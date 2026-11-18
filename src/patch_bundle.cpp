#include "aic/patch_bundle.hpp"

#include "aic/codegen.hpp"
#include "aic/errors.hpp"
#include "aic/json_min.hpp"
#include "aic/safetensors.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace aic {

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr const char* kMatHiPrefix = "delta.m.";
constexpr const char* kMatLoPrefix = "delta.mr.";
constexpr const char* kVecHiPrefix = "delta.v.";
constexpr const char* kVecLoPrefix = "delta.vr.";

bool is_threshold_route_tensor(const std::string& key) {
    return key.find(".mlp.aic_threshold_route.") != std::string::npos;
}

bool has_threshold_route_tensor(const std::map<std::string, Matrix>& matrices) {
    return std::any_of(matrices.begin(), matrices.end(), [](const auto& item) {
        return is_threshold_route_tensor(item.first);
    });
}

bool has_threshold_route_tensor(const std::map<std::string, std::string>& weight_map) {
    return std::any_of(weight_map.begin(), weight_map.end(), [](const auto& item) {
        return is_threshold_route_tensor(item.first);
    });
}

std::uint64_t fnv1a_u64(std::uint64_t h, std::uint64_t val) {
    for (int i = 0; i < 8; ++i) {
        const std::uint8_t b = static_cast<std::uint8_t>((val >> (8 * i)) & 0xFFu);
        h ^= b;
        h *= kFnvPrime;
    }
    return h;
}

std::uint64_t fnv1a_matrix(const Matrix& m) {
    std::uint64_t h = kFnvOffset;
    h = fnv1a_u64(h, static_cast<std::uint64_t>(m.rows));
    h = fnv1a_u64(h, static_cast<std::uint64_t>(m.cols));
    for (double x : m.data) {
        const double nx = (x == 0.0) ? 0.0 : x;
        h = fnv1a_u64(h, std::bit_cast<std::uint64_t>(nx));
    }
    return h;
}

std::uint64_t fnv1a_vector(const Vector& v) {
    std::uint64_t h = kFnvOffset;
    h = fnv1a_u64(h, static_cast<std::uint64_t>(v.size()));
    for (double x : v) {
        const double nx = (x == 0.0) ? 0.0 : x;
        h = fnv1a_u64(h, std::bit_cast<std::uint64_t>(nx));
    }
    return h;
}

std::string wisa_kind_to_str(WISAKind k) {
    switch (k) {
        case WISAKind::SWIGLU_POLE: return "SWIGLU_POLE";
        case WISAKind::THRESHOLDED_ROUTE: return "THRESHOLDED_ROUTE";
        case WISAKind::VPD_RANK1: return "VPD_RANK1";
        case WISAKind::ATTN_BILINEAR: return "ATTN_BILINEAR";
        case WISAKind::ASSOC_MEM_ROW: return "ASSOC_MEM_ROW";
        case WISAKind::FIELD_SPECTRAL_OP: return "FIELD_SPECTRAL_OP";
        case WISAKind::NORM_RESCALE: return "NORM_RESCALE";
    }
    return "VPD_RANK1";
}

WISAKind wisa_kind_from_str(const std::string& s) {
    if (s == "SWIGLU_POLE") return WISAKind::SWIGLU_POLE;
    if (s == "THRESHOLDED_ROUTE") return WISAKind::THRESHOLDED_ROUTE;
    if (s == "VPD_RANK1") return WISAKind::VPD_RANK1;
    if (s == "ATTN_BILINEAR") return WISAKind::ATTN_BILINEAR;
    if (s == "ASSOC_MEM_ROW") return WISAKind::ASSOC_MEM_ROW;
    if (s == "FIELD_SPECTRAL_OP") return WISAKind::FIELD_SPECTRAL_OP;
    if (s == "NORM_RESCALE") return WISAKind::NORM_RESCALE;
    throw StrataError(ErrorKind::IO, "Unknown WISAInstruction kind in patch bundle: " + s);
}

JsonValue inst_to_json(const WISAInstruction& wi) {
    JsonValue obj = JsonValue::object();
    obj["kind"] = wisa_kind_to_str(wi.kind);
    obj["layer"] = wi.layer;
    JsonValue lr = JsonValue::array();
    lr.as_array().emplace_back(wi.layer_range.first);
    lr.as_array().emplace_back(wi.layer_range.second);
    obj["layer_range"] = std::move(lr);
    obj["slot"] = wi.slot;
    obj["polarity"] = wi.polarity;
    obj["append_slot"] = wi.append_slot;
    obj["matrix_id"] = wi.matrix_id;
    obj["head_group"] = wi.head_group;
    obj["rope_mode"] = wi.rope_mode;
    obj["norm_id"] = wi.norm_id;
    obj["scale"] = wi.scale;
    obj["threshold"] = wi.threshold;
    obj["target_norm"] = wi.target_norm;
    return obj;
}

WISAInstruction inst_from_json(const JsonValue& obj) {
    if (!obj.is_object()) {
        throw StrataError(ErrorKind::IO, "Invalid instruction entry in patch bundle manifest");
    }
    WISAInstruction wi;
    if (obj.has("kind") && obj["kind"].is_string()) {
        wi.kind = wisa_kind_from_str(obj["kind"].as_string());
    }
    if (obj.has("layer") && obj["layer"].is_number()) {
        wi.layer = static_cast<int>(obj["layer"].as_int());
    }
    if (obj.has("layer_range") && obj["layer_range"].is_array() && obj["layer_range"].as_array().size() >= 2) {
        wi.layer_range = {
            static_cast<int>(obj["layer_range"].as_array()[0].as_int()),
            static_cast<int>(obj["layer_range"].as_array()[1].as_int())};
    }
    if (obj.has("slot") && obj["slot"].is_number()) {
        wi.slot = static_cast<int>(obj["slot"].as_int());
    }
    if (obj.has("polarity") && obj["polarity"].is_number()) {
        wi.polarity = static_cast<int>(obj["polarity"].as_int());
    }
    if (obj.has("append_slot") && obj["append_slot"].is_bool()) {
        wi.append_slot = obj["append_slot"].as_bool();
    }
    if (obj.has("matrix_id") && obj["matrix_id"].is_string()) {
        wi.matrix_id = obj["matrix_id"].as_string();
    }
    if (obj.has("head_group") && obj["head_group"].is_number()) {
        wi.head_group = static_cast<int>(obj["head_group"].as_int());
    }
    if (obj.has("rope_mode") && obj["rope_mode"].is_string()) {
        wi.rope_mode = obj["rope_mode"].as_string();
    }
    if (obj.has("norm_id") && obj["norm_id"].is_string()) {
        wi.norm_id = obj["norm_id"].as_string();
    }
    if (obj.has("scale") && obj["scale"].is_number()) {
        wi.scale = obj["scale"].as_number();
    }
    if (obj.has("threshold") && obj["threshold"].is_number()) {
        wi.threshold = obj["threshold"].as_number();
    }
    if (obj.has("target_norm") && obj["target_norm"].is_number()) {
        wi.target_norm = obj["target_norm"].as_number();
    }
    return wi;
}

void split_double_hi_lo(
    const std::vector<double>& src,
    std::vector<double>& hi_out,
    std::vector<double>& lo_out) {
    hi_out.resize(src.size());
    lo_out.resize(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) {
        const double hi = static_cast<double>(static_cast<float>(src[i]));
        const double lo = static_cast<double>(static_cast<float>(src[i] - hi));
        hi_out[i] = hi;
        lo_out[i] = lo;
    }
}

Matrix pad_zero_matrix(const Matrix& m, std::size_t rows, std::size_t cols) {
    Matrix out(rows, cols, 0.0);
    const std::size_t r_copy = std::min(rows, m.rows);
    const std::size_t c_copy = std::min(cols, m.cols);
    for (std::size_t r = 0; r < r_copy; ++r) {
        for (std::size_t c = 0; c < c_copy; ++c) {
            out(r, c) = m(r, c);
        }
    }
    return out;
}

Matrix crop_matrix(const Matrix& m, std::size_t rows, std::size_t cols) {
    if (m.rows == rows && m.cols == cols) {
        return m;
    }
    Matrix out(rows, cols, 0.0);
    const std::size_t r_copy = std::min(rows, m.rows);
    const std::size_t c_copy = std::min(cols, m.cols);
    for (std::size_t r = 0; r < r_copy; ++r) {
        for (std::size_t c = 0; c < c_copy; ++c) {
            out(r, c) = m(r, c);
        }
    }
    return out;
}

std::string lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string normalize_dtype_name(std::string dtype) {
    for (char& c : dtype) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    if (dtype == "FLOAT32" || dtype == "FP32") {
        return "F32";
    }
    if (dtype == "FLOAT16" || dtype == "FP16") {
        return "F16";
    }
    if (dtype == "BFLOAT16") {
        return "BF16";
    }
    if (dtype == "FP8" || dtype == "F8_E4M3" || dtype == "FLOAT8_E4M3FN") {
        return "F8_E4M3";
    }
    if (dtype == "I8" || dtype == "INT8") {
        return "I8";
    }
    if (dtype == "I4" || dtype == "INT4") {
        return "I4";
    }
    if (dtype != "F32" && dtype != "F16" && dtype != "BF16" &&
        dtype != "F8_E4M3" && dtype != "I8" && dtype != "I4") {
        return "F32";
    }
    return dtype;
}

int extract_layer_index(const std::string& key) {
    const std::string prefix = "model.layers.";
    if (!key.starts_with(prefix)) {
        return -1;
    }
    std::size_t pos = prefix.size();
    std::size_t end = key.find('.', pos);
    if (end == std::string::npos || end == pos) {
        return -1;
    }
    int idx = 0;
    for (std::size_t i = pos; i < end; ++i) {
        if (!std::isdigit(static_cast<unsigned char>(key[i]))) {
            return -1;
        }
        idx = idx * 10 + (key[i] - '0');
    }
    return idx;
}

std::size_t effective_head_dim(const ModelConfig& cfg) {
    if (cfg.head_dim > 0) {
        return static_cast<std::size_t>(cfg.head_dim);
    }
    const std::size_t n_q = static_cast<std::size_t>(std::max(1, cfg.num_attention_heads));
    return static_cast<std::size_t>(cfg.hidden_size) / n_q;
}

void build_zero_bypass_layer_tensors(
    const ModelConfig& cfg,
    int layer_idx,
    int d_ff_override,
    std::map<std::string, Matrix>& out_matrices,
    std::map<std::string, Vector>& out_vectors) {
    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t d_ff = static_cast<std::size_t>(std::max(cfg.intermediate_size, d_ff_override));
    const std::size_t n_q = static_cast<std::size_t>(std::max(1, cfg.num_attention_heads));
    const std::size_t hd = effective_head_dim(cfg);
    const std::size_t q_dim = n_q * hd;
    const std::string attn = lower_ascii(cfg.attention_type);
    const std::string prefix = "model.layers." + std::to_string(layer_idx);

    out_vectors[prefix + ".input_layernorm.weight"] = Vector(d, 1.0);
    out_matrices[prefix + ".self_attn.q_proj.weight"] = zeros(q_dim, d);

    std::size_t kv_dim = 0;
    if (attn == "mla") {
        const std::size_t rank = static_cast<std::size_t>(std::max(1, cfg.kv_lora_rank));
        out_matrices[prefix + ".self_attn.kv_a_proj.weight"] = zeros(rank, d);
        out_matrices[prefix + ".self_attn.kv_b_proj.weight"] = zeros(2 * q_dim, rank);
        out_matrices[prefix + ".self_attn.k_proj.weight"] = zeros(q_dim, d);
        out_matrices[prefix + ".self_attn.v_proj.weight"] = zeros(q_dim, d);
        kv_dim = q_dim;
    } else {
        kv_dim = static_cast<std::size_t>(std::max(1, cfg.num_key_value_heads)) * hd;
        out_matrices[prefix + ".self_attn.k_proj.weight"] = zeros(kv_dim, d);
        out_matrices[prefix + ".self_attn.v_proj.weight"] = zeros(kv_dim, d);
    }

    if (cfg.qkv_bias) {
        out_vectors[prefix + ".self_attn.q_proj.bias"] = Vector(q_dim, 0.0);
        out_vectors[prefix + ".self_attn.k_proj.bias"] = Vector(kv_dim, 0.0);
        out_vectors[prefix + ".self_attn.v_proj.bias"] = Vector(kv_dim, 0.0);
    }
    if (cfg.qk_norm) {
        out_vectors[prefix + ".self_attn.q_norm.weight"] = Vector(hd, 1.0);
        out_vectors[prefix + ".self_attn.k_norm.weight"] = Vector(hd, 1.0);
    }

    out_matrices[prefix + ".self_attn.o_proj.weight"] = zeros(d, q_dim);
    out_vectors[prefix + ".post_attention_layernorm.weight"] = Vector(d, 1.0);
    out_matrices[prefix + ".mlp.gate_proj.weight"] = zeros(d_ff, d);
    out_matrices[prefix + ".mlp.up_proj.weight"] = zeros(d_ff, d);
    out_matrices[prefix + ".mlp.down_proj.weight"] = zeros(d, d_ff);
}

std::string read_text_file(const std::filesystem::path& p) {
    std::ifstream ifs(p);
    if (!ifs) {
        throw IOError("Failed to open file for reading: " + p.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void write_text_file(const std::filesystem::path& p, const std::string& content) {
    if (!p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream ofs(p, std::ios::trunc);
    if (!ofs) {
        throw IOError("Failed to open file for writing: " + p.string());
    }
    ofs << content;
    if (!ofs.good()) {
        throw IOError("Failed writing file: " + p.string());
    }
}

struct ShardLayout {
    bool is_sharded = false;
    JsonValue index_json;
    std::map<std::string, std::string> weight_map;
    std::vector<std::string> shard_order;
};

ShardLayout load_shard_layout(const std::filesystem::path& dir) {
    ShardLayout layout;
    const std::filesystem::path index_path = dir / "model.safetensors.index.json";
    const std::filesystem::path single_path = dir / "model.safetensors";
    layout.is_sharded = std::filesystem::exists(index_path);
    if (!layout.is_sharded && !std::filesystem::exists(single_path)) {
        throw IOError("No model.safetensors or model.safetensors.index.json found in: " + dir.string());
    }
    if (layout.is_sharded) {
        layout.index_json = parse_json(read_text_file(index_path));
        if (!layout.index_json.is_object() ||
            !layout.index_json.has("weight_map") ||
            !layout.index_json["weight_map"].is_object()) {
            throw IOError("Invalid model.safetensors.index.json in: " + dir.string());
        }
        std::set<std::string> seen;
        for (const auto& [tname, s_val] : layout.index_json["weight_map"].as_object()) {
            const std::string sfile = s_val.as_string();
            layout.weight_map[tname] = sfile;
            if (!seen.contains(sfile)) {
                seen.insert(sfile);
                layout.shard_order.push_back(sfile);
            }
        }
        std::sort(layout.shard_order.begin(), layout.shard_order.end());
    } else {
        MmapSafetensors reader(single_path.string());
        for (const std::string& tname : reader.tensor_names()) {
            layout.weight_map[tname] = "model.safetensors";
        }
        layout.shard_order.push_back("model.safetensors");
    }
    return layout;
}

void write_aux_checkpoint_files(
    const std::filesystem::path& src_dir,
    const std::filesystem::path& dst_dir,
    bool same_dir,
    bool is_sharded,
    JsonValue& index_json,
    const std::map<std::string, std::string>& weight_map,
    std::int64_t total_byte_delta,
    JsonValue& cfg_json,
    int num_layers,
    int intermediate_size) {
    if (is_sharded) {
        JsonValue new_wmap = JsonValue::object();
        for (const auto& [tname, sfile] : weight_map) {
            new_wmap[tname] = sfile;
        }
        index_json["weight_map"] = std::move(new_wmap);
        if (index_json.has("metadata") && index_json["metadata"].is_object() &&
            index_json["metadata"].has("total_size") && index_json["metadata"]["total_size"].is_number()) {
            const std::int64_t old_sz = index_json["metadata"]["total_size"].as_int();
            const std::int64_t new_sz = std::max<std::int64_t>(0, old_sz + total_byte_delta);
            index_json["metadata"]["total_size"] = new_sz;
        }
        write_text_file(dst_dir / "model.safetensors.index.json", dump_json(index_json, 2) + "\n");
    }

    cfg_json["num_layers"] = num_layers;
    if (cfg_json.has("num_hidden_layers")) {
        cfg_json["num_hidden_layers"] = num_layers;
    }
    cfg_json["intermediate_size"] = intermediate_size;
    write_text_file(dst_dir / "config.json", dump_json(cfg_json, 2) + "\n");

    if (!same_dir) {
        const std::filesystem::path tok_src = src_dir / "tokenizer.json";
        if (std::filesystem::exists(tok_src)) {
            std::filesystem::copy_file(
                tok_src,
                dst_dir / "tokenizer.json",
                std::filesystem::copy_options::overwrite_existing);
        }
        const std::filesystem::path atlas_src = src_dir / "model.natlas";
        if (std::filesystem::exists(atlas_src)) {
            std::filesystem::copy_file(
                atlas_src,
                dst_dir / "model.natlas",
                std::filesystem::copy_options::overwrite_existing);
        }
    }
}

}

PatchBundle create_patch_bundle(
    const TransformerModel& base_model,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode,
    const std::string& module_name) {
    const std::string mode = target_mode.empty() ? std::string("in-place") : target_mode;
    auto [patched, stats] = apply_wisa(base_model, insts, mode);
    (void)stats;

    PatchBundle bundle;
    bundle.module_name = module_name.empty() ? std::string("patch") : module_name;
    bundle.arch_name = base_model.config.arch_name;
    bundle.target_mode = mode;
    bundle.num_layers = base_model.config.num_layers;
    bundle.hidden_size = base_model.config.hidden_size;
    bundle.orig_num_layers = base_model.config.num_layers;
    bundle.orig_intermediate_size = base_model.config.intermediate_size;
    bundle.instructions = insts;

    for (const auto& [key, w_patched] : patched.matrices) {
        auto bit = base_model.matrices.find(key);
        if (bit == base_model.matrices.end()) {
            if (frob_norm(w_patched) > 1e-14) {
                bundle.matrix_deltas[key] = w_patched;
            }
            continue;
        }
        const Matrix& w_base = bit->second;
        if (w_patched.rows == w_base.rows && w_patched.cols == w_base.cols) {
            Matrix delta = mat_sub(w_patched, w_base);
            if (frob_norm(delta) > 1e-14) {
                bundle.base_checksums[key] = fnv1a_matrix(w_base);
                bundle.matrix_deltas[key] = std::move(delta);
            }
        } else if (w_patched.rows >= w_base.rows && w_patched.cols >= w_base.cols) {
            Matrix delta = mat_sub(w_patched, pad_zero_matrix(w_base, w_patched.rows, w_patched.cols));
            if (frob_norm(delta) > 1e-14) {
                bundle.base_checksums[key] = fnv1a_matrix(w_base);
                bundle.matrix_deltas[key] = std::move(delta);
            }
        } else {
            throw StrataError(ErrorKind::IO, "Patched matrix shrank unexpectedly for " + key);
        }
    }

    for (const auto& [key, v_patched] : patched.vectors) {
        auto bit = base_model.vectors.find(key);
        if (bit == base_model.vectors.end()) {
            if (norm2(v_patched) > 1e-14) {
                bundle.vector_deltas[key] = v_patched;
            }
            continue;
        }
        const Vector& v_base = bit->second;
        if (v_patched.size() == v_base.size()) {
            Vector delta = vec_sub(v_patched, v_base);
            if (norm2(delta) > 1e-14) {
                bundle.base_checksums[key] = fnv1a_vector(v_base);
                bundle.vector_deltas[key] = std::move(delta);
            }
        } else {
            throw StrataError(ErrorKind::IO, "Patched vector shape changed unexpectedly for " + key);
        }
    }

    return bundle;
}

PatchBundle create_patch_bundle_streaming(
    const std::string& model_dir,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode,
    const std::string& module_name) {
    const std::string mode = target_mode.empty() ? std::string("in-place") : target_mode;
    if (mode != "in-place") {
        throw StrataError(
            ErrorKind::IO,
            "Patch-only streaming currently requires target=in-place");
    }

    const std::filesystem::path dir(model_dir);
    const ModelConfig cfg = ModelConfig::from_json(parse_json(read_text_file(dir / "config.json")));
    const ShardLayout layout = load_shard_layout(dir);
    std::set<int> target_layers;
    for (const WISAInstruction& wi : insts) {
        if (wi.kind == WISAKind::FIELD_SPECTRAL_OP) {
            for (int l = wi.layer_range.first; l < wi.layer_range.second; ++l) {
                target_layers.insert(l);
            }
        } else {
            target_layers.insert(wi.layer);
        }
    }
    for (int layer : target_layers) {
        if (layer < 0 || layer >= cfg.num_layers) {
            throw OutOfDepthError("Patch-only instruction layer out of range");
        }
    }

    std::map<std::string, std::unique_ptr<MmapSafetensors>> readers;
    auto get_reader = [&](const std::string& shard) -> MmapSafetensors& {
        auto it = readers.find(shard);
        if (it == readers.end()) {
            auto ptr = std::make_unique<MmapSafetensors>((dir / shard).string());
            it = readers.emplace(shard, std::move(ptr)).first;
        }
        return *it->second;
    };

    TransformerModel sparse;
    sparse.config = cfg;
    for (const auto& [name, shard] : layout.weight_map) {
        const int layer = extract_layer_index(name);
        if (!target_layers.contains(layer)) {
            continue;
        }
        const TensorSpec spec = get_reader(shard).read_tensor(name);
        if (spec.shape.size() == 1) {
            sparse.vectors[name] = spec.to_vector();
        } else if (spec.shape.size() == 2) {
            sparse.matrices[name] = spec.to_matrix();
        }
    }
    return create_patch_bundle(sparse, insts, mode, module_name);
}

void save_patch_bundle(const PatchBundle& bundle, const std::string& path) {
    JsonValue manifest = JsonValue::object();
    manifest["format"] = "npatch-v1";
    manifest["module_name"] = bundle.module_name;
    manifest["arch_name"] = bundle.arch_name;
    manifest["target_mode"] = bundle.target_mode;
    manifest["num_layers"] = bundle.num_layers;
    manifest["hidden_size"] = bundle.hidden_size;
    manifest["orig_num_layers"] = bundle.orig_num_layers;
    manifest["orig_intermediate_size"] = bundle.orig_intermediate_size;

    JsonValue cks = JsonValue::object();
    for (const auto& [key, hash] : bundle.base_checksums) {
        cks[key] = std::to_string(hash);
    }
    manifest["base_checksums"] = std::move(cks);

    JsonValue inst_arr = JsonValue::array();
    for (const auto& wi : bundle.instructions) {
        inst_arr.as_array().push_back(inst_to_json(wi));
    }
    manifest["instructions"] = std::move(inst_arr);

    std::map<std::string, TensorSpec> tensors;
    for (const auto& [key, m] : bundle.matrix_deltas) {
        std::vector<double> hi;
        std::vector<double> lo;
        split_double_hi_lo(m.data, hi, lo);
        tensors[std::string(kMatHiPrefix) + key] = TensorSpec{{m.rows, m.cols}, "F32", std::move(hi)};
        tensors[std::string(kMatLoPrefix) + key] = TensorSpec{{m.rows, m.cols}, "F32", std::move(lo)};
    }
    for (const auto& [key, v] : bundle.vector_deltas) {
        std::vector<double> hi;
        std::vector<double> lo;
        split_double_hi_lo(v, hi, lo);
        tensors[std::string(kVecHiPrefix) + key] = TensorSpec{{v.size()}, "F32", std::move(hi)};
        tensors[std::string(kVecLoPrefix) + key] = TensorSpec{{v.size()}, "F32", std::move(lo)};
    }

    std::map<std::string, std::string> metadata;
    metadata["npatch_manifest"] = dump_json(manifest, -1);
    metadata["module_name"] = bundle.module_name;
    metadata["arch_name"] = bundle.arch_name;
    metadata["target_mode"] = bundle.target_mode;

    save_safetensors(path, tensors, metadata);
}

PatchBundle load_patch_bundle(const std::string& path) {
    auto [tensors, metadata] = load_safetensors(path);
    auto mit = metadata.find("npatch_manifest");
    if (mit == metadata.end()) {
        throw StrataError(ErrorKind::IO, "Missing npatch_manifest metadata in patch bundle: " + path);
    }

    const JsonValue manifest = parse_json(mit->second);
    if (!manifest.is_object()) {
        throw StrataError(ErrorKind::IO, "Invalid npatch_manifest JSON in patch bundle: " + path);
    }

    PatchBundle bundle;
    if (manifest.has("module_name") && manifest["module_name"].is_string()) {
        bundle.module_name = manifest["module_name"].as_string();
    }
    if (manifest.has("arch_name") && manifest["arch_name"].is_string()) {
        bundle.arch_name = manifest["arch_name"].as_string();
    }
    if (manifest.has("target_mode") && manifest["target_mode"].is_string()) {
        bundle.target_mode = manifest["target_mode"].as_string();
    }
    if (manifest.has("num_layers") && manifest["num_layers"].is_number()) {
        bundle.num_layers = static_cast<int>(manifest["num_layers"].as_int());
    }
    if (manifest.has("hidden_size") && manifest["hidden_size"].is_number()) {
        bundle.hidden_size = static_cast<int>(manifest["hidden_size"].as_int());
    }
    if (manifest.has("orig_num_layers") && manifest["orig_num_layers"].is_number()) {
        bundle.orig_num_layers = static_cast<int>(manifest["orig_num_layers"].as_int());
    } else {
        bundle.orig_num_layers = bundle.num_layers;
    }
    if (manifest.has("orig_intermediate_size") && manifest["orig_intermediate_size"].is_number()) {
        bundle.orig_intermediate_size = static_cast<int>(manifest["orig_intermediate_size"].as_int());
    }

    if (manifest.has("base_checksums") && manifest["base_checksums"].is_object()) {
        for (const auto& [k, v] : manifest["base_checksums"].as_object()) {
            if (!v.is_string()) {
                throw StrataError(ErrorKind::IO, "Invalid checksum string for key: " + k);
            }
            bundle.base_checksums[k] = static_cast<std::uint64_t>(std::stoull(v.as_string()));
        }
    }

    if (manifest.has("instructions") && manifest["instructions"].is_array()) {
        for (const auto& ij : manifest["instructions"].as_array()) {
            bundle.instructions.push_back(inst_from_json(ij));
        }
    }

    const std::string m_hi(kMatHiPrefix);
    const std::string m_lo(kMatLoPrefix);
    const std::string v_hi(kVecHiPrefix);
    const std::string v_lo(kVecLoPrefix);

    for (const auto& [tname, spec] : tensors) {
        if (tname.starts_with(m_hi)) {
            const std::string key = tname.substr(m_hi.size());
            Matrix m = spec.to_matrix();
            auto lo_it = tensors.find(m_lo + key);
            if (lo_it != tensors.end()) {
                Matrix lo_m = lo_it->second.to_matrix();
                if (lo_m.rows == m.rows && lo_m.cols == m.cols) {
                    m = mat_add(m, lo_m);
                }
            }
            bundle.matrix_deltas[key] = std::move(m);
        } else if (tname.starts_with(v_hi)) {
            const std::string key = tname.substr(v_hi.size());
            Vector v = spec.to_vector();
            auto lo_it = tensors.find(v_lo + key);
            if (lo_it != tensors.end()) {
                Vector lo_v = lo_it->second.to_vector();
                if (lo_v.size() == v.size()) {
                    v = vec_add(v, lo_v);
                }
            }
            bundle.vector_deltas[key] = std::move(v);
        }
    }

    return bundle;
}

TransformerModel apply_patch_bundle(
    const TransformerModel& base_model,
    const PatchBundle& bundle,
    bool verify_checksum) {
    if (!bundle.arch_name.empty() && !base_model.config.arch_name.empty() &&
        bundle.arch_name != base_model.config.arch_name) {
        throw StrataError(
            ErrorKind::FrameMismatch,
            "PatchBundle arch_name '" + bundle.arch_name +
            "' mismatches model arch_name '" + base_model.config.arch_name + "'");
    }
    if (bundle.hidden_size > 0 && base_model.config.hidden_size != bundle.hidden_size) {
        throw StrataError(
            ErrorKind::IO,
            "PatchBundle hidden_size " + std::to_string(bundle.hidden_size) +
            " mismatches model hidden_size " + std::to_string(base_model.config.hidden_size));
    }
    if (bundle.num_layers > 0 && base_model.config.num_layers < bundle.num_layers) {
        throw StrataError(
            ErrorKind::OutOfDepth,
            "PatchBundle num_layers " + std::to_string(bundle.num_layers) +
            " exceeds model num_layers " + std::to_string(base_model.config.num_layers));
    }

    if (verify_checksum) {
        for (const auto& [key, expected] : bundle.base_checksums) {
            if (bundle.matrix_deltas.contains(key)) {
                auto it = base_model.matrices.find(key);
                if (it == base_model.matrices.end()) {
                    throw StrataError(ErrorKind::IO, "Missing base matrix for checksum check: " + key);
                }
                const std::uint64_t actual = fnv1a_matrix(it->second);
                if (actual != expected) {
                    throw StrataError(ErrorKind::IO, "Base matrix checksum mismatch for " + key);
                }
            } else if (bundle.vector_deltas.contains(key)) {
                auto it = base_model.vectors.find(key);
                if (it == base_model.vectors.end()) {
                    throw StrataError(ErrorKind::IO, "Missing base vector for checksum check: " + key);
                }
                const std::uint64_t actual = fnv1a_vector(it->second);
                if (actual != expected) {
                    throw StrataError(ErrorKind::IO, "Base vector checksum mismatch for " + key);
                }
            }
        }
    }

    TransformerModel out = base_model;
    int max_layer_idx = out.config.num_layers - 1;
    int new_d_ff = out.config.intermediate_size;
    for (const auto& [key, delta] : bundle.matrix_deltas) {
        const int l_idx = extract_layer_index(key);
        if (l_idx > max_layer_idx) {
            max_layer_idx = l_idx;
        }
        if (key.find(".mlp.gate_proj.") != std::string::npos ||
            key.find(".mlp.up_proj.") != std::string::npos) {
            new_d_ff = std::max(new_d_ff, static_cast<int>(delta.rows));
        } else if (key.find(".mlp.down_proj.") != std::string::npos) {
            new_d_ff = std::max(new_d_ff, static_cast<int>(delta.cols));
        }
    }
    for (const auto& [key, delta] : bundle.vector_deltas) {
        (void)delta;
        const int l_idx = extract_layer_index(key);
        if (l_idx > max_layer_idx) {
            max_layer_idx = l_idx;
        }
    }

    if (max_layer_idx >= out.config.num_layers) {
        for (int l = out.config.num_layers; l <= max_layer_idx; ++l) {
            build_zero_bypass_layer_tensors(out.config, l, new_d_ff, out.matrices, out.vectors);
        }
        out.config.num_layers = max_layer_idx + 1;
    }

    for (const auto& [key, delta] : bundle.matrix_deltas) {
        auto it = out.matrices.find(key);
        if (it == out.matrices.end()) {
            out.matrices[key] = delta;
            continue;
        }
        Matrix& m = it->second;
        if (m.rows == delta.rows && m.cols == delta.cols) {
            m = mat_add(m, delta);
        } else if (delta.rows >= m.rows && delta.cols >= m.cols) {
            m = mat_add(pad_zero_matrix(m, delta.rows, delta.cols), delta);
        } else {
            throw StrataError(ErrorKind::IO, "Matrix shape mismatch applying patch bundle to " + key);
        }
    }

    for (const auto& [key, delta] : bundle.vector_deltas) {
        auto it = out.vectors.find(key);
        if (it == out.vectors.end()) {
            out.vectors[key] = delta;
            continue;
        }
        Vector& v = it->second;
        if (v.size() != delta.size()) {
            throw StrataError(ErrorKind::IO, "Vector size mismatch applying patch bundle to " + key);
        }
        v = vec_add(v, delta);
    }

    out.config.aic_threshold_routes = has_threshold_route_tensor(out.matrices);

    if (new_d_ff > out.config.intermediate_size) {
        const std::size_t ud_ff = static_cast<std::size_t>(new_d_ff);
        for (int l = 0; l < out.config.num_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l);
            for (const char* suf : {".mlp.gate_proj.weight", ".mlp.up_proj.weight"}) {
                auto mit = out.matrices.find(p + suf);
                if (mit != out.matrices.end() && mit->second.rows < ud_ff) {
                    mit->second = pad_zero_matrix(mit->second, ud_ff, mit->second.cols);
                }
            }
            auto dit = out.matrices.find(p + ".mlp.down_proj.weight");
            if (dit != out.matrices.end() && dit->second.cols < ud_ff) {
                dit->second = pad_zero_matrix(dit->second, dit->second.rows, ud_ff);
            }
        }
        out.config.intermediate_size = new_d_ff;
    }

    return out;
}

TransformerModel revert_patch_bundle(
    const TransformerModel& patched_model,
    const PatchBundle& bundle) {
    if (!bundle.arch_name.empty() && !patched_model.config.arch_name.empty() &&
        bundle.arch_name != patched_model.config.arch_name) {
        throw StrataError(
            ErrorKind::FrameMismatch,
            "PatchBundle arch_name '" + bundle.arch_name +
            "' mismatches patched model arch_name '" + patched_model.config.arch_name + "'");
    }
    if (bundle.hidden_size > 0 && patched_model.config.hidden_size != bundle.hidden_size) {
        throw StrataError(
            ErrorKind::IO,
            "PatchBundle hidden_size mismatches patched model hidden_size");
    }

    TransformerModel out = patched_model;
    const bool shrink_elastic = (bundle.target_mode == "elastic");
    const int target_layers = (shrink_elastic && bundle.orig_num_layers > 0 &&
                               out.config.num_layers > bundle.orig_num_layers)
        ? bundle.orig_num_layers
        : out.config.num_layers;
    const int target_dff = (shrink_elastic && bundle.orig_intermediate_size > 0 &&
                            out.config.intermediate_size > bundle.orig_intermediate_size)
        ? bundle.orig_intermediate_size
        : out.config.intermediate_size;

    for (const auto& [key, delta] : bundle.matrix_deltas) {
        const int l_idx = extract_layer_index(key);
        if (l_idx >= target_layers) {
            continue;
        }
        if (is_threshold_route_tensor(key) && !bundle.base_checksums.contains(key)) {
            out.matrices.erase(key);
            continue;
        }
        auto it = out.matrices.find(key);
        if (it == out.matrices.end()) {
            throw StrataError(ErrorKind::IO, "Missing matrix when reverting patch bundle: " + key);
        }
        if (it->second.rows != delta.rows || it->second.cols != delta.cols) {
            throw StrataError(ErrorKind::IO, "Matrix shape mismatch reverting patch bundle on " + key);
        }
        it->second = mat_sub(it->second, delta);
    }

    for (const auto& [key, delta] : bundle.vector_deltas) {
        const int l_idx = extract_layer_index(key);
        if (l_idx >= target_layers) {
            continue;
        }
        if (is_threshold_route_tensor(key) && !bundle.base_checksums.contains(key)) {
            out.vectors.erase(key);
            continue;
        }
        auto it = out.vectors.find(key);
        if (it == out.vectors.end()) {
            throw StrataError(ErrorKind::IO, "Missing vector when reverting patch bundle: " + key);
        }
        if (it->second.size() != delta.size()) {
            throw StrataError(ErrorKind::IO, "Vector size mismatch reverting patch bundle on " + key);
        }
        it->second = vec_sub(it->second, delta);
    }

    if (target_layers < out.config.num_layers) {
        for (auto it = out.matrices.begin(); it != out.matrices.end();) {
            if (extract_layer_index(it->first) >= target_layers) {
                it = out.matrices.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = out.vectors.begin(); it != out.vectors.end();) {
            if (extract_layer_index(it->first) >= target_layers) {
                it = out.vectors.erase(it);
            } else {
                ++it;
            }
        }
        out.config.num_layers = target_layers;
    }

    out.config.aic_threshold_routes = has_threshold_route_tensor(out.matrices);

    if (target_dff < out.config.intermediate_size) {
        const std::size_t ud_ff = static_cast<std::size_t>(target_dff);
        for (int l = 0; l < out.config.num_layers; ++l) {
            const std::string p = "model.layers." + std::to_string(l);
            for (const char* suf : {".mlp.gate_proj.weight", ".mlp.up_proj.weight"}) {
                auto mit = out.matrices.find(p + suf);
                if (mit != out.matrices.end() && mit->second.rows > ud_ff) {
                    mit->second = crop_matrix(mit->second, ud_ff, mit->second.cols);
                }
            }
            auto dit = out.matrices.find(p + ".mlp.down_proj.weight");
            if (dit != out.matrices.end() && dit->second.cols > ud_ff) {
                dit->second = crop_matrix(dit->second, dit->second.rows, ud_ff);
            }
        }
        out.config.intermediate_size = target_dff;
    }

    return out;
}

PatchBundle compose_patch_bundles(const PatchBundle& a, const PatchBundle& b) {
    if (!a.arch_name.empty() && !b.arch_name.empty() && a.arch_name != b.arch_name) {
        throw StrataError(
            ErrorKind::FrameMismatch,
            "Cannot compose patch bundles with different arch_name: '" +
            a.arch_name + "' vs '" + b.arch_name + "'");
    }
    if (a.num_layers > 0 && b.num_layers > 0 && a.num_layers != b.num_layers) {
        throw StrataError(
            ErrorKind::OutOfDepth,
            "Cannot compose patch bundles with different num_layers");
    }
    if (a.hidden_size > 0 && b.hidden_size > 0 && a.hidden_size != b.hidden_size) {
        throw StrataError(
            ErrorKind::IO,
            "Cannot compose patch bundles with different hidden_size");
    }

    for (const auto& [key, hash_a] : a.base_checksums) {
        auto it = b.base_checksums.find(key);
        if (it != b.base_checksums.end() && it->second != hash_a) {
            throw StrataError(
                ErrorKind::IO,
                "Incompatible base tensor checksums while composing patches for " + key);
        }
    }

    std::set<std::pair<int, int>> used_slots;
    for (const auto& wi : a.instructions) {
        if (wi.kind == WISAKind::SWIGLU_POLE) {
            used_slots.emplace(wi.layer, wi.slot);
        }
    }
    for (const auto& wi : b.instructions) {
        if (wi.kind == WISAKind::SWIGLU_POLE && used_slots.contains({wi.layer, wi.slot})) {
            throw CrosstalkExceededError(
                "Conflicting SWIGLU_POLE slot (" + std::to_string(wi.layer) + ", " +
                std::to_string(wi.slot) + ") during patch composition");
        }
    }

    PatchBundle out;
    out.module_name = a.module_name + "+" + b.module_name;
    out.arch_name = !a.arch_name.empty() ? a.arch_name : b.arch_name;
    out.target_mode = (a.target_mode == b.target_mode)
        ? a.target_mode
        : (a.target_mode.empty() ? b.target_mode : a.target_mode);
    out.num_layers = std::max(a.num_layers, b.num_layers);
    out.hidden_size = std::max(a.hidden_size, b.hidden_size);
    out.orig_num_layers = (a.orig_num_layers > 0) ? a.orig_num_layers : b.orig_num_layers;
    out.orig_intermediate_size = (a.orig_intermediate_size > 0)
        ? a.orig_intermediate_size
        : b.orig_intermediate_size;
    out.base_checksums = a.base_checksums;
    for (const auto& [key, hash_b] : b.base_checksums) {
        out.base_checksums[key] = hash_b;
    }

    out.instructions = a.instructions;
    out.instructions.insert(out.instructions.end(), b.instructions.begin(), b.instructions.end());

    out.matrix_deltas = a.matrix_deltas;
    for (const auto& [key, db] : b.matrix_deltas) {
        auto it = out.matrix_deltas.find(key);
        if (it == out.matrix_deltas.end()) {
            out.matrix_deltas[key] = db;
            continue;
        }
        const Matrix& da = it->second;
        if (da.rows != db.rows || da.cols != db.cols) {
            throw StrataError(ErrorKind::IO, "Incompatible matrix delta shape while composing " + key);
        }
        const double na = frob_norm(da);
        const double nb = frob_norm(db);
        if (na > 1e-12 && nb > 1e-12) {
            long double inner = 0.0;
            for (std::size_t i = 0; i < da.data.size(); ++i) {
                inner += static_cast<long double>(da.data[i]) * static_cast<long double>(db.data[i]);
            }
            const double cos_overlap = std::abs(static_cast<double>(inner)) / (na * nb);
            if (cos_overlap > 0.1) {
                throw CrosstalkExceededError(
                    "Conflicting matrix delta overlap (" + std::to_string(cos_overlap) +
                    ") on " + key + " during patch composition");
            }
        }
        it->second = mat_add(da, db);
    }

    out.vector_deltas = a.vector_deltas;
    for (const auto& [key, vb] : b.vector_deltas) {
        auto it = out.vector_deltas.find(key);
        if (it == out.vector_deltas.end()) {
            out.vector_deltas[key] = vb;
            continue;
        }
        if (it->second.size() != vb.size()) {
            throw StrataError(ErrorKind::IO, "Incompatible vector delta size while composing " + key);
        }
        it->second = vec_add(it->second, vb);
    }

    return out;
}

void apply_patch_bundle_streaming(
    const std::string& src_model_dir,
    const std::string& dst_model_dir,
    const PatchBundle& bundle,
    bool verify_checksum) {
    const std::filesystem::path src_dir(src_model_dir);
    const std::filesystem::path dst_dir(dst_model_dir);
    std::filesystem::create_directories(dst_dir);

    JsonValue cfg_json = parse_json(read_text_file(src_dir / "config.json"));
    ModelConfig cfg = ModelConfig::from_json(cfg_json);

    if (!bundle.arch_name.empty() && !cfg.arch_name.empty() &&
        bundle.arch_name != cfg.arch_name) {
        throw StrataError(
            ErrorKind::FrameMismatch,
            "PatchBundle arch_name '" + bundle.arch_name +
            "' mismatches model arch_name '" + cfg.arch_name + "'");
    }
    if (bundle.hidden_size > 0 && cfg.hidden_size != bundle.hidden_size) {
        throw StrataError(
            ErrorKind::IO,
            "PatchBundle hidden_size " + std::to_string(bundle.hidden_size) +
            " mismatches model hidden_size " + std::to_string(cfg.hidden_size));
    }
    if (bundle.num_layers > 0 && cfg.num_layers < bundle.num_layers) {
        throw StrataError(
            ErrorKind::OutOfDepth,
            "PatchBundle num_layers " + std::to_string(bundle.num_layers) +
            " exceeds model num_layers " + std::to_string(cfg.num_layers));
    }

    ShardLayout layout = load_shard_layout(src_dir);
    std::string default_dtype = normalize_dtype_name(cfg.torch_dtype);
    if (!layout.weight_map.empty()) {
        const auto first_it = layout.weight_map.begin();
        MmapSafetensors probe((src_dir / first_it->second).string());
        default_dtype = normalize_dtype_name(probe.tensor_dtype(first_it->first));
    }

    std::map<std::string, std::unique_ptr<MmapSafetensors>> reader_cache;
    auto get_reader = [&](const std::string& shard_file) -> MmapSafetensors& {
        auto it = reader_cache.find(shard_file);
        if (it == reader_cache.end()) {
            auto ptr = std::make_unique<MmapSafetensors>((src_dir / shard_file).string());
            it = reader_cache.emplace(shard_file, std::move(ptr)).first;
        }
        return *(it->second);
    };

    if (verify_checksum) {
        for (const auto& [key, expected] : bundle.base_checksums) {
            if (!bundle.matrix_deltas.contains(key) && !bundle.vector_deltas.contains(key)) {
                continue;
            }
            auto wmit = layout.weight_map.find(key);
            if (wmit == layout.weight_map.end()) {
                throw StrataError(ErrorKind::IO, "Missing base tensor for checksum check: " + key);
            }
            const TensorSpec spec = get_reader(wmit->second).read_tensor(key);
            if (bundle.matrix_deltas.contains(key)) {
                if (fnv1a_matrix(spec.to_matrix()) != expected) {
                    throw StrataError(ErrorKind::IO, "Base matrix checksum mismatch for " + key);
                }
            } else {
                if (fnv1a_vector(spec.to_vector()) != expected) {
                    throw StrataError(ErrorKind::IO, "Base vector checksum mismatch for " + key);
                }
            }
        }
    }

    int new_num_layers = cfg.num_layers;
    int new_d_ff = cfg.intermediate_size;
    for (const auto& [key, delta] : bundle.matrix_deltas) {
        const int l_idx = extract_layer_index(key);
        if (l_idx >= new_num_layers) {
            new_num_layers = l_idx + 1;
        }
        if (key.find(".mlp.gate_proj.") != std::string::npos ||
            key.find(".mlp.up_proj.") != std::string::npos) {
            new_d_ff = std::max(new_d_ff, static_cast<int>(delta.rows));
        } else if (key.find(".mlp.down_proj.") != std::string::npos) {
            new_d_ff = std::max(new_d_ff, static_cast<int>(delta.cols));
        }
    }
    for (const auto& [key, delta] : bundle.vector_deltas) {
        (void)delta;
        const int l_idx = extract_layer_index(key);
        if (l_idx >= new_num_layers) {
            new_num_layers = l_idx + 1;
        }
    }

    std::set<std::string> touched_shards;
    std::map<std::string, std::map<std::string, TensorSpec>> shard_replacements;
    std::map<std::string, std::map<std::string, TensorSpec>> shard_additions;
    const std::string target_shard = layout.shard_order.empty()
        ? std::string("model.safetensors")
        : layout.shard_order.back();

    std::map<std::string, Matrix> pending_new_matrices;
    std::map<std::string, Vector> pending_new_vectors;
    if (new_num_layers > cfg.num_layers) {
        for (int l = cfg.num_layers; l < new_num_layers; ++l) {
            build_zero_bypass_layer_tensors(
                cfg,
                l,
                new_d_ff,
                pending_new_matrices,
                pending_new_vectors);
        }
    }

    for (const auto& [key, delta] : bundle.matrix_deltas) {
        auto wmit = layout.weight_map.find(key);
        if (wmit == layout.weight_map.end()) {
            auto pit = pending_new_matrices.find(key);
            if (pit != pending_new_matrices.end()) {
                if (pit->second.rows == delta.rows && pit->second.cols == delta.cols) {
                    pit->second = mat_add(pit->second, delta);
                } else if (delta.rows >= pit->second.rows && delta.cols >= pit->second.cols) {
                    pit->second = mat_add(pad_zero_matrix(pit->second, delta.rows, delta.cols), delta);
                } else {
                    throw StrataError(ErrorKind::IO, "Shape mismatch in new layer matrix: " + key);
                }
            } else {
                pending_new_matrices[key] = delta;
            }
            continue;
        }

        const std::string& shard_file = wmit->second;
        const TensorSpec spec = get_reader(shard_file).read_tensor(key);
        Matrix m = spec.to_matrix();
        if (m.rows == delta.rows && m.cols == delta.cols) {
            m = mat_add(m, delta);
        } else if (delta.rows >= m.rows && delta.cols >= m.cols) {
            m = mat_add(pad_zero_matrix(m, delta.rows, delta.cols), delta);
        } else {
            throw StrataError(ErrorKind::IO, "Matrix shape mismatch applying patch bundle to " + key);
        }
        const std::size_t ud_ff = static_cast<std::size_t>(std::max(0, new_d_ff));
        if ((key.find(".mlp.gate_proj.") != std::string::npos ||
             key.find(".mlp.up_proj.") != std::string::npos) &&
            m.rows < ud_ff) {
            m = pad_zero_matrix(m, ud_ff, m.cols);
        } else if (key.find(".mlp.down_proj.") != std::string::npos && m.cols < ud_ff) {
            m = pad_zero_matrix(m, m.rows, ud_ff);
        }
        shard_replacements[shard_file][key] = TensorSpec::from_matrix(m, spec.dtype);
        touched_shards.insert(shard_file);
    }

    for (const auto& [key, delta] : bundle.vector_deltas) {
        auto wmit = layout.weight_map.find(key);
        if (wmit == layout.weight_map.end()) {
            auto pit = pending_new_vectors.find(key);
            if (pit != pending_new_vectors.end()) {
                if (pit->second.size() != delta.size()) {
                    throw StrataError(ErrorKind::IO, "Size mismatch in new layer vector: " + key);
                }
                pit->second = vec_add(pit->second, delta);
            } else {
                pending_new_vectors[key] = delta;
            }
            continue;
        }

        const std::string& shard_file = wmit->second;
        const TensorSpec spec = get_reader(shard_file).read_tensor(key);
        Vector v = spec.to_vector();
        if (v.size() != delta.size()) {
            throw StrataError(ErrorKind::IO, "Vector size mismatch applying patch bundle to " + key);
        }
        shard_replacements[shard_file][key] = TensorSpec::from_vector(vec_add(v, delta), spec.dtype);
        touched_shards.insert(shard_file);
    }

    for (const auto& [key, m] : pending_new_matrices) {
        Matrix final_m = m;
        const std::size_t ud_ff = static_cast<std::size_t>(std::max(0, new_d_ff));
        if ((key.find(".mlp.gate_proj.") != std::string::npos ||
             key.find(".mlp.up_proj.") != std::string::npos) &&
            final_m.rows < ud_ff) {
            final_m = pad_zero_matrix(final_m, ud_ff, final_m.cols);
        } else if (key.find(".mlp.down_proj.") != std::string::npos && final_m.cols < ud_ff) {
            final_m = pad_zero_matrix(final_m, final_m.rows, ud_ff);
        }
        shard_additions[target_shard][key] = TensorSpec::from_matrix(final_m, default_dtype);
        layout.weight_map[key] = target_shard;
        touched_shards.insert(target_shard);
    }
    for (const auto& [key, v] : pending_new_vectors) {
        shard_additions[target_shard][key] = TensorSpec::from_vector(v, default_dtype);
        layout.weight_map[key] = target_shard;
        touched_shards.insert(target_shard);
    }

    if (new_d_ff > cfg.intermediate_size) {
        const std::size_t ud_ff = static_cast<std::size_t>(new_d_ff);
        for (int l = 0; l < cfg.num_layers; ++l) {
            const std::string prefix = "model.layers." + std::to_string(l);
            for (const char* suf : {".mlp.gate_proj.weight", ".mlp.up_proj.weight"}) {
                const std::string key = prefix + suf;
                auto wmit = layout.weight_map.find(key);
                if (wmit == layout.weight_map.end()) {
                    continue;
                }
                const std::string& sf = wmit->second;
                if (shard_replacements[sf].contains(key)) {
                    continue;
                }
                const TensorSpec spec = get_reader(sf).read_tensor(key);
                Matrix m = spec.to_matrix();
                if (m.rows < ud_ff) {
                    m = pad_zero_matrix(m, ud_ff, m.cols);
                    shard_replacements[sf][key] = TensorSpec::from_matrix(m, spec.dtype);
                    touched_shards.insert(sf);
                }
            }
            const std::string dk = prefix + ".mlp.down_proj.weight";
            auto dmit = layout.weight_map.find(dk);
            if (dmit != layout.weight_map.end()) {
                const std::string& sf = dmit->second;
                if (!shard_replacements[sf].contains(dk)) {
                    const TensorSpec spec = get_reader(sf).read_tensor(dk);
                    Matrix m = spec.to_matrix();
                    if (m.cols < ud_ff) {
                        m = pad_zero_matrix(m, m.rows, ud_ff);
                        shard_replacements[sf][dk] = TensorSpec::from_matrix(m, spec.dtype);
                        touched_shards.insert(sf);
                    }
                }
            }
        }
    }

    reader_cache.clear();

    std::error_code eq_ec;
    const bool same_dir = std::filesystem::equivalent(src_dir, dst_dir, eq_ec);

    std::int64_t total_byte_delta = 0;
    for (const std::string& shard_file : layout.shard_order) {
        const std::filesystem::path src_shard = src_dir / shard_file;
        const std::filesystem::path dst_shard = dst_dir / shard_file;
        if (touched_shards.contains(shard_file)) {
            std::int64_t shard_delta = 0;
            rewrite_safetensors_selective(
                src_shard.string(),
                dst_shard.string(),
                shard_replacements[shard_file],
                shard_additions[shard_file],
                {},
                &shard_delta);
            total_byte_delta += shard_delta;
        } else if (!same_dir) {
            std::filesystem::copy_file(
                src_shard,
                dst_shard,
                std::filesystem::copy_options::overwrite_existing);
        }
    }

    cfg_json["aic_threshold_routes"] = has_threshold_route_tensor(layout.weight_map);
    write_aux_checkpoint_files(
        src_dir,
        dst_dir,
        same_dir,
        layout.is_sharded,
        layout.index_json,
        layout.weight_map,
        total_byte_delta,
        cfg_json,
        new_num_layers,
        new_d_ff);
}

void revert_patch_bundle_streaming(
    const std::string& patched_model_dir,
    const std::string& restored_model_dir,
    const PatchBundle& bundle) {
    const std::filesystem::path src_dir(patched_model_dir);
    const std::filesystem::path dst_dir(restored_model_dir);
    std::filesystem::create_directories(dst_dir);

    JsonValue cfg_json = parse_json(read_text_file(src_dir / "config.json"));
    ModelConfig cfg = ModelConfig::from_json(cfg_json);

    if (!bundle.arch_name.empty() && !cfg.arch_name.empty() &&
        bundle.arch_name != cfg.arch_name) {
        throw StrataError(
            ErrorKind::FrameMismatch,
            "PatchBundle arch_name '" + bundle.arch_name +
            "' mismatches patched model arch_name '" + cfg.arch_name + "'");
    }
    if (bundle.hidden_size > 0 && cfg.hidden_size != bundle.hidden_size) {
        throw StrataError(
            ErrorKind::IO,
            "PatchBundle hidden_size mismatches patched model hidden_size");
    }

    ShardLayout layout = load_shard_layout(src_dir);

    std::map<std::string, std::unique_ptr<MmapSafetensors>> reader_cache;
    auto get_reader = [&](const std::string& shard_file) -> MmapSafetensors& {
        auto it = reader_cache.find(shard_file);
        if (it == reader_cache.end()) {
            auto ptr = std::make_unique<MmapSafetensors>((src_dir / shard_file).string());
            it = reader_cache.emplace(shard_file, std::move(ptr)).first;
        }
        return *(it->second);
    };

    const bool shrink_elastic = (bundle.target_mode == "elastic");
    const int target_layers = (shrink_elastic && bundle.orig_num_layers > 0 &&
                               cfg.num_layers > bundle.orig_num_layers)
        ? bundle.orig_num_layers
        : cfg.num_layers;
    const int target_dff = (shrink_elastic && bundle.orig_intermediate_size > 0 &&
                            cfg.intermediate_size > bundle.orig_intermediate_size)
        ? bundle.orig_intermediate_size
        : cfg.intermediate_size;

    std::set<std::string> touched_shards;
    std::map<std::string, std::map<std::string, TensorSpec>> shard_replacements;
    std::map<std::string, std::set<std::string>> shard_removals;
    std::map<std::string, std::string> out_weight_map = layout.weight_map;

    if (target_layers < cfg.num_layers) {
        for (const auto& [tname, sfile] : layout.weight_map) {
            if (extract_layer_index(tname) >= target_layers) {
                shard_removals[sfile].insert(tname);
                out_weight_map.erase(tname);
                touched_shards.insert(sfile);
            }
        }
    }

    const std::size_t ud_ff = static_cast<std::size_t>(std::max(0, target_dff));

    for (const auto& [key, delta] : bundle.matrix_deltas) {
        if (extract_layer_index(key) >= target_layers) {
            continue;
        }
        if (is_threshold_route_tensor(key) && !bundle.base_checksums.contains(key)) {
            auto wmit = out_weight_map.find(key);
            if (wmit != out_weight_map.end()) {
                shard_removals[wmit->second].insert(key);
                touched_shards.insert(wmit->second);
                out_weight_map.erase(wmit);
            }
            continue;
        }
        auto wmit = out_weight_map.find(key);
        if (wmit == out_weight_map.end()) {
            throw StrataError(ErrorKind::IO, "Missing matrix when reverting patch bundle: " + key);
        }
        const std::string& sf = wmit->second;
        const TensorSpec spec = get_reader(sf).read_tensor(key);
        Matrix m = spec.to_matrix();
        if (m.rows != delta.rows || m.cols != delta.cols) {
            throw StrataError(ErrorKind::IO, "Matrix shape mismatch reverting patch bundle on " + key);
        }
        m = mat_sub(m, delta);
        if (target_dff < cfg.intermediate_size) {
            if ((key.find(".mlp.gate_proj.") != std::string::npos ||
                 key.find(".mlp.up_proj.") != std::string::npos) &&
                m.rows > ud_ff) {
                m = crop_matrix(m, ud_ff, m.cols);
            } else if (key.find(".mlp.down_proj.") != std::string::npos && m.cols > ud_ff) {
                m = crop_matrix(m, m.rows, ud_ff);
            }
        }
        shard_replacements[sf][key] = TensorSpec::from_matrix(m, spec.dtype);
        touched_shards.insert(sf);
    }

    for (const auto& [key, delta] : bundle.vector_deltas) {
        if (extract_layer_index(key) >= target_layers) {
            continue;
        }
        if (is_threshold_route_tensor(key) && !bundle.base_checksums.contains(key)) {
            auto wmit = out_weight_map.find(key);
            if (wmit != out_weight_map.end()) {
                shard_removals[wmit->second].insert(key);
                touched_shards.insert(wmit->second);
                out_weight_map.erase(wmit);
            }
            continue;
        }
        auto wmit = out_weight_map.find(key);
        if (wmit == out_weight_map.end()) {
            throw StrataError(ErrorKind::IO, "Missing vector when reverting patch bundle: " + key);
        }
        const std::string& sf = wmit->second;
        const TensorSpec spec = get_reader(sf).read_tensor(key);
        Vector v = spec.to_vector();
        if (v.size() != delta.size()) {
            throw StrataError(ErrorKind::IO, "Vector size mismatch reverting patch bundle on " + key);
        }
        shard_replacements[sf][key] = TensorSpec::from_vector(vec_sub(v, delta), spec.dtype);
        touched_shards.insert(sf);
    }

    if (target_dff < cfg.intermediate_size) {
        for (int l = 0; l < target_layers; ++l) {
            const std::string prefix = "model.layers." + std::to_string(l);
            for (const char* suf : {".mlp.gate_proj.weight", ".mlp.up_proj.weight"}) {
                const std::string key = prefix + suf;
                auto wmit = out_weight_map.find(key);
                if (wmit == out_weight_map.end()) {
                    continue;
                }
                const std::string& sf = wmit->second;
                if (shard_replacements[sf].contains(key)) {
                    continue;
                }
                const TensorSpec spec = get_reader(sf).read_tensor(key);
                Matrix m = spec.to_matrix();
                if (m.rows > ud_ff) {
                    m = crop_matrix(m, ud_ff, m.cols);
                    shard_replacements[sf][key] = TensorSpec::from_matrix(m, spec.dtype);
                    touched_shards.insert(sf);
                }
            }
            const std::string dk = prefix + ".mlp.down_proj.weight";
            auto dmit = out_weight_map.find(dk);
            if (dmit != out_weight_map.end()) {
                const std::string& sf = dmit->second;
                if (!shard_replacements[sf].contains(dk)) {
                    const TensorSpec spec = get_reader(sf).read_tensor(dk);
                    Matrix m = spec.to_matrix();
                    if (m.cols > ud_ff) {
                        m = crop_matrix(m, m.rows, ud_ff);
                        shard_replacements[sf][dk] = TensorSpec::from_matrix(m, spec.dtype);
                        touched_shards.insert(sf);
                    }
                }
            }
        }
    }

    reader_cache.clear();

    std::error_code eq_ec;
    const bool same_dir = std::filesystem::equivalent(src_dir, dst_dir, eq_ec);

    std::int64_t total_byte_delta = 0;
    for (const std::string& shard_file : layout.shard_order) {
        const std::filesystem::path src_shard = src_dir / shard_file;
        const std::filesystem::path dst_shard = dst_dir / shard_file;
        if (touched_shards.contains(shard_file)) {
            std::int64_t shard_delta = 0;
            rewrite_safetensors_selective(
                src_shard.string(),
                dst_shard.string(),
                shard_replacements[shard_file],
                {},
                shard_removals[shard_file],
                &shard_delta);
            total_byte_delta += shard_delta;
        } else if (!same_dir) {
            std::filesystem::copy_file(
                src_shard,
                dst_shard,
                std::filesystem::copy_options::overwrite_existing);
        }
    }

    cfg_json["aic_threshold_routes"] = has_threshold_route_tensor(out_weight_map);
    write_aux_checkpoint_files(
        src_dir,
        dst_dir,
        same_dir,
        layout.is_sharded,
        layout.index_json,
        out_weight_map,
        total_byte_delta,
        cfg_json,
        target_layers,
        target_dff);
}

}
