#include "aic/accel_metal.hpp"
#include "aic/atlas.hpp"
#include "aic/cir.hpp"
#include "aic/codegen.hpp"
#include "aic/decompiler.hpp"
#include "aic/errors.hpp"
#include "aic/formatter.hpp"
#include "aic/json_min.hpp"
#include "aic/linker.hpp"
#include "aic/linter.hpp"
#include "aic/mir.hpp"
#include "aic/nelf.hpp"
#include "aic/patch_bundle.hpp"
#include "aic/runtime.hpp"
#include "aic/safetensors.hpp"
#include "aic/tensor.hpp"
#include "aic/tokenizer.hpp"
#include "aic/tracer.hpp"
#include "aic/wisa.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;

namespace {

std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw aic::IOError("Failed to open file for reading: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

void write_text_file(const std::string& path, const std::string& content) {
    const fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
    std::ofstream ofs(path, std::ios::trunc);
    if (!ofs) {
        throw aic::IOError("Failed to open file for writing: " + path);
    }
    ofs << content;
    if (!content.empty() && content.back() != '\n') {
        ofs << '\n';
    }
    if (!ofs.good()) {
        throw aic::IOError("Failed writing file: " + path);
    }
}

std::string to_upper_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
        s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void print_usage(std::ostream& os) {
    os << "Usage:\n"
       << "  aic init-demo-model --out <dir> [--arch qwen3-mini|llama-mini|deepseek-mini] [--dtype F32|BF16|F16|FP8|I8|INT4] [--shards 1] [--seed 42]\n"
       << "  aic check <file.strata> [-I <dir>]\n"
       << "  aic ir <file.strata> [--emit mir|cir|wisa] [--model <dir>] [--atlas <path>] [--target in-place|elastic] [-I <dir>] [--opt-level 0|1|2]\n"
       << "  aic compile <file.strata> --model <dir> [--out <out_dir>] [--atlas <path>] [--target in-place|elastic|lora] [--streaming] [--patch-only --emit-patch <file.npatch>] [-I <dir>] [--opt-level 0|1|2]\n"
       << "  aic lift --model <dir> --out <nelf_dir> [--strata-out <file.strata>] [--layers <l0..l1>] [--streaming]\n"
       << "  aic build-nelf <file.strata> --nelf <nelf_dir> --out <out_dir> [--target in-place|elastic] [-I <dir>] [--verify-orig <orig_dir>]\n"
       << "  aic link <file.strata> --target-model <dir> --out <out_dir> [--atlas <path>] [--target in-place|elastic|lora] [-I <dir>] [--opt-level 0|1|2]\n"
       << "  aic fmt <file.strata> [--write] [--check]\n"
       << "  aic lint <file.strata> [-I <dir>]\n"
       << "  aic test <file.strata> --model <dir> [-I <dir>] [--atlas <path>] [--target in-place|elastic|lora] [--opt-level 0|1|2]\n"
       << "  aic verify --orig <dir> --patched <dir> [--samples <n>]\n"
       << "  aic inspect --model <dir>\n"
       << "  aic run --model <dir> --prompt <text> [--max-tokens 8] [--temp 0.0] [--top-p 1.0] [--lora <lora_dir>] [--streaming]\n"
       << "  aic trace --model <dir> --prompt <text> [--strata <file.strata>] [-I <dir>] [--atlas <path>] [--lora <lora_dir>] [--gen-tokens <N>] [--compare <other_model_dir>]\n"
       << "  aic diff-nelf --lhs <nelf_dir> --rhs <nelf_dir>\n"
       << "  aic patch-apply <file.npatch> --model <dir> --out <out_dir> [--streaming]\n"
       << "  aic patch-revert <file.npatch> --model <patched_dir> --out <restored_dir> [--streaming]\n"
       << "  aic patch-compose <p1.npatch> <p2.npatch> --out <out.npatch>\n"
       << "  aic bench [--dim 512] [--layers 4]\n";
}

struct ParsedArgs {
    std::vector<std::string> positional;
    std::map<std::string, std::string> options;
};

ParsedArgs parse_args(
    int argc,
    char** argv,
    int start_idx,
    const std::set<std::string>& allowed_options,
    const std::set<std::string>& bool_flags = {}) {
    ParsedArgs out;
    for (int i = start_idx; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            out.options["help"] = "1";
            continue;
        }
        if (arg == "-I") {
            if (i + 1 < argc) {
                ++i;
            }
            continue;
        }
        if (arg.size() > 2 && arg[0] == '-' && arg[1] == 'I') {
            continue;
        }
        if (arg.size() > 2 && arg[0] == '-' && arg[1] == '-') {
            const std::size_t eq_pos = arg.find('=');
            std::string key;
            std::string val;
            if (eq_pos != std::string::npos) {
                key = arg.substr(2, eq_pos - 2);
                val = arg.substr(eq_pos + 1);
                if (!allowed_options.contains(key) && !bool_flags.contains(key)) {
                    throw std::invalid_argument("Unknown option --" + key);
                }
            } else {
                key = arg.substr(2);
                if (bool_flags.contains(key)) {
                    val = "1";
                } else if (allowed_options.contains(key)) {
                    if (i + 1 >= argc) {
                        throw std::invalid_argument("Missing value for option --" + key);
                    }
                    val = argv[++i];
                } else {
                    throw std::invalid_argument("Unknown option --" + key);
                }
            }
            out.options[key] = val;
        } else {
            out.positional.push_back(arg);
        }
    }
    return out;
}

std::string require_option(const ParsedArgs& args, const std::string& key, const std::string& cmd) {
    auto it = args.options.find(key);
    if (it == args.options.end() || it->second.empty()) {
        throw std::invalid_argument("Command '" + cmd + "' requires --" + key);
    }
    return it->second;
}

std::string get_option_or(const ParsedArgs& args, const std::string& key, const std::string& fallback) {
    auto it = args.options.find(key);
    if (it == args.options.end()) {
        return fallback;
    }
    return it->second;
}

bool get_bool_flag(const ParsedArgs& args, const std::string& key) {
    auto it = args.options.find(key);
    if (it == args.options.end()) {
        return false;
    }
    return it->second == "1" || it->second == "true" || it->second == "yes";
}

void validate_target_mode(const std::string& target) {
    if (target != "in-place" && target != "elastic" && target != "lora") {
        throw std::invalid_argument("Invalid --target '" + target + "' (expected in-place, elastic, or lora)");
    }
}

std::string validate_dtype(const std::string& raw) {
    const std::string dt = to_upper_ascii(raw);
    if (dt == "FP8") return "F8_E4M3";
    if (dt == "INT4" || dt == "I4") return "I4";
    if (dt != "F32" && dt != "BF16" && dt != "F16" && dt != "I8") {
        throw std::invalid_argument("Invalid --dtype '" + raw + "' (expected F32, BF16, F16, FP8, I8, or INT4)");
    }
    return dt;
}

aic::ModelConfig make_demo_config(const std::string& arch, const std::string& dtype) {
    aic::ModelConfig cfg;
    cfg.torch_dtype = dtype;
    if (arch == "qwen3-mini") {
        cfg.num_layers = 8;
        cfg.hidden_size = 32;
        cfg.intermediate_size = 96;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.vocab_size = 64;
        cfg.attention_type = "gqa";
        cfg.kv_lora_rank = 8;
        cfg.arch_name = "qwen3-mini";
        cfg.model_type = "qwen3";
        cfg.use_rope = true;
        cfg.qk_norm = true;
        return cfg;
    }
    if (arch == "llama-mini") {
        cfg.num_layers = 4;
        cfg.hidden_size = 24;
        cfg.intermediate_size = 64;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 2;
        cfg.vocab_size = 64;
        cfg.attention_type = "gqa";
        cfg.kv_lora_rank = 8;
        cfg.arch_name = "llama-mini";
        cfg.model_type = "llama";
        cfg.use_rope = true;
        cfg.qk_norm = false;
        return cfg;
    }
    if (arch == "mla-mini" || arch == "deepseek-mini") {
        cfg.num_layers = 4;
        cfg.hidden_size = 24;
        cfg.intermediate_size = 64;
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4;
        cfg.vocab_size = 64;
        cfg.attention_type = "mla";
        cfg.kv_lora_rank = 8;
        cfg.arch_name = "deepseek-mini";
        cfg.model_type = "deepseek_v3";
        cfg.use_rope = true;
        cfg.qk_norm = false;
        return cfg;
    }
    throw std::invalid_argument(
        "Unsupported --arch '" + arch + "' (expected qwen3-mini, llama-mini, or deepseek-mini)");
}

std::uint64_t parse_u64(const std::string& s, const std::string& opt_name) {
    try {
        std::size_t idx = 0;
        const unsigned long long v = std::stoull(s, &idx, 10);
        if (idx != s.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<std::uint64_t>(v);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid integer for --" + opt_name + ": " + s);
    }
}

int parse_int_strict(const std::string& s, const std::string& opt_name) {
    try {
        std::size_t idx = 0;
        const long v = std::stol(s, &idx, 10);
        if (idx != s.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return static_cast<int>(v);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid integer for --" + opt_name + ": " + s);
    }
}

std::size_t tensor_packed_bytes(const std::string& dt, std::size_t num_elems) {
    if (dt == "I4" || dt == "INT4") {
        return (num_elems + 1u) / 2u;
    }
    if (dt == "F8_E4M3" || dt == "FP8" || dt == "I8") {
        return num_elems;
    }
    if (dt == "BF16" || dt == "F16") {
        return num_elems * 2u;
    }
    if (dt == "F32") {
        return num_elems * 4u;
    }
    if (dt == "F64") {
        return num_elems * 8u;
    }
    return num_elems * 4u;
}

std::size_t simulate_shard_count(const std::vector<std::size_t>& sizes, std::size_t cap) {
    if (sizes.empty()) {
        return 1;
    }
    std::size_t shards = 1;
    std::size_t cur = 0;
    for (std::size_t sz : sizes) {
        if (cur > 0 && cap > 0 && cur + sz > cap) {
            ++shards;
            cur = 0;
        }
        cur += sz;
    }
    return shards;
}

std::size_t compute_max_shard_bytes(
    const std::map<std::string, aic::TensorSpec>& specs,
    std::size_t target_shards) {
    if (target_shards <= 1 || specs.size() <= 1) {
        return 0;
    }
    std::vector<std::size_t> sizes;
    sizes.reserve(specs.size());
    std::size_t total_bytes = 0;
    for (const auto& [name, spec] : specs) {
        (void)name;
        const std::size_t b = tensor_packed_bytes(spec.dtype, spec.data.size());
        sizes.push_back(b);
        total_bytes += b;
    }
    const std::size_t clamped_target = std::min(target_shards, specs.size());
    std::size_t lo = 1;
    std::size_t hi = std::max<std::size_t>(1, total_bytes);
    std::size_t best_cap = std::max<std::size_t>(1, total_bytes / clamped_target);
    std::size_t best_diff = specs.size();

    while (lo <= hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        const std::size_t cnt = simulate_shard_count(sizes, mid);
        const std::size_t diff = (cnt > clamped_target) ? (cnt - clamped_target) : (clamped_target - cnt);
        if (diff < best_diff || (diff == best_diff && cnt >= clamped_target)) {
            best_diff = diff;
            best_cap = mid;
        }
        if (cnt > clamped_target) {
            lo = mid + 1;
        } else {
            if (mid == 0) {
                break;
            }
            hi = mid - 1;
        }
    }
    return best_cap;
}

aic::Vector orthogonalize_against(
    aic::Vector v,
    const std::vector<aic::Vector>& basis,
    std::size_t fallback_idx) {
    const std::size_t d = v.size();
    if (d == 0) {
        return {};
    }
    for (int pass = 0; pass < 2; ++pass) {
        for (const auto& b : basis) {
            if (b.size() != d) {
                continue;
            }
            const double p = aic::dot(v, b);
            for (std::size_t i = 0; i < d; ++i) {
                v[i] -= p * b[i];
            }
        }
    }
    if (aic::norm2(v) > 1e-8) {
        return aic::normalize(v);
    }
    for (std::size_t attempt = 0; attempt < d; ++attempt) {
        aic::Vector cand(d, 0.0);
        cand[(fallback_idx + attempt) % d] = 1.0;
        for (int pass = 0; pass < 2; ++pass) {
            for (const auto& b : basis) {
                if (b.size() != d) {
                    continue;
                }
                const double p = aic::dot(cand, b);
                for (std::size_t i = 0; i < d; ++i) {
                    cand[i] -= p * b[i];
                }
            }
        }
        if (aic::norm2(cand) > 1e-8) {
            return aic::normalize(cand);
        }
    }
    aic::Vector e(d, 0.0);
    e[fallback_idx % d] = 1.0;
    return e;
}

void fill_missing_symbols_in_atlas(
    aic::NeuralAtlas& atlas,
    const aic::MIRModule& mir,
    const aic::TransformerModel& model) {
    bool needs_fill = false;
    for (const auto& [name, decl] : mir.declarations) {
        (void)decl;
        if (!atlas.symbols.contains(name)) {
            needs_fill = true;
            break;
        }
    }
    if (!needs_fill && !atlas.frame_bridges.empty()) {
        return;
    }

    const aic::NeuralAtlas def = aic::build_default_atlas(model, &mir, 42);
    for (const auto& [k, m] : def.frame_bridges) {
        if (!atlas.frame_bridges.contains(k)) {
            atlas.frame_bridges[k] = m;
        }
    }

    if (!needs_fill) {
        return;
    }

    const std::size_t d = static_cast<std::size_t>(std::max(0, model.config.hidden_size));
    std::vector<aic::Vector> raw_protected;
    for (const auto& [pname, dirs] : atlas.preserve_directions) {
        (void)pname;
        for (const auto& v : dirs) {
            if (v.size() == d) {
                raw_protected.push_back(v);
            }
        }
    }

    std::set<std::string> preserved_names;
    for (const auto& flow : mir.flows) {
        for (const auto& pn : flow.contract.preserve) {
            preserved_names.insert(pn);
        }
    }

    const aic::NeuralAtlas& def_full = def;
    for (const auto& pn : preserved_names) {
        auto look_in = [&](const std::map<std::string, aic::SymbolBinding>& syms) {
            auto it = syms.find(pn);
            if (it == syms.end()) {
                return;
            }
            const aic::SymbolBinding& sb = it->second;
            for (const auto& v : sb.direction.empty() ? std::vector<aic::Vector>{} : std::vector<aic::Vector>{sb.direction}) {
                if (v.size() == d) { raw_protected.push_back(v); }
            }
            for (const auto& v : sb.subspace_basis) {
                if (v.size() == d) { raw_protected.push_back(v); }
            }
            for (const auto& v : sb.manifold_plane) {
                if (v.size() == d) { raw_protected.push_back(v); }
            }
            for (const auto& m : sb.field_modes) {
                if (m.u.size() == d) { raw_protected.push_back(m.u); }
                if (m.v.size() == d) { raw_protected.push_back(m.v); }
            }
            for (const auto& v : sb.preserve_keys) {
                if (v.size() == d) { raw_protected.push_back(v); }
            }
        };
        look_in(atlas.symbols);
        look_in(def_full.symbols);
    }

    const std::vector<aic::Vector> protected_basis = aic::gram_schmidt(raw_protected, 1e-10);

    std::size_t sym_idx = 0;
    for (const auto& [name, decl] : mir.declarations) {
        (void)decl;
        if (atlas.symbols.contains(name)) {
            ++sym_idx;
            continue;
        }
        auto it = def.symbols.find(name);
        if (it == def.symbols.end()) {
            ++sym_idx;
            continue;
        }
        if (preserved_names.contains(name)) {
            atlas.symbols[name] = it->second;
            ++sym_idx;
            continue;
        }
        aic::SymbolBinding sb = it->second;
        if (d > 0) {
            if (!sb.manifold_plane.empty()) {
                std::vector<aic::Vector> basis = protected_basis;
                std::vector<aic::Vector> new_plane;
                for (std::size_t i = 0; i < sb.manifold_plane.size(); ++i) {
                    aic::Vector v = orthogonalize_against(sb.manifold_plane[i], basis, sym_idx * 7 + i);
                    basis.push_back(v);
                    new_plane.push_back(std::move(v));
                }
                sb.manifold_plane = new_plane;
                sb.subspace_basis = new_plane;
                sb.direction = (new_plane.size() > 1) ? new_plane[1] : new_plane[0];
            } else if (!sb.subspace_basis.empty()) {
                std::vector<aic::Vector> basis = protected_basis;
                std::vector<aic::Vector> new_sub;
                for (std::size_t i = 0; i < sb.subspace_basis.size(); ++i) {
                    aic::Vector v = orthogonalize_against(sb.subspace_basis[i], basis, sym_idx * 7 + i);
                    basis.push_back(v);
                    new_sub.push_back(std::move(v));
                }
                sb.subspace_basis = new_sub;
                sb.direction = new_sub[0];
            } else if (!sb.direction.empty()) {
                sb.direction = orthogonalize_against(sb.direction, protected_basis, sym_idx * 7);
            }
            if (!sb.preserve_keys.empty()) {
                std::vector<aic::Vector> basis = protected_basis;
                for (std::size_t i = 0; i < sb.preserve_keys.size(); ++i) {
                    sb.preserve_keys[i] = orthogonalize_against(
                        sb.preserve_keys[i], basis, sym_idx * 13 + i + 1);
                    basis.push_back(sb.preserve_keys[i]);
                }
            }
        }
        atlas.symbols[name] = std::move(sb);
        ++sym_idx;
    }
}

aic::NeuralAtlas resolve_atlas_for_compile(
    const aic::MIRModule& mir,
    const aic::TransformerModel& model,
    const std::string& atlas_path,
    const std::string& model_dir) {
    if (!atlas_path.empty()) {
        aic::NeuralAtlas atlas = aic::NeuralAtlas::load(atlas_path);
        fill_missing_symbols_in_atlas(atlas, mir, model);
        return atlas;
    }

    aic::NeuralAtlas atlas;
    if (!model_dir.empty()) {
        const fs::path default_atlas_path = fs::path(model_dir) / "model.natlas";
        if (fs::exists(default_atlas_path)) {
            atlas = aic::NeuralAtlas::load(default_atlas_path.string());
            fill_missing_symbols_in_atlas(atlas, mir, model);
        } else {
            atlas = aic::build_default_atlas(model, &mir, 42);
        }
    } else {
        atlas = aic::build_default_atlas(model, &mir, 42);
    }
    if (model.matrices.empty()) {
        return atlas;
    }

    aic::Tokenizer tokenizer;
    const aic::Tokenizer* tokenizer_ptr = nullptr;
    const fs::path tokenizer_path = fs::path(model_dir) / "tokenizer.json";
    if (!model_dir.empty() && fs::exists(tokenizer_path)) {
        tokenizer = aic::Tokenizer::load_from_file(tokenizer_path.string());
        tokenizer_ptr = &tokenizer;
    }
    const aic::NeuralAtlas resolved = aic::resolve_symbols_for_target(
        mir, model, nullptr, 42, tokenizer_ptr);
    for (const auto& [name, decl] : mir.declarations) {
        if (decl.kind != "pole" && decl.kind != "dipole") {
            continue;
        }
        const bool has_contrastive_pair = std::any_of(
            decl.anchors.begin(), decl.anchors.end(), [](const aic::MIRAnchor& anchor) {
                return anchor.kind == "contrastive" &&
                    anchor.named_args.contains("pos") && anchor.named_args.contains("neg");
            });
        if (has_contrastive_pair && resolved.symbols.contains(name)) {
            atlas.symbols[name] = resolved.symbols.at(name);
        }
    }
    return atlas;
}

void copy_tokenizer_if_present(const std::string& src_dir, const std::string& dst_dir) {
    std::error_code ec;
    if (fs::equivalent(src_dir, dst_dir, ec)) {
        return;
    }
    const fs::path tok_src = fs::path(src_dir) / "tokenizer.json";
    if (fs::exists(tok_src)) {
        fs::create_directories(dst_dir);
        fs::copy_file(
            tok_src,
            fs::path(dst_dir) / "tokenizer.json",
            fs::copy_options::overwrite_existing);
    }
}

std::pair<int, int> parse_layer_filter(const std::string& spec) {
    if (spec.empty()) {
        return {-1, -1};
    }
    bool inclusive = false;
    std::size_t sep = spec.find("..=");
    std::size_t sep_len = 3;
    if (sep != std::string::npos) {
        inclusive = true;
    } else {
        sep = spec.find("..");
        sep_len = 2;
        if (sep == std::string::npos) {
            sep = spec.find(':');
            sep_len = 1;
        }
    }
    if (sep == std::string::npos) {
        const int idx = parse_int_strict(spec, "layers");
        if (idx < 0) {
            throw std::invalid_argument("--layers start must be non-negative");
        }
        return {idx, idx + 1};
    }
    const int l0 = parse_int_strict(spec.substr(0, sep), "layers");
    int l1 = parse_int_strict(spec.substr(sep + sep_len), "layers");
    if (inclusive) {
        ++l1;
    }
    if (l0 < 0 || l1 <= l0) {
        throw std::invalid_argument("Invalid --layers range: " + spec);
    }
    return {l0, l1};
}

int parse_opt_level(const ParsedArgs& args) {
    const std::string s = get_option_or(args, "opt-level", "2");
    const int v = parse_int_strict(s, "opt-level");
    if (v < 0 || v > 2) {
        throw std::invalid_argument("--opt-level must be 0, 1, or 2");
    }
    return v;
}

std::vector<std::string> collect_include_dirs(int argc, char** argv, int start_idx) {
    std::vector<std::string> dirs;
    for (int i = start_idx; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-I" && i + 1 < argc) {
            dirs.push_back(argv[++i]);
        } else if (arg.size() > 2 && arg[0] == '-' && arg[1] == 'I') {
            dirs.push_back(arg.substr(2));
        }
    }
    return dirs;
}

int cmd_init_demo_model(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"out", "arch", "dtype", "shards", "seed"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const std::string out_dir = require_option(args, "out", "init-demo-model");
    const std::string arch = get_option_or(args, "arch", "qwen3-mini");
    const std::string dtype = validate_dtype(get_option_or(args, "dtype", "F32"));
    const int shards = parse_int_strict(get_option_or(args, "shards", "1"), "shards");
    if (shards < 1) {
        throw std::invalid_argument("--shards must be >= 1");
    }
    const std::uint64_t seed = parse_u64(get_option_or(args, "seed", "42"), "seed");

    const aic::ModelConfig cfg = make_demo_config(arch, dtype);
    const aic::TransformerModel model = aic::TransformerModel::create_synthetic(cfg, seed);

    if (shards > 1 || dtype != "F32") {
        fs::create_directories(out_dir);
        write_text_file((fs::path(out_dir) / "config.json").string(), aic::dump_json(cfg.to_json(), 2));

        std::map<std::string, aic::TensorSpec> specs;
        for (const auto& [name, mat] : model.matrices) {
            specs[name] = aic::TensorSpec::from_matrix(mat, dtype);
        }
        for (const auto& [name, vec] : model.vectors) {
            specs[name] = aic::TensorSpec::from_vector(vec, dtype);
        }
        const std::size_t max_shard_bytes = compute_max_shard_bytes(specs, static_cast<std::size_t>(shards));
        aic::save_safetensors_sharded(
            out_dir,
            specs,
            max_shard_bytes,
            {{"arch_name", cfg.arch_name}, {"attention_type", cfg.attention_type}, {"torch_dtype", dtype}});
    } else {
        model.save_checkpoint(out_dir);
    }

    const aic::NeuralAtlas atlas = aic::build_default_atlas(model, nullptr, seed);
    atlas.save((fs::path(out_dir) / "model.natlas").string());

    const aic::Tokenizer tok = aic::Tokenizer::create_byte_level_demo(cfg.vocab_size);
    tok.save((fs::path(out_dir) / "tokenizer.json").string());

    std::cout << "Initialized " << cfg.arch_name
              << " (layers=" << cfg.num_layers
              << ", hidden=" << cfg.hidden_size
              << ", intermediate=" << cfg.intermediate_size
              << ", attn=" << cfg.attention_type
              << ", dtype=" << dtype
              << ", shards=" << shards
              << ", rope=" << (cfg.use_rope ? "true" : "false")
              << ", qk_norm=" << (cfg.qk_norm ? "true" : "false")
              << ") in " << out_dir << '\n';
    return 0;
}

int cmd_check(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'check' requires <file.strata>");
    }

    const std::string& file_path = args.positional[0];
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);
    const std::string source = read_text_file(file_path);
    try {
        const aic::MIRModule mir = inc_dirs.empty()
            ? aic::compile_source_to_mir(source, file_path)
            : aic::compile_file_with_imports(file_path, inc_dirs);
        std::cout << "OK: module " << mir.name
                  << " (declarations=" << mir.declarations.size()
                  << ", flows=" << mir.flows.size() << ")\n";
        return 0;
    } catch (const aic::StrataError& e) {
        std::cerr << aic::format_diagnostic(e, source, file_path);
        return 1;
    }
}

int cmd_ir(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"emit", "model", "atlas", "target", "opt-level"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'ir' requires <file.strata>");
    }

    const std::string& file_path = args.positional[0];
    const std::string emit = get_option_or(args, "emit", "mir");
    const std::string model_dir = get_option_or(args, "model", "");
    const std::string atlas_path = get_option_or(args, "atlas", "");
    const std::string target = get_option_or(args, "target", "in-place");
    const int opt_level = parse_opt_level(args);
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);

    if (emit != "mir" && emit != "cir" && emit != "wisa") {
        throw std::invalid_argument("Invalid --emit '" + emit + "' (expected mir, cir, or wisa)");
    }
    validate_target_mode(target);

    const std::string source = read_text_file(file_path);
    const aic::MIRModule mir = inc_dirs.empty()
        ? aic::compile_source_to_mir(source, file_path)
        : aic::compile_file_with_imports(file_path, inc_dirs);
    if (emit == "mir") {
        std::cout << aic::format_mir(mir);
        return 0;
    }

    const aic::TransformerModel model = model_dir.empty()
        ? aic::TransformerModel::create_synthetic(aic::ModelConfig{}, 42)
        : aic::TransformerModel::load_checkpoint(model_dir);

    aic::NeuralAtlas atlas = resolve_atlas_for_compile(mir, model, atlas_path, model_dir);
    aic::CIRModule cir = aic::lower_mir_to_cir(mir, model.config, atlas, target);
    if (opt_level >= 1) {
        cir = aic::optimize_cir(cir, opt_level);
    }
    if (emit == "cir") {
        std::cout << aic::format_cir(cir);
        return 0;
    }

    std::vector<aic::WISAInstruction> wisa = aic::lower_cir_to_wisa(cir, model.config, atlas, target);
    if (opt_level >= 1) {
        wisa = aic::optimize_wisa(wisa, opt_level);
    }
    std::cout << aic::format_wisa(wisa);
    return 0;
}

int cmd_compile(int argc, char** argv) {
    const ParsedArgs args = parse_args(
        argc, argv, 2, {"model", "out", "atlas", "target", "opt-level", "emit-patch"}, {"streaming", "patch-only"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'compile' requires <file.strata>");
    }

    const std::string& file_path = args.positional[0];
    const std::string model_dir = require_option(args, "model", "compile");
    const bool patch_only = get_bool_flag(args, "patch-only");
    const std::string out_dir = get_option_or(args, "out", "");
    const std::string atlas_path = get_option_or(args, "atlas", "");
    const std::string target = get_option_or(args, "target", "in-place");
    const bool streaming = get_bool_flag(args, "streaming") || patch_only;
    const int opt_level = parse_opt_level(args);
    const std::string emit_patch_path = get_option_or(args, "emit-patch", "");
    if (patch_only && emit_patch_path.empty()) {
        throw std::invalid_argument("--patch-only requires --emit-patch");
    }
    if (patch_only && target != "in-place") {
        throw std::invalid_argument("--patch-only currently requires --target in-place");
    }
    if (!patch_only && out_dir.empty()) {
        throw std::invalid_argument("Command 'compile' requires --out unless --patch-only is used");
    }
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);
    validate_target_mode(target);

    aic::MIRModule mir;
    if (!inc_dirs.empty()) {
        mir = aic::compile_file_with_imports(file_path, inc_dirs);
    } else {
        const std::string source = read_text_file(file_path);
        mir = aic::compile_source_to_mir(source, file_path);
    }

    if (target == "lora") {
        const aic::TransformerModel model = aic::TransformerModel::load_checkpoint(model_dir);
        aic::NeuralAtlas atlas = resolve_atlas_for_compile(mir, model, atlas_path, model_dir);
        if (atlas_path.empty()) {
            const fs::path tok_p_lora = fs::path(model_dir) / "tokenizer.json";
            const aic::Tokenizer tok_lora = fs::exists(tok_p_lora)
                ? aic::Tokenizer::load_from_file(tok_p_lora.string())
                : aic::Tokenizer::create_byte_level_demo(model.config.vocab_size);
            aic::calibrate_atlas_empirical(atlas, model, &mir, &tok_lora);
        }
        const aic::CIRModule cir_opt = opt_level >= 1
            ? aic::optimize_cir(aic::lower_mir_to_cir(mir, model.config, atlas, target), opt_level)
            : aic::lower_mir_to_cir(mir, model.config, atlas, target);
        std::vector<aic::WISAInstruction> wisa = aic::lower_cir_to_wisa(cir_opt, model.config, atlas, target);
        if (opt_level >= 1) {
            wisa = aic::optimize_wisa(wisa, opt_level);
        }
        const aic::CompileStats stats = aic::emit_lora_adapter(model, wisa, out_dir);
        copy_tokenizer_if_present(model_dir, out_dir);
        if (!emit_patch_path.empty()) {
            const aic::PatchBundle bundle = aic::create_patch_bundle(model, wisa, target, mir.name);
            aic::save_patch_bundle(bundle, emit_patch_path);
        }
        std::cout << "Compiled (lora) " << file_path << " -> " << out_dir
                  << " (target=lora, opt_level=" << opt_level
                  << ", instructions=" << stats.instructions_applied
                  << ", added_poles=" << stats.added_poles << ")\n";
        return 0;
    }

    if (streaming) {
        const fs::path cfg_path = fs::path(model_dir) / "config.json";
        const aic::ModelConfig cfg = aic::ModelConfig::from_json(aic::parse_json(read_text_file(cfg_path.string())));
        aic::NeuralAtlas atlas;
        if (!atlas_path.empty()) {
            aic::TransformerModel cfg_only;
            cfg_only.config = cfg;
            atlas = resolve_atlas_for_compile(mir, cfg_only, atlas_path, model_dir);
        } else {
            const fs::path tok_path = fs::path(model_dir) / "tokenizer.json";
            const aic::Tokenizer tok = fs::exists(tok_path)
                ? aic::Tokenizer::load_from_file(tok_path.string())
                : aic::Tokenizer::create_byte_level_demo(cfg.vocab_size);
            atlas = aic::resolve_symbols_for_target_streaming(
                mir, model_dir, nullptr, 42, &tok);
        }
        aic::CIRModule cir = aic::lower_mir_to_cir(mir, cfg, atlas, target);
        if (opt_level >= 1) {
            cir = aic::optimize_cir(cir, opt_level);
        }
        std::vector<aic::WISAInstruction> wisa = aic::lower_cir_to_wisa(cir, cfg, atlas, target);
        if (opt_level >= 1) {
            wisa = aic::optimize_wisa(wisa, opt_level);
        }
        if (patch_only) {
            const aic::PatchBundle bundle = aic::create_patch_bundle_streaming(
                model_dir, wisa, target, mir.name);
            aic::save_patch_bundle(bundle, emit_patch_path);
            std::cout << "Compiled (patch-only streaming) " << file_path
                      << " -> " << emit_patch_path
                      << " (target=" << target
                      << ", opt_level=" << opt_level
                      << ", instructions=" << wisa.size() << ")\n";
            return 0;
        }

        aic::CompileStats stats = aic::apply_wisa_streaming_checkpoint(model_dir, out_dir, wisa, target);
        for (const auto& flow : cir.flows) {
            stats.max_approx_error = std::max(stats.max_approx_error, flow.max_approx_error);
            stats.max_crosstalk = std::max(stats.max_crosstalk, flow.max_crosstalk);
        }

        const aic::ModelConfig updated_cfg = aic::ModelConfig::from_json(
            aic::parse_json(read_text_file((fs::path(out_dir) / "config.json").string())));
        atlas.num_layers = updated_cfg.num_layers;
        atlas.save((fs::path(out_dir) / "model.natlas").string());
        copy_tokenizer_if_present(model_dir, out_dir);

        if (!emit_patch_path.empty()) {
            const aic::PatchBundle bundle = aic::create_patch_bundle_streaming(
                model_dir, wisa, target, mir.name);
            aic::save_patch_bundle(bundle, emit_patch_path);
        }

        std::cout << "Compiled (streaming) " << file_path << " -> " << out_dir
                  << " (target=" << target
                  << ", opt_level=" << opt_level
                  << ", instructions=" << stats.instructions_applied
                  << ", added_poles=" << stats.added_poles
                  << ", added_layers=" << stats.added_layers
                  << ", max_approx=" << stats.max_approx_error
                  << ", max_crosstalk=" << stats.max_crosstalk << ")\n";
        return 0;
    }

    const aic::TransformerModel model = aic::TransformerModel::load_checkpoint(model_dir);
    aic::NeuralAtlas atlas = resolve_atlas_for_compile(mir, model, atlas_path, model_dir);
    if (atlas_path.empty()) {
        const fs::path tok_p_full = fs::path(model_dir) / "tokenizer.json";
        const aic::Tokenizer tok_full = fs::exists(tok_p_full)
            ? aic::Tokenizer::load_from_file(tok_p_full.string())
            : aic::Tokenizer::create_byte_level_demo(model.config.vocab_size);
        aic::calibrate_atlas_empirical(atlas, model, &mir, &tok_full);
    }

    const aic::CompileOutput compiled = aic::compile_mir_to_model(mir, model, &atlas, target, opt_level);
    compiled.model.save_checkpoint(out_dir);

    atlas.num_layers = compiled.model.config.num_layers;
    atlas.save((fs::path(out_dir) / "model.natlas").string());
    copy_tokenizer_if_present(model_dir, out_dir);

    if (!emit_patch_path.empty()) {
        const aic::TransformerModel saved_patched = aic::TransformerModel::load_checkpoint(out_dir);
        aic::PatchBundle bundle = aic::create_patch_bundle(model, compiled.wisa, target, mir.name);
        for (const auto& [name, mat] : saved_patched.matrices) {
            const auto base_it = model.matrices.find(name);
            if (base_it == model.matrices.end()) continue;
            if (mat.rows != base_it->second.rows || mat.cols != base_it->second.cols) continue;
            const aic::Matrix delta = aic::mat_sub(mat, base_it->second);
            const double fn = aic::frob_norm(delta);
            if (fn > 1e-15) {
                bundle.matrix_deltas[name] = delta;
            }
        }
        for (const auto& [name, vec] : saved_patched.vectors) {
            const auto base_it = model.vectors.find(name);
            if (base_it == model.vectors.end()) continue;
            if (vec.size() != base_it->second.size()) continue;
            aic::Vector delta(vec.size());
            double sq = 0.0;
            for (std::size_t i = 0; i < vec.size(); ++i) {
                delta[i] = vec[i] - base_it->second[i];
                sq += delta[i] * delta[i];
            }
            if (sq > 1e-30) {
                bundle.vector_deltas[name] = delta;
            }
        }
        aic::save_patch_bundle(bundle, emit_patch_path);
    }

    std::cout << "Compiled " << file_path << " -> " << out_dir
              << " (target=" << target
              << ", opt_level=" << opt_level
              << ", instructions=" << compiled.stats.instructions_applied
              << ", added_poles=" << compiled.stats.added_poles
              << ", added_layers=" << compiled.stats.added_layers
              << ", max_approx=" << compiled.stats.max_approx_error
              << ", max_crosstalk=" << compiled.stats.max_crosstalk << ")\n";
    return 0;
}

int cmd_lift(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"model", "out", "strata-out", "layers"}, {"streaming"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const std::string model_dir = require_option(args, "model", "lift");
    const std::string nelf_dir = require_option(args, "out", "lift");
    const std::string strata_out = get_option_or(
        args, "strata-out", (fs::path(nelf_dir) / "lifted.strata").string());
    const std::pair<int, int> layer_filter = parse_layer_filter(get_option_or(args, "layers", ""));
    const bool streaming = get_bool_flag(args, "streaming");

    aic::NeuralELF nelf;
    std::string strata_src;
    if (streaming) {
        auto [n, s] = aic::lift_checkpoint_streaming(model_dir, "LiftedModel", layer_filter);
        nelf = std::move(n);
        strata_src = std::move(s);
    } else {
        const aic::TransformerModel model = aic::TransformerModel::load_checkpoint(model_dir);
        auto [n, s] = aic::lift_model(model, "LiftedModel", layer_filter);
        nelf = std::move(n);
        strata_src = std::move(s);
    }
    nelf.save(nelf_dir);
    strata_src = aic::format_strata_source(strata_src, strata_out);
    write_text_file(strata_out, strata_src);

    const auto ratios = nelf.section_energy_ratios();
    std::cout << "Lifted " << model_dir << " -> " << nelf_dir
              << " (strata=" << strata_out
              << ", streaming=" << (streaming ? "true" : "false")
              << ", .text=" << ratios.at(".text")
              << ", .rodata=" << ratios.at(".rodata")
              << ", .field=" << ratios.at(".field")
              << ", .residual=" << ratios.at(".residual") << ")\n";
    return 0;
}

int cmd_link(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"target-model", "out", "atlas", "target", "opt-level"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'link' requires <file.strata>");
    }

    const std::string& file_path = args.positional[0];
    const std::string target_model_dir = require_option(args, "target-model", "link");
    const std::string out_dir = require_option(args, "out", "link");
    const std::string atlas_path = get_option_or(args, "atlas", "");
    const std::string target = get_option_or(args, "target", "in-place");
    const int opt_level = parse_opt_level(args);
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);
    validate_target_mode(target);

    const std::string source = read_text_file(file_path);
    const aic::MIRModule mir = inc_dirs.empty()
        ? aic::compile_source_to_mir(source, file_path)
        : aic::compile_file_with_imports(file_path, inc_dirs);
    const aic::TransformerModel target_model = aic::TransformerModel::load_checkpoint(target_model_dir);

    aic::NeuralAtlas atlas_storage;
    const aic::NeuralAtlas* atlas_ptr = nullptr;
    if (!atlas_path.empty()) {
        atlas_storage = aic::NeuralAtlas::load(atlas_path);
        atlas_ptr = &atlas_storage;
    }

    aic::Tokenizer tok_storage;
    const aic::Tokenizer* tok_ptr = nullptr;
    const fs::path tok_path = fs::path(target_model_dir) / "tokenizer.json";
    if (fs::exists(tok_path)) {
        tok_storage = aic::Tokenizer::load_from_file(tok_path.string());
        tok_ptr = &tok_storage;
    }

    aic::NeuralAtlas resolved_atlas = aic::resolve_symbols_for_target(mir, target_model, atlas_ptr, 0, tok_ptr);
    aic::calibrate_atlas_empirical(resolved_atlas, target_model, &mir, tok_ptr);

    const aic::CompileOutput linked = aic::compile_mir_to_model(mir, target_model, &resolved_atlas, target, opt_level);
    linked.model.save_checkpoint(out_dir);

    resolved_atlas.num_layers = linked.model.config.num_layers;
    resolved_atlas.save((fs::path(out_dir) / "model.natlas").string());
    copy_tokenizer_if_present(target_model_dir, out_dir);

    std::cout << "Linked " << file_path << " onto " << target_model_dir << " -> " << out_dir
              << " (arch=" << linked.model.config.arch_name
              << ", target=" << target
              << ", tokenizer=" << (tok_ptr != nullptr ? "yes" : "no")
              << ", instructions=" << linked.stats.instructions_applied
              << ", added_poles=" << linked.stats.added_poles
              << ", added_layers=" << linked.stats.added_layers
              << ", max_crosstalk=" << linked.stats.max_crosstalk << ")\n";
    return 0;
}

int cmd_inspect(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"model"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const std::string model_dir = require_option(args, "model", "inspect");
    const fs::path dir_path(model_dir);
    const fs::path cfg_path = dir_path / "config.json";
    aic::ModelConfig cfg = aic::ModelConfig::from_json(aic::parse_json(read_text_file(cfg_path.string())));

    std::size_t shard_count = 0;
    std::size_t tensor_count = 0;
    std::string detected_dtype = cfg.torch_dtype;
    bool has_qk_norm_tensor = false;
    bool tied_absent_lm_head = true;

    const fs::path idx_path = dir_path / "model.safetensors.index.json";
    const fs::path single_path = dir_path / "model.safetensors";
    if (fs::exists(idx_path)) {
        const aic::JsonValue idx = aic::parse_json(read_text_file(idx_path.string()));
        if (idx.is_object() && idx.has("weight_map") && idx["weight_map"].is_object()) {
            std::set<std::string> shards;
            for (const auto& [tname, sfile] : idx["weight_map"].as_object()) {
                ++tensor_count;
                shards.insert(sfile.as_string());
                if (ends_with(tname, ".self_attn.q_norm.weight") ||
                    ends_with(tname, ".self_attn.k_norm.weight")) {
                    has_qk_norm_tensor = true;
                }
                if (tname == "lm_head.weight") {
                    tied_absent_lm_head = false;
                }
            }
            shard_count = shards.size();
            if (!shards.empty()) {
                aic::MmapSafetensors reader((dir_path / *shards.begin()).string());
                const auto names = reader.tensor_names();
                if (!names.empty()) {
                    detected_dtype = reader.tensor_dtype(names[0]);
                }
            }
        }
    } else if (fs::exists(single_path)) {
        shard_count = 1;
        aic::MmapSafetensors reader(single_path.string());
        const auto names = reader.tensor_names();
        tensor_count = names.size();
        for (const auto& tname : names) {
            if (ends_with(tname, ".self_attn.q_norm.weight") ||
                ends_with(tname, ".self_attn.k_norm.weight")) {
                has_qk_norm_tensor = true;
            }
            if (tname == "lm_head.weight") {
                tied_absent_lm_head = false;
            }
        }
        if (!names.empty()) {
            detected_dtype = reader.tensor_dtype(names[0]);
        }
    } else {
        throw aic::IOError("No model.safetensors or model.safetensors.index.json in: " + model_dir);
    }

    const bool qk_norm_eff = cfg.qk_norm || has_qk_norm_tensor;
    const bool tied_eff = cfg.tie_word_embeddings || tied_absent_lm_head;

    std::string tok_status = "absent";
    const fs::path tok_path = dir_path / "tokenizer.json";
    if (fs::exists(tok_path)) {
        const aic::Tokenizer tok = aic::Tokenizer::load_from_file(tok_path.string());
        tok_status = "present (vocab_size=" + std::to_string(tok.vocab_size()) + ")";
    }

    const fs::path atlas_path = dir_path / "model.natlas";
    const bool has_atlas = fs::exists(atlas_path);

#ifdef __APPLE__
    const std::string accel_status = "enabled (Accelerate BLAS/LAPACK)";
#else
    const std::string accel_status = "cpu-fallback";
#endif

    std::cout << "arch: " << cfg.arch_name << '\n'
              << "model_type: " << (cfg.model_type.empty() ? cfg.arch_name : cfg.model_type) << '\n'
              << "num_layers: " << cfg.num_layers << '\n'
              << "hidden_size: " << cfg.hidden_size << '\n'
              << "intermediate_size: " << cfg.intermediate_size << '\n'
              << "num_attention_heads: " << cfg.num_attention_heads << '\n'
              << "num_key_value_heads: " << cfg.num_key_value_heads << '\n'
              << "vocab_size: " << cfg.vocab_size << '\n'
              << "attention_type: " << cfg.attention_type << '\n'
              << "use_rope: " << (cfg.use_rope ? "true" : "false") << '\n'
              << "qk_norm: " << (qk_norm_eff ? "true" : "false") << '\n'
              << "aic_threshold_routes: " << (cfg.aic_threshold_routes ? "true" : "false") << '\n'
              << "tie_word_embeddings: " << (tied_eff ? "true" : "false") << '\n'
              << "dtype: " << detected_dtype << '\n'
              << "shards: " << shard_count << '\n'
              << "tensors: " << tensor_count << '\n'
              << "tokenizer: " << tok_status << '\n'
              << "natlas: " << (has_atlas ? "present" : "absent") << '\n'
              << "accelerate: " << accel_status << '\n'
              << "metal_available: " << (aic::metal_is_available() ? "true" : "false") << '\n'
              << "metal_device: " << aic::metal_device_name() << '\n';
    return 0;
}

int cmd_run(int argc, char** argv) {
    const ParsedArgs args = parse_args(
        argc, argv, 2,
        {"model", "prompt", "max-tokens", "temp", "top-p", "seed", "lora"},
        {"streaming"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const std::string model_dir = require_option(args, "model", "run");
    const std::string prompt = require_option(args, "prompt", "run");
    const int max_tokens = parse_int_strict(get_option_or(args, "max-tokens", "8"), "max-tokens");
    if (max_tokens < 0) {
        throw std::invalid_argument("--max-tokens must be >= 0");
    }
    const double temperature = [&]() -> double {
        const std::string s = get_option_or(args, "temp", "0.0");
        try { return std::stod(s); } catch (...) {
            throw std::invalid_argument("Invalid float for --temp: " + s);
        }
    }();
    const double top_p = [&]() -> double {
        const std::string s = get_option_or(args, "top-p", "1.0");
        try { return std::stod(s); } catch (...) {
            throw std::invalid_argument("Invalid float for --top-p: " + s);
        }
    }();
    const std::uint64_t seed = parse_u64(get_option_or(args, "seed", "42"), "seed");
    const std::string lora_dir = get_option_or(args, "lora", "");
    const bool streaming = get_bool_flag(args, "streaming");
    if (streaming && !lora_dir.empty()) {
        throw std::invalid_argument("--streaming does not support --lora; apply the adapter first");
    }

    const aic::ModelConfig cfg = aic::load_checkpoint_config(model_dir);
    const fs::path tok_path = fs::path(model_dir) / "tokenizer.json";
    const aic::Tokenizer tok = fs::exists(tok_path)
        ? aic::Tokenizer::load_from_file(tok_path.string())
        : aic::Tokenizer::create_byte_level_demo(cfg.vocab_size);

    std::vector<int> prompt_ids = tok.encode(prompt);
    if (prompt_ids.empty()) {
        prompt_ids.push_back(0);
    }
    const int vocab = std::max(1, cfg.vocab_size);
    for (int& tid : prompt_ids) {
        tid = ((tid % vocab) + vocab) % vocab;
    }

    std::vector<int> all_ids;
    if (streaming) {
        all_ids = aic::generate_checkpoint_streaming(
            model_dir, prompt_ids, max_tokens, temperature, top_p, seed);
    } else {
        aic::TransformerModel model = aic::TransformerModel::load_checkpoint(model_dir);
        if (!lora_dir.empty()) {
            model.apply_lora_adapter(lora_dir);
        }
        if (temperature <= 0.0 && top_p >= 1.0) {
            all_ids = model.generate_greedy(prompt_ids, max_tokens);
        } else {
            all_ids = model.generate_sampled(prompt_ids, max_tokens, temperature, top_p, seed);
        }
    }

    std::vector<int> gen_ids;
    if (all_ids.size() > prompt_ids.size()) {
        gen_ids.assign(
            all_ids.begin() + static_cast<std::ptrdiff_t>(prompt_ids.size()),
            all_ids.end());
    }

    std::cout << "prompt_ids:";
    for (int id : prompt_ids) {
        std::cout << ' ' << id;
    }
    std::cout << '\n';

    std::cout << "generated_ids:";
    for (int id : gen_ids) {
        std::cout << ' ' << id;
    }
    std::cout << '\n';

    std::cout << "all_ids:";
    for (int id : all_ids) {
        std::cout << ' ' << id;
    }
    std::cout << '\n';

    std::cout << "generated_text: " << tok.decode(gen_ids) << '\n';
    std::cout << "output_text: " << tok.decode(all_ids) << '\n';
    return 0;
}

int cmd_bench(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"dim", "layers"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const int dim = parse_int_strict(get_option_or(args, "dim", "512"), "dim");
    const int layers = parse_int_strict(get_option_or(args, "layers", "4"), "layers");
    if (dim < 16 || layers < 2) {
        throw std::invalid_argument("--dim must be >= 16 and --layers must be >= 2");
    }

    const std::size_t d = static_cast<std::size_t>(dim);
    aic::Matrix a(d, d, 0.0);
    aic::Matrix b(d, d, 0.0);
    for (std::size_t r = 0; r < d; ++r) {
        for (std::size_t c = 0; c < d; ++c) {
            a(r, c) = 0.001 * static_cast<double>((r * 31 + c) % 97) - 0.04;
            b(r, c) = 0.001 * static_cast<double>((r + c * 17) % 89) - 0.04;
        }
    }

    const double flops = 2.0 * static_cast<double>(d) * static_cast<double>(d) * static_cast<double>(d);

    const auto t0 = std::chrono::steady_clock::now();
    const aic::Matrix c_cpu = aic::mat_mul(a, b);
    const auto t1 = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double cpu_gflops = flops / (std::max(1e-9, cpu_ms * 1e-3) * 1e9);

    const auto t2 = std::chrono::steady_clock::now();
    const aic::Matrix c_metal = aic::metal_mat_mul(a, b);
    const auto t3 = std::chrono::steady_clock::now();
    const double metal_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    const double metal_gflops = flops / (std::max(1e-9, metal_ms * 1e-3) * 1e9);

    const double max_mat_diff = aic::frob_norm(aic::mat_sub(c_cpu, c_metal)) / std::max(1.0, aic::frob_norm(c_cpu));

    const std::string s1_src =
        "module BenchS1;\n"
        "manifold Ring : S1 in Frame<Mid> { curvature 1.0; }\n"
        "flow Rot(state: Residual in Frame<Mid>) -> Residual in Frame<Mid>\n"
        "    ![crosstalk <= 1e-3, approx <= 1e-2, energy <= 0.5, preserve(CoreFluency, Residual)]\n"
        "{\n"
        "    when state ~> Ring(chart: \"angle\") as th {\n"
        "        let nxt = state => Ring.rotate(angle: th);\n"
        "        state +> nxt(gain: 1.0);\n"
        "    }\n"
        "}\n";
    const aic::MIRModule mir = aic::compile_source_to_mir(s1_src);

    aic::ModelConfig cfg;
    cfg.num_layers = layers;
    cfg.hidden_size = dim;
    cfg.intermediate_size = dim * 2;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.vocab_size = 64;
    cfg.attention_type = "gqa";
    cfg.torch_dtype = "BF16";
    cfg.arch_name = "bench-mini";

    const aic::TransformerModel base = aic::TransformerModel::create_synthetic(cfg, 42);
    const aic::NeuralAtlas atlas = aic::build_default_atlas(base, &mir, 42);

    const auto t4 = std::chrono::steady_clock::now();
    const aic::CIRModule cir = aic::lower_mir_to_cir(mir, cfg, atlas, "in-place");
    const std::vector<aic::WISAInstruction> wisa = aic::lower_cir_to_wisa(cir, cfg, atlas, "in-place");
    const auto t5 = std::chrono::steady_clock::now();
    const double cir_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    const fs::path bench_dir = fs::temp_directory_path() / ("aic_bench_" + std::to_string(::getpid()));
    fs::remove_all(bench_dir);
    double stream_ms = 0.0;
    try {
        const fs::path src_dir = bench_dir / "src";
        const fs::path dst_dir = bench_dir / "dst";
        fs::create_directories(src_dir);
        write_text_file((src_dir / "config.json").string(), aic::dump_json(cfg.to_json(), 2));
        std::map<std::string, aic::TensorSpec> specs;
        for (const auto& [k, m] : base.matrices) {
            specs[k] = aic::TensorSpec::from_matrix(m, "BF16");
        }
        for (const auto& [k, v] : base.vectors) {
            specs[k] = aic::TensorSpec::from_vector(v, "BF16");
        }
        const std::size_t cap = compute_max_shard_bytes(specs, 2);
        aic::save_safetensors_sharded(src_dir.string(), specs, cap, {{"arch_name", cfg.arch_name}});

        const auto t6 = std::chrono::steady_clock::now();
        const aic::CompileStats st = aic::apply_wisa_streaming_checkpoint(
            src_dir.string(), dst_dir.string(), wisa, "in-place");
        const auto t7 = std::chrono::steady_clock::now();
        stream_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
        (void)st;
    } catch (...) {
        fs::remove_all(bench_dir);
        throw;
    }
    fs::remove_all(bench_dir);

    const int prefill_tokens = 8;
    std::vector<int> prompt_ids;
    prompt_ids.reserve(static_cast<std::size_t>(prefill_tokens));
    for (int i = 0; i < prefill_tokens; ++i) {
        prompt_ids.push_back(i % std::max(1, cfg.vocab_size));
    }

    const auto t8 = std::chrono::steady_clock::now();
    const std::vector<int> greedy_out = base.generate_greedy(prompt_ids, 4);
    const auto t9 = std::chrono::steady_clock::now();
    const double greedy_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();
    (void)greedy_out;

    aic::KVCache kv = base.init_kv_cache();
    for (int tid : prompt_ids) {
        base.forward_step(tid, kv);
    }
    const int kv_decode_steps = 4;
    const auto t10 = std::chrono::steady_clock::now();
    for (int s = 0; s < kv_decode_steps; ++s) {
        base.forward_step(0, kv);
    }
    const auto t11 = std::chrono::steady_clock::now();
    const double kv_ms = std::chrono::duration<double, std::milli>(t11 - t10).count();
    const double greedy_tps = static_cast<double>(4) / std::max(1e-9, greedy_ms * 1e-3);
    const double kv_tps = static_cast<double>(kv_decode_steps) / std::max(1e-9, kv_ms * 1e-3);

    const std::size_t svd_dim = static_cast<std::size_t>(std::max(dim, 600));
    aic::Matrix svd_mat(svd_dim, svd_dim, 0.0);
    for (std::size_t r = 0; r < svd_dim; ++r) {
        for (std::size_t c = 0; c < std::min(svd_dim, static_cast<std::size_t>(16)); ++c) {
            svd_mat(r, c) = 0.01 * static_cast<double>((r * 13 + c * 7 + 1) % 97) - 0.04;
        }
    }
    const auto t12 = std::chrono::steady_clock::now();
    const std::vector<aic::SVDMode> svd_modes = aic::randomized_svd(svd_mat, 16, 8, 2, 42);
    const auto t13 = std::chrono::steady_clock::now();
    const double svd_ms = std::chrono::duration<double, std::milli>(t13 - t12).count();

    std::cout << std::fixed << std::setprecision(3)
              << "Benchmark dim=" << dim << " layers=" << layers << '\n'
              << "  accelerate_matmul: " << cpu_ms << " ms (" << cpu_gflops << " GFLOPS)\n"
              << "  metal_matmul (" << (aic::metal_is_available() ? aic::metal_device_name() : "cpu-fallback")
              << "): " << metal_ms << " ms (" << metal_gflops << " GFLOPS, rel_diff=" << max_mat_diff << ")\n"
              << "  s1_tessellation_nullspace_cir: " << cir_ms << " ms (wisa_insts=" << wisa.size() << ")\n"
              << "  streaming_bf16_patch: " << stream_ms << " ms\n"
              << "  full_forward_greedy: " << greedy_ms << " ms (" << greedy_tps << " tok/s)\n"
              << "  KVCache forward_step: " << kv_ms << " ms (" << kv_tps << " tok/s, speedup="
              << kv_tps / std::max(1.0, greedy_tps) << "x)\n"
              << "  randomized_svd (" << svd_dim << "x" << svd_dim << " rank-16): "
              << svd_ms << " ms (k=" << svd_modes.size() << ")\n";
    return 0;
}

int cmd_fmt(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {}, {"write", "check"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'fmt' requires <file.strata>");
    }

    const std::string& file_path = args.positional[0];
    const bool do_write = get_bool_flag(args, "write");
    const bool do_check = get_bool_flag(args, "check");

    const std::string source = read_text_file(file_path);
    const std::string formatted = aic::format_strata_source(source, file_path);

    if (do_check) {
        if (formatted != source) {
            std::cerr << file_path << ": file is not formatted (run 'aic fmt --write' to fix)\n";
            return 1;
        }
        std::cout << file_path << ": OK (formatted)\n";
        return 0;
    }

    if (do_write) {
        write_text_file(file_path, formatted);
        std::cout << "Formatted " << file_path << '\n';
        return 0;
    }

    std::cout << formatted;
    return 0;
}

int cmd_verify(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"orig", "patched", "samples"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const std::string orig_dir = require_option(args, "orig", "verify");
    const std::string patched_dir = require_option(args, "patched", "verify");
    const int samples = parse_int_strict(get_option_or(args, "samples", "8"), "samples");
    if (samples < 1) {
        throw std::invalid_argument("--samples must be >= 1");
    }

    const aic::TransformerModel p_model = aic::TransformerModel::load_checkpoint(orig_dir);
    const aic::TransformerModel q_model = aic::TransformerModel::load_checkpoint(patched_dir);

    const int vocab = std::max(1, p_model.config.vocab_size);
    std::vector<std::vector<int>> seqs;
    seqs.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        std::vector<int> s;
        s.reserve(4);
        for (int j = 0; j < 4; ++j) {
            s.push_back(((i * 7 + j * 3 + 1) % vocab + vocab) % vocab);
        }
        seqs.push_back(std::move(s));
    }

    auto mat_frob_diff_mismatched = [](const aic::Matrix& A, const aic::Matrix& B) -> double {
        const std::size_t min_r = std::min(A.rows, B.rows);
        const std::size_t min_c = std::min(A.cols, B.cols);
        double acc = 0.0;
        for (std::size_t r = 0; r < min_r; ++r) {
            for (std::size_t c = 0; c < min_c; ++c) {
                const double d = A(r, c) - B(r, c);
                acc += d * d;
            }
            for (std::size_t c = min_c; c < A.cols; ++c) { const double v = A(r, c); acc += v * v; }
            for (std::size_t c = min_c; c < B.cols; ++c) { const double v = B(r, c); acc += v * v; }
        }
        for (std::size_t r = min_r; r < A.rows; ++r) {
            for (std::size_t c = 0; c < A.cols; ++c) { const double v = A(r, c); acc += v * v; }
        }
        for (std::size_t r = min_r; r < B.rows; ++r) {
            for (std::size_t c = 0; c < B.cols; ++c) { const double v = B(r, c); acc += v * v; }
        }
        return std::sqrt(acc);
    };

    auto vec_frob_diff_mismatched = [](const aic::Vector& a, const aic::Vector& b) -> double {
        const std::size_t min_n = std::min(a.size(), b.size());
        double acc = 0.0;
        for (std::size_t i = 0; i < min_n; ++i) { const double d = a[i] - b[i]; acc += d * d; }
        for (std::size_t i = min_n; i < a.size(); ++i) { acc += a[i] * a[i]; }
        for (std::size_t i = min_n; i < b.size(); ++i) { acc += b[i] * b[i]; }
        return std::sqrt(acc);
    };

    double frob_diff = 0.0;
    for (const auto& [name, mat] : p_model.matrices) {
        auto it = q_model.matrices.find(name);
        if (it != q_model.matrices.end()) {
            if (mat.rows == it->second.rows && mat.cols == it->second.cols) {
                frob_diff += aic::frob_norm(aic::mat_sub(mat, it->second));
            } else {
                frob_diff += mat_frob_diff_mismatched(mat, it->second);
            }
        } else {
            frob_diff += aic::frob_norm(mat);
        }
    }
    for (const auto& [name, mat] : q_model.matrices) {
        if (!p_model.matrices.contains(name)) {
            frob_diff += aic::frob_norm(mat);
        }
    }
    for (const auto& [name, vec] : p_model.vectors) {
        auto it = q_model.vectors.find(name);
        if (it != q_model.vectors.end()) {
            frob_diff += vec_frob_diff_mismatched(vec, it->second);
        } else {
            double acc = 0.0;
            for (double v : vec) acc += v * v;
            frob_diff += std::sqrt(acc);
        }
    }
    for (const auto& [name, vec] : q_model.vectors) {
        if (!p_model.vectors.contains(name)) {
            double acc = 0.0;
            for (double v : vec) acc += v * v;
            frob_diff += std::sqrt(acc);
        }
    }

    const int orig_layers = p_model.config.num_layers;
    int untouched_layers = 0;
    for (int l = 0; l < orig_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l) + ".";
        bool layer_untouched = true;
        for (const auto& [name, mat] : p_model.matrices) {
            if (name.substr(0, prefix.size()) != prefix) continue;
            auto it = q_model.matrices.find(name);
            if (it == q_model.matrices.end()) { layer_untouched = false; break; }
            if (mat.rows != it->second.rows || mat.cols != it->second.cols) { layer_untouched = false; break; }
            if (aic::frob_norm(aic::mat_sub(mat, it->second)) > 1e-12) { layer_untouched = false; break; }
        }
        if (layer_untouched) {
            for (const auto& [name, vec] : p_model.vectors) {
                if (name.substr(0, prefix.size()) != prefix) continue;
                auto it = q_model.vectors.find(name);
                if (it == q_model.vectors.end()) { layer_untouched = false; break; }
                if (vec_frob_diff_mismatched(vec, it->second) > 1e-12) { layer_untouched = false; break; }
            }
        }
        if (layer_untouched) ++untouched_layers;
    }

    const aic::ModelDivergenceStats div = aic::compute_model_divergence(p_model, q_model, seqs);

    std::cout << std::fixed << std::setprecision(6)
              << "weight_frob_diff: " << frob_diff << '\n'
              << "untouched_layers: " << untouched_layers << '/' << orig_layers << '\n'
              << "mean_kl: " << div.mean_kl << '\n'
              << "max_kl: " << div.max_kl << '\n'
              << "max_logit_diff: " << div.max_logit_diff << '\n';
    return 0;
}

int cmd_lint(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'lint' requires <file.strata>");
    }
    const std::string& file_path = args.positional[0];
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);

    std::vector<aic::LintDiagnostic> diags;
    if (!inc_dirs.empty()) {
        diags = aic::lint_file_with_imports(file_path, inc_dirs);
    } else {
        const std::string source = read_text_file(file_path);
        diags = aic::lint_source(source, file_path);
    }

    for (const auto& d : diags) {
        std::cerr << file_path << ":" << d.line << ":" << d.col
                  << ": " << d.code << ": " << d.message << '\n';
    }
    if (!diags.empty()) {
        return 1;
    }
    std::cout << "lint OK: " << file_path << " (0 warnings)\n";
    return 0;
}

int cmd_trace(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"model", "prompt", "strata", "atlas", "lora", "gen-tokens", "compare"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const std::string model_dir = require_option(args, "model", "trace");
    const std::string prompt = require_option(args, "prompt", "trace");
    const std::string strata_path = get_option_or(args, "strata", "");
    const std::string atlas_opt = get_option_or(args, "atlas", "");
    const std::string lora_dir = get_option_or(args, "lora", "");
    const std::string compare_dir = get_option_or(args, "compare", "");
    const int gen_tokens = parse_int_strict(get_option_or(args, "gen-tokens", "0"), "gen-tokens");
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);

    aic::TransformerModel model = aic::TransformerModel::load_checkpoint(model_dir);
    if (!lora_dir.empty()) {
        model.apply_lora_adapter(lora_dir);
    }

    const fs::path tok_path = fs::path(model_dir) / "tokenizer.json";
    const aic::Tokenizer tok = fs::exists(tok_path)
        ? aic::Tokenizer::load_from_file(tok_path.string())
        : aic::Tokenizer::create_byte_level_demo(model.config.vocab_size);

    std::vector<int> token_ids = tok.encode(prompt);
    if (token_ids.empty()) {
        token_ids.push_back(0);
    }
    const int vocab = std::max(1, model.config.vocab_size);
    for (int& tid : token_ids) {
        tid = ((tid % vocab) + vocab) % vocab;
    }

    aic::NeuralAtlas atlas;
    const fs::path natlas_path = !atlas_opt.empty()
        ? fs::path(atlas_opt)
        : fs::path(model_dir) / "model.natlas";
    if (fs::exists(natlas_path)) {
        atlas = aic::NeuralAtlas::load(natlas_path.string());
    } else {
        atlas = aic::build_default_atlas(model, nullptr, 42);
    }

    const aic::MIRModule* mir_ptr = nullptr;
    aic::MIRModule mir;
    if (!strata_path.empty()) {
        if (!inc_dirs.empty()) {
            mir = aic::compile_file_with_imports(strata_path, inc_dirs);
        } else {
            const std::string src = read_text_file(strata_path);
            mir = aic::compile_source_to_mir(src, strata_path);
        }
        mir_ptr = &mir;
    }

    if (!compare_dir.empty()) {
        const aic::TransformerModel patched_model = aic::TransformerModel::load_checkpoint(compare_dir);
        const aic::TraceComparisonReport cmp = aic::compare_execution_traces(model, patched_model, atlas, token_ids, mir_ptr);
        std::cout << aic::format_trace_comparison(cmp, &tok) << '\n';
        return 0;
    }

    if (gen_tokens > 0) {
        const aic::GenerationTraceReport gen_rep = aic::trace_generation(model, atlas, token_ids, gen_tokens, mir_ptr);
        std::cout << aic::format_generation_trace(gen_rep, &tok) << '\n';
        return 0;
    }

    const aic::ExecutionTraceReport rep = aic::trace_model_execution(model, atlas, token_ids, mir_ptr);
    std::cout << aic::format_execution_trace(rep, &tok) << '\n';
    return 0;
}

int cmd_diff_nelf(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"lhs", "rhs"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (!args.positional.empty()) {
        throw std::invalid_argument("Unexpected positional argument: " + args.positional[0]);
    }

    const std::string lhs_dir = require_option(args, "lhs", "diff-nelf");
    const std::string rhs_dir = require_option(args, "rhs", "diff-nelf");

    const aic::NeuralELF lhs = aic::NeuralELF::load(lhs_dir);
    const aic::NeuralELF rhs = aic::NeuralELF::load(rhs_dir);

    const aic::NelfDiffReport rep = aic::diff_nelf(lhs, rhs);
    std::cout << aic::format_nelf_diff(rep) << '\n';
    return 0;
}

int cmd_patch_apply(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"model", "out"}, {"streaming"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'patch-apply' requires <file.npatch>");
    }

    const std::string patch_path = args.positional[0];
    const std::string model_dir = require_option(args, "model", "patch-apply");
    const std::string out_dir = require_option(args, "out", "patch-apply");
    const bool streaming = get_bool_flag(args, "streaming");

    const aic::PatchBundle bundle = aic::load_patch_bundle(patch_path);

    if (streaming) {
        aic::apply_patch_bundle_streaming(model_dir, out_dir, bundle, true);
        const fs::path natlas_src = fs::path(model_dir) / "model.natlas";
        if (fs::exists(natlas_src)) {
            fs::create_directories(out_dir);
            fs::copy_file(natlas_src, fs::path(out_dir) / "model.natlas", fs::copy_options::overwrite_existing);
        }
        copy_tokenizer_if_present(model_dir, out_dir);
        std::cout << "patch-apply (streaming): " << patch_path << " -> " << out_dir
                  << " (module=" << bundle.module_name
                  << ", arch=" << bundle.arch_name << ")\n";
        return 0;
    }

    const aic::TransformerModel base_model = aic::TransformerModel::load_checkpoint(model_dir);
    const aic::TransformerModel patched = aic::apply_patch_bundle(base_model, bundle, true);
    patched.save_checkpoint(out_dir);
    copy_tokenizer_if_present(model_dir, out_dir);

    std::cout << "patch-apply: " << patch_path << " -> " << out_dir
              << " (module=" << bundle.module_name
              << ", arch=" << bundle.arch_name << ")\n";
    return 0;
}

int cmd_patch_revert(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"model", "out"}, {"streaming"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'patch-revert' requires <file.npatch>");
    }

    const std::string patch_path = args.positional[0];
    const std::string patched_dir = require_option(args, "model", "patch-revert");
    const std::string out_dir = require_option(args, "out", "patch-revert");
    const bool streaming = get_bool_flag(args, "streaming");

    const aic::PatchBundle bundle = aic::load_patch_bundle(patch_path);

    if (streaming) {
        aic::revert_patch_bundle_streaming(patched_dir, out_dir, bundle);
        const fs::path natlas_src = fs::path(patched_dir) / "model.natlas";
        if (fs::exists(natlas_src)) {
            fs::create_directories(out_dir);
            fs::copy_file(natlas_src, fs::path(out_dir) / "model.natlas", fs::copy_options::overwrite_existing);
        }
        copy_tokenizer_if_present(patched_dir, out_dir);
        std::cout << "patch-revert (streaming): " << patch_path << " -> " << out_dir << '\n';
        return 0;
    }

    const aic::TransformerModel patched_model = aic::TransformerModel::load_checkpoint(patched_dir);
    const aic::TransformerModel restored = aic::revert_patch_bundle(patched_model, bundle);
    restored.save_checkpoint(out_dir);
    copy_tokenizer_if_present(patched_dir, out_dir);

    std::cout << "patch-revert: " << patch_path << " -> " << out_dir << '\n';
    return 0;
}

int cmd_patch_compose(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"out"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 2) {
        throw std::invalid_argument("Command 'patch-compose' requires <p1.npatch> <p2.npatch>");
    }

    const std::string p1_path = args.positional[0];
    const std::string p2_path = args.positional[1];
    const std::string out_path = require_option(args, "out", "patch-compose");

    const aic::PatchBundle b1 = aic::load_patch_bundle(p1_path);
    const aic::PatchBundle b2 = aic::load_patch_bundle(p2_path);
    const aic::PatchBundle composed = aic::compose_patch_bundles(b1, b2);
    aic::save_patch_bundle(composed, out_path);

    std::cout << "patch-compose: " << p1_path << " + " << p2_path << " -> " << out_path << '\n';
    return 0;
}

int cmd_test(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"model", "atlas", "target", "opt-level"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'test' requires <file.strata>");
    }

    const std::string& file_path = args.positional[0];
    const std::string model_dir = require_option(args, "model", "test");
    const std::string atlas_path = get_option_or(args, "atlas", "");
    const std::string target = get_option_or(args, "target", "in-place");
    const int opt_level = parse_opt_level(args);
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);
    validate_target_mode(target);

    aic::MIRModule mir;
    if (!inc_dirs.empty()) {
        mir = aic::compile_file_with_imports(file_path, inc_dirs);
    } else {
        const std::string source = read_text_file(file_path);
        mir = aic::compile_source_to_mir(source, file_path);
    }

    const aic::TransformerModel base_model = aic::TransformerModel::load_checkpoint(model_dir);

    const fs::path tok_path = fs::path(model_dir) / "tokenizer.json";
    const aic::Tokenizer tok_obj = fs::exists(tok_path)
        ? aic::Tokenizer::load_from_file(tok_path.string())
        : aic::Tokenizer::create_byte_level_demo(base_model.config.vocab_size);
    const aic::Tokenizer* tok_ptr = &tok_obj;

    aic::NeuralAtlas atlas;
    if (!atlas_path.empty()) {
        atlas = resolve_atlas_for_compile(mir, base_model, atlas_path, model_dir);
    } else {
        aic::NeuralAtlas existing_atlas;
        const aic::NeuralAtlas* existing_atlas_ptr = nullptr;
        const fs::path default_atlas_path = fs::path(model_dir) / "model.natlas";
        if (fs::exists(default_atlas_path)) {
            existing_atlas = aic::NeuralAtlas::load(default_atlas_path.string());
            existing_atlas_ptr = &existing_atlas;
        }
        atlas = aic::resolve_symbols_for_target(mir, base_model, existing_atlas_ptr, 42, tok_ptr);
        aic::calibrate_atlas_empirical(atlas, base_model, &mir, tok_ptr);
    }

    aic::CompileOutput compiled;
    if (target == "lora") {
        const aic::CIRModule cir = opt_level >= 1
            ? aic::optimize_cir(aic::lower_mir_to_cir(mir, base_model.config, atlas, target), opt_level)
            : aic::lower_mir_to_cir(mir, base_model.config, atlas, target);
        std::vector<aic::WISAInstruction> wisa = aic::lower_cir_to_wisa(cir, base_model.config, atlas, target);
        if (opt_level >= 1) {
            wisa = aic::optimize_wisa(wisa, opt_level);
        }
        const fs::path lora_tmp = fs::temp_directory_path() / ("aic_test_lora_" + std::to_string(::getpid()));
        const aic::CompileStats lora_stats = aic::emit_lora_adapter(base_model, wisa, lora_tmp.string());
        compiled.model = aic::TransformerModel::load_with_lora(model_dir, lora_tmp.string());
        compiled.cir = cir;
        compiled.wisa = std::move(wisa);
        compiled.stats = lora_stats;
        for (const auto& flow : cir.flows) {
            compiled.stats.max_approx_error = std::max(compiled.stats.max_approx_error, flow.max_approx_error);
            compiled.stats.max_crosstalk = std::max(compiled.stats.max_crosstalk, flow.max_crosstalk);
        }
        fs::remove_all(lora_tmp);
    } else {
        compiled = aic::compile_mir_to_model(mir, base_model, &atlas, target, opt_level);
    }

    const aic::StrataTestSuiteReport report = aic::run_strata_tests(mir, base_model, compiled, atlas, tok_ptr);
    std::cout << aic::format_strata_test_report(report);

    return (report.passed_assertions == report.total_assertions) ? 0 : 1;
}

std::string inline_file_imports_for_nelf(
    const std::string& file_path,
    const std::string& source,
    const std::vector<std::string>& inc_dirs) {
    if (inc_dirs.empty()) {
        return source;
    }
    (void)aic::compile_file_with_imports(file_path, inc_dirs);
    std::vector<fs::path> search_roots;
    const fs::path parent = fs::path(file_path).parent_path();
    if (!parent.empty()) {
        search_roots.push_back(parent);
    }
    for (const std::string& d : inc_dirs) {
        search_roots.emplace_back(d);
    }

    std::istringstream iss(source);
    std::ostringstream out;
    std::string line;
    std::set<std::string> included;
    while (std::getline(iss, line)) {
        std::string trimmed = line;
        const std::size_t first = trimmed.find_first_not_of(" \t\r");
        if (first != std::string::npos) {
            trimmed = trimmed.substr(first);
        } else {
            trimmed.clear();
        }
        if (trimmed.rfind("import ", 0) == 0 && trimmed.find("std::") == std::string::npos) {
            const std::size_t semi = trimmed.find(';');
            if (semi != std::string::npos) {
                std::string imp_spec = trimmed.substr(7, semi - 7);
                const std::size_t as_pos = imp_spec.find(" as ");
                if (as_pos != std::string::npos) {
                    imp_spec = imp_spec.substr(0, as_pos);
                }
                while (!imp_spec.empty() && std::isspace(static_cast<unsigned char>(imp_spec.back()))) {
                    imp_spec.pop_back();
                }
                while (!imp_spec.empty() && std::isspace(static_cast<unsigned char>(imp_spec.front()))) {
                    imp_spec.erase(imp_spec.begin());
                }
                std::string rel_path;
                for (std::size_t i = 0; i < imp_spec.size(); ++i) {
                    if (i + 1 < imp_spec.size() && imp_spec[i] == ':' && imp_spec[i + 1] == ':') {
                        rel_path.push_back('/');
                        ++i;
                    } else {
                        rel_path.push_back(imp_spec[i]);
                    }
                }
                rel_path += ".strata";
                fs::path found;
                for (const fs::path& root : search_roots) {
                    const fs::path cand = root / rel_path;
                    if (fs::exists(cand)) {
                        found = fs::weakly_canonical(cand);
                        break;
                    }
                }
                if (!found.empty() && included.insert(found.string()).second) {
                    std::istringstream inc_ss(read_text_file(found.string()));
                    std::string inc_line;
                    while (std::getline(inc_ss, inc_line)) {
                        std::string itrim = inc_line;
                        const std::size_t ifirst = itrim.find_first_not_of(" \t\r");
                        if (ifirst != std::string::npos) {
                            itrim = itrim.substr(ifirst);
                        } else {
                            itrim.clear();
                        }
                        if (itrim.rfind("module ", 0) == 0 || itrim.rfind("import ", 0) == 0) {
                            continue;
                        }
                        out << inc_line << '\n';
                    }
                    continue;
                }
            }
        }
        out << line << '\n';
    }
    return out.str();
}

int cmd_build_nelf(int argc, char** argv) {
    const ParsedArgs args = parse_args(argc, argv, 2, {"nelf", "out", "target", "verify-orig"});
    if (args.options.contains("help")) {
        print_usage(std::cout);
        return 0;
    }
    if (args.positional.size() != 1) {
        throw std::invalid_argument("Command 'build-nelf' requires <file.strata>");
    }

    const std::string& file_path = args.positional[0];
    const std::string nelf_dir = require_option(args, "nelf", "build-nelf");
    const std::string out_dir = require_option(args, "out", "build-nelf");
    const std::string target = get_option_or(args, "target", "in-place");
    const std::string verify_orig_dir = get_option_or(args, "verify-orig", "");
    const std::vector<std::string> inc_dirs = collect_include_dirs(argc, argv, 2);
    if (target != "in-place" && target != "elastic") {
        throw std::invalid_argument("Invalid --target '" + target + "' for build-nelf (expected in-place or elastic)");
    }

    const aic::NeuralELF nelf = aic::NeuralELF::load(nelf_dir);
    const std::string raw_source = read_text_file(file_path);
    const std::string effective_source = inline_file_imports_for_nelf(file_path, raw_source, inc_dirs);

    const aic::TransformerModel rebuilt = aic::recompile_nelf_with_patch(nelf, effective_source, target);
    rebuilt.save_checkpoint(out_dir);

    aic::NeuralAtlas out_atlas = nelf.atlas;
    out_atlas.num_layers = rebuilt.config.num_layers;
    out_atlas.save((fs::path(out_dir) / "model.natlas").string());

    if (!verify_orig_dir.empty() && fs::exists(fs::path(verify_orig_dir) / "tokenizer.json")) {
        copy_tokenizer_if_present(verify_orig_dir, out_dir);
    } else {
        const aic::Tokenizer demo_tok = aic::Tokenizer::create_byte_level_demo(rebuilt.config.vocab_size);
        demo_tok.save((fs::path(out_dir) / "tokenizer.json").string());
    }

    std::cout << "Built NELF " << file_path << " + " << nelf_dir << " -> " << out_dir
              << " (target=" << target
              << ", layers=" << rebuilt.config.num_layers
              << ", hidden=" << rebuilt.config.hidden_size << ")\n";

    if (!verify_orig_dir.empty()) {
        const aic::TransformerModel orig = aic::TransformerModel::load_checkpoint(verify_orig_dir);
        const aic::FaithfulnessReport rep = aic::verify_nelf_faithfulness(orig, rebuilt, &nelf);
        std::cout << std::fixed << std::setprecision(6)
                  << "weight_frob_diff: " << rep.weight_frob_diff << '\n'
                  << "residual_energy_ratio: " << rep.residual_energy_ratio << '\n'
                  << "mean_kl_div: " << rep.mean_kl_div << '\n'
                  << "max_kl_div: " << rep.max_kl_div << '\n'
                  << "max_logit_diff: " << rep.max_logit_diff << '\n';
    }
    return 0;
}

}

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(std::cerr);
            return 1;
        }
        const std::string cmd = argv[1];
        if (cmd == "-h" || cmd == "--help") {
            print_usage(std::cout);
            return 0;
        }
        if (cmd == "init-demo-model") {
            return cmd_init_demo_model(argc, argv);
        }
        if (cmd == "check") {
            return cmd_check(argc, argv);
        }
        if (cmd == "ir") {
            return cmd_ir(argc, argv);
        }
        if (cmd == "compile") {
            return cmd_compile(argc, argv);
        }
        if (cmd == "lift") {
            return cmd_lift(argc, argv);
        }
        if (cmd == "link") {
            return cmd_link(argc, argv);
        }
        if (cmd == "inspect") {
            return cmd_inspect(argc, argv);
        }
        if (cmd == "run") {
            return cmd_run(argc, argv);
        }
        if (cmd == "bench") {
            return cmd_bench(argc, argv);
        }
        if (cmd == "fmt") {
            return cmd_fmt(argc, argv);
        }
        if (cmd == "verify") {
            return cmd_verify(argc, argv);
        }
        if (cmd == "lint") {
            return cmd_lint(argc, argv);
        }
        if (cmd == "trace") {
            return cmd_trace(argc, argv);
        }
        if (cmd == "diff-nelf") {
            return cmd_diff_nelf(argc, argv);
        }
        if (cmd == "patch-apply") {
            return cmd_patch_apply(argc, argv);
        }
        if (cmd == "patch-revert") {
            return cmd_patch_revert(argc, argv);
        }
        if (cmd == "patch-compose") {
            return cmd_patch_compose(argc, argv);
        }
        if (cmd == "test") {
            return cmd_test(argc, argv);
        }
        if (cmd == "build-nelf") {
            return cmd_build_nelf(argc, argv);
        }
        std::cerr << "Unknown command: " << cmd << '\n';
        print_usage(std::cerr);
        return 1;
    } catch (const aic::StrataError& e) {
        std::cerr << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
