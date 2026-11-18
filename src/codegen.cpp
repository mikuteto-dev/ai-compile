#include "aic/codegen.hpp"

#include "aic/errors.hpp"
#include "aic/json_min.hpp"
#include "aic/safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace aic {

namespace {

std::string to_lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string normalize_dtype_upper(std::string dtype) {
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

std::size_t effective_head_dim(const ModelConfig& cfg) {
    if (cfg.head_dim > 0) {
        return static_cast<std::size_t>(cfg.head_dim);
    }
    const std::size_t n_q = static_cast<std::size_t>(std::max(1, cfg.num_attention_heads));
    return static_cast<std::size_t>(cfg.hidden_size) / n_q;
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

Matrix append_zero_rows(const Matrix& m, std::size_t extra_rows) {
    if (extra_rows == 0) {
        return m;
    }
    Matrix out(m.rows + extra_rows, m.cols, 0.0);
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            out(r, c) = m(r, c);
        }
    }
    return out;
}

Matrix append_zero_cols(const Matrix& m, std::size_t extra_cols) {
    if (extra_cols == 0) {
        return m;
    }
    Matrix out(m.rows, m.cols + extra_cols, 0.0);
    for (std::size_t r = 0; r < m.rows; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            out(r, c) = m(r, c);
        }
    }
    return out;
}

Matrix extract_row_slice(const Matrix& m, std::size_t row_begin, std::size_t row_end) {
    Matrix out(row_end - row_begin, m.cols, 0.0);
    for (std::size_t r = row_begin; r < row_end; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            out(r - row_begin, c) = m(r, c);
        }
    }
    return out;
}

void add_row_slice(Matrix& dst, std::size_t row_begin, const Matrix& delta) {
    for (std::size_t r = 0; r < delta.rows; ++r) {
        for (std::size_t c = 0; c < delta.cols; ++c) {
            dst(row_begin + r, c) += delta(r, c);
        }
    }
}

void add_col_slice(Matrix& dst, std::size_t col_begin, const Matrix& delta) {
    for (std::size_t r = 0; r < delta.rows; ++r) {
        for (std::size_t c = 0; c < delta.cols; ++c) {
            dst(r, col_begin + c) += delta(r, c);
        }
    }
}

struct TensorAccess {
    std::function<Matrix&(const std::string&, bool)> get_matrix;
    std::function<Vector&(const std::string&, bool)> get_vector;
    std::function<bool(const std::string&)> has_matrix;
    std::function<bool(const std::string&)> has_vector;
};

void populate_zero_bypass_layer(
    const ModelConfig& cfg,
    int layer_idx,
    const std::function<void(const std::string&, Matrix)>& put_matrix,
    const std::function<void(const std::string&, Vector)>& put_vector) {
    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t d_ff = static_cast<std::size_t>(cfg.intermediate_size);
    const std::size_t n_q = static_cast<std::size_t>(std::max(1, cfg.num_attention_heads));
    const std::size_t hd = effective_head_dim(cfg);
    const std::size_t q_dim = n_q * hd;
    const std::string attn = to_lower_copy(cfg.attention_type);
    const std::string prefix = "model.layers." + std::to_string(layer_idx);

    put_vector(prefix + ".input_layernorm.weight", Vector(d, 1.0));
    put_matrix(prefix + ".self_attn.q_proj.weight", zeros(q_dim, d));

    std::size_t kv_dim = 0;
    if (attn == "mla") {
        const std::size_t rank = static_cast<std::size_t>(std::max(1, cfg.kv_lora_rank));
        put_matrix(prefix + ".self_attn.kv_a_proj.weight", zeros(rank, d));
        put_matrix(prefix + ".self_attn.kv_b_proj.weight", zeros(2 * q_dim, rank));
        put_matrix(prefix + ".self_attn.k_proj.weight", zeros(q_dim, d));
        put_matrix(prefix + ".self_attn.v_proj.weight", zeros(q_dim, d));
        kv_dim = q_dim;
    } else {
        kv_dim = static_cast<std::size_t>(std::max(1, cfg.num_key_value_heads)) * hd;
        put_matrix(prefix + ".self_attn.k_proj.weight", zeros(kv_dim, d));
        put_matrix(prefix + ".self_attn.v_proj.weight", zeros(kv_dim, d));
    }

    if (cfg.qkv_bias) {
        put_vector(prefix + ".self_attn.q_proj.bias", Vector(q_dim, 0.0));
        put_vector(prefix + ".self_attn.k_proj.bias", Vector(kv_dim, 0.0));
        put_vector(prefix + ".self_attn.v_proj.bias", Vector(kv_dim, 0.0));
    }
    if (cfg.qk_norm) {
        put_vector(prefix + ".self_attn.q_norm.weight", Vector(hd, 1.0));
        put_vector(prefix + ".self_attn.k_norm.weight", Vector(hd, 1.0));
    }

    put_matrix(prefix + ".self_attn.o_proj.weight", zeros(d, q_dim));
    put_vector(prefix + ".post_attention_layernorm.weight", Vector(d, 1.0));
    put_matrix(prefix + ".mlp.gate_proj.weight", zeros(d_ff, d));
    put_matrix(prefix + ".mlp.up_proj.weight", zeros(d_ff, d));
    put_matrix(prefix + ".mlp.down_proj.weight", zeros(d, d_ff));
}

void execute_wisa_instructions(
    const ModelConfig& cfg,
    const std::vector<WISAInstruction>& insts,
    bool elastic,
    bool lora,
    TensorAccess& access,
    CompileStats& stats) {
    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    const std::string attn_type = to_lower_copy(cfg.attention_type);
    auto resolve_layer = [&](int l, const char* op_name) -> int {
        if (l >= 0 && l < cfg.num_layers) {
            return l;
        }
        if (lora && cfg.num_layers > 0 && l >= 0) {
            return l % cfg.num_layers;
        }
        throw OutOfDepthError(std::string(op_name) + " layer out of range");
    };

    for (std::size_t idx = 0; idx < insts.size(); ++idx) {
        const auto& wi = insts[idx];
        if (wi.kind == WISAKind::SWIGLU_POLE) {
            const int layer_eff = resolve_layer(wi.layer, "SWIGLU_POLE");
            const std::string prefix = "model.layers." + std::to_string(layer_eff);
            Matrix& w_gate = access.get_matrix(prefix + ".mlp.gate_proj.weight", true);
            Matrix& w_up = access.get_matrix(prefix + ".mlp.up_proj.weight", true);
            Matrix& w_down = access.get_matrix(prefix + ".mlp.down_proj.weight", true);

            if (w_gate.cols != d || w_up.cols != d || w_down.rows != d) {
                throw std::invalid_argument("MLP hidden dimension mismatch in SWIGLU_POLE");
            }
            if (wi.w_gate.size() != d || wi.w_up.size() != d || wi.w_down.size() != d) {
                throw std::invalid_argument("SWIGLU_POLE vector size mismatch with hidden_size");
            }

            if (elastic && (wi.append_slot || wi.slot >= static_cast<int>(w_gate.rows))) {
                const std::size_t new_slot = w_gate.rows;
                w_gate = append_zero_rows(w_gate, 1);
                w_up = append_zero_rows(w_up, 1);
                w_down = append_zero_cols(w_down, 1);
                w_gate.set_row(new_slot, wi.w_gate);
                w_up.set_row(new_slot, wi.w_up);
                w_down.set_col(new_slot, vec_scale(wi.w_down, static_cast<double>(wi.polarity)));
                ++stats.added_poles;
            } else if (lora) {
                if (w_gate.rows == 0) {
                    throw ApproximationUnmetError("Empty MLP weight rows in lora mode");
                }
                const std::size_t s = (wi.slot >= 0 && static_cast<std::size_t>(wi.slot) < w_gate.rows)
                    ? static_cast<std::size_t>(wi.slot)
                    : (static_cast<std::size_t>(std::max(0, wi.slot)) % w_gate.rows);
                w_gate.set_row(s, vec_add(w_gate.row(s), wi.w_gate));
                w_up.set_row(s, vec_add(w_up.row(s), wi.w_up));
                w_down.set_col(s, vec_add(w_down.col(s), vec_scale(wi.w_down, static_cast<double>(wi.polarity))));
            } else {
                if (wi.slot < 0 || static_cast<std::size_t>(wi.slot) >= w_gate.rows) {
                    throw ApproximationUnmetError("SWIGLU_POLE slot index out of bounds in in-place mode");
                }
                const std::size_t s = static_cast<std::size_t>(wi.slot);
                w_gate.set_row(s, vec_add(w_gate.row(s), wi.w_gate));
                w_up.set_row(s, vec_add(w_up.row(s), wi.w_up));
                w_down.set_col(s, vec_add(w_down.col(s), vec_scale(wi.w_down, static_cast<double>(wi.polarity))));
            }
            ++stats.instructions_applied;
        } else if (wi.kind == WISAKind::THRESHOLDED_ROUTE) {
            const int layer_eff = resolve_layer(wi.layer, "THRESHOLDED_ROUTE");
            const std::string prefix = "model.layers." + std::to_string(layer_eff);
            Matrix& route = access.get_matrix(
                prefix + ".mlp.aic_threshold_route.weight", true);
            Vector& thresholds = access.get_vector(
                prefix + ".mlp.aic_threshold_route.threshold", true);
            Vector& polarities = access.get_vector(
                prefix + ".mlp.aic_threshold_route.polarity", true);
            Vector& mask = access.get_vector(
                prefix + ".mlp.aic_threshold_route.mask", true);
            if (wi.slot < 0 || static_cast<std::size_t>(wi.slot) >= route.cols ||
                route.rows != d || wi.u.size() != d || thresholds.size() != route.cols ||
                polarities.size() != route.cols || mask.size() != route.cols) {
                throw std::invalid_argument("THRESHOLDED_ROUTE tensor shape mismatch");
            }
            const std::size_t slot = static_cast<std::size_t>(wi.slot);
            const double polarity = static_cast<double>(wi.polarity);
            if (mask[slot] > 0.0 &&
                (std::abs(thresholds[slot] - wi.threshold) > 1e-12 ||
                 std::abs(polarities[slot] - polarity) > 1e-12)) {
                throw ApproximationUnmetError(
                    "Conflicting thresholded routes on the same detector slot");
            }
            thresholds[slot] = wi.threshold;
            polarities[slot] = polarity;
            mask[slot] = 1.0;
            route.set_col(slot, vec_add(route.col(slot), vec_scale(wi.u, wi.scale)));
            ++stats.instructions_applied;
        } else if (wi.kind == WISAKind::VPD_RANK1) {
            const int layer_eff = resolve_layer(wi.layer, "VPD_RANK1");
            const std::string prefix = "model.layers." + std::to_string(layer_eff);
            const std::string mat_key = prefix + "." + wi.matrix_id + ".weight";
            if (!access.has_matrix(mat_key)) {
                throw std::invalid_argument("Unknown matrix_id in VPD_RANK1: " + mat_key);
            }
            Matrix& w = access.get_matrix(mat_key, true);
            Vector v_eff = wi.v;
            if (wi.matrix_id == "mlp.down_proj" && wi.v.size() == d && w.cols != d) {
                const Matrix& w_up = access.get_matrix(prefix + ".mlp.up_proj.weight", false);
                v_eff = normalize(mat_vec(w_up, wi.v));
                if (norm2(v_eff) < 1e-12) {
                    v_eff = vec_zeros(w.cols);
                    if (!v_eff.empty()) {
                        v_eff[0] = 1.0;
                    }
                }
            }
            if (wi.u.size() != w.rows || v_eff.size() != w.cols) {
                throw std::invalid_argument("VPD_RANK1 vector shape mismatch for " + mat_key);
            }
            for (std::size_t r = 0; r < w.rows; ++r) {
                const double su = wi.scale * wi.u[r];
                for (std::size_t c = 0; c < w.cols; ++c) {
                    w(r, c) += su * v_eff[c];
                }
            }
            ++stats.instructions_applied;
        } else if (wi.kind == WISAKind::ATTN_BILINEAR) {
            const int layer_eff = resolve_layer(wi.layer, "ATTN_BILINEAR");
            const std::string prefix = "model.layers." + std::to_string(layer_eff);
            const std::string q_key = prefix + ".self_attn.q_proj.weight";
            const std::string o_key = prefix + ".self_attn.o_proj.weight";
            const std::string k_key = prefix + ".self_attn.k_proj.weight";
            const std::string v_key = prefix + ".self_attn.v_proj.weight";

            Matrix& w_q = access.get_matrix(q_key, true);
            Matrix& w_o = access.get_matrix(o_key, true);

            const std::size_t n_q = static_cast<std::size_t>(std::max(1, cfg.num_attention_heads));
            const std::size_t hd = effective_head_dim(cfg);

            if (attn_type == "gqa") {
                const std::size_t n_kv = static_cast<std::size_t>(std::max(1, cfg.num_key_value_heads));
                const std::size_t g = static_cast<std::size_t>(std::max(0, wi.head_group)) % n_kv;
                const std::size_t group_size = n_q / n_kv;
                const std::size_t hq = g * group_size + (group_size > 1 ? 1 : 0);

                const Matrix& k_ro = access.get_matrix(k_key, false);
                const Matrix& v_ro = access.get_matrix(v_key, false);
                Matrix k_g = extract_row_slice(k_ro, g * hd, (g + 1) * hd);
                Matrix v_g = extract_row_slice(v_ro, g * hd, (g + 1) * hd);

                if (frob_norm(k_g) < 1e-12 && wi.w_qk.rows == d && wi.w_qk.cols == d) {
                    Matrix& w_k = access.get_matrix(k_key, true);
                    const auto qk_modes = svd_top_k(wi.w_qk, hd, 30, 0);
                    for (std::size_t i = 0; i < qk_modes.size() && i < hd; ++i) {
                        const double s = std::sqrt(std::max(0.0, qk_modes[i].sigma));
                        w_k.set_row(g * hd + i, vec_scale(qk_modes[i].v, s));
                    }
                    k_g = extract_row_slice(w_k, g * hd, (g + 1) * hd);
                }
                if (frob_norm(v_g) < 1e-12 && wi.w_ov.rows == d && wi.w_ov.cols == d) {
                    Matrix& w_v = access.get_matrix(v_key, true);
                    const auto ov_modes = svd_top_k(wi.w_ov, hd, 30, 0);
                    for (std::size_t i = 0; i < ov_modes.size() && i < hd; ++i) {
                        const double s = std::sqrt(std::max(0.0, ov_modes[i].sigma));
                        w_v.set_row(g * hd + i, vec_scale(ov_modes[i].v, s));
                    }
                    v_g = extract_row_slice(w_v, g * hd, (g + 1) * hd);
                }

                if (wi.w_qk.rows == d && wi.w_qk.cols == d) {
                    Matrix kkt = mat_mul(k_g, transpose(k_g));
                    for (std::size_t i = 0; i < hd; ++i) {
                        kkt(i, i) += 1e-4;
                    }
                    const Matrix delta_q = mat_mul(mat_mul(mat_inv(kkt), k_g), transpose(wi.w_qk));
                    add_row_slice(w_q, hq * hd, delta_q);
                }
                if (wi.w_ov.rows == d && wi.w_ov.cols == d) {
                    Matrix vvt = mat_mul(v_g, transpose(v_g));
                    for (std::size_t i = 0; i < hd; ++i) {
                        vvt(i, i) += 1e-4;
                    }
                    const Matrix delta_o = mat_mul(mat_mul(wi.w_ov, transpose(v_g)), mat_inv(vvt));
                    add_col_slice(w_o, hq * hd, delta_o);
                }
            } else if (attn_type == "mha") {
                Matrix& w_k = access.get_matrix(k_key, true);
                Matrix& w_v = access.get_matrix(v_key, true);
                const std::size_t hq = static_cast<std::size_t>(std::max(0, wi.head_group)) % n_q;
                if (wi.w_qk.rows == d && wi.w_qk.cols == d) {
                    const auto qk_modes = svd_top_k(wi.w_qk, hd, 40, 0);
                    for (std::size_t i = 0; i < qk_modes.size() && i < hd; ++i) {
                        const double s = std::sqrt(std::max(0.0, qk_modes[i].sigma));
                        w_q.set_row(hq * hd + i, vec_add(w_q.row(hq * hd + i), vec_scale(qk_modes[i].u, s)));
                        w_k.set_row(hq * hd + i, vec_add(w_k.row(hq * hd + i), vec_scale(qk_modes[i].v, s)));
                    }
                }
                if (wi.w_ov.rows == d && wi.w_ov.cols == d) {
                    const auto ov_modes = svd_top_k(wi.w_ov, hd, 40, 0);
                    for (std::size_t i = 0; i < ov_modes.size() && i < hd; ++i) {
                        const double s = std::sqrt(std::max(0.0, ov_modes[i].sigma));
                        w_o.set_col(hq * hd + i, vec_add(w_o.col(hq * hd + i), vec_scale(ov_modes[i].u, s)));
                        w_v.set_row(hq * hd + i, vec_add(w_v.row(hq * hd + i), vec_scale(ov_modes[i].v, s)));
                    }
                }
            } else {
                const std::size_t hq = static_cast<std::size_t>(std::max(0, wi.head_group)) % n_q;
                const std::size_t q_dim = n_q * hd;
                const std::string kva_key = prefix + ".self_attn.kv_a_proj.weight";
                const std::string kvb_key = prefix + ".self_attn.kv_b_proj.weight";
                if (access.has_matrix(kva_key) && access.has_matrix(kvb_key)) {
                    const Matrix& kva_ro = access.get_matrix(kva_key, false);
                    const Matrix& kvb_ro = access.get_matrix(kvb_key, false);
                    if (frob_norm(kva_ro) < 1e-12 && wi.w_qk.rows == d && wi.w_qk.cols == d) {
                        Matrix& w_kva = access.get_matrix(kva_key, true);
                        Matrix& w_kvb = access.get_matrix(kvb_key, true);
                        Matrix& w_k = access.get_matrix(k_key, true);
                        Matrix& w_v = access.get_matrix(v_key, true);
                        const std::size_t rank = w_kva.rows;
                        const auto modes = svd_top_k(wi.w_qk, rank, 30, 0);
                        for (std::size_t i = 0; i < modes.size() && i < rank; ++i) {
                            w_kva.set_row(i, modes[i].v);
                        }
                        for (std::size_t r = 0; r < std::min(rank, q_dim); ++r) {
                            w_kvb(r, r) = 1.0;
                            w_kvb(q_dim + r, r) = 1.0;
                        }
                        w_k = mat_mul(extract_row_slice(w_kvb, 0, q_dim), w_kva);
                        w_v = mat_mul(extract_row_slice(w_kvb, q_dim, 2 * q_dim), w_kva);
                    } else if (access.has_matrix(k_key) && access.has_matrix(v_key)) {
                        (void)kvb_ro;
                    }
                }

                const Matrix k_all = access.has_matrix(k_key)
                    ? access.get_matrix(k_key, false)
                    : mat_mul(
                          extract_row_slice(access.get_matrix(kvb_key, false), 0, q_dim),
                          access.get_matrix(kva_key, false));
                const Matrix v_all = access.has_matrix(v_key)
                    ? access.get_matrix(v_key, false)
                    : mat_mul(
                          extract_row_slice(access.get_matrix(kvb_key, false), q_dim, 2 * q_dim),
                          access.get_matrix(kva_key, false));

                const Matrix k_h = extract_row_slice(k_all, hq * hd, (hq + 1) * hd);
                const Matrix v_h = extract_row_slice(v_all, hq * hd, (hq + 1) * hd);
                if (wi.w_qk.rows == d && wi.w_qk.cols == d) {
                    Matrix kkt = mat_mul(k_h, transpose(k_h));
                    for (std::size_t i = 0; i < hd; ++i) {
                        kkt(i, i) += 1e-4;
                    }
                    const Matrix delta_q = mat_mul(mat_mul(mat_inv(kkt), k_h), transpose(wi.w_qk));
                    add_row_slice(w_q, hq * hd, delta_q);
                }
                if (wi.w_ov.rows == d && wi.w_ov.cols == d) {
                    Matrix vvt = mat_mul(v_h, transpose(v_h));
                    for (std::size_t i = 0; i < hd; ++i) {
                        vvt(i, i) += 1e-4;
                    }
                    const Matrix delta_o = mat_mul(mat_mul(wi.w_ov, transpose(v_h)), mat_inv(vvt));
                    add_col_slice(w_o, hq * hd, delta_o);
                }
            }
            ++stats.instructions_applied;
        } else if (wi.kind == WISAKind::ASSOC_MEM_ROW) {
            const int layer_eff = resolve_layer(wi.layer, "ASSOC_MEM_ROW");
            std::size_t batch_end = idx + 1;
            while (batch_end < insts.size() &&
                   insts[batch_end].kind == WISAKind::ASSOC_MEM_ROW &&
                   resolve_layer(insts[batch_end].layer, "ASSOC_MEM_ROW") == layer_eff) {
                ++batch_end;
            }

            const std::string prefix = "model.layers." + std::to_string(layer_eff);
            Matrix& w_down = access.get_matrix(prefix + ".mlp.down_proj.weight", true);
            const std::size_t d_ff_l = w_down.cols;

            Matrix shared_c_inv = eye(d_ff_l);
            bool have_c_inv = false;

            std::vector<Vector> keys_list;
            std::vector<Vector> vals_list;
            keys_list.reserve(batch_end - idx);
            vals_list.reserve(batch_end - idx);

            for (std::size_t p = idx; p < batch_end; ++p) {
                const WISAInstruction& cur = insts[p];
                Vector k_eff;
                if (cur.k_star.size() == d_ff_l) {
                    k_eff = cur.k_star;
                    if (!have_c_inv && cur.c_inv.rows == d_ff_l && cur.c_inv.cols == d_ff_l) {
                        shared_c_inv = cur.c_inv;
                        have_c_inv = true;
                    }
                } else if (cur.k_star.size() == d) {
                    const Matrix& w_gate = access.get_matrix(prefix + ".mlp.gate_proj.weight", false);
                    const Matrix& w_up = access.get_matrix(prefix + ".mlp.up_proj.weight", false);
                    const Vector g_pre = mat_vec(w_gate, cur.k_star);
                    const Vector u_val = mat_vec(w_up, cur.k_star);
                    k_eff.resize(d_ff_l);
                    for (std::size_t i = 0; i < d_ff_l; ++i) {
                        k_eff[i] = silu(g_pre[i]) * u_val[i];
                    }
                    if (norm2(k_eff) < 1e-10) {
                        k_eff = u_val;
                    }
                    if (norm2(k_eff) < 1e-10 && d_ff_l > 0) {
                        k_eff.assign(d_ff_l, 0.0);
                        k_eff[0] = 1.0;
                    }
                } else if (!cur.k_star.empty() && cur.k_star.size() < d_ff_l) {
                    k_eff.assign(d_ff_l, 0.0);
                    std::copy(cur.k_star.begin(), cur.k_star.end(), k_eff.begin());
                    if (!have_c_inv &&
                        cur.c_inv.rows == cur.k_star.size() &&
                        cur.c_inv.cols == cur.k_star.size()) {
                        for (std::size_t r = 0; r < cur.k_star.size(); ++r) {
                            for (std::size_t c = 0; c < cur.k_star.size(); ++c) {
                                shared_c_inv(r, c) = cur.c_inv(r, c);
                            }
                        }
                        have_c_inv = true;
                    }
                } else {
                    throw std::invalid_argument("ASSOC_MEM_ROW k_star dimension mismatch");
                }

                if (cur.v_star.size() != d) {
                    throw std::invalid_argument("ASSOC_MEM_ROW v_star dimension mismatch");
                }

                if (norm2(k_eff) <= 1e-12) {
                    continue;
                }

                bool updated_dup = false;
                for (std::size_t j = 0; j < keys_list.size(); ++j) {
                    if (norm2(vec_sub(k_eff, keys_list[j])) <= 1e-12) {
                        vals_list[j] = cur.v_star;
                        updated_dup = true;
                        break;
                    }
                }
                if (!updated_dup) {
                    keys_list.push_back(std::move(k_eff));
                    vals_list.push_back(cur.v_star);
                }
            }

            const std::size_t m_cnt = keys_list.size();
            if (m_cnt > 0 && d > 0 && d_ff_l > 0) {
                Matrix k_mat(d_ff_l, m_cnt, 0.0);
                for (std::size_t m = 0; m < m_cnt; ++m) {
                    k_mat.set_col(m, keys_list[m]);
                }
                const Matrix u_mat = mat_mul(shared_c_inv, k_mat);
                const Matrix curr_mat = mat_mul(w_down, k_mat);
                Matrix r_mat(d, m_cnt, 0.0);
                for (std::size_t row = 0; row < d; ++row) {
                    for (std::size_t m = 0; m < m_cnt; ++m) {
                        r_mat(row, m) = vals_list[m][row] - curr_mat(row, m);
                    }
                }
                Matrix g_mat = mat_mul(transpose(k_mat), u_mat);
                if (m_cnt >= 2) {
                    for (std::size_t m = 0; m < m_cnt; ++m) {
                        g_mat(m, m) += 1e-6;
                    }
                }
                if (m_cnt >= 2 || std::abs(g_mat(0, 0)) > 1e-12) {
                    const Matrix z_mat = solve_linear_multi(g_mat, transpose(r_mat));
                    const Matrix delta = mat_mul(transpose(z_mat), transpose(u_mat));
                    w_down = mat_add(w_down, delta);
                }
            }

            stats.instructions_applied += static_cast<int>(batch_end - idx);
            idx = batch_end - 1;
        } else if (wi.kind == WISAKind::FIELD_SPECTRAL_OP) {
            int lo = std::max(0, wi.layer_range.first);
            int hi = std::min(cfg.num_layers, std::max(lo + 1, wi.layer_range.second));
            if (lo >= cfg.num_layers) {
                if (lora && cfg.num_layers > 0) {
                    lo = lo % cfg.num_layers;
                    hi = std::min(cfg.num_layers, lo + 1);
                } else {
                    throw OutOfDepthError("FIELD_SPECTRAL_OP layer range out of bounds");
                }
            }
            const double layer_scale = 1.0 / static_cast<double>(std::max(1, hi - lo));
            Matrix delta = zeros(d, d);
            for (std::size_t k = 0; k < wi.coeff_delta.size() && k < wi.u_k.size() && k < wi.v_k.size(); ++k) {
                if (wi.u_k[k].size() == d && wi.v_k[k].size() == d) {
                    const double s = wi.coeff_delta[k] * layer_scale;
                    for (std::size_t r = 0; r < d; ++r) {
                        const double su = s * wi.u_k[k][r];
                        for (std::size_t c = 0; c < d; ++c) {
                            delta(r, c) += su * wi.v_k[k][c];
                        }
                    }
                }
            }
            for (int l = lo; l < hi; ++l) {
                const std::string o_key = "model.layers." + std::to_string(l) + ".self_attn.o_proj.weight";
                Matrix& w_o = access.get_matrix(o_key, true);
                w_o = mat_add(w_o, delta);
            }
            ++stats.instructions_applied;
        } else if (wi.kind == WISAKind::NORM_RESCALE) {
            if (lora) {
                continue;
            }
            if (wi.layer < 0 || wi.layer >= cfg.num_layers) {
                throw OutOfDepthError("NORM_RESCALE layer out of range");
            }
            const std::string norm_key = (wi.norm_id == "model.norm")
                ? std::string("model.norm.weight")
                : ("model.layers." + std::to_string(wi.layer) + "." + wi.norm_id + ".weight");
            if (!access.has_vector(norm_key)) {
                throw std::invalid_argument("Unknown norm_id in NORM_RESCALE: " + norm_key);
            }
            Vector& gamma = access.get_vector(norm_key, true);
            for (std::size_t i = 0; i < gamma.size() && i < wi.gamma_scale.size(); ++i) {
                gamma[i] *= (1.0 + wi.gamma_scale[i]);
            }
            ++stats.instructions_applied;
        }
    }
}

void write_patched_shard_file(
    const std::filesystem::path& src_shard,
    const std::filesystem::path& dst_shard,
    const std::map<std::string, TensorSpec>& patched_specs,
    const std::map<std::string, TensorSpec>& extra_specs,
    std::int64_t& byte_delta) {
    std::int64_t delta = 0;
    rewrite_safetensors_selective(
        src_shard.string(),
        dst_shard.string(),
        patched_specs,
        extra_specs,
        {},
        &delta);
    byte_delta += delta;
}

}

std::pair<TransformerModel, CompileStats> apply_wisa(
    const TransformerModel& model,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode) {
    const std::string mode = target_mode.empty() ? std::string("in-place") : target_mode;
    const bool elastic = (mode == "elastic");
    const bool lora = (mode == "lora");

    TransformerModel out = model;
    CompileStats stats;

    int max_layer_needed = out.config.num_layers - 1;
    for (std::size_t idx = 0; idx < insts.size(); ++idx) {
        const auto& wi = insts[idx];
        if (wi.kind == WISAKind::FIELD_SPECTRAL_OP) {
            max_layer_needed = std::max(max_layer_needed, wi.layer_range.second - 1);
        } else {
            max_layer_needed = std::max(max_layer_needed, wi.layer);
        }
    }

    if (max_layer_needed >= out.config.num_layers) {
        if (!elastic && !lora) {
            throw OutOfDepthError(
                "W-ISA targets layer " + std::to_string(max_layer_needed) +
                " beyond num_layers=" + std::to_string(out.config.num_layers));
        }
        if (elastic) {
            for (int l = out.config.num_layers; l <= max_layer_needed; ++l) {
                populate_zero_bypass_layer(
                    out.config,
                    l,
                    [&](const std::string& k, Matrix m) { out.matrices[k] = std::move(m); },
                    [&](const std::string& k, Vector v) { out.vectors[k] = std::move(v); });
                ++stats.added_layers;
            }
            out.config.num_layers = max_layer_needed + 1;
        }
    }

    for (const auto& wi : insts) {
        if (wi.kind != WISAKind::THRESHOLDED_ROUTE) {
            continue;
        }
        if (wi.layer < 0 || wi.layer >= out.config.num_layers) {
            throw OutOfDepthError("THRESHOLDED_ROUTE layer out of range");
        }
        const std::string prefix = "model.layers." + std::to_string(wi.layer) +
            ".mlp.aic_threshold_route";
        const std::string gate_key = "model.layers." + std::to_string(wi.layer) +
            ".mlp.gate_proj.weight";
        const std::size_t d_ff = out.matrices.at(gate_key).rows;
        if (!out.matrices.contains(prefix + ".weight")) {
            out.matrices[prefix + ".weight"] = zeros(
                static_cast<std::size_t>(out.config.hidden_size), d_ff);
            out.vectors[prefix + ".threshold"] = Vector(d_ff, 0.0);
            out.vectors[prefix + ".polarity"] = Vector(d_ff, 0.0);
            out.vectors[prefix + ".mask"] = Vector(d_ff, 0.0);
        }
        out.config.aic_threshold_routes = true;
    }

    TensorAccess access{
        [&](const std::string& key, bool write) -> Matrix& {
            (void)write;
            return out.matrices.at(key);
        },
        [&](const std::string& key, bool write) -> Vector& {
            (void)write;
            return out.vectors.at(key);
        },
        [&](const std::string& key) -> bool {
            return out.matrices.contains(key);
        },
        [&](const std::string& key) -> bool {
            return out.vectors.contains(key);
        }};

    execute_wisa_instructions(out.config, insts, elastic, lora, access, stats);

    if (elastic && stats.added_poles > 0) {
        std::size_t max_d_ff = static_cast<std::size_t>(out.config.intermediate_size);
        for (int l = 0; l < out.config.num_layers; ++l) {
            const std::string gkey = "model.layers." + std::to_string(l) + ".mlp.gate_proj.weight";
            max_d_ff = std::max(max_d_ff, out.matrices.at(gkey).rows);
        }
        out.config.intermediate_size = static_cast<int>(max_d_ff);
        for (int l = 0; l < out.config.num_layers; ++l) {
            const std::string prefix = "model.layers." + std::to_string(l);
            Matrix& w_gate = out.matrices.at(prefix + ".mlp.gate_proj.weight");
            Matrix& w_up = out.matrices.at(prefix + ".mlp.up_proj.weight");
            Matrix& w_down = out.matrices.at(prefix + ".mlp.down_proj.weight");
            if (w_gate.rows < max_d_ff) {
                w_gate = append_zero_rows(w_gate, max_d_ff - w_gate.rows);
            }
            if (w_up.rows < max_d_ff) {
                w_up = append_zero_rows(w_up, max_d_ff - w_up.rows);
            }
            if (w_down.cols < max_d_ff) {
                w_down = append_zero_cols(w_down, max_d_ff - w_down.cols);
            }
        }
    }

    return {std::move(out), stats};
}

CompileStats apply_wisa_streaming_checkpoint(
    const std::string& src_model_dir,
    const std::string& dst_model_dir,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode) {
    const std::string mode = target_mode.empty() ? std::string("in-place") : target_mode;
    const bool elastic = (mode == "elastic");
    const bool lora = (mode == "lora");

    const std::filesystem::path src_dir(src_model_dir);
    const std::filesystem::path dst_dir(dst_model_dir);
    std::filesystem::create_directories(dst_dir);

    JsonValue cfg_json = parse_json(read_text_file(src_dir / "config.json"));
    ModelConfig cfg = ModelConfig::from_json(cfg_json);

    const std::filesystem::path index_path = src_dir / "model.safetensors.index.json";
    const std::filesystem::path single_path = src_dir / "model.safetensors";
    const bool is_sharded = std::filesystem::exists(index_path);
    if (!is_sharded && !std::filesystem::exists(single_path)) {
        throw IOError("No model.safetensors or model.safetensors.index.json found in: " + src_model_dir);
    }

    JsonValue index_json;
    std::map<std::string, std::string> weight_map;
    std::vector<std::string> shard_order;
    std::set<std::string> shard_seen;

    if (is_sharded) {
        index_json = parse_json(read_text_file(index_path));
        if (!index_json.is_object() || !index_json.has("weight_map") || !index_json["weight_map"].is_object()) {
            throw IOError("Invalid model.safetensors.index.json in: " + src_model_dir);
        }
        for (const auto& [tname, shard_val] : index_json["weight_map"].as_object()) {
            const std::string sfile = shard_val.as_string();
            weight_map[tname] = sfile;
            if (!shard_seen.contains(sfile)) {
                shard_seen.insert(sfile);
                shard_order.push_back(sfile);
            }
        }
        std::sort(shard_order.begin(), shard_order.end());
    } else {
        MmapSafetensors reader(single_path.string());
        for (const std::string& tname : reader.tensor_names()) {
            weight_map[tname] = "model.safetensors";
        }
        shard_order.push_back("model.safetensors");
    }

    CompileStats stats;
    int max_layer_needed = cfg.num_layers - 1;
    for (std::size_t idx = 0; idx < insts.size(); ++idx) {
        const auto& wi = insts[idx];
        if (wi.kind == WISAKind::FIELD_SPECTRAL_OP) {
            max_layer_needed = std::max(max_layer_needed, wi.layer_range.second - 1);
        } else {
            max_layer_needed = std::max(max_layer_needed, wi.layer);
        }
    }

    if (max_layer_needed >= cfg.num_layers) {
        if (!elastic && !lora) {
            throw OutOfDepthError(
                "W-ISA targets layer " + std::to_string(max_layer_needed) +
                " beyond num_layers=" + std::to_string(cfg.num_layers));
        }
    }

    std::string default_dtype = normalize_dtype_upper(cfg.torch_dtype);
    if (!weight_map.empty()) {
        const auto first_it = weight_map.begin();
        MmapSafetensors probe((src_dir / first_it->second).string());
        default_dtype = normalize_dtype_upper(probe.tensor_dtype(first_it->first));
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

    std::map<std::string, Matrix> loaded_matrices;
    std::map<std::string, Vector> loaded_vectors;
    std::map<std::string, std::string> tensor_dtypes;
    std::set<std::string> written_tensors;
    std::set<std::string> new_bypass_tensors;

    auto ensure_matrix = [&](const std::string& key, bool write) -> Matrix& {
        if (write) {
            written_tensors.insert(key);
        }
        auto it = loaded_matrices.find(key);
        if (it != loaded_matrices.end()) {
            return it->second;
        }
        auto wmit = weight_map.find(key);
        if (wmit == weight_map.end()) {
            throw std::out_of_range("Missing matrix tensor in checkpoint: " + key);
        }
        MmapSafetensors& rd = get_reader(wmit->second);
        TensorSpec spec = rd.read_tensor(key);
        tensor_dtypes[key] = spec.dtype;
        auto [ins_it, ok] = loaded_matrices.emplace(key, spec.to_matrix());
        (void)ok;
        return ins_it->second;
    };

    auto ensure_vector = [&](const std::string& key, bool write) -> Vector& {
        if (write) {
            written_tensors.insert(key);
        }
        auto it = loaded_vectors.find(key);
        if (it != loaded_vectors.end()) {
            return it->second;
        }
        auto wmit = weight_map.find(key);
        if (wmit == weight_map.end()) {
            throw std::out_of_range("Missing vector tensor in checkpoint: " + key);
        }
        MmapSafetensors& rd = get_reader(wmit->second);
        TensorSpec spec = rd.read_tensor(key);
        tensor_dtypes[key] = spec.dtype;
        auto [ins_it, ok] = loaded_vectors.emplace(key, spec.to_vector());
        (void)ok;
        return ins_it->second;
    };

    for (const auto& wi : insts) {
        if (wi.kind != WISAKind::THRESHOLDED_ROUTE) {
            continue;
        }
        if (wi.layer < 0 || wi.layer >= cfg.num_layers) {
            throw OutOfDepthError("THRESHOLDED_ROUTE layer out of range");
        }
        const std::string layer_prefix = "model.layers." + std::to_string(wi.layer);
        const std::string gate_key = layer_prefix + ".mlp.gate_proj.weight";
        auto gate_it = weight_map.find(gate_key);
        if (gate_it == weight_map.end()) {
            throw IOError("Missing detector layer gate projection: " + gate_key);
        }
        const auto gate_shape = get_reader(gate_it->second).tensor_shape(gate_key);
        if (gate_shape.size() != 2) {
            throw IOError("Invalid detector layer gate projection shape: " + gate_key);
        }
        const std::size_t d_ff = gate_shape[0];
        const std::string route_prefix = layer_prefix + ".mlp.aic_threshold_route";
        const std::string route_key = route_prefix + ".weight";
        if (!weight_map.contains(route_key)) {
            loaded_matrices[route_key] = zeros(
                static_cast<std::size_t>(cfg.hidden_size), d_ff);
            loaded_vectors[route_prefix + ".threshold"] = Vector(d_ff, 0.0);
            loaded_vectors[route_prefix + ".polarity"] = Vector(d_ff, 0.0);
            loaded_vectors[route_prefix + ".mask"] = Vector(d_ff, 0.0);
            for (const std::string& key : {
                     route_key,
                     route_prefix + ".threshold",
                     route_prefix + ".polarity",
                     route_prefix + ".mask"}) {
                tensor_dtypes[key] = default_dtype;
                written_tensors.insert(key);
                new_bypass_tensors.insert(key);
                weight_map[key] = gate_it->second;
            }
        }
        cfg.aic_threshold_routes = true;
    }

    if (elastic && max_layer_needed >= cfg.num_layers) {
        const std::string target_shard = shard_order.empty() ? std::string("model.safetensors") : shard_order.back();
        for (int l = cfg.num_layers; l <= max_layer_needed; ++l) {
            populate_zero_bypass_layer(
                cfg,
                l,
                [&](const std::string& k, Matrix m) {
                    loaded_matrices[k] = std::move(m);
                    tensor_dtypes[k] = default_dtype;
                    written_tensors.insert(k);
                    new_bypass_tensors.insert(k);
                    weight_map[k] = target_shard;
                },
                [&](const std::string& k, Vector v) {
                    loaded_vectors[k] = std::move(v);
                    tensor_dtypes[k] = default_dtype;
                    written_tensors.insert(k);
                    new_bypass_tensors.insert(k);
                    weight_map[k] = target_shard;
                });
            ++stats.added_layers;
        }
        cfg.num_layers = max_layer_needed + 1;
    }

    TensorAccess access{
        ensure_matrix,
        ensure_vector,
        [&](const std::string& key) -> bool {
            return loaded_matrices.contains(key) || weight_map.contains(key);
        },
        [&](const std::string& key) -> bool {
            return loaded_vectors.contains(key) || weight_map.contains(key);
        }};

    execute_wisa_instructions(cfg, insts, elastic, lora, access, stats);

    if (elastic && stats.added_poles > 0) {
        std::size_t max_d_ff = static_cast<std::size_t>(cfg.intermediate_size);
        for (int l = 0; l < cfg.num_layers; ++l) {
            const std::string gkey = "model.layers." + std::to_string(l) + ".mlp.gate_proj.weight";
            if (loaded_matrices.contains(gkey)) {
                max_d_ff = std::max(max_d_ff, loaded_matrices.at(gkey).rows);
            } else if (weight_map.contains(gkey)) {
                const auto shape = get_reader(weight_map.at(gkey)).tensor_shape(gkey);
                if (!shape.empty()) {
                    max_d_ff = std::max(max_d_ff, shape[0]);
                }
            }
        }
        cfg.intermediate_size = static_cast<int>(max_d_ff);
        for (int l = 0; l < cfg.num_layers; ++l) {
            const std::string prefix = "model.layers." + std::to_string(l);
            const std::string gk = prefix + ".mlp.gate_proj.weight";
            const std::string uk = prefix + ".mlp.up_proj.weight";
            const std::string dk = prefix + ".mlp.down_proj.weight";

            auto check_rows = [&](const std::string& k) -> std::size_t {
                if (loaded_matrices.contains(k)) {
                    return loaded_matrices.at(k).rows;
                }
                if (weight_map.contains(k)) {
                    const auto sh = get_reader(weight_map.at(k)).tensor_shape(k);
                    return sh.empty() ? max_d_ff : sh[0];
                }
                return max_d_ff;
            };
            auto check_cols = [&](const std::string& k) -> std::size_t {
                if (loaded_matrices.contains(k)) {
                    return loaded_matrices.at(k).cols;
                }
                if (weight_map.contains(k)) {
                    const auto sh = get_reader(weight_map.at(k)).tensor_shape(k);
                    return (sh.size() < 2) ? max_d_ff : sh[1];
                }
                return max_d_ff;
            };

            if (check_rows(gk) < max_d_ff) {
                Matrix& w_gate = ensure_matrix(gk, true);
                w_gate = append_zero_rows(w_gate, max_d_ff - w_gate.rows);
            }
            if (check_rows(uk) < max_d_ff) {
                Matrix& w_up = ensure_matrix(uk, true);
                w_up = append_zero_rows(w_up, max_d_ff - w_up.rows);
            }
            if (check_cols(dk) < max_d_ff) {
                Matrix& w_down = ensure_matrix(dk, true);
                w_down = append_zero_cols(w_down, max_d_ff - w_down.cols);
            }
        }
    }

    reader_cache.clear();

    std::set<std::string> touched_shards;
    std::map<std::string, std::map<std::string, TensorSpec>> shard_patched_specs;
    std::map<std::string, std::map<std::string, TensorSpec>> shard_extra_specs;

    for (const std::string& tname : written_tensors) {
        const std::string& shard_file = weight_map.at(tname);
        touched_shards.insert(shard_file);
        const std::string dt = tensor_dtypes.contains(tname) ? tensor_dtypes.at(tname) : default_dtype;
        TensorSpec spec;
        if (loaded_matrices.contains(tname)) {
            spec = TensorSpec::from_matrix(loaded_matrices.at(tname), dt);
        } else {
            spec = TensorSpec::from_vector(loaded_vectors.at(tname), dt);
        }
        if (new_bypass_tensors.contains(tname)) {
            shard_extra_specs[shard_file][tname] = std::move(spec);
        } else {
            shard_patched_specs[shard_file][tname] = std::move(spec);
        }
    }

    std::error_code eq_ec;
    const bool same_dir = std::filesystem::equivalent(src_dir, dst_dir, eq_ec);

    std::int64_t total_byte_delta = 0;
    for (const std::string& shard_file : shard_order) {
        const std::filesystem::path src_shard = src_dir / shard_file;
        const std::filesystem::path dst_shard = dst_dir / shard_file;
        if (touched_shards.contains(shard_file)) {
            write_patched_shard_file(
                src_shard,
                dst_shard,
                shard_patched_specs[shard_file],
                shard_extra_specs[shard_file],
                total_byte_delta);
        } else if (!same_dir) {
            std::filesystem::copy_file(
                src_shard,
                dst_shard,
                std::filesystem::copy_options::overwrite_existing);
        }
    }

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

    cfg_json["num_layers"] = cfg.num_layers;
    if (cfg_json.has("num_hidden_layers")) {
        cfg_json["num_hidden_layers"] = cfg.num_layers;
    }
    cfg_json["intermediate_size"] = cfg.intermediate_size;
    cfg_json["aic_threshold_routes"] = cfg.aic_threshold_routes;
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
        if (std::filesystem::exists(atlas_src) && !elastic) {
            std::filesystem::copy_file(
                atlas_src,
                dst_dir / "model.natlas",
                std::filesystem::copy_options::overwrite_existing);
        }
    }

    return stats;
}

CompileStats emit_lora_adapter(
    const TransformerModel& base_model,
    const std::vector<WISAInstruction>& insts,
    const std::string& out_lora_dir,
    int max_lora_rank,
    double lora_alpha) {
    auto [patched, stats] = apply_wisa(base_model, insts, "lora");
    const int rank_cap = std::max(1, max_lora_rank);

    struct DeltaFactor {
        std::string key;
        std::size_t rows = 0;
        std::size_t cols = 0;
        Matrix delta;
        std::vector<SVDMode> modes;
    };
    std::vector<DeltaFactor> factors;
    int max_used_rank = 0;

    for (const auto& [key, w_patched] : patched.matrices) {
        auto bit = base_model.matrices.find(key);
        if (bit == base_model.matrices.end()) {
            continue;
        }
        const Matrix& w_base = bit->second;
        if (w_patched.rows != w_base.rows || w_patched.cols != w_base.cols) {
            continue;
        }
        Matrix delta = mat_sub(w_patched, w_base);
        if (frob_norm(delta) <= 1e-12) {
            continue;
        }
        const std::size_t k_lim = std::min({
            static_cast<std::size_t>(rank_cap),
            delta.rows,
            delta.cols});
        std::vector<SVDMode> raw_modes = svd_top_k(delta, k_lim, 50, 0);
        std::vector<SVDMode> kept_modes;
        for (auto& m : raw_modes) {
            if (m.sigma > 1e-12) {
                kept_modes.push_back(std::move(m));
            }
        }
        if (kept_modes.empty()) {
            continue;
        }
        max_used_rank = std::max(max_used_rank, static_cast<int>(kept_modes.size()));
        factors.push_back(DeltaFactor{
            key,
            delta.rows,
            delta.cols,
            std::move(delta),
            std::move(kept_modes)});
    }

    const int r_global = std::max(1, std::min(rank_cap, max_used_rank));
    const double alpha_eff = (std::abs(lora_alpha) > 1e-12)
        ? lora_alpha
        : static_cast<double>(r_global);
    const double scale_inv = static_cast<double>(r_global) / alpha_eff;
    const double apply_scale = alpha_eff / static_cast<double>(r_global);

    std::map<std::string, TensorSpec> lora_tensors;
    std::set<std::string> target_modules;

    for (const auto& df : factors) {
        Matrix a_mat(static_cast<std::size_t>(r_global), df.cols, 0.0);
        Matrix b_mat(df.rows, static_cast<std::size_t>(r_global), 0.0);
        const std::size_t n_copy = std::min(
            df.modes.size(),
            static_cast<std::size_t>(r_global));
        for (std::size_t j = 0; j < n_copy; ++j) {
            const double s_root = std::sqrt(std::max(0.0, df.modes[j].sigma * scale_inv));
            b_mat.set_col(j, vec_scale(df.modes[j].u, s_root));
            a_mat.set_row(j, vec_scale(df.modes[j].v, s_root));
        }

        const Matrix recon = mat_scale(mat_mul(b_mat, a_mat), apply_scale);
        const double err = frob_norm(mat_sub(df.delta, recon));
        stats.max_approx_error = std::max(stats.max_approx_error, err);

        const std::string stem = (df.key.size() > 7 && df.key.ends_with(".weight"))
            ? df.key.substr(0, df.key.size() - 7)
            : df.key;
        const std::string peft_stem = "base_model.model." + stem;
        lora_tensors[peft_stem + ".lora_A.weight"] = TensorSpec::from_matrix(a_mat, "F32");
        lora_tensors[peft_stem + ".lora_B.weight"] = TensorSpec::from_matrix(b_mat, "F32");

        const std::size_t dot_pos = stem.rfind('.');
        const std::string leaf = (dot_pos == std::string::npos)
            ? stem
            : stem.substr(dot_pos + 1);
        if (!leaf.empty()) {
            target_modules.insert(leaf);
        }
    }

    const std::filesystem::path out_dir(out_lora_dir);
    std::filesystem::create_directories(out_dir);

    std::map<std::string, std::string> meta;
    meta["format"] = "pt";
    meta["aic_target"] = "lora";
    save_safetensors((out_dir / "adapter_model.safetensors").string(), lora_tensors, meta);

    JsonValue cfg_obj = JsonValue::object();
    cfg_obj["peft_type"] = "LORA";
    cfg_obj["r"] = r_global;
    cfg_obj["lora_alpha"] = alpha_eff;
    JsonValue mod_arr = JsonValue::array();
    for (const std::string& tm : target_modules) {
        mod_arr.as_array().emplace_back(tm);
    }
    cfg_obj["target_modules"] = std::move(mod_arr);
    cfg_obj["bias"] = "none";
    cfg_obj["task_type"] = "CAUSAL_LM";
    write_text_file(out_dir / "adapter_config.json", dump_json(cfg_obj, 2) + "\n");

    return stats;
}

CompileOutput compile_mir_to_model(
    const MIRModule& mir,
    const TransformerModel& model,
    const NeuralAtlas* atlas,
    const std::string& target_mode,
    int opt_level) {
    NeuralAtlas default_atlas;
    const NeuralAtlas* active_atlas = atlas;
    if (active_atlas == nullptr) {
        default_atlas = build_default_atlas(model, &mir);
        active_atlas = &default_atlas;
    }

    CIRModule cir = lower_mir_to_cir(mir, model.config, *active_atlas, target_mode);
    if (opt_level >= 1) {
        cir = optimize_cir(cir, opt_level);
    }
    std::vector<WISAInstruction> wisa = lower_cir_to_wisa(cir, model.config, *active_atlas, target_mode);
    if (opt_level >= 1) {
        wisa = optimize_wisa(wisa, opt_level);
    }
    auto [patched_model, stats] = apply_wisa(model, wisa, target_mode);

    for (const auto& flow : cir.flows) {
        stats.max_approx_error = std::max(stats.max_approx_error, flow.max_approx_error);
        stats.max_crosstalk = std::max(stats.max_crosstalk, flow.max_crosstalk);
    }

    return CompileOutput{
        std::move(patched_model),
        std::move(cir),
        std::move(wisa),
        stats};
}

}
