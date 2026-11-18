#include "aic/wisa.hpp"

#include "aic/errors.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace aic {

namespace {

std::string lower_str(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

std::string wisa_kind_name(WISAKind k) {
    switch (k) {
        case WISAKind::SWIGLU_POLE: return "SWIGLU_POLE";
        case WISAKind::THRESHOLDED_ROUTE: return "THRESHOLDED_ROUTE";
        case WISAKind::VPD_RANK1: return "VPD_RANK1";
        case WISAKind::ATTN_BILINEAR: return "ATTN_BILINEAR";
        case WISAKind::ASSOC_MEM_ROW: return "ASSOC_MEM_ROW";
        case WISAKind::FIELD_SPECTRAL_OP: return "FIELD_SPECTRAL_OP";
        case WISAKind::NORM_RESCALE: return "NORM_RESCALE";
    }
    return "UNKNOWN";
}

Matrix truncate_matrix_rank(const Matrix& m, std::size_t max_rank) {
    if (m.rows == 0 || m.cols == 0 || max_rank == 0) {
        return m;
    }
    const std::vector<SVDMode> modes = svd_top_k(m, max_rank, 40, 0);
    Matrix out = zeros(m.rows, m.cols);
    for (const auto& mode : modes) {
        out = mat_add(out, mat_scale(outer(mode.u, mode.v), mode.sigma));
    }
    return out;
}

}

std::vector<WISAInstruction> lower_cir_to_wisa(
    const CIRModule& cir,
    const ModelConfig& cfg,
    const NeuralAtlas& atlas,
    const std::string& target_mode) {
    const std::string mode = target_mode.empty()
        ? (cir.target_mode.empty() ? std::string("in-place") : cir.target_mode)
        : target_mode;
    const bool elastic = (mode == "elastic");
    const bool lora = (mode == "lora");
    const std::string attn_type = lower_str(cfg.attention_type);
    const std::size_t d = static_cast<std::size_t>(std::max(0, cfg.hidden_size));
    const std::size_t d_ff = static_cast<std::size_t>(std::max(0, cfg.intermediate_size));

    std::set<std::pair<int, int>> used_dormant;
    std::set<std::pair<int, int>> used_lora_slots;
    std::map<int, int> elastic_slot_cursor;
    std::map<int, int> lora_slot_cursor;

    int max_elastic_layer = cfg.num_layers - 1;
    for (const auto& flow : cir.flows) {
        for (const auto& op : flow.ops) {
            if (op.layer > max_elastic_layer) {
                max_elastic_layer = op.layer;
            }
        }
    }

    auto alloc_dormant = [&](int pref_layer, int l0, int l1) -> std::pair<int, int> {
        for (const auto& [dl, ds] : atlas.dormant_poles) {
            if (dl == pref_layer && !used_dormant.contains({dl, ds})) {
                used_dormant.emplace(dl, ds);
                return {dl, ds};
            }
        }
        for (const auto& [dl, ds] : atlas.dormant_poles) {
            if (dl >= l0 && dl < l1 && !used_dormant.contains({dl, ds})) {
                used_dormant.emplace(dl, ds);
                return {dl, ds};
            }
        }
        if (lora && cfg.num_layers > 0 && cfg.intermediate_size > 0) {
            const int target_l = std::clamp(pref_layer, 0, cfg.num_layers - 1);
            const int n_slots = cfg.intermediate_size;
            int& cur = lora_slot_cursor[target_l];
            for (int probe = 0; probe < n_slots; ++probe) {
                const int cand = (cur + probe) % n_slots;
                if (!used_dormant.contains({target_l, cand}) &&
                    !used_lora_slots.contains({target_l, cand})) {
                    used_lora_slots.emplace(target_l, cand);
                    cur = cand + 1;
                    return {target_l, cand};
                }
            }
            const int wrapped = cur % n_slots;
            ++cur;
            return {target_l, wrapped};
        }
        throw ApproximationUnmetError(
            "Insufficient dormant SwiGLU pole slots in layers [" +
            std::to_string(l0) + ", " + std::to_string(l1) + ") for in-place target");
    };

    std::vector<WISAInstruction> insts;

    for (const auto& flow : cir.flows) {
        const int fl0 = flow.layer_range.first;
        const int fl1 = flow.layer_range.second;

        for (const auto& op : flow.ops) {
            if (op.kind == "swiglu_spline") {
                for (const auto& s : op.spline_slots) {
                    WISAInstruction wi;
                    wi.kind = WISAKind::SWIGLU_POLE;
                    wi.polarity = 1;
                    wi.w_gate = s.w_gate;
                    wi.w_up = s.w_up;
                    wi.w_down = s.w_down;
                    if (elastic) {
                        wi.layer = op.layer;
                        wi.append_slot = true;
                        int& cur = elastic_slot_cursor[op.layer];
                        wi.slot = cfg.intermediate_size + cur;
                        ++cur;
                    } else {
                        auto [al, as] = alloc_dormant(op.layer, fl0, fl1);
                        wi.layer = al;
                        wi.slot = as;
                        wi.append_slot = false;
                    }
                    insts.push_back(std::move(wi));
                }
            } else if (op.kind == "swiglu_pole") {
                WISAInstruction wi;
                wi.kind = WISAKind::SWIGLU_POLE;
                wi.polarity = op.polarity;
                const Vector in_v = op.input_dir.empty() ? vec_zeros(d) : op.input_dir;
                const Vector out_v = op.output_dir.empty() ? vec_zeros(d) : op.output_dir;
                wi.w_gate = vec_scale(in_v, 4.0);
                wi.w_up = in_v;
                wi.w_down = vec_scale(out_v, op.scalar * 0.25);

                if (elastic) {
                    wi.layer = op.layer;
                    wi.append_slot = true;
                    int& cur = elastic_slot_cursor[op.layer];
                    wi.slot = cfg.intermediate_size + cur;
                    ++cur;
                } else {
                    bool bound = false;
                    auto sit = atlas.symbols.find(op.symbol);
                    if (sit != atlas.symbols.end()) {
                        const std::string pkey = op.pole.empty() ? std::string("+pole") : op.pole;
                        auto pit = sit->second.pole_slots.find(pkey);
                        if (pit != sit->second.pole_slots.end()) {
                            const int bl = std::get<0>(pit->second);
                            const int bs = std::get<1>(pit->second);
                            if (bl >= fl0 && bl < fl1 && bs >= 0 && bs < cfg.intermediate_size) {
                                wi.layer = bl;
                                wi.slot = bs;
                                bound = true;
                            }
                        }
                    }
                    if (!bound) {
                        auto [al, as] = alloc_dormant(op.layer, fl0, fl1);
                        wi.layer = al;
                        wi.slot = as;
                    }
                    wi.append_slot = false;
                }
                const int pole_layer = wi.layer;
                insts.push_back(std::move(wi));

                if (!lora && op.scalar > 0.0 && d > 0) {
                    WISAInstruction nr;
                    nr.kind = WISAKind::NORM_RESCALE;
                    nr.layer = pole_layer;
                    nr.norm_id = "post_attention_layernorm";
                    nr.gamma_scale.assign(d, std::min(0.05, 0.01 * op.scalar));
                    insts.push_back(std::move(nr));
                }
            } else if (op.kind == "thresholded_route") {
                if (lora) {
                    throw ApproximationUnmetError(
                        "Thresholded routes require a checkpoint target, not LoRA");
                }
                auto sit = atlas.symbols.find(op.symbol);
                if (sit == atlas.symbols.end()) {
                    throw UndefinedSymbolError(
                        "Thresholded route detector is unresolved: " + op.symbol);
                }
                const std::string pkey = op.pole.empty() ? std::string("+pole") : op.pole;
                auto pit = sit->second.pole_slots.find(pkey);
                if (pit == sit->second.pole_slots.end()) {
                    throw ApproximationUnmetError(
                        "Thresholded route requires a resolved SwiGLU detector slot for " + op.symbol);
                }
                WISAInstruction wi;
                wi.kind = WISAKind::THRESHOLDED_ROUTE;
                wi.layer = std::get<0>(pit->second);
                wi.slot = std::get<1>(pit->second);
                wi.polarity = std::get<2>(pit->second);
                wi.u = op.output_dir;
                wi.scale = op.scalar;
                wi.threshold = op.threshold;
                insts.push_back(std::move(wi));
            } else if (op.kind == "vpd_rank1") {
                WISAInstruction wi;
                wi.kind = WISAKind::VPD_RANK1;
                wi.layer = op.layer;
                wi.u = op.output_dir;
                wi.scale = op.scalar;

                bool used_pole_slot = false;
                if (op.matrix_id == "mlp.down_proj" && d_ff > 0) {
                    auto sit = atlas.symbols.find(op.symbol);
                    if (sit != atlas.symbols.end()) {
                        const std::string pkey = op.pole.empty() ? std::string("+pole") : op.pole;
                        auto pit = sit->second.pole_slots.find(pkey);
                        if (pit != sit->second.pole_slots.end()) {
                            const int bl = std::get<0>(pit->second);
                            const int bs = std::get<1>(pit->second);
                            const int bp = std::get<2>(pit->second);
                            if (bl >= 0 && bl < cfg.num_layers && bs >= 0 &&
                                static_cast<std::size_t>(bs) < d_ff && !elastic) {
                                wi.layer = bl;
                                wi.matrix_id = "mlp.down_proj";
                                wi.v = vec_zeros(d_ff);
                                wi.v[static_cast<std::size_t>(bs)] = 1.0;
                                wi.scale = op.scalar * static_cast<double>(bp);
                                used_pole_slot = true;
                            }
                        }
                    }
                }
                if (!used_pole_slot) {
                    wi.matrix_id = "self_attn.o_proj";
                    wi.v = op.input_dir;
                }
                insts.push_back(std::move(wi));
            } else if (op.kind == "frame_bridge") {
                if (op.bridge_matrix.rows == d && op.bridge_matrix.cols == d && d > 0) {
                    const Matrix delta = mat_sub(op.bridge_matrix, eye(d));
                    if (frob_norm(delta) > 1e-9) {
                        const std::size_t k_modes = std::min<std::size_t>(4, d);
                        const std::vector<SVDMode> modes = svd_top_k(delta, k_modes, 40, 0);
                        for (const auto& m : modes) {
                            if (m.sigma > 1e-9) {
                                WISAInstruction wi;
                                wi.kind = WISAKind::VPD_RANK1;
                                wi.layer = op.layer;
                                wi.matrix_id = "self_attn.o_proj";
                                wi.u = m.u;
                                wi.v = m.v;
                                wi.scale = m.sigma;
                                insts.push_back(std::move(wi));
                            }
                        }
                    }
                }
            } else if (op.kind == "attn_bilinear") {
                WISAInstruction wi;
                wi.kind = WISAKind::ATTN_BILINEAR;
                wi.rope_mode = op.rope_mode;
                wi.w_qk = op.w_qk;
                wi.w_ov = op.w_ov;

                if (attn_type == "gqa") {
                    const int n_kv = std::max(1, cfg.num_key_value_heads);
                    bool found = false;
                    const int scan_lo = (op.layer < cfg.num_layers) ? fl0 : op.layer;
                    const int scan_hi = (op.layer < cfg.num_layers) ? fl1 : (op.layer + 1);

                    for (int cand_l = op.layer; cand_l < scan_hi && !found; ++cand_l) {
                        for (int g = 0; g < n_kv; ++g) {
                            if (!atlas.occupied_kv_groups.contains({cand_l, g})) {
                                wi.layer = cand_l;
                                wi.head_group = g;
                                found = true;
                                break;
                            }
                        }
                    }
                    for (int cand_l = scan_lo; cand_l < op.layer && !found; ++cand_l) {
                        for (int g = 0; g < n_kv; ++g) {
                            if (!atlas.occupied_kv_groups.contains({cand_l, g})) {
                                wi.layer = cand_l;
                                wi.head_group = g;
                                found = true;
                                break;
                            }
                        }
                    }
                    if (!found) {
                        if (!elastic) {
                            throw GQAHeadSharingConflictError(
                                "All GQA KV head groups occupied in layers [" +
                                std::to_string(scan_lo) + ", " + std::to_string(scan_hi) + ")");
                        }
                        ++max_elastic_layer;
                        wi.layer = max_elastic_layer;
                        wi.head_group = 0;
                    }
                } else if (attn_type == "mla") {
                    wi.layer = op.layer;
                    wi.head_group = 0;
                    const std::size_t rank_lim = static_cast<std::size_t>(std::max(1, cfg.kv_lora_rank));
                    wi.w_qk = truncate_matrix_rank(op.w_qk, rank_lim);
                    wi.w_ov = truncate_matrix_rank(op.w_ov, rank_lim);
                } else {
                    const int n_q = std::max(1, cfg.num_attention_heads);
                    int chosen_h = 0;
                    for (int h = 0; h < n_q; ++h) {
                        if (!atlas.occupied_kv_groups.contains({op.layer, h})) {
                            chosen_h = h;
                            break;
                        }
                    }
                    wi.layer = op.layer;
                    wi.head_group = chosen_h;
                }
                insts.push_back(std::move(wi));
            } else if (op.kind == "assoc_mem_row") {
                WISAInstruction wi;
                wi.kind = WISAKind::ASSOC_MEM_ROW;
                wi.layer = op.layer;
                wi.k_star = op.key_vec;
                wi.v_star = op.val_vec;
                wi.c_inv = op.cov_inv;
                wi.target_norm = op.target_norm;
                insts.push_back(std::move(wi));
            } else if (op.kind == "field_spectral") {
                WISAInstruction wi;
                wi.kind = WISAKind::FIELD_SPECTRAL_OP;
                wi.layer = op.layer;
                wi.layer_range = {op.layer, std::max(op.layer + 1, op.layer_end)};
                for (const auto& m : op.field_modes) {
                    wi.u_k.push_back(m.u);
                    wi.v_k.push_back(m.v);
                    wi.coeff_delta.push_back(m.sigma);
                }
                insts.push_back(std::move(wi));
            }
        }
    }

    return insts;
}

std::vector<WISAInstruction> optimize_wisa(
    const std::vector<WISAInstruction>& insts,
    int opt_level) {
    if (opt_level <= 0) {
        return insts;
    }

    std::vector<WISAInstruction> filtered;
    filtered.reserve(insts.size());
    for (const auto& wi : insts) {
        if (wi.kind == WISAKind::VPD_RANK1) {
            if (std::abs(wi.scale) < 1e-12 || norm2(wi.u) < 1e-12 || norm2(wi.v) < 1e-12) {
                continue;
            }
            filtered.push_back(wi);
        } else if (wi.kind == WISAKind::THRESHOLDED_ROUTE) {
            if (wi.threshold <= 0.0 || std::abs(wi.scale) < 1e-12 || norm2(wi.u) < 1e-12) {
                continue;
            }
            filtered.push_back(wi);
        } else if (wi.kind == WISAKind::SWIGLU_POLE) {
            if (norm2(wi.w_down) < 1e-12) {
                continue;
            }
            filtered.push_back(wi);
        } else if (wi.kind == WISAKind::FIELD_SPECTRAL_OP) {
            WISAInstruction kept = wi;
            kept.u_k.clear();
            kept.v_k.clear();
            kept.coeff_delta.clear();
            for (std::size_t k = 0; k < wi.coeff_delta.size() && k < wi.u_k.size() && k < wi.v_k.size(); ++k) {
                if (std::abs(wi.coeff_delta[k]) >= 1e-12 &&
                    norm2(wi.u_k[k]) >= 1e-12 &&
                    norm2(wi.v_k[k]) >= 1e-12) {
                    kept.u_k.push_back(wi.u_k[k]);
                    kept.v_k.push_back(wi.v_k[k]);
                    kept.coeff_delta.push_back(wi.coeff_delta[k]);
                }
            }
            if (!kept.coeff_delta.empty()) {
                filtered.push_back(std::move(kept));
            }
        } else {
            filtered.push_back(wi);
        }
    }

    std::vector<WISAInstruction> out;
    out.reserve(filtered.size());

    std::size_t idx = 0;
    while (idx < filtered.size()) {
        if (filtered[idx].kind == WISAKind::VPD_RANK1) {
            std::size_t end = idx + 1;
            const int lay = filtered[idx].layer;
            const std::string& mid = filtered[idx].matrix_id;
            const std::size_t rows = filtered[idx].u.size();
            const std::size_t cols = filtered[idx].v.size();
            while (end < filtered.size() &&
                   filtered[end].kind == WISAKind::VPD_RANK1 &&
                   filtered[end].layer == lay &&
                   filtered[end].matrix_id == mid &&
                   filtered[end].u.size() == rows &&
                   filtered[end].v.size() == cols) {
                ++end;
            }
            const std::size_t run_len = end - idx;
            if (run_len >= 2 && rows > 0 && cols > 0 && rows <= 256 && cols <= 256) {
                Matrix delta = zeros(rows, cols);
                for (std::size_t p = idx; p < end; ++p) {
                    const auto& w = filtered[p];
                    for (std::size_t r = 0; r < rows; ++r) {
                        const double su = w.scale * w.u[r];
                        for (std::size_t c = 0; c < cols; ++c) {
                            delta(r, c) += su * w.v[c];
                        }
                    }
                }
                const std::size_t k_max = std::min({run_len, rows, cols});
                const std::vector<SVDMode> modes = svd_top_k(delta, k_max, 50, 0);
                std::vector<SVDMode> kept_modes;
                Matrix recon = zeros(rows, cols);
                for (const auto& m : modes) {
                    if (m.sigma > 1e-11) {
                        kept_modes.push_back(m);
                        for (std::size_t r = 0; r < rows; ++r) {
                            const double su = m.sigma * m.u[r];
                            for (std::size_t c = 0; c < cols; ++c) {
                                recon(r, c) += su * m.v[c];
                            }
                        }
                    }
                }
                const double err = frob_norm(mat_sub(delta, recon));
                if (err < 1e-11 && kept_modes.size() <= run_len) {
                    for (const auto& m : kept_modes) {
                        WISAInstruction cwi;
                        cwi.kind = WISAKind::VPD_RANK1;
                        cwi.layer = lay;
                        cwi.matrix_id = mid;
                        cwi.u = m.u;
                        cwi.v = m.v;
                        cwi.scale = m.sigma;
                        out.push_back(std::move(cwi));
                    }
                    idx = end;
                    continue;
                }
            }
            for (std::size_t p = idx; p < end; ++p) {
                out.push_back(filtered[p]);
            }
            idx = end;
            continue;
        }

        if (filtered[idx].kind == WISAKind::FIELD_SPECTRAL_OP) {
            std::size_t end = idx + 1;
            const auto lrange = filtered[idx].layer_range;
            const std::size_t d = filtered[idx].u_k.empty() ? 0u : filtered[idx].u_k[0].size();
            while (end < filtered.size() &&
                   filtered[end].kind == WISAKind::FIELD_SPECTRAL_OP &&
                   filtered[end].layer_range == lrange) {
                const std::size_t d2 = filtered[end].u_k.empty() ? 0u : filtered[end].u_k[0].size();
                if (d2 != d) {
                    break;
                }
                ++end;
            }
            std::size_t total_modes = 0;
            for (std::size_t p = idx; p < end; ++p) {
                total_modes += filtered[p].coeff_delta.size();
            }
            if (total_modes >= 2 && d > 0 && d <= 256) {
                Matrix delta = zeros(d, d);
                for (std::size_t p = idx; p < end; ++p) {
                    const auto& w = filtered[p];
                    for (std::size_t k = 0; k < w.coeff_delta.size() && k < w.u_k.size() && k < w.v_k.size(); ++k) {
                        if (w.u_k[k].size() == d && w.v_k[k].size() == d) {
                            const double s = w.coeff_delta[k];
                            for (std::size_t r = 0; r < d; ++r) {
                                const double su = s * w.u_k[k][r];
                                for (std::size_t c = 0; c < d; ++c) {
                                    delta(r, c) += su * w.v_k[k][c];
                                }
                            }
                        }
                    }
                }
                const std::size_t k_max = std::min(d, total_modes);
                const std::vector<SVDMode> modes = svd_top_k(delta, k_max, 50, 0);
                std::vector<SVDMode> kept_modes;
                Matrix recon = zeros(d, d);
                for (const auto& m : modes) {
                    if (m.sigma > 1e-11) {
                        kept_modes.push_back(m);
                        for (std::size_t r = 0; r < d; ++r) {
                            const double su = m.sigma * m.u[r];
                            for (std::size_t c = 0; c < d; ++c) {
                                recon(r, c) += su * m.v[c];
                            }
                        }
                    }
                }
                const double err = frob_norm(mat_sub(delta, recon));
                if (err < 1e-11 && kept_modes.size() <= total_modes) {
                    if (!kept_modes.empty()) {
                        WISAInstruction cwi;
                        cwi.kind = WISAKind::FIELD_SPECTRAL_OP;
                        cwi.layer = lrange.first;
                        cwi.layer_range = lrange;
                        for (const auto& m : kept_modes) {
                            cwi.u_k.push_back(m.u);
                            cwi.v_k.push_back(m.v);
                            cwi.coeff_delta.push_back(m.sigma);
                        }
                        out.push_back(std::move(cwi));
                    }
                    idx = end;
                    continue;
                }
            }
            for (std::size_t p = idx; p < end; ++p) {
                out.push_back(filtered[p]);
            }
            idx = end;
            continue;
        }

        out.push_back(filtered[idx]);
        ++idx;
    }

    if (opt_level >= 2) {
        std::map<int, bool> has_norm_rescale;
        std::map<int, double> net_pos_energy;
        std::map<int, std::size_t> layer_dim;

        for (const auto& wi : out) {
            if (wi.kind == WISAKind::NORM_RESCALE) {
                has_norm_rescale[wi.layer] = true;
            } else if (wi.kind == WISAKind::SWIGLU_POLE) {
                net_pos_energy[wi.layer] += norm2(wi.w_down) * std::max(1.0, norm2(wi.w_up));
                if (!wi.w_down.empty()) {
                    layer_dim[wi.layer] = wi.w_down.size();
                }
            } else if (wi.kind == WISAKind::VPD_RANK1) {
                net_pos_energy[wi.layer] += wi.scale * norm2(wi.u) * norm2(wi.v);
                if (!wi.u.empty()) {
                    layer_dim[wi.layer] = wi.u.size();
                }
            } else if (wi.kind == WISAKind::THRESHOLDED_ROUTE) {
                net_pos_energy[wi.layer] += std::abs(wi.scale) * norm2(wi.u);
                if (!wi.u.empty()) {
                    layer_dim[wi.layer] = wi.u.size();
                }
            }
        }

        for (const auto& [lay, net_e] : net_pos_energy) {
            if (net_e > 0.05 && !has_norm_rescale[lay] && layer_dim[lay] > 0) {
                const double g = std::max(-0.25, 1.0 / std::sqrt(1.0 + net_e) - 1.0);
                WISAInstruction nr;
                nr.kind = WISAKind::NORM_RESCALE;
                nr.layer = lay;
                nr.norm_id = "post_attention_layernorm";
                nr.gamma_scale.assign(layer_dim[lay], g);
                out.push_back(std::move(nr));
            }
        }
    }

    return out;
}

std::string format_wisa(const std::vector<WISAInstruction>& insts) {
    std::ostringstream oss;
    for (std::size_t i = 0; i < insts.size(); ++i) {
        const auto& wi = insts[i];
        oss << i << ": " << wisa_kind_name(wi.kind) << "(";
        switch (wi.kind) {
            case WISAKind::SWIGLU_POLE:
                oss << "layer=" << wi.layer
                    << ", slot=" << wi.slot
                    << ", polarity=" << wi.polarity
                    << ", append=" << (wi.append_slot ? "true" : "false")
                    << ", d_gate=" << wi.w_gate.size();
                break;
            case WISAKind::THRESHOLDED_ROUTE:
                oss << "layer=" << wi.layer
                    << ", slot=" << wi.slot
                    << ", polarity=" << wi.polarity
                    << ", threshold=" << wi.threshold
                    << ", scale=" << wi.scale;
                break;
            case WISAKind::VPD_RANK1:
                oss << "layer=" << wi.layer
                    << ", matrix=\"" << wi.matrix_id << "\""
                    << ", scale=" << wi.scale;
                break;
            case WISAKind::ATTN_BILINEAR:
                oss << "layer=" << wi.layer
                    << ", head_group=" << wi.head_group
                    << ", rope_mode=\"" << wi.rope_mode << "\"";
                break;
            case WISAKind::ASSOC_MEM_ROW:
                oss << "layer=" << wi.layer
                    << ", dim=" << wi.k_star.size()
                    << ", target_norm=" << wi.target_norm;
                break;
            case WISAKind::FIELD_SPECTRAL_OP:
                oss << "layers=[" << wi.layer_range.first << ", " << wi.layer_range.second << ")"
                    << ", modes=" << wi.coeff_delta.size();
                break;
            case WISAKind::NORM_RESCALE:
                oss << "layer=" << wi.layer
                    << ", norm_id=\"" << wi.norm_id << "\""
                    << ", dim=" << wi.gamma_scale.size();
                break;
        }
        oss << ")\n";
    }
    return oss.str();
}

}
