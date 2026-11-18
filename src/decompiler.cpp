#include "aic/decompiler.hpp"

#include "aic/codegen.hpp"
#include "aic/errors.hpp"
#include "aic/json_min.hpp"
#include "aic/linker.hpp"
#include "aic/mir.hpp"
#include "aic/safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <vector>

namespace aic {

namespace {

std::string sanitize_ident(const std::string& name) {
    std::string out;
    for (unsigned char c : name) {
        if (std::isalnum(c) || c == '_') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) {
        return "LiftedModel";
    }
    if (!std::isalpha(static_cast<unsigned char>(out[0])) && out[0] != '_') {
        out = "M_" + out;
    }
    return out;
}

int parse_layer_index(const std::string& key) {
    const std::string prefix = "model.layers.";
    if (key.rfind(prefix, 0) != 0) {
        return -1;
    }
    std::size_t pos = prefix.size();
    std::size_t end = key.find('.', pos);
    if (end == std::string::npos || end == pos) {
        return -1;
    }
    try {
        return std::stoi(key.substr(pos, end - pos));
    } catch (...) {
        return -1;
    }
}

std::string read_file_text(const std::filesystem::path& path) {
    std::ifstream ifs(path.string());
    if (!ifs) {
        throw IOError("Failed to open file for reading: " + path.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string lower_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

void apply_rope_local(Vector& head_vec, std::size_t pos, double rope_theta) {
    const std::size_t hd = head_vec.size();
    const std::size_t half = hd / 2;
    const double d_hd = static_cast<double>(hd);
    const double d_pos = static_cast<double>(pos);
    for (std::size_t i = 0; i < half; ++i) {
        const double exponent = (2.0 * static_cast<double>(i)) / d_hd;
        const double inv_freq = 1.0 / std::pow(rope_theta, exponent);
        const double angle = d_pos * inv_freq;
        const double cos_a = std::cos(angle);
        const double sin_a = std::sin(angle);
        const double x0 = head_vec[i];
        const double x1 = head_vec[i + half];
        head_vec[i] = x0 * cos_a - x1 * sin_a;
        head_vec[i + half] = x1 * cos_a + x0 * sin_a;
    }
}

bool has_full_layer_weights(const TransformerModel& m, int l) {
    const std::string prefix = "model.layers." + std::to_string(l);
    return m.vectors.contains(prefix + ".input_layernorm.weight") &&
        m.vectors.contains(prefix + ".post_attention_layernorm.weight") &&
        m.matrices.contains(prefix + ".self_attn.q_proj.weight") &&
        m.matrices.contains(prefix + ".self_attn.o_proj.weight") &&
        m.matrices.contains(prefix + ".mlp.gate_proj.weight") &&
        m.matrices.contains(prefix + ".mlp.up_proj.weight") &&
        m.matrices.contains(prefix + ".mlp.down_proj.weight");
}

struct PartialTraceData {
    std::vector<std::vector<Vector>> res_inputs;
    std::vector<std::vector<Vector>> res_outputs;
    std::vector<std::vector<Vector>> res_deltas;
    std::vector<std::vector<Vector>> single_outputs;
    std::vector<std::vector<Vector>> single_deltas;
    std::vector<std::vector<double>> pole_pos;
    std::vector<std::vector<double>> pole_neg;
    std::vector<std::vector<double>> pole_max;
};

void trace_sequence_range(
    const TransformerModel& model,
    const std::vector<int>& seq,
    int trace_start,
    int trace_end,
    int fl0,
    int fl1,
    bool is_single_probe,
    PartialTraceData& out) {
    const std::size_t d = static_cast<std::size_t>(std::max(0, model.config.hidden_size));
    const std::size_t d_ff = static_cast<std::size_t>(std::max(0, model.config.intermediate_size));
    if (d == 0 || seq.empty() || trace_start >= trace_end) {
        return;
    }

    const std::size_t seq_len = seq.size();
    std::vector<Vector> h(seq_len, Vector(d, 0.0));
    const auto emb_it = model.matrices.find("model.embed_tokens.weight");
    if (emb_it != model.matrices.end() && emb_it->second.rows > 0 && emb_it->second.cols == d) {
        const std::size_t vocab_n = emb_it->second.rows;
        for (std::size_t t = 0; t < seq_len; ++t) {
            const std::size_t tid = static_cast<std::size_t>(seq[t]) % vocab_n;
            h[t] = emb_it->second.row(tid);
        }
    } else {
        for (std::size_t t = 0; t < seq_len; ++t) {
            h[t][static_cast<std::size_t>(seq[t]) % d] = 1.0;
        }
    }

    const std::size_t num_q_heads = static_cast<std::size_t>(std::max(1, model.config.num_attention_heads));
    const std::size_t hd = (model.config.head_dim > 0)
        ? static_cast<std::size_t>(model.config.head_dim)
        : (d / num_q_heads);
    const double inv_sqrt_hd = 1.0 / std::sqrt(static_cast<double>(std::max<std::size_t>(1, hd)));
    const double eps = model.config.rms_norm_eps;
    const std::string attn_mode = lower_copy(model.config.attention_type);

    for (int l = trace_start; l < trace_end; ++l) {
        if (!has_full_layer_weights(model, l)) {
            continue;
        }
        const std::string prefix = "model.layers." + std::to_string(l);
        const Vector& in_ln = model.vectors.at(prefix + ".input_layernorm.weight");

        std::vector<Vector> x_norm(seq_len);
        for (std::size_t t = 0; t < seq_len; ++t) {
            x_norm[t] = rms_norm(h[t], in_ln, eps);
        }

        const Matrix& w_q = model.matrices.at(prefix + ".self_attn.q_proj.weight");
        const Matrix& w_o = model.matrices.at(prefix + ".self_attn.o_proj.weight");
        const auto qb_it = model.vectors.find(prefix + ".self_attn.q_proj.bias");

        std::vector<Vector> q_all(seq_len);
        for (std::size_t t = 0; t < seq_len; ++t) {
            q_all[t] = mat_vec(w_q, x_norm[t]);
            if (qb_it != model.vectors.end() && qb_it->second.size() == q_all[t].size()) {
                q_all[t] = vec_add(q_all[t], qb_it->second);
            }
        }

        std::vector<Vector> k_all(seq_len);
        std::vector<Vector> v_all(seq_len);
        const std::string kva_key = prefix + ".self_attn.kv_a_proj.weight";
        const std::string kvb_key = prefix + ".self_attn.kv_b_proj.weight";
        if (attn_mode == "mla" && model.matrices.contains(kva_key) && model.matrices.contains(kvb_key)) {
            const Matrix& w_kva = model.matrices.at(kva_key);
            const Matrix& w_kvb = model.matrices.at(kvb_key);
            for (std::size_t t = 0; t < seq_len; ++t) {
                const Vector c_kv = mat_vec(w_kva, x_norm[t]);
                const Vector kv = mat_vec(w_kvb, c_kv);
                const std::size_t half = kv.size() / 2;
                k_all[t].assign(kv.begin(), kv.begin() + static_cast<std::ptrdiff_t>(half));
                v_all[t].assign(kv.begin() + static_cast<std::ptrdiff_t>(half), kv.end());
            }
        } else {
            const Matrix& w_k = model.matrices.at(prefix + ".self_attn.k_proj.weight");
            const Matrix& w_v = model.matrices.at(prefix + ".self_attn.v_proj.weight");
            const auto kb_it = model.vectors.find(prefix + ".self_attn.k_proj.bias");
            const auto vb_it = model.vectors.find(prefix + ".self_attn.v_proj.bias");
            for (std::size_t t = 0; t < seq_len; ++t) {
                k_all[t] = mat_vec(w_k, x_norm[t]);
                v_all[t] = mat_vec(w_v, x_norm[t]);
                if (kb_it != model.vectors.end() && kb_it->second.size() == k_all[t].size()) {
                    k_all[t] = vec_add(k_all[t], kb_it->second);
                }
                if (vb_it != model.vectors.end() && vb_it->second.size() == v_all[t].size()) {
                    v_all[t] = vec_add(v_all[t], vb_it->second);
                }
            }
        }

        const std::size_t kv_heads_eff = std::max<std::size_t>(1, k_all[0].size() / hd);
        const std::size_t group_size = std::max<std::size_t>(1, num_q_heads / kv_heads_eff);

        const auto qn_it = model.vectors.find(prefix + ".self_attn.q_norm.weight");
        const auto kn_it = model.vectors.find(prefix + ".self_attn.k_norm.weight");
        if (qn_it != model.vectors.end() || kn_it != model.vectors.end() || model.config.use_rope) {
            for (std::size_t t = 0; t < seq_len; ++t) {
                for (std::size_t hq = 0; hq < num_q_heads; ++hq) {
                    const std::size_t q_start = hq * hd;
                    Vector q_head(
                        q_all[t].begin() + static_cast<std::ptrdiff_t>(q_start),
                        q_all[t].begin() + static_cast<std::ptrdiff_t>(q_start + hd));
                    if (qn_it != model.vectors.end() && qn_it->second.size() == hd) {
                        q_head = rms_norm(q_head, qn_it->second, eps);
                    }
                    if (model.config.use_rope) {
                        apply_rope_local(q_head, t, model.config.rope_theta);
                    }
                    std::copy(q_head.begin(), q_head.end(), q_all[t].begin() + static_cast<std::ptrdiff_t>(q_start));
                }
                for (std::size_t hkv = 0; hkv < kv_heads_eff; ++hkv) {
                    const std::size_t kv_start = hkv * hd;
                    Vector k_head(
                        k_all[t].begin() + static_cast<std::ptrdiff_t>(kv_start),
                        k_all[t].begin() + static_cast<std::ptrdiff_t>(kv_start + hd));
                    if (kn_it != model.vectors.end() && kn_it->second.size() == hd) {
                        k_head = rms_norm(k_head, kn_it->second, eps);
                    }
                    if (model.config.use_rope) {
                        apply_rope_local(k_head, t, model.config.rope_theta);
                    }
                    std::copy(k_head.begin(), k_head.end(), k_all[t].begin() + static_cast<std::ptrdiff_t>(kv_start));
                }
            }
        }

        std::vector<Vector> h_mid(seq_len);
        for (std::size_t t = 0; t < seq_len; ++t) {
            Vector concat_ctx(num_q_heads * hd, 0.0);
            for (std::size_t hq = 0; hq < num_q_heads; ++hq) {
                const std::size_t hkv = hq / group_size;
                const std::size_t q_start = hq * hd;
                const std::size_t kv_start = hkv * hd;

                Vector scores(t + 1, 0.0);
                for (std::size_t j = 0; j <= t; ++j) {
                    double acc = 0.0;
                    for (std::size_t m = 0; m < hd; ++m) {
                        acc += q_all[t][q_start + m] * k_all[j][kv_start + m];
                    }
                    scores[j] = acc * inv_sqrt_hd;
                }
                const Vector probs = softmax(scores);
                for (std::size_t j = 0; j <= t; ++j) {
                    const double p = probs[j];
                    for (std::size_t m = 0; m < hd; ++m) {
                        concat_ctx[q_start + m] += p * v_all[j][kv_start + m];
                    }
                }
            }
            const Vector attn_out = mat_vec(w_o, concat_ctx);
            h_mid[t] = vec_add(h[t], attn_out);
        }

        const Vector& post_ln = model.vectors.at(prefix + ".post_attention_layernorm.weight");
        const Matrix& w_gate = model.matrices.at(prefix + ".mlp.gate_proj.weight");
        const Matrix& w_up = model.matrices.at(prefix + ".mlp.up_proj.weight");
        const Matrix& w_down = model.matrices.at(prefix + ".mlp.down_proj.weight");
        const std::size_t d_ff_actual = w_gate.rows;

        std::vector<Vector> h_next(seq_len);
        const std::size_t ul = static_cast<std::size_t>(l);
        const bool collect_l = (l >= fl0 && l < fl1);

        for (std::size_t t = 0; t < seq_len; ++t) {
            const Vector x_post = rms_norm(h_mid[t], post_ln, eps);
            const Vector g_pre = mat_vec(w_gate, x_post);
            const Vector u_val = mat_vec(w_up, x_post);
            Vector acts(d_ff_actual);
            for (std::size_t i = 0; i < d_ff_actual; ++i) {
                acts[i] = silu(g_pre[i]) * u_val[i];
                if (collect_l && i < d_ff) {
                    const double abs_a = std::abs(acts[i]);
                    out.pole_max[ul][i] = std::max(out.pole_max[ul][i], abs_a);
                    if (u_val[i] > 0.0 && g_pre[i] > 0.0) {
                        out.pole_pos[ul][i] = std::max(out.pole_pos[ul][i], abs_a);
                    } else if (u_val[i] < 0.0 && g_pre[i] > 0.0) {
                        out.pole_neg[ul][i] = std::max(out.pole_neg[ul][i], abs_a);
                    }
                }
            }
            const Vector mlp_out = mat_vec(w_down, acts);
            h_next[t] = vec_add(h_mid[t], mlp_out);

            if (collect_l) {
                if (is_single_probe) {
                    out.single_outputs[ul].push_back(h_next[t]);
                    out.single_deltas[ul].push_back(vec_sub(h_next[t], h[t]));
                } else {
                    out.res_inputs[ul].push_back(h[t]);
                    out.res_outputs[ul].push_back(h_next[t]);
                    out.res_deltas[ul].push_back(vec_sub(h_next[t], h[t]));
                }
            }
        }
        h = std::move(h_next);
    }
}

PartialTraceData collect_partial_traces(const TransformerModel& model, int fl0, int fl1) {
    const int n_layers = std::max(0, model.config.num_layers);
    const std::size_t d = static_cast<std::size_t>(std::max(0, model.config.hidden_size));
    const std::size_t d_ff = static_cast<std::size_t>(std::max(0, model.config.intermediate_size));
    const int vocab = std::max(1, model.config.vocab_size);

    PartialTraceData out;
    out.res_inputs.resize(static_cast<std::size_t>(n_layers));
    out.res_outputs.resize(static_cast<std::size_t>(n_layers));
    out.res_deltas.resize(static_cast<std::size_t>(n_layers));
    out.single_outputs.resize(static_cast<std::size_t>(n_layers));
    out.single_deltas.resize(static_cast<std::size_t>(n_layers));
    out.pole_pos.assign(static_cast<std::size_t>(n_layers), std::vector<double>(d_ff, 0.0));
    out.pole_neg.assign(static_cast<std::size_t>(n_layers), std::vector<double>(d_ff, 0.0));
    out.pole_max.assign(static_cast<std::size_t>(n_layers), std::vector<double>(d_ff, 0.0));

    if (n_layers <= 0 || d == 0 || fl0 >= fl1) {
        return out;
    }

    int trace_start = fl0;
    bool all_prefix_present = true;
    for (int l = 0; l < fl0; ++l) {
        if (!has_full_layer_weights(model, l)) {
            all_prefix_present = false;
            break;
        }
    }
    if (all_prefix_present) {
        trace_start = 0;
    }

    const std::vector<std::vector<int>> cal_seqs = {
        {1 % vocab, 3 % vocab, 7 % vocab, 11 % vocab, 13 % vocab, 17 % vocab},
        {2 % vocab, 5 % vocab, 9 % vocab, 14 % vocab, 19 % vocab, 23 % vocab},
        {4 % vocab, 4 % vocab, 4 % vocab, 8 % vocab, 12 % vocab, 16 % vocab},
        {6 % vocab, 10 % vocab, 15 % vocab, 21 % vocab, 28 % vocab, 35 % vocab}
    };
    for (const auto& seq : cal_seqs) {
        trace_sequence_range(model, seq, trace_start, fl1, fl0, fl1, false, out);
    }

    const int single_count = std::min(16, vocab);
    for (int tid = 0; tid < single_count; ++tid) {
        trace_sequence_range(model, {tid}, trace_start, fl1, fl0, fl1, true, out);
    }

    return out;
}

double vec_concentration(const Vector& v) {
    const double n2 = norm2(v);
    if (n2 <= 1e-12 || v.empty()) {
        return 1.0;
    }
    double max_abs = 0.0;
    for (double x : v) {
        max_abs = std::max(max_abs, std::abs(x));
    }
    return (max_abs * std::sqrt(static_cast<double>(v.size()))) / n2;
}

double activation_kurtosis_score(const Vector& dir, const std::vector<Vector>& samples) {
    if (samples.empty() || dir.empty() || samples[0].size() != dir.size()) {
        return 1.0;
    }
    double max_sq = 0.0;
    double sum_sq = 0.0;
    for (const Vector& x : samples) {
        const double d = dot(dir, x);
        const double sq = d * d;
        max_sq = std::max(max_sq, sq);
        sum_sq += sq;
    }
    const double mean_sq = sum_sq / static_cast<double>(samples.size());
    return max_sq / (mean_sq + 1e-12);
}

void accumulate_mode(Matrix& dst, const SVDMode& m, double scale) {
    if (dst.rows != m.u.size() || dst.cols != m.v.size()) {
        return;
    }
    const double w = m.sigma * scale;
    for (std::size_t r = 0; r < dst.rows; ++r) {
        const double wu = w * m.u[r];
        for (std::size_t c = 0; c < dst.cols; ++c) {
            dst(r, c) += wu * m.v[c];
        }
    }
}

Vector subtract_basis_local(Vector v, const std::vector<Vector>& basis) {
    const std::size_t d = v.size();
    for (int pass = 0; pass < 2; ++pass) {
        for (const Vector& b : basis) {
            if (b.size() != d) {
                continue;
            }
            const double p = dot(v, b);
            for (std::size_t i = 0; i < d; ++i) {
                v[i] -= p * b[i];
            }
        }
    }
    return v;
}

std::vector<Vector> orthonormal_in_comp(
    const std::vector<Vector>& cands,
    std::size_t count,
    std::size_t d,
    const std::vector<Vector>& forbidden) {
    if (d == 0 || count == 0) {
        return {};
    }
    std::vector<Vector> basis = gram_schmidt(forbidden, 1e-10);
    std::vector<Vector> out;
    out.reserve(count);
    for (const Vector& raw : cands) {
        if (out.size() >= count) {
            break;
        }
        if (raw.size() != d) {
            continue;
        }
        Vector w = subtract_basis_local(raw, basis);
        if (norm2(w) > 1e-7) {
            w = normalize(w);
            out.push_back(w);
            basis.push_back(w);
        }
    }
    for (std::size_t i = 0; i < d && out.size() < count; ++i) {
        Vector e(d, 0.0);
        e[i] = 1.0;
        Vector w = subtract_basis_local(e, basis);
        if (norm2(w) > 1e-7) {
            w = normalize(w);
            out.push_back(w);
            basis.push_back(w);
        }
    }
    return out;
}

struct CenteredSamples {
    std::vector<Vector> pts;
    Vector mu;
    Matrix centered;
};

CenteredSamples dedup_and_center_samples(
    const std::vector<Vector>& raw_samples,
    std::size_t d,
    std::size_t min_pts,
    const std::vector<Vector>& project_basis) {
    CenteredSamples cs;
    cs.pts.reserve(raw_samples.size());
    for (const Vector& v : raw_samples) {
        if (v.size() != d) {
            continue;
        }
        bool dup = false;
        for (const Vector& prev : cs.pts) {
            if (norm2(vec_sub(v, prev)) < 1e-6) {
                dup = true;
                break;
            }
        }
        if (!dup) {
            cs.pts.push_back(v);
        }
    }
    if (cs.pts.size() < min_pts) {
        cs.pts.clear();
        return cs;
    }

    cs.mu.assign(d, 0.0);
    for (const Vector& v : cs.pts) {
        for (std::size_t i = 0; i < d; ++i) {
            cs.mu[i] += v[i];
        }
    }
    const double inv_n = 1.0 / static_cast<double>(cs.pts.size());
    for (double& x : cs.mu) {
        x *= inv_n;
    }

    cs.centered = Matrix(cs.pts.size(), d, 0.0);
    for (std::size_t i = 0; i < cs.pts.size(); ++i) {
        Vector c = vec_sub(cs.pts[i], cs.mu);
        if (!project_basis.empty()) {
            c = subtract_basis_local(std::move(c), project_basis);
        }
        cs.centered.set_row(i, c);
    }
    return cs;
}

struct PlaneRadiusStats {
    double mean_r = 0.0;
    double cv = 1e9;
    int quad_count = 0;
    int oct_count = 0;
};

PlaneRadiusStats compute_plane_radius_stats(
    const Matrix& centered,
    const Vector& va,
    const Vector& vb) {
    PlaneRadiusStats stats;
    const std::size_t n = centered.rows;
    if (n == 0) {
        return stats;
    }
    std::vector<double> radii;
    radii.reserve(n);
    int quad_mask = 0;
    int oct_mask = 0;
    double sum_r = 0.0;
    constexpr double kTwoPi = 6.28318530717958647692;
    for (std::size_t i = 0; i < n; ++i) {
        const Vector c = centered.row(i);
        const double x = dot(c, va);
        const double y = dot(c, vb);
        const double r = std::hypot(x, y);
        radii.push_back(r);
        sum_r += r;
        if (r > 1e-6) {
            const int q = (x >= 0.0 ? 0 : 1) + (y >= 0.0 ? 0 : 2);
            quad_mask |= (1 << q);
            double ang = std::atan2(y, x);
            if (ang < 0.0) {
                ang += kTwoPi;
            }
            const int oct = std::clamp(static_cast<int>((ang / kTwoPi) * 8.0), 0, 7);
            oct_mask |= (1 << oct);
        }
    }

    stats.mean_r = sum_r / static_cast<double>(radii.size());
    if (stats.mean_r > 1e-12) {
        double var_r = 0.0;
        for (double r : radii) {
            const double dr = r - stats.mean_r;
            var_r += dr * dr;
        }
        stats.cv = std::sqrt(var_r / static_cast<double>(radii.size())) / stats.mean_r;
    }
    for (int b = 0; b < 4; ++b) {
        if ((quad_mask & (1 << b)) != 0) {
            ++stats.quad_count;
        }
    }
    for (int b = 0; b < 8; ++b) {
        if ((oct_mask & (1 << b)) != 0) {
            ++stats.oct_count;
        }
    }
    return stats;
}

struct RingDetection {
    bool found = false;
    Vector e0;
    Vector e1;
    Vector e2;
};

RingDetection detect_s1_ring_in_samples(
    const std::vector<Vector>& raw_samples,
    std::size_t d,
    const std::vector<Vector>& prot_gs) {
    RingDetection res;
    if (d < 3 || raw_samples.size() < 4) {
        return res;
    }

    const CenteredSamples cs = dedup_and_center_samples(raw_samples, d, 4, {});
    if (cs.pts.size() < 4) {
        return res;
    }

    const std::vector<SVDMode> modes = svd_top_k(cs.centered, 2, 40, 42);
    if (modes.size() < 2 || modes[0].sigma <= 1e-12 || modes[1].sigma < 0.45 * modes[0].sigma) {
        return res;
    }
    const double total_fn = frob_norm(cs.centered);
    const double total_energy = total_fn * total_fn;
    const double plane_energy = modes[0].sigma * modes[0].sigma + modes[1].sigma * modes[1].sigma;
    if (total_energy <= 1e-12 || (plane_energy / total_energy) < 0.70) {
        return res;
    }
    const Vector& v1 = modes[0].v;
    const Vector& v2 = modes[1].v;

    const PlaneRadiusStats rstats = compute_plane_radius_stats(cs.centered, v1, v2);
    if (rstats.mean_r <= 1e-4) {
        return res;
    }
    const bool angular_ok = (cs.pts.size() >= 8)
        ? (rstats.quad_count >= 3 && rstats.oct_count >= 5)
        : (rstats.quad_count >= 3);
    if (rstats.cv >= 0.20 || !angular_ok) {
        return res;
    }

    std::vector<Vector> plane_basis = orthonormal_in_comp({cs.mu, v1, v2}, 3, d, prot_gs);
    if (plane_basis.size() < 3) {
        return res;
    }
    res.found = true;
    res.e0 = plane_basis[0];
    res.e1 = plane_basis[1];
    res.e2 = plane_basis[2];
    return res;
}

struct TorusDetection {
    bool found = false;
    std::vector<Vector> basis;
};

TorusDetection detect_t2_torus_in_samples(
    const std::vector<Vector>& raw_samples,
    std::size_t d,
    const std::vector<Vector>& prot_gs) {
    TorusDetection res;
    if (d < 5 || raw_samples.size() < 16) {
        return res;
    }

    const CenteredSamples cs = dedup_and_center_samples(raw_samples, d, 16, prot_gs);
    if (cs.pts.size() < 16) {
        return res;
    }

    const std::vector<SVDMode> modes = svd_top_k(cs.centered, 4, 60, 42);
    if (modes.size() < 4) {
        return res;
    }
    const double s1 = modes[0].sigma;
    const double s2 = modes[1].sigma;
    const double s3 = modes[2].sigma;
    const double s4 = modes[3].sigma;
    if (s1 <= 1e-3 || (s2 / s1) <= 0.60 || (s3 / s2) <= 0.45 || (s4 / s3) <= 0.60) {
        return res;
    }

    const PlaneRadiusStats p12 = compute_plane_radius_stats(cs.centered, modes[0].v, modes[1].v);
    const PlaneRadiusStats p34 = compute_plane_radius_stats(cs.centered, modes[2].v, modes[3].v);
    if (p12.mean_r <= 1e-3 || p12.cv >= 0.28 || p34.mean_r <= 1e-3 || p34.cv >= 0.28) {
        return res;
    }

    const std::vector<Vector> v_orth = orthonormal_in_comp(
        {modes[0].v, modes[1].v, modes[2].v, modes[3].v}, 4, d, prot_gs);
    if (v_orth.size() < 4) {
        return res;
    }
    std::vector<Vector> forbidden_e0 = prot_gs;
    forbidden_e0.insert(forbidden_e0.end(), v_orth.begin(), v_orth.end());
    const std::vector<Vector> e0_vec = orthonormal_in_comp({cs.mu}, 1, d, forbidden_e0);
    if (e0_vec.empty()) {
        return res;
    }

    res.found = true;
    res.basis = {e0_vec[0], v_orth[0], v_orth[1], v_orth[2], v_orth[3]};
    return res;
}

bool op_mutates_model(const MIROp& op) {
    if (op.op == "excite" || op.op == "annihilate" || op.op == "transport_map" ||
        op.op == "frame_transport" || op.op == "attend" || op.op == "rodata_insert" ||
        op.op == "rodata_prune" || op.op == "rodata_rewrite") {
        return true;
    }
    if (op.op == "when") {
        if (op.body.empty() && op.else_body.empty()) {
            return true;
        }
        for (const auto& sub : op.body) {
            if (op_mutates_model(sub)) {
                return true;
            }
        }
        for (const auto& sub : op.else_body) {
            if (op_mutates_model(sub)) {
                return true;
            }
        }
    }
    return false;
}

bool mir_has_mutations(const MIRModule& mir) {
    for (const auto& flow : mir.flows) {
        for (const auto& op : flow.ops) {
            if (op_mutates_model(op)) {
                return true;
            }
        }
    }
    return false;
}

std::pair<NeuralELF, std::string> lift_loaded_model(
    const TransformerModel& model,
    const std::string& module_name,
    std::pair<int, int> layer_filter) {
    NeuralELF nelf;
    nelf.config = model.config;

    const int n_layers = std::max(0, model.config.num_layers);
    const bool filter_active = (layer_filter.first >= 0 && layer_filter.second > layer_filter.first);
    const int fl0 = filter_active ? std::clamp(layer_filter.first, 0, n_layers) : 0;
    const int fl1 = filter_active ? std::clamp(layer_filter.second, fl0, n_layers) : n_layers;
    const std::size_t d = static_cast<std::size_t>(std::max(0, model.config.hidden_size));
    const std::size_t d_ff = static_cast<std::size_t>(std::max(0, model.config.intermediate_size));

    const PartialTraceData traces = collect_partial_traces(model, fl0, fl1);

    constexpr double kTargetResidualRatio = 0.03;
    constexpr double kTargetRodataShare = 0.15;
    constexpr double kTargetFieldShare = 0.15;

    std::vector<Vector> residual_dir_cands;
    std::vector<Vector> text_dir_cands;
    std::vector<Vector> rodata_dir_cands;
    std::vector<SVDMode> lifted_field_modes;
    double top_rodata_curvature = 18.5;

    for (const auto& [key, w] : model.matrices) {
        const int layer_idx = parse_layer_index(key);
        if (layer_idx < 0 || (filter_active && (layer_idx < fl0 || layer_idx >= fl1))) {
            nelf.passthrough_matrices[key] = w;
            continue;
        }

        const std::size_t rows = w.rows;
        const std::size_t cols = w.cols;
        Matrix w_text = zeros(rows, cols);
        Matrix w_rodata = zeros(rows, cols);
        Matrix w_field = zeros(rows, cols);

        const double total_fn = frob_norm(w);
        const double total_e = total_fn * total_fn;
        const std::size_t min_dim = std::min(rows, cols);

        if (total_e > 0.0 && min_dim > 0) {
            const std::vector<SVDMode> modes = svd_top_k(w, min_dim, 50, 42);
            const std::size_t m_cnt = modes.size();
            if (m_cnt > 0) {
                std::vector<double> energies(m_cnt, 0.0);
                double sum_modes_e = 0.0;
                for (std::size_t i = 0; i < m_cnt; ++i) {
                    energies[i] = modes[i].sigma * modes[i].sigma;
                    sum_modes_e += energies[i];
                }

                const double target_res_e = kTargetResidualRatio * total_e;
                std::size_t keep_full = 0;
                double boundary_alpha = 0.0;
                double tail_e = sum_modes_e;

                for (std::size_t i = 0; i < m_cnt; ++i) {
                    const double next_tail = std::max(0.0, tail_e - energies[i]);
                    if (next_tail >= target_res_e) {
                        tail_e = next_tail;
                        ++keep_full;
                    } else {
                        const double rem_needed = std::max(0.0, target_res_e - next_tail);
                        const double ratio = std::clamp(rem_needed / std::max(1e-24, energies[i]), 0.0, 1.0);
                        boundary_alpha = std::clamp(1.0 - std::sqrt(ratio), 0.0, 1.0);
                        break;
                    }
                }

                struct ModeEntry {
                    std::size_t idx = 0;
                    double scale = 1.0;
                    double energy = 0.0;
                    double curv = 1.0;
                };
                std::vector<ModeEntry> kept;
                for (std::size_t i = 0; i < keep_full; ++i) {
                    const SVDMode& m = modes[i];
                    double curv = vec_concentration(m.u) * vec_concentration(m.v);
                    if (static_cast<std::size_t>(layer_idx) < traces.res_inputs.size()) {
                        const auto& samps = traces.res_inputs[static_cast<std::size_t>(layer_idx)];
                        if (m.v.size() == d) {
                            curv *= activation_kurtosis_score(m.v, samps);
                        } else if (m.u.size() == d) {
                            curv *= activation_kurtosis_score(m.u, samps);
                        }
                    }
                    kept.push_back(ModeEntry{i, 1.0, energies[i], curv});
                }
                if (keep_full < m_cnt && boundary_alpha > 1e-6) {
                    const SVDMode& m = modes[keep_full];
                    const double curv = vec_concentration(m.u) * vec_concentration(m.v);
                    kept.push_back(ModeEntry{keep_full, boundary_alpha, energies[keep_full] * boundary_alpha * boundary_alpha, curv});
                }

                if (kept.size() >= 3) {
                    std::sort(kept.begin(), kept.end(), [](const ModeEntry& a, const ModeEntry& b) {
                        return a.curv > b.curv;
                    });

                    const double ro_cap = kTargetRodataShare * total_e;
                    const double fl_cap = kTargetFieldShare * total_e;
                    double ro_e = 0.0;
                    double fl_e = 0.0;

                    std::size_t ro_end = 0;
                    while (ro_end < kept.size() - 2 && ro_e + kept[ro_end].energy <= ro_cap * 1.15) {
                        ro_e += kept[ro_end].energy;
                        ++ro_end;
                    }
                    if (ro_end == 0) {
                        ro_end = 1;
                    }

                    std::size_t fl_begin = kept.size();
                    while (fl_begin > ro_end + 1 && fl_e + kept[fl_begin - 1].energy <= fl_cap * 1.15) {
                        --fl_begin;
                        fl_e += kept[fl_begin].energy;
                    }
                    if (fl_begin == kept.size()) {
                        fl_begin = kept.size() - 1;
                    }

                    for (std::size_t p = 0; p < ro_end; ++p) {
                        accumulate_mode(w_rodata, modes[kept[p].idx], kept[p].scale);
                        if (modes[kept[p].idx].u.size() == d) {
                            rodata_dir_cands.push_back(modes[kept[p].idx].u);
                        }
                        top_rodata_curvature = std::max(top_rodata_curvature, 10.0 + kept[p].curv);
                    }
                    for (std::size_t p = ro_end; p < fl_begin; ++p) {
                        accumulate_mode(w_text, modes[kept[p].idx], kept[p].scale);
                        if (modes[kept[p].idx].u.size() == d) {
                            text_dir_cands.push_back(modes[kept[p].idx].u);
                        }
                    }
                    for (std::size_t p = fl_begin; p < kept.size(); ++p) {
                        accumulate_mode(w_field, modes[kept[p].idx], kept[p].scale);
                        if (modes[kept[p].idx].u.size() == d && modes[kept[p].idx].v.size() == d &&
                            lifted_field_modes.size() < 16) {
                            SVDMode fm = modes[kept[p].idx];
                            fm.sigma *= kept[p].scale;
                            lifted_field_modes.push_back(std::move(fm));
                        }
                    }
                } else {
                    for (const auto& ke : kept) {
                        accumulate_mode(w_text, modes[ke.idx], ke.scale * 0.70);
                        accumulate_mode(w_rodata, modes[ke.idx], ke.scale * 0.15);
                        accumulate_mode(w_field, modes[ke.idx], ke.scale * 0.15);
                    }
                }

                if (keep_full < m_cnt && modes[keep_full].u.size() == d) {
                    residual_dir_cands.push_back(modes[keep_full].u);
                } else if (m_cnt > 0 && modes.back().u.size() == d) {
                    residual_dir_cands.push_back(modes.back().u);
                }
            }
        }

        Matrix w_known = mat_add(w_text, mat_add(w_rodata, w_field));
        Matrix w_res = mat_sub(w, w_known);

        nelf.text_weights[key] = std::move(w_text);
        nelf.rodata_weights[key] = std::move(w_rodata);
        nelf.field_weights[key] = std::move(w_field);
        nelf.residual_weights[key] = std::move(w_res);
    }

    for (const auto& [key, vec] : model.vectors) {
        nelf.passthrough_vectors[key] = vec;
    }

    NeuralAtlas atlas = build_default_atlas(model, nullptr, 42);
    atlas.symbols.clear();

    if (n_layers > 0 && d_ff > 0) {
        atlas.dormant_poles.clear();
        for (int l = 0; l < n_layers; ++l) {
            const std::size_t ul = static_cast<std::size_t>(l);
            std::vector<std::pair<double, int>> ranked;
            ranked.reserve(d_ff);
            for (std::size_t i = 0; i < d_ff; ++i) {
                ranked.emplace_back(traces.pole_max[ul][i], static_cast<int>(i));
            }
            std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });
            const std::size_t d_cnt = std::max<std::size_t>(1, d_ff / 2);
            for (std::size_t i = 0; i < d_cnt && i < ranked.size(); ++i) {
                atlas.dormant_poles.emplace_back(l, ranked[i].second);
            }
        }
    }

    if (d > 0) {
        const std::size_t res_cap = std::max<std::size_t>(1, std::min<std::size_t>(3, d / 8));
        atlas.preserve_directions["Residual"] = orthonormal_in_comp(residual_dir_cands, res_cap, d, {});
        std::vector<Vector> used_pres = atlas.preserve_directions["Residual"];
        const std::size_t core_cap = std::max<std::size_t>(1, std::min<std::size_t>(2, d / 8));
        atlas.preserve_directions["CoreFluency"] = orthonormal_in_comp({}, core_cap, d, used_pres);
        used_pres.insert(
            used_pres.end(),
            atlas.preserve_directions["CoreFluency"].begin(),
            atlas.preserve_directions["CoreFluency"].end());
        atlas.preserve_directions["CoreReasoning"] = orthonormal_in_comp({}, core_cap, d, used_pres);
    }

    std::vector<Vector> all_protected;
    for (const auto& [pk, pvecs] : atlas.preserve_directions) {
        (void)pk;
        all_protected.insert(all_protected.end(), pvecs.begin(), pvecs.end());
    }
    const std::vector<Vector> prot_gs = gram_schmidt(all_protected, 1e-10);

    const std::size_t sub_k = std::max<std::size_t>(1, std::min<std::size_t>(4, d > 4 ? d / 8 : 1));
    const std::size_t field_r = std::max<std::size_t>(1, std::min<std::size_t>(8, d > 4 ? d / 4 : 1));

    const auto mid_range = model.config.stage_to_layers("Mid");
    const auto mid_late_range = model.config.stage_to_layers("Mid", "Late");
    const bool use_filtered_frame = filter_active && fl0 < fl1;
    const std::pair<int, int> micro_range = use_filtered_frame ? std::pair<int, int>{fl0, fl1} : mid_range;
    const std::pair<int, int> rodata_range = use_filtered_frame ? std::pair<int, int>{fl0, fl1} : mid_late_range;
    const std::pair<int, int> field_range = use_filtered_frame
        ? std::pair<int, int>{fl0, fl1}
        : model.config.stage_to_layers("All");
    const int best_layer = (n_layers > 0)
        ? std::clamp(micro_range.first, 0, n_layers - 1)
        : 0;

    int best_bipolar_slot = 0;
    if (n_layers > 0 && d_ff > 0) {
        double best_bipolar_score = -1.0;
        const std::size_t ul = static_cast<std::size_t>(best_layer);
        for (std::size_t i = 0; i < d_ff; ++i) {
            const double bscore = std::min(traces.pole_pos[ul][i], traces.pole_neg[ul][i]) +
                0.1 * traces.pole_max[ul][i];
            if (bscore > best_bipolar_score) {
                best_bipolar_score = bscore;
                best_bipolar_slot = static_cast<int>(i);
            }
        }
    }

    {
        SymbolBinding sb;
        sb.name = "LiftedDipole";
        sb.kind = "dipole";
        sb.layer_range = micro_range;
        Vector cand_dir;
        const std::string down_key = "model.layers." + std::to_string(best_layer) + ".mlp.down_proj.weight";
        if (nelf.text_weights.contains(down_key) &&
            static_cast<std::size_t>(best_bipolar_slot) < nelf.text_weights.at(down_key).cols) {
            cand_dir = nelf.text_weights.at(down_key).col(static_cast<std::size_t>(best_bipolar_slot));
        } else if (!text_dir_cands.empty()) {
            cand_dir = text_dir_cands[0];
        }
        auto basis = orthonormal_in_comp({cand_dir}, 1, d, prot_gs);
        if (!basis.empty()) {
            sb.direction = basis[0];
            sb.subspace_basis = basis;
        }
        sb.pole_slots["+pole"] = {best_layer, best_bipolar_slot, 1};
        sb.pole_slots["-pole"] = {best_layer, best_bipolar_slot, -1};
        atlas.symbols[sb.name] = std::move(sb);
    }

    {
        SymbolBinding sb;
        sb.name = "LiftedSubspace";
        sb.kind = "subspace";
        sb.layer_range = micro_range;
        sb.subspace_basis = orthonormal_in_comp(text_dir_cands, sub_k, d, prot_gs);
        if (!sb.subspace_basis.empty()) {
            sb.direction = sb.subspace_basis[0];
        }
        atlas.symbols[sb.name] = std::move(sb);
    }

    {
        SymbolBinding sb;
        sb.name = "LiftedRodata";
        sb.kind = "rodata";
        sb.layer_range = rodata_range;
        sb.preserve_keys = orthonormal_in_comp(rodata_dir_cands, 1, d, prot_gs);
        if (!sb.preserve_keys.empty()) {
            sb.direction = sb.preserve_keys[0];
        }
        atlas.symbols[sb.name] = std::move(sb);
    }

    {
        SymbolBinding sb;
        sb.name = "LiftedField";
        sb.kind = "field";
        sb.layer_range = field_range;
        if (lifted_field_modes.size() > field_r) {
            lifted_field_modes.resize(field_r);
        }
        if (lifted_field_modes.empty() && d > 0) {
            auto fbasis = orthonormal_in_comp({}, field_r, d, prot_gs);
            for (std::size_t i = 0; i < fbasis.size(); ++i) {
                lifted_field_modes.push_back(SVDMode{1.0 / static_cast<double>(i + 1), fbasis[i], fbasis[i]});
            }
        }
        sb.field_modes = std::move(lifted_field_modes);
        if (!sb.field_modes.empty()) {
            sb.direction = sb.field_modes[0].u;
        }
        atlas.symbols[sb.name] = std::move(sb);
    }

    struct LiftedRingSpec {
        int layer = 0;
        std::string name;
    };
    std::vector<LiftedRingSpec> detected_rings;
    std::vector<LiftedRingSpec> detected_tori;
    for (int l = fl0; l < fl1; ++l) {
        const std::size_t ul = static_cast<std::size_t>(l);
        RingDetection det = detect_s1_ring_in_samples(traces.single_deltas[ul], d, prot_gs);
        if (!det.found) {
            det = detect_s1_ring_in_samples(traces.single_outputs[ul], d, prot_gs);
        }
        if (!det.found) {
            det = detect_s1_ring_in_samples(traces.res_deltas[ul], d, prot_gs);
        }
        if (!det.found) {
            det = detect_s1_ring_in_samples(traces.res_outputs[ul], d, prot_gs);
        }
        if (det.found) {
            const std::string rname = "RingManifold_L" + std::to_string(l);
            SymbolBinding sb;
            sb.name = rname;
            sb.kind = "manifold";
            sb.layer_range = {l, l + 1};
            sb.manifold_plane = {det.e0, det.e1, det.e2};
            sb.subspace_basis = sb.manifold_plane;
            sb.direction = det.e1;
            atlas.symbols[rname] = std::move(sb);
            detected_rings.push_back(LiftedRingSpec{l, rname});
        } else if (d >= 5) {
            TorusDetection tdet = detect_t2_torus_in_samples(traces.single_deltas[ul], d, prot_gs);
            if (!tdet.found) {
                tdet = detect_t2_torus_in_samples(traces.single_outputs[ul], d, prot_gs);
            }
            if (!tdet.found) {
                tdet = detect_t2_torus_in_samples(traces.res_deltas[ul], d, prot_gs);
            }
            if (!tdet.found) {
                tdet = detect_t2_torus_in_samples(traces.res_outputs[ul], d, prot_gs);
            }
            if (tdet.found) {
                const std::string tname = "TorusManifold_L" + std::to_string(l);
                SymbolBinding sb;
                sb.name = tname;
                sb.kind = "manifold";
                sb.layer_range = {l, l + 1};
                sb.manifold_plane = tdet.basis;
                sb.subspace_basis = tdet.basis;
                sb.direction = tdet.basis[1];
                atlas.symbols[tname] = std::move(sb);
                detected_tori.push_back(LiftedRingSpec{l, tname});
            }
        }
    }

    nelf.atlas = std::move(atlas);

    const std::string filter_stage = use_filtered_frame
        ? (std::to_string(fl0) + ".." + std::to_string(fl1))
        : std::string();
    const std::string micro_frame_tag = use_filtered_frame ? filter_stage : std::string("Mid");
    const std::string rodata_frame_tag = use_filtered_frame ? filter_stage : std::string("Mid..Late");
    const std::string field_frame_tag = use_filtered_frame ? filter_stage : std::string("All");
    const std::string flow_frame_tag = use_filtered_frame ? filter_stage : std::string("All");

    const std::string mod_id = sanitize_ident(module_name);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "module " << mod_id << ";\n\n";
    oss << "frame LiftedEarly in Early {\n";
    oss << "    connect to LiftedMid via StandardResidualBridge;\n";
    oss << "}\n\n";
    oss << "frame LiftedMid in Mid {\n";
    oss << "    connect to LiftedLate via StandardResidualBridge;\n";
    oss << "}\n\n";
    oss << "frame LiftedLate in Late;\n\n";
    oss << "dipole LiftedDipole in Frame<" << micro_frame_tag << "> {\n";
    oss << "    +pole: PositivePole {\n";
    oss << "        anchor contrastive(\n";
    oss << "            pos: [\"lifted positive pole activation\"],\n";
    oss << "            neg: [\"lifted negative pole activation\"]\n";
    oss << "        );\n";
    oss << "    }\n";
    oss << "    -pole: NegativePole {\n";
    oss << "        anchor probe(\"lifted complementary pole state\");\n";
    oss << "    }\n";
    oss << "}\n\n";
    oss << "subspace LiftedSubspace<" << sub_k << "> in Frame<" << micro_frame_tag << "> {\n";
    oss << "    anchor probe(\"lifted principal reasoning subspace\");\n";
    oss << "}\n\n";
    for (const auto& ring : detected_rings) {
        oss << "manifold " << ring.name << " : S1 in Frame<"
            << ring.layer << ".." << (ring.layer + 1) << "> {\n";
        oss << "    curvature 1.0;\n";
        oss << "    anchor cyclic([\"step0\", \"step1\", \"step2\", \"step3\"]);\n";
        oss << "}\n\n";
    }
    for (const auto& torus : detected_tori) {
        oss << "manifold " << torus.name << " : T2 in Frame<"
            << torus.layer << ".." << (torus.layer + 1) << "> {\n";
        oss << "    curvature 1.0;\n";
        oss << "    anchor cyclic([\"torus_u0\", \"torus_u1\", \"torus_v0\", \"torus_v1\"]);\n";
        oss << "}\n\n";
    }
    oss << "rodata LiftedRodata<Entity, Value> in Frame<" << rodata_frame_tag << "> {\n";
    oss << "    anchor factual(\"The attribute of {subject} is {object}.\");\n";
    oss << "    entry \"anchor_0\" => \"val_0\" with (curvature: " << top_rodata_curvature << ");\n";
    oss << "}\n\n";
    oss << "field LiftedField : LowCurvature in Frame<" << field_frame_tag << "> {\n";
    oss << "    rank " << field_r << ";\n";
    oss << "    metric Fisher;\n";
    oss << "    anchor spectral(prompts: [\"lifted coherent fluency and style mode\"], curvature_band: \"low\");\n";
    oss << "}\n\n";
    oss << "flow LiftedProgram(state: Residual in Frame<" << flow_frame_tag << ">) -> Residual in Frame<" << flow_frame_tag << ">\n";
    oss << "    ![crosstalk <= 1e-2, approx <= 1e-2, energy <= 0.15, preserve(CoreFluency, Residual)]\n";
    oss << "{\n";
    oss << "}\n\n";
    oss << "test VerifyLiftedFaithfulness {\n";
    oss << "    expect crosstalk <= 0.01;\n";
    oss << "    expect approx <= 0.01;\n";
    oss << "}\n";

    return {std::move(nelf), oss.str()};
}

}

std::pair<NeuralELF, std::string> lift_model(
    const TransformerModel& model,
    const std::string& module_name,
    std::pair<int, int> layer_filter) {
    return lift_loaded_model(model, module_name, layer_filter);
}

std::pair<NeuralELF, std::string> lift_checkpoint_streaming(
    const std::string& model_dir,
    const std::string& module_name,
    std::pair<int, int> layer_filter) {
    const std::filesystem::path dir(model_dir);
    if (!std::filesystem::exists(dir)) {
        throw IOError("Checkpoint directory does not exist: " + model_dir);
    }

    const std::filesystem::path cfg_path = dir / "config.json";
    if (!std::filesystem::exists(cfg_path)) {
        throw IOError("Missing config.json in checkpoint directory: " + model_dir);
    }
    const ModelConfig cfg = ModelConfig::from_json(parse_json(read_file_text(cfg_path)));
    const int n_layers = std::max(0, cfg.num_layers);
    const bool filter_active = (layer_filter.first >= 0 && layer_filter.second > layer_filter.first);
    const int fl0 = filter_active ? std::clamp(layer_filter.first, 0, n_layers) : 0;
    const int fl1 = filter_active ? std::clamp(layer_filter.second, fl0, n_layers) : n_layers;

    std::map<std::string, std::vector<std::string>> shard_needed;
    const std::filesystem::path idx_path = dir / "model.safetensors.index.json";
    const std::filesystem::path single_path = dir / "model.safetensors";

    auto should_decode = [&](const std::string& tname) -> bool {
        const int lid = parse_layer_index(tname);
        if (lid < 0) {
            return true;
        }
        if (!filter_active) {
            return true;
        }
        return lid >= fl0 && lid < fl1;
    };

    if (std::filesystem::exists(idx_path)) {
        const JsonValue idx = parse_json(read_file_text(idx_path));
        if (!idx.is_object() || !idx.has("weight_map") || !idx["weight_map"].is_object()) {
            throw IOError("Invalid model.safetensors.index.json in: " + model_dir);
        }
        for (const auto& [tname, sname_v] : idx["weight_map"].as_object()) {
            if (should_decode(tname)) {
                shard_needed[sname_v.as_string()].push_back(tname);
            }
        }
    } else if (std::filesystem::exists(single_path)) {
        MmapSafetensors reader(single_path.string());
        for (const std::string& tname : reader.tensor_names()) {
            if (should_decode(tname)) {
                shard_needed["model.safetensors"].push_back(tname);
            }
        }
    } else {
        throw IOError("No model.safetensors or model.safetensors.index.json in: " + model_dir);
    }

    TransformerModel partial_model;
    partial_model.config = cfg;

    for (const auto& [shard_name, tnames] : shard_needed) {
        if (tnames.empty()) {
            continue;
        }
        MmapSafetensors reader((dir / shard_name).string());
        for (const std::string& tname : tnames) {
            const TensorSpec spec = reader.read_tensor(tname);
            if (spec.shape.size() == 2) {
                partial_model.matrices[tname] = spec.to_matrix();
            } else if (spec.shape.size() == 1) {
                partial_model.vectors[tname] = spec.to_vector();
            } else {
                throw IOError("Unsupported tensor rank in checkpoint for: " + tname);
            }
        }
    }

    return lift_loaded_model(partial_model, module_name, layer_filter);
}

TransformerModel recompile_nelf_with_patch(
    const NeuralELF& nelf,
    const std::string& strata_source,
    const std::string& target_mode) {
    MIRModule mir = compile_source_to_mir(strata_source);
    TransformerModel base_model = nelf.reconstruct_model();

    if (!mir_has_mutations(mir)) {
        return base_model;
    }

    NeuralAtlas patch_atlas = nelf.atlas;
    bool has_unbound = false;
    for (const auto& [sym_name, decl] : mir.declarations) {
        (void)decl;
        if (!patch_atlas.symbols.contains(sym_name)) {
            has_unbound = true;
            break;
        }
    }
    if (has_unbound) {
        patch_atlas = resolve_symbols_for_target(mir, base_model, &nelf.atlas, 0);
    }

    if (!patch_atlas.preserve_directions.contains("Residual") ||
        patch_atlas.preserve_directions["Residual"].empty()) {
        const std::size_t d = static_cast<std::size_t>(std::max(0, base_model.config.hidden_size));
        std::vector<Vector> res_cands;
        for (const auto& [k, rw] : nelf.residual_weights) {
            (void)k;
            const std::size_t min_d = std::min(rw.rows, rw.cols);
            if (min_d > 0) {
                auto ms = svd_top_k(rw, 1, 20, 0);
                if (!ms.empty() && ms[0].u.size() == d) {
                    res_cands.push_back(ms[0].u);
                }
            }
            if (res_cands.size() >= 4) {
                break;
            }
        }
        const std::size_t res_cap = std::max<std::size_t>(1, std::min<std::size_t>(3, d > 4 ? d / 8 : 1));
        patch_atlas.preserve_directions["Residual"] = orthonormal_in_comp(res_cands, res_cap, d, {});
    }

    for (auto& flow : mir.flows) {
        if (std::find(flow.contract.preserve.begin(), flow.contract.preserve.end(), "Residual") ==
            flow.contract.preserve.end()) {
            flow.contract.preserve.push_back("Residual");
        }
    }

    CompileOutput out = compile_mir_to_model(mir, base_model, &patch_atlas, target_mode);

    const std::size_t d = static_cast<std::size_t>(std::max(0, out.model.config.hidden_size));
    if (d > 0 && patch_atlas.preserve_directions.contains("Residual") &&
        !patch_atlas.preserve_directions["Residual"].empty()) {
        const std::vector<Vector> res_gs = gram_schmidt(patch_atlas.preserve_directions["Residual"], 1e-10);
        if (!res_gs.empty()) {
            Matrix p_res_perp = eye(d);
            for (const Vector& q : res_gs) {
                if (q.size() == d) {
                    p_res_perp = mat_sub(p_res_perp, outer(q, q));
                }
            }
            for (auto& [k, new_m] : out.model.matrices) {
                const auto bit = base_model.matrices.find(k);
                if (bit == base_model.matrices.end()) {
                    if (new_m.rows == d) {
                        new_m = mat_mul(p_res_perp, new_m);
                    }
                    if (new_m.cols == d) {
                        new_m = mat_mul(new_m, p_res_perp);
                    }
                    continue;
                }
                const Matrix& base_m = bit->second;
                if (new_m.rows == base_m.rows && new_m.cols == base_m.cols) {
                    Matrix delta = mat_sub(new_m, base_m);
                    if (frob_norm(delta) > 0.0) {
                        if (delta.rows == d) {
                            delta = mat_mul(p_res_perp, delta);
                        }
                        if (delta.cols == d) {
                            delta = mat_mul(delta, p_res_perp);
                        }
                        new_m = mat_add(base_m, delta);
                    }
                } else {
                    Matrix padded_base(new_m.rows, new_m.cols, 0.0);
                    for (std::size_t r = 0; r < base_m.rows && r < new_m.rows; ++r) {
                        for (std::size_t c = 0; c < base_m.cols && c < new_m.cols; ++c) {
                            padded_base(r, c) = base_m(r, c);
                        }
                    }
                    Matrix delta = mat_sub(new_m, padded_base);
                    if (frob_norm(delta) > 0.0) {
                        if (delta.rows == d) {
                            delta = mat_mul(p_res_perp, delta);
                        }
                        if (delta.cols == d) {
                            delta = mat_mul(delta, p_res_perp);
                        }
                        new_m = mat_add(padded_base, delta);
                    }
                }
            }
        }
    }

    return std::move(out.model);
}

FaithfulnessReport verify_nelf_faithfulness(
    const TransformerModel& orig,
    const TransformerModel& recompiled,
    const NeuralELF* nelf) {
    FaithfulnessReport rep;

    long double sum_sq = 0.0;
    std::set<std::string> mat_keys;
    for (const auto& [k, m] : orig.matrices) {
        (void)m;
        mat_keys.insert(k);
    }
    for (const auto& [k, m] : recompiled.matrices) {
        (void)m;
        mat_keys.insert(k);
    }
    for (const std::string& k : mat_keys) {
        const auto it_a = orig.matrices.find(k);
        const auto it_b = recompiled.matrices.find(k);
        if (it_a != orig.matrices.end() && it_b != recompiled.matrices.end()) {
            const Matrix& ma = it_a->second;
            const Matrix& mb = it_b->second;
            const std::size_t max_r = std::max(ma.rows, mb.rows);
            const std::size_t max_c = std::max(ma.cols, mb.cols);
            for (std::size_t r = 0; r < max_r; ++r) {
                for (std::size_t c = 0; c < max_c; ++c) {
                    const double va = (r < ma.rows && c < ma.cols) ? ma(r, c) : 0.0;
                    const double vb = (r < mb.rows && c < mb.cols) ? mb(r, c) : 0.0;
                    const long double d = static_cast<long double>(va - vb);
                    sum_sq += d * d;
                }
            }
        } else if (it_a != orig.matrices.end()) {
            const double fn = frob_norm(it_a->second);
            sum_sq += static_cast<long double>(fn) * static_cast<long double>(fn);
        } else if (it_b != recompiled.matrices.end()) {
            const double fn = frob_norm(it_b->second);
            sum_sq += static_cast<long double>(fn) * static_cast<long double>(fn);
        }
    }

    std::set<std::string> vec_keys;
    for (const auto& [k, v] : orig.vectors) {
        (void)v;
        vec_keys.insert(k);
    }
    for (const auto& [k, v] : recompiled.vectors) {
        (void)v;
        vec_keys.insert(k);
    }
    for (const std::string& k : vec_keys) {
        const auto it_a = orig.vectors.find(k);
        const auto it_b = recompiled.vectors.find(k);
        if (it_a != orig.vectors.end() && it_b != recompiled.vectors.end()) {
            const Vector& va = it_a->second;
            const Vector& vb = it_b->second;
            const std::size_t max_n = std::max(va.size(), vb.size());
            for (std::size_t i = 0; i < max_n; ++i) {
                const double xa = (i < va.size()) ? va[i] : 0.0;
                const double xb = (i < vb.size()) ? vb[i] : 0.0;
                const long double d = static_cast<long double>(xa - xb);
                sum_sq += d * d;
            }
        } else if (it_a != orig.vectors.end()) {
            const double n2 = norm2(it_a->second);
            sum_sq += static_cast<long double>(n2) * static_cast<long double>(n2);
        } else if (it_b != recompiled.vectors.end()) {
            const double n2 = norm2(it_b->second);
            sum_sq += static_cast<long double>(n2) * static_cast<long double>(n2);
        }
    }
    rep.weight_frob_diff = std::sqrt(static_cast<double>(sum_sq));

    if (nelf != nullptr) {
        const auto ratios = nelf->section_energy_ratios();
        const auto rit = ratios.find(".residual");
        if (rit != ratios.end()) {
            rep.residual_energy_ratio = rit->second;
        }
    }

    const int vocab = std::max(1, std::min(orig.config.vocab_size, recompiled.config.vocab_size));
    const std::vector<std::vector<int>> probes = {
        {0 % vocab, 1 % vocab, 2 % vocab, 3 % vocab, 4 % vocab, 5 % vocab},
        {2 % vocab, 5 % vocab, 9 % vocab, 14 % vocab, 19 % vocab, 23 % vocab},
        {4 % vocab, 7 % vocab, 11 % vocab, 15 % vocab, 20 % vocab, 26 % vocab},
        {6 % vocab, 10 % vocab, 13 % vocab, 18 % vocab, 22 % vocab, 29 % vocab}
    };
    const ModelDivergenceStats div = compute_model_divergence(orig, recompiled, probes);
    rep.mean_kl_div = div.mean_kl;
    rep.max_kl_div = div.max_kl;
    rep.max_logit_diff = div.max_logit_diff;

    return rep;
}

}
