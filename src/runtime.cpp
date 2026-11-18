#include "aic/runtime.hpp"

#include "aic/safetensors.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace aic {

namespace {

std::string to_lower_trim(std::string s) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    std::string out = s.substr(start, end - start);
    for (char& c : out) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return out;
}

std::string normalize_storage_dtype(const std::string& raw) {
    const std::string low = to_lower_trim(raw);
    if (low == "bf16" || low == "bfloat16") {
        return "BF16";
    }
    if (low == "f16" || low == "float16") {
        return "F16";
    }
    if (low == "fp8" || low == "f8_e4m3" || low == "float8_e4m3fn") {
        return "F8_E4M3";
    }
    if (low == "i8" || low == "int8") {
        return "I8";
    }
    if (low == "i4" || low == "int4") {
        return "I4";
    }
    if (low == "f32" || low == "float32" || low.empty()) {
        return "F32";
    }
    return "F32";
}

bool is_rope_default_model_type(const std::string& model_type) {
    const std::string low = to_lower_trim(model_type);
    return low == "qwen2" || low == "qwen3" || low == "llama" ||
        low == "mistral" || low == "gemma2" || low == "deepseek_v3";
}

bool try_parse_int(const std::string& s, int& out) {
    std::size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    std::size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    if (start == end) {
        return false;
    }
    std::string trimmed = s.substr(start, end - start);
    const char* begin_ptr = trimmed.data();
    if (trimmed[0] == '+') {
        ++begin_ptr;
    }
    const char* end_ptr = trimmed.data() + trimmed.size();
    if (begin_ptr == end_ptr) {
        return false;
    }
    int val = 0;
    auto [ptr, ec] = std::from_chars(begin_ptr, end_ptr, val);
    if (ec != std::errc{} || ptr != end_ptr) {
        return false;
    }
    out = val;
    return true;
}

std::pair<double, double> stage_fractions(const std::string& name) {
    const std::string key = to_lower_trim(name);
    if (key == "early") {
        return {0.0, 0.25};
    }
    if (key == "mid") {
        return {0.25, 0.75};
    }
    if (key == "late") {
        return {0.75, 0.95};
    }
    if (key == "logit") {
        return {0.95, 1.0};
    }
    if (key == "all") {
        return {0.0, 1.0};
    }
    throw std::invalid_argument("Unknown stage name: " + name);
}

int effective_head_dim(const ModelConfig& cfg) {
    if (cfg.head_dim > 0) {
        return cfg.head_dim;
    }
    return cfg.hidden_size / cfg.num_attention_heads;
}

void validate_config(ModelConfig& cfg) {
    cfg.attention_type = to_lower_trim(cfg.attention_type);
    if (cfg.attention_type != "mha" && cfg.attention_type != "gqa" && cfg.attention_type != "mla") {
        throw std::invalid_argument("Unsupported attention_type: " + cfg.attention_type);
    }
    if (cfg.num_layers < 0) {
        throw std::invalid_argument("num_layers must be non-negative");
    }
    if (cfg.hidden_size <= 0 || cfg.intermediate_size <= 0 || cfg.vocab_size <= 0) {
        throw std::invalid_argument("hidden_size, intermediate_size, and vocab_size must be positive");
    }
    if (cfg.num_attention_heads <= 0) {
        throw std::invalid_argument("num_attention_heads must be positive");
    }
    if (cfg.head_dim <= 0 && (cfg.hidden_size % cfg.num_attention_heads != 0)) {
        throw std::invalid_argument("hidden_size must be divisible by num_attention_heads");
    }
    if (cfg.attention_type == "mha") {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }
    if (cfg.num_key_value_heads <= 0) {
        throw std::invalid_argument("num_key_value_heads must be positive");
    }
    if ((cfg.attention_type == "mha" || cfg.attention_type == "gqa") &&
        (cfg.num_attention_heads % cfg.num_key_value_heads != 0)) {
        throw std::invalid_argument("num_attention_heads must be divisible by num_key_value_heads");
    }
    if (cfg.attention_type == "mla" && cfg.kv_lora_rank <= 0) {
        throw std::invalid_argument("kv_lora_rank must be positive for mla");
    }
    if (cfg.rope_theta <= 0.0) {
        throw std::invalid_argument("rope_theta must be positive");
    }
}

Matrix rand_matrix(std::mt19937_64& rng, std::size_t rows, std::size_t cols, double stddev) {
    Matrix m(rows, cols, 0.0);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (double& x : m.data) {
        x = dist(rng) * stddev;
    }
    return m;
}

Matrix slice_rows(const Matrix& m, std::size_t row_begin, std::size_t row_end) {
    Matrix out(row_end - row_begin, m.cols, 0.0);
    for (std::size_t r = row_begin; r < row_end; ++r) {
        for (std::size_t c = 0; c < m.cols; ++c) {
            out(r - row_begin, c) = m(r, c);
        }
    }
    return out;
}

Matrix pack_rows(const std::vector<Vector>& rows, std::size_t cols) {
    Matrix out(rows.size(), cols, 0.0);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (rows[r].size() != cols) {
            throw std::invalid_argument("Row size mismatch in pack_rows");
        }
        std::copy(rows[r].begin(), rows[r].end(), out.data.begin() + static_cast<std::ptrdiff_t>(r * cols));
    }
    return out;
}

std::vector<Vector> unpack_rows(const Matrix& mat) {
    std::vector<Vector> out(mat.rows);
    for (std::size_t r = 0; r < mat.rows; ++r) {
        out[r] = mat.row(r);
    }
    return out;
}

const Matrix& require_matrix(const std::map<std::string, Matrix>& mats, const std::string& key) {
    const auto it = mats.find(key);
    if (it == mats.end()) {
        throw std::runtime_error("Missing matrix tensor: " + key);
    }
    return it->second;
}

const Vector& require_vector(const std::map<std::string, Vector>& vecs, const std::string& key) {
    const auto it = vecs.find(key);
    if (it == vecs.end()) {
        throw std::runtime_error("Missing vector tensor: " + key);
    }
    return it->second;
}

void add_bias_inplace(Vector& x, const Vector& bias) {
    if (x.size() != bias.size()) {
        throw std::invalid_argument("Bias length mismatch in forward");
    }
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] += bias[i];
    }
}

void add_bias_matrix_inplace(Matrix& mat, const Vector& bias) {
    if (mat.cols != bias.size()) {
        throw std::invalid_argument("Bias length mismatch in batched forward");
    }
    for (std::size_t r = 0; r < mat.rows; ++r) {
        const std::size_t base = r * mat.cols;
        for (std::size_t c = 0; c < mat.cols; ++c) {
            mat.data[base + c] += bias[c];
        }
    }
}

void apply_rope_inplace(Vector& head_vec, std::size_t pos, double rope_theta) {
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

void apply_head_norm_and_rope(
    Vector& vec,
    std::size_t num_heads,
    std::size_t hd,
    const Vector* norm_w,
    bool use_rope,
    std::size_t pos,
    double rope_theta,
    double eps) {
    if (norm_w == nullptr && !use_rope) {
        return;
    }
    for (std::size_t h_idx = 0; h_idx < num_heads; ++h_idx) {
        const std::size_t start = h_idx * hd;
        Vector head(
            vec.begin() + static_cast<std::ptrdiff_t>(start),
            vec.begin() + static_cast<std::ptrdiff_t>(start + hd));
        if (norm_w != nullptr) {
            head = rms_norm(head, *norm_w, eps);
        }
        if (use_rope) {
            apply_rope_inplace(head, pos, rope_theta);
        }
        std::copy(head.begin(), head.end(), vec.begin() + static_cast<std::ptrdiff_t>(start));
    }
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
        s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

const Matrix& resolve_lm_head(const TransformerModel& model) {
    const auto lm_it = model.matrices.find("lm_head.weight");
    if (lm_it != model.matrices.end()) {
        return lm_it->second;
    }
    if (model.config.tie_word_embeddings) {
        return require_matrix(model.matrices, "model.embed_tokens.weight");
    }
    return require_matrix(model.matrices, "lm_head.weight");
}

Vector threshold_route_delta(
    const TransformerModel& model,
    const std::string& prefix,
    const Vector& acts) {
    const std::string route_key = prefix + ".mlp.aic_threshold_route.weight";
    const std::string threshold_key = prefix + ".mlp.aic_threshold_route.threshold";
    const std::string polarity_key = prefix + ".mlp.aic_threshold_route.polarity";
    const std::string mask_key = prefix + ".mlp.aic_threshold_route.mask";
    auto route_it = model.matrices.find(route_key);
    if (route_it == model.matrices.end()) {
        return Vector(static_cast<std::size_t>(model.config.hidden_size), 0.0);
    }
    const Matrix& route = route_it->second;
    const Vector& thresholds = require_vector(model.vectors, threshold_key);
    const Vector& polarities = require_vector(model.vectors, polarity_key);
    const Vector& mask = require_vector(model.vectors, mask_key);
    if (route.rows != static_cast<std::size_t>(model.config.hidden_size) ||
        route.cols != acts.size() || thresholds.size() != acts.size() ||
        polarities.size() != acts.size() || mask.size() != acts.size()) {
        throw std::invalid_argument("Invalid AIC threshold route tensor shape at " + prefix);
    }
    Vector delta(route.rows, 0.0);
    for (std::size_t slot = 0; slot < acts.size(); ++slot) {
        if (mask[slot] <= 0.0) {
            continue;
        }
        const double detector = polarities[slot] * acts[slot];
        if (detector <= thresholds[slot]) {
            continue;
        }
        const double gate = detector;
        for (std::size_t r = 0; r < route.rows; ++r) {
            delta[r] += gate * route(r, slot);
        }
    }
    return delta;
}

void run_transformer_layers(
    const TransformerModel& model,
    std::vector<Vector>& h,
    bool record_trace,
    ForwardTrace* trace_out,
    KVCache* cache_out) {
    const ModelConfig& config = model.config;
    const std::size_t seq_len = h.size();
    const std::size_t d = static_cast<std::size_t>(config.hidden_size);
    const std::size_t num_q_heads = static_cast<std::size_t>(config.num_attention_heads);
    const std::size_t hd = static_cast<std::size_t>(effective_head_dim(config));
    const double inv_sqrt_hd = 1.0 / std::sqrt(static_cast<double>(hd));
    const double eps = config.rms_norm_eps;
    const std::string attn_mode = to_lower_trim(config.attention_type);

    if (record_trace && trace_out != nullptr) {
        trace_out->residual.reserve(static_cast<std::size_t>(config.num_layers) + 1);
        trace_out->mlp_poles.reserve(static_cast<std::size_t>(config.num_layers));
        trace_out->residual.push_back(h);
    }

    for (int l = 0; l < config.num_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l);
        const Vector& in_ln = require_vector(model.vectors, prefix + ".input_layernorm.weight");

        std::vector<Vector> x_norm(seq_len);
        for (std::size_t t = 0; t < seq_len; ++t) {
            x_norm[t] = rms_norm(h[t], in_ln, eps);
        }

        const Matrix& w_q = require_matrix(model.matrices, prefix + ".self_attn.q_proj.weight");
        const Matrix& w_o = require_matrix(model.matrices, prefix + ".self_attn.o_proj.weight");
        const auto qb_it = model.vectors.find(prefix + ".self_attn.q_proj.bias");

        std::vector<Vector> q_all(seq_len);
        std::vector<Vector> k_all(seq_len);
        std::vector<Vector> v_all(seq_len);

        const std::string kva_key = prefix + ".self_attn.kv_a_proj.weight";
        const std::string kvb_key = prefix + ".self_attn.kv_b_proj.weight";
        const bool is_mla = (attn_mode == "mla") &&
            model.matrices.contains(kva_key) &&
            model.matrices.contains(kvb_key);

        if (seq_len > 1) {
            const Matrix X_norm = pack_rows(x_norm, d);
            Matrix Q_mat = mat_mul(X_norm, transpose(w_q));
            if (qb_it != model.vectors.end()) {
                add_bias_matrix_inplace(Q_mat, qb_it->second);
            }
            q_all = unpack_rows(Q_mat);

            if (is_mla) {
                const Matrix& w_kva = require_matrix(model.matrices, kva_key);
                const Matrix& w_kvb = require_matrix(model.matrices, kvb_key);
                const Matrix C_kv_mat = mat_mul(X_norm, transpose(w_kva));
                const Matrix KV_mat = mat_mul(C_kv_mat, transpose(w_kvb));
                const std::size_t half = KV_mat.cols / 2;
                for (std::size_t t = 0; t < seq_len; ++t) {
                    const std::size_t base = t * KV_mat.cols;
                    k_all[t].assign(
                        KV_mat.data.begin() + static_cast<std::ptrdiff_t>(base),
                        KV_mat.data.begin() + static_cast<std::ptrdiff_t>(base + half));
                    v_all[t].assign(
                        KV_mat.data.begin() + static_cast<std::ptrdiff_t>(base + half),
                        KV_mat.data.begin() + static_cast<std::ptrdiff_t>(base + KV_mat.cols));
                    if (cache_out != nullptr) {
                        cache_out->layers[static_cast<std::size_t>(l)].c_kv.push_back(C_kv_mat.row(t));
                    }
                }
            } else {
                const Matrix& w_k = require_matrix(model.matrices, prefix + ".self_attn.k_proj.weight");
                const Matrix& w_v = require_matrix(model.matrices, prefix + ".self_attn.v_proj.weight");
                const auto kb_it = model.vectors.find(prefix + ".self_attn.k_proj.bias");
                const auto vb_it = model.vectors.find(prefix + ".self_attn.v_proj.bias");
                Matrix K_mat = mat_mul(X_norm, transpose(w_k));
                Matrix V_mat = mat_mul(X_norm, transpose(w_v));
                if (kb_it != model.vectors.end()) {
                    add_bias_matrix_inplace(K_mat, kb_it->second);
                }
                if (vb_it != model.vectors.end()) {
                    add_bias_matrix_inplace(V_mat, vb_it->second);
                }
                k_all = unpack_rows(K_mat);
                v_all = unpack_rows(V_mat);
            }
        } else {
            q_all[0] = mat_vec(w_q, x_norm[0]);
            if (qb_it != model.vectors.end()) {
                add_bias_inplace(q_all[0], qb_it->second);
            }
            if (is_mla) {
                const Matrix& w_kva = require_matrix(model.matrices, kva_key);
                const Matrix& w_kvb = require_matrix(model.matrices, kvb_key);
                const Vector c_kv = mat_vec(w_kva, x_norm[0]);
                const Vector kv = mat_vec(w_kvb, c_kv);
                const std::size_t half = kv.size() / 2;
                k_all[0].assign(kv.begin(), kv.begin() + static_cast<std::ptrdiff_t>(half));
                v_all[0].assign(kv.begin() + static_cast<std::ptrdiff_t>(half), kv.end());
                if (cache_out != nullptr) {
                    cache_out->layers[static_cast<std::size_t>(l)].c_kv.push_back(c_kv);
                }
            } else {
                const Matrix& w_k = require_matrix(model.matrices, prefix + ".self_attn.k_proj.weight");
                const Matrix& w_v = require_matrix(model.matrices, prefix + ".self_attn.v_proj.weight");
                const auto kb_it = model.vectors.find(prefix + ".self_attn.k_proj.bias");
                const auto vb_it = model.vectors.find(prefix + ".self_attn.v_proj.bias");
                k_all[0] = mat_vec(w_k, x_norm[0]);
                v_all[0] = mat_vec(w_v, x_norm[0]);
                if (kb_it != model.vectors.end()) {
                    add_bias_inplace(k_all[0], kb_it->second);
                }
                if (vb_it != model.vectors.end()) {
                    add_bias_inplace(v_all[0], vb_it->second);
                }
            }
        }

        const std::size_t kv_heads_eff = k_all[0].size() / hd;
        const std::size_t group_size = num_q_heads / kv_heads_eff;

        const auto qn_it = model.vectors.find(prefix + ".self_attn.q_norm.weight");
        const auto kn_it = model.vectors.find(prefix + ".self_attn.k_norm.weight");
        const Vector* q_norm_w = (qn_it != model.vectors.end()) ? &qn_it->second : nullptr;
        const Vector* k_norm_w = (kn_it != model.vectors.end()) ? &kn_it->second : nullptr;

        if (q_norm_w != nullptr || k_norm_w != nullptr || config.use_rope) {
            for (std::size_t t = 0; t < seq_len; ++t) {
                apply_head_norm_and_rope(
                    q_all[t], num_q_heads, hd, q_norm_w, config.use_rope, t, config.rope_theta, eps);
                apply_head_norm_and_rope(
                    k_all[t], kv_heads_eff, hd, k_norm_w, config.use_rope, t, config.rope_theta, eps);
            }
        }

        if (cache_out != nullptr) {
            LayerKVCache& lcache = cache_out->layers[static_cast<std::size_t>(l)];
            for (std::size_t t = 0; t < seq_len; ++t) {
                lcache.k.push_back(k_all[t]);
                lcache.v.push_back(v_all[t]);
            }
        }

        const std::size_t q_dim = num_q_heads * hd;
        std::vector<Vector> concat_all(seq_len, Vector(q_dim, 0.0));
        for (std::size_t t = 0; t < seq_len; ++t) {
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
                        concat_all[t][q_start + m] += p * v_all[j][kv_start + m];
                    }
                }
            }
        }

        std::vector<Vector> h_mid(seq_len);
        if (seq_len > 1) {
            const Matrix Ctx_mat = pack_rows(concat_all, q_dim);
            const Matrix Attn_mat = mat_mul(Ctx_mat, transpose(w_o));
            for (std::size_t t = 0; t < seq_len; ++t) {
                h_mid[t] = vec_add(h[t], Attn_mat.row(t));
            }
        } else {
            h_mid[0] = vec_add(h[0], mat_vec(w_o, concat_all[0]));
        }

        const Vector& post_ln = require_vector(model.vectors, prefix + ".post_attention_layernorm.weight");
        const Matrix& w_gate = require_matrix(model.matrices, prefix + ".mlp.gate_proj.weight");
        const Matrix& w_up = require_matrix(model.matrices, prefix + ".mlp.up_proj.weight");
        const Matrix& w_down = require_matrix(model.matrices, prefix + ".mlp.down_proj.weight");
        const std::size_t d_ff_actual = w_gate.rows;

        std::vector<PoleTrace> layer_poles;
        if (record_trace && trace_out != nullptr) {
            layer_poles.resize(d_ff_actual);
            for (std::size_t i = 0; i < d_ff_actual; ++i) {
                layer_poles[i].gate_pre.resize(seq_len, 0.0);
                layer_poles[i].up_val.resize(seq_len, 0.0);
                layer_poles[i].act.resize(seq_len, 0.0);
            }
        }

        std::vector<Vector> h_next(seq_len);
        if (seq_len > 1) {
            std::vector<Vector> x_post(seq_len);
            for (std::size_t t = 0; t < seq_len; ++t) {
                x_post[t] = rms_norm(h_mid[t], post_ln, eps);
            }
            const Matrix X_post = pack_rows(x_post, d);
            const Matrix G_mat = mat_mul(X_post, transpose(w_gate));
            const Matrix U_mat = mat_mul(X_post, transpose(w_up));
            Matrix Act_mat(seq_len, d_ff_actual, 0.0);
            for (std::size_t t = 0; t < seq_len; ++t) {
                const std::size_t row_base = t * d_ff_actual;
                for (std::size_t i = 0; i < d_ff_actual; ++i) {
                    const double gv = G_mat.data[row_base + i];
                    const double uv = U_mat.data[row_base + i];
                    const double av = silu(gv) * uv;
                    Act_mat.data[row_base + i] = av;
                    if (record_trace && trace_out != nullptr) {
                        layer_poles[i].gate_pre[t] = gv;
                        layer_poles[i].up_val[t] = uv;
                        layer_poles[i].act[t] = av;
                    }
                }
            }
            const Matrix Mlp_mat = mat_mul(Act_mat, transpose(w_down));
            for (std::size_t t = 0; t < seq_len; ++t) {
                h_next[t] = vec_add(h_mid[t], Mlp_mat.row(t));
                if (model.config.aic_threshold_routes) {
                    h_next[t] = vec_add(
                        h_next[t], threshold_route_delta(model, prefix, Act_mat.row(t)));
                }
            }
        } else {
            const Vector x_post = rms_norm(h_mid[0], post_ln, eps);
            const Vector g_pre = mat_vec(w_gate, x_post);
            const Vector u_val = mat_vec(w_up, x_post);
            Vector acts(d_ff_actual);
            for (std::size_t i = 0; i < d_ff_actual; ++i) {
                acts[i] = silu(g_pre[i]) * u_val[i];
                if (record_trace && trace_out != nullptr) {
                    layer_poles[i].gate_pre[0] = g_pre[i];
                    layer_poles[i].up_val[0] = u_val[i];
                    layer_poles[i].act[0] = acts[i];
                }
            }
            h_next[0] = vec_add(h_mid[0], mat_vec(w_down, acts));
            if (model.config.aic_threshold_routes) {
                h_next[0] = vec_add(h_next[0], threshold_route_delta(model, prefix, acts));
            }
        }

        h = std::move(h_next);
        if (record_trace && trace_out != nullptr) {
            trace_out->residual.push_back(h);
            trace_out->mlp_poles.push_back(std::move(layer_poles));
        }
    }

    if (cache_out != nullptr) {
        cache_out->seq_len += seq_len;
    }
}

Vector prefill_last_logits(
    const TransformerModel& model,
    const std::vector<int>& prompt_ids,
    KVCache& cache) {
    if (prompt_ids.size() == 1) {
        return model.forward_step(prompt_ids[0], cache);
    }
    if (cache.layers.size() != static_cast<std::size_t>(model.config.num_layers)) {
        throw std::invalid_argument("KVCache layer count mismatch in prefill");
    }
    if (cache.seq_len != 0) {
        throw std::invalid_argument("Expected empty KVCache in prefill_last_logits");
    }

    const Matrix& embed_w = require_matrix(model.matrices, "model.embed_tokens.weight");
    const int vocab_n = static_cast<int>(embed_w.rows);
    std::vector<Vector> h;
    h.reserve(prompt_ids.size());
    for (int tid : prompt_ids) {
        if (tid < 0 || tid >= vocab_n) {
            throw std::out_of_range("Token ID out of range in prefill");
        }
        h.push_back(embed_w.row(static_cast<std::size_t>(tid)));
    }

    run_transformer_layers(model, h, false, nullptr, &cache);

    const Vector& final_ln = require_vector(model.vectors, "model.norm.weight");
    const Matrix& w_lm_head = resolve_lm_head(model);
    const Vector h_final = rms_norm(h.back(), final_ln, model.config.rms_norm_eps);
    return mat_vec(w_lm_head, h_final);
}

int argmax_index(const Vector& logits) {
    if (logits.empty()) {
        throw std::runtime_error("Empty logit vector");
    }
    std::size_t best_id = 0;
    double best_val = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = i;
        }
    }
    return static_cast<int>(best_id);
}

int sample_nucleus(const Vector& logits, double temperature, double top_p, std::mt19937_64& rng) {
    if (logits.empty()) {
        throw std::runtime_error("Empty logit vector");
    }
    Vector scaled(logits.size());
    const double inv_t = 1.0 / temperature;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        scaled[i] = logits[i] * inv_t;
    }
    const Vector probs = softmax(scaled);

    if (top_p >= 1.0) {
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        const double r = u01(rng);
        double cum = 0.0;
        for (std::size_t i = 0; i < probs.size(); ++i) {
            cum += probs[i];
            if (r <= cum) {
                return static_cast<int>(i);
            }
        }
        return static_cast<int>(probs.size() - 1);
    }

    std::vector<std::pair<double, int>> items;
    items.reserve(probs.size());
    for (std::size_t i = 0; i < probs.size(); ++i) {
        items.emplace_back(probs[i], static_cast<int>(i));
    }
    std::sort(items.begin(), items.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) {
            return a.first > b.first;
        }
        return a.second < b.second;
    });

    std::vector<std::pair<double, int>> nucleus;
    double cum = 0.0;
    for (const auto& item : items) {
        nucleus.push_back(item);
        cum += item.first;
        if (cum >= top_p) {
            break;
        }
    }

    double sum_nuc = 0.0;
    for (const auto& item : nucleus) {
        sum_nuc += item.first;
    }
    if (sum_nuc <= 0.0) {
        return nucleus[0].second;
    }

    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double r = u01(rng) * sum_nuc;
    double acc = 0.0;
    for (const auto& item : nucleus) {
        acc += item.first;
        if (r <= acc) {
            return item.second;
        }
    }
    return nucleus.back().second;
}

}

std::pair<int, int> ModelConfig::stage_to_layers(
    const std::string& start,
    const std::string& end,
    bool inclusive) const {
    const int L = num_layers;
    if (L <= 0) {
        return {0, 0};
    }

    if (end.empty()) {
        int val = 0;
        if (try_parse_int(start, val)) {
            const int lo = std::clamp(val, 0, L);
            const int hi = std::clamp(val + 1, lo, L);
            return {lo, hi};
        }
        const auto [f_lo, f_hi] = stage_fractions(start);
        const int lo = std::min(L - 1, static_cast<int>(f_lo * static_cast<double>(L)));
        const int raw_hi = (f_hi >= 1.0) ? L : static_cast<int>(f_hi * static_cast<double>(L));
        const int hi = std::min(L, std::max(lo + 1, raw_hi));
        return {lo, hi};
    }

    int start_int = 0;
    const bool has_start_int = try_parse_int(start, start_int);
    int lo = 0;
    if (has_start_int) {
        lo = std::clamp(start_int, 0, L);
    } else {
        const auto [f_lo, f_hi] = stage_fractions(start);
        (void)f_hi;
        lo = std::min(L - 1, static_cast<int>(f_lo * static_cast<double>(L)));
    }

    int end_int = 0;
    if (try_parse_int(end, end_int)) {
        const int raw_hi = end_int + (inclusive ? 1 : 0);
        const int hi = std::clamp(raw_hi, lo, L);
        return {lo, hi};
    }

    const auto [f_lo, f_hi] = stage_fractions(end);
    (void)f_lo;
    const int raw_hi = (f_hi >= 1.0) ? L : static_cast<int>(f_hi * static_cast<double>(L));
    const int hi = std::min(L, std::max(lo + 1, raw_hi));
    return {lo, hi};
}

JsonValue ModelConfig::to_json() const {
    JsonValue j = JsonValue::object();
    j["num_layers"] = num_layers;
    j["hidden_size"] = hidden_size;
    j["intermediate_size"] = intermediate_size;
    j["num_attention_heads"] = num_attention_heads;
    j["num_key_value_heads"] = num_key_value_heads;
    j["vocab_size"] = vocab_size;
    j["attention_type"] = attention_type;
    j["kv_lora_rank"] = kv_lora_rank;
    j["rms_norm_eps"] = rms_norm_eps;
    j["arch_name"] = arch_name;
    j["model_type"] = model_type;
    j["head_dim"] = head_dim;
    j["use_rope"] = use_rope;
    j["rope_theta"] = rope_theta;
    j["tie_word_embeddings"] = tie_word_embeddings;
    j["qk_norm"] = qk_norm;
    j["qkv_bias"] = qkv_bias;
    j["aic_threshold_routes"] = aic_threshold_routes;
    j["torch_dtype"] = torch_dtype;
    return j;
}

ModelConfig ModelConfig::from_json(const JsonValue& j) {
    if (!j.is_object()) {
        throw std::invalid_argument("ModelConfig JSON must be an object");
    }
    ModelConfig cfg;
    if (j.has("num_layers") && j["num_layers"].is_number()) {
        cfg.num_layers = static_cast<int>(j["num_layers"].as_int());
    } else if (j.has("num_hidden_layers") && j["num_hidden_layers"].is_number()) {
        cfg.num_layers = static_cast<int>(j["num_hidden_layers"].as_int());
    }
    if (j.has("hidden_size") && j["hidden_size"].is_number()) {
        cfg.hidden_size = static_cast<int>(j["hidden_size"].as_int());
    }
    if (j.has("intermediate_size") && j["intermediate_size"].is_number()) {
        cfg.intermediate_size = static_cast<int>(j["intermediate_size"].as_int());
    }
    if (j.has("num_attention_heads") && j["num_attention_heads"].is_number()) {
        cfg.num_attention_heads = static_cast<int>(j["num_attention_heads"].as_int());
    }
    if (j.has("num_key_value_heads") && j["num_key_value_heads"].is_number()) {
        cfg.num_key_value_heads = static_cast<int>(j["num_key_value_heads"].as_int());
    } else if (j.has("num_attention_heads") && j["num_attention_heads"].is_number() &&
               !j.has("attention_type")) {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }
    if (j.has("vocab_size") && j["vocab_size"].is_number()) {
        cfg.vocab_size = static_cast<int>(j["vocab_size"].as_int());
    }
    if (j.has("kv_lora_rank") && j["kv_lora_rank"].is_number()) {
        cfg.kv_lora_rank = static_cast<int>(j["kv_lora_rank"].as_int());
    }
    if (j.has("attention_type") && j["attention_type"].is_string()) {
        cfg.attention_type = j["attention_type"].as_string();
    } else if (j.has("kv_lora_rank") && j["kv_lora_rank"].is_number()) {
        cfg.attention_type = "mla";
    } else if (j.has("num_key_value_heads") || j.has("num_attention_heads")) {
        cfg.attention_type = (cfg.num_key_value_heads < cfg.num_attention_heads) ? "gqa" : "mha";
    }
    if (j.has("rms_norm_eps") && j["rms_norm_eps"].is_number()) {
        cfg.rms_norm_eps = j["rms_norm_eps"].as_number();
    }
    if (j.has("model_type") && j["model_type"].is_string()) {
        cfg.model_type = j["model_type"].as_string();
    }
    if (j.has("arch_name") && j["arch_name"].is_string()) {
        cfg.arch_name = j["arch_name"].as_string();
    } else if (j.has("architectures") && j["architectures"].is_array() &&
               !j["architectures"].as_array().empty() &&
               j["architectures"].as_array()[0].is_string()) {
        cfg.arch_name = j["architectures"].as_array()[0].as_string();
    } else if (!cfg.model_type.empty()) {
        cfg.arch_name = cfg.model_type;
    }
    if (j.has("head_dim") && j["head_dim"].is_number()) {
        cfg.head_dim = static_cast<int>(j["head_dim"].as_int());
    }
    if (j.has("use_rope") && j["use_rope"].is_bool()) {
        cfg.use_rope = j["use_rope"].as_bool();
    } else if (is_rope_default_model_type(cfg.model_type)) {
        cfg.use_rope = true;
    }
    if (j.has("rope_theta") && j["rope_theta"].is_number()) {
        cfg.rope_theta = j["rope_theta"].as_number();
    }
    if (j.has("tie_word_embeddings") && j["tie_word_embeddings"].is_bool()) {
        cfg.tie_word_embeddings = j["tie_word_embeddings"].as_bool();
    }
    if (j.has("qk_norm") && j["qk_norm"].is_bool()) {
        cfg.qk_norm = j["qk_norm"].as_bool();
    } else if (j.has("use_qk_norm") && j["use_qk_norm"].is_bool()) {
        cfg.qk_norm = j["use_qk_norm"].as_bool();
    }
    if (j.has("qkv_bias") && j["qkv_bias"].is_bool()) {
        cfg.qkv_bias = j["qkv_bias"].as_bool();
    } else if (j.has("attention_bias") && j["attention_bias"].is_bool()) {
        cfg.qkv_bias = j["attention_bias"].as_bool();
    }
    if (j.has("aic_threshold_routes") && j["aic_threshold_routes"].is_bool()) {
        cfg.aic_threshold_routes = j["aic_threshold_routes"].as_bool();
    }
    if (j.has("torch_dtype") && j["torch_dtype"].is_string()) {
        cfg.torch_dtype = normalize_storage_dtype(j["torch_dtype"].as_string());
    }
    validate_config(cfg);
    return cfg;
}

TransformerModel TransformerModel::create_synthetic(ModelConfig cfg, std::uint64_t seed) {
    validate_config(cfg);
    std::mt19937_64 rng(seed);

    const std::size_t d = static_cast<std::size_t>(cfg.hidden_size);
    const std::size_t d_ff = static_cast<std::size_t>(cfg.intermediate_size);
    const std::size_t vocab = static_cast<std::size_t>(cfg.vocab_size);
    const std::size_t hd = static_cast<std::size_t>(effective_head_dim(cfg));
    const std::size_t q_dim = static_cast<std::size_t>(cfg.num_attention_heads) * hd;

    TransformerModel model;
    model.config = cfg;

    model.matrices["model.embed_tokens.weight"] = rand_matrix(rng, vocab, d, 0.5);

    const double attn_std = 1.0 / std::sqrt(static_cast<double>(d));
    const double o_std = 0.5 / std::sqrt(static_cast<double>(q_dim));
    const double mlp_in_std = 1.0 / std::sqrt(static_cast<double>(d));
    const double mlp_down_std = 0.5 / std::sqrt(static_cast<double>(d_ff));

    for (int l = 0; l < cfg.num_layers; ++l) {
        const std::string prefix = "model.layers." + std::to_string(l);
        model.vectors[prefix + ".input_layernorm.weight"] = Vector(d, 1.0);
        model.matrices[prefix + ".self_attn.q_proj.weight"] = rand_matrix(rng, q_dim, d, attn_std);

        std::size_t kv_dim = d;
        if (cfg.attention_type == "mla") {
            const std::size_t rank = static_cast<std::size_t>(cfg.kv_lora_rank);
            Matrix kv_a = rand_matrix(rng, rank, d, attn_std);
            Matrix kv_b = rand_matrix(rng, 2 * q_dim, rank, 1.0 / std::sqrt(static_cast<double>(rank)));
            Matrix k_proj = mat_mul(slice_rows(kv_b, 0, q_dim), kv_a);
            Matrix v_proj = mat_mul(slice_rows(kv_b, q_dim, 2 * q_dim), kv_a);
            model.matrices[prefix + ".self_attn.kv_a_proj.weight"] = std::move(kv_a);
            model.matrices[prefix + ".self_attn.kv_b_proj.weight"] = std::move(kv_b);
            model.matrices[prefix + ".self_attn.k_proj.weight"] = std::move(k_proj);
            model.matrices[prefix + ".self_attn.v_proj.weight"] = std::move(v_proj);
            kv_dim = q_dim;
        } else {
            kv_dim = static_cast<std::size_t>(cfg.num_key_value_heads) * hd;
            model.matrices[prefix + ".self_attn.k_proj.weight"] = rand_matrix(rng, kv_dim, d, attn_std);
            model.matrices[prefix + ".self_attn.v_proj.weight"] = rand_matrix(rng, kv_dim, d, attn_std);
        }

        if (cfg.qkv_bias) {
            model.vectors[prefix + ".self_attn.q_proj.bias"] = Vector(q_dim, 0.0);
            model.vectors[prefix + ".self_attn.k_proj.bias"] = Vector(kv_dim, 0.0);
            model.vectors[prefix + ".self_attn.v_proj.bias"] = Vector(kv_dim, 0.0);
        }
        if (cfg.qk_norm) {
            model.vectors[prefix + ".self_attn.q_norm.weight"] = Vector(hd, 1.0);
            model.vectors[prefix + ".self_attn.k_norm.weight"] = Vector(hd, 1.0);
        }

        model.matrices[prefix + ".self_attn.o_proj.weight"] = rand_matrix(rng, d, q_dim, o_std);
        model.vectors[prefix + ".post_attention_layernorm.weight"] = Vector(d, 1.0);
        model.matrices[prefix + ".mlp.gate_proj.weight"] = rand_matrix(rng, d_ff, d, mlp_in_std);
        model.matrices[prefix + ".mlp.up_proj.weight"] = rand_matrix(rng, d_ff, d, mlp_in_std);
        model.matrices[prefix + ".mlp.down_proj.weight"] = rand_matrix(rng, d, d_ff, mlp_down_std);
    }

    model.vectors["model.norm.weight"] = Vector(d, 1.0);
    if (!cfg.tie_word_embeddings) {
        model.matrices["lm_head.weight"] = rand_matrix(rng, vocab, d, 1.0 / std::sqrt(static_cast<double>(d)));
    }
    return model;
}

std::pair<std::vector<Vector>, ForwardTrace> TransformerModel::forward(
    const std::vector<int>& token_ids,
    const std::vector<Vector>& hidden_inputs,
    bool record_trace) const {
    const bool has_tokens = !token_ids.empty();
    const bool has_hidden = !hidden_inputs.empty();
    if (has_tokens == has_hidden) {
        throw std::invalid_argument("Provide exactly one of non-empty token_ids or hidden_inputs");
    }

    const std::size_t d = static_cast<std::size_t>(config.hidden_size);
    std::vector<Vector> h;

    if (has_tokens) {
        const Matrix& embed_w = require_matrix(matrices, "model.embed_tokens.weight");
        const int vocab_n = static_cast<int>(embed_w.rows);
        h.reserve(token_ids.size());
        for (int tid : token_ids) {
            if (tid < 0 || tid >= vocab_n) {
                throw std::out_of_range("Token ID out of range in forward");
            }
            h.push_back(embed_w.row(static_cast<std::size_t>(tid)));
        }
    } else {
        h.reserve(hidden_inputs.size());
        for (const Vector& v : hidden_inputs) {
            if (v.size() != d) {
                throw std::invalid_argument("Hidden vector dimension mismatch in forward");
            }
            h.push_back(v);
        }
    }

    const std::size_t seq_len = h.size();
    ForwardTrace trace;
    run_transformer_layers(*this, h, record_trace, record_trace ? &trace : nullptr, nullptr);

    const Vector& final_ln = require_vector(vectors, "model.norm.weight");
    const Matrix& w_lm_head = resolve_lm_head(*this);

    std::vector<Vector> logits(seq_len);
    if (seq_len > 1) {
        std::vector<Vector> h_final(seq_len);
        for (std::size_t t = 0; t < seq_len; ++t) {
            h_final[t] = rms_norm(h[t], final_ln, config.rms_norm_eps);
        }
        const Matrix H_final = pack_rows(h_final, d);
        const Matrix Logits_mat = mat_mul(H_final, transpose(w_lm_head));
        logits = unpack_rows(Logits_mat);
    } else {
        const Vector h_final = rms_norm(h[0], final_ln, config.rms_norm_eps);
        logits[0] = mat_vec(w_lm_head, h_final);
    }

    return {std::move(logits), std::move(trace)};
}

std::pair<std::vector<Vector>, ForwardTrace> TransformerModel::forward_hidden(
    const std::vector<Vector>& hidden_inputs,
    bool record_trace) const {
    const std::size_t d = static_cast<std::size_t>(config.hidden_size);
    if (hidden_inputs.empty()) {
        throw std::invalid_argument("forward_hidden requires at least one hidden vector");
    }
    for (const Vector& v : hidden_inputs) {
        if (v.size() != d) {
            throw std::invalid_argument("Hidden vector dimension mismatch in forward_hidden");
        }
    }
    std::vector<Vector> h = hidden_inputs;
    ForwardTrace trace;
    run_transformer_layers(*this, h, record_trace, record_trace ? &trace : nullptr, nullptr);
    return {std::move(h), std::move(trace)};
}

KVCache TransformerModel::init_kv_cache() const {
    KVCache cache;
    cache.layers.resize(static_cast<std::size_t>(config.num_layers));
    cache.seq_len = 0;
    return cache;
}

Vector forward_step_impl(
    const TransformerModel& model,
    int token_id,
    KVCache& cache,
    ForwardStepTrace* step_trace) {
    const ModelConfig& config = model.config;
    if (cache.layers.size() != static_cast<std::size_t>(config.num_layers)) {
        throw std::invalid_argument("KVCache layer count mismatch in forward_step");
    }

    const Matrix& embed_w = require_matrix(model.matrices, "model.embed_tokens.weight");
    const int vocab_n = static_cast<int>(embed_w.rows);
    if (token_id < 0 || token_id >= vocab_n) {
        throw std::out_of_range("Token ID out of range in forward_step");
    }

    Vector h = embed_w.row(static_cast<std::size_t>(token_id));
    const std::size_t pos = cache.seq_len;
    const std::size_t num_q_heads = static_cast<std::size_t>(config.num_attention_heads);
    const std::size_t hd = static_cast<std::size_t>(effective_head_dim(config));
    const double inv_sqrt_hd = 1.0 / std::sqrt(static_cast<double>(hd));
    const double eps = config.rms_norm_eps;
    const std::string attn_mode = to_lower_trim(config.attention_type);

    if (step_trace != nullptr) {
        step_trace->residual.clear();
        step_trace->mlp_poles.clear();
        step_trace->residual.reserve(static_cast<std::size_t>(config.num_layers) + 1);
        step_trace->mlp_poles.reserve(static_cast<std::size_t>(config.num_layers));
        step_trace->residual.push_back(h);
    }

    for (int l = 0; l < config.num_layers; ++l) {
        LayerKVCache& lcache = cache.layers[static_cast<std::size_t>(l)];
        if (lcache.k.size() != pos || lcache.v.size() != pos) {
            throw std::invalid_argument("KVCache sequence length mismatch in forward_step");
        }

        const std::string prefix = "model.layers." + std::to_string(l);
        const Vector& in_ln = require_vector(model.vectors, prefix + ".input_layernorm.weight");
        const Vector x_norm = rms_norm(h, in_ln, eps);

        const Matrix& w_q = require_matrix(model.matrices, prefix + ".self_attn.q_proj.weight");
        const Matrix& w_o = require_matrix(model.matrices, prefix + ".self_attn.o_proj.weight");
        Vector q = mat_vec(w_q, x_norm);
        const auto qb_it = model.vectors.find(prefix + ".self_attn.q_proj.bias");
        if (qb_it != model.vectors.end()) {
            add_bias_inplace(q, qb_it->second);
        }

        Vector k;
        Vector v;
        const std::string kva_key = prefix + ".self_attn.kv_a_proj.weight";
        const std::string kvb_key = prefix + ".self_attn.kv_b_proj.weight";
        if (attn_mode == "mla" && model.matrices.contains(kva_key) && model.matrices.contains(kvb_key)) {
            const Matrix& w_kva = require_matrix(model.matrices, kva_key);
            const Matrix& w_kvb = require_matrix(model.matrices, kvb_key);
            const Vector c_kv = mat_vec(w_kva, x_norm);
            lcache.c_kv.push_back(c_kv);
            const Vector kv = mat_vec(w_kvb, c_kv);
            const std::size_t half = kv.size() / 2;
            k.assign(kv.begin(), kv.begin() + static_cast<std::ptrdiff_t>(half));
            v.assign(kv.begin() + static_cast<std::ptrdiff_t>(half), kv.end());
        } else {
            const Matrix& w_k = require_matrix(model.matrices, prefix + ".self_attn.k_proj.weight");
            const Matrix& w_v = require_matrix(model.matrices, prefix + ".self_attn.v_proj.weight");
            k = mat_vec(w_k, x_norm);
            v = mat_vec(w_v, x_norm);
            const auto kb_it = model.vectors.find(prefix + ".self_attn.k_proj.bias");
            const auto vb_it = model.vectors.find(prefix + ".self_attn.v_proj.bias");
            if (kb_it != model.vectors.end()) {
                add_bias_inplace(k, kb_it->second);
            }
            if (vb_it != model.vectors.end()) {
                add_bias_inplace(v, vb_it->second);
            }
        }

        const std::size_t kv_heads_eff = k.size() / hd;
        const std::size_t group_size = num_q_heads / kv_heads_eff;

        const auto qn_it = model.vectors.find(prefix + ".self_attn.q_norm.weight");
        const auto kn_it = model.vectors.find(prefix + ".self_attn.k_norm.weight");
        const Vector* q_norm_w = (qn_it != model.vectors.end()) ? &qn_it->second : nullptr;
        const Vector* k_norm_w = (kn_it != model.vectors.end()) ? &kn_it->second : nullptr;

        apply_head_norm_and_rope(
            q, num_q_heads, hd, q_norm_w, config.use_rope, pos, config.rope_theta, eps);
        apply_head_norm_and_rope(
            k, kv_heads_eff, hd, k_norm_w, config.use_rope, pos, config.rope_theta, eps);

        lcache.k.push_back(std::move(k));
        lcache.v.push_back(std::move(v));

        const std::size_t total_len = pos + 1;
        Vector concat_ctx(num_q_heads * hd, 0.0);
        for (std::size_t hq = 0; hq < num_q_heads; ++hq) {
            const std::size_t hkv = hq / group_size;
            const std::size_t q_start = hq * hd;
            const std::size_t kv_start = hkv * hd;

            Vector scores(total_len, 0.0);
            for (std::size_t j = 0; j < total_len; ++j) {
                double acc = 0.0;
                const Vector& kj = lcache.k[j];
                for (std::size_t m = 0; m < hd; ++m) {
                    acc += q[q_start + m] * kj[kv_start + m];
                }
                scores[j] = acc * inv_sqrt_hd;
            }
            const Vector probs = softmax(scores);
            for (std::size_t j = 0; j < total_len; ++j) {
                const double p = probs[j];
                const Vector& vj = lcache.v[j];
                for (std::size_t m = 0; m < hd; ++m) {
                    concat_ctx[q_start + m] += p * vj[kv_start + m];
                }
            }
        }

        const Vector attn_out = mat_vec(w_o, concat_ctx);
        const Vector h_mid = vec_add(h, attn_out);

        const Vector& post_ln = require_vector(model.vectors, prefix + ".post_attention_layernorm.weight");
        const Matrix& w_gate = require_matrix(model.matrices, prefix + ".mlp.gate_proj.weight");
        const Matrix& w_up = require_matrix(model.matrices, prefix + ".mlp.up_proj.weight");
        const Matrix& w_down = require_matrix(model.matrices, prefix + ".mlp.down_proj.weight");
        const std::size_t d_ff_actual = w_gate.rows;

        const Vector x_post = rms_norm(h_mid, post_ln, eps);
        const Vector g_pre = mat_vec(w_gate, x_post);
        const Vector u_val = mat_vec(w_up, x_post);
        Vector acts(d_ff_actual);
        for (std::size_t i = 0; i < d_ff_actual; ++i) {
            acts[i] = silu(g_pre[i]) * u_val[i];
        }
        Vector mlp_out = mat_vec(w_down, acts);
        if (model.config.aic_threshold_routes) {
            mlp_out = vec_add(mlp_out, threshold_route_delta(model, prefix, acts));
        }
        h = vec_add(h_mid, mlp_out);

        if (step_trace != nullptr) {
            step_trace->residual.push_back(h);
            step_trace->mlp_poles.push_back(PoleTrace{g_pre, u_val, acts});
        }
    }

    cache.seq_len += 1;

    const Vector& final_ln = require_vector(model.vectors, "model.norm.weight");
    const Matrix& w_lm_head = resolve_lm_head(model);
    const Vector h_final = rms_norm(h, final_ln, eps);
    return mat_vec(w_lm_head, h_final);
}

Vector TransformerModel::forward_step(int token_id, KVCache& cache) const {
    return forward_step_impl(*this, token_id, cache, nullptr);
}

ForwardStepTrace TransformerModel::forward_step_traced(int token_id, KVCache& cache) const {
    ForwardStepTrace out;
    out.logits = forward_step_impl(*this, token_id, cache, &out);
    return out;
}

std::vector<int> TransformerModel::generate_greedy(
    const std::vector<int>& prompt_ids,
    int max_new_tokens) const {
    if (prompt_ids.empty()) {
        throw std::invalid_argument("prompt_ids must be non-empty in generate_greedy");
    }
    if (max_new_tokens < 0) {
        throw std::invalid_argument("max_new_tokens must be non-negative in generate_greedy");
    }
    std::vector<int> ids = prompt_ids;
    if (max_new_tokens == 0) {
        return ids;
    }
    ids.reserve(prompt_ids.size() + static_cast<std::size_t>(max_new_tokens));

    KVCache cache = init_kv_cache();
    Vector last_logits = prefill_last_logits(*this, prompt_ids, cache);
    for (int step = 0; step < max_new_tokens; ++step) {
        const int next_id = argmax_index(last_logits);
        ids.push_back(next_id);
        if (step + 1 < max_new_tokens) {
            last_logits = forward_step(next_id, cache);
        }
    }
    return ids;
}

std::vector<int> TransformerModel::generate_sampled(
    const std::vector<int>& prompt_ids,
    int max_new_tokens,
    double temperature,
    double top_p,
    std::uint64_t seed) const {
    if (temperature <= 0.0) {
        return generate_greedy(prompt_ids, max_new_tokens);
    }
    if (prompt_ids.empty()) {
        throw std::invalid_argument("prompt_ids must be non-empty in generate_sampled");
    }
    if (max_new_tokens < 0) {
        throw std::invalid_argument("max_new_tokens must be non-negative in generate_sampled");
    }
    if (top_p <= 0.0) {
        throw std::invalid_argument("top_p must be positive in generate_sampled");
    }

    std::vector<int> ids = prompt_ids;
    if (max_new_tokens == 0) {
        return ids;
    }
    ids.reserve(prompt_ids.size() + static_cast<std::size_t>(max_new_tokens));

    std::mt19937_64 rng(seed);
    KVCache cache = init_kv_cache();
    Vector last_logits = prefill_last_logits(*this, prompt_ids, cache);
    for (int step = 0; step < max_new_tokens; ++step) {
        const int next_id = sample_nucleus(last_logits, temperature, top_p, rng);
        ids.push_back(next_id);
        if (step + 1 < max_new_tokens) {
            last_logits = forward_step(next_id, cache);
        }
    }
    return ids;
}

void TransformerModel::save_checkpoint(const std::string& dir) const {
    const std::filesystem::path out_dir(dir);
    std::filesystem::create_directories(out_dir);

    {
        const std::string cfg_path = (out_dir / "config.json").string();
        std::ofstream ofs(cfg_path, std::ios::trunc);
        if (!ofs) {
            throw std::runtime_error("Failed to open config.json for writing: " + cfg_path);
        }
        ofs << dump_json(config.to_json(), 2) << '\n';
        if (!ofs.good()) {
            throw std::runtime_error("Failed writing config.json: " + cfg_path);
        }
    }

    const std::string storage_dtype = normalize_storage_dtype(config.torch_dtype);
    std::map<std::string, TensorSpec> specs;
    for (const auto& [name, mat] : matrices) {
        specs[name] = TensorSpec::from_matrix(mat, storage_dtype);
    }
    for (const auto& [name, vec] : vectors) {
        specs[name] = TensorSpec::from_vector(vec, storage_dtype);
    }

    save_safetensors(
        (out_dir / "model.safetensors").string(),
        specs,
        {{"arch_name", config.arch_name}, {"attention_type", config.attention_type}});
}

namespace {

class StreamingCheckpointReader {
public:
    explicit StreamingCheckpointReader(const std::string& model_dir)
        : dir_(model_dir) {
        const std::filesystem::path index_path = dir_ / "model.safetensors.index.json";
        const std::filesystem::path single_path = dir_ / "model.safetensors";
        if (std::filesystem::exists(index_path)) {
            std::ifstream ifs(index_path);
            if (!ifs) {
                throw std::runtime_error("Failed to open safetensors index: " + index_path.string());
            }
            std::ostringstream oss;
            oss << ifs.rdbuf();
            const JsonValue index = parse_json(oss.str());
            if (!index.has("weight_map") || !index["weight_map"].is_object()) {
                throw std::runtime_error("Invalid safetensors index weight_map");
            }
            for (const auto& [name, shard] : index["weight_map"].as_object()) {
                if (!shard.is_string()) {
                    throw std::runtime_error("Invalid safetensors shard entry for " + name);
                }
                weight_map_[name] = shard.as_string();
            }
        } else if (std::filesystem::exists(single_path)) {
            MmapSafetensors probe(single_path.string());
            for (const std::string& name : probe.tensor_names()) {
                weight_map_[name] = "model.safetensors";
            }
        } else {
            throw std::runtime_error("No safetensors checkpoint in: " + model_dir);
        }
    }

    bool has(const std::string& name) const {
        return weight_map_.contains(name);
    }

    const std::map<std::string, std::string>& weight_map() const {
        return weight_map_;
    }

    TensorSpec read(const std::string& name) {
        return reader(name).read_tensor(name);
    }

    std::vector<std::size_t> shape(const std::string& name) {
        return reader(name).tensor_shape(name);
    }

    Vector row(const std::string& name, std::size_t row_idx) {
        return reader(name).read_matrix_row(name, row_idx);
    }

    Vector mat_vec(const std::string& name, const Vector& x) {
        return reader(name).mat_vec(name, x);
    }

private:
    MmapSafetensors& reader(const std::string& name) {
        auto wit = weight_map_.find(name);
        if (wit == weight_map_.end()) {
            throw std::out_of_range("Missing tensor in checkpoint: " + name);
        }
        auto rit = readers_.find(wit->second);
        if (rit == readers_.end()) {
            auto ptr = std::make_unique<MmapSafetensors>((dir_ / wit->second).string());
            rit = readers_.emplace(wit->second, std::move(ptr)).first;
        }
        return *rit->second;
    }

    std::filesystem::path dir_;
    std::map<std::string, std::string> weight_map_;
    std::map<std::string, std::unique_ptr<MmapSafetensors>> readers_;
};

std::vector<Vector> run_checkpoint_layers_streaming(
    StreamingCheckpointReader& checkpoint,
    const ModelConfig& cfg,
    const std::vector<int>& token_ids,
    ForwardTrace* trace) {
    const std::string embed_key = "model.embed_tokens.weight";
    if (!checkpoint.has(embed_key)) {
        throw std::runtime_error("Streaming checkpoint is missing " + embed_key);
    }
    const std::vector<std::size_t> embed_shape = checkpoint.shape(embed_key);
    if (embed_shape.size() != 2) {
        throw std::runtime_error("Embedding tensor must be rank 2");
    }
    const std::size_t vocab = embed_shape[0];
    std::vector<Vector> hidden;
    hidden.reserve(token_ids.size());
    for (int token_id : token_ids) {
        if (token_id < 0 || static_cast<std::size_t>(token_id) >= vocab) {
            throw std::out_of_range("Token ID out of range in streaming forward");
        }
        hidden.push_back(checkpoint.row(embed_key, static_cast<std::size_t>(token_id)));
    }
    if (hidden.empty()) {
        throw std::invalid_argument("Streaming forward requires at least one token");
    }

    if (trace != nullptr) {
        trace->residual.reserve(static_cast<std::size_t>(cfg.num_layers) + 1);
        trace->mlp_poles.reserve(static_cast<std::size_t>(cfg.num_layers));
        trace->residual.push_back(hidden);
    }

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const std::string source_prefix = "model.layers." + std::to_string(layer);
        const std::string local_prefix = "model.layers.0";
        TransformerModel local;
        local.config = cfg;
        local.config.num_layers = 1;
        for (const auto& [name, shard] : checkpoint.weight_map()) {
            (void)shard;
            if (!name.starts_with(source_prefix + ".")) {
                continue;
            }
            TensorSpec spec = checkpoint.read(name);
            const std::string local_name = local_prefix + name.substr(source_prefix.size());
            if (spec.shape.size() == 1) {
                local.vectors[local_name] = spec.to_vector();
            } else if (spec.shape.size() == 2) {
                local.matrices[local_name] = spec.to_matrix();
            }
        }
        auto [next_hidden, local_trace] = local.forward_hidden(hidden, trace != nullptr);
        hidden = std::move(next_hidden);
        if (trace != nullptr) {
            trace->residual.push_back(hidden);
            if (local_trace.mlp_poles.empty()) {
                trace->mlp_poles.emplace_back();
            } else {
                trace->mlp_poles.push_back(std::move(local_trace.mlp_poles[0]));
            }
        }
    }
    return hidden;
}

} // namespace

ModelConfig load_checkpoint_config(const std::string& model_dir) {
    const std::filesystem::path cfg_path = std::filesystem::path(model_dir) / "config.json";
    std::ifstream ifs(cfg_path);
    if (!ifs) {
        throw std::runtime_error("Failed to open config.json: " + cfg_path.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return ModelConfig::from_json(parse_json(oss.str()));
}

std::pair<std::vector<Vector>, ForwardTrace> forward_checkpoint_streaming(
    const std::string& model_dir,
    const std::vector<int>& token_ids,
    bool record_trace) {
    const ModelConfig cfg = load_checkpoint_config(model_dir);
    StreamingCheckpointReader checkpoint(model_dir);
    ForwardTrace trace;
    std::vector<Vector> hidden = run_checkpoint_layers_streaming(
        checkpoint, cfg, token_ids, record_trace ? &trace : nullptr);
    const Vector final_norm = checkpoint.read("model.norm.weight").to_vector();
    const std::string head_key = checkpoint.has("lm_head.weight")
        ? std::string("lm_head.weight")
        : std::string("model.embed_tokens.weight");
    std::vector<Vector> logits;
    logits.reserve(hidden.size());
    for (const Vector& h : hidden) {
        logits.push_back(checkpoint.mat_vec(
            head_key, rms_norm(h, final_norm, cfg.rms_norm_eps)));
    }
    return {std::move(logits), std::move(trace)};
}

ForwardTrace trace_checkpoint_streaming(
    const std::string& model_dir,
    const std::vector<int>& token_ids) {
    const ModelConfig cfg = load_checkpoint_config(model_dir);
    StreamingCheckpointReader checkpoint(model_dir);
    ForwardTrace trace;
    (void)run_checkpoint_layers_streaming(checkpoint, cfg, token_ids, &trace);
    return trace;
}

std::vector<int> generate_checkpoint_streaming(
    const std::string& model_dir,
    const std::vector<int>& prompt_ids,
    int max_new_tokens,
    double temperature,
    double top_p,
    std::uint64_t seed) {
    if (prompt_ids.empty()) {
        throw std::invalid_argument("Streaming generation requires a non-empty prompt");
    }
    if (max_new_tokens < 0 || top_p <= 0.0) {
        throw std::invalid_argument("Invalid streaming generation parameters");
    }
    std::vector<int> ids = prompt_ids;
    std::mt19937_64 rng(seed);
    for (int step = 0; step < max_new_tokens; ++step) {
        auto [logits, trace] = forward_checkpoint_streaming(model_dir, ids, false);
        (void)trace;
        const Vector& last = logits.back();
        int next = 0;
        if (temperature <= 0.0) {
            next = static_cast<int>(std::distance(last.begin(), std::max_element(last.begin(), last.end())));
        } else {
            next = sample_nucleus(last, temperature, top_p, rng);
        }
        ids.push_back(next);
    }
    return ids;
}

TransformerModel TransformerModel::load_checkpoint(const std::string& dir) {
    const std::filesystem::path in_dir(dir);
    const std::string cfg_path = (in_dir / "config.json").string();
    std::ifstream ifs(cfg_path);
    if (!ifs) {
        throw std::runtime_error("Failed to open config.json: " + cfg_path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();

    ModelConfig cfg = ModelConfig::from_json(parse_json(oss.str()));
    auto [specs, metadata] = load_safetensors_sharded(dir);
    (void)metadata;

    TransformerModel model;
    model.config = std::move(cfg);
    bool saw_qk_norm = false;
    bool saw_qkv_bias = false;
    for (const auto& [name, spec] : specs) {
        if (spec.shape.size() == 1) {
            if (ends_with(name, ".self_attn.q_norm.weight") || ends_with(name, ".self_attn.k_norm.weight")) {
                saw_qk_norm = true;
            }
            if (ends_with(name, ".self_attn.q_proj.bias") ||
                ends_with(name, ".self_attn.k_proj.bias") ||
                ends_with(name, ".self_attn.v_proj.bias")) {
                saw_qkv_bias = true;
            }
            model.vectors[name] = spec.to_vector();
        } else if (spec.shape.size() == 2) {
            model.matrices[name] = spec.to_matrix();
        } else {
            throw std::runtime_error("Unsupported tensor rank in checkpoint for: " + name);
        }
    }
    if (saw_qk_norm) {
        model.config.qk_norm = true;
    }
    if (saw_qkv_bias) {
        model.config.qkv_bias = true;
    }
    if (!model.matrices.contains("lm_head.weight") &&
        model.matrices.contains("model.embed_tokens.weight")) {
        model.config.tie_word_embeddings = true;
    }
    return model;
}

void TransformerModel::apply_lora_adapter(
    const std::string& lora_dir,
    double scale_override) {
    const std::filesystem::path dir(lora_dir);
    const std::string cfg_path = (dir / "adapter_config.json").string();
    std::ifstream ifs(cfg_path);
    if (!ifs) {
        throw std::runtime_error("Failed to open adapter_config.json: " + cfg_path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    const JsonValue cfg_json = parse_json(oss.str());
    if (!cfg_json.is_object()) {
        throw std::runtime_error("adapter_config.json must be a JSON object");
    }
    const int r_val = cfg_json.has("r") && cfg_json["r"].is_number()
        ? static_cast<int>(cfg_json["r"].as_int())
        : 8;
    const double alpha_val = cfg_json.has("lora_alpha") && cfg_json["lora_alpha"].is_number()
        ? cfg_json["lora_alpha"].as_number()
        : static_cast<double>(std::max(1, r_val));
    const double scale = (scale_override >= 0.0)
        ? scale_override
        : (alpha_val / static_cast<double>(std::max(1, r_val)));

    auto [specs, meta] = load_safetensors((dir / "adapter_model.safetensors").string());
    (void)meta;

    struct LoraPair {
        bool has_a = false;
        bool has_b = false;
        Matrix a;
        Matrix b;
    };
    std::map<std::string, LoraPair> pairs;

    auto strip_suffix = [](const std::string& s, const std::string& suf, std::string& stem_out) -> bool {
        if (s.size() <= suf.size()) {
            return false;
        }
        if (s.compare(s.size() - suf.size(), suf.size(), suf) != 0) {
            return false;
        }
        stem_out = s.substr(0, s.size() - suf.size());
        return true;
    };

    for (const auto& [name, spec] : specs) {
        std::string stem;
        if (strip_suffix(name, ".lora_A.weight", stem) ||
            strip_suffix(name, ".lora_A.default.weight", stem)) {
            pairs[stem].has_a = true;
            pairs[stem].a = spec.to_matrix();
        } else if (strip_suffix(name, ".lora_B.weight", stem) ||
                   strip_suffix(name, ".lora_B.default.weight", stem)) {
            pairs[stem].has_b = true;
            pairs[stem].b = spec.to_matrix();
        }
    }

    auto resolve_target_key = [&](const std::string& stem) -> std::string {
        const std::string peft_prefix = "base_model.model.";
        if (stem.rfind(peft_prefix, 0) == 0) {
            const std::string stripped = stem.substr(peft_prefix.size());
            if (matrices.contains(stripped + ".weight")) {
                return stripped + ".weight";
            }
            if (matrices.contains("model." + stripped + ".weight")) {
                return "model." + stripped + ".weight";
            }
        }
        if (matrices.contains(stem + ".weight")) {
            return stem + ".weight";
        }
        if (matrices.contains("model." + stem + ".weight")) {
            return "model." + stem + ".weight";
        }
        throw std::runtime_error("LoRA target weight not found in model: " + stem);
    };

    for (const auto& [stem, pair] : pairs) {
        if (!pair.has_a || !pair.has_b) {
            throw std::runtime_error("Incomplete LoRA A/B pair for: " + stem);
        }
        const std::string target_key = resolve_target_key(stem);
        Matrix& target = matrices.at(target_key);
        if (pair.a.rows != pair.b.cols ||
            pair.b.rows != target.rows ||
            pair.a.cols != target.cols) {
            throw std::invalid_argument("LoRA matrix shape mismatch for: " + target_key);
        }
        const Matrix delta = mat_scale(mat_mul(pair.b, pair.a), scale);
        for (std::size_t i = 0; i < target.data.size(); ++i) {
            target.data[i] += delta.data[i];
        }
    }
}

TransformerModel TransformerModel::load_with_lora(
    const std::string& base_dir,
    const std::string& lora_dir,
    double scale_override) {
    TransformerModel model = load_checkpoint(base_dir);
    model.apply_lora_adapter(lora_dir, scale_override);
    return model;
}

ModelDivergenceStats compute_model_divergence(
    const TransformerModel& p_model,
    const TransformerModel& q_model,
    const std::vector<std::vector<int>>& token_sequences) {
    ModelDivergenceStats stats;
    double sum_kl = 0.0;
    std::size_t token_count = 0;

    for (const auto& seq : token_sequences) {
        auto [logits_p, tr_p] = p_model.forward(seq, {}, false);
        auto [logits_q, tr_q] = q_model.forward(seq, {}, false);
        (void)tr_p;
        (void)tr_q;

        if (logits_p.size() != logits_q.size()) {
            throw std::invalid_argument("Sequence length mismatch in compute_model_divergence");
        }
        for (std::size_t t = 0; t < logits_p.size(); ++t) {
            const Vector& lp = logits_p[t];
            const Vector& lq = logits_q[t];
            if (lp.size() != lq.size()) {
                throw std::invalid_argument("Vocab size mismatch in compute_model_divergence");
            }
            for (std::size_t v = 0; v < lp.size(); ++v) {
                stats.max_logit_diff = std::max(stats.max_logit_diff, std::abs(lp[v] - lq[v]));
            }
            const Vector p = softmax(lp);
            const Vector q = softmax(lq);
            double kl = 0.0;
            for (std::size_t v = 0; v < p.size(); ++v) {
                if (p[v] > 0.0) {
                    const double pv = std::max(p[v], 1e-30);
                    const double qv = std::max(q[v], 1e-30);
                    kl += p[v] * (std::log(pv) - std::log(qv));
                }
            }
            if (kl < 0.0) {
                kl = 0.0;
            }
            sum_kl += kl;
            stats.max_kl = std::max(stats.max_kl, kl);
            ++token_count;
        }
    }

    if (token_count > 0) {
        stats.mean_kl = sum_kl / static_cast<double>(token_count);
    }
    return stats;
}

std::vector<Matrix> compute_mlp_key_covariances(
    const TransformerModel& model,
    const std::vector<std::vector<int>>& calibration_sequences,
    double ridge_lambda) {
    if (ridge_lambda <= 0.0) {
        throw std::invalid_argument("ridge_lambda must be positive in compute_mlp_key_covariances");
    }
    const int L = model.config.num_layers;
    if (L <= 0) {
        return {};
    }

    std::vector<std::size_t> d_ff_per_layer(static_cast<std::size_t>(L));
    for (int l = 0; l < L; ++l) {
        const std::string gate_key = "model.layers." + std::to_string(l) + ".mlp.gate_proj.weight";
        const auto it = model.matrices.find(gate_key);
        d_ff_per_layer[static_cast<std::size_t>(l)] = (it != model.matrices.end())
            ? it->second.rows
            : static_cast<std::size_t>(model.config.intermediate_size);
    }

    std::vector<std::vector<Vector>> per_layer_rows(static_cast<std::size_t>(L));
    std::size_t total_tokens = 0;

    for (const auto& seq : calibration_sequences) {
        if (seq.empty()) {
            continue;
        }
        auto [logits, trace] = model.forward(seq, {}, true);
        (void)logits;
        const std::size_t seq_len = seq.size();
        total_tokens += seq_len;
        for (int l = 0; l < L; ++l) {
            const std::size_t ul = static_cast<std::size_t>(l);
            const std::size_t d_ff = d_ff_per_layer[ul];
            for (std::size_t t = 0; t < seq_len; ++t) {
                Vector k_vec(d_ff, 0.0);
                for (std::size_t i = 0; i < d_ff; ++i) {
                    k_vec[i] = trace.mlp_poles[ul][i].act[t];
                }
                per_layer_rows[ul].push_back(std::move(k_vec));
            }
        }
    }

    std::vector<Matrix> out;
    out.reserve(static_cast<std::size_t>(L));
    for (int l = 0; l < L; ++l) {
        const std::size_t ul = static_cast<std::size_t>(l);
        const std::size_t d_ff = d_ff_per_layer[ul];
        if (total_tokens == 0) {
            out.push_back(mat_scale(eye(d_ff), 1.0 / ridge_lambda));
            continue;
        }
        const Matrix K = pack_rows(per_layer_rows[ul], d_ff);
        Matrix C = mat_scale(mat_mul(transpose(K), K), 1.0 / static_cast<double>(total_tokens));
        for (std::size_t i = 0; i < d_ff; ++i) {
            C(i, i) += ridge_lambda;
        }
        Matrix C_inv = mat_inv(C);
        for (std::size_t r = 0; r < d_ff; ++r) {
            for (std::size_t c = r + 1; c < d_ff; ++c) {
                const double sym = 0.5 * (C_inv(r, c) + C_inv(c, r));
                C_inv(r, c) = sym;
                C_inv(c, r) = sym;
            }
        }
        out.push_back(std::move(C_inv));
    }
    return out;
}

std::vector<LayerFisherDiag> compute_fisher_diagonals(
    const TransformerModel& model,
    const std::vector<std::vector<int>>& calibration_sequences,
    double eps_floor) {
    if (eps_floor <= 0.0) {
        throw std::invalid_argument("eps_floor must be positive in compute_fisher_diagonals");
    }
    const int L = model.config.num_layers;
    const std::size_t d = static_cast<std::size_t>(model.config.hidden_size);
    if (L <= 0) {
        return {};
    }

    std::size_t total_tokens = 0;
    for (const auto& seq : calibration_sequences) {
        total_tokens += seq.size();
    }
    if (total_tokens == 0) {
        return std::vector<LayerFisherDiag>(
            static_cast<std::size_t>(L),
            LayerFisherDiag{Vector(d, 1.0), Vector(d, 1.0)});
    }

    const Matrix& w_lm = resolve_lm_head(model);
    const Matrix w_lm_t = transpose(w_lm);
    std::vector<const Vector*> w_in_ln(static_cast<std::size_t>(L));
    std::vector<Matrix> w_o_t(static_cast<std::size_t>(L));
    for (int l = 0; l < L; ++l) {
        const std::size_t ul = static_cast<std::size_t>(l);
        const std::string prefix = "model.layers." + std::to_string(l);
        w_in_ln[ul] = &require_vector(model.vectors, prefix + ".input_layernorm.weight");
        w_o_t[ul] = transpose(require_matrix(model.matrices, prefix + ".self_attn.o_proj.weight"));
    }

    const double inv_n = 1.0 / static_cast<double>(total_tokens);
    const double eps = model.config.rms_norm_eps;
    std::vector<LayerFisherDiag> result(
        static_cast<std::size_t>(L),
        LayerFisherDiag{Vector(d, 0.0), Vector(d, 0.0)});

    for (const auto& seq : calibration_sequences) {
        if (seq.empty()) {
            continue;
        }
        auto [logits, trace] = model.forward(seq, {}, true);
        const std::size_t seq_len = seq.size();
        for (std::size_t t = 0; t < seq_len; ++t) {
            for (int l = 0; l < L; ++l) {
                const std::size_t ul = static_cast<std::size_t>(l);
                const Vector x_in = rms_norm(trace.residual[ul][t], *w_in_ln[ul], eps);
                for (std::size_t j = 0; j < d; ++j) {
                    result[ul].g_in[j] += (x_in[j] * x_in[j]) * inv_n;
                }
            }

            const Vector p = softmax(logits[t]);
            const Vector w_bar = mat_vec(w_lm_t, p);
            Vector f_L(d, 0.0);
            const std::size_t vocab = w_lm.rows;
            for (std::size_t v = 0; v < vocab; ++v) {
                const double pv = p[v];
                const std::size_t row_base = v * d;
                for (std::size_t i = 0; i < d; ++i) {
                    const double diff = w_lm.data[row_base + i] - w_bar[i];
                    f_L[i] += pv * (diff * diff);
                }
            }

            Vector s = std::move(f_L);
            for (int l = L - 1; l >= 0; --l) {
                const std::size_t ul = static_cast<std::size_t>(l);
                for (std::size_t i = 0; i < d; ++i) {
                    result[ul].g_out[i] += s[i] * inv_n;
                }
                if (l > 0) {
                    Vector sqrt_s(d);
                    for (std::size_t i = 0; i < d; ++i) {
                        sqrt_s[i] = std::sqrt(std::max(0.0, s[i]));
                    }
                    const Vector wo_t_s = mat_vec(w_o_t[ul], sqrt_s);
                    const double wo_energy = dot(wo_t_s, wo_t_s) / static_cast<double>(d);
                    const Vector& ln_w = *w_in_ln[ul];
                    Vector s_prev(d);
                    for (std::size_t j = 0; j < d; ++j) {
                        s_prev[j] = s[j] + wo_energy * (ln_w[j] * ln_w[j]);
                    }
                    s = std::move(s_prev);
                }
            }
        }
    }

    auto normalize_diag = [&](Vector& v) {
        if (v.empty()) {
            return;
        }
        for (double& x : v) {
            if (!std::isfinite(x) || x < eps_floor) {
                x = eps_floor;
            }
        }
        double sum = 0.0;
        for (double x : v) {
            sum += x;
        }
        const double scale = static_cast<double>(v.size()) / sum;
        for (double& x : v) {
            x *= scale;
            if (x < eps_floor) {
                x = eps_floor;
            }
        }
    };

    for (auto& diag : result) {
        normalize_diag(diag.g_in);
        normalize_diag(diag.g_out);
    }
    return result;
}

}
