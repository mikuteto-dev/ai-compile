#pragma once

#include "aic/json_min.hpp"
#include "aic/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace aic {

struct ModelConfig {
    int num_layers = 8;
    int hidden_size = 32;
    int intermediate_size = 96;
    int num_attention_heads = 4;
    int num_key_value_heads = 2;
    int vocab_size = 64;
    std::string attention_type = "gqa";
    int kv_lora_rank = 8;
    double rms_norm_eps = 1e-6;
    std::string arch_name = "qwen3-mini";
    std::string model_type;
    int head_dim = 0;
    bool use_rope = false;
    double rope_theta = 10000.0;
    bool tie_word_embeddings = false;
    bool qk_norm = false;
    bool qkv_bias = false;
    bool aic_threshold_routes = false;
    std::string torch_dtype = "F32";

    std::pair<int, int> stage_to_layers(
        const std::string& start,
        const std::string& end = "",
        bool inclusive = false) const;
    JsonValue to_json() const;
    static ModelConfig from_json(const JsonValue& j);
};

struct PoleTrace {
    Vector gate_pre;
    Vector up_val;
    Vector act;
};

struct ForwardTrace {
    std::vector<std::vector<Vector>> residual;
    std::vector<std::vector<PoleTrace>> mlp_poles;
};

struct ForwardStepTrace {
    Vector logits;
    std::vector<Vector> residual;
    std::vector<PoleTrace> mlp_poles;
};

struct LayerKVCache {
    std::vector<Vector> k;
    std::vector<Vector> v;
    std::vector<Vector> c_kv;
};

struct KVCache {
    std::vector<LayerKVCache> layers;
    std::size_t seq_len = 0;
};

struct ModelDivergenceStats {
    double mean_kl = 0.0;
    double max_kl = 0.0;
    double max_logit_diff = 0.0;
};

class TransformerModel {
public:
    ModelConfig config;
    std::map<std::string, Matrix> matrices;
    std::map<std::string, Vector> vectors;

    static TransformerModel create_synthetic(ModelConfig cfg = {}, std::uint64_t seed = 42);

    std::pair<std::vector<Vector>, ForwardTrace> forward(
        const std::vector<int>& token_ids,
        const std::vector<Vector>& hidden_inputs = {},
        bool record_trace = false) const;

    std::pair<std::vector<Vector>, ForwardTrace> forward_hidden(
        const std::vector<Vector>& hidden_inputs,
        bool record_trace = false) const;

    KVCache init_kv_cache() const;
    Vector forward_step(int token_id, KVCache& cache) const;
    ForwardStepTrace forward_step_traced(int token_id, KVCache& cache) const;

    std::vector<int> generate_greedy(
        const std::vector<int>& prompt_ids,
        int max_new_tokens) const;

    std::vector<int> generate_sampled(
        const std::vector<int>& prompt_ids,
        int max_new_tokens,
        double temperature = 0.0,
        double top_p = 1.0,
        std::uint64_t seed = 42) const;

    void save_checkpoint(const std::string& dir) const;
    static TransformerModel load_checkpoint(const std::string& dir);

    void apply_lora_adapter(const std::string& lora_dir, double scale_override = -1.0);
    static TransformerModel load_with_lora(
        const std::string& base_dir,
        const std::string& lora_dir,
        double scale_override = -1.0);
};

ModelConfig load_checkpoint_config(const std::string& model_dir);

std::pair<std::vector<Vector>, ForwardTrace> forward_checkpoint_streaming(
    const std::string& model_dir,
    const std::vector<int>& token_ids,
    bool record_trace = false);

ForwardTrace trace_checkpoint_streaming(
    const std::string& model_dir,
    const std::vector<int>& token_ids);

std::vector<int> generate_checkpoint_streaming(
    const std::string& model_dir,
    const std::vector<int>& prompt_ids,
    int max_new_tokens,
    double temperature = 0.0,
    double top_p = 1.0,
    std::uint64_t seed = 42);

ModelDivergenceStats compute_model_divergence(
    const TransformerModel& p_model,
    const TransformerModel& q_model,
    const std::vector<std::vector<int>>& token_sequences);

std::vector<Matrix> compute_mlp_key_covariances(
    const TransformerModel& model,
    const std::vector<std::vector<int>>& calibration_sequences,
    double ridge_lambda = 1e-2);

struct LayerFisherDiag {
    Vector g_in;
    Vector g_out;
};

std::vector<LayerFisherDiag> compute_fisher_diagonals(
    const TransformerModel& model,
    const std::vector<std::vector<int>>& calibration_sequences,
    double eps_floor = 1e-4);

}
