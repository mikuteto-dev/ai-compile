#include "aic/tracer.hpp"

#include "aic/errors.hpp"
#include "aic/linker.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace aic {

namespace {

double vector_rms(const Vector& v) {
    if (v.empty()) {
        return 0.0;
    }
    const double n = norm2(v);
    return n / std::sqrt(static_cast<double>(v.size()));
}

int argmax_token(const Vector& logits) {
    if (logits.empty()) {
        return 0;
    }
    int best_id = 0;
    double best_val = logits[0];
    for (std::size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_id = static_cast<int>(i);
        }
    }
    return best_id;
}

std::vector<double> compute_layer_residual_rms(const ForwardTrace& trace) {
    std::vector<double> rms;
    rms.reserve(trace.residual.size());
    for (const auto& layer_vecs : trace.residual) {
        if (layer_vecs.empty()) {
            rms.push_back(0.0);
            continue;
        }
        double sum_rms = 0.0;
        for (const Vector& v : layer_vecs) {
            sum_rms += vector_rms(v);
        }
        rms.push_back(sum_rms / static_cast<double>(layer_vecs.size()));
    }
    return rms;
}

std::pair<int, int> resolve_symbol_layer_bounds(
    const SymbolBinding& sb,
    int num_layers,
    const std::string& preferred_pole = "") {
    if (num_layers <= 0) {
        return {0, 0};
    }
    int l0 = std::clamp(sb.layer_range.first, 0, num_layers - 1);
    int l1 = std::clamp(std::max(l0 + 1, sb.layer_range.second), l0 + 1, num_layers);

    if ((sb.kind == "pole" || sb.kind == "dipole" || !preferred_pole.empty()) &&
        !sb.pole_slots.empty()) {
        auto pit = !preferred_pole.empty() ? sb.pole_slots.find(preferred_pole) : sb.pole_slots.end();
        if (pit == sb.pole_slots.end()) {
            pit = sb.pole_slots.find("+pole");
        }
        if (pit == sb.pole_slots.end()) {
            pit = sb.pole_slots.begin();
        }
        const int bl = std::get<0>(pit->second);
        if (bl >= 0 && bl < num_layers) {
            l0 = bl;
            l1 = bl + 1;
        }
    }
    return {l0, l1};
}

int find_pole_slot(const SymbolBinding& sb, const std::string& preferred_pole = "") {
    if (sb.pole_slots.empty()) {
        return -1;
    }
    auto pit = !preferred_pole.empty() ? sb.pole_slots.find(preferred_pole) : sb.pole_slots.end();
    if (pit == sb.pole_slots.end()) {
        pit = sb.pole_slots.find("+pole");
    }
    if (pit == sb.pole_slots.end()) {
        pit = sb.pole_slots.begin();
    }
    return std::get<1>(pit->second);
}

struct LayerProjectionSummary {
    double mean_proj = 0.0;
    double max_proj = 0.0;
    double active_ratio = 0.0;
};

LayerProjectionSummary score_symbol_layer(
    const SymbolBinding& sb,
    const Vector& norm_dir,
    const Vector* vecs,
    std::size_t seq_len,
    const double* pole_acts,
    std::size_t pole_act_len,
    double rms_ref) {
    LayerProjectionSummary summary;
    if (vecs == nullptr || seq_len == 0) {
        return summary;
    }
    const std::size_t d = vecs[0].size();
    const double thresh = std::max(0.05, 0.10 * rms_ref);
    std::vector<double> projs(seq_len, 0.0);

    if ((sb.kind == "subspace" || sb.kind == "manifold") && !sb.subspace_basis.empty()) {
        const auto& basis = (sb.kind == "manifold" && sb.manifold_plane.size() >= 2)
            ? sb.manifold_plane
            : sb.subspace_basis;
        for (std::size_t t = 0; t < seq_len; ++t) {
            double sq_sum = 0.0;
            for (const Vector& b : basis) {
                if (b.size() == d) {
                    const double p = dot(vecs[t], b);
                    sq_sum += p * p;
                }
            }
            projs[t] = std::sqrt(sq_sum);
        }
    } else if (sb.kind == "field" && !sb.field_modes.empty()) {
        for (std::size_t t = 0; t < seq_len; ++t) {
            double sq_sum = 0.0;
            for (const auto& fm : sb.field_modes) {
                if (fm.u.size() == d) {
                    const double p = dot(vecs[t], fm.u);
                    sq_sum += p * p;
                }
            }
            projs[t] = std::sqrt(sq_sum);
        }
    } else if (sb.kind == "rodata" && !sb.preserve_keys.empty()) {
        for (std::size_t t = 0; t < seq_len; ++t) {
            double best_k = 0.0;
            for (const Vector& kv : sb.preserve_keys) {
                if (kv.size() == d) {
                    best_k = std::max(best_k, std::abs(dot(vecs[t], kv)));
                }
            }
            projs[t] = best_k;
        }
    } else if (norm_dir.size() == d) {
        for (std::size_t t = 0; t < seq_len; ++t) {
            projs[t] = dot(vecs[t], norm_dir);
        }
    } else if (pole_acts != nullptr) {
        for (std::size_t t = 0; t < seq_len && t < pole_act_len; ++t) {
            projs[t] = pole_acts[t];
        }
    }

    double sum_p = 0.0;
    double max_abs_p = 0.0;
    std::size_t active_cnt = 0;
    for (double p : projs) {
        sum_p += p;
        const double ap = std::abs(p);
        max_abs_p = std::max(max_abs_p, ap);
        if (ap > thresh) {
            ++active_cnt;
        }
    }

    summary.mean_proj = sum_p / static_cast<double>(seq_len);
    summary.max_proj = max_abs_p;
    summary.active_ratio = static_cast<double>(active_cnt) / static_cast<double>(seq_len);
    return summary;
}

SymbolActivation eval_symbol_on_trace(
    const SymbolBinding& sb,
    const ForwardTrace& trace,
    const std::vector<double>& layer_rms,
    int num_layers) {
    SymbolActivation act;
    act.symbol = sb.name;
    act.kind = sb.kind;

    if (num_layers <= 0 || trace.residual.size() <= 1) {
        return act;
    }

    const auto [l0, l1] = resolve_symbol_layer_bounds(sb, num_layers);
    const Vector norm_dir = (norm2(sb.direction) > 1e-12) ? normalize(sb.direction) : Vector{};
    const int bs = find_pole_slot(sb);

    double best_score = -1.0;
    bool initialized = false;

    for (int l = l0; l < l1; ++l) {
        const std::size_t res_idx = static_cast<std::size_t>(l + 1);
        if (res_idx >= trace.residual.size() || trace.residual[res_idx].empty()) {
            continue;
        }
        const auto& seq_vecs = trace.residual[res_idx];
        const double rms_ref = (res_idx < layer_rms.size()) ? layer_rms[res_idx] : 1.0;
        const double* pole_acts = nullptr;
        std::size_t pole_act_len = 0;
        if (bs >= 0 && static_cast<std::size_t>(l) < trace.mlp_poles.size()) {
            const auto& layer_poles = trace.mlp_poles[static_cast<std::size_t>(l)];
            if (static_cast<std::size_t>(bs) < layer_poles.size()) {
                pole_acts = layer_poles[static_cast<std::size_t>(bs)].act.data();
                pole_act_len = layer_poles[static_cast<std::size_t>(bs)].act.size();
            }
        }

        const LayerProjectionSummary s = score_symbol_layer(
            sb,
            norm_dir,
            seq_vecs.data(),
            seq_vecs.size(),
            pole_acts,
            pole_act_len,
            rms_ref);
        if (!initialized || s.max_proj > best_score) {
            best_score = s.max_proj;
            initialized = true;
            act.layer = l;
            act.mean_proj = s.mean_proj;
            act.max_proj = s.max_proj;
            act.active_ratio = s.active_ratio;
        }
    }

    return act;
}

SymbolActivation eval_symbol_on_step_trace(
    const SymbolBinding& sb,
    const ForwardStepTrace& step_trace,
    int num_layers) {
    SymbolActivation act;
    act.symbol = sb.name;
    act.kind = sb.kind;

    if (num_layers <= 0 || step_trace.residual.size() <= 1) {
        return act;
    }

    const auto [l0, l1] = resolve_symbol_layer_bounds(sb, num_layers);
    const Vector norm_dir = (norm2(sb.direction) > 1e-12) ? normalize(sb.direction) : Vector{};
    const int bs = find_pole_slot(sb);

    double best_score = -1.0;
    bool initialized = false;

    for (int l = l0; l < l1; ++l) {
        const std::size_t res_idx = static_cast<std::size_t>(l + 1);
        if (res_idx >= step_trace.residual.size() || step_trace.residual[res_idx].empty()) {
            continue;
        }
        const Vector& res_vec = step_trace.residual[res_idx];
        const double rms_ref = vector_rms(res_vec);
        double single_pole_act = 0.0;
        const double* pole_acts = nullptr;
        std::size_t pole_act_len = 0;
        if (bs >= 0 && static_cast<std::size_t>(l) < step_trace.mlp_poles.size()) {
            const auto& layer_act = step_trace.mlp_poles[static_cast<std::size_t>(l)].act;
            if (static_cast<std::size_t>(bs) < layer_act.size()) {
                single_pole_act = layer_act[static_cast<std::size_t>(bs)];
                pole_acts = &single_pole_act;
                pole_act_len = 1;
            }
        }

        const LayerProjectionSummary s = score_symbol_layer(
            sb,
            norm_dir,
            &res_vec,
            1,
            pole_acts,
            pole_act_len,
            rms_ref);
        if (!initialized || s.max_proj > best_score) {
            best_score = s.max_proj;
            initialized = true;
            act.layer = l;
            act.mean_proj = s.mean_proj;
            act.max_proj = s.max_proj;
            act.active_ratio = s.active_ratio;
        }
    }

    return act;
}

NeuralAtlas resolve_effective_atlas(
    const TransformerModel& model,
    const NeuralAtlas& atlas,
    const MIRModule* mir,
    const Tokenizer* tokenizer = nullptr) {
    NeuralAtlas effective = atlas;
    if (mir == nullptr) {
        return effective;
    }
    bool has_missing = false;
    for (const auto& [name, decl] : mir->declarations) {
        (void)decl;
        if (!effective.symbols.contains(name)) {
            has_missing = true;
            break;
        }
    }
    if (!has_missing) {
        for (const auto& tc : mir->tests) {
            for (const auto& exp : tc.expectations) {
                if (exp.kind == "resonance" && !exp.symbol.empty() &&
                    !effective.symbols.contains(exp.symbol)) {
                    has_missing = true;
                    break;
                }
            }
            if (has_missing) {
                break;
            }
        }
    }
    if (has_missing) {
        effective = resolve_symbols_for_target(*mir, model, &atlas, 0, tokenizer);
    }
    return effective;
}

std::vector<std::string> collect_ordered_symbol_names(
    const NeuralAtlas& effective_atlas,
    const MIRModule* mir) {
    std::vector<std::string> ordered_names;
    std::set<std::string> seen;
    if (mir != nullptr) {
        for (const auto& [name, decl] : mir->declarations) {
            (void)decl;
            if (effective_atlas.symbols.contains(name) && seen.insert(name).second) {
                ordered_names.push_back(name);
            }
        }
    }
    for (const auto& [name, sb] : effective_atlas.symbols) {
        (void)sb;
        if (seen.insert(name).second) {
            ordered_names.push_back(name);
        }
    }
    return ordered_names;
}

std::vector<int> tokenize_contract_string(
    const std::string& text,
    const Tokenizer* tokenizer,
    int vocab_size) {
    const int safe_vocab = std::max(1, vocab_size);
    std::vector<int> ids;
    if (tokenizer != nullptr) {
        ids = tokenizer->encode(text);
    } else {
        ids.reserve(text.size());
        for (unsigned char c : text) {
            ids.push_back(static_cast<int>(c) % safe_vocab);
        }
    }
    if (ids.empty()) {
        ids.push_back(0);
    }
    for (int& id : ids) {
        if (id < 0 || id >= safe_vocab) {
            id = ((id % safe_vocab) + safe_vocab) % safe_vocab;
        }
    }
    return ids;
}

double measure_resonance_on_trace(
    const SymbolBinding& sb,
    const std::string& pole_side,
    const ForwardTrace& trace,
    const std::vector<double>& layer_rms,
    int num_layers) {
    if (sb.kind == "pole" || sb.kind == "dipole" || !pole_side.empty()) {
        if (num_layers <= 0 || trace.residual.size() <= 1) {
            return 0.0;
        }
        const double pol_sign = (pole_side == "-pole") ? -1.0 : 1.0;
        const auto [l0, l1] = resolve_symbol_layer_bounds(sb, num_layers, pole_side);
        const Vector dir = (norm2(sb.direction) > 1e-12) ? normalize(sb.direction) : Vector{};
        const int slot = find_pole_slot(sb, pole_side);

        double best_val = 0.0;
        for (int l = l0; l < l1; ++l) {
            const std::size_t res_idx = static_cast<std::size_t>(l + 1);
            if (res_idx >= trace.residual.size() || trace.residual[res_idx].empty()) {
                continue;
            }
            const auto& seq_vecs = trace.residual[res_idx];
            const std::size_t d = seq_vecs[0].size();
            if (dir.size() == d) {
                for (const Vector& v : seq_vecs) {
                    best_val = std::max(best_val, std::max(0.0, pol_sign * dot(v, dir)));
                }
            } else if (slot >= 0 && static_cast<std::size_t>(l) < trace.mlp_poles.size()) {
                const auto& layer_poles = trace.mlp_poles[static_cast<std::size_t>(l)];
                if (static_cast<std::size_t>(slot) < layer_poles.size()) {
                    for (double a : layer_poles[static_cast<std::size_t>(slot)].act) {
                        best_val = std::max(best_val, std::max(0.0, pol_sign * a));
                    }
                }
            }
        }
        return best_val;
    }

    return eval_symbol_on_trace(sb, trace, layer_rms, num_layers).max_proj;
}

bool evaluate_comparison(double measured, const std::string& cmp_op, double rhs) {
    constexpr double kTol = 1e-9;
    if (cmp_op == "<") {
        return measured < rhs;
    }
    if (cmp_op == "<=") {
        return measured <= rhs;
    }
    if (cmp_op == ">") {
        return measured > rhs;
    }
    if (cmp_op == ">=") {
        return measured >= rhs;
    }
    if (cmp_op == "==") {
        return std::abs(measured - rhs) <= kTol;
    }
    if (cmp_op == "!=") {
        return std::abs(measured - rhs) > kTol;
    }
    throw ParseError("Unsupported comparison operator in test expectation: " + cmp_op);
}

std::string format_rhs_with_base(bool rhs_is_base, double rhs_val) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    if (rhs_is_base) {
        oss << "base";
        if (rhs_val > 0.0) {
            oss << " + " << rhs_val;
        } else if (rhs_val < 0.0) {
            oss << " - " << (-rhs_val);
        }
    } else {
        oss << rhs_val;
    }
    return oss.str();
}

std::string format_expectation_description(const MIRExpectation& exp) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    if (exp.kind == "resonance") {
        oss << "resonance(\"" << exp.prompt << "\", " << exp.symbol;
        if (!exp.pole.empty()) {
            oss << "." << exp.pole;
        }
        oss << ") " << exp.cmp_op << " " << format_rhs_with_base(exp.rhs_is_base, exp.rhs_val);
        return oss.str();
    }
    if (exp.kind == "logit_margin") {
        oss << "logit_margin(\"" << exp.prompt << "\", \"" << exp.target_token_str
            << "\", \"" << exp.foil_token_str << "\") " << exp.cmp_op << " "
            << format_rhs_with_base(exp.rhs_is_base, exp.rhs_val);
        return oss.str();
    }
    if (exp.kind == "top_token") {
        oss << "top_token(\"" << exp.prompt << "\") " << exp.cmp_op
            << " \"" << exp.target_token_str << "\"";
        return oss.str();
    }
    if (exp.kind == "kl_drift") {
        if (exp.prompts.size() <= 1) {
            const std::string p = exp.prompts.empty() ? exp.prompt : exp.prompts[0];
            oss << "kl_drift(\"" << p << "\") " << exp.cmp_op << " " << exp.rhs_val;
        } else {
            oss << "kl_drift(" << exp.prompts.size() << " prompts) "
                << exp.cmp_op << " " << exp.rhs_val;
        }
        return oss.str();
    }
    oss << exp.kind << " " << exp.cmp_op << " " << exp.rhs_val;
    return oss.str();
}

}

ExecutionTraceReport trace_model_execution(
    const TransformerModel& model,
    const NeuralAtlas& atlas,
    const std::vector<int>& token_ids,
    const MIRModule* mir) {
    if (token_ids.empty()) {
        throw std::invalid_argument("trace_model_execution requires non-empty token_ids");
    }

    auto [logits, trace] = model.forward(token_ids, {}, true);

    ExecutionTraceReport rep;
    rep.token_ids = token_ids;
    rep.layer_residual_rms = compute_layer_residual_rms(trace);

    if (!logits.empty()) {
        rep.predicted_next_token = argmax_token(logits.back());
    }

    const NeuralAtlas effective_atlas = resolve_effective_atlas(model, atlas, mir);
    const std::vector<std::string> ordered_names =
        collect_ordered_symbol_names(effective_atlas, mir);

    rep.activations.reserve(ordered_names.size());
    for (const std::string& name : ordered_names) {
        rep.activations.push_back(eval_symbol_on_trace(
            effective_atlas.symbols.at(name),
            trace,
            rep.layer_residual_rms,
            model.config.num_layers));
    }

    return rep;
}

std::string format_execution_trace(
    const ExecutionTraceReport& rep,
    const Tokenizer* tokenizer) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "ExecutionTrace (tokens=" << rep.token_ids.size()
        << ", predicted_next_token=" << rep.predicted_next_token;
    if (tokenizer != nullptr) {
        oss << ", decoded_next=\"" << tokenizer->decode({rep.predicted_next_token}) << "\"";
    }
    oss << ")\n";
    if (tokenizer != nullptr) {
        oss << "Prompt: \"" << tokenizer->decode(rep.token_ids) << "\"\n";
    }
    oss << "Layer Residual RMS:";
    for (std::size_t l = 0; l < rep.layer_residual_rms.size(); ++l) {
        oss << " L" << l << "=" << rep.layer_residual_rms[l];
    }
    oss << "\n";
    oss << "Symbol Activations (" << rep.activations.size() << "):\n";
    for (const auto& act : rep.activations) {
        oss << "  " << act.symbol << " (" << act.kind << ", layer=" << act.layer
            << "): mean_proj=" << act.mean_proj
            << ", max_proj=" << act.max_proj
            << ", active_ratio=" << act.active_ratio << "\n";
    }
    return oss.str();
}

GenerationTraceReport trace_generation(
    const TransformerModel& model,
    const NeuralAtlas& atlas,
    const std::vector<int>& prompt_ids,
    int max_new_tokens,
    const MIRModule* mir) {
    if (prompt_ids.empty()) {
        throw std::invalid_argument("trace_generation requires non-empty prompt_ids");
    }
    if (max_new_tokens < 0) {
        throw std::invalid_argument("trace_generation requires non-negative max_new_tokens");
    }

    GenerationTraceReport rep;
    rep.prompt_ids = prompt_ids;
    rep.prefill_trace = trace_model_execution(model, atlas, prompt_ids, mir);
    if (max_new_tokens == 0) {
        return rep;
    }

    const NeuralAtlas effective_atlas = resolve_effective_atlas(model, atlas, mir);
    const std::vector<std::string> ordered_names =
        collect_ordered_symbol_names(effective_atlas, mir);

    KVCache cache = model.init_kv_cache();
    Vector last_logits;
    for (int tok : prompt_ids) {
        ForwardStepTrace pre_step = model.forward_step_traced(tok, cache);
        last_logits = std::move(pre_step.logits);
    }

    int cur_tok = argmax_token(last_logits);
    rep.generated_ids.reserve(static_cast<std::size_t>(max_new_tokens));
    rep.decode_steps.reserve(static_cast<std::size_t>(max_new_tokens));

    for (int step = 0; step < max_new_tokens; ++step) {
        rep.generated_ids.push_back(cur_tok);
        const ForwardStepTrace step_trace = model.forward_step_traced(cur_tok, cache);

        GenerationStepActivation step_act;
        step_act.step = step;
        step_act.token_id = cur_tok;
        step_act.activations.reserve(ordered_names.size());
        for (const std::string& name : ordered_names) {
            step_act.activations.push_back(eval_symbol_on_step_trace(
                effective_atlas.symbols.at(name),
                step_trace,
                model.config.num_layers));
        }
        rep.decode_steps.push_back(std::move(step_act));

        cur_tok = argmax_token(step_trace.logits);
    }

    return rep;
}

std::string format_generation_trace(
    const GenerationTraceReport& rep,
    const Tokenizer* tokenizer) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "GenerationTrace (prompt_tokens=" << rep.prompt_ids.size()
        << ", generated_tokens=" << rep.generated_ids.size() << ")\n";
    if (tokenizer != nullptr) {
        oss << "Prompt: \"" << tokenizer->decode(rep.prompt_ids) << "\"\n";
        oss << "Generated: \"" << tokenizer->decode(rep.generated_ids) << "\"\n";
    }
    oss << format_execution_trace(rep.prefill_trace, tokenizer);
    oss << "Decode Steps (" << rep.decode_steps.size() << "):\n";
    for (const auto& ds : rep.decode_steps) {
        oss << "  Step " << ds.step << ": token_id=" << ds.token_id;
        if (tokenizer != nullptr) {
            oss << ", token=\"" << tokenizer->decode({ds.token_id}) << "\"";
        }
        oss << "\n";
        for (const auto& act : ds.activations) {
            oss << "    " << act.symbol << " (" << act.kind << ", layer=" << act.layer
                << "): mean_proj=" << act.mean_proj
                << ", max_proj=" << act.max_proj
                << ", active_ratio=" << act.active_ratio << "\n";
        }
    }
    return oss.str();
}

TraceComparisonReport compare_execution_traces(
    const TransformerModel& base_model,
    const TransformerModel& patched_model,
    const NeuralAtlas& atlas,
    const std::vector<int>& token_ids,
    const MIRModule* mir) {
    const ExecutionTraceReport base_rep = trace_model_execution(base_model, atlas, token_ids, mir);
    const ExecutionTraceReport patched_rep = trace_model_execution(patched_model, atlas, token_ids, mir);

    TraceComparisonReport cmp;
    cmp.token_ids = token_ids;
    cmp.base_next_token = base_rep.predicted_next_token;
    cmp.patched_next_token = patched_rep.predicted_next_token;
    cmp.kl_divergence = compute_model_divergence(base_model, patched_model, {token_ids}).mean_kl;

    std::map<std::string, SymbolActivation> patched_map;
    for (const auto& a : patched_rep.activations) {
        patched_map[a.symbol] = a;
    }

    std::set<std::string> seen;
    for (const auto& b : base_rep.activations) {
        seen.insert(b.symbol);
        SymbolTraceDelta d;
        d.symbol = b.symbol;
        d.kind = b.kind;
        d.layer = b.layer;
        d.base_mean_proj = b.mean_proj;
        d.base_max_proj = b.max_proj;
        auto it = patched_map.find(b.symbol);
        if (it != patched_map.end()) {
            d.kind = it->second.kind;
            d.layer = it->second.layer;
            d.patched_mean_proj = it->second.mean_proj;
            d.patched_max_proj = it->second.max_proj;
        }
        d.delta_mean_proj = d.patched_mean_proj - d.base_mean_proj;
        d.delta_max_proj = d.patched_max_proj - d.base_max_proj;
        cmp.deltas.push_back(std::move(d));
    }
    for (const auto& p : patched_rep.activations) {
        if (!seen.contains(p.symbol)) {
            SymbolTraceDelta d;
            d.symbol = p.symbol;
            d.kind = p.kind;
            d.layer = p.layer;
            d.patched_mean_proj = p.mean_proj;
            d.delta_mean_proj = p.mean_proj;
            d.patched_max_proj = p.max_proj;
            d.delta_max_proj = p.max_proj;
            cmp.deltas.push_back(std::move(d));
        }
    }

    return cmp;
}

std::string format_trace_comparison(
    const TraceComparisonReport& cmp,
    const Tokenizer* tokenizer) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "TraceComparison (tokens=" << cmp.token_ids.size()
        << ", base_next_token=" << cmp.base_next_token;
    if (tokenizer != nullptr) {
        oss << ", base_decoded=\"" << tokenizer->decode({cmp.base_next_token}) << "\"";
    }
    oss << ", patched_next_token=" << cmp.patched_next_token;
    if (tokenizer != nullptr) {
        oss << ", patched_decoded=\"" << tokenizer->decode({cmp.patched_next_token}) << "\"";
    }
    oss << ", kl_divergence=" << cmp.kl_divergence << ")\n";
    oss << "Symbol Deltas (" << cmp.deltas.size() << "):\n";
    for (const auto& d : cmp.deltas) {
        oss << "  " << d.symbol << " (" << d.kind << ", layer=" << d.layer
            << "): base_mean=" << d.base_mean_proj
            << ", patched_mean=" << d.patched_mean_proj
            << ", delta_mean=" << d.delta_mean_proj
            << ", base_max=" << d.base_max_proj
            << ", patched_max=" << d.patched_max_proj
            << ", delta_max=" << d.delta_max_proj << "\n";
    }
    return oss.str();
}

StrataTestSuiteReport run_strata_tests(
    const MIRModule& mir,
    const TransformerModel& base_model,
    const CompileOutput& compiled,
    const NeuralAtlas& atlas,
    const Tokenizer* tokenizer) {
    StrataTestSuiteReport report;
    const NeuralAtlas effective_atlas = resolve_effective_atlas(base_model, atlas, &mir, tokenizer);

    struct CachedForward {
        Vector last_logits;
        ForwardTrace trace;
        std::vector<double> rms;
        bool has_trace = false;
    };
    std::map<std::vector<int>, CachedForward> compiled_cache;
    std::map<std::vector<int>, CachedForward> base_cache;

    auto get_cached = [](const TransformerModel& mdl,
                         std::map<std::vector<int>, CachedForward>& cache,
                         const std::vector<int>& ids,
                         bool need_trace) -> const CachedForward& {
        auto it = cache.find(ids);
        if (it == cache.end() || (need_trace && !it->second.has_trace)) {
            auto [logits, tr] = mdl.forward(ids, {}, need_trace);
            CachedForward cf;
            if (!logits.empty()) {
                cf.last_logits = std::move(logits.back());
            }
            if (need_trace) {
                cf.rms = compute_layer_residual_rms(tr);
                cf.trace = std::move(tr);
                cf.has_trace = true;
            }
            if (it == cache.end()) {
                it = cache.emplace(ids, std::move(cf)).first;
            } else {
                it->second = std::move(cf);
            }
        }
        return it->second;
    };

    for (const auto& tc : mir.tests) {
        ++report.total_tests;
        std::vector<StrataAssertionResult> results;
        results.reserve(tc.expectations.size());
        bool all_passed = true;

        for (const auto& exp : tc.expectations) {
            StrataAssertionResult res;
            res.description = format_expectation_description(exp);
            res.cmp_op = exp.cmp_op;

            if (exp.kind == "resonance") {
                auto sit = effective_atlas.symbols.find(exp.symbol);
                if (sit == effective_atlas.symbols.end()) {
                    throw UndefinedSymbolError(
                        "Symbol not resolved in atlas for test expectation: " + exp.symbol);
                }
                const SymbolBinding& sb = sit->second;
                const std::vector<int> ids = tokenize_contract_string(
                    exp.prompt, tokenizer, compiled.model.config.vocab_size);
                const CachedForward& ct = get_cached(compiled.model, compiled_cache, ids, true);
                res.measured = measure_resonance_on_trace(
                    sb, exp.pole, ct.trace, ct.rms, compiled.model.config.num_layers);

                if (exp.rhs_is_base) {
                    const std::vector<int> base_ids = tokenize_contract_string(
                        exp.prompt, tokenizer, base_model.config.vocab_size);
                    const CachedForward& bt = get_cached(base_model, base_cache, base_ids, true);
                    const double base_measured = measure_resonance_on_trace(
                        sb, exp.pole, bt.trace, bt.rms, base_model.config.num_layers);
                    res.expected_rhs = base_measured + exp.rhs_val;
                } else {
                    res.expected_rhs = exp.rhs_val;
                }
            } else if (exp.kind == "top_token") {
                const int vocab_cap = std::max(1, compiled.model.config.vocab_size);
                const std::vector<int> prompt_ids = tokenize_contract_string(
                    exp.prompt, tokenizer, vocab_cap);
                const std::vector<int> target_ids = tokenize_contract_string(
                    exp.target_token_str, tokenizer, vocab_cap);
                const int expected_tok = target_ids[0] % vocab_cap;
                const CachedForward& cf = get_cached(compiled.model, compiled_cache, prompt_ids, false);
                const int pred_tok = argmax_token(cf.last_logits);
                res.measured = static_cast<double>(pred_tok);
                res.expected_rhs = static_cast<double>(expected_tok);

                std::ostringstream doss;
                doss << res.description << " (pred=" << pred_tok;
                if (tokenizer != nullptr) {
                    doss << " \"" << tokenizer->decode({pred_tok}) << "\"";
                }
                doss << ", target=" << expected_tok;
                if (tokenizer != nullptr) {
                    doss << " \"" << tokenizer->decode({expected_tok}) << "\"";
                }
                doss << ")";
                res.description = doss.str();
            } else if (exp.kind == "logit_margin") {
                const int vocab_cap = std::max(
                    1,
                    exp.rhs_is_base
                        ? std::min(base_model.config.vocab_size, compiled.model.config.vocab_size)
                        : compiled.model.config.vocab_size);
                const std::vector<int> prompt_ids = tokenize_contract_string(
                    exp.prompt, tokenizer, vocab_cap);
                const int pos_tok = tokenize_contract_string(
                    exp.target_token_str, tokenizer, vocab_cap)[0] % vocab_cap;
                const int foil_tok = tokenize_contract_string(
                    exp.foil_token_str, tokenizer, vocab_cap)[0] % vocab_cap;
                const CachedForward& cf = get_cached(compiled.model, compiled_cache, prompt_ids, false);
                const double pos_logit = (static_cast<std::size_t>(pos_tok) < cf.last_logits.size())
                    ? cf.last_logits[static_cast<std::size_t>(pos_tok)]
                    : 0.0;
                const double foil_logit = (static_cast<std::size_t>(foil_tok) < cf.last_logits.size())
                    ? cf.last_logits[static_cast<std::size_t>(foil_tok)]
                    : 0.0;
                res.measured = pos_logit - foil_logit;

                if (exp.rhs_is_base) {
                    const CachedForward& bf = get_cached(base_model, base_cache, prompt_ids, false);
                    const double b_pos = (static_cast<std::size_t>(pos_tok) < bf.last_logits.size())
                        ? bf.last_logits[static_cast<std::size_t>(pos_tok)]
                        : 0.0;
                    const double b_foil = (static_cast<std::size_t>(foil_tok) < bf.last_logits.size())
                        ? bf.last_logits[static_cast<std::size_t>(foil_tok)]
                        : 0.0;
                    res.expected_rhs = (b_pos - b_foil) + exp.rhs_val;
                } else {
                    res.expected_rhs = exp.rhs_val;
                }
            } else if (exp.kind == "kl_drift") {
                std::vector<std::string> raw_prompts = exp.prompts;
                if (raw_prompts.empty()) {
                    raw_prompts.push_back(exp.prompt);
                }
                std::vector<std::vector<int>> seqs;
                seqs.reserve(raw_prompts.size());
                const int vocab_cap = std::min(
                    base_model.config.vocab_size, compiled.model.config.vocab_size);
                for (const std::string& p : raw_prompts) {
                    seqs.push_back(tokenize_contract_string(p, tokenizer, vocab_cap));
                }
                res.measured = compute_model_divergence(base_model, compiled.model, seqs).mean_kl;
                res.expected_rhs = exp.rhs_val;
            } else if (exp.kind == "crosstalk") {
                res.measured = compiled.stats.max_crosstalk;
                res.expected_rhs = exp.rhs_val;
            } else if (exp.kind == "approx") {
                res.measured = compiled.stats.max_approx_error;
                res.expected_rhs = exp.rhs_val;
            } else {
                throw ParseError("Unknown test expectation kind: " + exp.kind);
            }

            res.passed = evaluate_comparison(res.measured, res.cmp_op, res.expected_rhs);
            ++report.total_assertions;
            if (res.passed) {
                ++report.passed_assertions;
            } else {
                all_passed = false;
            }
            results.push_back(std::move(res));
        }

        if (all_passed) {
            ++report.passed_tests;
        }
        report.test_results.emplace_back(tc.name, std::move(results));
    }

    return report;
}

std::string format_strata_test_report(const StrataTestSuiteReport& report) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "StrataTestSuite (" << report.passed_tests << "/" << report.total_tests
        << " tests passed, " << report.passed_assertions << "/" << report.total_assertions
        << " assertions passed)\n";
    for (const auto& [name, results] : report.test_results) {
        int pass_cnt = 0;
        for (const auto& r : results) {
            if (r.passed) {
                ++pass_cnt;
            }
        }
        const bool ok = (pass_cnt == static_cast<int>(results.size()));
        oss << "  TEST " << name << ": " << (ok ? "PASS" : "FAIL")
            << " (" << pass_cnt << "/" << results.size() << ")\n";
        for (const auto& r : results) {
            oss << "    [" << (r.passed ? "PASS" : "FAIL") << "] " << r.description
                << " (measured=" << r.measured
                << ", op=" << r.cmp_op
                << ", expected=" << r.expected_rhs << ")\n";
        }
    }
    return oss.str();
}

}
