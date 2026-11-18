#pragma once

#include "aic/atlas.hpp"
#include "aic/codegen.hpp"
#include "aic/mir.hpp"
#include "aic/runtime.hpp"
#include "aic/tokenizer.hpp"

#include <string>
#include <utility>
#include <vector>

namespace aic {

struct SymbolActivation {
    std::string symbol;
    std::string kind;
    int layer = 0;
    double mean_proj = 0.0;
    double max_proj = 0.0;
    double active_ratio = 0.0;
};

struct ExecutionTraceReport {
    std::vector<int> token_ids;
    std::vector<double> layer_residual_rms;
    std::vector<SymbolActivation> activations;
    int predicted_next_token = 0;
};

struct GenerationStepActivation {
    int step = 0;
    int token_id = 0;
    std::vector<SymbolActivation> activations;
};

struct GenerationTraceReport {
    std::vector<int> prompt_ids;
    std::vector<int> generated_ids;
    ExecutionTraceReport prefill_trace;
    std::vector<GenerationStepActivation> decode_steps;
};

struct SymbolTraceDelta {
    std::string symbol;
    std::string kind;
    int layer = 0;
    double base_mean_proj = 0.0;
    double patched_mean_proj = 0.0;
    double delta_mean_proj = 0.0;
    double base_max_proj = 0.0;
    double patched_max_proj = 0.0;
    double delta_max_proj = 0.0;
};

struct TraceComparisonReport {
    std::vector<int> token_ids;
    int base_next_token = 0;
    int patched_next_token = 0;
    double kl_divergence = 0.0;
    std::vector<SymbolTraceDelta> deltas;
};

struct StrataAssertionResult {
    std::string description;
    bool passed = false;
    double measured = 0.0;
    double expected_rhs = 0.0;
    std::string cmp_op;
};

struct StrataTestSuiteReport {
    int total_tests = 0;
    int passed_tests = 0;
    int total_assertions = 0;
    int passed_assertions = 0;
    std::vector<std::pair<std::string, std::vector<StrataAssertionResult>>> test_results;
};

ExecutionTraceReport trace_model_execution(
    const TransformerModel& model,
    const NeuralAtlas& atlas,
    const std::vector<int>& token_ids,
    const MIRModule* mir = nullptr);

std::string format_execution_trace(
    const ExecutionTraceReport& rep,
    const Tokenizer* tokenizer = nullptr);

GenerationTraceReport trace_generation(
    const TransformerModel& model,
    const NeuralAtlas& atlas,
    const std::vector<int>& prompt_ids,
    int max_new_tokens,
    const MIRModule* mir = nullptr);

std::string format_generation_trace(
    const GenerationTraceReport& rep,
    const Tokenizer* tokenizer = nullptr);

TraceComparisonReport compare_execution_traces(
    const TransformerModel& base_model,
    const TransformerModel& patched_model,
    const NeuralAtlas& atlas,
    const std::vector<int>& token_ids,
    const MIRModule* mir = nullptr);

std::string format_trace_comparison(
    const TraceComparisonReport& cmp,
    const Tokenizer* tokenizer = nullptr);

StrataTestSuiteReport run_strata_tests(
    const MIRModule& mir,
    const TransformerModel& base_model,
    const CompileOutput& compiled,
    const NeuralAtlas& atlas,
    const Tokenizer* tokenizer = nullptr);

std::string format_strata_test_report(const StrataTestSuiteReport& report);

}
