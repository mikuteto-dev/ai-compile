#pragma once

#include "aic/atlas.hpp"
#include "aic/codegen.hpp"
#include "aic/mir.hpp"
#include "aic/runtime.hpp"
#include "aic/tokenizer.hpp"

#include <cstdint>
#include <string>

namespace aic {

NeuralAtlas resolve_symbols_for_target(
    const MIRModule& mir,
    const TransformerModel& target_model,
    const NeuralAtlas* target_atlas = nullptr,
    std::uint64_t seed = 0,
    const Tokenizer* tokenizer = nullptr);

NeuralAtlas resolve_symbols_for_target_streaming(
    const MIRModule& mir,
    const std::string& model_dir,
    const NeuralAtlas* target_atlas = nullptr,
    std::uint64_t seed = 0,
    const Tokenizer* tokenizer = nullptr);

CompileOutput cross_compile(
    const std::string& strata_source,
    const TransformerModel& target_model,
    const NeuralAtlas* target_atlas = nullptr,
    const std::string& target_mode = "in-place",
    const Tokenizer* tokenizer = nullptr);

}
