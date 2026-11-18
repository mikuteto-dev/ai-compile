#pragma once

#include "aic/atlas.hpp"
#include "aic/cir.hpp"
#include "aic/mir.hpp"
#include "aic/runtime.hpp"
#include "aic/wisa.hpp"

#include <string>
#include <utility>
#include <vector>

namespace aic {

struct CompileStats {
    int instructions_applied = 0;
    int added_poles = 0;
    int added_layers = 0;
    double max_approx_error = 0.0;
    double max_crosstalk = 0.0;
};

struct CompileOutput {
    TransformerModel model;
    CIRModule cir;
    std::vector<WISAInstruction> wisa;
    CompileStats stats;
};

std::pair<TransformerModel, CompileStats> apply_wisa(
    const TransformerModel& model,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode = "in-place");

CompileStats apply_wisa_streaming_checkpoint(
    const std::string& src_model_dir,
    const std::string& dst_model_dir,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode = "in-place");

CompileStats emit_lora_adapter(
    const TransformerModel& base_model,
    const std::vector<WISAInstruction>& insts,
    const std::string& out_lora_dir,
    int max_lora_rank = 16,
    double lora_alpha = 16.0);

CompileOutput compile_mir_to_model(
    const MIRModule& mir,
    const TransformerModel& model,
    const NeuralAtlas* atlas = nullptr,
    const std::string& target_mode = "in-place",
    int opt_level = 2);

}
