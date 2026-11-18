#pragma once

#include "aic/nelf.hpp"
#include "aic/runtime.hpp"

#include <string>
#include <utility>

namespace aic {

struct FaithfulnessReport {
    double weight_frob_diff = 0.0;
    double mean_kl_div = 0.0;
    double max_kl_div = 0.0;
    double max_logit_diff = 0.0;
    double residual_energy_ratio = 0.0;
};

std::pair<NeuralELF, std::string> lift_model(
    const TransformerModel& model,
    const std::string& module_name = "LiftedModel",
    std::pair<int, int> layer_filter = {-1, -1});

std::pair<NeuralELF, std::string> lift_checkpoint_streaming(
    const std::string& model_dir,
    const std::string& module_name = "LiftedModel",
    std::pair<int, int> layer_filter = {-1, -1});

TransformerModel recompile_nelf_with_patch(
    const NeuralELF& nelf,
    const std::string& strata_source,
    const std::string& target_mode = "in-place");

FaithfulnessReport verify_nelf_faithfulness(
    const TransformerModel& orig,
    const TransformerModel& recompiled,
    const NeuralELF* nelf = nullptr);

}
