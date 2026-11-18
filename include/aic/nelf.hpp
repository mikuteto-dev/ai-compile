#pragma once

#include "aic/atlas.hpp"
#include "aic/runtime.hpp"
#include "aic/tensor.hpp"

#include <map>
#include <string>

namespace aic {

struct NelfDiffReport {
    double text_frob_diff = 0.0;
    double rodata_frob_diff = 0.0;
    double field_frob_diff = 0.0;
    double residual_frob_diff = 0.0;
    int changed_symbols = 0;
};

struct NeuralELF {
    ModelConfig config;
    std::map<std::string, Matrix> text_weights;
    std::map<std::string, Matrix> rodata_weights;
    std::map<std::string, Matrix> field_weights;
    std::map<std::string, Matrix> residual_weights;
    std::map<std::string, Matrix> passthrough_matrices;
    std::map<std::string, Vector> passthrough_vectors;
    NeuralAtlas atlas;

    TransformerModel reconstruct_model() const;
    std::map<std::string, double> section_energy_ratios() const;
    void save(const std::string& dir) const;
    static NeuralELF load(const std::string& dir);
};

NelfDiffReport diff_nelf(const NeuralELF& lhs, const NeuralELF& rhs);
std::string format_nelf_diff(const NelfDiffReport& rep);

}
