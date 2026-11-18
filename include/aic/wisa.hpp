#pragma once

#include "aic/atlas.hpp"
#include "aic/cir.hpp"
#include "aic/runtime.hpp"
#include "aic/tensor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace aic {

enum class WISAKind {
    SWIGLU_POLE,
    THRESHOLDED_ROUTE,
    VPD_RANK1,
    ATTN_BILINEAR,
    ASSOC_MEM_ROW,
    FIELD_SPECTRAL_OP,
    NORM_RESCALE
};

struct WISAInstruction {
    WISAKind kind = WISAKind::VPD_RANK1;
    int layer = 0;
    std::pair<int, int> layer_range{0, 1};
    int slot = 0;
    int polarity = 1;
    bool append_slot = false;
    std::string matrix_id;
    int head_group = 0;
    std::string rope_mode = "none";
    std::string norm_id = "post_attention_layernorm";
    Vector w_gate;
    Vector w_up;
    Vector w_down;
    Vector u;
    Vector v;
    double scale = 1.0;
    double threshold = 0.0;
    Matrix w_qk;
    Matrix w_ov;
    Vector k_star;
    Vector v_star;
    Matrix c_inv;
    double target_norm = 1.0;
    std::vector<Vector> u_k;
    std::vector<Vector> v_k;
    Vector coeff_delta;
    Vector gamma_scale;
};

std::vector<WISAInstruction> lower_cir_to_wisa(
    const CIRModule& cir,
    const ModelConfig& cfg,
    const NeuralAtlas& atlas,
    const std::string& target_mode = "in-place");

std::vector<WISAInstruction> optimize_wisa(
    const std::vector<WISAInstruction>& insts,
    int opt_level = 2);

std::string format_wisa(const std::vector<WISAInstruction>& insts);

}
