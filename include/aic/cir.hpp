#pragma once

#include "aic/atlas.hpp"
#include "aic/mir.hpp"
#include "aic/runtime.hpp"
#include "aic/tensor.hpp"

#include <string>
#include <utility>
#include <vector>

namespace aic {

struct TessellatedSlot {
    int chart_index = 0;
    Vector w_gate;
    Vector w_up;
    Vector w_down;
};

struct CIROp {
    std::string kind;
    std::string source_op;
    std::string symbol;
    std::string pole;
    std::string matrix_id;
    int layer = 0;
    int layer_end = 1;
    int polarity = 1;
    double scalar = 1.0;
    double threshold = 0.5;
    double approx_error = 0.0;
    double crosstalk = 0.0;
    Vector input_dir;
    Vector output_dir;
    std::vector<TessellatedSlot> spline_slots;
    Matrix bridge_matrix;
    Matrix w_qk;
    Matrix w_ov;
    std::string rope_mode = "none";
    Vector key_vec;
    Vector val_vec;
    Matrix cov_inv;
    double target_norm = 1.0;
    std::vector<SVDMode> field_modes;
};

struct CIRFlow {
    std::string name;
    std::pair<int, int> layer_range{0, 1};
    std::string target_mode_hint;
    MIREffectContract contract;
    double max_approx_error = 0.0;
    double max_crosstalk = 0.0;
    int sequential_depth = 0;
    int added_layers_needed = 0;
    std::vector<CIROp> ops;
};

struct CIRModule {
    std::string name;
    std::string arch_name;
    int hidden_size = 0;
    int num_layers = 0;
    std::string target_mode = "in-place";
    std::vector<CIRFlow> flows;
};

std::pair<std::vector<TessellatedSlot>, double> build_s1_swiglu_spline(
    const Vector& e0,
    const Vector& e1,
    const Vector& e2,
    int k_charts,
    double delta_angle,
    double beta = 0.0);

double eval_s1_spline_error(
    const std::vector<TessellatedSlot>& slots,
    const Vector& e0,
    const Vector& e1,
    const Vector& e2,
    double delta_angle,
    std::size_t eval_points = 360);

std::pair<std::vector<TessellatedSlot>, double> build_t2_swiglu_spline(
    const Vector& e0,
    const Vector& c1,
    const Vector& s1,
    const Vector& c2,
    const Vector& s2,
    int k_per_axis,
    double dtheta1,
    double dtheta2,
    double beta = 0.0);

double eval_t2_spline_error(
    const std::vector<TessellatedSlot>& slots,
    const Vector& e0,
    const Vector& c1,
    const Vector& s1,
    const Vector& c2,
    const Vector& s2,
    double dtheta1,
    double dtheta2,
    std::size_t grid_per_axis = 24);

double fisher_inner(
    const std::vector<SVDMode>& a,
    const std::vector<SVDMode>& b,
    const Vector& g_out,
    const Vector& g_in);

double fisher_cosine(
    const std::vector<SVDMode>& a,
    const std::vector<SVDMode>& b,
    const Vector& g_out,
    const Vector& g_in);

std::vector<SVDMode> fisher_decouple(
    const std::vector<SVDMode>& f1,
    const std::vector<SVDMode>& f2,
    const Vector& g_out,
    const Vector& g_in,
    double reg = 1e-4);

std::vector<SVDMode> temper_field_modes(
    const std::vector<SVDMode>& modes,
    const std::string& band,
    double gain);

std::vector<SVDMode> blend_field_modes(
    const std::vector<SVDMode>& fa,
    const std::vector<SVDMode>& fb,
    double alpha);

CIRModule lower_mir_to_cir(
    const MIRModule& mir,
    const ModelConfig& cfg,
    const NeuralAtlas& atlas,
    const std::string& target_mode = "in-place");

CIRModule optimize_cir(const CIRModule& cir, int opt_level = 2);

std::string format_cir(const CIRModule& cir);

}
