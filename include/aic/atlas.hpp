#pragma once

#include "aic/mir.hpp"
#include "aic/runtime.hpp"
#include "aic/tensor.hpp"
#include "aic/tokenizer.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace aic {

struct SymbolBinding {
    std::string name;
    std::string kind;
    std::pair<int, int> layer_range{0, 1};
    Vector direction;
    std::vector<Vector> subspace_basis;
    std::map<std::string, std::tuple<int, int, int>> pole_slots;
    std::vector<SVDMode> field_modes;
    std::vector<Vector> preserve_keys;
    std::vector<Vector> manifold_plane;
    std::map<std::string, Vector> rodata_key_vectors;
    std::map<std::string, Vector> rodata_val_vectors;
};

struct NeuralAtlas {
    std::string arch_name;
    int hidden_size = 0;
    int num_layers = 0;
    std::map<std::string, SymbolBinding> symbols;
    std::map<std::string, std::vector<Vector>> preserve_directions;
    std::vector<std::pair<int, int>> dormant_poles;
    std::set<std::pair<int, int>> occupied_kv_groups;
    std::map<std::string, Matrix> frame_bridges;
    std::map<int, Matrix> mlp_cov_inv;
    std::map<int, LayerFisherDiag> fisher_metrics;

    void save(const std::string& path) const;
    static NeuralAtlas load(const std::string& path);
};

NeuralAtlas build_default_atlas(
    const TransformerModel& model,
    const MIRModule* mir = nullptr,
    std::uint64_t seed = 42);

void calibrate_atlas_empirical(
    NeuralAtlas& atlas,
    const TransformerModel& model,
    const MIRModule* mir = nullptr,
    const Tokenizer* tokenizer = nullptr);

}
