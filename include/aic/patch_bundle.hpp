#pragma once

#include "aic/runtime.hpp"
#include "aic/tensor.hpp"
#include "aic/wisa.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace aic {

struct PatchBundle {
    std::string module_name;
    std::string arch_name;
    std::string target_mode;
    int num_layers = 0;
    int hidden_size = 0;
    int orig_num_layers = 0;
    int orig_intermediate_size = 0;
    std::map<std::string, std::uint64_t> base_checksums;
    std::vector<WISAInstruction> instructions;
    std::map<std::string, Matrix> matrix_deltas;
    std::map<std::string, Vector> vector_deltas;
};

PatchBundle create_patch_bundle(
    const TransformerModel& base_model,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode = "in-place",
    const std::string& module_name = "patch");

PatchBundle create_patch_bundle_streaming(
    const std::string& model_dir,
    const std::vector<WISAInstruction>& insts,
    const std::string& target_mode = "in-place",
    const std::string& module_name = "patch");

void save_patch_bundle(const PatchBundle& bundle, const std::string& path);

PatchBundle load_patch_bundle(const std::string& path);

TransformerModel apply_patch_bundle(
    const TransformerModel& base_model,
    const PatchBundle& bundle,
    bool verify_checksum = true);

TransformerModel revert_patch_bundle(
    const TransformerModel& patched_model,
    const PatchBundle& bundle);

PatchBundle compose_patch_bundles(const PatchBundle& a, const PatchBundle& b);

void apply_patch_bundle_streaming(
    const std::string& src_model_dir,
    const std::string& dst_model_dir,
    const PatchBundle& bundle,
    bool verify_checksum = true);

void revert_patch_bundle_streaming(
    const std::string& patched_model_dir,
    const std::string& restored_model_dir,
    const PatchBundle& bundle);

}
