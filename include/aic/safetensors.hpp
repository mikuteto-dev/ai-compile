#pragma once

#include "aic/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace aic {

struct TensorSpec {
    std::vector<std::size_t> shape;
    std::string dtype = "F32";
    std::vector<double> data;

    static TensorSpec from_matrix(const Matrix& m, std::string dtype = "F32");
    static TensorSpec from_vector(const Vector& v, std::string dtype = "F32");
    Matrix to_matrix() const;
    Vector to_vector() const;
};

class MmapSafetensors {
public:
    explicit MmapSafetensors(const std::string& path);
    ~MmapSafetensors();

    MmapSafetensors(const MmapSafetensors&) = delete;
    MmapSafetensors& operator=(const MmapSafetensors&) = delete;
    MmapSafetensors(MmapSafetensors&& other) noexcept;
    MmapSafetensors& operator=(MmapSafetensors&& other) noexcept;

    bool has_tensor(const std::string& name) const;
    std::vector<std::string> tensor_names() const;
    std::vector<std::size_t> tensor_shape(const std::string& name) const;
    std::string tensor_dtype(const std::string& name) const;
    TensorSpec read_tensor(const std::string& name) const;
    Vector read_matrix_row(const std::string& name, std::size_t row) const;
    Vector mat_vec(const std::string& name, const Vector& x) const;
    std::pair<std::size_t, std::size_t> raw_byte_range(const std::string& name) const;
    std::string_view raw_tensor_bytes(const std::string& name) const;
    const std::map<std::string, std::string>& metadata() const;

private:
    struct Entry {
        std::string dtype;
        std::vector<std::size_t> shape;
        std::size_t start = 0;
        std::size_t end = 0;
    };

    void* mapped_ = nullptr;
    std::size_t file_size_ = 0;
    std::size_t data_offset_ = 0;
    std::map<std::string, Entry> entries_;
    std::map<std::string, std::string> metadata_;

    void reset() noexcept;
    const Entry& require_entry(const std::string& name) const;
};

void save_safetensors(
    const std::string& path,
    const std::map<std::string, TensorSpec>& tensors,
    const std::map<std::string, std::string>& metadata = {});

std::pair<std::map<std::string, TensorSpec>, std::map<std::string, std::string>>
load_safetensors(const std::string& path);

void save_safetensors_sharded(
    const std::string& dir,
    const std::map<std::string, TensorSpec>& tensors,
    std::size_t max_shard_bytes,
    const std::map<std::string, std::string>& metadata = {});

std::pair<std::map<std::string, TensorSpec>, std::map<std::string, std::string>>
load_safetensors_sharded(const std::string& dir_or_file);

TensorSpec load_single_tensor_sharded(
    const std::string& dir_or_file,
    const std::string& tensor_name);

void rewrite_safetensors_selective(
    const std::string& src_path,
    const std::string& dst_path,
    const std::map<std::string, TensorSpec>& replacements,
    const std::map<std::string, TensorSpec>& additions = {},
    const std::set<std::string>& removals = {},
    std::int64_t* out_byte_delta = nullptr);

}
