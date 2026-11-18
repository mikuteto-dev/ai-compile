#include "aic/safetensors.hpp"

#include "aic/json_min.hpp"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace aic {

namespace {

std::string normalize_dtype(std::string dtype) {
    for (char& c : dtype) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    if (dtype == "FP8" || dtype == "FLOAT8_E4M3FN") {
        dtype = "F8_E4M3";
    } else if (dtype == "INT8") {
        dtype = "I8";
    } else if (dtype == "INT4") {
        dtype = "I4";
    }
    if (dtype != "F32" && dtype != "F16" && dtype != "BF16" &&
        dtype != "F8_E4M3" && dtype != "I8" && dtype != "I4") {
        throw std::invalid_argument("Unsupported safetensors dtype: " + dtype);
    }
    return dtype;
}

std::size_t dtype_packed_bytes(const std::string& dtype, std::size_t num_elems) {
    if (dtype == "F32") {
        if (num_elems > std::numeric_limits<std::size_t>::max() / 4u) {
            throw std::overflow_error("Tensor byte count overflow");
        }
        return num_elems * 4u;
    }
    if (dtype == "F16" || dtype == "BF16") {
        if (num_elems > std::numeric_limits<std::size_t>::max() / 2u) {
            throw std::overflow_error("Tensor byte count overflow");
        }
        return num_elems * 2u;
    }
    if (dtype == "F8_E4M3" || dtype == "I8") {
        return num_elems;
    }
    if (dtype == "I4") {
        return (num_elems / 2u) + (num_elems % 2u);
    }
    throw std::invalid_argument("Unsupported safetensors dtype: " + dtype);
}

std::size_t checked_num_elements(const std::vector<std::size_t>& shape) {
    std::size_t total = 1;
    for (std::size_t d : shape) {
        if (d != 0 && total > std::numeric_limits<std::size_t>::max() / d) {
            throw std::overflow_error("Tensor element count overflow");
        }
        total *= d;
    }
    return total;
}

std::uint16_t double_to_bf16_bits(double val) {
    const float f = static_cast<float>(val);
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return static_cast<std::uint16_t>(((bits >> 16) | 0x0040u) & 0xFFFFu);
    }
    const std::uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>((bits + rounding_bias) >> 16);
}

double bf16_bits_to_double(std::uint16_t h) {
    const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
    return static_cast<double>(std::bit_cast<float>(bits));
}

std::uint16_t double_to_f16_bits(double val) {
    const float f = static_cast<float>(val);
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t abs_bits = bits & 0x7FFFFFFFu;

    if (abs_bits > 0x7F800000u) {
        std::uint32_t nan_payload = (abs_bits >> 13) & 0x03FFu;
        if (nan_payload == 0) {
            nan_payload = 1u;
        }
        return static_cast<std::uint16_t>(sign | 0x7C00u | nan_payload);
    }
    if (abs_bits >= 0x477FF000u) {
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }
    if (abs_bits >= 0x38800000u) {
        std::uint32_t exp = (abs_bits >> 23) - 127u + 15u;
        const std::uint32_t mant = abs_bits & 0x007FFFFFu;
        std::uint32_t rounded_mant = mant >> 13;
        const std::uint32_t rem = mant & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (rounded_mant & 1u))) {
            ++rounded_mant;
            if (rounded_mant == 0x0400u) {
                rounded_mant = 0u;
                ++exp;
            }
        }
        if (exp >= 31u) {
            return static_cast<std::uint16_t>(sign | 0x7C00u);
        }
        return static_cast<std::uint16_t>(sign | (exp << 10) | rounded_mant);
    }
    if (abs_bits <= 0x33000000u) {
        return static_cast<std::uint16_t>(sign);
    }
    const int e = static_cast<int>(abs_bits >> 23) - 127;
    const std::uint32_t mant_full = 0x00800000u | (abs_bits & 0x007FFFFFu);
    const std::uint32_t shift = static_cast<std::uint32_t>(-1 - e);
    std::uint32_t m = mant_full >> shift;
    const std::uint32_t rem = mant_full & ((1u << shift) - 1u);
    const std::uint32_t half = 1u << (shift - 1u);
    if (rem > half || (rem == half && (m & 1u))) {
        ++m;
    }
    return static_cast<std::uint16_t>(sign | m);
}

double f16_bits_to_double(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1Fu;
    std::uint32_t mant = h & 0x03FFu;

    if (exp == 0u) {
        if (mant == 0u) {
            return static_cast<double>(std::bit_cast<float>(sign));
        }
        std::uint32_t exp_f32 = 113u;
        while ((mant & 0x0400u) == 0u) {
            mant <<= 1;
            --exp_f32;
        }
        mant &= 0x03FFu;
        return static_cast<double>(std::bit_cast<float>(sign | (exp_f32 << 23) | (mant << 13)));
    }
    if (exp == 31u) {
        if (mant == 0u) {
            return static_cast<double>(std::bit_cast<float>(sign | 0x7F800000u));
        }
        return static_cast<double>(std::bit_cast<float>(sign | 0x7FC00000u | (mant << 13)));
    }
    const std::uint32_t exp_f32 = exp + (127u - 15u);
    return static_cast<double>(std::bit_cast<float>(sign | (exp_f32 << 23) | (mant << 13)));
}

std::uint8_t double_to_f8_e4m3(double val) {
    const std::uint8_t sign = std::signbit(val) ? 0x80u : 0x00u;
    if (std::isnan(val) || std::isinf(val)) {
        return static_cast<std::uint8_t>(sign | 0x7Fu);
    }
    const double ax = std::abs(val);
    if (ax == 0.0) {
        return sign;
    }
    if (ax >= 448.0) {
        return static_cast<std::uint8_t>(sign | 0x7Eu);
    }
    if (ax < 0.015625) {
        const double scaled = ax * 512.0;
        const double fl = std::floor(scaled);
        const double rem = scaled - fl;
        int m = static_cast<int>(fl);
        if (rem > 0.5 || (rem == 0.5 && (m & 1))) {
            ++m;
        }
        return static_cast<std::uint8_t>(sign | static_cast<std::uint8_t>(m));
    }
    int exp_unbiased = 0;
    const double frac = std::frexp(ax, &exp_unbiased);
    const double m_norm = frac * 2.0;
    int exp_bits = (exp_unbiased - 1) + 7;
    const double mant_scaled = (m_norm - 1.0) * 8.0;
    const double fl = std::floor(mant_scaled);
    const double rem = mant_scaled - fl;
    int mant_bits = static_cast<int>(fl);
    if (rem > 0.5 || (rem == 0.5 && (mant_bits & 1))) {
        ++mant_bits;
        if (mant_bits == 8) {
            mant_bits = 0;
            ++exp_bits;
        }
    }
    if (exp_bits > 15 || (exp_bits == 15 && mant_bits >= 7)) {
        return static_cast<std::uint8_t>(sign | 0x7Eu);
    }
    return static_cast<std::uint8_t>(
        sign | (static_cast<std::uint8_t>(exp_bits) << 3) | static_cast<std::uint8_t>(mant_bits));
}

double f8_e4m3_to_double(std::uint8_t b) {
    const double sign = ((b & 0x80u) != 0u) ? -1.0 : 1.0;
    const int exp = static_cast<int>((b >> 3) & 0x0Fu);
    const int mant = static_cast<int>(b & 0x07u);
    if (exp == 15 && mant == 7) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (exp == 0) {
        if (mant == 0) {
            return sign * 0.0;
        }
        return sign * (static_cast<double>(mant) / 512.0);
    }
    const double frac = 1.0 + static_cast<double>(mant) / 8.0;
    return sign * std::ldexp(frac, exp - 7);
}

std::uint8_t double_to_i8(double val) {
    if (std::isnan(val)) {
        return 0u;
    }
    if (val >= 127.0) {
        return static_cast<std::uint8_t>(static_cast<std::int8_t>(127));
    }
    if (val <= -128.0) {
        return static_cast<std::uint8_t>(static_cast<std::int8_t>(-128));
    }
    const long long rounded = std::llround(val);
    const long long clamped = std::clamp(rounded, -128LL, 127LL);
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(clamped));
}

double i8_to_double(std::uint8_t b) {
    return static_cast<double>(static_cast<std::int8_t>(b));
}

void write_u16_le(std::vector<std::uint8_t>& out, std::uint16_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFu));
}

void write_u32_le(std::vector<std::uint8_t>& out, std::uint32_t v) {
    out.push_back(static_cast<std::uint8_t>(v & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 8) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 16) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((v >> 24) & 0xFFu));
}

std::uint16_t read_u16_le(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
}

std::uint32_t read_u32_le(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])
        | (static_cast<std::uint32_t>(p[1]) << 8)
        | (static_cast<std::uint32_t>(p[2]) << 16)
        | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t read_u64_le(const std::uint8_t* p) {
    std::uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

double decode_value_at(
    const std::uint8_t* base,
    const std::string& dtype,
    std::size_t index,
    double scale) {
    if (dtype == "F32") {
        return static_cast<double>(std::bit_cast<float>(read_u32_le(base + index * 4u)));
    }
    if (dtype == "F16") {
        return f16_bits_to_double(read_u16_le(base + index * 2u));
    }
    if (dtype == "BF16") {
        return bf16_bits_to_double(read_u16_le(base + index * 2u));
    }
    if (dtype == "F8_E4M3") {
        return f8_e4m3_to_double(base[index]);
    }
    if (dtype == "I8") {
        return i8_to_double(base[index]);
    }
    if (dtype == "I4") {
        const std::uint8_t byte = base[index / 2u];
        const std::uint8_t nib = ((index & 1u) == 0u)
            ? (byte & 0x0Fu)
            : ((byte >> 4) & 0x0Fu);
        const int q = (nib >= 8u) ? (static_cast<int>(nib) - 16) : static_cast<int>(nib);
        return static_cast<double>(q) * scale;
    }
    throw std::invalid_argument("Unsupported streaming tensor dtype: " + dtype);
}

std::string format_scale_17g(double scale) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", scale);
    return std::string(buf);
}

std::pair<std::vector<std::uint8_t>, double> encode_tensor_bytes_with_scale(
    const std::vector<double>& data,
    const std::string& dtype) {
    const std::string norm = normalize_dtype(dtype);
    const std::size_t expected_bytes = dtype_packed_bytes(norm, data.size());
    std::vector<std::uint8_t> bytes;
    bytes.reserve(expected_bytes);

    if (norm == "F32") {
        for (double x : data) {
            const float f = static_cast<float>(x);
            write_u32_le(bytes, std::bit_cast<std::uint32_t>(f));
        }
        return {std::move(bytes), 1.0};
    }
    if (norm == "F16") {
        for (double x : data) {
            write_u16_le(bytes, double_to_f16_bits(x));
        }
        return {std::move(bytes), 1.0};
    }
    if (norm == "BF16") {
        for (double x : data) {
            write_u16_le(bytes, double_to_bf16_bits(x));
        }
        return {std::move(bytes), 1.0};
    }
    if (norm == "F8_E4M3") {
        for (double x : data) {
            bytes.push_back(double_to_f8_e4m3(x));
        }
        return {std::move(bytes), 1.0};
    }
    if (norm == "I8") {
        for (double x : data) {
            bytes.push_back(double_to_i8(x));
        }
        return {std::move(bytes), 1.0};
    }

    double max_abs = 0.0;
    for (double x : data) {
        const double xv = std::isnan(x) ? 0.0 : x;
        max_abs = std::max(max_abs, std::abs(xv));
    }
    const double scale = (max_abs > 1e-30) ? (max_abs / 7.0) : 1.0;
    bytes.assign(expected_bytes, 0u);
    for (std::size_t i = 0; i < data.size(); ++i) {
        const double xv = std::isnan(data[i]) ? 0.0 : data[i];
        const long long q_ll = std::clamp<long long>(std::llround(xv / scale), -8LL, 7LL);
        const int q = static_cast<int>(q_ll);
        const std::uint8_t nib = static_cast<std::uint8_t>(q < 0 ? (q + 16) : q);
        if ((i & 1u) == 0u) {
            bytes[i / 2u] |= (nib & 0x0Fu);
        } else {
            bytes[i / 2u] |= static_cast<std::uint8_t>((nib & 0x0Fu) << 4);
        }
    }
    return {std::move(bytes), scale};
}

std::vector<double> decode_tensor_bytes(
    const std::uint8_t* data_ptr,
    std::size_t byte_len,
    const std::string& dtype,
    std::size_t num_elems,
    double scale = 1.0) {
    const std::string norm = normalize_dtype(dtype);
    const std::size_t expected_bytes = dtype_packed_bytes(norm, num_elems);
    if (byte_len != expected_bytes) {
        throw std::runtime_error("Safetensors byte length mismatch for tensor");
    }
    std::vector<double> out(num_elems);
    if (norm == "F32") {
        for (std::size_t i = 0; i < num_elems; ++i) {
            const std::uint32_t u = read_u32_le(data_ptr + i * 4);
            out[i] = static_cast<double>(std::bit_cast<float>(u));
        }
    } else if (norm == "F16") {
        for (std::size_t i = 0; i < num_elems; ++i) {
            out[i] = f16_bits_to_double(read_u16_le(data_ptr + i * 2));
        }
    } else if (norm == "BF16") {
        for (std::size_t i = 0; i < num_elems; ++i) {
            out[i] = bf16_bits_to_double(read_u16_le(data_ptr + i * 2));
        }
    } else if (norm == "F8_E4M3") {
        for (std::size_t i = 0; i < num_elems; ++i) {
            out[i] = f8_e4m3_to_double(data_ptr[i]);
        }
    } else if (norm == "I8") {
        for (std::size_t i = 0; i < num_elems; ++i) {
            out[i] = i8_to_double(data_ptr[i]);
        }
    } else {
        for (std::size_t i = 0; i < num_elems; ++i) {
            const std::uint8_t byte = data_ptr[i / 2u];
            const std::uint8_t nib = ((i & 1u) == 0u) ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
            const int q = (nib >= 8u) ? (static_cast<int>(nib) - 16) : static_cast<int>(nib);
            out[i] = static_cast<double>(q) * scale;
        }
    }
    return out;
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

std::string format_shard_name(std::size_t index_1based, std::size_t total_shards) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "model-%05zu-of-%05zu.safetensors", index_1based, total_shards);
    return std::string(buf);
}

}

TensorSpec TensorSpec::from_matrix(const Matrix& m, std::string dtype) {
    if (m.data.size() != m.rows * m.cols) {
        throw std::invalid_argument("Matrix storage size mismatch in TensorSpec::from_matrix");
    }
    TensorSpec spec;
    spec.shape = {m.rows, m.cols};
    spec.dtype = normalize_dtype(std::move(dtype));
    spec.data = m.data;
    return spec;
}

TensorSpec TensorSpec::from_vector(const Vector& v, std::string dtype) {
    TensorSpec spec;
    spec.shape = {v.size()};
    spec.dtype = normalize_dtype(std::move(dtype));
    spec.data = v;
    return spec;
}

Matrix TensorSpec::to_matrix() const {
    if (shape.size() != 2) {
        throw std::invalid_argument("Expected 2D shape in TensorSpec::to_matrix");
    }
    const std::size_t r = shape[0];
    const std::size_t c = shape[1];
    if (data.size() != r * c) {
        throw std::invalid_argument("TensorSpec data length mismatch in to_matrix");
    }
    Matrix out(r, c);
    out.data = data;
    return out;
}

Vector TensorSpec::to_vector() const {
    if (shape.size() != 1) {
        throw std::invalid_argument("Expected 1D shape in TensorSpec::to_vector");
    }
    if (data.size() != shape[0]) {
        throw std::invalid_argument("TensorSpec data length mismatch in to_vector");
    }
    return data;
}

MmapSafetensors::MmapSafetensors(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open safetensors file for mmap: " + path);
    }

    struct stat st{};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("Failed to stat safetensors file: " + path);
    }
    if (st.st_size < 8) {
        ::close(fd);
        throw std::runtime_error("Invalid safetensors file (too short): " + path);
    }

    const std::size_t fsize = static_cast<std::size_t>(st.st_size);
    void* ptr = ::mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error("mmap failed for safetensors file: " + path);
    }

    mapped_ = ptr;
    file_size_ = fsize;

    try {
        const auto* bytes = static_cast<const std::uint8_t*>(mapped_);
        const std::uint64_t header_len_u64 = read_u64_le(bytes);
        if (header_len_u64 > file_size_ - 8u) {
            throw std::runtime_error("Corrupt safetensors header length exceeds file size");
        }
        const std::size_t header_len = static_cast<std::size_t>(header_len_u64);
        data_offset_ = 8u + header_len;
        const std::size_t payload_sz = file_size_ - data_offset_;

        const std::string_view header_view(
            static_cast<const char*>(mapped_) + 8,
            header_len);
        const JsonValue header = parse_json(header_view);
        if (!header.is_object()) {
            throw std::runtime_error("Safetensors header must be a JSON object");
        }

        for (const auto& [name, entry] : header.as_object()) {
            if (name == "__metadata__") {
                if (entry.is_object()) {
                    for (const auto& [mk, mv] : entry.as_object()) {
                        if (mv.is_string()) {
                            metadata_[mk] = mv.as_string();
                        }
                    }
                }
                continue;
            }
            if (!entry.is_object()) {
                throw std::runtime_error("Invalid tensor entry in safetensors header: " + name);
            }
            Entry info;
            info.dtype = normalize_dtype(entry["dtype"].as_string());

            const auto& shape_arr = entry["shape"].as_array();
            info.shape.reserve(shape_arr.size());
            for (const JsonValue& dv : shape_arr) {
                const std::int64_t dim = dv.as_int();
                if (dim < 0) {
                    throw std::runtime_error("Negative dimension in safetensors shape");
                }
                info.shape.push_back(static_cast<std::size_t>(dim));
            }

            const auto& offsets = entry["data_offsets"].as_array();
            if (offsets.size() != 2) {
                throw std::runtime_error("Invalid data_offsets in safetensors header for " + name);
            }
            const std::int64_t start_i = offsets[0].as_int();
            const std::int64_t end_i = offsets[1].as_int();
            if (start_i < 0 || end_i < start_i) {
                throw std::runtime_error("Invalid data_offsets range for " + name);
            }
            info.start = static_cast<std::size_t>(start_i);
            info.end = static_cast<std::size_t>(end_i);
            if (info.end > payload_sz) {
                throw std::runtime_error("Out-of-bounds data_offsets for " + name);
            }
            const std::size_t expected_bytes =
                dtype_packed_bytes(info.dtype, checked_num_elements(info.shape));
            if (info.end - info.start != expected_bytes) {
                throw std::runtime_error("Safetensors byte length mismatch for " + name);
            }
            entries_[name] = std::move(info);
        }
    } catch (...) {
        reset();
        throw;
    }
}

MmapSafetensors::~MmapSafetensors() {
    reset();
}

MmapSafetensors::MmapSafetensors(MmapSafetensors&& other) noexcept
    : mapped_(other.mapped_),
      file_size_(other.file_size_),
      data_offset_(other.data_offset_),
      entries_(std::move(other.entries_)),
      metadata_(std::move(other.metadata_)) {
    other.mapped_ = nullptr;
    other.file_size_ = 0;
    other.data_offset_ = 0;
}

MmapSafetensors& MmapSafetensors::operator=(MmapSafetensors&& other) noexcept {
    if (this != &other) {
        reset();
        mapped_ = other.mapped_;
        file_size_ = other.file_size_;
        data_offset_ = other.data_offset_;
        entries_ = std::move(other.entries_);
        metadata_ = std::move(other.metadata_);
        other.mapped_ = nullptr;
        other.file_size_ = 0;
        other.data_offset_ = 0;
    }
    return *this;
}

void MmapSafetensors::reset() noexcept {
    if (mapped_ != nullptr && mapped_ != MAP_FAILED && file_size_ > 0) {
        ::munmap(mapped_, file_size_);
    }
    mapped_ = nullptr;
    file_size_ = 0;
    data_offset_ = 0;
    entries_.clear();
    metadata_.clear();
}

const MmapSafetensors::Entry& MmapSafetensors::require_entry(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end()) {
        throw std::out_of_range("Missing tensor in safetensors file: " + name);
    }
    return it->second;
}

bool MmapSafetensors::has_tensor(const std::string& name) const {
    return entries_.contains(name);
}

std::vector<std::string> MmapSafetensors::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(entries_.size());
    for (const auto& [name, entry] : entries_) {
        (void)entry;
        names.push_back(name);
    }
    return names;
}

std::vector<std::size_t> MmapSafetensors::tensor_shape(const std::string& name) const {
    return require_entry(name).shape;
}

std::string MmapSafetensors::tensor_dtype(const std::string& name) const {
    return require_entry(name).dtype;
}

std::pair<std::size_t, std::size_t> MmapSafetensors::raw_byte_range(const std::string& name) const {
    const Entry& e = require_entry(name);
    return {e.start, e.end};
}

std::string_view MmapSafetensors::raw_tensor_bytes(const std::string& name) const {
    const Entry& e = require_entry(name);
    return std::string_view(
        static_cast<const char*>(mapped_) + data_offset_ + e.start,
        e.end - e.start);
}

TensorSpec MmapSafetensors::read_tensor(const std::string& name) const {
    const Entry& e = require_entry(name);
    const std::size_t num_elems = checked_num_elements(e.shape);
    const auto* base = static_cast<const std::uint8_t*>(mapped_) + data_offset_ + e.start;
    double scale = 1.0;
    if (e.dtype == "I4") {
        const auto sit = metadata_.find("__scale__" + name);
        if (sit != metadata_.end()) {
            char* end_ptr = nullptr;
            const double parsed = std::strtod(sit->second.c_str(), &end_ptr);
            if (end_ptr != sit->second.c_str() && std::isfinite(parsed)) {
                scale = parsed;
            }
        }
    }
    std::vector<double> data = decode_tensor_bytes(base, e.end - e.start, e.dtype, num_elems, scale);
    return TensorSpec{e.shape, e.dtype, std::move(data)};
}

Vector MmapSafetensors::read_matrix_row(const std::string& name, std::size_t row) const {
    const Entry& e = require_entry(name);
    if (e.shape.size() != 2 || row >= e.shape[0]) {
        throw std::out_of_range("Invalid matrix row for tensor: " + name);
    }
    const auto* base = static_cast<const std::uint8_t*>(mapped_) + data_offset_ + e.start;
    double scale = 1.0;
    if (e.dtype == "I4") {
        const auto it = metadata_.find("__scale__" + name);
        if (it != metadata_.end()) {
            scale = std::stod(it->second);
        }
    }
    const std::size_t cols = e.shape[1];
    Vector out(cols, 0.0);
    const std::size_t offset = row * cols;
    for (std::size_t c = 0; c < cols; ++c) {
        out[c] = decode_value_at(base, e.dtype, offset + c, scale);
    }
    return out;
}

Vector MmapSafetensors::mat_vec(const std::string& name, const Vector& x) const {
    const Entry& e = require_entry(name);
    if (e.shape.size() != 2 || e.shape[1] != x.size()) {
        throw std::invalid_argument("Matrix-vector shape mismatch for tensor: " + name);
    }
    const auto* base = static_cast<const std::uint8_t*>(mapped_) + data_offset_ + e.start;
    double scale = 1.0;
    if (e.dtype == "I4") {
        const auto it = metadata_.find("__scale__" + name);
        if (it != metadata_.end()) {
            scale = std::stod(it->second);
        }
    }
    const std::size_t rows = e.shape[0];
    const std::size_t cols = e.shape[1];
    Vector out(rows, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        long double sum = 0.0;
        const std::size_t offset = r * cols;
        for (std::size_t c = 0; c < cols; ++c) {
            sum += static_cast<long double>(decode_value_at(base, e.dtype, offset + c, scale)) *
                static_cast<long double>(x[c]);
        }
        out[r] = static_cast<double>(sum);
    }
    return out;
}

const std::map<std::string, std::string>& MmapSafetensors::metadata() const {
    return metadata_;
}

void save_safetensors(
    const std::string& path,
    const std::map<std::string, TensorSpec>& tensors,
    const std::map<std::string, std::string>& metadata) {
    const std::filesystem::path out_path(path);
    if (!out_path.parent_path().empty()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::map<std::string, std::string> meta_out = metadata;
    std::vector<std::vector<std::uint8_t>> chunks;
    chunks.reserve(tensors.size());
    std::size_t offset = 0;

    JsonValue header = JsonValue::object();

    for (const auto& [name, spec] : tensors) {
        if (name == "__metadata__") {
            throw std::invalid_argument("Tensor name cannot be '__metadata__'");
        }
        const std::string norm_dtype = normalize_dtype(spec.dtype);
        const std::size_t expected = checked_num_elements(spec.shape);
        if (spec.data.size() != expected) {
            throw std::invalid_argument("TensorSpec data length mismatch for " + name);
        }

        auto [raw, scale] = encode_tensor_bytes_with_scale(spec.data, norm_dtype);
        if (norm_dtype == "I4") {
            meta_out["__scale__" + name] = format_scale_17g(scale);
        }
        const std::size_t end = offset + raw.size();

        JsonValue shape_arr = JsonValue::array();
        for (std::size_t d : spec.shape) {
            shape_arr.as_array().emplace_back(d);
        }
        JsonValue offsets_arr = JsonValue::array();
        offsets_arr.as_array().emplace_back(offset);
        offsets_arr.as_array().emplace_back(end);

        JsonValue entry = JsonValue::object();
        entry["dtype"] = norm_dtype;
        entry["shape"] = std::move(shape_arr);
        entry["data_offsets"] = std::move(offsets_arr);
        header[name] = std::move(entry);

        chunks.push_back(std::move(raw));
        offset = end;
    }

    JsonValue meta_obj = JsonValue::object();
    for (const auto& [k, v] : meta_out) {
        meta_obj[k] = v;
    }
    header["__metadata__"] = std::move(meta_obj);

    std::string header_str = dump_json(header, -1);
    const std::size_t pad_len = (8u - (header_str.size() % 8u)) % 8u;
    header_str.append(pad_len, ' ');

    const std::uint64_t header_len = static_cast<std::uint64_t>(header_str.size());
    std::uint8_t len_buf[8];
    for (int i = 0; i < 8; ++i) {
        len_buf[i] = static_cast<std::uint8_t>((header_len >> (8 * i)) & 0xFFu);
    }

    std::ofstream ofs(out_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open safetensors file for writing: " + path);
    }
    ofs.write(reinterpret_cast<const char*>(len_buf), 8);
    ofs.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
    for (const auto& chunk : chunks) {
        if (!chunk.empty()) {
            ofs.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(chunk.size()));
        }
    }
    if (!ofs.good()) {
        throw std::runtime_error("Error writing safetensors file: " + path);
    }
}

std::pair<std::map<std::string, TensorSpec>, std::map<std::string, std::string>>
load_safetensors(const std::string& path) {
    MmapSafetensors reader(path);
    std::map<std::string, TensorSpec> tensors;
    for (const std::string& name : reader.tensor_names()) {
        tensors[name] = reader.read_tensor(name);
    }
    return {std::move(tensors), reader.metadata()};
}

void save_safetensors_sharded(
    const std::string& dir,
    const std::map<std::string, TensorSpec>& tensors,
    std::size_t max_shard_bytes,
    const std::map<std::string, std::string>& metadata) {
    const std::filesystem::path out_dir(dir);
    std::filesystem::create_directories(out_dir);

    std::vector<std::map<std::string, TensorSpec>> shards;
    std::map<std::string, TensorSpec> current;
    std::size_t current_bytes = 0;
    std::size_t total_bytes = 0;

    for (const auto& [name, spec] : tensors) {
        if (name == "__metadata__") {
            throw std::invalid_argument("Tensor name cannot be '__metadata__'");
        }
        const std::string norm_dtype = normalize_dtype(spec.dtype);
        const std::size_t num_elems = checked_num_elements(spec.shape);
        if (spec.data.size() != num_elems) {
            throw std::invalid_argument("TensorSpec data length mismatch for " + name);
        }
        const std::size_t tensor_bytes = dtype_packed_bytes(norm_dtype, num_elems);
        total_bytes += tensor_bytes;

        if (!current.empty() && max_shard_bytes > 0 && current_bytes + tensor_bytes > max_shard_bytes) {
            shards.push_back(std::move(current));
            current.clear();
            current_bytes = 0;
        }
        current[name] = spec;
        current_bytes += tensor_bytes;
    }
    if (!current.empty() || shards.empty()) {
        shards.push_back(std::move(current));
    }

    if (shards.size() == 1) {
        std::error_code ec;
        std::filesystem::remove(out_dir / "model.safetensors.index.json", ec);
        save_safetensors((out_dir / "model.safetensors").string(), shards[0], metadata);
        return;
    }

    {
        std::error_code ec;
        std::filesystem::remove(out_dir / "model.safetensors", ec);
    }

    JsonValue index_json = JsonValue::object();
    JsonValue index_meta = JsonValue::object();
    index_meta["total_size"] = total_bytes;
    index_json["metadata"] = std::move(index_meta);
    JsonValue weight_map = JsonValue::object();

    const std::size_t total_shards = shards.size();
    for (std::size_t i = 0; i < total_shards; ++i) {
        const std::string shard_name = format_shard_name(i + 1, total_shards);
        save_safetensors((out_dir / shard_name).string(), shards[i], metadata);
        for (const auto& [tensor_name, spec] : shards[i]) {
            (void)spec;
            weight_map[tensor_name] = shard_name;
        }
    }
    index_json["weight_map"] = std::move(weight_map);

    const std::string idx_path = (out_dir / "model.safetensors.index.json").string();
    std::ofstream ofs(idx_path, std::ios::trunc);
    if (!ofs) {
        throw std::runtime_error("Failed to open index file for writing: " + idx_path);
    }
    ofs << dump_json(index_json, 2) << '\n';
    if (!ofs.good()) {
        throw std::runtime_error("Failed writing index file: " + idx_path);
    }
}

std::pair<std::map<std::string, TensorSpec>, std::map<std::string, std::string>>
load_safetensors_sharded(const std::string& dir_or_file) {
    const std::filesystem::path p(dir_or_file);
    if (std::filesystem::is_regular_file(p)) {
        return load_safetensors(dir_or_file);
    }
    if (!std::filesystem::is_directory(p)) {
        throw std::runtime_error("Safetensors path does not exist: " + dir_or_file);
    }

    const std::filesystem::path index_path = p / "model.safetensors.index.json";
    if (std::filesystem::exists(index_path)) {
        const JsonValue idx = parse_json(read_text_file(index_path));
        if (!idx.is_object() || !idx.has("weight_map") || !idx["weight_map"].is_object()) {
            throw std::runtime_error("Invalid model.safetensors.index.json");
        }
        std::map<std::string, std::vector<std::string>> shard_to_tensors;
        for (const auto& [tname, sname_v] : idx["weight_map"].as_object()) {
            shard_to_tensors[sname_v.as_string()].push_back(tname);
        }
        std::map<std::string, TensorSpec> tensors;
        std::map<std::string, std::string> metadata;
        for (const auto& [shard_file, tnames] : shard_to_tensors) {
            MmapSafetensors reader((p / shard_file).string());
            for (const auto& [mk, mv] : reader.metadata()) {
                metadata[mk] = mv;
            }
            for (const std::string& tname : tnames) {
                tensors[tname] = reader.read_tensor(tname);
            }
        }
        return {std::move(tensors), std::move(metadata)};
    }

    const std::filesystem::path single_path = p / "model.safetensors";
    if (std::filesystem::exists(single_path)) {
        return load_safetensors(single_path.string());
    }
    throw std::runtime_error("No model.safetensors or model.safetensors.index.json found in: " + dir_or_file);
}

TensorSpec load_single_tensor_sharded(
    const std::string& dir_or_file,
    const std::string& tensor_name) {
    const std::filesystem::path p(dir_or_file);
    if (std::filesystem::is_regular_file(p)) {
        MmapSafetensors reader(dir_or_file);
        return reader.read_tensor(tensor_name);
    }
    if (!std::filesystem::is_directory(p)) {
        throw std::runtime_error("Safetensors path does not exist: " + dir_or_file);
    }

    const std::filesystem::path index_path = p / "model.safetensors.index.json";
    if (std::filesystem::exists(index_path)) {
        const JsonValue idx = parse_json(read_text_file(index_path));
        if (!idx.is_object() || !idx.has("weight_map") || !idx["weight_map"].is_object()) {
            throw std::runtime_error("Invalid model.safetensors.index.json");
        }
        const JsonValue& wmap = idx["weight_map"];
        if (!wmap.has(tensor_name)) {
            throw std::out_of_range("Tensor not found in weight_map: " + tensor_name);
        }
        const std::string shard_file = wmap[tensor_name].as_string();
        MmapSafetensors reader((p / shard_file).string());
        return reader.read_tensor(tensor_name);
    }

    const std::filesystem::path single_path = p / "model.safetensors";
    if (std::filesystem::exists(single_path)) {
        MmapSafetensors reader(single_path.string());
        return reader.read_tensor(tensor_name);
    }
    throw std::runtime_error("No model.safetensors or model.safetensors.index.json found in: " + dir_or_file);
}

void rewrite_safetensors_selective(
    const std::string& src_path,
    const std::string& dst_path,
    const std::map<std::string, TensorSpec>& replacements,
    const std::map<std::string, TensorSpec>& additions,
    const std::set<std::string>& removals,
    std::int64_t* out_byte_delta) {
    auto reader_ptr = std::make_unique<MmapSafetensors>(src_path);

    std::size_t orig_data_bytes = 0;
    std::set<std::string> final_names;
    for (const std::string& name : reader_ptr->tensor_names()) {
        orig_data_bytes += reader_ptr->raw_tensor_bytes(name).size();
        if (!removals.contains(name) || replacements.contains(name) || additions.contains(name)) {
            final_names.insert(name);
        }
    }
    for (const auto& [k, spec] : replacements) {
        (void)spec;
        if (k == "__metadata__") {
            throw std::invalid_argument("Tensor name cannot be '__metadata__'");
        }
        final_names.insert(k);
    }
    for (const auto& [k, spec] : additions) {
        (void)spec;
        if (k == "__metadata__") {
            throw std::invalid_argument("Tensor name cannot be '__metadata__'");
        }
        final_names.insert(k);
    }

    std::map<std::string, std::string> meta_out = reader_ptr->metadata();
    for (const std::string& rem : removals) {
        if (!replacements.contains(rem) && !additions.contains(rem)) {
            meta_out.erase("__scale__" + rem);
        }
    }

    struct ChunkRef {
        std::string_view raw_view;
        std::vector<std::uint8_t> owned;
        bool is_owned = false;
    };

    std::vector<ChunkRef> chunks;
    chunks.reserve(final_names.size());
    JsonValue header = JsonValue::object();
    std::size_t offset = 0;

    for (const std::string& name : final_names) {
        const auto add_it = additions.find(name);
        const auto rep_it = replacements.find(name);
        std::string norm_dtype;
        std::vector<std::size_t> shape;
        ChunkRef chunk;

        if (add_it != additions.end() || rep_it != replacements.end()) {
            const TensorSpec& spec = (add_it != additions.end()) ? add_it->second : rep_it->second;
            norm_dtype = normalize_dtype(spec.dtype);
            shape = spec.shape;
            const std::size_t expected = checked_num_elements(shape);
            if (spec.data.size() != expected) {
                throw std::invalid_argument("TensorSpec data length mismatch for " + name);
            }
            auto [encoded, scale] = encode_tensor_bytes_with_scale(spec.data, norm_dtype);
            if (norm_dtype == "I4") {
                meta_out["__scale__" + name] = format_scale_17g(scale);
            } else {
                meta_out.erase("__scale__" + name);
            }
            chunk.owned = std::move(encoded);
            chunk.is_owned = true;
        } else {
            norm_dtype = reader_ptr->tensor_dtype(name);
            shape = reader_ptr->tensor_shape(name);
            chunk.raw_view = reader_ptr->raw_tensor_bytes(name);
            chunk.is_owned = false;
        }

        const std::size_t byte_len = chunk.is_owned ? chunk.owned.size() : chunk.raw_view.size();
        const std::size_t end = offset + byte_len;

        JsonValue shape_arr = JsonValue::array();
        for (std::size_t d : shape) {
            shape_arr.as_array().emplace_back(d);
        }
        JsonValue offsets_arr = JsonValue::array();
        offsets_arr.as_array().emplace_back(offset);
        offsets_arr.as_array().emplace_back(end);

        JsonValue entry = JsonValue::object();
        entry["dtype"] = norm_dtype;
        entry["shape"] = std::move(shape_arr);
        entry["data_offsets"] = std::move(offsets_arr);
        header[name] = std::move(entry);

        chunks.push_back(std::move(chunk));
        offset = end;
    }

    JsonValue meta_obj = JsonValue::object();
    for (const auto& [k, v] : meta_out) {
        meta_obj[k] = v;
    }
    header["__metadata__"] = std::move(meta_obj);

    std::string header_str = dump_json(header, -1);
    const std::size_t pad_len = (8u - (header_str.size() % 8u)) % 8u;
    header_str.append(pad_len, ' ');

    const std::uint64_t header_len = static_cast<std::uint64_t>(header_str.size());
    std::uint8_t len_buf[8];
    for (int i = 0; i < 8; ++i) {
        len_buf[i] = static_cast<std::uint8_t>((header_len >> (8 * i)) & 0xFFu);
    }

    const std::filesystem::path dst_fs(dst_path);
    const std::filesystem::path parent_dir = dst_fs.parent_path();
    if (!parent_dir.empty()) {
        std::filesystem::create_directories(parent_dir);
    }
    const std::filesystem::path tmp_path =
        (parent_dir.empty() ? std::filesystem::path(".") : parent_dir) /
        (dst_fs.filename().string() + ".tmp." + std::to_string(::getpid()));

    try {
        {
            std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
            if (!ofs) {
                throw std::runtime_error("Failed to open temp safetensors file for writing: " + tmp_path.string());
            }
            ofs.write(reinterpret_cast<const char*>(len_buf), 8);
            ofs.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
            for (const ChunkRef& chunk : chunks) {
                if (chunk.is_owned) {
                    if (!chunk.owned.empty()) {
                        ofs.write(
                            reinterpret_cast<const char*>(chunk.owned.data()),
                            static_cast<std::streamsize>(chunk.owned.size()));
                    }
                } else if (!chunk.raw_view.empty()) {
                    ofs.write(
                        chunk.raw_view.data(),
                        static_cast<std::streamsize>(chunk.raw_view.size()));
                }
            }
            if (!ofs.good()) {
                throw std::runtime_error("Error writing temp safetensors file: " + tmp_path.string());
            }
        }
        reader_ptr.reset();
        std::filesystem::rename(tmp_path, dst_fs);
    } catch (...) {
        std::error_code ec;
        std::filesystem::remove(tmp_path, ec);
        throw;
    }

    if (out_byte_delta != nullptr) {
        *out_byte_delta = static_cast<std::int64_t>(offset) - static_cast<std::int64_t>(orig_data_bytes);
    }
}

}
