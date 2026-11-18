#pragma once

#include "aic/tensor.hpp"

#include <memory>
#include <string>

namespace aic {

class MetalContext {
public:
    static MetalContext& instance();

    bool is_available() const;
    std::string device_name() const;
    Matrix mat_mul_f32(const Matrix& A, const Matrix& B) const;
    void add_rank1_f32(Matrix& W, const Vector& u, const Vector& v, double scale) const;
    Vector swiglu_poles_f32(const Vector& gate_pre, const Vector& up_val) const;
    double fisher_inner_f32(
        const Matrix& a,
        const Matrix& b,
        const Vector& g_out,
        const Vector& g_in) const;

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

private:
    MetalContext();
    ~MetalContext();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

bool metal_is_available();
std::string metal_device_name();
Matrix metal_mat_mul(const Matrix& A, const Matrix& B);
void metal_add_rank1(Matrix& W, const Vector& u, const Vector& v, double scale);
Vector metal_swiglu_poles(const Vector& gate_pre, const Vector& up_val);
double metal_fisher_inner(
    const Matrix& a,
    const Matrix& b,
    const Vector& g_out,
    const Vector& g_in);

}
