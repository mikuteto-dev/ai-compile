#include "aic/accel_metal.hpp"

#ifndef __APPLE__

#include <stdexcept>

namespace aic {

struct MetalContext::Impl {};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}
MetalContext::~MetalContext() = default;

MetalContext& MetalContext::instance() {
    static MetalContext context;
    return context;
}

bool MetalContext::is_available() const {
    return false;
}

std::string MetalContext::device_name() const {
    return "unavailable";
}

Matrix MetalContext::mat_mul_f32(const Matrix& a, const Matrix& b) const {
    return mat_mul(a, b);
}

void MetalContext::add_rank1_f32(
    Matrix& weight,
    const Vector& u,
    const Vector& v,
    double scale) const {
    if (weight.rows != u.size() || weight.cols != v.size()) {
        throw std::invalid_argument("Dimension mismatch in metal_add_rank1 fallback");
    }
    for (std::size_t r = 0; r < weight.rows; ++r) {
        for (std::size_t c = 0; c < weight.cols; ++c) {
            weight(r, c) += scale * u[r] * v[c];
        }
    }
}

Vector MetalContext::swiglu_poles_f32(
    const Vector& gate_pre,
    const Vector& up_val) const {
    if (gate_pre.size() != up_val.size()) {
        throw std::invalid_argument("Length mismatch in metal_swiglu_poles fallback");
    }
    Vector out(gate_pre.size(), 0.0);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = silu(gate_pre[i]) * up_val[i];
    }
    return out;
}

double MetalContext::fisher_inner_f32(
    const Matrix& a,
    const Matrix& b,
    const Vector& g_out,
    const Vector& g_in) const {
    if (a.rows != b.rows || a.cols != b.cols ||
        g_out.size() != a.rows || g_in.size() != a.cols) {
        throw std::invalid_argument("Shape mismatch in metal_fisher_inner fallback");
    }
    double total = 0.0;
    for (std::size_t r = 0; r < a.rows; ++r) {
        for (std::size_t c = 0; c < a.cols; ++c) {
            total += g_out[r] * g_in[c] * a(r, c) * b(r, c);
        }
    }
    return total;
}

bool metal_is_available() {
    return false;
}

std::string metal_device_name() {
    return "unavailable";
}

Matrix metal_mat_mul(const Matrix& a, const Matrix& b) {
    return MetalContext::instance().mat_mul_f32(a, b);
}

void metal_add_rank1(Matrix& weight, const Vector& u, const Vector& v, double scale) {
    MetalContext::instance().add_rank1_f32(weight, u, v, scale);
}

Vector metal_swiglu_poles(const Vector& gate_pre, const Vector& up_val) {
    return MetalContext::instance().swiglu_poles_f32(gate_pre, up_val);
}

double metal_fisher_inner(
    const Matrix& a,
    const Matrix& b,
    const Vector& g_out,
    const Vector& g_in) {
    return MetalContext::instance().fisher_inner_f32(a, b, g_out, g_in);
}

} // namespace aic

#endif
