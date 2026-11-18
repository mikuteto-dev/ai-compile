#include "aic/accel_metal.hpp"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace aic {

namespace {

struct MatMulDims {
    std::uint32_t M;
    std::uint32_t N;
    std::uint32_t K;
};

struct Rank1Params {
    std::uint32_t rows;
    std::uint32_t cols;
    float scale;
};

struct FisherParams {
    std::uint32_t rows;
    std::uint32_t cols;
};

const char* kMetalKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct MatMulDims {
    uint M;
    uint N;
    uint K;
};

struct Rank1Params {
    uint rows;
    uint cols;
    float scale;
};

struct FisherParams {
    uint rows;
    uint cols;
};

kernel void matmul_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatMulDims& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dims.N || gid.y >= dims.M) {
        return;
    }
    float acc = 0.0f;
    uint a_row = gid.y * dims.K;
    for (uint k = 0; k < dims.K; ++k) {
        acc += A[a_row + k] * B[k * dims.N + gid.x];
    }
    C[gid.y * dims.N + gid.x] = acc;
}

kernel void swiglu_poles_f32(
    device const float* gate_pre [[buffer(0)]],
    device const float* up_val [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= count) {
        return;
    }
    float g = gate_pre[gid];
    float sig = (g >= 0.0f) ? (1.0f / (1.0f + exp(-g))) : (exp(g) / (1.0f + exp(g)));
    out[gid] = (g * sig) * up_val[gid];
}

kernel void add_rank1_f32(
    device float* W [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const float* v [[buffer(2)]],
    constant Rank1Params& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= params.cols || gid.y >= params.rows) {
        return;
    }
    W[gid.y * params.cols + gid.x] += params.scale * u[gid.y] * v[gid.x];
}

kernel void fisher_inner_f32(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* g_out [[buffer(2)]],
    device const float* g_in [[buffer(3)]],
    device float* partials [[buffer(4)]],
    constant FisherParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint grid_size [[threads_per_grid]],
    uint tpg [[threads_per_threadgroup]]) {
    uint total = params.rows * params.cols;
    float local_sum = 0.0f;
    for (uint idx = gid; idx < total; idx += grid_size) {
        uint r = idx / params.cols;
        uint c = idx % params.cols;
        local_sum += g_out[r] * g_in[c] * A[idx] * B[idx];
    }
    threadgroup float tg_buf[256];
    tg_buf[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tpg >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            tg_buf[lid] += tg_buf[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partials[group_id] = tg_buf[0];
    }
}
)MSL";

double cpu_fisher_inner(
    const Matrix& a,
    const Matrix& b,
    const Vector& g_out,
    const Vector& g_in) {
    const std::size_t rows = a.rows;
    const std::size_t cols = a.cols;
    if (rows == 0 || cols == 0) {
        return 0.0;
    }
    Vector scaled_row(cols, 0.0);
    double total = 0.0;
    const bool use_blas = cols <= static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t r = 0; r < rows; ++r) {
        const double* a_ptr = a.data.data() + r * cols;
        const double* b_ptr = b.data.data() + r * cols;
        double row_dot = 0.0;
        if (use_blas) {
            vDSP_vmulD(a_ptr, 1, g_in.data(), 1, scaled_row.data(), 1, static_cast<vDSP_Length>(cols));
            row_dot = cblas_ddot(static_cast<int>(cols), scaled_row.data(), 1, b_ptr, 1);
        } else {
            for (std::size_t c = 0; c < cols; ++c) {
                row_dot += g_in[c] * a_ptr[c] * b_ptr[c];
            }
        }
        total += g_out[r] * row_dot;
    }
    return total;
}

}

struct MetalContext::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> pso_matmul = nil;
    id<MTLComputePipelineState> pso_swiglu = nil;
    id<MTLComputePipelineState> pso_rank1 = nil;
    id<MTLComputePipelineState> pso_fisher = nil;
    std::string device_name = "unavailable";
    bool ready = false;

    Impl() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                return;
            }
            if ([device name]) {
                device_name = std::string([[device name] UTF8String]);
            }
            queue = [device newCommandQueue];
            if (!queue) {
                return;
            }
            NSString* src = [NSString stringWithUTF8String:kMetalKernelSource];
            if (!src) {
                return;
            }
            NSError* err = nil;
            library = [device newLibraryWithSource:src options:nil error:&err];
            if (!library) {
                return;
            }
            id<MTLFunction> fn_matmul = [library newFunctionWithName:@"matmul_f32"];
            id<MTLFunction> fn_swiglu = [library newFunctionWithName:@"swiglu_poles_f32"];
            id<MTLFunction> fn_rank1 = [library newFunctionWithName:@"add_rank1_f32"];
            id<MTLFunction> fn_fisher = [library newFunctionWithName:@"fisher_inner_f32"];
            if (!fn_matmul || !fn_swiglu || !fn_rank1 || !fn_fisher) {
                return;
            }
            pso_matmul = [device newComputePipelineStateWithFunction:fn_matmul error:&err];
            pso_swiglu = [device newComputePipelineStateWithFunction:fn_swiglu error:&err];
            pso_rank1 = [device newComputePipelineStateWithFunction:fn_rank1 error:&err];
            pso_fisher = [device newComputePipelineStateWithFunction:fn_fisher error:&err];
            if (!pso_matmul || !pso_swiglu || !pso_rank1 || !pso_fisher) {
                return;
            }
            ready = true;
        }
    }
};

MetalContext::MetalContext() : impl_(std::make_unique<Impl>()) {}
MetalContext::~MetalContext() = default;

MetalContext& MetalContext::instance() {
    static MetalContext ctx;
    return ctx;
}

bool MetalContext::is_available() const {
    return impl_->ready;
}

std::string MetalContext::device_name() const {
    return impl_->device_name;
}

Matrix MetalContext::mat_mul_f32(const Matrix& A, const Matrix& B) const {
    if (A.data.size() != A.rows * A.cols || B.data.size() != B.rows * B.cols) {
        throw std::invalid_argument("Matrix storage size mismatch in metal_mat_mul");
    }
    if (A.cols != B.rows) {
        throw std::invalid_argument("Dimension mismatch in metal_mat_mul");
    }
    const std::size_t M = A.rows;
    const std::size_t N = B.cols;
    const std::size_t K = A.cols;
    if (M == 0 || N == 0 || K == 0) {
        return Matrix(M, N, 0.0);
    }

    if (!impl_->ready || M > 0xFFFFFFFFu || N > 0xFFFFFFFFu || K > 0xFFFFFFFFu) {
        return mat_mul(A, B);
    }

    @autoreleasepool {
        std::vector<float> a_f32(A.data.size());
        std::vector<float> b_f32(B.data.size());
        for (std::size_t i = 0; i < A.data.size(); ++i) {
            a_f32[i] = static_cast<float>(A.data[i]);
        }
        for (std::size_t i = 0; i < B.data.size(); ++i) {
            b_f32[i] = static_cast<float>(B.data[i]);
        }

        const NSUInteger a_bytes = static_cast<NSUInteger>(a_f32.size() * sizeof(float));
        const NSUInteger b_bytes = static_cast<NSUInteger>(b_f32.size() * sizeof(float));
        const NSUInteger c_bytes = static_cast<NSUInteger>(M * N * sizeof(float));

        id<MTLBuffer> buf_a = [impl_->device newBufferWithBytes:a_f32.data()
                                                           length:a_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [impl_->device newBufferWithBytes:b_f32.data()
                                                           length:b_bytes
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_c = [impl_->device newBufferWithLength:c_bytes
                                                           options:MTLResourceStorageModeShared];
        if (!buf_a || !buf_b || !buf_c) {
            return mat_mul(A, B);
        }

        const MatMulDims dims{
            static_cast<std::uint32_t>(M),
            static_cast<std::uint32_t>(N),
            static_cast<std::uint32_t>(K)};

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            return mat_mul(A, B);
        }

        [enc setComputePipelineState:impl_->pso_matmul];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_c offset:0 atIndex:2];
        [enc setBytes:&dims length:sizeof(dims) atIndex:3];

        const NSUInteger max_t = [impl_->pso_matmul maxTotalThreadsPerThreadgroup];
        const NSUInteger tg_side = (max_t >= 256) ? 16 : 8;
        const MTLSize tg_size = MTLSizeMake(tg_side, tg_side, 1);
        const MTLSize tg_groups = MTLSizeMake(
            (static_cast<NSUInteger>(N) + tg_side - 1) / tg_side,
            (static_cast<NSUInteger>(M) + tg_side - 1) / tg_side,
            1);
        [enc dispatchThreadgroups:tg_groups threadsPerThreadgroup:tg_size];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            return mat_mul(A, B);
        }

        const float* c_ptr = static_cast<const float*>([buf_c contents]);
        Matrix out(M, N, 0.0);
        for (std::size_t i = 0; i < M * N; ++i) {
            out.data[i] = static_cast<double>(c_ptr[i]);
        }
        return out;
    }
}

void MetalContext::add_rank1_f32(Matrix& W, const Vector& u, const Vector& v, double scale) const {
    if (W.data.size() != W.rows * W.cols) {
        throw std::invalid_argument("Matrix storage size mismatch in metal_add_rank1");
    }
    if (W.rows != u.size() || W.cols != v.size()) {
        throw std::invalid_argument("Dimension mismatch in metal_add_rank1");
    }
    if (W.rows == 0 || W.cols == 0) {
        return;
    }

    auto cpu_fallback = [&]() {
        for (std::size_t r = 0; r < W.rows; ++r) {
            const double su = scale * u[r];
            for (std::size_t c = 0; c < W.cols; ++c) {
                W(r, c) += su * v[c];
            }
        }
    };

    if (!impl_->ready || W.rows > 0xFFFFFFFFu || W.cols > 0xFFFFFFFFu) {
        cpu_fallback();
        return;
    }

    @autoreleasepool {
        std::vector<float> w_f32(W.data.size());
        std::vector<float> u_f32(u.size());
        std::vector<float> v_f32(v.size());
        for (std::size_t i = 0; i < W.data.size(); ++i) {
            w_f32[i] = static_cast<float>(W.data[i]);
        }
        for (std::size_t i = 0; i < u.size(); ++i) {
            u_f32[i] = static_cast<float>(u[i]);
        }
        for (std::size_t i = 0; i < v.size(); ++i) {
            v_f32[i] = static_cast<float>(v[i]);
        }

        id<MTLBuffer> buf_w = [impl_->device newBufferWithBytes:w_f32.data()
                                                           length:static_cast<NSUInteger>(w_f32.size() * sizeof(float))
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_u = [impl_->device newBufferWithBytes:u_f32.data()
                                                           length:static_cast<NSUInteger>(u_f32.size() * sizeof(float))
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_v = [impl_->device newBufferWithBytes:v_f32.data()
                                                           length:static_cast<NSUInteger>(v_f32.size() * sizeof(float))
                                                          options:MTLResourceStorageModeShared];
        if (!buf_w || !buf_u || !buf_v) {
            cpu_fallback();
            return;
        }

        const Rank1Params params{
            static_cast<std::uint32_t>(W.rows),
            static_cast<std::uint32_t>(W.cols),
            static_cast<float>(scale)};

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            cpu_fallback();
            return;
        }

        [enc setComputePipelineState:impl_->pso_rank1];
        [enc setBuffer:buf_w offset:0 atIndex:0];
        [enc setBuffer:buf_u offset:0 atIndex:1];
        [enc setBuffer:buf_v offset:0 atIndex:2];
        [enc setBytes:&params length:sizeof(params) atIndex:3];

        const NSUInteger max_t = [impl_->pso_rank1 maxTotalThreadsPerThreadgroup];
        const NSUInteger tg_side = (max_t >= 256) ? 16 : 8;
        const MTLSize tg_size = MTLSizeMake(tg_side, tg_side, 1);
        const MTLSize tg_groups = MTLSizeMake(
            (static_cast<NSUInteger>(W.cols) + tg_side - 1) / tg_side,
            (static_cast<NSUInteger>(W.rows) + tg_side - 1) / tg_side,
            1);
        [enc dispatchThreadgroups:tg_groups threadsPerThreadgroup:tg_size];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            cpu_fallback();
            return;
        }

        const float* w_ptr = static_cast<const float*>([buf_w contents]);
        for (std::size_t i = 0; i < W.data.size(); ++i) {
            W.data[i] = static_cast<double>(w_ptr[i]);
        }
    }
}

Vector MetalContext::swiglu_poles_f32(const Vector& gate_pre, const Vector& up_val) const {
    if (gate_pre.size() != up_val.size()) {
        throw std::invalid_argument("Length mismatch in metal_swiglu_poles");
    }
    const std::size_t n = gate_pre.size();
    if (n == 0) {
        return {};
    }

    auto cpu_fallback = [&]() {
        Vector out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = silu(gate_pre[i]) * up_val[i];
        }
        return out;
    };

    if (!impl_->ready || n > 0xFFFFFFFFu) {
        return cpu_fallback();
    }

    @autoreleasepool {
        std::vector<float> g_f32(n);
        std::vector<float> u_f32(n);
        for (std::size_t i = 0; i < n; ++i) {
            g_f32[i] = static_cast<float>(gate_pre[i]);
            u_f32[i] = static_cast<float>(up_val[i]);
        }

        const NSUInteger bytes = static_cast<NSUInteger>(n * sizeof(float));
        id<MTLBuffer> buf_g = [impl_->device newBufferWithBytes:g_f32.data() length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_u = [impl_->device newBufferWithBytes:u_f32.data() length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_o = [impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf_g || !buf_u || !buf_o) {
            return cpu_fallback();
        }

        const std::uint32_t count = static_cast<std::uint32_t>(n);
        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            return cpu_fallback();
        }

        [enc setComputePipelineState:impl_->pso_swiglu];
        [enc setBuffer:buf_g offset:0 atIndex:0];
        [enc setBuffer:buf_u offset:0 atIndex:1];
        [enc setBuffer:buf_o offset:0 atIndex:2];
        [enc setBytes:&count length:sizeof(count) atIndex:3];

        const NSUInteger tg_w = std::min<NSUInteger>([impl_->pso_swiglu maxTotalThreadsPerThreadgroup], 256);
        const MTLSize tg_size = MTLSizeMake(tg_w, 1, 1);
        const MTLSize tg_groups = MTLSizeMake((static_cast<NSUInteger>(n) + tg_w - 1) / tg_w, 1, 1);
        [enc dispatchThreadgroups:tg_groups threadsPerThreadgroup:tg_size];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            return cpu_fallback();
        }

        const float* o_ptr = static_cast<const float*>([buf_o contents]);
        Vector out(n);
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = static_cast<double>(o_ptr[i]);
        }
        return out;
    }
}

double MetalContext::fisher_inner_f32(
    const Matrix& a,
    const Matrix& b,
    const Vector& g_out,
    const Vector& g_in) const {
    if (a.data.size() != a.rows * a.cols || b.data.size() != b.rows * b.cols) {
        throw std::invalid_argument("Matrix storage size mismatch in fisher_inner_f32");
    }
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::invalid_argument("Matrix shape mismatch in fisher_inner_f32");
    }
    if (g_out.size() != a.rows || g_in.size() != a.cols) {
        throw std::invalid_argument("Metric vector length mismatch in fisher_inner_f32");
    }
    if (a.rows == 0 || a.cols == 0) {
        return 0.0;
    }

    const std::size_t total = a.rows * a.cols;
    if (!impl_->ready || total < 4096u || a.rows > 0xFFFFFFFFu || a.cols > 0xFFFFFFFFu ||
        total > 0xFFFFFFFFu) {
        return cpu_fisher_inner(a, b, g_out, g_in);
    }

    @autoreleasepool {
        std::vector<float> a_f32(total);
        std::vector<float> b_f32(total);
        std::vector<float> go_f32(a.rows);
        std::vector<float> gi_f32(a.cols);
        for (std::size_t i = 0; i < total; ++i) {
            a_f32[i] = static_cast<float>(a.data[i]);
            b_f32[i] = static_cast<float>(b.data[i]);
        }
        for (std::size_t i = 0; i < a.rows; ++i) {
            go_f32[i] = static_cast<float>(g_out[i]);
        }
        for (std::size_t j = 0; j < a.cols; ++j) {
            gi_f32[j] = static_cast<float>(g_in[j]);
        }

        const NSUInteger max_t = [impl_->pso_fisher maxTotalThreadsPerThreadgroup];
        const NSUInteger tg_w = (max_t >= 256) ? 256 : ((max_t >= 128) ? 128 : 64);
        const NSUInteger num_groups = std::min<NSUInteger>(
            (static_cast<NSUInteger>(total) + tg_w - 1) / tg_w, 256);

        id<MTLBuffer> buf_a = [impl_->device newBufferWithBytes:a_f32.data()
                                                           length:static_cast<NSUInteger>(total * sizeof(float))
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [impl_->device newBufferWithBytes:b_f32.data()
                                                           length:static_cast<NSUInteger>(total * sizeof(float))
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_go = [impl_->device newBufferWithBytes:go_f32.data()
                                                            length:static_cast<NSUInteger>(a.rows * sizeof(float))
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_gi = [impl_->device newBufferWithBytes:gi_f32.data()
                                                            length:static_cast<NSUInteger>(a.cols * sizeof(float))
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_p = [impl_->device newBufferWithLength:static_cast<NSUInteger>(num_groups * sizeof(float))
                                                           options:MTLResourceStorageModeShared];
        if (!buf_a || !buf_b || !buf_go || !buf_gi || !buf_p) {
            return cpu_fisher_inner(a, b, g_out, g_in);
        }

        const FisherParams params{
            static_cast<std::uint32_t>(a.rows),
            static_cast<std::uint32_t>(a.cols)};

        id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        if (!cmd || !enc) {
            return cpu_fisher_inner(a, b, g_out, g_in);
        }

        [enc setComputePipelineState:impl_->pso_fisher];
        [enc setBuffer:buf_a offset:0 atIndex:0];
        [enc setBuffer:buf_b offset:0 atIndex:1];
        [enc setBuffer:buf_go offset:0 atIndex:2];
        [enc setBuffer:buf_gi offset:0 atIndex:3];
        [enc setBuffer:buf_p offset:0 atIndex:4];
        [enc setBytes:&params length:sizeof(params) atIndex:5];

        [enc dispatchThreadgroups:MTLSizeMake(num_groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_w, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        if ([cmd status] != MTLCommandBufferStatusCompleted) {
            return cpu_fisher_inner(a, b, g_out, g_in);
        }

        const float* partials = static_cast<const float*>([buf_p contents]);
        double total_sum = 0.0;
        for (NSUInteger g = 0; g < num_groups; ++g) {
            total_sum += static_cast<double>(partials[g]);
        }
        return total_sum;
    }
}

bool metal_is_available() {
    return MetalContext::instance().is_available();
}

std::string metal_device_name() {
    return MetalContext::instance().device_name();
}

Matrix metal_mat_mul(const Matrix& A, const Matrix& B) {
    return MetalContext::instance().mat_mul_f32(A, B);
}

void metal_add_rank1(Matrix& W, const Vector& u, const Vector& v, double scale) {
    MetalContext::instance().add_rank1_f32(W, u, v, scale);
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

}
