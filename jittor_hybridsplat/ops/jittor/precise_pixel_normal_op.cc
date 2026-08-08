#include "precise_pixel_normal_op.h"

namespace jittor {

#ifndef JIT

PrecisePixelNormalOp::PrecisePixelNormalOp(Var* normal_raw) : normal_raw(normal_raw) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(normal_raw->dtype() == ns_float32)
        << "precise_pixel_normal input must be FP32";
    CHECK(
        normal_raw->shape.size() >= 1 &&
        normal_raw->shape[normal_raw->shape.size() - 1] == 3)
        << "precise_pixel_normal input must have final extent 3";
    output = create_output(normal_raw->shape, ns_float32);
}

void PrecisePixelNormalOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "VERSION", JK::hex1(1));
}

#else

#ifdef JIT_cuda

__device__ __forceinline__ float precise_pixel_normal_divide_rn_f32(
    float numerator,
    float denominator) {
    float result;
    asm volatile(
        "div.rn.f32 %0, %1, %2;"
        : "=f"(result)
        : "f"(numerator), "f"(denominator));
    return result;
}

__global__ static void precise_pixel_normal_l2_kernel(
    const float* normal_raw,
    float* output,
    index_t vector_count) {
    const index_t vector_index =
        static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector_index >= vector_count) {
        return;
    }
    const index_t offset = vector_index * 3;
    const float x = normal_raw[offset + 0];
    const float y = normal_raw[offset + 1];
    const float z = normal_raw[offset + 2];
    const float xz = __fadd_rn(
        __fmul_rn(x, x),
        __fmul_rn(z, z));
    const float squared_norm = __fadd_rn(
        xz,
        __fmul_rn(y, y));
    output[offset] = __fsqrt_rn(squared_norm);
}

__global__ static void precise_pixel_normal_divide_kernel(
    const float* normal_raw,
    float* in_out,
    index_t vector_count) {
    const index_t vector_index =
        static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vector_index >= vector_count) {
        return;
    }
    const index_t offset = vector_index * 3;
    const float denominator = __fadd_rn(
        in_out[offset],
        1.0e-8f);
    in_out[offset + 0] = precise_pixel_normal_divide_rn_f32(
        normal_raw[offset + 0],
        denominator);
    in_out[offset + 1] = precise_pixel_normal_divide_rn_f32(
        normal_raw[offset + 1],
        denominator);
    in_out[offset + 2] = precise_pixel_normal_divide_rn_f32(
        normal_raw[offset + 2],
        denominator);
}

#endif

void PrecisePixelNormalOp::jit_run() {
#ifdef JIT_cuda
    const index_t vector_count = output->num / 3;
    if (vector_count == 0) {
        return;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (vector_count + threads - 1) / threads);
    precise_pixel_normal_l2_kernel<<<blocks, threads>>>(
        normal_raw->ptr<float32>(),
        output->ptr<float32>(),
        vector_count);
    cudaError_t error = cudaGetLastError();
    CHECK(error == cudaSuccess)
        << "precise_pixel_normal L2 launch failed: "
        << cudaGetErrorString(error);
    precise_pixel_normal_divide_kernel<<<blocks, threads>>>(
        normal_raw->ptr<float32>(),
        output->ptr<float32>(),
        vector_count);
    error = cudaGetLastError();
    CHECK(error == cudaSuccess)
        << "precise_pixel_normal divide launch failed: "
        << cudaGetErrorString(error);
#else
    CHECK(false) << "PrecisePixelNormalOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
