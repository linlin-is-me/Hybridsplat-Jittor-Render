#include "precise_depth_divide_op.h"

namespace jittor {

#ifndef JIT

PreciseDepthDivideOp::PreciseDepthDivideOp(Var* numerator, Var* denominator) : numerator(numerator), denominator(denominator) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(numerator->dtype() == ns_float32)
        << "precise_depth_divide numerator must be FP32";
    CHECK(denominator->dtype() == ns_float32)
        << "precise_depth_divide denominator must be FP32";
    CHECK(numerator->shape == denominator->shape)
        << "precise_depth_divide inputs must have identical shapes";
    output = create_output(numerator->shape, ns_float32);
}

void PreciseDepthDivideOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "VERSION", JK::hex1(2));
}

#else

#ifdef JIT_cuda

__device__ __forceinline__ float precise_divide_rn_f32(
    float numerator,
    float denominator) {
    float result;
    asm volatile(
        "div.rn.f32 %0, %1, %2;"
        : "=f"(result)
        : "f"(numerator), "f"(denominator));
    return result;
}

__global__ static void precise_depth_divide_kernel(
    const float* __restrict__ numerator,
    const float* __restrict__ denominator,
    float* __restrict__ output,
    index_t count) {
    const index_t index =
        static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    float value = precise_divide_rn_f32(
        numerator[index],
        denominator[index]);
    const unsigned int bits = __float_as_uint(value);
    const unsigned int exponent = bits & 0x7f800000u;
    if (exponent == 0x7f800000u) {
        const unsigned int fraction = bits & 0x007fffffu;
        const unsigned int sign = bits & 0x80000000u;
        if (fraction != 0u || sign == 0u) {
            value = 0.0f;
        } else {
            value = __uint_as_float(0xff7fffffu);
        }
    }
    output[index] = value;
}

#endif

void PreciseDepthDivideOp::jit_run() {
#ifdef JIT_cuda
    const index_t count = output->num;
    if (count == 0) {
        return;
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>(
        (count + threads - 1) / threads);
    precise_depth_divide_kernel<<<blocks, threads>>>(
        numerator->ptr<float32>(),
        denominator->ptr<float32>(),
        output->ptr<float32>(),
        count);
    const cudaError_t error = cudaGetLastError();
    CHECK(error == cudaSuccess)
        << "precise_depth_divide kernel launch failed: "
        << cudaGetErrorString(error);
#else
    CHECK(false) << "PreciseDepthDivideOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
