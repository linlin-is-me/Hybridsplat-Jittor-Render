#include "precise_activation.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreads = 256;

__global__ void precise_sigmoid_kernel(const float* input, float* output, uint64_t count)
{
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float denominator = __fadd_rn(1.0f, expf(__fsub_rn(0.0f, input[index])));
    output[index] = __fdiv_rn(1.0f, denominator);
}

__global__ void precise_normalize4_kernel(const float* input, float* output, uint64_t count)
{
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const uint64_t offset = index * 4;
    const float s0 = __fmul_rn(input[offset], input[offset]);
    const float s1 = __fmul_rn(input[offset + 1], input[offset + 1]);
    const float s2 = __fmul_rn(input[offset + 2], input[offset + 2]);
    const float s3 = __fmul_rn(input[offset + 3], input[offset + 3]);
    const float norm = __fsqrt_rn(__fadd_rn(__fadd_rn(s0, s1), __fadd_rn(s2, s3)));
    const float denominator = norm < 1.0e-12f ? 1.0e-12f : norm;
    output[offset] = __fdiv_rn(input[offset], denominator);
    output[offset + 1] = __fdiv_rn(input[offset + 1], denominator);
    output[offset + 2] = __fdiv_rn(input[offset + 2], denominator);
    output[offset + 3] = __fdiv_rn(input[offset + 3], denominator);
}

__global__ void precise_fuse_maps_kernel(
    const float* base_rgb,
    const float* reflection_rgb,
    const float* specular,
    float* diffuse,
    float* final_rgb,
    float* stored_reflection,
    uint64_t value_count)
{
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= value_count) return;
    const float weight = specular[index / 3];
    const float diffuse_value = __fmul_rn(base_rgb[index], __fsub_rn(1.0f, weight));
    const float reflected_value = __fmul_rn(
        __fmul_rn(reflection_rgb[index], weight),
        2.0f);
    diffuse[index] = diffuse_value;
    stored_reflection[index] = reflected_value;
    final_rgb[index] = __fadd_rn(diffuse_value, __fmul_rn(0.5f, reflected_value));
}

}  // namespace

cudaError_t hs_launch_precise_sigmoid_f32(
    const float* input,
    float* output,
    uint64_t count) noexcept
{
    if (count == 0) return cudaSuccess;
    const uint64_t blocks = (count + kThreads - 1) / kThreads;
    if (blocks > UINT32_MAX) return cudaErrorInvalidValue;
    precise_sigmoid_kernel<<<static_cast<uint32_t>(blocks), kThreads, 0, 0>>>(input, output, count);
    return cudaGetLastError();
}

cudaError_t hs_launch_precise_normalize4_f32(
    const float* input4,
    float* output4,
    uint64_t count) noexcept
{
    if (count == 0) return cudaSuccess;
    const uint64_t blocks = (count + kThreads - 1) / kThreads;
    if (blocks > UINT32_MAX) return cudaErrorInvalidValue;
    precise_normalize4_kernel<<<static_cast<uint32_t>(blocks), kThreads, 0, 0>>>(input4, output4, count);
    return cudaGetLastError();
}

cudaError_t hs_launch_precise_fuse_maps_f32(
    const float* base_rgb,
    const float* reflection_rgb,
    const float* specular,
    float* diffuse,
    float* final_rgb,
    float* stored_reflection,
    uint64_t pixel_count) noexcept
{
    if (pixel_count == 0) return cudaSuccess;
    if (pixel_count > UINT64_MAX / 3) return cudaErrorInvalidValue;
    const uint64_t value_count = pixel_count * 3;
    const uint64_t blocks = (value_count + kThreads - 1) / kThreads;
    if (blocks > UINT32_MAX) return cudaErrorInvalidValue;
    precise_fuse_maps_kernel<<<static_cast<uint32_t>(blocks), kThreads, 0, 0>>>(
        base_rgb,
        reflection_rgb,
        specular,
        diffuse,
        final_rgb,
        stored_reflection,
        value_count);
    return cudaGetLastError();
}
