#include "precise_exp.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

__global__ void precise_exp_f32_kernel(
    const float* input,
    float* output,
    uint64_t count) {
    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = expf(input[index]);
    }
}

}  // namespace

cudaError_t hs_launch_precise_exp_f32(
    const float* input,
    float* output,
    uint64_t count) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    const uint64_t block_count =
        (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (block_count > static_cast<uint64_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    precise_exp_f32_kernel<<<
        static_cast<uint32_t>(block_count),
        kThreadsPerBlock,
        0,
        0>>>(input, output, count);
    return cudaGetLastError();
}
