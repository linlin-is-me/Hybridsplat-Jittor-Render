#include "precise_direction.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

__global__ void precise_direction_f32_kernel(
    const float* points,
    const float* center,
    uint64_t count,
    float* output) {
    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const float x = points[index * 3] - center[0];
    const float y = points[index * 3 + 1] - center[1];
    const float z = points[index * 3 + 2] - center[2];
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    // PyTorch Reduce.cuh maps a three-value inner reduction to block.x=2:
    // lane 0 accumulates x and z, then combines lane 1's y.
    float squared_norm = xx + zz;
    squared_norm = squared_norm + yy;
    const float norm = sqrtf(squared_norm);
    output[index * 3] = x / norm;
    output[index * 3 + 1] = y / norm;
    output[index * 3 + 2] = z / norm;
}

}  // namespace

cudaError_t hs_launch_precise_direction_f32(
    const float* points,
    const float* center,
    uint64_t count,
    float* output) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    const uint64_t block_count =
        (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (block_count > static_cast<uint64_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    precise_direction_f32_kernel<<<
        static_cast<uint32_t>(block_count),
        kThreadsPerBlock,
        0,
        0>>>(points, center, count, output);
    return cudaGetLastError();
}
