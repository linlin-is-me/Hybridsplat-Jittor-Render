#include "precise_camera_center.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void negative_rotation3x3_kernel(
    const float* rotation3x3,
    float* negative_rotation3x3) {
    const uint32_t index = threadIdx.x;
    if (index < 9) {
        // PyTorch's unary negation preserves the [1,3] strides of R.mT.
        // Writing -R in physical row-major order reproduces that buffer.
        negative_rotation3x3[index] = -rotation3x3[index];
    }
}

}  // namespace

cudaError_t hs_launch_negative_rotation3x3_f32(
    const float* rotation3x3,
    float* negative_rotation3x3) {
    negative_rotation3x3_kernel<<<1, 32, 0, 0>>>(
        rotation3x3,
        negative_rotation3x3);
    return cudaGetLastError();
}
