#include "precise_matrix.h"

#include <cuda_runtime.h>

namespace {

__global__ void column_major_to_row_major_f32_kernel(
    const float* input,
    uint32_t dimension,
    float* output) {
    const uint32_t index = threadIdx.x;
    const uint32_t count = dimension * dimension;
    if (index >= count) {
        return;
    }
    const uint32_t row = index / dimension;
    const uint32_t column = index - row * dimension;
    output[index] = input[row + column * dimension];
}

}  // namespace

cudaError_t hs_launch_column_major_to_row_major_f32(
    const float* input,
    uint32_t dimension,
    float* output) noexcept {
    column_major_to_row_major_f32_kernel<<<1, 16, 0, nullptr>>>(
        input,
        dimension,
        output);
    return cudaGetLastError();
}
