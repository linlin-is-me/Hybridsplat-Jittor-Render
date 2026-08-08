#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_column_major_to_row_major_f32(
    const float* input,
    uint32_t dimension,
    float* output) noexcept;
