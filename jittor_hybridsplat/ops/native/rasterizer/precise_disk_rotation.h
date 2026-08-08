#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_precise_disk_rotation_f32(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9) noexcept;
