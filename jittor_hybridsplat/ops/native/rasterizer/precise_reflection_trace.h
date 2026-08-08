#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_precise_reflection_prefix_f32(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9) noexcept;

cudaError_t hs_launch_precise_reflection_suffix_f32(
    const float* xyz,
    const float* camera_center,
    const float* normal_raw3,
    uint64_t count,
    float* output36) noexcept;
