#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_precise_sigmoid_f32(
    const float* input,
    float* output,
    uint64_t count) noexcept;

cudaError_t hs_launch_precise_normalize4_f32(
    const float* input4,
    float* output4,
    uint64_t count) noexcept;

cudaError_t hs_launch_precise_fuse_maps_f32(
    const float* base_rgb,
    const float* reflection_rgb,
    const float* specular,
    float* diffuse,
    float* final_rgb,
    float* stored_reflection,
    uint64_t pixel_count) noexcept;
