#pragma once

#include "context_internal.h"

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_full_rasterizer_forward_f32(
    HsRasterizerContext* context,
    int32_t point_count,
    const float* background5,
    int32_t width,
    int32_t height,
    const float* means3d,
    const float* colors5,
    const float* opacities,
    const float* scales2,
    float scale_modifier,
    const float* rotations4,
    const float* viewmatrix4x4,
    const float* projmatrix4x4,
    const float* camera_position3,
    float tan_fovx,
    float tan_fovy,
    bool prefiltered,
    float* output_color5,
    float* output_aux7,
    float* output_weight,
    int32_t* output_radii,
    int32_t* output_rendered) noexcept;
