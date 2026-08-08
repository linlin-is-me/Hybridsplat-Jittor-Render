#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

struct HsRasterizerContext;

cudaError_t hs_launch_flashgs_forward_f32(
    HsRasterizerContext* context,
    int32_t point_count,
    int32_t width,
    int32_t height,
    const float* positions3,
    const float* shs48,
    const float* colors3,
    bool use_sh,
    const float* opacities1,
    const float* covariance6,
    const float* scales2,
    float scale_modifier,
    const float* rotations4,
    const float* specular1,
    const float* roughness1,
    const float camera_position3[3],
    const float camera_rotation9[9],
    float focal_x,
    float focal_y,
    float z_far,
    float z_near,
    const float background3[3],
    uint64_t initial_pair_capacity,
    float* output_color3,
    float* output_other2,
    float* output_depth1,
    float* output_alpha1,
    float* output_normal3,
    float* output_median_depth1,
    float* output_distortion1,
    int32_t* output_num_rendered) noexcept(false);

cudaError_t hs_flashgs_release_context(HsRasterizerContext* context) noexcept;

cudaError_t hs_flashgs_copy_debug_to_host(
    HsRasterizerContext* context,
    uint64_t point_capacity,
    uint64_t pair_capacity,
    uint64_t tile_capacity,
    float* points_xy2,
    float* rgb3,
    float* other2,
    float* conic_opacity4,
    float* normal_opacity4,
    float* tu3,
    float* tv3,
    float* tw3,
    uint64_t* sorted_keys,
    uint32_t* sorted_values,
    int32_t* tile_ranges2) noexcept;
