#ifndef HYBRIDSPLAT_RASTERIZER_C_API_H_
#define HYBRIDSPLAT_RASTERIZER_C_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(HS_RASTERIZER_BUILDING_LIBRARY)
#define HS_API __declspec(dllexport)
#else
#define HS_API __declspec(dllimport)
#endif
#else
#define HS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define HS_NOEXCEPT noexcept
extern "C" {
#else
#define HS_NOEXCEPT
#endif

#define HS_RASTERIZER_ABI_VERSION 2u

typedef struct HsRasterizerContext* HsRasterizerHandle;

typedef enum HsStatus {
    HS_STATUS_OK = 0,
    HS_STATUS_INVALID_ARGUMENT = 1,
    HS_STATUS_ABI_MISMATCH = 2,
    HS_STATUS_RANGE_ERROR = 3,
    HS_STATUS_CUDA_ERROR = 4,
    HS_STATUS_OUT_OF_MEMORY = 5,
    HS_STATUS_BUSY = 6,
    HS_STATUS_INTERNAL_ERROR = 7
} HsStatus;

typedef struct HsRasterizerWorkspaceStatsV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device;
    uint32_t reserved;
    uint64_t geometry_bytes;
    uint64_t binning_bytes;
    uint64_t image_bytes;
    uint64_t current_bytes;
    uint64_t peak_bytes;
    uint64_t growth_count;
} HsRasterizerWorkspaceStatsV1;

typedef struct HsOptixWorkspaceStatsV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device;
    uint32_t scene_ready;
    uint64_t params_bytes;
    uint64_t gas_output_bytes;
    uint64_t gas_scratch_bytes;
    uint64_t dist3_bytes;
    uint64_t current_bytes;
    uint64_t peak_bytes;
    uint64_t growth_count;
    uint64_t scene_signature;
    uint64_t gas_build_count;
    uint64_t pipeline_create_count;
} HsOptixWorkspaceStatsV1;

typedef struct HsMathWorkspaceStatsV2 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device;
    uint32_t handles_ready;
    uint64_t workspace_bytes;
    uint64_t current_bytes;
    uint64_t peak_bytes;
    uint64_t growth_count;
    uint64_t handle_create_count;
    uint64_t camera_center_calls;
    uint64_t general_inverse_calls;
    uint64_t matmul_calls;
    uint64_t pixel_world_calls;
} HsMathWorkspaceStatsV2;

typedef struct HsFlashgsWorkspaceStatsV1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device;
    uint32_t initialized;
    uint64_t gaussian_capacity;
    uint64_t pair_capacity;
    uint64_t range_capacity;
    uint64_t output_pixel_capacity;
    uint64_t gaussian_bytes;
    uint64_t pair_bytes;
    uint64_t cub_bytes;
    uint64_t ranges_bytes;
    uint64_t output_bytes;
    uint64_t current_bytes;
    uint64_t peak_bytes;
    uint64_t growth_count;
    uint64_t overflow_retry_count;
    uint64_t forward_calls;
    uint64_t last_point_count;
    uint64_t last_num_rendered;
    uint64_t last_tile_count;
} HsFlashgsWorkspaceStatsV1;

HS_API uint32_t hs_rasterizer_abi_version(void) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_build_info(
    char* output_text,
    uint64_t output_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_create(
    int32_t device,
    HsRasterizerHandle* handle,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_synchronize(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_workspace_stats(
    HsRasterizerHandle handle,
    HsRasterizerWorkspaceStatsV1* stats,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_release_workspace(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_math_workspace_stats_v2(
    HsRasterizerHandle handle,
    HsMathWorkspaceStatsV2* stats,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_destroy(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/* Deterministic FlashGS 16x16 inference forward. Gaussian and output
 * buffers are contiguous device FP32 buffers. camera_position3,
 * camera_rotation9, and background3 point to host FP32 arrays and are read
 * before this call returns. Exactly one of shs48 and colors3 is required,
 * selected by use_sh. Outputs use channel-first layouts:
 * color[3,H,W], other[2,H,W], depth[1,H,W], alpha[1,H,W],
 * normal[3,H,W], median_depth[1,H,W], distortion[1,H,W], rendered[1]. */
HS_API int32_t hs_flashgs_forward_f32_v1(
    HsRasterizerHandle handle,
    uint64_t point_count,
    int32_t width,
    int32_t height,
    const float* positions3,
    const float* shs48,
    const float* colors3,
    int32_t use_sh,
    const float* opacities1,
    const float* covariance6,
    const float* scales2,
    float scale_modifier,
    const float* rotations4,
    const float* specular1,
    const float* roughness1,
    const float* camera_position3,
    const float* camera_rotation9,
    float focal_x,
    float focal_y,
    float z_far,
    float z_near,
    const float* background3,
    uint64_t initial_pair_capacity,
    float* output_color3,
    float* output_other2,
    float* output_depth1,
    float* output_alpha1,
    float* output_normal3,
    float* output_median_depth1,
    float* output_distortion1,
    int32_t* output_num_rendered,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_flashgs_workspace_stats_v1(
    HsRasterizerHandle handle,
    HsFlashgsWorkspaceStatsV1* stats,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/* Exact host-side activation used by the frozen deterministic FlashGS NPZ
 * loader. All buffers are contiguous host FP32 arrays. The implementation
 * calls the same sigmoid_f32, std::exp, glm::normalize, and computeCov3D
 * functions as HybridSplat-FlashGS-deterministic/csrc/pybind.cpp. */
HS_API int32_t hs_flashgs_prepare_npz_host_f32_v1(
    uint64_t point_count,
    const float* opacity_raw1,
    const float* scaling_raw2,
    const float* rotation_raw4,
    const float* specular_raw1,
    const float* roughness_raw1,
    float* opacity1,
    float* scaling2,
    float* rotation4,
    float* covariance6,
    float* specular1,
    float* roughness1,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/* Diagnostic-only synchronized copy of the most recent deterministic
 * FlashGS preprocess, sorted pairs, and tile ranges. All destination buffers
 * are host memory. Capacities must cover the counts returned by
 * hs_flashgs_workspace_stats_v1. This function is excluded from production
 * rendering and exists solely for staged numerical validation. */
HS_API int32_t hs_flashgs_debug_copy_v1(
    HsRasterizerHandle handle,
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
    int32_t* tile_ranges2,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * Inference-only five-channel 2D Gaussian rasterizer. All tensor buffers
 * are contiguous device buffers. Matrix storage and quaternion order match
 * the original HybridSplat extension. Outputs are color[5,H,W], aux[7,H,W],
 * weight[P,1], radii[P], and rendered[1]. The context owns only temporary
 * geometry, binning, sorting, and image workspaces.
 */
HS_API int32_t hs_rasterizer_forward_f32_v1(
    HsRasterizerHandle handle,
    uint64_t point_count,
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
    int32_t prefiltered,
    float* output_color5,
    float* output_aux7,
    float* output_weight,
    int32_t* output_radii,
    int32_t* output_rendered,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/* Inference-only OptiX scene build from [V,3] FP32 vertices and
 * [F,3] int32 triangles. Repeating the same signature with rebuild=0 is a
 * no-op. output_status is a one-element int32 device buffer. */
HS_API int32_t hs_optix_build_scene_v1(
    HsRasterizerHandle handle,
    const float* vertices3,
    uint64_t vertex_count,
    const int32_t* triangles3,
    uint64_t triangle_count,
    uint64_t scene_signature,
    int32_t rebuild,
    int32_t* output_status,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/* Inference-only reflection tracing. Outputs match the Python inference
 * ABI: rgb[P,3], dpt[P,1], acc[P,1], norm[P,3], dist[P,1], aux[P,2],
 * mid[P,32], and wet[P]. */
HS_API int32_t hs_optix_trace_f32_v1(
    HsRasterizerHandle handle,
    uint64_t ray_count,
    const float* ray_origins3,
    const float* ray_directions3,
    const float* background3,
    const float* means3d,
    const float* shs_m3,
    uint32_t coefficient_count,
    uint32_t degree,
    const float* others2,
    const float* opacities,
    const float* scales2,
    float scale_modifier,
    const float* rotations4,
    const float* viewmatrix4x4,
    const float* projmatrix4x4,
    const float* camera_position3,
    int32_t start_from_first,
    int32_t max_trace_depth,
    float specular_threshold,
    int32_t use_others,
    float* output_rgb3,
    float* output_depth1,
    float* output_acc1,
    float* output_normal3,
    float* output_dist1,
    float* output_aux2,
    float* output_mid32,
    float* output_wet1,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_optix_workspace_stats_v1(
    HsRasterizerHandle handle,
    HsOptixWorkspaceStatsV1* stats,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_optix_release_scene_v1(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * 精确 FP32 exp。输入和输出均为当前 device 上的连续 FP32 缓冲区。
 * 运算在 CUDA stream 0 上执行；count 为零时允许两个缓冲区均为空。
 */
HS_API int32_t hs_rasterizer_precise_exp_f32_v1(
    const float* input,
    float* output,
    uint64_t count,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_precise_sigmoid_f32_v1(
    const float* input,
    float* output,
    uint64_t count,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_precise_normalize4_f32_v1(
    const float* input4,
    float* output4,
    uint64_t count,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_precise_fuse_maps_f32_v1(
    const float* base_rgb,
    const float* reflection_rgb,
    const float* specular,
    float* diffuse,
    float* final_rgb,
    float* stored_reflection,
    uint64_t pixel_count,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * 精确 SH-to-RGB。sh 布局为 [count,3,coefficient_count]，
 * directions 为 [count,3]，output 为 [count,3]。结果包含 +0.5 与 clamp_min(0)。
 */
HS_API int32_t hs_rasterizer_precise_sh_to_rgb_f32_v1(
    const float* sh,
    const float* directions,
    uint64_t count,
    uint32_t coefficient_count,
    uint32_t degree,
    float* output,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * 精确视线方向。points 为 [count,3]，center 为 [3]，
 * output 为 normalize(points-center)，全部使用逐步 FP32 舍入。
 */
HS_API int32_t hs_rasterizer_precise_direction_f32_v1(
    const float* points,
    const float* center,
    uint64_t count,
    float* output,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * Exact GaussianCamera center. rotation3x3 and translation3x1 are
 * contiguous row-major FP32 buffers; output3 is -rotation^T*translation.
 * The implementation reproduces PyTorch 2.6 SGEMM(N,T,1,3,3) on stream 0.
 * Handles and scratch storage belong to the persistent session context.
 */
HS_API int32_t hs_rasterizer_precise_camera_center_f32_v2(
    HsRasterizerHandle handle,
    const float* rotation3x3,
    const float* translation3x1,
    float* output3,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * PyTorch 2.6 CUDA general inverse for one contiguous row-major FP32
 * matrix. dimension must be 3 or 4. The implementation reproduces
 * linalg_inv_ex: identity RHS, LU of the row-major buffer interpreted as
 * column-major A^T, and cublasSgetrsBatched(CUBLAS_OP_T). output is returned
 * as a contiguous row-major matrix. Handles and scratch storage belong to
 * the persistent session context.
 */
HS_API int32_t hs_rasterizer_precise_general_inverse_f32_v2(
    HsRasterizerHandle handle,
    const float* input,
    uint32_t dimension,
    float* output,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * PyTorch 2.6 TF32 left[N,3] @ right[3,3]^T. left, right, and output are
 * contiguous row-major buffers. The reference renderer uses allow_tf32=true and
 * float32_matmul_precision=high, so this calls cublasSgemm with
 * CUBLAS_TF32_TENSOR_OP_MATH. The implementation uses CUDA stream 0 and
 * accepts count == 0.
 */
HS_API int32_t hs_rasterizer_precise_matmul_nt_n3k3_f32_v2(
    HsRasterizerHandle handle,
    const float* left_n3,
    const float* right3x3,
    uint64_t count,
    float* output_n3,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

HS_API int32_t hs_rasterizer_precise_pixel_world_f32_v2(
    HsRasterizerHandle handle,
    const float* rotation3x3,
    const float* vectors_n3,
    uint64_t count,
    float* output_n3,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * Exact rotation matrix for OptiX disk geometry. rotation_wxyz is the
 * raw checkpoint [count,4]; rotation_matrix9 is row-major [count,3,3].
 * The implementation reproduces the first profiling invocation of
 * HardwareRendering.get_disks: F.normalize, build_rotation's second
 * normalization, and unfused FP32 quaternion arithmetic.
 */
HS_API int32_t hs_rasterizer_precise_disk_rotation_f32_v1(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * Reflection prefix. rotation_wxyz is the raw checkpoint [count,4];
 * rotation_matrix9 is the twice-normalized row-major [count,3,3] matrix.
 */
HS_API int32_t hs_rasterizer_precise_reflection_prefix_f32_v1(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

/*
 * Reflection suffix after applying the diagonal scale to the
 * TensorExpr-aligned rotation matrix. xyz and normal_raw3 are [count,3],
 * camera_center is [3]. output36 starts with
 * [view3, normal3, reflected3, ray_d3, ray_o3, positive1]; remaining fields
 * preserve deterministic diagnostics.
 */
HS_API int32_t hs_rasterizer_precise_reflection_suffix_f32_v1(
    const float* xyz,
    const float* camera_center,
    const float* normal_raw3,
    uint64_t count,
    float* output36,
    char* error_text,
    uint64_t error_capacity) HS_NOEXCEPT;

#ifdef __cplusplus
}
#endif

#endif
