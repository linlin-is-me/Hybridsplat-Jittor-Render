#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <cstdint>
#include <mutex>

struct HsRasterizerContext {
    explicit HsRasterizerContext(int32_t device_id) : device(device_id) {}

    int32_t device;
    std::mutex mutex;
    void* geometry = nullptr;
    void* binning = nullptr;
    void* image = nullptr;
    uint64_t geometry_bytes = 0;
    uint64_t binning_bytes = 0;
    uint64_t image_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t growth_count = 0;
    cublasHandle_t math_tf32_blas = nullptr;
    cublasHandle_t math_default_blas = nullptr;
    cusolverDnHandle_t math_solver = nullptr;
    void* math_workspace = nullptr;
    uint64_t math_workspace_bytes = 0;
    uint64_t math_peak_bytes = 0;
    uint64_t math_growth_count = 0;
    int32_t math_inverse_workspace_count = 0;
    int32_t math_solve_info[16] = {};
    uint64_t math_handle_create_count = 0;
    uint64_t math_camera_center_calls = 0;
    uint64_t math_general_inverse_calls = 0;
    uint64_t math_matmul_calls = 0;
    uint64_t math_pixel_world_calls = 0;
    void* optix_state = nullptr;
    void* optix_params = nullptr;
    void* optix_gas_scratch = nullptr;
    void* optix_dist3 = nullptr;
    uint64_t optix_params_bytes = 0;
    uint64_t optix_gas_scratch_bytes = 0;
    uint64_t optix_dist3_bytes = 0;
    uint64_t optix_peak_bytes = 0;
    uint64_t optix_growth_count = 0;
    uint64_t optix_scene_signature = 0;
    uint64_t optix_gas_build_count = 0;
    uint64_t optix_pipeline_create_count = 0;
    void* flashgs_gaussian_workspace = nullptr;
    void* flashgs_pair_workspace = nullptr;
    void* flashgs_cub_workspace = nullptr;
    void* flashgs_ranges_workspace = nullptr;
    void* flashgs_output_workspace = nullptr;
    int32_t* flashgs_counter_device = nullptr;
    int32_t* flashgs_counter_host = nullptr;
    uint64_t flashgs_gaussian_bytes = 0;
    uint64_t flashgs_pair_bytes = 0;
    uint64_t flashgs_cub_bytes = 0;
    uint64_t flashgs_ranges_bytes = 0;
    uint64_t flashgs_output_bytes = 0;
    uint64_t flashgs_gaussian_capacity = 0;
    uint64_t flashgs_pair_capacity = 0;
    uint64_t flashgs_range_capacity = 0;
    uint64_t flashgs_output_pixel_capacity = 0;
    uint64_t flashgs_peak_bytes = 0;
    uint64_t flashgs_growth_count = 0;
    uint64_t flashgs_overflow_retry_count = 0;
    uint64_t flashgs_forward_calls = 0;
    uint64_t flashgs_last_point_count = 0;
    uint64_t flashgs_last_num_rendered = 0;
    uint64_t flashgs_last_tile_count = 0;
};

cudaError_t hs_context_reserve_geometry(
    HsRasterizerContext* context,
    uint64_t required,
    char** pointer) noexcept;

cudaError_t hs_context_reserve_binning(
    HsRasterizerContext* context,
    uint64_t required,
    char** pointer) noexcept;

cudaError_t hs_context_reserve_image(
    HsRasterizerContext* context,
    uint64_t required,
    char** pointer) noexcept;

cudaError_t hs_context_release_buffers(HsRasterizerContext* context) noexcept;

cudaError_t hs_context_initialize_math(HsRasterizerContext* context) noexcept;
cudaError_t hs_context_release_math(HsRasterizerContext* context) noexcept;
