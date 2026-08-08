#include "hybridsplat_rasterizer_c_api.h"

#include "context_internal.h"
#include "flashgs_backend.h"
#include "full_rasterizer.h"
#include "optix_backend.h"
#include "precise_camera_center.h"
#include "precise_activation.h"
#include "precise_direction.h"
#include "precise_disk_rotation.h"
#include "precise_exp.h"
#include "precise_matrix.h"
#include "precise_reflection_trace.h"
#include "precise_sh.h"

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>

#ifndef HS_RASTERIZER_BUILD_ID
#error "HS_RASTERIZER_BUILD_ID must be defined by the native build"
#endif

namespace {

constexpr uint64_t kWorkspaceAlignment = 2ull * 1024ull * 1024ull;
constexpr uint64_t kMathWorkspaceBytes = kWorkspaceAlignment;
constexpr uint64_t kMathCameraOffset = 0;
constexpr uint64_t kMathIdentity3Offset = 256;
constexpr uint64_t kMathIdentity4Offset = 384;
constexpr uint64_t kMathFactorOffset = 512;
constexpr uint64_t kMathRhsOffset = 768;
constexpr uint64_t kMathPivotsOffset = 1024;
constexpr uint64_t kMathInfoOffset = 1280;
constexpr uint64_t kMathFactorArrayOffset = 1536;
constexpr uint64_t kMathRhsArrayOffset = 1792;
constexpr uint64_t kMathSolverWorkspaceOffset = 2048;

struct MathWorkspacePointers {
    float* negative_rotation3x3;
    float* identity3x3;
    float* identity4x4;
    float* factor4x4;
    float* rhs4x4;
    int* pivots4;
    int* device_info;
    float** factor_array;
    float** rhs_array;
    float* solver_workspace;
};

MathWorkspacePointers math_pointers(HsRasterizerContext* context) noexcept {
    auto* base = static_cast<unsigned char*>(context->math_workspace);
    return {
        reinterpret_cast<float*>(base + kMathCameraOffset),
        reinterpret_cast<float*>(base + kMathIdentity3Offset),
        reinterpret_cast<float*>(base + kMathIdentity4Offset),
        reinterpret_cast<float*>(base + kMathFactorOffset),
        reinterpret_cast<float*>(base + kMathRhsOffset),
        reinterpret_cast<int*>(base + kMathPivotsOffset),
        reinterpret_cast<int*>(base + kMathInfoOffset),
        reinterpret_cast<float**>(base + kMathFactorArrayOffset),
        reinterpret_cast<float**>(base + kMathRhsArrayOffset),
        reinterpret_cast<float*>(base + kMathSolverWorkspaceOffset),
    };
}

uint64_t align_workspace(uint64_t value) noexcept {
    if (value == 0) {
        return kWorkspaceAlignment;
    }
    return (value + kWorkspaceAlignment - 1) & ~(kWorkspaceAlignment - 1);
}

cudaError_t reserve_buffer(
    void** buffer,
    uint64_t* capacity,
    uint64_t required,
    HsRasterizerContext* context,
    char** pointer) noexcept {
    if (buffer == nullptr || capacity == nullptr || context == nullptr ||
        pointer == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (*capacity >= required && *buffer != nullptr) {
        *pointer = static_cast<char*>(*buffer);
        return cudaSuccess;
    }
    const uint64_t grown =
        *capacity == 0 ? required : std::max(required, *capacity + *capacity / 4);
    const uint64_t target = align_workspace(grown);
    void* replacement = nullptr;
    cudaError_t error = cudaMalloc(&replacement, target);
    if (error != cudaSuccess) {
        return error;
    }
    if (*buffer != nullptr) {
        error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            cudaFree(replacement);
            return error;
        }
        error = cudaFree(*buffer);
        if (error != cudaSuccess) {
            cudaFree(replacement);
            return error;
        }
    }
    *buffer = replacement;
    *capacity = target;
    ++context->growth_count;
    context->peak_bytes = std::max(
        context->peak_bytes,
        context->geometry_bytes + context->binning_bytes + context->image_bytes);
    *pointer = static_cast<char*>(replacement);
    return cudaSuccess;
}

void write_text(char* destination, uint64_t capacity, const char* text) noexcept {
    if (destination == nullptr || capacity == 0) {
        return;
    }
    const char* source = text == nullptr ? "" : text;
    const size_t source_size = std::strlen(source);
    const size_t copy_size =
        std::min(source_size, static_cast<size_t>(capacity - 1));
    if (copy_size != 0) {
        std::memcpy(destination, source, copy_size);
    }
    destination[copy_size] = '\0';
}

void clear_error(char* error_text, uint64_t error_capacity) noexcept {
    write_text(error_text, error_capacity, "");
}

int32_t fail(
    HsStatus status,
    const std::string& message,
    char* error_text,
    uint64_t error_capacity) noexcept {
    write_text(error_text, error_capacity, message.c_str());
    return static_cast<int32_t>(status);
}

int32_t cuda_failure(
    cudaError_t error,
    const char* operation,
    char* error_text,
    uint64_t error_capacity) noexcept {
    const HsStatus status =
        error == cudaErrorMemoryAllocation ? HS_STATUS_OUT_OF_MEMORY
                                           : HS_STATUS_CUDA_ERROR;
    std::string message(operation);
    message += ": ";
    message += cudaGetErrorString(error);
    return fail(status, message, error_text, error_capacity);
}

int32_t cublas_failure(
    cublasStatus_t status,
    const char* operation,
    char* error_text,
    uint64_t error_capacity) noexcept {
    std::string message(operation);
    message += ": ";
    message += cublasGetStatusString(status);
    return fail(
        HS_STATUS_CUDA_ERROR,
        message,
        error_text,
        error_capacity);
}

int32_t cusolver_failure(
    cusolverStatus_t status,
    const char* operation,
    char* error_text,
    uint64_t error_capacity) noexcept {
    char message[256] = {};
    std::snprintf(
        message,
        sizeof(message),
        "%s: cuSOLVER status %d",
        operation,
        static_cast<int>(status));
    return fail(
        status == CUSOLVER_STATUS_ALLOC_FAILED
            ? HS_STATUS_OUT_OF_MEMORY
            : HS_STATUS_CUDA_ERROR,
        message,
        error_text,
        error_capacity);
}

int32_t validate_error_buffer(
    char* error_text,
    uint64_t error_capacity) noexcept {
    if (error_capacity != 0 && error_text == nullptr) {
        return static_cast<int32_t>(HS_STATUS_INVALID_ARGUMENT);
    }
    return static_cast<int32_t>(HS_STATUS_OK);
}

int32_t select_device(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) noexcept {
    if (handle == nullptr) {
        return fail(
            HS_STATUS_INVALID_ARGUMENT,
            "handle is null",
            error_text,
            error_capacity);
    }
    const cudaError_t error = cudaSetDevice(handle->device);
    if (error != cudaSuccess) {
        return cuda_failure(
            error, "cudaSetDevice", error_text, error_capacity);
    }
    return static_cast<int32_t>(HS_STATUS_OK);
}

int32_t initialize_math_context(
    HsRasterizerContext* context,
    char* error_text,
    uint64_t error_capacity) noexcept {
    cublasStatus_t blas_status = cublasCreate(&context->math_tf32_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_failure(
            blas_status, "math TF32 cublasCreate", error_text, error_capacity);
    }
    ++context->math_handle_create_count;
    blas_status = cublasSetStream(context->math_tf32_blas, nullptr);
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSetPointerMode(
            context->math_tf32_blas, CUBLAS_POINTER_MODE_HOST);
    }
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSetMathMode(
            context->math_tf32_blas, CUBLAS_TF32_TENSOR_OP_MATH);
    }
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_failure(
            blas_status, "math TF32 handle setup", error_text, error_capacity);
    }

    blas_status = cublasCreate(&context->math_default_blas);
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_failure(
            blas_status, "math FP32 cublasCreate", error_text, error_capacity);
    }
    ++context->math_handle_create_count;
    blas_status = cublasSetStream(context->math_default_blas, nullptr);
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSetPointerMode(
            context->math_default_blas, CUBLAS_POINTER_MODE_HOST);
    }
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSetMathMode(
            context->math_default_blas, CUBLAS_DEFAULT_MATH);
    }
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return cublas_failure(
            blas_status, "math FP32 handle setup", error_text, error_capacity);
    }

    cusolverStatus_t solver_status = cusolverDnCreate(&context->math_solver);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        return cusolver_failure(
            solver_status, "math cusolverDnCreate", error_text, error_capacity);
    }
    ++context->math_handle_create_count;
    solver_status = cusolverDnSetStream(context->math_solver, nullptr);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        return cusolver_failure(
            solver_status, "math cusolverDnSetStream", error_text, error_capacity);
    }

    cudaError_t cuda_status = cudaMalloc(
        &context->math_workspace, kMathWorkspaceBytes);
    if (cuda_status != cudaSuccess) {
        return cuda_failure(
            cuda_status, "math workspace cudaMalloc", error_text, error_capacity);
    }
    context->math_workspace_bytes = kMathWorkspaceBytes;
    context->math_peak_bytes = kMathWorkspaceBytes;
    ++context->math_growth_count;
    const MathWorkspacePointers pointers = math_pointers(context);
    float identity3[9] = {};
    for (int index = 0; index < 3; ++index) {
        identity3[index + index * 3] = 1.0f;
    }
    float identity4[16] = {};
    for (int index = 0; index < 4; ++index) {
        identity4[index + index * 4] = 1.0f;
    }
    cuda_status = cudaMemcpy(
        pointers.identity3x3,
        identity3,
        sizeof(identity3),
        cudaMemcpyHostToDevice);
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpy(
            pointers.identity4x4,
            identity4,
            sizeof(identity4),
            cudaMemcpyHostToDevice);
    }
    if (cuda_status == cudaSuccess) {
        float* factor = pointers.factor4x4;
        float* rhs = pointers.rhs4x4;
        cuda_status = cudaMemcpy(
            pointers.factor_array,
            &factor,
            sizeof(factor),
            cudaMemcpyHostToDevice);
        if (cuda_status == cudaSuccess) {
            cuda_status = cudaMemcpy(
                pointers.rhs_array,
                &rhs,
                sizeof(rhs),
                cudaMemcpyHostToDevice);
        }
    }
    if (cuda_status != cudaSuccess) {
        return cuda_failure(
            cuda_status, "math workspace initialization", error_text, error_capacity);
    }
    solver_status = cusolverDnSgetrf_bufferSize(
        context->math_solver,
        4,
        4,
        pointers.factor4x4,
        4,
        &context->math_inverse_workspace_count);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
        return cusolver_failure(
            solver_status,
            "math inverse workspace query",
            error_text,
            error_capacity);
    }
    const uint64_t solver_bytes =
        static_cast<uint64_t>(context->math_inverse_workspace_count) * sizeof(float);
    if (kMathSolverWorkspaceOffset + solver_bytes > kMathWorkspaceBytes) {
        return fail(
            HS_STATUS_RANGE_ERROR,
            "math inverse workspace exceeds persistent allocation",
            error_text,
            error_capacity);
    }
    return static_cast<int32_t>(HS_STATUS_OK);
}

template <typename Function>
int32_t guard(
    char* error_text,
    uint64_t error_capacity,
    Function&& function) noexcept {
    const int32_t buffer_status =
        validate_error_buffer(error_text, error_capacity);
    if (buffer_status != HS_STATUS_OK) {
        return buffer_status;
    }
    try {
        return function();
    } catch (const std::bad_alloc& exception) {
        return fail(
            HS_STATUS_OUT_OF_MEMORY,
            exception.what(),
            error_text,
            error_capacity);
    } catch (const std::exception& exception) {
        return fail(
            HS_STATUS_INTERNAL_ERROR,
            exception.what(),
            error_text,
            error_capacity);
    } catch (...) {
        return fail(
            HS_STATUS_INTERNAL_ERROR,
            "unknown native exception",
            error_text,
            error_capacity);
    }
}

}  // namespace

cudaError_t hs_context_reserve_geometry(
    HsRasterizerContext* context,
    uint64_t required,
    char** pointer) noexcept {
    return reserve_buffer(
        &context->geometry,
        &context->geometry_bytes,
        required,
        context,
        pointer);
}

cudaError_t hs_context_reserve_binning(
    HsRasterizerContext* context,
    uint64_t required,
    char** pointer) noexcept {
    return reserve_buffer(
        &context->binning,
        &context->binning_bytes,
        required,
        context,
        pointer);
}

cudaError_t hs_context_reserve_image(
    HsRasterizerContext* context,
    uint64_t required,
    char** pointer) noexcept {
    return reserve_buffer(
        &context->image,
        &context->image_bytes,
        required,
        context,
        pointer);
}

cudaError_t hs_context_release_buffers(HsRasterizerContext* context) noexcept {
    cudaError_t first = cudaSuccess;
    for (void** buffer : {&context->geometry, &context->binning, &context->image}) {
        if (*buffer == nullptr) {
            continue;
        }
        const cudaError_t error = cudaFree(*buffer);
        if (first == cudaSuccess && error != cudaSuccess) {
            first = error;
        }
        *buffer = nullptr;
    }
    context->geometry_bytes = 0;
    context->binning_bytes = 0;
    context->image_bytes = 0;
    return first;
}

cudaError_t hs_context_initialize_math(HsRasterizerContext* context) noexcept {
    if (context == nullptr) {
        return cudaErrorInvalidValue;
    }
    char error_text[1] = {};
    return initialize_math_context(context, error_text, sizeof(error_text)) ==
            HS_STATUS_OK
        ? cudaSuccess
        : cudaErrorUnknown;
}

cudaError_t hs_context_release_math(HsRasterizerContext* context) noexcept {
    if (context == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t first = cudaSuccess;
    if (context->math_workspace != nullptr) {
        const cudaError_t status = cudaFree(context->math_workspace);
        if (first == cudaSuccess && status != cudaSuccess) {
            first = status;
        }
        context->math_workspace = nullptr;
    }
    if (context->math_tf32_blas != nullptr) {
        if (cublasDestroy(context->math_tf32_blas) != CUBLAS_STATUS_SUCCESS &&
            first == cudaSuccess) {
            first = cudaErrorUnknown;
        }
        context->math_tf32_blas = nullptr;
    }
    if (context->math_default_blas != nullptr) {
        if (cublasDestroy(context->math_default_blas) != CUBLAS_STATUS_SUCCESS &&
            first == cudaSuccess) {
            first = cudaErrorUnknown;
        }
        context->math_default_blas = nullptr;
    }
    if (context->math_solver != nullptr) {
        if (cusolverDnDestroy(context->math_solver) != CUSOLVER_STATUS_SUCCESS &&
            first == cudaSuccess) {
            first = cudaErrorUnknown;
        }
        context->math_solver = nullptr;
    }
    context->math_workspace_bytes = 0;
    return first;
}

uint32_t hs_rasterizer_abi_version(void) noexcept {
    return HS_RASTERIZER_ABI_VERSION;
}

int32_t hs_rasterizer_build_info(
    char* output_text,
    uint64_t output_capacity) noexcept {
    if (output_text == nullptr || output_capacity == 0) {
        return static_cast<int32_t>(HS_STATUS_INVALID_ARGUMENT);
    }
    const char* info =
        "{\"abi\":2,\"build_id\":\"" HS_RASTERIZER_BUILD_ID "\","
        "\"cuda\":\"12.6\",\"sm\":89,"
        "\"cxx_standard\":17,\"cuda_standard\":17,"
        "\"fast_math\":false,\"stream\":0,"
        "\"precise_exp\":\"v1\",\"precise_sh_to_rgb\":\"v1\","
        "\"precise_direction\":\"v1\","
        "\"precise_camera_center\":\"pytorch_sgemm_nt_m1n3k3_tf32_v3\","
        "\"precise_general_inverse\":"
        "\"pytorch_cusolver_sgetrf_cublas_sgetrs_batched_t_v2\","
        "\"precise_matmul_nt_n3k3\":\"pytorch_sgemm_tn_tf32_v3\","
        "\"precise_pixel_world\":\"cublas_fp32_persistent_v2\","
        "\"math_resources\":\"persistent_context_v2\","
        "\"precise_disk_rotation\":\"profiling_fp32_v1\","
        "\"precise_reflection_prefix\":\"texpr_fma_v1\","
        "\"precise_reflection_suffix\":\"fp32_v1\","
        "\"five_channel_rasterizer\":\"forward_v1\","
        "\"flashgs_forward\":\"deterministic_two_pass_v1\","
        "\"flashgs_preprocess_invisible_slots\":\"zero\","
        "\"optix_reflection_tracer\":\"forward_v1\","
        "\"optix_ptx\":\"cuda12.6_ptx8.5_compute89\","
        "\"workspace_growth\":\"1.25x_align_2mib\"}";
    if (std::strlen(info) + 1 > output_capacity) {
        write_text(output_text, output_capacity, info);
        return static_cast<int32_t>(HS_STATUS_RANGE_ERROR);
    }
    write_text(output_text, output_capacity, info);
    return static_cast<int32_t>(HS_STATUS_OK);
}

int32_t hs_rasterizer_create(
    int32_t device,
    HsRasterizerHandle* handle,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "handle output is null",
                error_text,
                error_capacity);
        }
        *handle = nullptr;

        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess) {
            return cuda_failure(
                error, "cudaGetDeviceCount", error_text, error_capacity);
        }
        if (device < 0 || device >= device_count) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "device is outside the visible CUDA device range",
                error_text,
                error_capacity);
        }
        error = cudaSetDevice(device);
        if (error != cudaSuccess) {
            return cuda_failure(
                error, "cudaSetDevice", error_text, error_capacity);
        }

        *handle = new HsRasterizerContext(device);
        const int32_t math_status = initialize_math_context(
            *handle, error_text, error_capacity);
        if (math_status != HS_STATUS_OK) {
            hs_context_release_math(*handle);
            delete *handle;
            *handle = nullptr;
            return math_status;
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_synchronize(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        const cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            return cuda_failure(
                error, "cudaDeviceSynchronize", error_text, error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_workspace_stats(
    HsRasterizerHandle handle,
    HsRasterizerWorkspaceStatsV1* stats,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr || stats == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "handle or stats is null",
                error_text,
                error_capacity);
        }
        if (stats->struct_size != sizeof(HsRasterizerWorkspaceStatsV1) ||
            stats->abi_version != HS_RASTERIZER_ABI_VERSION) {
            return fail(
                HS_STATUS_ABI_MISMATCH,
                "workspace stats ABI does not match",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        stats->device = handle->device;
        stats->reserved = 0;
        stats->geometry_bytes = handle->geometry_bytes;
        stats->binning_bytes = handle->binning_bytes;
        stats->image_bytes = handle->image_bytes;
        stats->current_bytes = handle->geometry_bytes +
                               handle->binning_bytes +
                               handle->image_bytes;
        stats->peak_bytes = handle->peak_bytes;
        stats->growth_count = handle->growth_count;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_release_workspace(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        cudaError_t error = cudaDeviceSynchronize();
        if (error != cudaSuccess) {
            return cuda_failure(
                error, "cudaDeviceSynchronize", error_text, error_capacity);
        }
        error = hs_context_release_buffers(handle);
        if (error != cudaSuccess) {
            return cuda_failure(
                error, "release rasterizer workspace", error_text, error_capacity);
        }
        error = hs_flashgs_release_context(handle);
        if (error != cudaSuccess) {
            return cuda_failure(
                error, "release FlashGS workspace", error_text, error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_math_workspace_stats_v2(
    HsRasterizerHandle handle,
    HsMathWorkspaceStatsV2* stats,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr || stats == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "handle or math stats is null",
                error_text,
                error_capacity);
        }
        if (stats->struct_size != sizeof(HsMathWorkspaceStatsV2) ||
            stats->abi_version != HS_RASTERIZER_ABI_VERSION) {
            return fail(
                HS_STATUS_ABI_MISMATCH,
                "math workspace stats ABI does not match",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        stats->device = handle->device;
        stats->handles_ready =
            handle->math_tf32_blas != nullptr &&
            handle->math_default_blas != nullptr &&
            handle->math_solver != nullptr
            ? 1u
            : 0u;
        stats->workspace_bytes = handle->math_workspace_bytes;
        stats->current_bytes = handle->math_workspace_bytes;
        stats->peak_bytes = handle->math_peak_bytes;
        stats->growth_count = handle->math_growth_count;
        stats->handle_create_count = handle->math_handle_create_count;
        stats->camera_center_calls = handle->math_camera_center_calls;
        stats->general_inverse_calls = handle->math_general_inverse_calls;
        stats->matmul_calls = handle->math_matmul_calls;
        stats->pixel_world_calls = handle->math_pixel_world_calls;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_destroy(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "handle is null",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        {
            std::lock_guard<std::mutex> lock(handle->mutex);
            const cudaError_t error = cudaDeviceSynchronize();
            if (error != cudaSuccess) {
                return cuda_failure(
                    error,
                    "cudaDeviceSynchronize",
                    error_text,
                    error_capacity);
            }
            const cudaError_t release_error = hs_context_release_buffers(handle);
            if (release_error != cudaSuccess) {
                return cuda_failure(
                    release_error,
                    "release rasterizer workspace",
                    error_text,
                    error_capacity);
            }
            const cudaError_t flashgs_error = hs_flashgs_release_context(handle);
            if (flashgs_error != cudaSuccess) {
                return cuda_failure(
                    flashgs_error,
                    "release FlashGS context",
                    error_text,
                    error_capacity);
            }
            hs_optix_release_context(handle);
            const cudaError_t math_error = hs_context_release_math(handle);
            if (math_error != cudaSuccess) {
                return cuda_failure(
                    math_error,
                    "release math context",
                    error_text,
                    error_capacity);
            }
        }
        delete handle;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_flashgs_forward_f32_v1(
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
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr || camera_position3 == nullptr ||
            camera_rotation9 == nullptr || background3 == nullptr ||
            output_color3 == nullptr || output_other2 == nullptr ||
            output_depth1 == nullptr || output_alpha1 == nullptr ||
            output_normal3 == nullptr || output_median_depth1 == nullptr ||
            output_distortion1 == nullptr || output_num_rendered == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "FlashGS received a null required handle or buffer",
                error_text,
                error_capacity);
        }
        if (point_count > static_cast<uint64_t>(INT_MAX) || width <= 0 ||
            height <= 0 || (use_sh != 0 && use_sh != 1) ||
            initial_pair_capacity == 0 ||
            initial_pair_capacity > static_cast<uint64_t>(INT_MAX) ||
            !std::isfinite(scale_modifier) || scale_modifier <= 0.0f ||
            !std::isfinite(focal_x) || focal_x <= 0.0f ||
            !std::isfinite(focal_y) || focal_y <= 0.0f ||
            !std::isfinite(z_near) || z_near <= 0.0f ||
            !std::isfinite(z_far) || z_far <= z_near) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "FlashGS dimensions or scalar arguments are invalid",
                error_text,
                error_capacity);
        }
        for (int index = 0; index < 3; ++index) {
            if (!std::isfinite(camera_position3[index]) ||
                !std::isfinite(background3[index])) {
                return fail(
                    HS_STATUS_RANGE_ERROR,
                    "FlashGS camera position or background is not finite",
                    error_text,
                    error_capacity);
            }
        }
        for (int index = 0; index < 9; ++index) {
            if (!std::isfinite(camera_rotation9[index])) {
                return fail(
                    HS_STATUS_RANGE_ERROR,
                    "FlashGS camera rotation is not finite",
                    error_text,
                    error_capacity);
            }
        }
        if (point_count != 0 &&
            (positions3 == nullptr || opacities1 == nullptr ||
             covariance6 == nullptr || scales2 == nullptr ||
             rotations4 == nullptr || specular1 == nullptr ||
             roughness1 == nullptr || (use_sh != 0 && shs48 == nullptr) ||
             (use_sh == 0 && colors3 == nullptr))) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "non-empty FlashGS input contains a null buffer",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) return device_status;
        std::lock_guard<std::mutex> lock(handle->mutex);
        const cudaError_t status = hs_launch_flashgs_forward_f32(
            handle,
            static_cast<int32_t>(point_count),
            width,
            height,
            positions3,
            shs48,
            colors3,
            use_sh != 0,
            opacities1,
            covariance6,
            scales2,
            scale_modifier,
            rotations4,
            specular1,
            roughness1,
            camera_position3,
            camera_rotation9,
            focal_x,
            focal_y,
            z_far,
            z_near,
            background3,
            initial_pair_capacity,
            output_color3,
            output_other2,
            output_depth1,
            output_alpha1,
            output_normal3,
            output_median_depth1,
            output_distortion1,
            output_num_rendered);
        if (status != cudaSuccess) {
            return cuda_failure(
                status, "deterministic FlashGS forward", error_text, error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_flashgs_workspace_stats_v1(
    HsRasterizerHandle handle,
    HsFlashgsWorkspaceStatsV1* stats,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr || stats == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "handle or FlashGS stats is null",
                error_text,
                error_capacity);
        }
        if (stats->struct_size != sizeof(HsFlashgsWorkspaceStatsV1) ||
            stats->abi_version != HS_RASTERIZER_ABI_VERSION) {
            return fail(
                HS_STATUS_ABI_MISMATCH,
                "FlashGS workspace stats ABI does not match",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) return device_status;
        std::lock_guard<std::mutex> lock(handle->mutex);
        stats->device = handle->device;
        stats->initialized = handle->flashgs_pair_workspace != nullptr ? 1u : 0u;
        stats->gaussian_capacity = handle->flashgs_gaussian_capacity;
        stats->pair_capacity = handle->flashgs_pair_capacity;
        stats->range_capacity = handle->flashgs_range_capacity;
        stats->output_pixel_capacity = handle->flashgs_output_pixel_capacity;
        stats->gaussian_bytes = handle->flashgs_gaussian_bytes;
        stats->pair_bytes = handle->flashgs_pair_bytes;
        stats->cub_bytes = handle->flashgs_cub_bytes;
        stats->ranges_bytes = handle->flashgs_ranges_bytes;
        stats->output_bytes = handle->flashgs_output_bytes;
        stats->current_bytes =
            handle->flashgs_gaussian_bytes + handle->flashgs_pair_bytes +
            handle->flashgs_cub_bytes + handle->flashgs_ranges_bytes +
            handle->flashgs_output_bytes +
            (handle->flashgs_counter_device == nullptr ? 0ull : sizeof(int32_t));
        stats->peak_bytes = handle->flashgs_peak_bytes;
        stats->growth_count = handle->flashgs_growth_count;
        stats->overflow_retry_count = handle->flashgs_overflow_retry_count;
        stats->forward_calls = handle->flashgs_forward_calls;
        stats->last_point_count = handle->flashgs_last_point_count;
        stats->last_num_rendered = handle->flashgs_last_num_rendered;
        stats->last_tile_count = handle->flashgs_last_tile_count;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_flashgs_debug_copy_v1(
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
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "FlashGS debug copy received a null context",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) return device_status;
        std::lock_guard<std::mutex> lock(handle->mutex);
        const cudaError_t status = hs_flashgs_copy_debug_to_host(
            handle,
            point_capacity,
            pair_capacity,
            tile_capacity,
            points_xy2,
            rgb3,
            other2,
            conic_opacity4,
            normal_opacity4,
            tu3,
            tv3,
            tw3,
            sorted_keys,
            sorted_values,
            tile_ranges2);
        if (status != cudaSuccess) {
            return cuda_failure(status, "FlashGS debug copy", error_text, error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_forward_f32_v1(
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
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr || background5 == nullptr ||
            output_color5 == nullptr || output_aux7 == nullptr ||
            output_rendered == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "rasterizer received a null required handle or buffer",
                error_text,
                error_capacity);
        }
        if (point_count > static_cast<uint64_t>(INT_MAX) || width <= 0 ||
            height <= 0 || !std::isfinite(scale_modifier) ||
            !std::isfinite(tan_fovx) || !std::isfinite(tan_fovy) ||
            tan_fovx <= 0.0f || tan_fovy <= 0.0f ||
            (prefiltered != 0 && prefiltered != 1)) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "rasterizer dimensions or scalar arguments are invalid",
                error_text,
                error_capacity);
        }
        if (point_count != 0 &&
            (means3d == nullptr || colors5 == nullptr ||
             opacities == nullptr || scales2 == nullptr ||
             rotations4 == nullptr || viewmatrix4x4 == nullptr ||
             projmatrix4x4 == nullptr || camera_position3 == nullptr ||
             output_weight == nullptr || output_radii == nullptr)) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "non-empty rasterizer input contains a null buffer",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        const cudaError_t error = hs_launch_full_rasterizer_forward_f32(
            handle,
            static_cast<int32_t>(point_count),
            background5,
            width,
            height,
            means3d,
            colors5,
            opacities,
            scales2,
            scale_modifier,
            rotations4,
            viewmatrix4x4,
            projmatrix4x4,
            camera_position3,
            tan_fovx,
            tan_fovy,
            prefiltered != 0,
            output_color5,
            output_aux7,
            output_weight,
            output_radii,
            output_rendered);
        if (error != cudaSuccess) {
            return cuda_failure(
                error,
                "five-channel rasterizer forward",
                error_text,
                error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_exp_f32_v1(
    const float* input,
    float* output,
    uint64_t count,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (input == nullptr || output == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_exp received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error =
            hs_launch_precise_exp_f32(input, output, count);
        if (error != cudaSuccess) {
            return cuda_failure(
                error,
                "precise_exp launch",
                error_text,
                error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_sigmoid_f32_v1(
    const float* input,
    float* output,
    uint64_t count,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (input == nullptr || output == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_sigmoid received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error = hs_launch_precise_sigmoid_f32(input, output, count);
        if (error != cudaSuccess) {
            return cuda_failure(error, "precise_sigmoid launch", error_text, error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_normalize4_f32_v1(
    const float* input4,
    float* output4,
    uint64_t count,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (input4 == nullptr || output4 == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_normalize4 received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error = hs_launch_precise_normalize4_f32(input4, output4, count);
        if (error != cudaSuccess) {
            return cuda_failure(error, "precise_normalize4 launch", error_text, error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_fuse_maps_f32_v1(
    const float* base_rgb,
    const float* reflection_rgb,
    const float* specular,
    float* diffuse,
    float* final_rgb,
    float* stored_reflection,
    uint64_t pixel_count,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (pixel_count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (base_rgb == nullptr || reflection_rgb == nullptr || specular == nullptr ||
            diffuse == nullptr || final_rgb == nullptr || stored_reflection == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_fuse_maps received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error = hs_launch_precise_fuse_maps_f32(
            base_rgb,
            reflection_rgb,
            specular,
            diffuse,
            final_rgb,
            stored_reflection,
            pixel_count);
        if (error != cudaSuccess) {
            return cuda_failure(error, "precise_fuse_maps launch", error_text, error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_sh_to_rgb_f32_v1(
    const float* sh,
    const float* directions,
    uint64_t count,
    uint32_t coefficient_count,
    uint32_t degree,
    float* output,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (degree > 3) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "precise_sh_to_rgb supports degrees 0 through 3",
                error_text,
                error_capacity);
        }
        const uint32_t required = (degree + 1) * (degree + 1);
        if (coefficient_count < required) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "precise_sh_to_rgb coefficient count is too small",
                error_text,
                error_capacity);
        }
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (sh == nullptr || directions == nullptr || output == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_sh_to_rgb received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error = hs_launch_precise_sh_to_rgb_f32(
            sh,
            directions,
            count,
            coefficient_count,
            degree,
            output);
        if (error != cudaSuccess) {
            return cuda_failure(
                error,
                "precise_sh_to_rgb launch",
                error_text,
                error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_direction_f32_v1(
    const float* points,
    const float* center,
    uint64_t count,
    float* output,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (points == nullptr || center == nullptr || output == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_direction received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error =
            hs_launch_precise_direction_f32(points, center, count, output);
        if (error != cudaSuccess) {
            return cuda_failure(
                error,
                "precise_direction launch",
                error_text,
                error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_camera_center_f32_v2(
    HsRasterizerHandle handle,
    const float* rotation3x3,
    const float* translation3x1,
    float* output3,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (handle == nullptr || rotation3x3 == nullptr ||
            translation3x1 == nullptr ||
            output3 == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_camera_center received a null buffer",
                error_text,
                error_capacity);
        }

        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (handle->math_workspace == nullptr ||
            handle->math_tf32_blas == nullptr) {
            return fail(
                HS_STATUS_INTERNAL_ERROR,
                "precise_camera_center math context is not initialized",
                error_text,
                error_capacity);
        }
        const MathWorkspacePointers pointers = math_pointers(handle);
        cudaError_t cuda_status = hs_launch_negative_rotation3x3_f32(
            rotation3x3,
            pointers.negative_rotation3x3);

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
        if (cuda_status == cudaSuccess) {
            // PyTorch 2.6 sees the contiguous [3,1] result as column-major
            // [1,3]. Its unary negation preserves R.mT's [1,3] strides, so
            // the physical B buffer is contiguous row-major -R.
            status = cublasSgemm(
                handle->math_tf32_blas,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                1,
                3,
                3,
                &alpha,
                translation3x1,
                1,
                pointers.negative_rotation3x3,
                3,
                &beta,
                output3,
                1);
        }

        if (cuda_status != cudaSuccess) {
            return cuda_failure(
                cuda_status,
                "precise_camera_center negative-R",
                error_text,
                error_capacity);
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            return cublas_failure(
                status,
                "precise_camera_center SGEMM",
                error_text,
                error_capacity);
        }
        ++handle->math_camera_center_calls;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_general_inverse_f32_v2(
    HsRasterizerHandle handle,
    const float* input,
    uint32_t dimension,
    float* output,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (dimension != 3 && dimension != 4) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "precise_general_inverse dimension must be 3 or 4",
                error_text,
                error_capacity);
        }
        if (handle == nullptr || input == nullptr || output == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_general_inverse received a null buffer",
                error_text,
                error_capacity);
        }

        const cudaError_t prior_error = cudaGetLastError();
        if (prior_error != cudaSuccess) {
            return cuda_failure(
                prior_error,
                "precise_general_inverse preexisting CUDA error",
                error_text,
                error_capacity);
        }

        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (handle->math_workspace == nullptr ||
            handle->math_solver == nullptr ||
            handle->math_tf32_blas == nullptr) {
            return fail(
                HS_STATUS_INTERNAL_ERROR,
                "precise_general_inverse math context is not initialized",
                error_text,
                error_capacity);
        }
        const MathWorkspacePointers pointers = math_pointers(handle);
        cusolverDnHandle_t solver = handle->math_solver;
        cublasHandle_t blas = handle->math_tf32_blas;
        float* factor = pointers.factor4x4;
        float* rhs = pointers.rhs4x4;
        float* workspace = pointers.solver_workspace;
        int* pivots = pointers.pivots4;
        int* device_info = pointers.device_info;
        float** factor_array = pointers.factor_array;
        float** rhs_array = pointers.rhs_array;
        int32_t result = static_cast<int32_t>(HS_STATUS_OK);
        const size_t matrix_count =
            static_cast<size_t>(dimension) * static_cast<size_t>(dimension);
        const size_t matrix_bytes = matrix_count * sizeof(float);
        cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
        cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;
        do {
            cudaError_t cuda_status = cudaSuccess;

            // PyTorch initializes the identity result before solve_ex copies
            // contiguous A.mT into its column-major LU buffer.
            cuda_status = cudaMemcpyAsync(
                rhs,
                dimension == 3
                    ? pointers.identity3x3
                    : pointers.identity4x4,
                matrix_bytes,
                cudaMemcpyDeviceToDevice,
                nullptr);
            if (cuda_status == cudaSuccess) {
                cuda_status = cudaMemcpyAsync(
                    factor,
                    input,
                    matrix_bytes,
                    cudaMemcpyDeviceToDevice,
                    nullptr);
            }
            if (cuda_status != cudaSuccess) {
                result = cuda_failure(
                    cuda_status,
                    "precise_general_inverse identity/input copy",
                    error_text,
                    error_capacity);
                break;
            }

            int workspace_count = 0;
            solver_status = cusolverDnSgetrf_bufferSize(
                solver,
                static_cast<int>(dimension),
                static_cast<int>(dimension),
                factor,
                static_cast<int>(dimension),
                &workspace_count);
            if (solver_status != CUSOLVER_STATUS_SUCCESS) {
                result = cusolver_failure(
                    solver_status,
                    "precise_general_inverse cusolverDnSgetrf_bufferSize",
                    error_text,
                    error_capacity);
                break;
            }
            if (workspace_count > handle->math_inverse_workspace_count) {
                result = fail(
                    HS_STATUS_RANGE_ERROR,
                    "precise_general_inverse persistent workspace is too small",
                    error_text,
                    error_capacity);
                break;
            }
            solver_status = cusolverDnSgetrf(
                solver,
                static_cast<int>(dimension),
                static_cast<int>(dimension),
                factor,
                static_cast<int>(dimension),
                workspace,
                pivots,
                device_info);
            if (solver_status != CUSOLVER_STATUS_SUCCESS) {
                result = cusolver_failure(
                    solver_status,
                    "precise_general_inverse cusolverDnSgetrf",
                    error_text,
                    error_capacity);
                break;
            }

            int32_t* solve_info = &handle->math_solve_info[
                handle->math_general_inverse_calls % 16];
            *solve_info = 0;
            blas_status = cublasSgetrsBatched(
                blas,
                CUBLAS_OP_T,
                static_cast<int>(dimension),
                static_cast<int>(dimension),
                factor_array,
                static_cast<int>(dimension),
                pivots,
                rhs_array,
                static_cast<int>(dimension),
                solve_info,
                1);
            if (blas_status != CUBLAS_STATUS_SUCCESS) {
                result = cublas_failure(
                    blas_status,
                    "precise_general_inverse cublasSgetrsBatched",
                    error_text,
                    error_capacity);
                break;
            }
            cuda_status = hs_launch_column_major_to_row_major_f32(
                rhs,
                dimension,
                output);
            if (cuda_status != cudaSuccess) {
                result = cuda_failure(
                    cuda_status,
                    "precise_general_inverse output transpose",
                    error_text,
                    error_capacity);
                break;
            }
        } while (false);

        const cudaError_t trailing_error = cudaGetLastError();

        if (result != HS_STATUS_OK) {
            return result;
        }
        if (trailing_error != cudaSuccess) {
            return cuda_failure(
                trailing_error,
                "precise_general_inverse trailing CUDA error",
                error_text,
                error_capacity);
        }
        ++handle->math_general_inverse_calls;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_matmul_nt_n3k3_f32_v2(
    HsRasterizerHandle handle,
    const float* left_n3,
    const float* right3x3,
    uint64_t count,
    float* output_n3,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (count > static_cast<uint64_t>(INT_MAX)) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "precise_matmul_nt_n3k3 count exceeds cuBLAS int range",
                error_text,
                error_capacity);
        }
        if (handle == nullptr || left_n3 == nullptr || right3x3 == nullptr ||
            output_n3 == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_matmul_nt_n3k3 received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t prior_error = cudaGetLastError();
        if (prior_error != cudaSuccess) {
            return cuda_failure(
                prior_error,
                "precise_matmul_nt_n3k3 preexisting CUDA error",
                error_text,
                error_capacity);
        }

        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (handle->math_tf32_blas == nullptr) {
            return fail(
                HS_STATUS_INTERNAL_ERROR,
                "precise_matmul_nt_n3k3 math context is not initialized",
                error_text,
                error_capacity);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t status = cublasSgemm(
            handle->math_tf32_blas,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            3,
            static_cast<int>(count),
            3,
            &alpha,
            right3x3,
            3,
            left_n3,
            3,
            &beta,
            output_n3,
            3);
        const cudaError_t cuda_status = cudaGetLastError();
        const cudaError_t trailing_error = cudaGetLastError();
        if (status != CUBLAS_STATUS_SUCCESS) {
            return cublas_failure(
                status,
                "precise_matmul_nt_n3k3 cublasSgemm",
                error_text,
                error_capacity);
        }
        if (cuda_status != cudaSuccess) {
            return cuda_failure(
                cuda_status,
                "precise_matmul_nt_n3k3 execution",
                error_text,
                error_capacity);
        }
        if (trailing_error != cudaSuccess) {
            return cuda_failure(
                trailing_error,
                "precise_matmul_nt_n3k3 trailing CUDA error",
                error_text,
                error_capacity);
        }
        ++handle->math_matmul_calls;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_pixel_world_f32_v2(
    HsRasterizerHandle handle,
    const float* rotation3x3,
    const float* vectors_n3,
    uint64_t count,
    float* output_n3,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (count > static_cast<uint64_t>(INT_MAX)) {
            return fail(
                HS_STATUS_RANGE_ERROR,
                "precise_pixel_world count exceeds cuBLAS int range",
                error_text,
                error_capacity);
        }
        if (handle == nullptr || rotation3x3 == nullptr ||
            vectors_n3 == nullptr || output_n3 == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_pixel_world received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const int32_t device_status =
            select_device(handle, error_text, error_capacity);
        if (device_status != HS_STATUS_OK) {
            return device_status;
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (handle->math_default_blas == nullptr) {
            return fail(
                HS_STATUS_INTERNAL_ERROR,
                "precise_pixel_world math context is not initialized",
                error_text,
                error_capacity);
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const cublasStatus_t status = cublasGemmStridedBatchedEx(
            handle->math_default_blas,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            3,
            1,
            3,
            &alpha,
            rotation3x3,
            CUDA_R_32F,
            3,
            0,
            vectors_n3,
            CUDA_R_32F,
            3,
            3,
            &beta,
            output_n3,
            CUDA_R_32F,
            3,
            3,
            static_cast<int>(count),
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return cublas_failure(
                status,
                "precise_pixel_world batched GEMM",
                error_text,
                error_capacity);
        }
        const cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            return cuda_failure(
                cuda_status,
                "precise_pixel_world execution",
                error_text,
                error_capacity);
        }
        ++handle->math_pixel_world_calls;
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_disk_rotation_f32_v1(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (rotation_wxyz == nullptr || rotation_matrix9 == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_disk_rotation received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error = hs_launch_precise_disk_rotation_f32(
            rotation_wxyz,
            count,
            rotation_matrix9);
        if (error != cudaSuccess) {
            return cuda_failure(
                error,
                "precise_disk_rotation launch",
                error_text,
                error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_reflection_prefix_f32_v1(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (rotation_wxyz == nullptr || rotation_matrix9 == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_reflection_prefix received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error = hs_launch_precise_reflection_prefix_f32(
            rotation_wxyz,
            count,
            rotation_matrix9);
        if (error != cudaSuccess) {
            return cuda_failure(
                error,
                "precise_reflection_prefix launch",
                error_text,
                error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}

int32_t hs_rasterizer_precise_reflection_suffix_f32_v1(
    const float* xyz,
    const float* camera_center,
    const float* normal_raw3,
    uint64_t count,
    float* output36,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guard(error_text, error_capacity, [&]() -> int32_t {
        if (count == 0) {
            clear_error(error_text, error_capacity);
            return static_cast<int32_t>(HS_STATUS_OK);
        }
        if (xyz == nullptr || camera_center == nullptr ||
            normal_raw3 == nullptr || output36 == nullptr) {
            return fail(
                HS_STATUS_INVALID_ARGUMENT,
                "precise_reflection_suffix received a null non-empty buffer",
                error_text,
                error_capacity);
        }
        const cudaError_t error = hs_launch_precise_reflection_suffix_f32(
            xyz,
            camera_center,
            normal_raw3,
            count,
            output36);
        if (error != cudaSuccess) {
            return cuda_failure(
                error,
                "precise_reflection_suffix launch",
                error_text,
                error_capacity);
        }
        clear_error(error_text, error_capacity);
        return static_cast<int32_t>(HS_STATUS_OK);
    });
}
