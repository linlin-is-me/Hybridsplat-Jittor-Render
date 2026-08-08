#include "hybridsplat_rasterizer_c_api.h"

#include "context_internal.h"
#include "optix_backend.h"
#include "params.h"

#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

namespace {

constexpr uint64_t kAlignment = 2ull * 1024ull * 1024ull;
const int kLibraryLocationAnchor = 0;

struct EmptyRecordData {};

template <typename T>
struct SbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct NativeOptixState {
    OptixDeviceContext context = nullptr;
    OptixModule module = nullptr;
    OptixProgramGroup raygen = nullptr;
    OptixProgramGroup miss = nullptr;
    OptixProgramGroup hitgroup = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixShaderBindingTable sbt = {};
    void* raygen_record = nullptr;
    void* miss_record = nullptr;
    void* hitgroup_record = nullptr;
    void* gas_output = nullptr;
    uint64_t gas_output_bytes = 0;
    OptixTraversableHandle gas_handle = 0;
};

struct BackendError : std::runtime_error {
    BackendError(HsStatus status_value, const std::string& message)
        : std::runtime_error(message), status(status_value) {}
    HsStatus status;
};

uint64_t align_capacity(uint64_t value) {
    return (std::max(value, uint64_t{1}) + kAlignment - 1) & ~(kAlignment - 1);
}

void write_error(char* destination, uint64_t capacity, const std::string& text) noexcept {
    if (destination == nullptr || capacity == 0) return;
    const size_t count = std::min<size_t>(text.size(), capacity - 1);
    if (count) std::memcpy(destination, text.data(), count);
    destination[count] = '\0';
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    const HsStatus status = error == cudaErrorMemoryAllocation
        ? HS_STATUS_OUT_OF_MEMORY
        : HS_STATUS_CUDA_ERROR;
    throw BackendError(
        status,
        std::string(operation) + ": " + cudaGetErrorString(error));
}

void check_optix(OptixResult result, const char* operation) {
    if (result == OPTIX_SUCCESS) return;
    throw BackendError(
        HS_STATUS_INTERNAL_ERROR,
        std::string(operation) + ": " + optixGetErrorString(result));
}

void optix_log(unsigned int, const char*, const char*, void*) {}

std::filesystem::path resolve_ptx_path() {
    const char* override_path = std::getenv("HYBRIDSPLAT_OPTIX_PTX_PATH");
    if (override_path != nullptr && override_path[0] != '\0') {
        return std::filesystem::absolute(override_path).lexically_normal();
    }
    Dl_info library_info = {};
    if (dladdr(static_cast<const void*>(&kLibraryLocationAnchor), &library_info) == 0 ||
        library_info.dli_fname == nullptr || library_info.dli_fname[0] == '\0') {
        throw BackendError(
            HS_STATUS_INTERNAL_ERROR,
            "cannot locate libhybridsplat_rasterizer.so for OptiX PTX discovery");
    }
    const std::filesystem::path library_path =
        std::filesystem::absolute(library_info.dli_fname).lexically_normal();
    return (
        library_path.parent_path().parent_path() /
        "share" / "hybridsplat" / "forward.ptx").lexically_normal();
}

std::string read_ptx() {
    const std::filesystem::path ptx_path = resolve_ptx_path();
    std::ifstream stream(ptx_path, std::ios::binary);
    if (!stream) {
        throw BackendError(
            HS_STATUS_INTERNAL_ERROR,
            std::string("cannot open OptiX PTX: ") + ptx_path.string());
    }
    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

void destroy_state(NativeOptixState* state) noexcept {
    if (state == nullptr) return;
    if (state->pipeline) optixPipelineDestroy(state->pipeline);
    if (state->raygen) optixProgramGroupDestroy(state->raygen);
    if (state->miss) optixProgramGroupDestroy(state->miss);
    if (state->hitgroup) optixProgramGroupDestroy(state->hitgroup);
    if (state->module) optixModuleDestroy(state->module);
    if (state->raygen_record) cudaFree(state->raygen_record);
    if (state->miss_record) cudaFree(state->miss_record);
    if (state->hitgroup_record) cudaFree(state->hitgroup_record);
    if (state->gas_output) cudaFree(state->gas_output);
    if (state->context) optixDeviceContextDestroy(state->context);
    delete state;
}

NativeOptixState* initialize_state(HsRasterizerContext* context) {
    if (context->optix_state != nullptr) {
        return static_cast<NativeOptixState*>(context->optix_state);
    }
    check_cuda(cudaSetDevice(context->device), "cudaSetDevice");
    check_cuda(cudaFree(nullptr), "initialize CUDA context");
    check_optix(optixInit(), "optixInit");

    NativeOptixState* state = new NativeOptixState();
    try {
        OptixDeviceContextOptions context_options = {};
        context_options.logCallbackFunction = optix_log;
        context_options.logCallbackLevel = 3;
        context_options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
        check_optix(
            optixDeviceContextCreate(nullptr, &context_options, &state->context),
            "optixDeviceContextCreate");

        const std::string ptx = read_ptx();
        OptixModuleCompileOptions module_options = {};
        module_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
        OptixPipelineCompileOptions compile_options = {};
        compile_options.usesMotionBlur = false;
        compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        compile_options.numPayloadValues = 2;
        compile_options.numAttributeValues = 2;
        compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        compile_options.pipelineLaunchParamsVariableName = "params";
        compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        char log[4096] = {};
        size_t log_size = sizeof(log);
        const OptixResult module_result = optixModuleCreate(
            state->context,
            &module_options,
            &compile_options,
            ptx.data(),
            ptx.size(),
            log,
            &log_size,
            &state->module);
        if (module_result != OPTIX_SUCCESS) {
            throw BackendError(
                HS_STATUS_INTERNAL_ERROR,
                std::string("optixModuleCreate: ") + optixGetErrorString(module_result) +
                    " " + std::string(log, std::min(log_size, sizeof(log))));
        }

        OptixProgramGroupOptions group_options = {};
        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = state->module;
        raygen_desc.raygen.entryFunctionName = "__raygen__ot";
        check_optix(
            optixProgramGroupCreate(
                state->context, &raygen_desc, 1, &group_options,
                nullptr, nullptr, &state->raygen),
            "create raygen program group");

        OptixProgramGroupDesc miss_desc = {};
        miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        check_optix(
            optixProgramGroupCreate(
                state->context, &miss_desc, 1, &group_options,
                nullptr, nullptr, &state->miss),
            "create miss program group");

        OptixProgramGroupDesc hit_desc = {};
        hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_desc.hitgroup.moduleAH = state->module;
        hit_desc.hitgroup.entryFunctionNameAH = "__anyhit__ot";
        check_optix(
            optixProgramGroupCreate(
                state->context, &hit_desc, 1, &group_options,
                nullptr, nullptr, &state->hitgroup),
            "create hitgroup program group");

        OptixProgramGroup groups[] = {state->raygen, state->miss, state->hitgroup};
        OptixPipelineLinkOptions link_options = {};
        link_options.maxTraceDepth = 31;
        check_optix(
            optixPipelineCreate(
                state->context,
                &compile_options,
                &link_options,
                groups,
                3,
                nullptr,
                nullptr,
                &state->pipeline),
            "optixPipelineCreate");
        OptixStackSizes stack = {};
        for (OptixProgramGroup group : groups) {
            check_optix(
                optixUtilAccumulateStackSizes(group, &stack, state->pipeline),
                "optixUtilAccumulateStackSizes");
        }
        uint32_t traversal = 0;
        uint32_t direct = 0;
        uint32_t continuation = 0;
        check_optix(
            optixUtilComputeStackSizes(
                &stack, 31, 0, 0, &traversal, &direct, &continuation),
            "optixUtilComputeStackSizes");
        check_optix(
            optixPipelineSetStackSize(
                state->pipeline, traversal, direct, continuation, 1),
            "optixPipelineSetStackSize");

        using Record = SbtRecord<EmptyRecordData>;
        Record raygen_record = {};
        Record miss_record = {};
        Record hit_record = {};
        check_optix(optixSbtRecordPackHeader(state->raygen, &raygen_record), "pack raygen SBT");
        check_optix(optixSbtRecordPackHeader(state->miss, &miss_record), "pack miss SBT");
        check_optix(optixSbtRecordPackHeader(state->hitgroup, &hit_record), "pack hit SBT");
        check_cuda(cudaMalloc(&state->raygen_record, sizeof(Record)), "allocate raygen SBT");
        check_cuda(cudaMalloc(&state->miss_record, sizeof(Record)), "allocate miss SBT");
        check_cuda(cudaMalloc(&state->hitgroup_record, sizeof(Record)), "allocate hit SBT");
        check_cuda(cudaMemcpy(state->raygen_record, &raygen_record, sizeof(Record), cudaMemcpyHostToDevice), "upload raygen SBT");
        check_cuda(cudaMemcpy(state->miss_record, &miss_record, sizeof(Record), cudaMemcpyHostToDevice), "upload miss SBT");
        check_cuda(cudaMemcpy(state->hitgroup_record, &hit_record, sizeof(Record), cudaMemcpyHostToDevice), "upload hit SBT");
        state->sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(state->raygen_record);
        state->sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(state->miss_record);
        state->sbt.missRecordStrideInBytes = sizeof(Record);
        state->sbt.missRecordCount = 1;
        state->sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(state->hitgroup_record);
        state->sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        state->sbt.hitgroupRecordCount = 1;
    } catch (...) {
        destroy_state(state);
        throw;
    }
    context->optix_state = state;
    ++context->optix_pipeline_create_count;
    return state;
}

void update_peak(HsRasterizerContext* context, NativeOptixState* state) {
    const uint64_t current = context->optix_params_bytes +
        context->optix_gas_scratch_bytes + context->optix_dist3_bytes +
        (state ? state->gas_output_bytes : 0);
    context->optix_peak_bytes = std::max(context->optix_peak_bytes, current);
}

void reserve_buffer(
    void** buffer,
    uint64_t* capacity,
    uint64_t required,
    HsRasterizerContext* context,
    NativeOptixState* state) {
    if (*buffer != nullptr && *capacity >= required) return;
    const uint64_t grown = *capacity == 0
        ? required
        : std::max(required, *capacity + *capacity / 4);
    const uint64_t target = align_capacity(grown);
    void* replacement = nullptr;
    check_cuda(cudaMalloc(&replacement, target), "allocate OptiX workspace");
    if (*buffer != nullptr) {
        check_cuda(cudaDeviceSynchronize(), "synchronize before OptiX workspace growth");
        check_cuda(cudaFree(*buffer), "free previous OptiX workspace");
    }
    *buffer = replacement;
    *capacity = target;
    ++context->optix_growth_count;
    update_peak(context, state);
}

template <typename Function>
int32_t guarded(
    char* error_text,
    uint64_t error_capacity,
    Function&& function) noexcept {
    if (error_capacity != 0 && error_text == nullptr) {
        return HS_STATUS_INVALID_ARGUMENT;
    }
    try {
        function();
        write_error(error_text, error_capacity, "");
        return HS_STATUS_OK;
    } catch (const BackendError& error) {
        write_error(error_text, error_capacity, error.what());
        return error.status;
    } catch (const std::bad_alloc& error) {
        write_error(error_text, error_capacity, error.what());
        return HS_STATUS_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        write_error(error_text, error_capacity, error.what());
        return HS_STATUS_INTERNAL_ERROR;
    } catch (...) {
        write_error(error_text, error_capacity, "unknown OptiX backend exception");
        return HS_STATUS_INTERNAL_ERROR;
    }
}

void release_scene_locked(HsRasterizerContext* context, bool destroy_pipeline) {
    NativeOptixState* state = static_cast<NativeOptixState*>(context->optix_state);
    check_cuda(cudaDeviceSynchronize(), "synchronize before OptiX release");
    if (context->optix_params) check_cuda(cudaFree(context->optix_params), "free OptiX params");
    if (context->optix_gas_scratch) check_cuda(cudaFree(context->optix_gas_scratch), "free GAS scratch");
    if (context->optix_dist3) check_cuda(cudaFree(context->optix_dist3), "free OptiX dist workspace");
    context->optix_params = nullptr;
    context->optix_gas_scratch = nullptr;
    context->optix_dist3 = nullptr;
    context->optix_params_bytes = 0;
    context->optix_gas_scratch_bytes = 0;
    context->optix_dist3_bytes = 0;
    context->optix_scene_signature = 0;
    if (state) {
        if (destroy_pipeline) {
            context->optix_state = nullptr;
            destroy_state(state);
        } else {
            if (state->gas_output) check_cuda(cudaFree(state->gas_output), "free GAS output");
            state->gas_output = nullptr;
            state->gas_output_bytes = 0;
            state->gas_handle = 0;
        }
    }
}

}  // namespace

void hs_optix_release_context(HsRasterizerContext* context) noexcept {
    if (context == nullptr) return;
    try {
        release_scene_locked(context, true);
    } catch (...) {
        NativeOptixState* state = static_cast<NativeOptixState*>(context->optix_state);
        context->optix_state = nullptr;
        destroy_state(state);
    }
}

int32_t hs_optix_build_scene_v1(
    HsRasterizerHandle handle,
    const float* vertices3,
    uint64_t vertex_count,
    const int32_t* triangles3,
    uint64_t triangle_count,
    uint64_t scene_signature,
    int32_t rebuild,
    int32_t* output_status,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guarded(error_text, error_capacity, [&]() {
        if (handle == nullptr || vertices3 == nullptr || triangles3 == nullptr ||
            output_status == nullptr) {
            throw BackendError(HS_STATUS_INVALID_ARGUMENT, "OptiX scene build received a null buffer");
        }
        if (vertex_count == 0 || triangle_count == 0 ||
            vertex_count > UINT32_MAX || triangle_count > UINT32_MAX ||
            scene_signature == 0 || (rebuild != 0 && rebuild != 1)) {
            throw BackendError(HS_STATUS_RANGE_ERROR, "OptiX scene dimensions or scalars are invalid");
        }
        check_cuda(cudaSetDevice(handle->device), "cudaSetDevice");
        std::lock_guard<std::mutex> lock(handle->mutex);
        NativeOptixState* state = initialize_state(handle);
        if (!rebuild && state->gas_handle != 0 &&
            handle->optix_scene_signature == scene_signature) {
            check_cuda(cudaMemsetAsync(output_status, 0, sizeof(int32_t), nullptr), "write scene status");
            return;
        }

        OptixAccelBuildOptions options = {};
        options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
            OPTIX_BUILD_FLAG_ALLOW_UPDATE |
            OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
        options.operation = OPTIX_BUILD_OPERATION_BUILD;
        CUdeviceptr vertex_pointer = reinterpret_cast<CUdeviceptr>(vertices3);
        const uint32_t flags[] = {OPTIX_GEOMETRY_FLAG_NONE};
        OptixBuildInput input = {};
        input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        input.triangleArray.numVertices = static_cast<uint32_t>(vertex_count);
        input.triangleArray.vertexBuffers = &vertex_pointer;
        input.triangleArray.flags = flags;
        input.triangleArray.numSbtRecords = 1;
        input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        input.triangleArray.numIndexTriplets = static_cast<uint32_t>(triangle_count);
        input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(triangles3);
        OptixAccelBufferSizes sizes = {};
        check_optix(
            optixAccelComputeMemoryUsage(state->context, &options, &input, 1, &sizes),
            "optixAccelComputeMemoryUsage");
        reserve_buffer(
            &handle->optix_gas_scratch,
            &handle->optix_gas_scratch_bytes,
            sizes.tempSizeInBytes,
            handle,
            state);
        reserve_buffer(
            &state->gas_output,
            &state->gas_output_bytes,
            sizes.outputSizeInBytes,
            handle,
            state);
        check_optix(
            optixAccelBuild(
                state->context,
                nullptr,
                &options,
                &input,
                1,
                reinterpret_cast<CUdeviceptr>(handle->optix_gas_scratch),
                sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>(state->gas_output),
                sizes.outputSizeInBytes,
                &state->gas_handle,
                nullptr,
                0),
            "optixAccelBuild");
        handle->optix_scene_signature = scene_signature;
        ++handle->optix_gas_build_count;
        update_peak(handle, state);
        check_cuda(cudaMemsetAsync(output_status, 0, sizeof(int32_t), nullptr), "write scene status");
    });
}

int32_t hs_optix_trace_f32_v1(
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
    uint64_t error_capacity) noexcept {
    return guarded(error_text, error_capacity, [&]() {
        if (handle == nullptr) {
            throw BackendError(HS_STATUS_INVALID_ARGUMENT, "OptiX trace handle is null");
        }
        if (ray_count > UINT32_MAX || coefficient_count == 0 ||
            coefficient_count > 16 || degree > 3 ||
            coefficient_count < (degree + 1) * (degree + 1) ||
            (start_from_first != 0 && start_from_first != 1) ||
            (use_others != 0 && use_others != 1) ||
            max_trace_depth < 0 || max_trace_depth > 1 ||
            !std::isfinite(scale_modifier) || scale_modifier <= 0.0f ||
            !std::isfinite(specular_threshold)) {
            throw BackendError(HS_STATUS_RANGE_ERROR, "OptiX trace dimensions or scalars are invalid");
        }
        if (ray_count != 0 &&
            (ray_origins3 == nullptr || ray_directions3 == nullptr ||
             background3 == nullptr || means3d == nullptr || shs_m3 == nullptr ||
             others2 == nullptr || opacities == nullptr || scales2 == nullptr ||
             rotations4 == nullptr || viewmatrix4x4 == nullptr ||
             projmatrix4x4 == nullptr || camera_position3 == nullptr ||
             output_rgb3 == nullptr || output_depth1 == nullptr ||
             output_acc1 == nullptr || output_normal3 == nullptr ||
             output_dist1 == nullptr || output_aux2 == nullptr ||
             output_mid32 == nullptr || output_wet1 == nullptr)) {
            throw BackendError(HS_STATUS_INVALID_ARGUMENT, "OptiX trace received a null non-empty buffer");
        }
        check_cuda(cudaSetDevice(handle->device), "cudaSetDevice");
        std::lock_guard<std::mutex> lock(handle->mutex);
        NativeOptixState* state = static_cast<NativeOptixState*>(handle->optix_state);
        if (state == nullptr || state->gas_handle == 0) {
            throw BackendError(HS_STATUS_INVALID_ARGUMENT, "OptiX scene is not built");
        }
        if (ray_count == 0) return;

        reserve_buffer(
            &handle->optix_params,
            &handle->optix_params_bytes,
            sizeof(Params),
            handle,
            state);
        reserve_buffer(
            &handle->optix_dist3,
            &handle->optix_dist3_bytes,
            ray_count * 3 * sizeof(float),
            handle,
            state);
        update_peak(handle, state);

        check_cuda(cudaMemsetAsync(output_rgb3, 0, ray_count * 3 * sizeof(float), nullptr), "zero OptiX rgb");
        check_cuda(cudaMemsetAsync(output_depth1, 0, ray_count * sizeof(float), nullptr), "zero OptiX depth");
        check_cuda(cudaMemsetAsync(output_acc1, 0, ray_count * sizeof(float), nullptr), "zero OptiX alpha");
        check_cuda(cudaMemsetAsync(output_normal3, 0, ray_count * 3 * sizeof(float), nullptr), "zero OptiX normal");
        check_cuda(cudaMemsetAsync(output_dist1, 0, ray_count * sizeof(float), nullptr), "zero OptiX distance");
        check_cuda(cudaMemsetAsync(output_aux2, 0, ray_count * 2 * sizeof(float), nullptr), "zero OptiX auxiliary");
        check_cuda(cudaMemsetAsync(output_mid32, 0, ray_count * 32 * sizeof(float), nullptr), "zero OptiX middle values");
        check_cuda(cudaMemsetAsync(output_wet1, 0, ray_count * sizeof(float), nullptr), "zero OptiX wet values");
        check_cuda(cudaMemsetAsync(handle->optix_dist3, 0, ray_count * 3 * sizeof(float), nullptr), "zero OptiX dist3 workspace");

        Params params = {};
        params.handle = state->gas_handle;
        params.training = false;
        params.max_trace_depth = max_trace_depth;
        params.start_from_first = start_from_first != 0;
        params.specular_threshold = specular_threshold;
        params.P = static_cast<int>(ray_count);
        params.H = static_cast<int>(ray_count);
        params.W = 3;
        params.D = static_cast<int>(degree);
        params.M = static_cast<int>(coefficient_count);
        params.ray_o = reinterpret_cast<float3*>(const_cast<float*>(ray_origins3));
        params.ray_d = reinterpret_cast<float3*>(const_cast<float*>(ray_directions3));
        params.vertices = nullptr;
        params.background = const_cast<float*>(background3);
        params.means3D = reinterpret_cast<float3*>(const_cast<float*>(means3d));
        params.shs = reinterpret_cast<float3*>(const_cast<float*>(shs_m3));
        params.colors_precomp = nullptr;
        params.others_precomp = use_others ? const_cast<float*>(others2) : nullptr;
        params.opacities = const_cast<float*>(opacities);
        params.scales = reinterpret_cast<float2*>(const_cast<float*>(scales2));
        params.scale_modifier = scale_modifier;
        params.rotations = reinterpret_cast<float4*>(const_cast<float*>(rotations4));
        params.transMat_precomp = nullptr;
        params.viewmatrix = const_cast<float*>(viewmatrix4x4);
        params.projmatrix = const_cast<float*>(projmatrix4x4);
        params.campos = reinterpret_cast<float3*>(const_cast<float*>(camera_position3));
        params.out_rgb = output_rgb3;
        params.out_dpt = output_depth1;
        params.out_acc = output_acc1;
        params.out_norm = reinterpret_cast<float3*>(output_normal3);
        params.out_dist = reinterpret_cast<float3*>(handle->optix_dist3);
        params.out_aux = output_aux2;
        params.mid_val = output_mid32;
        params.a_weights = nullptr;
        check_cuda(
            cudaMemcpyAsync(
                handle->optix_params,
                &params,
                sizeof(params),
                cudaMemcpyHostToDevice,
                nullptr),
            "upload OptiX launch params");
        check_optix(
            optixLaunch(
                state->pipeline,
                nullptr,
                reinterpret_cast<CUdeviceptr>(handle->optix_params),
                sizeof(params),
                &state->sbt,
                static_cast<uint32_t>(ray_count),
                1,
                1),
            "optixLaunch");
        check_cuda(
            cudaMemcpy2DAsync(
                output_dist1,
                sizeof(float),
                handle->optix_dist3,
                3 * sizeof(float),
                sizeof(float),
                ray_count,
                cudaMemcpyDeviceToDevice,
                nullptr),
            "extract OptiX distance channel");
        check_cuda(cudaGetLastError(), "OptiX trace launch");
    });
}

int32_t hs_optix_workspace_stats_v1(
    HsRasterizerHandle handle,
    HsOptixWorkspaceStatsV1* stats,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guarded(error_text, error_capacity, [&]() {
        if (handle == nullptr || stats == nullptr) {
            throw BackendError(HS_STATUS_INVALID_ARGUMENT, "OptiX stats received a null pointer");
        }
        if (stats->struct_size != sizeof(HsOptixWorkspaceStatsV1) ||
            stats->abi_version != HS_RASTERIZER_ABI_VERSION) {
            throw BackendError(HS_STATUS_ABI_MISMATCH, "OptiX stats ABI mismatch");
        }
        std::lock_guard<std::mutex> lock(handle->mutex);
        NativeOptixState* state = static_cast<NativeOptixState*>(handle->optix_state);
        stats->device = handle->device;
        stats->scene_ready = state && state->gas_handle != 0;
        stats->params_bytes = handle->optix_params_bytes;
        stats->gas_output_bytes = state ? state->gas_output_bytes : 0;
        stats->gas_scratch_bytes = handle->optix_gas_scratch_bytes;
        stats->dist3_bytes = handle->optix_dist3_bytes;
        stats->current_bytes = stats->params_bytes + stats->gas_output_bytes +
            stats->gas_scratch_bytes + stats->dist3_bytes;
        stats->peak_bytes = handle->optix_peak_bytes;
        stats->growth_count = handle->optix_growth_count;
        stats->scene_signature = handle->optix_scene_signature;
        stats->gas_build_count = handle->optix_gas_build_count;
        stats->pipeline_create_count = handle->optix_pipeline_create_count;
    });
}

int32_t hs_optix_release_scene_v1(
    HsRasterizerHandle handle,
    char* error_text,
    uint64_t error_capacity) noexcept {
    return guarded(error_text, error_capacity, [&]() {
        if (handle == nullptr) {
            throw BackendError(HS_STATUS_INVALID_ARGUMENT, "OptiX release handle is null");
        }
        check_cuda(cudaSetDevice(handle->device), "cudaSetDevice");
        std::lock_guard<std::mutex> lock(handle->mutex);
        release_scene_locked(handle, false);
    });
}
