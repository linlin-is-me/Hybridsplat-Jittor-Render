#include "flashgs_backend.h"

#include "context_internal.h"
#include "ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

struct FlashGsProfileEvents {
    cudaEvent_t events[6]{};
    bool enabled = false;

    explicit FlashGsProfileEvents(bool requested) : enabled(requested) {
        if (!enabled) return;
        for (cudaEvent_t& event : events) {
            if (cudaEventCreate(&event) != cudaSuccess) {
                enabled = false;
                break;
            }
        }
    }

    ~FlashGsProfileEvents() {
        for (cudaEvent_t event : events) {
            if (event != nullptr) cudaEventDestroy(event);
        }
    }

    void record(int index) const {
        if (enabled) cudaEventRecord(events[index], nullptr);
    }

    float elapsed(int begin, int end) const {
        float milliseconds = 0.0f;
        if (enabled) cudaEventElapsedTime(&milliseconds, events[begin], events[end]);
        return milliseconds;
    }
};

constexpr uint64_t kDeviceAlignment = 2ull * 1024ull * 1024ull;
constexpr uint64_t kFieldAlignment = 256ull;

uint64_t align_up(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

cudaError_t replace_device_buffer(
    void** buffer,
    uint64_t* allocated_bytes,
    uint64_t required_bytes,
    uint64_t* growth_count) {
    const uint64_t target = align_up(std::max<uint64_t>(required_bytes, 1), kDeviceAlignment);
    if (*buffer != nullptr && *allocated_bytes >= target) {
        return cudaSuccess;
    }
    void* replacement = nullptr;
    cudaError_t status = cudaMalloc(&replacement, target);
    if (status != cudaSuccess) {
        return status;
    }
    if (*buffer != nullptr) {
        status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            cudaFree(replacement);
            return status;
        }
        status = cudaFree(*buffer);
        if (status != cudaSuccess) {
            cudaFree(replacement);
            return status;
        }
    }
    *buffer = replacement;
    *allocated_bytes = target;
    ++*growth_count;
    return cudaSuccess;
}

uint64_t gaussian_layout_bytes(uint64_t count) {
    uint64_t offset = 0;
    for (uint64_t bytes_per_point : {8ull, 12ull, 8ull, 16ull, 16ull, 12ull, 12ull, 12ull}) {
        offset = align_up(offset, kFieldAlignment);
        offset += count * bytes_per_point;
    }
    return offset;
}

struct GaussianPointers {
    float2* points_xy;
    float3* rgb;
    float2* other;
    float4* conic_opacity;
    float4* normal_opacity;
    float3* tu;
    float3* tv;
    float3* tw;
};

GaussianPointers gaussian_pointers(void* workspace, uint64_t count) {
    auto* base = static_cast<unsigned char*>(workspace);
    uint64_t offset = 0;
    auto take = [&](uint64_t bytes) {
        offset = align_up(offset, kFieldAlignment);
        unsigned char* result = base + offset;
        offset += bytes;
        return result;
    };
    return {
        reinterpret_cast<float2*>(take(count * 8ull)),
        reinterpret_cast<float3*>(take(count * 12ull)),
        reinterpret_cast<float2*>(take(count * 8ull)),
        reinterpret_cast<float4*>(take(count * 16ull)),
        reinterpret_cast<float4*>(take(count * 16ull)),
        reinterpret_cast<float3*>(take(count * 12ull)),
        reinterpret_cast<float3*>(take(count * 12ull)),
        reinterpret_cast<float3*>(take(count * 12ull)),
    };
}

uint64_t pair_layout_bytes(uint64_t capacity) {
    uint64_t offset = 0;
    for (uint64_t bytes_per_pair : {8ull, 4ull, 8ull, 4ull, 8ull, 4ull}) {
        offset = align_up(offset, kFieldAlignment);
        offset += capacity * bytes_per_pair;
    }
    return offset;
}

struct PairPointers {
    uint64_t* keys_unsorted;
    uint32_t* values_unsorted;
    uint64_t* keys_secondary;
    uint32_t* values_secondary;
    uint64_t* keys_sorted;
    uint32_t* values_sorted;
};

PairPointers pair_pointers(void* workspace, uint64_t capacity) {
    auto* base = static_cast<unsigned char*>(workspace);
    uint64_t offset = 0;
    auto take = [&](uint64_t bytes) {
        offset = align_up(offset, kFieldAlignment);
        unsigned char* result = base + offset;
        offset += bytes;
        return result;
    };
    return {
        reinterpret_cast<uint64_t*>(take(capacity * 8ull)),
        reinterpret_cast<uint32_t*>(take(capacity * 4ull)),
        reinterpret_cast<uint64_t*>(take(capacity * 8ull)),
        reinterpret_cast<uint32_t*>(take(capacity * 4ull)),
        reinterpret_cast<uint64_t*>(take(capacity * 8ull)),
        reinterpret_cast<uint32_t*>(take(capacity * 4ull)),
    };
}

uint64_t output_layout_bytes(uint64_t pixels) {
    uint64_t offset = 0;
    for (uint64_t bytes_per_pixel : {12ull, 8ull, 12ull}) {
        offset = align_up(offset, kFieldAlignment);
        offset += pixels * bytes_per_pixel;
    }
    return offset;
}

struct OutputPointers {
    float3* color;
    float2* other;
    float3* normal;
};

OutputPointers output_pointers(void* workspace, uint64_t pixels) {
    auto* base = static_cast<unsigned char*>(workspace);
    uint64_t offset = 0;
    auto take = [&](uint64_t bytes) {
        offset = align_up(offset, kFieldAlignment);
        unsigned char* result = base + offset;
        offset += bytes;
        return result;
    };
    return {
        reinterpret_cast<float3*>(take(pixels * 12ull)),
        reinterpret_cast<float2*>(take(pixels * 8ull)),
        reinterpret_cast<float3*>(take(pixels * 12ull)),
    };
}

uint64_t current_bytes(const HsRasterizerContext* context) {
    return context->flashgs_gaussian_bytes + context->flashgs_pair_bytes +
           context->flashgs_cub_bytes + context->flashgs_ranges_bytes +
           context->flashgs_output_bytes +
           (context->flashgs_counter_device == nullptr ? 0ull : sizeof(int32_t));
}

void update_peak(HsRasterizerContext* context) {
    context->flashgs_peak_bytes = std::max(context->flashgs_peak_bytes, current_bytes(context));
}

cudaError_t ensure_counters(HsRasterizerContext* context) {
    cudaError_t status = cudaSuccess;
    if (context->flashgs_counter_device == nullptr) {
        status = cudaMalloc(
            reinterpret_cast<void**>(&context->flashgs_counter_device),
            sizeof(int32_t));
        if (status != cudaSuccess) return status;
        ++context->flashgs_growth_count;
    }
    if (context->flashgs_counter_host == nullptr) {
        status = cudaHostAlloc(
            reinterpret_cast<void**>(&context->flashgs_counter_host),
            sizeof(int32_t),
            cudaHostAllocPortable);
        if (status != cudaSuccess) return status;
        ++context->flashgs_growth_count;
    }
    update_peak(context);
    return cudaSuccess;
}

cudaError_t ensure_gaussians(HsRasterizerContext* context, uint64_t count) {
    if (count <= context->flashgs_gaussian_capacity &&
        context->flashgs_gaussian_workspace != nullptr) {
        return cudaSuccess;
    }
    const cudaError_t status = replace_device_buffer(
        &context->flashgs_gaussian_workspace,
        &context->flashgs_gaussian_bytes,
        gaussian_layout_bytes(count),
        &context->flashgs_growth_count);
    if (status == cudaSuccess) {
        context->flashgs_gaussian_capacity = count;
        update_peak(context);
    }
    return status;
}

cudaError_t ensure_pairs(HsRasterizerContext* context, uint64_t capacity) {
    if (capacity == 0 || capacity > static_cast<uint64_t>(INT_MAX)) {
        return cudaErrorInvalidValue;
    }
    if (capacity > context->flashgs_pair_capacity ||
        context->flashgs_pair_workspace == nullptr) {
        cudaError_t status = replace_device_buffer(
            &context->flashgs_pair_workspace,
            &context->flashgs_pair_bytes,
            pair_layout_bytes(capacity),
            &context->flashgs_growth_count);
        if (status != cudaSuccess) return status;
        context->flashgs_pair_capacity = capacity;
    }
    const size_t cub_required = flashgs::get_sort_buffer_size(static_cast<int>(capacity));
    cudaError_t status = replace_device_buffer(
        &context->flashgs_cub_workspace,
        &context->flashgs_cub_bytes,
        static_cast<uint64_t>(cub_required),
        &context->flashgs_growth_count);
    if (status == cudaSuccess) update_peak(context);
    return status;
}

cudaError_t ensure_ranges(HsRasterizerContext* context, uint64_t tile_count) {
    if (tile_count <= context->flashgs_range_capacity &&
        context->flashgs_ranges_workspace != nullptr) {
        return cudaSuccess;
    }
    const cudaError_t status = replace_device_buffer(
        &context->flashgs_ranges_workspace,
        &context->flashgs_ranges_bytes,
        tile_count * sizeof(int2),
        &context->flashgs_growth_count);
    if (status == cudaSuccess) {
        context->flashgs_range_capacity = tile_count;
        update_peak(context);
    }
    return status;
}

cudaError_t ensure_outputs(HsRasterizerContext* context, uint64_t pixels) {
    if (pixels <= context->flashgs_output_pixel_capacity &&
        context->flashgs_output_workspace != nullptr) {
        return cudaSuccess;
    }
    const cudaError_t status = replace_device_buffer(
        &context->flashgs_output_workspace,
        &context->flashgs_output_bytes,
        output_layout_bytes(pixels),
        &context->flashgs_growth_count);
    if (status == cudaSuccess) {
        context->flashgs_output_pixel_capacity = pixels;
        update_peak(context);
    }
    return status;
}

__global__ void fill_empty_color(float3* color, uint64_t pixels, float3 background) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    const auto encode = [](float value) {
        return static_cast<float>(static_cast<unsigned char>(
            fminf(fmaxf(0.0f, value * 255.0f), 255.0f)));
    };
    color[index] = make_float3(encode(background.x), encode(background.y), encode(background.z));
}

__global__ void pack_channels(
    const float3* color_hwc,
    const float2* other_hwc,
    const float3* normal_hwc,
    uint64_t pixels,
    float* color_chw,
    float* other_chw,
    float* normal_chw) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= pixels) return;
    const float3 color = color_hwc[index];
    const float2 other = other_hwc[index];
    const float3 normal = normal_hwc[index];
    color_chw[index] = color.x;
    color_chw[pixels + index] = color.y;
    color_chw[2 * pixels + index] = color.z;
    other_chw[index] = other.x;
    other_chw[pixels + index] = other.y;
    normal_chw[index] = normal.x;
    normal_chw[pixels + index] = normal.y;
    normal_chw[2 * pixels + index] = normal.z;
}

}  // namespace

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
    int32_t* output_num_rendered) noexcept(false) {
    const uint64_t pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
    const uint64_t tile_count =
        static_cast<uint64_t>((width + 15) / 16) * static_cast<uint64_t>((height + 15) / 16);
    cudaError_t status = cudaMemsetAsync(output_color3, 0, pixels * 3 * sizeof(float), nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(output_other2, 0, pixels * 2 * sizeof(float), nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(output_depth1, 0, pixels * sizeof(float), nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(output_alpha1, 0, pixels * sizeof(float), nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(output_normal3, 0, pixels * 3 * sizeof(float), nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(output_median_depth1, 0, pixels * sizeof(float), nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(output_distortion1, 0, pixels * sizeof(float), nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(output_num_rendered, 0, sizeof(int32_t), nullptr);
    if (status != cudaSuccess) return status;

    status = ensure_outputs(context, pixels);
    if (status != cudaSuccess) return status;
    status = ensure_ranges(context, tile_count);
    if (status != cudaSuccess) return status;
    OutputPointers output = output_pointers(
        context->flashgs_output_workspace,
        context->flashgs_output_pixel_capacity);
    status = cudaMemsetAsync(
        context->flashgs_output_workspace, 0, context->flashgs_output_bytes, nullptr);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(
        context->flashgs_ranges_workspace, 0, tile_count * sizeof(int2), nullptr);
    if (status != cudaSuccess) return status;

    const char* profile_environment = std::getenv("HS_FLASHGS_PROFILE");
    FlashGsProfileEvents profile(
        profile_environment != nullptr && profile_environment[0] == '1');
    double count_host_milliseconds = 0.0;
    int32_t num_rendered = 0;
    if (point_count != 0) {
        status = ensure_counters(context);
        if (status != cudaSuccess) return status;
        status = ensure_gaussians(context, static_cast<uint64_t>(point_count));
        if (status != cudaSuccess) return status;
        const uint64_t requested_capacity =
            context->flashgs_pair_capacity == 0 ? std::max<uint64_t>(initial_pair_capacity, 1)
                                                : context->flashgs_pair_capacity;
        status = ensure_pairs(context, requested_capacity);
        if (status != cudaSuccess) return status;

        GaussianPointers gaussian = gaussian_pointers(
            context->flashgs_gaussian_workspace,
            context->flashgs_gaussian_capacity);
        const glm::vec3 camera_position(
            camera_position3[0], camera_position3[1], camera_position3[2]);
        const glm::mat3 camera_rotation({
            {camera_rotation9[0], camera_rotation9[1], camera_rotation9[2]},
            {camera_rotation9[3], camera_rotation9[4], camera_rotation9[5]},
            {camera_rotation9[6], camera_rotation9[7], camera_rotation9[8]},
        });

        for (;;) {
            PairPointers pairs = pair_pointers(
                context->flashgs_pair_workspace,
                context->flashgs_pair_capacity);
            status = cudaMemsetAsync(
                context->flashgs_counter_device, 0, sizeof(int32_t), nullptr);
            if (status != cudaSuccess) return status;
            profile.record(0);
            flashgs::preprocess(
                point_count,
                reinterpret_cast<glm::vec3*>(const_cast<float*>(positions3)),
                reinterpret_cast<flashgs::shs_deg3_t*>(const_cast<float*>(shs48)),
                reinterpret_cast<const glm::vec3*>(colors3),
                use_sh,
                const_cast<float*>(opacities1),
                reinterpret_cast<flashgs::cov3d_t*>(const_cast<float*>(covariance6)),
                width,
                height,
                16,
                16,
                camera_position,
                camera_rotation,
                focal_x,
                focal_y,
                z_far,
                z_near,
                reinterpret_cast<glm::vec2*>(const_cast<float*>(scales2)),
                scale_modifier,
                reinterpret_cast<glm::vec4*>(const_cast<float*>(rotations4)),
                const_cast<float*>(specular1),
                const_cast<float*>(roughness1),
                gaussian.points_xy,
                gaussian.rgb,
                gaussian.other,
                gaussian.conic_opacity,
                gaussian.normal_opacity,
                gaussian.tu,
                gaussian.tv,
                gaussian.tw,
                pairs.keys_unsorted,
                pairs.values_unsorted,
                static_cast<int>(context->flashgs_pair_capacity),
                context->flashgs_counter_device,
                nullptr);
            status = cudaPeekAtLastError();
            if (status != cudaSuccess) return status;
            profile.record(1);
            const auto count_host_started = std::chrono::steady_clock::now();
            status = cudaMemcpyAsync(
                context->flashgs_counter_host,
                context->flashgs_counter_device,
                sizeof(int32_t),
                cudaMemcpyDeviceToHost,
                nullptr);
            if (status != cudaSuccess) return status;
            status = cudaStreamSynchronize(nullptr);
            if (status != cudaSuccess) return status;
            count_host_milliseconds += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - count_host_started).count();
            profile.record(2);
            num_rendered = *context->flashgs_counter_host;
            if (num_rendered < 0) return cudaErrorInvalidValue;
            if (static_cast<uint64_t>(num_rendered) < context->flashgs_pair_capacity) break;
            const uint64_t grown = std::max<uint64_t>(
                static_cast<uint64_t>(num_rendered) + 1,
                context->flashgs_pair_capacity + context->flashgs_pair_capacity / 4);
            status = ensure_pairs(context, grown);
            if (status != cudaSuccess) return status;
            ++context->flashgs_overflow_retry_count;
        }

        if (num_rendered != 0) {
            PairPointers pairs = pair_pointers(
                context->flashgs_pair_workspace,
                context->flashgs_pair_capacity);
            flashgs::sort_gaussian(
                num_rendered,
                width,
                height,
                16,
                16,
                static_cast<char*>(context->flashgs_cub_workspace),
                static_cast<size_t>(context->flashgs_cub_bytes),
                pairs.keys_unsorted,
                pairs.values_unsorted,
                pairs.keys_secondary,
                pairs.values_secondary,
                pairs.keys_sorted,
                pairs.values_sorted,
                nullptr);
            status = cudaPeekAtLastError();
            if (status != cudaSuccess) return status;
            profile.record(3);
            flashgs::render_16x16(
                num_rendered,
                width,
                height,
                gaussian.points_xy,
                gaussian.rgb,
                gaussian.other,
                gaussian.conic_opacity,
                gaussian.normal_opacity,
                gaussian.tu,
                gaussian.tv,
                gaussian.tw,
                pairs.keys_sorted,
                pairs.values_sorted,
                static_cast<int2*>(context->flashgs_ranges_workspace),
                make_float3(background3[0], background3[1], background3[2]),
                output.color,
                output.other,
                output_depth1,
                output_alpha1,
                output.normal,
                output_median_depth1,
                output_distortion1,
                nullptr);
            status = cudaPeekAtLastError();
            if (status != cudaSuccess) return status;
            profile.record(4);
            status = cudaMemcpyAsync(
                output_num_rendered,
                context->flashgs_counter_device,
                sizeof(int32_t),
                cudaMemcpyDeviceToDevice,
                nullptr);
            if (status != cudaSuccess) return status;
        }
    }

    if (num_rendered == 0) {
        const int blocks = static_cast<int>((pixels + 255) / 256);
        fill_empty_color<<<blocks, 256, 0, nullptr>>>(
            output.color,
            pixels,
            make_float3(background3[0], background3[1], background3[2]));
        status = cudaPeekAtLastError();
        if (status != cudaSuccess) return status;
    }
    const int blocks = static_cast<int>((pixels + 255) / 256);
    pack_channels<<<blocks, 256, 0, nullptr>>>(
        output.color,
        output.other,
        output.normal,
        pixels,
        output_color3,
        output_other2,
        output_normal3);
    status = cudaPeekAtLastError();
    if (status != cudaSuccess) return status;
    profile.record(5);
    if (profile.enabled && point_count != 0 && num_rendered != 0) {
        status = cudaEventSynchronize(profile.events[5]);
        if (status != cudaSuccess) return status;
        std::fprintf(
            stderr,
            "HS_FLASHGS_PROFILE point_count=%d pairs=%d preprocess_ms=%.9f "
            "count_device_ms=%.9f count_host_ms=%.9f sort_ms=%.9f "
            "render_ms=%.9f pack_ms=%.9f\n",
            point_count,
            num_rendered,
            profile.elapsed(0, 1),
            profile.elapsed(1, 2),
            count_host_milliseconds,
            profile.elapsed(2, 3),
            profile.elapsed(3, 4),
            profile.elapsed(4, 5));
    }

    ++context->flashgs_forward_calls;
    context->flashgs_last_point_count = static_cast<uint64_t>(point_count);
    context->flashgs_last_num_rendered = static_cast<uint64_t>(num_rendered);
    context->flashgs_last_tile_count = tile_count;
    update_peak(context);
    return cudaSuccess;
}

cudaError_t hs_flashgs_release_context(HsRasterizerContext* context) noexcept {
    if (context == nullptr) return cudaErrorInvalidValue;
    cudaError_t first = cudaSuccess;
    for (void** buffer : {
             &context->flashgs_gaussian_workspace,
             &context->flashgs_pair_workspace,
             &context->flashgs_cub_workspace,
             &context->flashgs_ranges_workspace,
             &context->flashgs_output_workspace}) {
        if (*buffer == nullptr) continue;
        const cudaError_t status = cudaFree(*buffer);
        if (first == cudaSuccess && status != cudaSuccess) first = status;
        *buffer = nullptr;
    }
    if (context->flashgs_counter_device != nullptr) {
        const cudaError_t status = cudaFree(context->flashgs_counter_device);
        if (first == cudaSuccess && status != cudaSuccess) first = status;
        context->flashgs_counter_device = nullptr;
    }
    if (context->flashgs_counter_host != nullptr) {
        const cudaError_t status = cudaFreeHost(context->flashgs_counter_host);
        if (first == cudaSuccess && status != cudaSuccess) first = status;
        context->flashgs_counter_host = nullptr;
    }
    context->flashgs_gaussian_bytes = 0;
    context->flashgs_pair_bytes = 0;
    context->flashgs_cub_bytes = 0;
    context->flashgs_ranges_bytes = 0;
    context->flashgs_output_bytes = 0;
    context->flashgs_gaussian_capacity = 0;
    context->flashgs_pair_capacity = 0;
    context->flashgs_range_capacity = 0;
    context->flashgs_output_pixel_capacity = 0;
    return first;
}

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
    int32_t* tile_ranges2) noexcept {
    if (context == nullptr || context->flashgs_gaussian_workspace == nullptr ||
        context->flashgs_pair_workspace == nullptr ||
        context->flashgs_ranges_workspace == nullptr) {
        return cudaErrorInvalidResourceHandle;
    }
    const uint64_t point_count = context->flashgs_last_point_count;
    const uint64_t pair_count = context->flashgs_last_num_rendered;
    const uint64_t tile_count = context->flashgs_last_tile_count;
    if (point_capacity < point_count || pair_capacity < pair_count ||
        tile_capacity < tile_count || points_xy2 == nullptr || rgb3 == nullptr ||
        other2 == nullptr || conic_opacity4 == nullptr ||
        normal_opacity4 == nullptr || tu3 == nullptr || tv3 == nullptr ||
        tw3 == nullptr || sorted_keys == nullptr || sorted_values == nullptr ||
        tile_ranges2 == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) return status;
    const GaussianPointers gaussian = gaussian_pointers(
        context->flashgs_gaussian_workspace,
        context->flashgs_gaussian_capacity);
    const PairPointers pairs = pair_pointers(
        context->flashgs_pair_workspace,
        context->flashgs_pair_capacity);
    struct Copy {
        void* destination;
        const void* source;
        uint64_t bytes;
    };
    const Copy copies[] = {
        {points_xy2, gaussian.points_xy, point_count * 2ull * sizeof(float)},
        {rgb3, gaussian.rgb, point_count * 3ull * sizeof(float)},
        {other2, gaussian.other, point_count * 2ull * sizeof(float)},
        {conic_opacity4, gaussian.conic_opacity, point_count * 4ull * sizeof(float)},
        {normal_opacity4, gaussian.normal_opacity, point_count * 4ull * sizeof(float)},
        {tu3, gaussian.tu, point_count * 3ull * sizeof(float)},
        {tv3, gaussian.tv, point_count * 3ull * sizeof(float)},
        {tw3, gaussian.tw, point_count * 3ull * sizeof(float)},
        {sorted_keys, pairs.keys_sorted, pair_count * sizeof(uint64_t)},
        {sorted_values, pairs.values_sorted, pair_count * sizeof(uint32_t)},
        {tile_ranges2, context->flashgs_ranges_workspace, tile_count * 2ull * sizeof(int32_t)},
    };
    for (const Copy& copy : copies) {
        if (copy.bytes == 0) continue;
        status = cudaMemcpy(copy.destination, copy.source, copy.bytes, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) return status;
    }
    return cudaSuccess;
}
