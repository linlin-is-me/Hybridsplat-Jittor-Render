#include "full_rasterizer.h"

#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "forward.h"
#include "auxiliary.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace cg = cooperative_groups;

namespace {

template <typename T>
void obtain(char*& chunk, T*& pointer, size_t count, size_t alignment) {
    const size_t offset =
        (reinterpret_cast<uintptr_t>(chunk) + alignment - 1) &
        ~(alignment - 1);
    pointer = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(pointer + count);
}

struct GeometryState {
    size_t scan_size = 0;
    float* depths = nullptr;
    char* scanning_space = nullptr;
    bool* clamped = nullptr;
    int32_t* internal_radii = nullptr;
    float2* means2d = nullptr;
    float* transform = nullptr;
    float4* normal_opacity = nullptr;
    float* rgb = nullptr;
    uint32_t* point_offsets = nullptr;
    uint32_t* tiles_touched = nullptr;

    static GeometryState from_chunk(char*& chunk, size_t count) {
        GeometryState state;
        obtain(chunk, state.depths, count, 128);
        obtain(chunk, state.clamped, count * 3, 128);
        obtain(chunk, state.internal_radii, count, 128);
        obtain(chunk, state.means2d, count, 128);
        obtain(chunk, state.transform, count * 9, 128);
        obtain(chunk, state.normal_opacity, count, 128);
        obtain(chunk, state.rgb, count * 3, 128);
        obtain(chunk, state.tiles_touched, count, 128);
        cub::DeviceScan::InclusiveSum(
            nullptr,
            state.scan_size,
            state.tiles_touched,
            state.tiles_touched,
            count);
        obtain(chunk, state.scanning_space, state.scan_size, 128);
        obtain(chunk, state.point_offsets, count, 128);
        return state;
    }
};

struct ImageState {
    uint2* ranges = nullptr;
    uint32_t* contributors = nullptr;
    float* accumulated_alpha = nullptr;

    static ImageState from_chunk(char*& chunk, size_t count) {
        ImageState state;
        obtain(chunk, state.accumulated_alpha, count * 3, 128);
        obtain(chunk, state.contributors, count * 2, 128);
        obtain(chunk, state.ranges, count, 128);
        return state;
    }
};

struct BinningState {
    size_t sorting_size = 0;
    uint64_t* keys_unsorted = nullptr;
    uint64_t* keys = nullptr;
    uint32_t* points_unsorted = nullptr;
    uint32_t* points = nullptr;
    char* sorting_space = nullptr;

    static BinningState from_chunk(char*& chunk, size_t count) {
        BinningState state;
        obtain(chunk, state.points, count, 128);
        obtain(chunk, state.points_unsorted, count, 128);
        obtain(chunk, state.keys, count, 128);
        obtain(chunk, state.keys_unsorted, count, 128);
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            state.sorting_size,
            state.keys_unsorted,
            state.keys,
            state.points_unsorted,
            state.points,
            count);
        obtain(chunk, state.sorting_space, state.sorting_size, 128);
        return state;
    }
};

template <typename State>
size_t required(size_t count) {
    char* end = nullptr;
    State::from_chunk(end, count);
    return reinterpret_cast<size_t>(end) + 128;
}

uint32_t higher_msb(uint32_t value) {
    uint32_t msb = sizeof(value) * 4;
    uint32_t step = msb;
    while (step > 1) {
        step /= 2;
        if (value >> msb) {
            msb += step;
        } else {
            msb -= step;
        }
    }
    if (value >> msb) {
        ++msb;
    }
    return msb;
}

__global__ void duplicate_with_keys(
    int32_t point_count,
    const float2* points_xy,
    const float* depths,
    const uint32_t* offsets,
    uint64_t* keys,
    uint32_t* values,
    const int32_t* radii,
    dim3 grid) {
    const uint32_t index = cg::this_grid().thread_rank();
    if (index >= static_cast<uint32_t>(point_count) || radii[index] <= 0) {
        return;
    }
    uint32_t offset = index == 0 ? 0 : offsets[index - 1];
    uint2 rect_min;
    uint2 rect_max;
    getRect(points_xy[index], radii[index], rect_min, rect_max, grid);
    for (uint32_t y = rect_min.y; y < rect_max.y; ++y) {
        for (uint32_t x = rect_min.x; x < rect_max.x; ++x) {
            uint64_t key = static_cast<uint64_t>(y * grid.x + x) << 32;
            key |= *reinterpret_cast<const uint32_t*>(&depths[index]);
            keys[offset] = key;
            values[offset] = index;
            ++offset;
        }
    }
}

__global__ void identify_tile_ranges(
    int32_t count,
    const uint64_t* keys,
    uint2* ranges) {
    const uint32_t index = cg::this_grid().thread_rank();
    if (index >= static_cast<uint32_t>(count)) {
        return;
    }
    const uint32_t current = static_cast<uint32_t>(keys[index] >> 32);
    if (index == 0) {
        ranges[current].x = 0;
    } else {
        const uint32_t previous = static_cast<uint32_t>(keys[index - 1] >> 32);
        if (current != previous) {
            ranges[previous].y = index;
            ranges[current].x = index;
        }
    }
    if (index == static_cast<uint32_t>(count - 1)) {
        ranges[current].y = count;
    }
}

cudaError_t zero_outputs(
    int32_t point_count,
    int32_t width,
    int32_t height,
    float* color,
    float* aux,
    float* weight,
    int32_t* radii,
    int32_t* rendered) {
    const size_t pixels = static_cast<size_t>(width) * height;
    cudaError_t error = cudaMemset(color, 0, 5 * pixels * sizeof(float));
    if (error != cudaSuccess) return error;
    error = cudaMemset(aux, 0, 7 * pixels * sizeof(float));
    if (error != cudaSuccess) return error;
    if (point_count > 0) {
        error = cudaMemset(weight, 0, point_count * sizeof(float));
        if (error != cudaSuccess) return error;
        error = cudaMemset(radii, 0, point_count * sizeof(int32_t));
        if (error != cudaSuccess) return error;
    }
    return cudaMemset(rendered, 0, sizeof(int32_t));
}

}  // namespace

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
    int32_t* output_rendered) noexcept {
    cudaError_t error = zero_outputs(
        point_count,
        width,
        height,
        output_color5,
        output_aux7,
        output_weight,
        output_radii,
        output_rendered);
    if (error != cudaSuccess || point_count == 0) {
        return error;
    }

    const dim3 tile_grid(
        (width + BLOCK_X - 1) / BLOCK_X,
        (height + BLOCK_Y - 1) / BLOCK_Y,
        1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    char* geometry_buffer = nullptr;
    char* image_buffer = nullptr;
    error = hs_context_reserve_geometry(
        context,
        required<GeometryState>(point_count),
        &geometry_buffer);
    if (error != cudaSuccess) return error;
    error = hs_context_reserve_image(
        context,
        required<ImageState>(static_cast<size_t>(width) * height),
        &image_buffer);
    if (error != cudaSuccess) return error;
    char* geometry_cursor = geometry_buffer;
    char* image_cursor = image_buffer;
    GeometryState geometry = GeometryState::from_chunk(
        geometry_cursor,
        point_count);
    ImageState image = ImageState::from_chunk(
        image_cursor,
        static_cast<size_t>(width) * height);

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);
    FORWARD::preprocess(
        point_count,
        0,
        0,
        means3d,
        reinterpret_cast<const glm::vec2*>(scales2),
        scale_modifier,
        reinterpret_cast<const glm::vec4*>(rotations4),
        opacities,
        nullptr,
        geometry.clamped,
        nullptr,
        colors5,
        viewmatrix4x4,
        projmatrix4x4,
        reinterpret_cast<const glm::vec3*>(camera_position3),
        width,
        height,
        focal_x,
        focal_y,
        tan_fovx,
        tan_fovy,
        output_radii,
        geometry.means2d,
        geometry.depths,
        geometry.transform,
        geometry.rgb,
        geometry.normal_opacity,
        tile_grid,
        geometry.tiles_touched,
        prefiltered);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;

    error = cub::DeviceScan::InclusiveSum(
        geometry.scanning_space,
        geometry.scan_size,
        geometry.tiles_touched,
        geometry.point_offsets,
        point_count);
    if (error != cudaSuccess) return error;
    int32_t rendered = 0;
    error = cudaMemcpy(
        &rendered,
        geometry.point_offsets + point_count - 1,
        sizeof(rendered),
        cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return error;
    error = cudaMemcpy(
        output_rendered,
        &rendered,
        sizeof(rendered),
        cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return error;

    char* binning_buffer = nullptr;
    error = hs_context_reserve_binning(
        context,
        required<BinningState>(rendered),
        &binning_buffer);
    if (error != cudaSuccess) return error;
    char* binning_cursor = binning_buffer;
    BinningState binning = BinningState::from_chunk(
        binning_cursor,
        rendered);

    if (rendered > 0) {
        duplicate_with_keys<<<(point_count + 255) / 256, 256>>>(
            point_count,
            geometry.means2d,
            geometry.depths,
            geometry.point_offsets,
            binning.keys_unsorted,
            binning.points_unsorted,
            output_radii,
            tile_grid);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
        const uint32_t bit = higher_msb(tile_grid.x * tile_grid.y);
        error = cub::DeviceRadixSort::SortPairs(
            binning.sorting_space,
            binning.sorting_size,
            binning.keys_unsorted,
            binning.keys,
            binning.points_unsorted,
            binning.points,
            rendered,
            0,
            32 + bit);
        if (error != cudaSuccess) return error;
    }

    error = cudaMemset(
        image.ranges,
        0,
        static_cast<size_t>(tile_grid.x) * tile_grid.y * sizeof(uint2));
    if (error != cudaSuccess) return error;
    if (rendered > 0) {
        identify_tile_ranges<<<(rendered + 255) / 256, 256>>>(
            rendered,
            binning.keys,
            image.ranges);
        error = cudaGetLastError();
        if (error != cudaSuccess) return error;
    }

    FORWARD::render(
        tile_grid,
        block,
        image.ranges,
        binning.points,
        width,
        height,
        focal_x,
        focal_y,
        geometry.means2d,
        colors5,
        geometry.transform,
        geometry.depths,
        geometry.normal_opacity,
        image.accumulated_alpha,
        image.contributors,
        background5,
        output_color5,
        output_aux7,
        output_weight);
    return cudaGetLastError();
}
