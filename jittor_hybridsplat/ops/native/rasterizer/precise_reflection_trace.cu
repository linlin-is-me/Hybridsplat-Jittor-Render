#include "precise_reflection_trace.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr float kNormalizeEpsilon = 1.0e-8f;
constexpr float kQuaternionEpsilon = 1.0e-12f;
constexpr float kNormalEpsilon = 1.0e-20f;
constexpr float kRayOriginOffset = 0.05f;

__device__ __forceinline__ float add_rn(float a, float b) {
    return __fadd_rn(a, b);
}

__device__ __forceinline__ float sub_rn(float a, float b) {
    return __fsub_rn(a, b);
}

__device__ __forceinline__ float mul_rn(float a, float b) {
    return __fmul_rn(a, b);
}

__device__ __forceinline__ float div_rn(float a, float b) {
    return __fdiv_rn(a, b);
}

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}

__device__ __forceinline__ float norm3_pytorch(float x, float y, float z) {
    const float xx = mul_rn(x, x);
    const float yy = mul_rn(y, y);
    const float zz = mul_rn(z, z);
    // PyTorch Reduce.cuh maps an inner reduction of length three to
    // block.x=2: lane 0 accumulates c0 and c2, then combines lane 1's c1.
    return __fsqrt_rn(add_rn(add_rn(xx, zz), yy));
}

__device__ __forceinline__ float norm4_pytorch(
    float c0,
    float c1,
    float c2,
    float c3) {
    const float s0 = mul_rn(c0, c0);
    const float s1 = mul_rn(c1, c1);
    const float s2 = mul_rn(c2, c2);
    const float s3 = mul_rn(c3, c3);
    // Reduce.cuh uses block.x=4 for length four and combines lanes as a tree.
    return __fsqrt_rn(add_rn(add_rn(s0, s1), add_rn(s2, s3)));
}

__device__ __forceinline__ float normalize3_add_epsilon(
    float x,
    float y,
    float z,
    float epsilon,
    float& ox,
    float& oy,
    float& oz) {
    const float denominator = add_rn(norm3_pytorch(x, y, z), epsilon);
    ox = div_rn(x, denominator);
    oy = div_rn(y, denominator);
    oz = div_rn(z, denominator);
    return denominator;
}

__device__ __forceinline__ float normalize3_clamp_epsilon(
    float x,
    float y,
    float z,
    float epsilon,
    float& ox,
    float& oy,
    float& oz) {
    const float norm = norm3_pytorch(x, y, z);
    const float denominator = norm < epsilon ? epsilon : norm;
    ox = div_rn(x, denominator);
    oy = div_rn(y, denominator);
    oz = div_rn(z, denominator);
    return denominator;
}

__device__ __forceinline__ void normalize4_clamp_epsilon(
    float c0,
    float c1,
    float c2,
    float c3,
    float epsilon,
    float& o0,
    float& o1,
    float& o2,
    float& o3) {
    const float norm = norm4_pytorch(c0, c1, c2, c3);
    const float denominator = norm < epsilon ? epsilon : norm;
    o0 = div_rn(c0, denominator);
    o1 = div_rn(c1, denominator);
    o2 = div_rn(c2, denominator);
    o3 = div_rn(c3, denominator);
}

__device__ __forceinline__ float dot3_pytorch(
    float ax,
    float ay,
    float az,
    float bx,
    float by,
    float bz) {
    const float c0 = mul_rn(ax, bx);
    const float c1 = mul_rn(ay, by);
    const float c2 = mul_rn(az, bz);
    return add_rn(add_rn(c0, c2), c1);
}

__global__ void precise_reflection_prefix_f32_kernel(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9) {
    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const uint64_t rotation_offset = index * 4;
    // GaussianModel.get_rotation first applies F.normalize to the checkpoint
    // quaternion. gaussian2d_utils.py build_rotation then normalizes it again.
    float active_w;
    float active_x;
    float active_y;
    float active_z;
    normalize4_clamp_epsilon(
        rotation_wxyz[rotation_offset],
        rotation_wxyz[rotation_offset + 1],
        rotation_wxyz[rotation_offset + 2],
        rotation_wxyz[rotation_offset + 3],
        kQuaternionEpsilon,
        active_w,
        active_x,
        active_y,
        active_z);
    const float rotation_norm =
        norm4_pytorch(active_w, active_x, active_y, active_z);
    const float w = div_rn(active_w, rotation_norm);
    const float x = div_rn(active_x, rotation_norm);
    const float y = div_rn(active_y, rotation_norm);
    const float z = div_rn(active_z, rotation_norm);

    // gaussian2d_utils.py:186-196 is executed by TorchScript's TensorExpr
    // fuser after the first profiling call. Its NVRTC kernel contracts the
    // leading product and the addition/subtraction in every matrix entry.
    // Keep those exact contractions explicit while --fmad=false protects the
    // normalization and all remaining stages from unintended contraction.
    const float wz = mul_rn(w, z);
    const float wy = mul_rn(w, y);
    const float wx = mul_rn(w, x);
    const float xx = mul_rn(x, x);
    const float yy = mul_rn(y, y);
    const float zz = mul_rn(z, z);
    const uint64_t output_offset = index * 9;
    rotation_matrix9[output_offset] =
        sub_rn(1.0f, mul_rn(2.0f, fma_rn(y, y, zz)));
    rotation_matrix9[output_offset + 1] =
        mul_rn(2.0f, fma_rn(x, y, sub_rn(0.0f, wz)));
    rotation_matrix9[output_offset + 2] =
        mul_rn(2.0f, fma_rn(x, z, wy));
    rotation_matrix9[output_offset + 3] =
        mul_rn(2.0f, fma_rn(x, y, wz));
    rotation_matrix9[output_offset + 4] =
        sub_rn(1.0f, mul_rn(2.0f, fma_rn(x, x, zz)));
    rotation_matrix9[output_offset + 5] =
        mul_rn(2.0f, fma_rn(y, z, sub_rn(0.0f, wx)));
    rotation_matrix9[output_offset + 6] =
        mul_rn(2.0f, fma_rn(x, z, sub_rn(0.0f, wy)));
    rotation_matrix9[output_offset + 7] =
        mul_rn(2.0f, fma_rn(y, z, wx));
    rotation_matrix9[output_offset + 8] =
        sub_rn(1.0f, mul_rn(2.0f, fma_rn(x, x, yy)));
}

__global__ void precise_reflection_suffix_f32_kernel(
    const float* xyz,
    const float* camera_center,
    const float* normal_raw3,
    uint64_t count,
    float* output36) {
    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const uint64_t xyz_offset = index * 3;
    const uint64_t output_offset = index * 36;
    const float px = xyz[xyz_offset];
    const float py = xyz[xyz_offset + 1];
    const float pz = xyz[xyz_offset + 2];

    // math_utils.py:5-8 and optix_utils.py:396-398.
    const float view_x_raw = sub_rn(camera_center[0], px);
    const float view_y_raw = sub_rn(camera_center[1], py);
    const float view_z_raw = sub_rn(camera_center[2], pz);
    float view_x;
    float view_y;
    float view_z;
    normalize3_add_epsilon(
        view_x_raw,
        view_y_raw,
        view_z_raw,
        kNormalizeEpsilon,
        view_x,
        view_y,
        view_z);

    const float input_normal_x = normal_raw3[xyz_offset];
    const float input_normal_y = normal_raw3[xyz_offset + 1];
    const float input_normal_z = normal_raw3[xyz_offset + 2];
    float normal_x_raw = input_normal_x;
    float normal_y_raw = input_normal_y;
    float normal_z_raw = input_normal_z;

    // gaussian2d_utils.py:210-223 flips against means3D-camera_center,
    // then performs a clamp-based length-three normalization.
    const float to_point_x = sub_rn(px, camera_center[0]);
    const float to_point_y = sub_rn(py, camera_center[1]);
    const float to_point_z = sub_rn(pz, camera_center[2]);
    const float neg_to_point_x = sub_rn(0.0f, to_point_x);
    const float neg_to_point_y = sub_rn(0.0f, to_point_y);
    const float neg_to_point_z = sub_rn(0.0f, to_point_z);
    const float flip_dot = dot3_pytorch(
        normal_x_raw,
        normal_y_raw,
        normal_z_raw,
        neg_to_point_x,
        neg_to_point_y,
        neg_to_point_z);
    const bool positive = flip_dot >= 0.0f;
    const float flip_sign = positive ? 1.0f : -1.0f;
    normal_x_raw = mul_rn(normal_x_raw, flip_sign);
    normal_y_raw = mul_rn(normal_y_raw, flip_sign);
    normal_z_raw = mul_rn(normal_z_raw, flip_sign);
    float normal_x;
    float normal_y;
    float normal_z;
    const float normal_denominator = normalize3_clamp_epsilon(
        normal_x_raw,
        normal_y_raw,
        normal_z_raw,
        kNormalEpsilon,
        normal_x,
        normal_y,
        normal_z);

    // optix_utils.py:403-408 keeps multiply, sum, scale, multiply,
    // subtraction and the final normalization as distinct FP32 stages.
    const float reflection_product_x = mul_rn(normal_x, view_x);
    const float reflection_product_y = mul_rn(normal_y, view_y);
    const float reflection_product_z = mul_rn(normal_z, view_z);
    const float dot_nv = add_rn(
        add_rn(reflection_product_x, reflection_product_z),
        reflection_product_y);
    const float two_dot = mul_rn(2.0f, dot_nv);
    const float scaled_normal_x = mul_rn(two_dot, normal_x);
    const float scaled_normal_y = mul_rn(two_dot, normal_y);
    const float scaled_normal_z = mul_rn(two_dot, normal_z);
    const float reflected_x = sub_rn(scaled_normal_x, view_x);
    const float reflected_y = sub_rn(scaled_normal_y, view_y);
    const float reflected_z = sub_rn(scaled_normal_z, view_z);
    float ray_x;
    float ray_y;
    float ray_z;
    const float ray_denominator = normalize3_add_epsilon(
        reflected_x,
        reflected_y,
        reflected_z,
        kNormalizeEpsilon,
        ray_x,
        ray_y,
        ray_z);

    // optix_utils.py:410 has two eager ATen stages. Keep the product rounded
    // before the addition, so NVCC cannot replace them with one FFMA.
    const float offset_x = mul_rn(kRayOriginOffset, ray_x);
    const float offset_y = mul_rn(kRayOriginOffset, ray_y);
    const float offset_z = mul_rn(kRayOriginOffset, ray_z);
    const float origin_x = add_rn(px, offset_x);
    const float origin_y = add_rn(py, offset_y);
    const float origin_z = add_rn(pz, offset_z);

    output36[output_offset] = view_x;
    output36[output_offset + 1] = view_y;
    output36[output_offset + 2] = view_z;
    output36[output_offset + 3] = normal_x;
    output36[output_offset + 4] = normal_y;
    output36[output_offset + 5] = normal_z;
    output36[output_offset + 6] = reflected_x;
    output36[output_offset + 7] = reflected_y;
    output36[output_offset + 8] = reflected_z;
    output36[output_offset + 9] = ray_x;
    output36[output_offset + 10] = ray_y;
    output36[output_offset + 11] = ray_z;
    output36[output_offset + 12] = origin_x;
    output36[output_offset + 13] = origin_y;
    output36[output_offset + 14] = origin_z;
    output36[output_offset + 15] = positive ? 1.0f : 0.0f;
    output36[output_offset + 16] = input_normal_x;
    output36[output_offset + 17] = input_normal_y;
    output36[output_offset + 18] = input_normal_z;
    output36[output_offset + 19] = normal_x_raw;
    output36[output_offset + 20] = normal_y_raw;
    output36[output_offset + 21] = normal_z_raw;
    output36[output_offset + 22] = normal_denominator;
    output36[output_offset + 23] = reflection_product_x;
    output36[output_offset + 24] = reflection_product_y;
    output36[output_offset + 25] = reflection_product_z;
    output36[output_offset + 26] = dot_nv;
    output36[output_offset + 27] = two_dot;
    output36[output_offset + 28] = scaled_normal_x;
    output36[output_offset + 29] = scaled_normal_y;
    output36[output_offset + 30] = scaled_normal_z;
    output36[output_offset + 31] = ray_denominator;
    output36[output_offset + 32] = offset_x;
    output36[output_offset + 33] = offset_y;
    output36[output_offset + 34] = offset_z;
    output36[output_offset + 35] = flip_dot;
}

}  // namespace

cudaError_t hs_launch_precise_reflection_prefix_f32(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    const uint64_t block_count =
        (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (block_count > static_cast<uint64_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    precise_reflection_prefix_f32_kernel<<<
        static_cast<uint32_t>(block_count),
        kThreadsPerBlock,
        0,
        0>>>(rotation_wxyz, count, rotation_matrix9);
    return cudaGetLastError();
}

cudaError_t hs_launch_precise_reflection_suffix_f32(
    const float* xyz,
    const float* camera_center,
    const float* normal_raw3,
    uint64_t count,
    float* output36) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    const uint64_t block_count =
        (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (block_count > static_cast<uint64_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    precise_reflection_suffix_f32_kernel<<<
        static_cast<uint32_t>(block_count),
        kThreadsPerBlock,
        0,
        0>>>(xyz, camera_center, normal_raw3, count, output36);
    return cudaGetLastError();
}
