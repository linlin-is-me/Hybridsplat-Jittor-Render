#include "precise_disk_rotation.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;
constexpr float kQuaternionEpsilon = 1.0e-12f;

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

__device__ __forceinline__ float norm4_pytorch(
    float c0,
    float c1,
    float c2,
    float c3) {
    const float s0 = mul_rn(c0, c0);
    const float s1 = mul_rn(c1, c1);
    const float s2 = mul_rn(c2, c2);
    const float s3 = mul_rn(c3, c3);
    return __fsqrt_rn(add_rn(add_rn(s0, s1), add_rn(s2, s3)));
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

__global__ void precise_disk_rotation_f32_kernel(
    const float* rotation_wxyz,
    uint64_t count,
    float* rotation_matrix9) {
    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const uint64_t input_offset = index * 4;
    // GaussianParameters.rotation follows torch.nn.functional.normalize.
    float active_w;
    float active_x;
    float active_y;
    float active_z;
    normalize4_clamp_epsilon(
        rotation_wxyz[input_offset],
        rotation_wxyz[input_offset + 1],
        rotation_wxyz[input_offset + 2],
        rotation_wxyz[input_offset + 3],
        kQuaternionEpsilon,
        active_w,
        active_x,
        active_y,
        active_z);

    // build_rotation performs a second torch.norm followed by division.
    const float second_norm =
        norm4_pytorch(active_w, active_x, active_y, active_z);
    const float w = div_rn(active_w, second_norm);
    const float x = div_rn(active_x, second_norm);
    const float y = div_rn(active_y, second_norm);
    const float z = div_rn(active_z, second_norm);

    // The first reflection build_cov call runs the profiling graph. Every
    // multiplication, addition and subtraction therefore remains a distinct
    // round-to-nearest FP32 stage; unlike get_normal's later call, no
    // TensorExpr FMA contraction is applied here.
    const float xy = mul_rn(x, y);
    const float xz = mul_rn(x, z);
    const float yz = mul_rn(y, z);
    const float wz = mul_rn(w, z);
    const float wy = mul_rn(w, y);
    const float wx = mul_rn(w, x);
    const float xx = mul_rn(x, x);
    const float yy = mul_rn(y, y);
    const float zz = mul_rn(z, z);
    const uint64_t output_offset = index * 9;
    rotation_matrix9[output_offset] =
        sub_rn(1.0f, mul_rn(2.0f, add_rn(yy, zz)));
    rotation_matrix9[output_offset + 1] =
        mul_rn(2.0f, sub_rn(xy, wz));
    rotation_matrix9[output_offset + 2] =
        mul_rn(2.0f, add_rn(xz, wy));
    rotation_matrix9[output_offset + 3] =
        mul_rn(2.0f, add_rn(xy, wz));
    rotation_matrix9[output_offset + 4] =
        sub_rn(1.0f, mul_rn(2.0f, add_rn(xx, zz)));
    rotation_matrix9[output_offset + 5] =
        mul_rn(2.0f, sub_rn(yz, wx));
    rotation_matrix9[output_offset + 6] =
        mul_rn(2.0f, sub_rn(xz, wy));
    rotation_matrix9[output_offset + 7] =
        mul_rn(2.0f, add_rn(yz, wx));
    rotation_matrix9[output_offset + 8] =
        sub_rn(1.0f, mul_rn(2.0f, add_rn(xx, yy)));
}

}  // namespace

cudaError_t hs_launch_precise_disk_rotation_f32(
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
    precise_disk_rotation_f32_kernel<<<
        static_cast<uint32_t>(block_count),
        kThreadsPerBlock,
        0,
        0>>>(rotation_wxyz, count, rotation_matrix9);
    return cudaGetLastError();
}
