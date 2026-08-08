#include "precise_camera_inverse_op.h"

namespace jittor {

#ifndef JIT

PreciseCameraInverseOp::PreciseCameraInverseOp(Var* matrix) : matrix(matrix) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(matrix->dtype() == ns_float32)
        << "precise_camera_inverse matrix must be FP32";
    CHECK(
        matrix->shape.size() == 2 &&
        matrix->shape[0] == 3 &&
        matrix->shape[1] == 3)
        << "precise_camera_inverse matrix must have shape (3,3)";
    output = create_output(matrix->shape, ns_float32);
}

void PreciseCameraInverseOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "VERSION", JK::hex1(1));
}

#else

#ifdef JIT_cuda

__device__ __forceinline__ float precise_camera_divide_rn_f32(
    float numerator,
    float denominator) {
    float result;
    asm volatile(
        "div.rn.f32 %0, %1, %2;"
        : "=f"(result)
        : "f"(numerator), "f"(denominator));
    return result;
}

__global__ static void precise_camera_inverse_minors_kernel(
    const float* matrix,
    float* minors) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const float r00 = matrix[0];
    const float r01 = matrix[1];
    const float r02 = matrix[2];
    const float r10 = matrix[3];
    const float r11 = matrix[4];
    const float r12 = matrix[5];
    const float r20 = matrix[6];
    const float r21 = matrix[7];
    const float r22 = matrix[8];
    minors[0] = r11 * r22 - r21 * r12;
    minors[3] = (0.0f - r10) * r22 + r20 * r12;
    minors[6] = r10 * r21 - r20 * r11;
    minors[1] = (0.0f - r01) * r22 + r21 * r02;
    minors[4] = r00 * r22 - r20 * r02;
    minors[7] = (0.0f - r00) * r21 + r20 * r01;
    minors[2] = r01 * r12 - r11 * r02;
    minors[5] = (0.0f - r00) * r12 + r10 * r02;
    minors[8] = r00 * r11 - r10 * r01;
}

__global__ static void precise_camera_inverse_divide_kernel(
    const float* matrix,
    float* in_out) {
    __shared__ float minors[9];
    const int index = static_cast<int>(threadIdx.x);
    if (blockIdx.x != 0 || index >= 9) {
        return;
    }
    minors[index] = in_out[index];
    __syncthreads();
    const float denominator =
        ((matrix[0] * minors[0] + matrix[1] * minors[3])
         + matrix[2] * minors[6])
        + 1.1920928955078125e-7f;
    in_out[index] = precise_camera_divide_rn_f32(
        minors[index],
        denominator);
}

#endif

void PreciseCameraInverseOp::jit_run() {
#ifdef JIT_cuda
    precise_camera_inverse_minors_kernel<<<1, 1>>>(
        matrix->ptr<float32>(),
        output->ptr<float32>());
    cudaError_t error = cudaGetLastError();
    CHECK(error == cudaSuccess)
        << "precise_camera_inverse minors launch failed: "
        << cudaGetErrorString(error);
    precise_camera_inverse_divide_kernel<<<1, 9>>>(
        matrix->ptr<float32>(),
        output->ptr<float32>());
    error = cudaGetLastError();
    CHECK(error == cudaSuccess)
        << "precise_camera_inverse divide launch failed: "
        << cudaGetErrorString(error);
#else
    CHECK(false) << "PreciseCameraInverseOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
