#include "precise_sh.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr uint32_t kThreadsPerBlock = 256;

__global__ void precise_sh_to_rgb_f32_kernel(
    const float* sh,
    const float* directions,
    uint64_t count,
    uint32_t coefficient_count,
    uint32_t degree,
    float* output) {
    const uint64_t index =
        static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    constexpr float c0 = 0.28209479177387814f;
    constexpr float c1 = 0.4886025119029199f;
    constexpr float c20 = 1.0925484305920792f;
    constexpr float c21 = -1.0925484305920792f;
    constexpr float c22 = 0.31539156525252005f;
    constexpr float c23 = -1.0925484305920792f;
    constexpr float c24 = 0.5462742152960396f;
    constexpr float c30 = -0.5900435899266435f;
    constexpr float c31 = 2.890611442640554f;
    constexpr float c32 = -0.4570457994644658f;
    constexpr float c33 = 0.3731763325901154f;
    constexpr float c34 = -0.4570457994644658f;
    constexpr float c35 = 1.445305721320277f;
    constexpr float c36 = -0.5900435899266435f;

    const float x = directions[index * 3];
    const float y = directions[index * 3 + 1];
    const float z = directions[index * 3 + 2];
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float yz = y * z;
    const float xz = x * z;

    for (uint32_t channel = 0; channel < 3; ++channel) {
        const float* coefficients =
            sh + (index * 3 + channel) * coefficient_count;
        float result = c0 * coefficients[0];

        if (degree > 0) {
            float term = c1 * y;
            term = term * coefficients[1];
            result = result - term;
            term = c1 * z;
            term = term * coefficients[2];
            result = result + term;
            term = c1 * x;
            term = term * coefficients[3];
            result = result - term;
        }

        if (degree > 1) {
            float term = c20 * xy;
            term = term * coefficients[4];
            result = result + term;
            term = c21 * yz;
            term = term * coefficients[5];
            result = result + term;
            float polynomial = 2.0f * zz;
            polynomial = polynomial - xx;
            polynomial = polynomial - yy;
            term = c22 * polynomial;
            term = term * coefficients[6];
            result = result + term;
            term = c23 * xz;
            term = term * coefficients[7];
            result = result + term;
            polynomial = xx - yy;
            term = c24 * polynomial;
            term = term * coefficients[8];
            result = result + term;
        }

        if (degree > 2) {
            float polynomial = 3.0f * xx;
            polynomial = polynomial - yy;
            float term = c30 * y;
            term = term * polynomial;
            term = term * coefficients[9];
            result = result + term;
            term = c31 * xy;
            term = term * z;
            term = term * coefficients[10];
            result = result + term;
            polynomial = 4.0f * zz;
            polynomial = polynomial - xx;
            polynomial = polynomial - yy;
            term = c32 * y;
            term = term * polynomial;
            term = term * coefficients[11];
            result = result + term;
            polynomial = 2.0f * zz;
            term = 3.0f * xx;
            polynomial = polynomial - term;
            term = 3.0f * yy;
            polynomial = polynomial - term;
            term = c33 * z;
            term = term * polynomial;
            term = term * coefficients[12];
            result = result + term;
            polynomial = 4.0f * zz;
            polynomial = polynomial - xx;
            polynomial = polynomial - yy;
            term = c34 * x;
            term = term * polynomial;
            term = term * coefficients[13];
            result = result + term;
            polynomial = xx - yy;
            term = c35 * z;
            term = term * polynomial;
            term = term * coefficients[14];
            result = result + term;
            polynomial = 3.0f * yy;
            polynomial = xx - polynomial;
            term = c36 * x;
            term = term * polynomial;
            term = term * coefficients[15];
            result = result + term;
        }

        result = result + 0.5f;
        output[index * 3 + channel] = fmaxf(result, 0.0f);
    }
}

}  // namespace

cudaError_t hs_launch_precise_sh_to_rgb_f32(
    const float* sh,
    const float* directions,
    uint64_t count,
    uint32_t coefficient_count,
    uint32_t degree,
    float* output) noexcept {
    if (count == 0) {
        return cudaSuccess;
    }
    const uint64_t block_count =
        (count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (block_count > static_cast<uint64_t>(UINT32_MAX)) {
        return cudaErrorInvalidValue;
    }
    precise_sh_to_rgb_f32_kernel<<<
        static_cast<uint32_t>(block_count),
        kThreadsPerBlock,
        0,
        0>>>(
        sh,
        directions,
        count,
        coefficient_count,
        degree,
        output);
    return cudaGetLastError();
}
