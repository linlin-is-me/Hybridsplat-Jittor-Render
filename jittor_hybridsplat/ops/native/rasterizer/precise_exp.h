#ifndef HYBRIDSPLAT_PRECISE_EXP_H_
#define HYBRIDSPLAT_PRECISE_EXP_H_

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_precise_exp_f32(
    const float* input,
    float* output,
    uint64_t count) noexcept;

#endif
