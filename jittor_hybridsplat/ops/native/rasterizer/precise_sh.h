#ifndef HYBRIDSPLAT_PRECISE_SH_H_
#define HYBRIDSPLAT_PRECISE_SH_H_

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_precise_sh_to_rgb_f32(
    const float* sh,
    const float* directions,
    uint64_t count,
    uint32_t coefficient_count,
    uint32_t degree,
    float* output) noexcept;

#endif
