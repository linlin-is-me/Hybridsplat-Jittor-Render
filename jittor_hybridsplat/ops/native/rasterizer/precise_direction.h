#ifndef HYBRIDSPLAT_PRECISE_DIRECTION_H_
#define HYBRIDSPLAT_PRECISE_DIRECTION_H_

#include <cuda_runtime_api.h>

#include <cstdint>

cudaError_t hs_launch_precise_direction_f32(
    const float* points,
    const float* center,
    uint64_t count,
    float* output) noexcept;

#endif
