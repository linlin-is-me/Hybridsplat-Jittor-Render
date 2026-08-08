#ifndef HYBRIDSPLAT_PRECISE_CAMERA_CENTER_H_
#define HYBRIDSPLAT_PRECISE_CAMERA_CENTER_H_

#include <cuda_runtime_api.h>

cudaError_t hs_launch_negative_rotation3x3_f32(
    const float* rotation3x3,
    float* negative_rotation3x3);

#endif
