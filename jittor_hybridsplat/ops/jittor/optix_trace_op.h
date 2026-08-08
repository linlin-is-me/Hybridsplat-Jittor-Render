#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct OptixTraceOp : Op {
    Var* ray_origins;
    Var* ray_directions;
    Var* background;
    Var* means3d;
    Var* shs;
    Var* others;
    Var* opacities;
    Var* scales;
    Var* rotations;
    Var* viewmatrix;
    Var* projmatrix;
    Var* camera_position;
    Var* rgb;
    Var* depth;
    Var* acc;
    Var* normal;
    Var* dist;
    Var* auxiliary;
    Var* mid;
    Var* wet;
    int64 degree;
    float64 scale_modifier;
    int64 start_from_first;
    int64 max_trace_depth;
    float64 specular_threshold;
    int64 use_others;
    int64 handle_value;

    // @attrs(multiple_outputs)
    OptixTraceOp(Var* ray_origins, Var* ray_directions, Var* background, Var* means3d, Var* shs, Var* others, Var* opacities, Var* scales, Var* rotations, Var* viewmatrix, Var* projmatrix, Var* camera_position, int64 degree, float64 scale_modifier, int64 start_from_first, int64 max_trace_depth, float64 specular_threshold, int64 use_others, int64 handle_value);

    const char* name() const override { return "optix_trace"; }
    DECLARE_jit_run;
};

}  // namespace jittor
