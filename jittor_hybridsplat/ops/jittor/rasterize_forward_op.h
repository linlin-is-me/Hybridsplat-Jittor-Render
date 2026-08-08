#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct RasterizeForwardOp : Op {
    Var* background;
    Var* means3d;
    Var* colors;
    Var* opacities;
    Var* scales;
    Var* rotations;
    Var* viewmatrix;
    Var* projmatrix;
    Var* camera_position;
    Var* color;
    Var* auxiliary;
    Var* weight;
    Var* radii;
    Var* rendered;
    int64 width;
    int64 height;
    float64 scale_modifier;
    float64 tan_fovx;
    float64 tan_fovy;
    int64 prefiltered;
    int64 handle_value;

    // @attrs(multiple_outputs)
    RasterizeForwardOp(Var* background, Var* means3d, Var* colors, Var* opacities, Var* scales, Var* rotations, Var* viewmatrix, Var* projmatrix, Var* camera_position, int64 width, int64 height, float64 scale_modifier, float64 tan_fovx, float64 tan_fovy, int64 prefiltered, int64 handle_value);

    const char* name() const override { return "rasterize_forward"; }
    DECLARE_jit_run;
};

}  // namespace jittor
