#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct FlashgsForwardOp : Op {
    Var* positions;
    Var* shs;
    Var* colors;
    Var* opacities;
    Var* covariance;
    Var* scales;
    Var* rotations;
    Var* specular;
    Var* roughness;
    Var* color;
    Var* other;
    Var* depth;
    Var* alpha;
    Var* normal;
    Var* median_depth;
    Var* distortion;
    Var* rendered;
    int64 width;
    int64 height;
    int64 use_sh;
    int64 initial_pair_capacity;
    int64 handle_value;
    float64 scale_modifier;
    float64 focal_x;
    float64 focal_y;
    float64 z_far;
    float64 z_near;
    float64 camera_x;
    float64 camera_y;
    float64 camera_z;
    float64 rotation_0;
    float64 rotation_1;
    float64 rotation_2;
    float64 rotation_3;
    float64 rotation_4;
    float64 rotation_5;
    float64 rotation_6;
    float64 rotation_7;
    float64 rotation_8;
    float64 background_r;
    float64 background_g;
    float64 background_b;

    // @attrs(multiple_outputs)
    FlashgsForwardOp(Var* positions, Var* shs, Var* colors, Var* opacities, Var* covariance, Var* scales, Var* rotations, Var* specular, Var* roughness, int64 width, int64 height, int64 use_sh, int64 initial_pair_capacity, int64 handle_value, float64 scale_modifier, float64 focal_x, float64 focal_y, float64 z_far, float64 z_near, float64 camera_x, float64 camera_y, float64 camera_z, float64 rotation_0, float64 rotation_1, float64 rotation_2, float64 rotation_3, float64 rotation_4, float64 rotation_5, float64 rotation_6, float64 rotation_7, float64 rotation_8, float64 background_r, float64 background_g, float64 background_b);

    const char* name() const override { return "flashgs_forward"; }
    DECLARE_jit_run;
};

}  // namespace jittor
