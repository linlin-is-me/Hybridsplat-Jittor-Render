#pragma once
#include "op.h"
#include "var.h"
namespace jittor {
struct PreciseFuseMapsOp : Op {
    Var* base_rgb;
    Var* reflection_rgb;
    Var* specular;
    Var* diffuse;
    Var* final_rgb;
    Var* stored_reflection;
    // @attrs(multiple_outputs)
    PreciseFuseMapsOp(Var* base_rgb, Var* reflection_rgb, Var* specular);
    const char* name() const override { return "precise_fuse_maps"; }
    DECLARE_jit_run;
};
}
