#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct OptixBuildSceneOp : Op {
    Var* vertices;
    Var* triangles;
    Var* status;
    int64 scene_signature;
    int64 rebuild;
    int64 handle_value;

    OptixBuildSceneOp(Var* vertices, Var* triangles, int64 scene_signature, int64 rebuild, int64 handle_value);

    const char* name() const override { return "optix_build_scene"; }
    DECLARE_jit_run;
};

}  // namespace jittor
