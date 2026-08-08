#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseCameraInverseOp : Op {
    Var* matrix;
    Var* output;

    PreciseCameraInverseOp(Var* matrix);

    const char* name() const override {
        return "precise_camera_inverse";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
