#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseCameraCenterOp : Op {
    Var* rotation;
    Var* translation;
    Var* output;
    int64 handle_value;

    PreciseCameraCenterOp(Var* rotation, Var* translation, int64 handle_value);

    const char* name() const override {
        return "precise_camera_center";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
