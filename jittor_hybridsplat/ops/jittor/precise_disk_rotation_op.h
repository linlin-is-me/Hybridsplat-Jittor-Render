#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseDiskRotationOp : Op {
    Var* rotation_wxyz;
    Var* output;

    PreciseDiskRotationOp(Var* rotation_wxyz);

    const char* name() const override {
        return "precise_disk_rotation";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
