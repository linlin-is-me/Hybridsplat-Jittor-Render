#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseDirectionOp : Op {
    Var* points;
    Var* center;
    Var* output;

    PreciseDirectionOp(Var* points, Var* center);

    const char* name() const override { return "precise_direction"; }
    DECLARE_jit_run;
};

}  // namespace jittor
