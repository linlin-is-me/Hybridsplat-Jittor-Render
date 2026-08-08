#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseShToRgbOp : Op {
    Var* sh;
    Var* directions;
    Var* output;
    int64 degree;
    int64 coefficient_count;

    PreciseShToRgbOp(Var* sh, Var* directions, int64 degree);

    const char* name() const override { return "precise_sh_to_rgb"; }
    DECLARE_jit_run;
};

}  // namespace jittor
