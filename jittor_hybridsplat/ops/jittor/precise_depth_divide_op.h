#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseDepthDivideOp : Op {
    Var* numerator;
    Var* denominator;
    Var* output;

    PreciseDepthDivideOp(Var* numerator, Var* denominator);

    const char* name() const override { return "precise_depth_divide"; }
    DECLARE_jit_run;
};

}  // namespace jittor
