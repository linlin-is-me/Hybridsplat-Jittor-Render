#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseExpOp : Op {
    Var* input;
    Var* output;

    PreciseExpOp(Var* input);

    const char* name() const override { return "precise_exp"; }
    DECLARE_jit_run;
};

}  // namespace jittor
