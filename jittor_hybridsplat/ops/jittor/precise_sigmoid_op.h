#pragma once
#include "op.h"
#include "var.h"
namespace jittor {
struct PreciseSigmoidOp : Op {
    Var* input;
    Var* output;
    PreciseSigmoidOp(Var* input);
    const char* name() const override { return "precise_sigmoid"; }
    DECLARE_jit_run;
};
}
