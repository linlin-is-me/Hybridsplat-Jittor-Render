#pragma once
#include "op.h"
#include "var.h"
namespace jittor {
struct PreciseNormalize4Op : Op {
    Var* input;
    Var* output;
    PreciseNormalize4Op(Var* input);
    const char* name() const override { return "precise_normalize4"; }
    DECLARE_jit_run;
};
}
