#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseGeneralInverseOp : Op {
    Var* input;
    Var* output;
    int64 handle_value;

    PreciseGeneralInverseOp(Var* input, int64 handle_value);

    const char* name() const override {
        return "precise_general_inverse";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
