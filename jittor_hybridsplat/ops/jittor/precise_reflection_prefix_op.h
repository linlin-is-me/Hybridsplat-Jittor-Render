#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseReflectionPrefixOp : Op {
    Var* rotation_wxyz;
    Var* output;

    PreciseReflectionPrefixOp(Var* rotation_wxyz);

    const char* name() const override {
        return "precise_reflection_prefix";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
