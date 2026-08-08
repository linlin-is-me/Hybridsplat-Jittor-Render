#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PrecisePixelNormalOp : Op {
    Var* normal_raw;
    Var* output;

    PrecisePixelNormalOp(Var* normal_raw);

    const char* name() const override {
        return "precise_pixel_normal";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
