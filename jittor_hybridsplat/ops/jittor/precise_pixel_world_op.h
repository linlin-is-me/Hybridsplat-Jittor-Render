#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PrecisePixelWorldOp : Op {
    Var* rotation;
    Var* vectors;
    Var* output;
    int64 handle_value;

    PrecisePixelWorldOp(Var* rotation, Var* vectors, int64 handle_value);

    const char* name() const override {
        return "precise_pixel_world";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
