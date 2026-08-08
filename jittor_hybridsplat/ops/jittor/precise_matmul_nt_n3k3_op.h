#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseMatmulNtN3k3Op : Op {
    Var* left;
    Var* right;
    Var* output;
    int64 handle_value;

    PreciseMatmulNtN3k3Op(Var* left, Var* right, int64 handle_value);

    const char* name() const override {
        return "precise_matmul_nt_n3k3";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
