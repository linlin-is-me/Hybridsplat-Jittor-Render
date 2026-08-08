#pragma once

#include "op.h"
#include "var.h"

namespace jittor {

struct PreciseReflectionSuffixOp : Op {
    Var* xyz;
    Var* camera_center;
    Var* normal_raw3;
    Var* output;

    PreciseReflectionSuffixOp(Var* xyz, Var* camera_center, Var* normal_raw3);

    const char* name() const override {
        return "precise_reflection_suffix";
    }
    DECLARE_jit_run;
};

}  // namespace jittor
