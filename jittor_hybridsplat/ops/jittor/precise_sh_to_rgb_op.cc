#include "precise_sh_to_rgb_op.h"

#include "hybridsplat_rasterizer_c_api.h"

namespace jittor {

#ifndef JIT

PreciseShToRgbOp::PreciseShToRgbOp(
    Var* sh,
    Var* directions,
    int64 degree)
    : sh(sh), directions(directions), degree(degree) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(sh->dtype() == ns_float32)
        << "precise_sh_to_rgb SH input must be FP32";
    CHECK(directions->dtype() == ns_float32)
        << "precise_sh_to_rgb directions must be FP32";
    CHECK(sh->shape.size() == 3)
        << "precise_sh_to_rgb SH input must have three dimensions";
    CHECK(directions->shape.size() == 2)
        << "precise_sh_to_rgb directions must have two dimensions";
    CHECK(sh->shape[0] == directions->shape[0])
        << "precise_sh_to_rgb point counts differ";
    CHECK(sh->shape[1] == 3)
        << "precise_sh_to_rgb SH channel count must be three";
    CHECK(directions->shape[1] == 3)
        << "precise_sh_to_rgb direction width must be three";
    CHECK(degree >= 0 && degree <= 3)
        << "precise_sh_to_rgb degree must be in [0,3]";
    coefficient_count = sh->shape[2];
    CHECK(coefficient_count >= (degree + 1) * (degree + 1))
        << "precise_sh_to_rgb coefficient count is too small";
    output = create_output({directions->shape[0], 3}, ns_float32);
}

void PreciseShToRgbOp::jit_prepare(JK& jk) {
    add_jit_define(
        jk,
        "ABI",
        JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PreciseShToRgbOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_sh_to_rgb_f32_v1(
        sh->ptr<float32>(),
        directions->ptr<float32>(),
        static_cast<uint64_t>(directions->shape[0]),
        static_cast<uint32_t>(coefficient_count),
        static_cast<uint32_t>(degree),
        output->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseShToRgbOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
