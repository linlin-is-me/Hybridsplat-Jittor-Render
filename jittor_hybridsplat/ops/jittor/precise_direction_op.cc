#include "precise_direction_op.h"

#include "hybridsplat_rasterizer_c_api.h"

namespace jittor {

#ifndef JIT

PreciseDirectionOp::PreciseDirectionOp(Var* points, Var* center)
    : points(points), center(center) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(points->dtype() == ns_float32)
        << "precise_direction points must be FP32";
    CHECK(center->dtype() == ns_float32)
        << "precise_direction center must be FP32";
    CHECK(points->shape.size() == 2 && points->shape[1] == 3)
        << "precise_direction points must have shape (P,3)";
    CHECK(center->num == 3)
        << "precise_direction center must contain three values";
    output = create_output(points->shape, ns_float32);
}

void PreciseDirectionOp::jit_prepare(JK& jk) {
    add_jit_define(
        jk,
        "ABI",
        JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PreciseDirectionOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_direction_f32_v1(
        points->ptr<float32>(),
        center->ptr<float32>(),
        static_cast<uint64_t>(points->shape[0]),
        output->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseDirectionOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
