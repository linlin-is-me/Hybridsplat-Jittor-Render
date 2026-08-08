#include "precise_reflection_prefix_op.h"

#include "hybridsplat_rasterizer_c_api.h"

namespace jittor {

#ifndef JIT

PreciseReflectionPrefixOp::PreciseReflectionPrefixOp(Var* rotation_wxyz)
    : rotation_wxyz(rotation_wxyz) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(rotation_wxyz->dtype() == ns_float32)
        << "precise_reflection_prefix rotation must be FP32";
    CHECK(
        rotation_wxyz->shape.size() == 2 &&
        rotation_wxyz->shape[1] == 4)
        << "precise_reflection_prefix rotation must have shape (P,4)";
    output = create_output(
        {rotation_wxyz->shape[0], 3, 3},
        ns_float32);
}

void PreciseReflectionPrefixOp::jit_prepare(JK& jk) {
    add_jit_define(
        jk,
        "ABI",
        JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PreciseReflectionPrefixOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_reflection_prefix_f32_v1(
        rotation_wxyz->ptr<float32>(),
        static_cast<uint64_t>(rotation_wxyz->shape[0]),
        output->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseReflectionPrefixOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
