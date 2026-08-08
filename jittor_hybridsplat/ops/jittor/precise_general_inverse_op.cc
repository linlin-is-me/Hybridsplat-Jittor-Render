#include "precise_general_inverse_op.h"

#include "hybridsplat_rasterizer_c_api.h"

namespace jittor {

#ifndef JIT

PreciseGeneralInverseOp::PreciseGeneralInverseOp(Var* input, int64 handle_value)
    : input(input), handle_value(handle_value) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(input->dtype() == ns_float32)
        << "precise_general_inverse input must be FP32";
    CHECK(
        input->shape.size() == 2 &&
        input->shape[0] == input->shape[1] &&
        (input->shape[0] == 3 || input->shape[0] == 4))
        << "precise_general_inverse input must have shape (3,3) or (4,4)";
    output = create_output(input->shape, ns_float32);
}

void PreciseGeneralInverseOp::jit_prepare(JK& jk) {
    add_jit_define(
        jk,
        "ABI",
        JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PreciseGeneralInverseOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    const int32_t status = hs_rasterizer_precise_general_inverse_f32_v2(
        handle,
        input->ptr<float32>(),
        static_cast<uint32_t>(input->shape[0]),
        output->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseGeneralInverseOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
