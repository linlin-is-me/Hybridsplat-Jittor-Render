#include "precise_sigmoid_op.h"
#include "hybridsplat_rasterizer_c_api.h"
namespace jittor {
#ifndef JIT
PreciseSigmoidOp::PreciseSigmoidOp(Var* input) : input(input) {
    flags.set(NodeFlags::_cpu, 0); flags.set(NodeFlags::_cuda, 1);
    CHECK(input->dtype() == ns_float32) << "precise_sigmoid input must be FP32";
    output = create_output(input->shape, ns_float32);
}
void PreciseSigmoidOp::jit_prepare(JK& jk) { add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION)); }
#else
void PreciseSigmoidOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_sigmoid_f32_v1(input->ptr<float32>(), output->ptr<float32>(), static_cast<uint64_t>(input->num), error_text, sizeof(error_text));
    CHECK(status == HS_STATUS_OK) << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseSigmoidOp requires CUDA";
#endif
}
#endif
}
