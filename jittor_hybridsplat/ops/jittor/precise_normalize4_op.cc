#include "precise_normalize4_op.h"
#include "hybridsplat_rasterizer_c_api.h"
namespace jittor {
#ifndef JIT
PreciseNormalize4Op::PreciseNormalize4Op(Var* input) : input(input) {
    flags.set(NodeFlags::_cpu, 0); flags.set(NodeFlags::_cuda, 1);
    CHECK(input->dtype() == ns_float32) << "precise_normalize4 input must be FP32";
    CHECK(input->shape.size() == 2 && input->shape[1] == 4) << "precise_normalize4 input must have shape [P,4]";
    output = create_output(input->shape, ns_float32);
}
void PreciseNormalize4Op::jit_prepare(JK& jk) { add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION)); }
#else
void PreciseNormalize4Op::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_normalize4_f32_v1(input->ptr<float32>(), output->ptr<float32>(), static_cast<uint64_t>(input->shape[0]), error_text, sizeof(error_text));
    CHECK(status == HS_STATUS_OK) << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseNormalize4Op requires CUDA";
#endif
}
#endif
}
