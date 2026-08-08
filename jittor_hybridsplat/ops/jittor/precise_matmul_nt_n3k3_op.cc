#include "precise_matmul_nt_n3k3_op.h"

#include "hybridsplat_rasterizer_c_api.h"

namespace jittor {

#ifndef JIT

PreciseMatmulNtN3k3Op::PreciseMatmulNtN3k3Op(
    Var* left,
    Var* right,
    int64 handle_value)
    : left(left), right(right), handle_value(handle_value) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(left->dtype() == ns_float32)
        << "precise_matmul_nt_n3k3 left must be FP32";
    CHECK(right->dtype() == ns_float32)
        << "precise_matmul_nt_n3k3 right must be FP32";
    CHECK(left->shape.size() == 2 && left->shape[1] == 3)
        << "precise_matmul_nt_n3k3 left must have shape (N,3)";
    CHECK(
        right->shape.size() == 2 &&
        right->shape[0] == 3 &&
        right->shape[1] == 3)
        << "precise_matmul_nt_n3k3 right must have shape (3,3)";
    output = create_output(left->shape, ns_float32);
}

void PreciseMatmulNtN3k3Op::jit_prepare(JK& jk) {
    add_jit_define(
        jk,
        "ABI",
        JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PreciseMatmulNtN3k3Op::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    const int32_t status =
        hs_rasterizer_precise_matmul_nt_n3k3_f32_v2(
            handle,
            left->ptr<float32>(),
            right->ptr<float32>(),
            static_cast<uint64_t>(left->shape[0]),
            output->ptr<float32>(),
            error_text,
            sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseMatmulNtN3k3Op requires CUDA";
#endif
}

#endif

}  // namespace jittor
