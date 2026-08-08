#include "precise_camera_center_op.h"

#include "hybridsplat_rasterizer_c_api.h"

namespace jittor {

#ifndef JIT

PreciseCameraCenterOp::PreciseCameraCenterOp(
    Var* rotation,
    Var* translation,
    int64 handle_value)
    : rotation(rotation), translation(translation), handle_value(handle_value) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(rotation->dtype() == ns_float32)
        << "precise_camera_center rotation must be FP32";
    CHECK(translation->dtype() == ns_float32)
        << "precise_camera_center translation must be FP32";
    CHECK(
        rotation->shape.size() == 2 &&
        rotation->shape[0] == 3 &&
        rotation->shape[1] == 3)
        << "precise_camera_center rotation must have shape (3,3)";
    CHECK(
        translation->shape.size() == 2 &&
        translation->shape[0] == 3 &&
        translation->shape[1] == 1)
        << "precise_camera_center translation must have shape (3,1)";
    output = create_output({3}, ns_float32);
}

void PreciseCameraCenterOp::jit_prepare(JK& jk) {
    add_jit_define(
        jk,
        "ABI",
        JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PreciseCameraCenterOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    const int32_t status = hs_rasterizer_precise_camera_center_f32_v2(
        handle,
        rotation->ptr<float32>(),
        translation->ptr<float32>(),
        output->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseCameraCenterOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
