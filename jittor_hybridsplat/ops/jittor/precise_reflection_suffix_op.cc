#include "precise_reflection_suffix_op.h"

#include "hybridsplat_rasterizer_c_api.h"

namespace jittor {

#ifndef JIT

PreciseReflectionSuffixOp::PreciseReflectionSuffixOp(
    Var* xyz,
    Var* camera_center,
    Var* normal_raw3)
    : xyz(xyz),
      camera_center(camera_center),
      normal_raw3(normal_raw3) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(xyz->dtype() == ns_float32)
        << "precise_reflection_suffix xyz must be FP32";
    CHECK(camera_center->dtype() == ns_float32)
        << "precise_reflection_suffix camera center must be FP32";
    CHECK(normal_raw3->dtype() == ns_float32)
        << "precise_reflection_suffix normal must be FP32";
    CHECK(xyz->shape.size() == 2 && xyz->shape[1] == 3)
        << "precise_reflection_suffix xyz must have shape (P,3)";
    CHECK(camera_center->num == 3)
        << "precise_reflection_suffix camera center must contain three values";
    CHECK(
        normal_raw3->shape.size() == 2 &&
        normal_raw3->shape[0] == xyz->shape[0] &&
        normal_raw3->shape[1] == 3)
        << "precise_reflection_suffix normal must have shape (P,3)";
    output = create_output({xyz->shape[0], 36}, ns_float32);
}

void PreciseReflectionSuffixOp::jit_prepare(JK& jk) {
    add_jit_define(
        jk,
        "ABI",
        JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PreciseReflectionSuffixOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_reflection_suffix_f32_v1(
        xyz->ptr<float32>(),
        camera_center->ptr<float32>(),
        normal_raw3->ptr<float32>(),
        static_cast<uint64_t>(xyz->shape[0]),
        output->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseReflectionSuffixOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
