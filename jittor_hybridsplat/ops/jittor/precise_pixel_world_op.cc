#include "precise_pixel_world_op.h"

#include "hybridsplat_rasterizer_c_api.h"

#include <limits>

namespace jittor {

#ifndef JIT

PrecisePixelWorldOp::PrecisePixelWorldOp(Var* rotation, Var* vectors, int64 handle_value) : rotation(rotation), vectors(vectors), handle_value(handle_value) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(rotation->dtype() == ns_float32)
        << "precise_pixel_world rotation must be FP32";
    CHECK(vectors->dtype() == ns_float32)
        << "precise_pixel_world vectors must be FP32";
    CHECK(
        rotation->shape.size() == 2 &&
        rotation->shape[0] == 3 &&
        rotation->shape[1] == 3)
        << "precise_pixel_world rotation must have shape (3,3)";
    CHECK(
        vectors->shape.size() >= 1 &&
        vectors->shape[vectors->shape.size() - 1] == 3)
        << "precise_pixel_world vectors must have final extent 3";
    CHECK(vectors->num / 3 <= std::numeric_limits<int>::max())
        << "precise_pixel_world vector count exceeds cuBLAS int range";
    output = create_output(vectors->shape, ns_float32);
}

void PrecisePixelWorldOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void PrecisePixelWorldOp::jit_run() {
#ifdef JIT_cuda
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_pixel_world_f32_v2(
        handle,
        rotation->ptr<float32>(),
        vectors->ptr<float32>(),
        static_cast<uint64_t>(vectors->num / 3),
        output->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PrecisePixelWorldOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
