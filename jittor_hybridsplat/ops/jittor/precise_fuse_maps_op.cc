#include "precise_fuse_maps_op.h"
#include "hybridsplat_rasterizer_c_api.h"
namespace jittor {
#ifndef JIT
PreciseFuseMapsOp::PreciseFuseMapsOp(Var* base_rgb, Var* reflection_rgb, Var* specular) : base_rgb(base_rgb), reflection_rgb(reflection_rgb), specular(specular) {
    flags.set(NodeFlags::_cpu, 0); flags.set(NodeFlags::_cuda, 1);
    CHECK(base_rgb->dtype() == ns_float32 && reflection_rgb->dtype() == ns_float32 && specular->dtype() == ns_float32) << "precise_fuse_maps inputs must be FP32";
    CHECK(base_rgb->shape == reflection_rgb->shape) << "precise_fuse_maps RGB shapes must match";
    CHECK(base_rgb->shape.size() == 3 && base_rgb->shape[0] == 1 && base_rgb->shape[2] == 3) << "precise_fuse_maps RGB inputs must have shape [1,P,3]";
    CHECK(specular->shape.size() == 3 && specular->shape[0] == 1 && specular->shape[1] == base_rgb->shape[1] && specular->shape[2] == 1) << "precise_fuse_maps specular must have shape [1,P,1]";
    diffuse = create_output(base_rgb->shape, ns_float32);
    final_rgb = create_output(base_rgb->shape, ns_float32);
    stored_reflection = create_output(base_rgb->shape, ns_float32);
}
void PreciseFuseMapsOp::jit_prepare(JK& jk) { add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION)); }
#else
void PreciseFuseMapsOp::jit_run() {
#ifdef JIT_cuda
    char error_text[512] = {};
    const int32_t status = hs_rasterizer_precise_fuse_maps_f32_v1(base_rgb->ptr<float32>(), reflection_rgb->ptr<float32>(), specular->ptr<float32>(), diffuse->ptr<float32>(), final_rgb->ptr<float32>(), stored_reflection->ptr<float32>(), static_cast<uint64_t>(base_rgb->shape[1]), error_text, sizeof(error_text));
    CHECK(status == HS_STATUS_OK) << "HybridSplat native status " << status << ": " << error_text;
#else
    CHECK(false) << "PreciseFuseMapsOp requires CUDA";
#endif
}
#endif
}
