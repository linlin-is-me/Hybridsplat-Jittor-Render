#include "optix_build_scene_op.h"

#include "hybridsplat_rasterizer_c_api.h"

#include <cstdint>
#include <limits>

namespace jittor {

#ifndef JIT

OptixBuildSceneOp::OptixBuildSceneOp(
    Var* vertices,
    Var* triangles,
    int64 scene_signature,
    int64 rebuild,
    int64 handle_value)
    : vertices(vertices),
      triangles(triangles),
      scene_signature(scene_signature),
      rebuild(rebuild),
      handle_value(handle_value) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    CHECK(vertices->dtype() == ns_float32);
    CHECK(triangles->dtype() == ns_int32);
    CHECK(vertices->shape.size() == 2 && vertices->shape[1] == 3);
    CHECK(triangles->shape.size() == 2 && triangles->shape[1] == 3);
    CHECK(vertices->shape[0] > 0 && triangles->shape[0] > 0);
    CHECK(vertices->shape[0] <= std::numeric_limits<int>::max());
    CHECK(triangles->shape[0] <= std::numeric_limits<int>::max());
    CHECK(scene_signature > 0);
    CHECK(rebuild == 0 || rebuild == 1);
    CHECK(handle_value != 0);
    status = create_output({1}, ns_int32);
}

void OptixBuildSceneOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void OptixBuildSceneOp::jit_run() {
#ifdef JIT_cuda
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    char error_text[1024] = {};
    const int32_t result = hs_optix_build_scene_v1(
        handle,
        vertices->ptr<float32>(),
        static_cast<uint64_t>(vertices->shape[0]),
        triangles->ptr<int32>(),
        static_cast<uint64_t>(triangles->shape[0]),
        static_cast<uint64_t>(scene_signature),
        static_cast<int32_t>(rebuild),
        status->ptr<int32>(),
        error_text,
        sizeof(error_text));
    CHECK(result == HS_STATUS_OK)
        << "HybridSplat OptiX build status " << result << ": " << error_text;
#else
    CHECK(false) << "OptixBuildSceneOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
