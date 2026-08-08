#include "optix_trace_op.h"

#include "hybridsplat_rasterizer_c_api.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace jittor {

#ifndef JIT

OptixTraceOp::OptixTraceOp(
    Var* ray_origins,
    Var* ray_directions,
    Var* background,
    Var* means3d,
    Var* shs,
    Var* others,
    Var* opacities,
    Var* scales,
    Var* rotations,
    Var* viewmatrix,
    Var* projmatrix,
    Var* camera_position,
    int64 degree,
    float64 scale_modifier,
    int64 start_from_first,
    int64 max_trace_depth,
    float64 specular_threshold,
    int64 use_others,
    int64 handle_value)
    : ray_origins(ray_origins),
      ray_directions(ray_directions),
      background(background),
      means3d(means3d),
      shs(shs),
      others(others),
      opacities(opacities),
      scales(scales),
      rotations(rotations),
      viewmatrix(viewmatrix),
      projmatrix(projmatrix),
      camera_position(camera_position),
      degree(degree),
      scale_modifier(scale_modifier),
      start_from_first(start_from_first),
      max_trace_depth(max_trace_depth),
      specular_threshold(specular_threshold),
      use_others(use_others),
      handle_value(handle_value) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    for (Var* value : {
             ray_origins, ray_directions, background, means3d, shs, others,
             opacities, scales, rotations, viewmatrix, projmatrix,
             camera_position}) {
        CHECK(value->dtype() == ns_float32)
            << "OptiX trace inputs must be FP32";
    }
    const int64 count = ray_origins->shape[0];
    CHECK(count > 0 && count <= std::numeric_limits<int>::max());
    CHECK(ray_origins->shape.size() == 2 && ray_origins->shape[1] == 3);
    CHECK(ray_directions->shape.size() == 2 && ray_directions->shape[0] == count && ray_directions->shape[1] == 3);
    CHECK(background->shape.size() == 1 && background->shape[0] == 3);
    CHECK(means3d->shape.size() == 2 && means3d->shape[0] == count && means3d->shape[1] == 3);
    CHECK(shs->shape.size() == 3 && shs->shape[0] == count && shs->shape[2] == 3);
    CHECK(shs->shape[1] > 0 && shs->shape[1] <= std::numeric_limits<uint32_t>::max());
    CHECK(others->shape.size() == 2 && others->shape[0] == count && others->shape[1] == 2);
    CHECK(opacities->shape.size() == 2 && opacities->shape[0] == count && opacities->shape[1] == 1);
    CHECK(scales->shape.size() == 2 && scales->shape[0] == count && scales->shape[1] == 2);
    CHECK(rotations->shape.size() == 2 && rotations->shape[0] == count && rotations->shape[1] == 4);
    CHECK(viewmatrix->shape.size() == 2 && viewmatrix->shape[0] == 4 && viewmatrix->shape[1] == 4);
    CHECK(projmatrix->shape.size() == 2 && projmatrix->shape[0] == 4 && projmatrix->shape[1] == 4);
    CHECK(camera_position->shape.size() == 1 && camera_position->shape[0] == 3);
    CHECK(degree >= 0 && degree <= 3);
    CHECK(std::isfinite(scale_modifier) && scale_modifier > 0.0);
    CHECK(start_from_first == 0 || start_from_first == 1);
    CHECK(max_trace_depth >= 0 && max_trace_depth <= 1);
    CHECK(use_others == 0 || use_others == 1);
    CHECK(std::isfinite(specular_threshold));
    CHECK(handle_value != 0);
    rgb = create_output({count, 3}, ns_float32);
    depth = create_output({count, 1}, ns_float32);
    acc = create_output({count, 1}, ns_float32);
    normal = create_output({count, 3}, ns_float32);
    dist = create_output({count, 1}, ns_float32);
    auxiliary = create_output({count, 2}, ns_float32);
    mid = create_output({count, 32}, ns_float32);
    wet = create_output({count}, ns_float32);
}

void OptixTraceOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void OptixTraceOp::jit_run() {
#ifdef JIT_cuda
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    char error_text[1024] = {};
    const int32_t status = hs_optix_trace_f32_v1(
        handle,
        static_cast<uint64_t>(ray_origins->shape[0]),
        ray_origins->ptr<float32>(),
        ray_directions->ptr<float32>(),
        background->ptr<float32>(),
        means3d->ptr<float32>(),
        shs->ptr<float32>(),
        static_cast<uint32_t>(shs->shape[1]),
        static_cast<uint32_t>(degree),
        others->ptr<float32>(),
        opacities->ptr<float32>(),
        scales->ptr<float32>(),
        static_cast<float>(scale_modifier),
        rotations->ptr<float32>(),
        viewmatrix->ptr<float32>(),
        projmatrix->ptr<float32>(),
        camera_position->ptr<float32>(),
        static_cast<int32_t>(start_from_first),
        static_cast<int32_t>(max_trace_depth),
        static_cast<float>(specular_threshold),
        static_cast<int32_t>(use_others),
        rgb->ptr<float32>(),
        depth->ptr<float32>(),
        acc->ptr<float32>(),
        normal->ptr<float32>(),
        dist->ptr<float32>(),
        auxiliary->ptr<float32>(),
        mid->ptr<float32>(),
        wet->ptr<float32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat OptiX trace status " << status << ": " << error_text;
#else
    CHECK(false) << "OptixTraceOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
