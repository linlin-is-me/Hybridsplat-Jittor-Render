#include "rasterize_forward_op.h"

#include "hybridsplat_rasterizer_c_api.h"

#include <cstdint>
#include <limits>

namespace jittor {

#ifndef JIT

RasterizeForwardOp::RasterizeForwardOp(
    Var* background,
    Var* means3d,
    Var* colors,
    Var* opacities,
    Var* scales,
    Var* rotations,
    Var* viewmatrix,
    Var* projmatrix,
    Var* camera_position,
    int64 width,
    int64 height,
    float64 scale_modifier,
    float64 tan_fovx,
    float64 tan_fovy,
    int64 prefiltered,
    int64 handle_value)
    : background(background),
      means3d(means3d),
      colors(colors),
      opacities(opacities),
      scales(scales),
      rotations(rotations),
      viewmatrix(viewmatrix),
      projmatrix(projmatrix),
      camera_position(camera_position),
      width(width),
      height(height),
      scale_modifier(scale_modifier),
      tan_fovx(tan_fovx),
      tan_fovy(tan_fovy),
      prefiltered(prefiltered),
      handle_value(handle_value) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    for (Var* value : {
             background,
             means3d,
             colors,
             opacities,
             scales,
             rotations,
             viewmatrix,
             projmatrix,
             camera_position}) {
        CHECK(value->dtype() == ns_float32)
            << "rasterize_forward inputs must be FP32";
    }
    const int64 count = means3d->shape[0];
    CHECK(means3d->shape.size() == 2 && means3d->shape[0] == count && means3d->shape[1] == 3);
    CHECK(colors->shape.size() == 2 && colors->shape[0] == count && colors->shape[1] == 5);
    CHECK(opacities->shape.size() == 2 && opacities->shape[0] == count && opacities->shape[1] == 1);
    CHECK(scales->shape.size() == 2 && scales->shape[0] == count && scales->shape[1] == 2);
    CHECK(rotations->shape.size() == 2 && rotations->shape[0] == count && rotations->shape[1] == 4);
    CHECK(background->shape.size() == 1 && background->shape[0] == 5);
    CHECK(viewmatrix->shape.size() == 2 && viewmatrix->shape[0] == 4 && viewmatrix->shape[1] == 4);
    CHECK(projmatrix->shape.size() == 2 && projmatrix->shape[0] == 4 && projmatrix->shape[1] == 4);
    CHECK(camera_position->shape.size() == 1 && camera_position->shape[0] == 3);
    CHECK(count <= std::numeric_limits<int>::max());
    CHECK(width > 0 && width <= std::numeric_limits<int>::max());
    CHECK(height > 0 && height <= std::numeric_limits<int>::max());
    CHECK(prefiltered == 0 || prefiltered == 1);
    CHECK(handle_value != 0);
    color = create_output({5, height, width}, ns_float32);
    auxiliary = create_output({7, height, width}, ns_float32);
    weight = create_output({count, 1}, ns_float32);
    radii = create_output({count}, ns_int32);
    rendered = create_output({1}, ns_int32);
}

void RasterizeForwardOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void RasterizeForwardOp::jit_run() {
#ifdef JIT_cuda
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    char error_text[1024] = {};
    const int32_t status = hs_rasterizer_forward_f32_v1(
        handle,
        static_cast<uint64_t>(means3d->shape[0]),
        background->ptr<float32>(),
        static_cast<int32_t>(width),
        static_cast<int32_t>(height),
        means3d->ptr<float32>(),
        colors->ptr<float32>(),
        opacities->ptr<float32>(),
        scales->ptr<float32>(),
        static_cast<float>(scale_modifier),
        rotations->ptr<float32>(),
        viewmatrix->ptr<float32>(),
        projmatrix->ptr<float32>(),
        camera_position->ptr<float32>(),
        static_cast<float>(tan_fovx),
        static_cast<float>(tan_fovy),
        static_cast<int32_t>(prefiltered),
        color->ptr<float32>(),
        auxiliary->ptr<float32>(),
        weight->ptr<float32>(),
        radii->ptr<int32>(),
        rendered->ptr<int32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat rasterizer status " << status << ": " << error_text;
#else
    CHECK(false) << "RasterizeForwardOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
