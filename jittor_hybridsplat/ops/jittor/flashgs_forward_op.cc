#include "flashgs_forward_op.h"

#include "hybridsplat_rasterizer_c_api.h"

#include <cstdint>
#include <limits>

namespace jittor {

#ifndef JIT

FlashgsForwardOp::FlashgsForwardOp(
    Var* positions,
    Var* shs,
    Var* colors,
    Var* opacities,
    Var* covariance,
    Var* scales,
    Var* rotations,
    Var* specular,
    Var* roughness,
    int64 width,
    int64 height,
    int64 use_sh,
    int64 initial_pair_capacity,
    int64 handle_value,
    float64 scale_modifier,
    float64 focal_x,
    float64 focal_y,
    float64 z_far,
    float64 z_near,
    float64 camera_x,
    float64 camera_y,
    float64 camera_z,
    float64 rotation_0,
    float64 rotation_1,
    float64 rotation_2,
    float64 rotation_3,
    float64 rotation_4,
    float64 rotation_5,
    float64 rotation_6,
    float64 rotation_7,
    float64 rotation_8,
    float64 background_r,
    float64 background_g,
    float64 background_b)
    : positions(positions),
      shs(shs),
      colors(colors),
      opacities(opacities),
      covariance(covariance),
      scales(scales),
      rotations(rotations),
      specular(specular),
      roughness(roughness),
      width(width),
      height(height),
      use_sh(use_sh),
      initial_pair_capacity(initial_pair_capacity),
      handle_value(handle_value),
      scale_modifier(scale_modifier),
      focal_x(focal_x),
      focal_y(focal_y),
      z_far(z_far),
      z_near(z_near),
      camera_x(camera_x),
      camera_y(camera_y),
      camera_z(camera_z),
      rotation_0(rotation_0),
      rotation_1(rotation_1),
      rotation_2(rotation_2),
      rotation_3(rotation_3),
      rotation_4(rotation_4),
      rotation_5(rotation_5),
      rotation_6(rotation_6),
      rotation_7(rotation_7),
      rotation_8(rotation_8),
      background_r(background_r),
      background_g(background_g),
      background_b(background_b) {
    flags.set(NodeFlags::_cpu, 0);
    flags.set(NodeFlags::_cuda, 1);
    for (Var* value : {
             positions,
             shs,
             colors,
             opacities,
             covariance,
             scales,
             rotations,
             specular,
             roughness}) {
        CHECK(value->dtype() == ns_float32) << "FlashGS inputs must be FP32";
    }
    const int64 count = positions->shape[0];
    CHECK(positions->shape.size() == 2 && positions->shape[1] == 3);
    CHECK(shs->shape.size() == 2 && shs->shape[0] == count && shs->shape[1] == 48);
    CHECK(colors->shape.size() == 2 && colors->shape[0] == count && colors->shape[1] == 3);
    CHECK(opacities->shape.size() == 2 && opacities->shape[0] == count && opacities->shape[1] == 1);
    CHECK(covariance->shape.size() == 2 && covariance->shape[0] == count && covariance->shape[1] == 6);
    CHECK(scales->shape.size() == 2 && scales->shape[0] == count && scales->shape[1] == 2);
    CHECK(rotations->shape.size() == 2 && rotations->shape[0] == count && rotations->shape[1] == 4);
    CHECK(specular->shape.size() == 2 && specular->shape[0] == count && specular->shape[1] == 1);
    CHECK(roughness->shape.size() == 2 && roughness->shape[0] == count && roughness->shape[1] == 1);
    CHECK(count <= std::numeric_limits<int>::max());
    CHECK(width > 0 && width <= std::numeric_limits<int>::max());
    CHECK(height > 0 && height <= std::numeric_limits<int>::max());
    CHECK(use_sh == 0 || use_sh == 1);
    CHECK(initial_pair_capacity > 0 && initial_pair_capacity <= std::numeric_limits<int>::max());
    CHECK(handle_value != 0);
    color = create_output({3, height, width}, ns_float32);
    other = create_output({2, height, width}, ns_float32);
    depth = create_output({1, height, width}, ns_float32);
    alpha = create_output({1, height, width}, ns_float32);
    normal = create_output({3, height, width}, ns_float32);
    median_depth = create_output({1, height, width}, ns_float32);
    distortion = create_output({1, height, width}, ns_float32);
    rendered = create_output({1}, ns_int32);
}

void FlashgsForwardOp::jit_prepare(JK& jk) {
    add_jit_define(jk, "ABI", JK::hex1(HS_RASTERIZER_ABI_VERSION));
}

#else

void FlashgsForwardOp::jit_run() {
#ifdef JIT_cuda
    auto handle = reinterpret_cast<HsRasterizerHandle>(
        static_cast<uintptr_t>(handle_value));
    const float camera_position[3] = {
        static_cast<float>(camera_x),
        static_cast<float>(camera_y),
        static_cast<float>(camera_z),
    };
    const float camera_rotation[9] = {
        static_cast<float>(rotation_0),
        static_cast<float>(rotation_1),
        static_cast<float>(rotation_2),
        static_cast<float>(rotation_3),
        static_cast<float>(rotation_4),
        static_cast<float>(rotation_5),
        static_cast<float>(rotation_6),
        static_cast<float>(rotation_7),
        static_cast<float>(rotation_8),
    };
    const float background[3] = {
        static_cast<float>(background_r),
        static_cast<float>(background_g),
        static_cast<float>(background_b),
    };
    char error_text[1024] = {};
    const int32_t status = hs_flashgs_forward_f32_v1(
        handle,
        static_cast<uint64_t>(positions->shape[0]),
        static_cast<int32_t>(width),
        static_cast<int32_t>(height),
        positions->ptr<float32>(),
        use_sh != 0 ? shs->ptr<float32>() : nullptr,
        use_sh == 0 ? colors->ptr<float32>() : nullptr,
        static_cast<int32_t>(use_sh),
        opacities->ptr<float32>(),
        covariance->ptr<float32>(),
        scales->ptr<float32>(),
        static_cast<float>(scale_modifier),
        rotations->ptr<float32>(),
        specular->ptr<float32>(),
        roughness->ptr<float32>(),
        camera_position,
        camera_rotation,
        static_cast<float>(focal_x),
        static_cast<float>(focal_y),
        static_cast<float>(z_far),
        static_cast<float>(z_near),
        background,
        static_cast<uint64_t>(initial_pair_capacity),
        color->ptr<float32>(),
        other->ptr<float32>(),
        depth->ptr<float32>(),
        alpha->ptr<float32>(),
        normal->ptr<float32>(),
        median_depth->ptr<float32>(),
        distortion->ptr<float32>(),
        rendered->ptr<int32>(),
        error_text,
        sizeof(error_text));
    CHECK(status == HS_STATUS_OK)
        << "HybridSplat FlashGS status " << status << ": " << error_text;
#else
    CHECK(false) << "FlashgsForwardOp requires CUDA";
#endif
}

#endif

}  // namespace jittor
