from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from .math import (
    _nan_to_num_like_pytorch,
    fuse_maps,
    gaussian_disks,
    pad_background,
    postprocess_raster,
    prepare_reflection_trace,
)
from .ops.optix import trace_optix
from .ops.rasterizer import RasterizerOutputs, rasterize_forward
from .ops.flashgs import FlashgsOutputs, flashgs_forward


@dataclass
class PreparedGaussian:
    point_count: int
    active_sh_degree_value: int
    xyz: Any
    features: Any
    scaling: Any
    rotation: Any
    rotation_raw: Any
    opacity: Any
    specular: Any
    roughness: Any

    def close(self) -> None:
        for name in (
            "xyz",
            "features",
            "scaling",
            "rotation",
            "rotation_raw",
            "opacity",
            "specular",
            "roughness",
        ):
            setattr(self, name, None)


@dataclass
class PreparedFlashgsGaussian:
    point_count: int
    active_sh_degree_value: int
    xyz: Any
    shs: Any
    opacity: Any
    scaling: Any
    rotation: Any
    covariance: Any
    specular: Any
    roughness: Any

    def close(self) -> None:
        for name in (
            "xyz",
            "shs",
            "opacity",
            "scaling",
            "rotation",
            "covariance",
            "specular",
            "roughness",
        ):
            setattr(self, name, None)


def prepare_gaussian(group, *, exact_trace_activation: bool = False) -> PreparedGaussian:
    if exact_trace_activation:
        from .ops.precise_activation import precise_normalize4, precise_sigmoid

        rotation = precise_normalize4(group.rotation_raw)
        opacity = precise_sigmoid(group.opacity_raw)
        specular = precise_sigmoid(group.specular_raw)
        roughness = precise_sigmoid(group.roughness_raw)
    else:
        rotation = group.rotation
        opacity = group.opacity
        specular = group.specular
        roughness = group.roughness
    values = PreparedGaussian(
        point_count=group.point_count,
        active_sh_degree_value=group.active_sh_degree_value,
        xyz=group.xyz,
        features=group.features,
        scaling=group.scaling,
        rotation=rotation,
        rotation_raw=group.rotation_raw,
        opacity=opacity,
        specular=specular,
        roughness=roughness,
    )
    for value in (
        values.xyz,
        values.features,
        values.scaling,
        values.rotation,
        values.rotation_raw,
        values.opacity,
        values.specular,
        values.roughness,
    ):
        value.stop_grad()
    return values


def prepare_flashgs_gaussian(group) -> PreparedFlashgsGaussian:
    if not hasattr(group, "flashgs_covariance"):
        raise ValueError("checkpoint was not prepared for FlashGS")
    values = PreparedFlashgsGaussian(
        point_count=group.point_count,
        active_sh_degree_value=group.active_sh_degree_value,
        xyz=group.xyz_raw,
        shs=group.features.reshape((group.point_count, 48)),
        opacity=group.flashgs_opacity,
        scaling=group.flashgs_scaling,
        rotation=group.flashgs_rotation,
        covariance=group.flashgs_covariance,
        specular=group.flashgs_specular,
        roughness=group.flashgs_roughness,
    )
    for value in (
        values.xyz,
        values.shs,
        values.opacity,
        values.scaling,
        values.rotation,
        values.covariance,
        values.specular,
        values.roughness,
    ):
        value.stop_grad()
    return values


def _flashgs_rasterize(
    gaussian: PreparedFlashgsGaussian,
    colors,
    camera,
    context,
    *,
    use_sh: bool,
) -> FlashgsOutputs:
    return flashgs_forward(
        gaussian.xyz,
        gaussian.shs,
        colors,
        gaussian.opacity,
        gaussian.covariance,
        gaussian.scaling,
        gaussian.rotation,
        gaussian.specular,
        gaussian.roughness,
        width=camera.image_width,
        height=camera.image_height,
        use_sh=use_sh,
        camera_position=camera.position.reshape(-1).tolist(),
        camera_rotation=camera.rotation.reshape(-1).tolist(),
        focal_x=camera.focal_x,
        focal_y=camera.focal_y,
        z_far=camera.zfar,
        z_near=camera.znear,
        background=(0.0, 0.0, 0.0),
        scale_modifier=1.0,
        initial_pair_capacity=2_000_000,
        context=context,
    )


def _flashgs_postprocess(raster: FlashgsOutputs, viewpoint_camera):
    import jittor as jt

    from .ops.precise_depth import precise_depth_divide

    height, width = int(raster.alpha.shape[1]), int(raster.alpha.shape[2])
    pixels = height * width
    alpha = raster.alpha.transpose([1, 2, 0]).reshape((1, pixels, 1))
    depth = _nan_to_num_like_pytorch(
        precise_depth_divide(raster.depth, raster.alpha)
    ).transpose([1, 2, 0]).reshape((1, pixels, 1))
    normal_hwc = raster.normal.transpose([1, 2, 0])
    normal_hwc = normal_hwc @ viewpoint_camera.world_view_transform[:3, :3].transpose(0, 1)
    normal = normal_hwc.reshape((1, pixels, 3))
    rgb = raster.color.transpose([1, 2, 0]).reshape((1, pixels, 3))
    specular = raster.other[0:1].transpose([1, 2, 0]).reshape((1, pixels, 1))
    zeros3 = jt.zeros((1, pixels, 3), dtype="float32")
    zeros1 = jt.zeros((1, pixels, 1), dtype="float32")
    return {
        "rgb_map": rgb,
        "acc_map": alpha,
        "dpt_map": depth,
        "norm_map": normal,
        "surf_norm_map": zeros3,
        "dist_map": zeros1,
        "spec_map": specular,
    }


def _rasterize(
    gaussian: PreparedGaussian,
    colors,
    background,
    camera,
    context,
) -> RasterizerOutputs:
    return rasterize_forward(
        pad_background(background, channels=5),
        gaussian.xyz,
        colors,
        gaussian.opacity,
        gaussian.scaling,
        gaussian.rotation,
        camera.world_view_transform,
        camera.full_proj_transform,
        camera.camera_center,
        width=camera.image_width,
        height=camera.image_height,
        scale_modifier=1.0,
        tan_fovx=camera.tanfovx,
        tan_fovy=camera.tanfovy,
        prefiltered=False,
        context=context,
    )


def render_hybridsplat_view(
    *,
    base: PreparedGaussian,
    reflection: PreparedGaussian,
    background,
    reflection_background,
    camera,
    context,
    synchronize_stages: bool = False,
    stage_timings: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import jittor as jt

    from .ops.precise_direction import precise_direction
    from .ops.precise_sh import precise_sh_to_rgb

    started = time.perf_counter()
    previous = started

    def mark(name: str) -> None:
        nonlocal previous
        if stage_timings is None:
            return
        jt.sync_all(True)
        current = time.perf_counter()
        stage_timings[name] = current - previous
        previous = current

    base_direction = precise_direction(base.xyz, camera.camera_center)
    base_rgb = precise_sh_to_rgb(
        base.features.transpose(1, 2),
        base_direction,
        base.active_sh_degree_value,
    )
    base_colors = jt.concat((base_rgb, base.specular, base.roughness), dim=-1)
    base_raster = _rasterize(base, base_colors, background, camera, context)
    if synchronize_stages:
        jt.sync_all(True)
    mark("base_raster_seconds")
    base_maps = postprocess_raster(
        camera,
        base_raster.color,
        base_raster.auxiliary,
        context=context,
    )

    trace_inputs = prepare_reflection_trace(reflection, camera.camera_center)
    others = jt.concat((reflection.specular, reflection.roughness), dim=-1)
    mark("reflection_prepare_seconds")
    trace = trace_optix(
        trace_inputs["trace_ray_o"],
        trace_inputs["trace_ray_d"],
        reflection_background,
        reflection.xyz,
        reflection.features,
        others,
        reflection.opacity,
        reflection.scaling,
        reflection.rotation,
        camera.world_view_transform,
        camera.full_proj_transform,
        camera.camera_center,
        degree=reflection.active_sh_degree_value,
        scale_modifier=1.0,
        start_from_first=False,
        max_trace_depth=0,
        specular_threshold=0.0,
        use_others=False,
        context=context,
    )
    if synchronize_stages:
        jt.sync_all(True)
    mark("optix_seconds")

    reflection_colors = jt.concat(
        (trace.rgb, reflection.specular, reflection.roughness), dim=-1
    )
    reflection_raster = _rasterize(
        reflection,
        reflection_colors,
        reflection_background,
        camera,
        context,
    )
    if synchronize_stages:
        jt.sync_all(True)
    mark("reflection_raster_seconds")
    reflection_rgb = reflection_raster.color[:3].transpose([1, 2, 0]).reshape(
        (1, -1, 3)
    )
    fused = fuse_maps(base_maps["rgb_map"], reflection_rgb, base_maps["spec_map"])
    outputs = {
        "rgb_map": fused["rgb_map"],
        "acc_map": base_maps["acc_map"],
        "dpt_map": base_maps["dpt_map"],
        "norm_map": base_maps["norm_map"],
        "surf_norm_map": base_maps["surf_norm_map"],
        "spec_map": base_maps["spec_map"],
        "dif_rgb_map": fused["dif_rgb_map"],
        "ref_rgb_map": fused["ref_rgb_map"],
    }
    mark("fusion_seconds")
    intermediates = {
        "base_raster": base_raster,
        "trace": trace,
        "reflection_raster": reflection_raster,
        "trace_inputs": trace_inputs,
        "stage_timings": stage_timings,
    }
    return outputs, intermediates


def render_hybridsplat_flashgs_view(
    *,
    base: PreparedFlashgsGaussian,
    reflection: PreparedFlashgsGaussian,
    trace_reflection: PreparedGaussian,
    reflection_background,
    flashgs_camera,
    viewpoint_camera,
    context,
    synchronize_stages: bool = False,
    stage_timings: dict[str, float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import jittor as jt

    previous = time.perf_counter()

    def mark(name: str) -> None:
        nonlocal previous
        if stage_timings is None:
            return
        jt.sync_all(True)
        current = time.perf_counter()
        stage_timings[name] = current - previous
        previous = current

    base_colors = jt.zeros((base.point_count, 3), dtype="float32").stop_grad()
    base_raster = _flashgs_rasterize(
        base, base_colors, flashgs_camera, context, use_sh=True
    )
    if synchronize_stages:
        jt.sync_all(True)
    mark("base_raster_seconds")
    base_maps = _flashgs_postprocess(base_raster, viewpoint_camera)

    trace_inputs = prepare_reflection_trace(
        trace_reflection, viewpoint_camera.camera_center
    )
    others = jt.concat(
        (trace_reflection.specular, trace_reflection.roughness), dim=-1
    )
    mark("reflection_prepare_seconds")
    trace = trace_optix(
        trace_inputs["trace_ray_o"],
        trace_inputs["trace_ray_d"],
        reflection_background,
        trace_reflection.xyz,
        trace_reflection.features,
        others,
        trace_reflection.opacity,
        trace_reflection.scaling,
        trace_reflection.rotation,
        viewpoint_camera.world_view_transform,
        viewpoint_camera.full_proj_transform,
        viewpoint_camera.camera_center,
        degree=trace_reflection.active_sh_degree_value,
        scale_modifier=1.0,
        start_from_first=False,
        max_trace_depth=0,
        specular_threshold=0.0,
        use_others=False,
        context=context,
    )
    if synchronize_stages:
        jt.sync_all(True)
    mark("optix_seconds")

    reflection_raster = _flashgs_rasterize(
        reflection,
        trace.rgb,
        flashgs_camera,
        context,
        use_sh=False,
    )
    if synchronize_stages:
        jt.sync_all(True)
    mark("reflection_raster_seconds")
    reflection_rgb = reflection_raster.color.transpose([1, 2, 0]).reshape(
        (1, -1, 3)
    )
    from .ops.precise_fuse import precise_fuse_maps

    fused = precise_fuse_maps(
        base_maps["rgb_map"], reflection_rgb, base_maps["spec_map"]
    )
    outputs = {
        "rgb_map": fused["rgb_map"],
        "acc_map": base_maps["acc_map"],
        "dpt_map": base_maps["dpt_map"],
        "norm_map": base_maps["norm_map"],
        "surf_norm_map": base_maps["surf_norm_map"],
        "spec_map": base_maps["spec_map"],
        "dif_rgb_map": fused["dif_rgb_map"],
        "ref_rgb_map": fused["ref_rgb_map"],
    }
    mark("fusion_seconds")
    return outputs, {
        "base_raster": base_raster,
        "base_maps": base_maps,
        "trace": trace,
        "reflection_raster": reflection_raster,
        "trace_inputs": trace_inputs,
        "stage_timings": stage_timings,
    }
