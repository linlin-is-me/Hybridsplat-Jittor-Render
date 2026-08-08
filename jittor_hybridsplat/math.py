from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def _jt():
    import jittor as jt

    return jt


def safe_normalize(x, eps: float = 1e-20, *, clamp: bool = True):
    norm = (x * x).sum(dim=-1, keepdims=True).sqrt()
    norm = norm.maximum(eps) if clamp else norm + eps
    return x / norm


def build_rotation(quaternion_wxyz):
    q = quaternion_wxyz / (
        (quaternion_wxyz * quaternion_wxyz).sum(dim=-1, keepdims=True).sqrt()
    )
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    jt = _jt()
    row0 = jt.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ),
        dim=-1,
    )
    row1 = jt.stack(
        (
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ),
        dim=-1,
    )
    row2 = jt.stack(
        (
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    )
    return jt.stack((row0, row1, row2), dim=-2)


def _diagonal3(diagonal):
    jt = _jt()
    zero = diagonal[..., 0] * 0.0
    row0 = jt.stack((diagonal[..., 0], zero, zero), dim=-1)
    row1 = jt.stack((zero, diagonal[..., 1], zero), dim=-1)
    row2 = jt.stack((zero, zero, diagonal[..., 2]), dim=-1)
    return jt.stack((row0, row1, row2), dim=-2)


def build_scaling_rotation(scales3, quaternion_wxyz):
    return build_rotation(quaternion_wxyz) @ _diagonal3(scales3)


def build_inverse_scaling_rotation(scales3, quaternion_wxyz):
    return _diagonal3(1.0 / scales3) @ build_rotation(quaternion_wxyz).transpose(-2, -1)


def build_covariance(center, scales2, quaternion_wxyz, scaling_modifier: float = 1.0):
    jt = _jt()
    ones = jt.ones_like(scales2[..., :1])
    scales3 = jt.concat((scales2 * scaling_modifier, ones), dim=-1)
    linear = build_scaling_rotation(scales3, quaternion_wxyz).transpose(-2, -1)
    upper = jt.concat((linear, jt.zeros_like(linear[..., :1])), dim=-1)
    bottom = jt.concat((center, ones), dim=-1).unsqueeze(-2)
    return jt.concat((upper, bottom), dim=-2)


def build_inverse_covariance(
    center,
    scales2,
    quaternion_wxyz,
    scaling_modifier: float = 1.0,
):
    jt = _jt()
    ones = jt.ones_like(scales2[..., :1])
    scales3 = jt.concat((scales2 * scaling_modifier, ones), dim=-1)
    linear = build_inverse_scaling_rotation(scales3, quaternion_wxyz)
    translation = -(linear @ center.unsqueeze(-1))
    upper = jt.concat((linear, translation), dim=-1)
    zero = jt.zeros_like(center[..., :1])
    bottom = jt.concat((zero, zero, zero, ones), dim=-1).unsqueeze(-2)
    return jt.concat((upper, bottom), dim=-2)


def flip_align_view(normal, view_direction) -> Tuple[object, object]:
    jt = _jt()
    dot = (normal * -view_direction).sum(dim=-1, keepdims=True)
    non_flip = dot >= 0
    sign = jt.where(non_flip, jt.ones_like(dot), -jt.ones_like(dot))
    return normal * sign, non_flip


def gaussian_normals(group, camera_center):
    covariance = build_covariance(
        group.xyz,
        group.scaling,
        group.rotation,
        scaling_modifier=1.0,
    )
    normal_raw = covariance[:, 2, :3]
    normal_raw, positive = flip_align_view(normal_raw, group.xyz - camera_center)
    return safe_normalize(normal_raw, eps=1e-20, clamp=True), positive


def eval_sh(degree: int, sh, directions):
    if degree < 0 or degree > 4:
        raise ValueError(f"unsupported SH degree: {degree}")
    required = (degree + 1) ** 2
    if sh.shape[-1] < required:
        raise ValueError(
            f"SH coefficient count {sh.shape[-1]} is smaller than {required}"
        )
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    c2 = (
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396,
    )
    c3 = (
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435,
    )
    c4 = (
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    )
    result = c0 * sh[..., 0]
    if degree > 0:
        x, y, z = directions[..., 0:1], directions[..., 1:2], directions[..., 2:3]
        result = result - c1 * y * sh[..., 1] + c1 * z * sh[..., 2] - c1 * x * sh[..., 3]
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + c2[0] * xy * sh[..., 4]
                + c2[1] * yz * sh[..., 5]
                + c2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + c2[3] * xz * sh[..., 7]
                + c2[4] * (xx - yy) * sh[..., 8]
            )
            if degree > 2:
                result = (
                    result
                    + c3[0] * y * (3.0 * xx - yy) * sh[..., 9]
                    + c3[1] * xy * z * sh[..., 10]
                    + c3[2] * y * (4.0 * zz - xx - yy) * sh[..., 11]
                    + c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[..., 12]
                    + c3[4] * x * (4.0 * zz - xx - yy) * sh[..., 13]
                    + c3[5] * z * (xx - yy) * sh[..., 14]
                    + c3[6] * x * (xx - 3.0 * yy) * sh[..., 15]
                )
                if degree > 3:
                    result = (
                        result
                        + c4[0] * xy * (xx - yy) * sh[..., 16]
                        + c4[1] * yz * (3.0 * xx - yy) * sh[..., 17]
                        + c4[2] * xy * (7.0 * zz - 1.0) * sh[..., 18]
                        + c4[3] * yz * (7.0 * zz - 3.0) * sh[..., 19]
                        + c4[4] * (zz * (35.0 * zz - 30.0) + 3.0) * sh[..., 20]
                        + c4[5] * xz * (7.0 * zz - 3.0) * sh[..., 21]
                        + c4[6] * (xx - yy) * (7.0 * zz - 1.0) * sh[..., 22]
                        + c4[7] * xz * (xx - 3.0 * yy) * sh[..., 23]
                        + c4[8]
                        * (xx * (xx - 3.0 * yy) - yy * (3.0 * xx - yy))
                        * sh[..., 24]
                    )
    return result


def pad_background(background, channels: int = 5):
    if len(background.shape) != 1:
        raise ValueError(f"background must be one-dimensional, got {background.shape}")
    if background.shape[0] > channels:
        raise ValueError(
            f"background has {background.shape[0]} channels, expected at most {channels}"
        )
    if background.shape[0] == channels:
        return background
    jt = _jt()
    return jt.concat(
        (background, jt.zeros((channels - background.shape[0],), dtype=background.dtype)),
        dim=0,
    )


def prepare_base_raster_inputs(group, camera_center, background) -> Dict[str, object]:
    jt = _jt()
    from .ops.precise_direction import precise_direction
    from .ops.precise_sh import precise_sh_to_rgb

    features = group.features.transpose(1, 2)
    direction = precise_direction(group.xyz, camera_center)
    rgb = precise_sh_to_rgb(
        features,
        direction,
        group.active_sh_degree_value,
    )
    colors = jt.concat((rgb, group.specular, group.roughness), dim=-1)
    return {
        "means3D": group.xyz,
        "opacity": group.opacity,
        "scales": group.scaling,
        "rotations": group.rotation,
        "colors_precomp": colors,
        "background": pad_background(background, channels=5),
    }


def prepare_reflection_trace(group, camera_center) -> Dict[str, object]:
    from .ops.precise_reflection_trace import precise_reflection_trace

    # The native prefix reproduces the profiled TorchScript TensorExpr
    # rotation, including its explicit FMA contractions. Jittor applies the
    # diagonal scale, then the native suffix preserves optix_utils.py's eager
    # FP32 reflection stages.
    return precise_reflection_trace(
        group.xyz,
        group.rotation_raw,
        group.scaling,
        camera_center,
    )


def gaussian_disks(group):
    jt = _jt()
    from .ops.precise_disk_rotation import precise_disk_rotation

    # get_disks makes the first build_cov invocation in the profiled
    # PyTorch path. Its double quaternion normalization and unfused FP32
    # rotation differ from the later TensorExpr-fused normal computation.
    rotation = precise_disk_rotation(group.rotation_raw)
    scales3 = jt.concat(
        (group.scaling, jt.ones_like(group.scaling[..., :1])),
        dim=-1,
    )
    linear = rotation @ _diagonal3(scales3)
    upper = jt.concat((linear, group.xyz.unsqueeze(-1)), dim=-1)
    zero = jt.zeros_like(group.xyz[..., :1])
    one = jt.ones_like(zero)
    bottom = jt.concat((zero, zero, zero, one), dim=-1).unsqueeze(-2)
    transform = jt.concat((upper, bottom), dim=-2)
    transform = jt.concat(
        (
            transform[..., :2],
            jt.zeros_like(transform[..., 2:3]),
            transform[..., 3:4],
        ),
        dim=-1,
    )
    sigma3 = jt.array(
        (
            (-3.0, 3.0, 1.0, 1.0),
            (-3.0, -3.0, 1.0, 1.0),
            (3.0, 3.0, 1.0, 1.0),
            (3.0, -3.0, 1.0, 1.0),
        ),
        dtype="float32",
    )
    transforms = transform.unsqueeze(1).expand((-1, 4, -1, -1))
    coordinates = sigma3.unsqueeze(0).repeat((group.point_count, 1, 1))
    vertices = transforms.reshape((-1, 4, 4)) @ coordinates.reshape(
        (-1, 4, 1)
    )
    vertices = vertices[..., :3, 0]
    indices = jt.arange(group.point_count * 4, dtype="int32").reshape(
        (group.point_count, 4)
    )
    faces = jt.stack((indices[:, :3], indices[:, 1:]), dim=1).reshape((-1, 3))
    return vertices, faces


def fuse_maps(base_rgb, reflection_rgb, specular):
    diffuse = base_rgb * (1.0 - specular)
    stored_reflection = reflection_rgb * specular * 2.0
    # Keep the externally frozen storage identity exact. The PyTorch baseline
    # stores these three maps with this elementwise relationship.
    final = diffuse + 0.5 * stored_reflection
    return {
        "dif_rgb_map": diffuse,
        "rgb_map": final,
        "ref_rgb_map": stored_reflection,
    }


def pixel_reflection_rays(ray_origin, ray_direction, normal_map, depth_map):
    from .ops.precise_camera_rays import precise_pixel_normal

    normal = precise_pixel_normal(normal_map)
    reflection_product = (ray_direction * normal).stop_fuse()
    reflection_dot = reflection_product.sum(
        dim=-1,
        keepdims=True,
    ).stop_fuse()
    reflection_twice = (2.0 * reflection_dot).stop_fuse()
    reflection_scaled = (reflection_twice * normal).stop_fuse()
    reflected_direction = ray_direction - reflection_scaled
    origin_product = (ray_direction * depth_map).stop_fuse()
    reflected_origin = ray_origin + origin_product
    return reflected_origin, reflected_direction


def _nan_to_num_like_pytorch(value):
    jt = _jt()
    result = jt.where(value.isnan(), jt.zeros_like(value), value)
    result = jt.where(result.isposinf(), jt.zeros_like(result), result)
    minimum = jt.ones_like(result) * -3.4028234663852886e38
    return jt.where(result.isneginf(), minimum, result)


def _pytorch_focal_from_fov(length: int, fov: float) -> np.float32:
    """复现 PyTorch 路径在 CPU 上构造焦距时的 FP32 舍入顺序。"""
    fov32 = np.float32(fov)
    half = np.divide(fov32, np.float32(2.0), dtype=np.float32)
    denominator = np.float32(2.0 * math.tan(float(half)))
    return np.divide(
        np.float32(length),
        denominator,
        dtype=np.float32,
    )


def depth_to_normal(camera, depth, *, context=None):
    jt = _jt()
    from .ops.precise_matrix import (
        precise_general_inverse,
        precise_matmul_nt_n3k3,
    )

    height, width = camera.image_height, camera.image_width
    v, u = jt.meshgrid(
        jt.arange(height, dtype="float32"),
        jt.arange(width, dtype="float32"),
    )
    pixels = jt.stack((u, v, jt.ones_like(u)), dim=-1).reshape((-1, 3))
    intrinsics = jt.array(
        np.asarray(
            (
                (
                    _pytorch_focal_from_fov(width, camera.FoVx),
                    np.float32(0.0),
                    np.divide(
                        np.float32(width),
                        np.float32(2.0),
                        dtype=np.float32,
                    ),
                ),
                (
                    np.float32(0.0),
                    _pytorch_focal_from_fov(height, camera.FoVy),
                    np.divide(
                        np.float32(height),
                        np.float32(2.0),
                        dtype=np.float32,
                    ),
                ),
                (
                    np.float32(0.0),
                    np.float32(0.0),
                    np.float32(1.0),
                ),
            ),
            dtype=np.float32,
            order="C",
        ),
        dtype="float32",
    )
    intrinsics.stop_grad()
    inverse_intrinsics = precise_general_inverse(intrinsics, context=context)
    world_to_view = camera.world_view_transform.transpose(0, 1)
    c2w = precise_general_inverse(world_to_view, context=context)
    camera_rays = precise_matmul_nt_n3k3(
        pixels,
        inverse_intrinsics,
        context=context,
    )
    ray_direction = precise_matmul_nt_n3k3(
        camera_rays,
        c2w[:3, :3],
        context=context,
    )
    depth_scaled_rays = (
        depth.reshape((-1, 1)) * ray_direction
    ).stop_fuse()
    xyz = (
        depth_scaled_rays + c2w[:3, 3]
    ).reshape((height, width, 3)).stop_fuse()
    dx = (xyz[2:, 1:-1] - xyz[:-2, 1:-1]).stop_fuse()
    dy = (xyz[1:-1, 2:] - xyz[1:-1, :-2]).stop_fuse()
    cross = jt.cross(dx, dy, dim=-1).stop_fuse()
    inner = safe_normalize(cross, eps=1e-12, clamp=True)
    side = jt.zeros((height - 2, 1, 3), dtype="float32")
    middle = jt.concat((side, inner, side), dim=1)
    border = jt.zeros((1, width, 3), dtype="float32")
    return jt.concat((border, middle, border), dim=0)


def postprocess_raster(
    camera,
    rendered_image,
    allmap,
    depth_ratio: float = 0.0,
    *,
    context=None,
):
    from .ops.precise_depth import precise_depth_divide

    alpha = allmap[1:2]
    normal_hwc = allmap[2:5].transpose([1, 2, 0])
    normal_hwc = normal_hwc @ camera.world_view_transform[:3, :3].transpose(0, 1)
    normal = normal_hwc.transpose([2, 0, 1])
    depth_median = _nan_to_num_like_pytorch(allmap[5:6])
    depth_expected = precise_depth_divide(allmap[0:1], alpha)
    surface_depth = (
        depth_expected * (1.0 - depth_ratio) + depth_median * depth_ratio
    )
    surface_normal = depth_to_normal(
        camera, surface_depth, context=context
    ).transpose([2, 0, 1])
    surface_normal = surface_normal * alpha
    distortion = allmap[6:7]

    def flatten(chw):
        return chw.transpose([1, 2, 0]).reshape((1, -1, chw.shape[0]))

    return {
        "acc_map": flatten(alpha),
        "dpt_map": flatten(surface_depth),
        "norm_map": flatten(normal),
        "dist_map": flatten(distortion),
        "surf_norm_map": flatten(surface_normal),
        "spec_map": flatten(rendered_image[3:4]),
        "rough_map": flatten(rendered_image[4:5]),
        "rgb_map": flatten(rendered_image[:3]),
    }
