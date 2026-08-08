"""HybridSplat OptiX 反射追踪的 Jittor 前向接口。"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from .native_runtime import NativeRendererContext, load_jittor_ops


class OptixTraceOutputs(NamedTuple):
    rgb: Any
    depth: Any
    acc: Any
    normal: Any
    dist: Any
    auxiliary: Any
    mid: Any
    wet: Any


def _validate_var(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: str,
) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if str(value.dtype) != dtype:
        raise TypeError(f"{name} must be {dtype}, got {value.dtype}")


def build_optix_scene(
    vertices: Any,
    triangles: Any,
    *,
    scene_signature: int,
    rebuild: bool = False,
    context: NativeRendererContext,
) -> None:
    vertex_count = int(vertices.shape[0])
    triangle_count = int(triangles.shape[0])
    _validate_var("vertices", vertices, (vertex_count, 3), "float32")
    _validate_var("triangles", triangles, (triangle_count, 3), "int32")
    if vertex_count <= 0 or triangle_count <= 0:
        raise ValueError("OptiX scene must contain vertices and triangles")
    if isinstance(scene_signature, bool) or not isinstance(scene_signature, int):
        raise TypeError("scene_signature must be a CPU integer")
    if scene_signature <= 0 or scene_signature >= 2**63:
        raise ValueError("scene_signature must be in [1, 2**63)")
    if not isinstance(rebuild, bool):
        raise TypeError("rebuild must be bool")
    status = load_jittor_ops().optix_build_scene(
        vertices,
        triangles,
        scene_signature,
        int(rebuild),
        context.handle_value,
    )
    import jittor as jt

    # GAS build belongs to session control and completes before tracing starts.
    fetched_status = jt.fetch_sync([status])[0]
    if int(fetched_status.reshape(-1)[0]) != 0:
        raise RuntimeError("OptiX scene build returned a nonzero device status")


def trace_optix(
    ray_origins: Any,
    ray_directions: Any,
    background: Any,
    means3d: Any,
    shs: Any,
    others: Any,
    opacities: Any,
    scales: Any,
    rotations: Any,
    viewmatrix: Any,
    projmatrix: Any,
    camera_position: Any,
    *,
    degree: int,
    scale_modifier: float = 1.0,
    start_from_first: bool = False,
    max_trace_depth: int = 0,
    specular_threshold: float = 0.0,
    use_others: bool = True,
    context: NativeRendererContext,
) -> OptixTraceOutputs:
    count = int(ray_origins.shape[0])
    coefficient_count = int(shs.shape[1])
    _validate_var("ray_origins", ray_origins, (count, 3), "float32")
    _validate_var("ray_directions", ray_directions, (count, 3), "float32")
    _validate_var("background", background, (3,), "float32")
    _validate_var("means3d", means3d, (count, 3), "float32")
    _validate_var("shs", shs, (count, coefficient_count, 3), "float32")
    _validate_var("others", others, (count, 2), "float32")
    _validate_var("opacities", opacities, (count, 1), "float32")
    _validate_var("scales", scales, (count, 2), "float32")
    _validate_var("rotations", rotations, (count, 4), "float32")
    _validate_var("viewmatrix", viewmatrix, (4, 4), "float32")
    _validate_var("projmatrix", projmatrix, (4, 4), "float32")
    _validate_var("camera_position", camera_position, (3,), "float32")
    if count <= 0 or coefficient_count <= 0:
        raise ValueError("OptiX trace requires nonempty rays and SH coefficients")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree not in range(4):
        raise ValueError("degree must be an integer in [0, 3]")
    if not isinstance(start_from_first, bool):
        raise TypeError("start_from_first must be bool")
    if not isinstance(use_others, bool):
        raise TypeError("use_others must be bool")
    if (
        isinstance(max_trace_depth, bool)
        or not isinstance(max_trace_depth, int)
        or max_trace_depth not in (0, 1)
    ):
        raise ValueError("max_trace_depth must be 0 or 1")
    for name, value in (
        ("scale_modifier", scale_modifier),
        ("specular_threshold", specular_threshold),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a CPU real scalar")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if float(scale_modifier) <= 0.0:
        raise ValueError("scale_modifier must be positive")
    outputs = load_jittor_ops().optix_trace(
        ray_origins,
        ray_directions,
        background,
        means3d,
        shs,
        others,
        opacities,
        scales,
        rotations,
        viewmatrix,
        projmatrix,
        camera_position,
        degree,
        float(scale_modifier),
        int(start_from_first),
        max_trace_depth,
        float(specular_threshold),
        int(use_others),
        context.handle_value,
    )
    return OptixTraceOutputs(*outputs)
