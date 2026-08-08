"""HybridSplat 五通道标准光栅器的 Jittor 前向接口。"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

from .native_runtime import NativeRendererContext, load_jittor_ops


class RasterizerOutputs(NamedTuple):
    color: Any
    auxiliary: Any
    weight: Any
    radii: Any
    rendered: Any


def _validate_var(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: str = "float32",
) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if str(value.dtype) != dtype:
        raise TypeError(f"{name} must be {dtype}, got {value.dtype}")


def _positive_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a CPU real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def rasterize_forward(
    background: Any,
    means3d: Any,
    colors: Any,
    opacities: Any,
    scales: Any,
    rotations: Any,
    viewmatrix: Any,
    projmatrix: Any,
    camera_position: Any,
    *,
    width: int,
    height: int,
    scale_modifier: float,
    tan_fovx: float,
    tan_fovy: float,
    prefiltered: bool = False,
    context: NativeRendererContext,
) -> RasterizerOutputs:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive CPU integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive CPU integer")
    if not isinstance(prefiltered, bool):
        raise TypeError("prefiltered must be bool")
    count = int(means3d.shape[0])
    _validate_var("background", background, (5,))
    _validate_var("means3d", means3d, (count, 3))
    _validate_var("colors", colors, (count, 5))
    _validate_var("opacities", opacities, (count, 1))
    _validate_var("scales", scales, (count, 2))
    _validate_var("rotations", rotations, (count, 4))
    _validate_var("viewmatrix", viewmatrix, (4, 4))
    _validate_var("projmatrix", projmatrix, (4, 4))
    _validate_var("camera_position", camera_position, (3,))
    scale = _positive_finite("scale_modifier", scale_modifier)
    fovx = _positive_finite("tan_fovx", tan_fovx)
    fovy = _positive_finite("tan_fovy", tan_fovy)
    operations = load_jittor_ops()
    outputs = operations.rasterize_forward(
        background,
        means3d,
        colors,
        opacities,
        scales,
        rotations,
        viewmatrix,
        projmatrix,
        camera_position,
        width,
        height,
        scale,
        fovx,
        fovy,
        int(prefiltered),
        context.handle_value,
    )
    return RasterizerOutputs(*outputs)
