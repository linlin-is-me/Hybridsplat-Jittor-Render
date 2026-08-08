"""Deterministic FlashGS 16x16 forward operator for Jittor."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, NamedTuple

from .native_runtime import NativeRendererContext, load_jittor_ops


class FlashgsOutputs(NamedTuple):
    color: Any
    other: Any
    depth: Any
    alpha: Any
    normal: Any
    median_depth: Any
    distortion: Any
    rendered: Any


def _validate_var(name: str, value: Any, shape: tuple[int, ...]) -> None:
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if str(value.dtype) != "float32":
        raise TypeError(f"{name} must be float32, got {value.dtype}")


def _finite(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a CPU real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _vector(name: str, value: Sequence[float], count: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a CPU sequence")
    flattened: list[float] = []
    for item in value:
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            flattened.extend(item)
        else:
            flattened.append(item)
    if len(flattened) != count:
        raise ValueError(f"{name} must contain {count} values")
    return tuple(_finite(f"{name}[{index}]", item) for index, item in enumerate(flattened))


def flashgs_forward(
    positions: Any,
    shs: Any,
    colors: Any,
    opacities: Any,
    covariance: Any,
    scales: Any,
    rotations: Any,
    specular: Any,
    roughness: Any,
    *,
    width: int,
    height: int,
    use_sh: bool,
    camera_position: Sequence[float],
    camera_rotation: Sequence[float],
    focal_x: float,
    focal_y: float,
    z_far: float = 100.0,
    z_near: float = 0.01,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    scale_modifier: float = 1.0,
    initial_pair_capacity: int = 1 << 20,
    context: NativeRendererContext,
) -> FlashgsOutputs:
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive CPU integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive CPU integer")
    if not isinstance(use_sh, bool):
        raise TypeError("use_sh must be bool")
    if (
        isinstance(initial_pair_capacity, bool)
        or not isinstance(initial_pair_capacity, int)
        or initial_pair_capacity <= 0
    ):
        raise ValueError("initial_pair_capacity must be a positive CPU integer")
    count = int(positions.shape[0])
    _validate_var("positions", positions, (count, 3))
    _validate_var("shs", shs, (count, 48))
    _validate_var("colors", colors, (count, 3))
    _validate_var("opacities", opacities, (count, 1))
    _validate_var("covariance", covariance, (count, 6))
    _validate_var("scales", scales, (count, 2))
    _validate_var("rotations", rotations, (count, 4))
    _validate_var("specular", specular, (count, 1))
    _validate_var("roughness", roughness, (count, 1))
    position = _vector("camera_position", camera_position, 3)
    rotation = _vector("camera_rotation", camera_rotation, 9)
    background_value = _vector("background", background, 3)
    scale = _finite("scale_modifier", scale_modifier, positive=True)
    fx = _finite("focal_x", focal_x, positive=True)
    fy = _finite("focal_y", focal_y, positive=True)
    far = _finite("z_far", z_far, positive=True)
    near = _finite("z_near", z_near, positive=True)
    if far <= near:
        raise ValueError("z_far must be greater than z_near")
    operations = load_jittor_ops()
    outputs = operations.flashgs_forward(
        positions,
        shs,
        colors,
        opacities,
        covariance,
        scales,
        rotations,
        specular,
        roughness,
        width,
        height,
        int(use_sh),
        initial_pair_capacity,
        context.handle_value,
        scale,
        fx,
        fy,
        far,
        near,
        *position,
        *rotation,
        *background_value,
    )
    return FlashgsOutputs(*outputs)
