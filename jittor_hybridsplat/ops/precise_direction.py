"""与 PyTorch CUDA FP32 归一化顺序对齐的视线方向算子。"""

from __future__ import annotations

from typing import Any

from .native_runtime import load_jittor_ops, load_native_library


def precise_direction(points: Any, center: Any) -> Any:
    if str(points.dtype) != "float32" or str(center.dtype) != "float32":
        raise TypeError("precise_direction inputs must be float32")
    point_shape = tuple(points.shape)
    if len(point_shape) != 2 or point_shape[1] != 3:
        raise ValueError(f"points must have shape (P,3), got {point_shape}")
    if int(center.numel()) != 3:
        raise ValueError(f"center must contain three values, got {center.shape}")
    load_native_library()
    return load_jittor_ops().precise_direction(points, center)
