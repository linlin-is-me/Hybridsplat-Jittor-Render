"""精确复现 PyTorch FP32 深度除法及其非有限值处理。"""

from __future__ import annotations

from typing import Any

from .native_runtime import load_jittor_ops


def precise_depth_divide(numerator: Any, denominator: Any) -> Any:
    if str(numerator.dtype) != "float32":
        raise TypeError("precise_depth_divide numerator must be float32")
    if str(denominator.dtype) != "float32":
        raise TypeError("precise_depth_divide denominator must be float32")
    numerator_shape = tuple(numerator.shape)
    denominator_shape = tuple(denominator.shape)
    if numerator_shape != denominator_shape:
        raise ValueError(
            "precise_depth_divide inputs must have identical shapes, "
            f"got {numerator_shape} and {denominator_shape}"
        )
    return load_jittor_ops().precise_depth_divide(
        numerator,
        denominator,
    )
