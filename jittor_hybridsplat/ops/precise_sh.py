"""与 PyTorch CUDA 逐算子 FP32 舍入对齐的 SH-to-RGB 算子。"""

from __future__ import annotations

from typing import Any

from .native_runtime import load_jittor_ops, load_native_library


def precise_sh_to_rgb(sh: Any, directions: Any, degree: int) -> Any:
    if str(sh.dtype) != "float32" or str(directions.dtype) != "float32":
        raise TypeError("precise_sh_to_rgb inputs must be float32")
    sh_shape = tuple(sh.shape)
    direction_shape = tuple(directions.shape)
    if len(sh_shape) != 3 or sh_shape[1] != 3:
        raise ValueError(f"SH input must have shape (P,3,C), got {sh_shape}")
    if direction_shape != (sh_shape[0], 3):
        raise ValueError(
            f"directions must have shape ({sh_shape[0]},3), got {direction_shape}"
        )
    degree = int(degree)
    if degree < 0 or degree > 3:
        raise ValueError(f"precise_sh_to_rgb degree must be in [0,3], got {degree}")
    if sh_shape[2] < (degree + 1) ** 2:
        raise ValueError("SH coefficient count is too small for the degree")
    load_native_library()
    return load_jittor_ops().precise_sh_to_rgb(sh, directions, degree)
