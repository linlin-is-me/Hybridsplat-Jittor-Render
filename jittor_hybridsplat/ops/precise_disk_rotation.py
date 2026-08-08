"""PyTorch profiling-path quaternion rotation for OptiX disk geometry."""

from __future__ import annotations

from typing import Any

from .native_runtime import load_jittor_ops, load_native_library


def precise_disk_rotation(rotation_wxyz: Any) -> Any:
    if str(rotation_wxyz.dtype) != "float32":
        raise TypeError("rotation_wxyz must be float32")
    shape = tuple(rotation_wxyz.shape)
    if len(shape) != 2 or shape[1] != 4:
        raise ValueError(
            f"rotation_wxyz must have shape (P,4), got {shape}"
        )
    load_native_library()
    return load_jittor_ops().precise_disk_rotation(rotation_wxyz)
