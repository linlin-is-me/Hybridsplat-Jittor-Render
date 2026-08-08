from __future__ import annotations

from typing import Any

from .native_runtime import load_jittor_ops, load_native_library


def precise_sigmoid(value: Any) -> Any:
    if str(value.dtype) != "float32":
        raise TypeError(f"precise_sigmoid input must be float32, got {value.dtype}")
    load_native_library()
    return load_jittor_ops().precise_sigmoid(value)


def precise_normalize4(value: Any) -> Any:
    if str(value.dtype) != "float32" or len(value.shape) != 2 or int(value.shape[1]) != 4:
        raise TypeError("precise_normalize4 input must be FP32 [P,4]")
    load_native_library()
    return load_jittor_ops().precise_normalize4(value)
