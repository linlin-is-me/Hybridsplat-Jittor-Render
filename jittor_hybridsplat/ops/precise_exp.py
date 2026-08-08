"""与 PyTorch CUDA FP32 exp 对齐的原生 Jittor 算子。"""

from __future__ import annotations

from typing import Any

from .native_runtime import load_jittor_ops, load_native_library


def precise_exp(value: Any) -> Any:
    """在 CUDA stream 0 上计算不启用 fast-math 的逐元素 FP32 exp。"""
    if str(value.dtype) != "float32":
        raise TypeError(f"precise_exp input must be float32, got {value.dtype}")
    load_native_library()
    operations = load_jittor_ops()
    return operations.precise_exp(value)
