"""PyTorch CUDA FP32 逆矩阵与局部精确矩阵乘接口。"""

from __future__ import annotations

from typing import Any

from .native_runtime import default_math_context, load_jittor_ops, load_native_library


def precise_general_inverse(matrix: Any, *, context=None) -> Any:
    if str(matrix.dtype) != "float32":
        raise TypeError(
            f"matrix must be float32, got {matrix.dtype}"
        )
    shape = tuple(matrix.shape)
    if shape not in ((3, 3), (4, 4)):
        raise ValueError(
            f"matrix must have shape (3,3) or (4,4), got {shape}"
        )
    load_native_library()
    context = default_math_context() if context is None else context
    return load_jittor_ops().precise_general_inverse(
        matrix, context.handle_value
    )


def precise_matmul_nt_n3k3(left: Any, right: Any, *, context=None) -> Any:
    if str(left.dtype) != "float32" or str(right.dtype) != "float32":
        raise TypeError("precise_matmul_nt_n3k3 inputs must be float32")
    left_shape = tuple(left.shape)
    right_shape = tuple(right.shape)
    if len(left_shape) != 2 or left_shape[1] != 3:
        raise ValueError(
            f"left must have shape (N,3), got {left_shape}"
        )
    if right_shape != (3, 3):
        raise ValueError(
            f"right must have shape (3,3), got {right_shape}"
        )
    load_native_library()
    context = default_math_context() if context is None else context
    return load_jittor_ops().precise_matmul_nt_n3k3(
        left, right, context.handle_value
    )
