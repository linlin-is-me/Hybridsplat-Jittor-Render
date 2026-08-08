"""Camera-ray CUDA operations that preserve PyTorch 2.6 FP32 order."""

from __future__ import annotations

from typing import Any

from .native_runtime import default_math_context, load_jittor_ops


def precise_camera_inverse(matrix: Any) -> Any:
    if str(matrix.dtype) != "float32":
        raise TypeError("precise_camera_inverse matrix must be float32")
    shape = tuple(matrix.shape)
    if shape != (3, 3):
        raise ValueError(
            f"matrix must have shape (3,3), got {shape}"
        )
    return load_jittor_ops().precise_camera_inverse(matrix)


def precise_pixel_world(rotation: Any, vectors: Any, *, context=None) -> Any:
    if str(rotation.dtype) != "float32":
        raise TypeError("precise_pixel_world rotation must be float32")
    if str(vectors.dtype) != "float32":
        raise TypeError("precise_pixel_world vectors must be float32")
    rotation_shape = tuple(rotation.shape)
    vector_shape = tuple(vectors.shape)
    if rotation_shape != (3, 3):
        raise ValueError(
            f"rotation must have shape (3,3), got {rotation_shape}"
        )
    if not vector_shape or vector_shape[-1] != 3:
        raise ValueError(
            f"vectors must have final extent 3, got {vector_shape}"
        )
    context = default_math_context() if context is None else context
    return load_jittor_ops().precise_pixel_world(
        rotation, vectors, context.handle_value
    )


def precise_pixel_normal(normal_raw: Any) -> Any:
    if str(normal_raw.dtype) != "float32":
        raise TypeError("precise_pixel_normal input must be float32")
    shape = tuple(normal_raw.shape)
    if not shape or shape[-1] != 3:
        raise ValueError(
            f"normal_raw must have final extent 3, got {shape}"
        )
    return load_jittor_ops().precise_pixel_normal(normal_raw)
