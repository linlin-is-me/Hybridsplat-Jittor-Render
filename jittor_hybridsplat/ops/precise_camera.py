"""与 PyTorch 2.6 SGEMM(N,T,1,3,3) 逐位对齐的相机中心算子。"""

from __future__ import annotations

from typing import Any

from .native_runtime import default_math_context, load_jittor_ops, load_native_library


def precise_camera_center(rotation: Any, translation: Any, *, context=None) -> Any:
    if str(rotation.dtype) != "float32":
        raise TypeError("rotation must be float32")
    if str(translation.dtype) != "float32":
        raise TypeError("translation must be float32")
    rotation_shape = tuple(rotation.shape)
    translation_shape = tuple(translation.shape)
    if rotation_shape != (3, 3):
        raise ValueError(
            f"rotation must have shape (3,3), got {rotation_shape}"
        )
    if translation_shape != (3, 1):
        raise ValueError(
            f"translation must have shape (3,1), got {translation_shape}"
        )
    load_native_library()
    context = default_math_context() if context is None else context
    return load_jittor_ops().precise_camera_center(
        rotation, translation, context.handle_value
    )
