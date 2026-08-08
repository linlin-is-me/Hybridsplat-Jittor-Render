"""PyTorch CUDA FP32 stages for HybridSplat reflection preprocessing."""

from __future__ import annotations

from typing import Any, Dict

from .native_runtime import load_jittor_ops, load_native_library


def precise_reflection_trace(
    xyz: Any,
    rotation_wxyz: Any,
    scaling_xy: Any,
    camera_center: Any,
) -> Dict[str, Any]:
    if (
        str(xyz.dtype) != "float32"
        or str(rotation_wxyz.dtype) != "float32"
        or str(scaling_xy.dtype) != "float32"
        or str(camera_center.dtype) != "float32"
    ):
        raise TypeError("precise_reflection_trace inputs must be float32")
    xyz_shape = tuple(xyz.shape)
    rotation_shape = tuple(rotation_wxyz.shape)
    scaling_shape = tuple(scaling_xy.shape)
    if len(xyz_shape) != 2 or xyz_shape[1] != 3:
        raise ValueError(f"xyz must have shape (P,3), got {xyz_shape}")
    if rotation_shape != (xyz_shape[0], 4):
        raise ValueError(
            "rotation_wxyz must have shape "
            f"({xyz_shape[0]},4), got {rotation_shape}"
        )
    if scaling_shape != (xyz_shape[0], 2):
        raise ValueError(
            "scaling_xy must have shape "
            f"({xyz_shape[0]},2), got {scaling_shape}"
        )
    if int(camera_center.numel()) != 3:
        raise ValueError(
            "camera_center must contain three values, "
            f"got {camera_center.shape}"
        )
    load_native_library()
    import jittor as jt

    operations = load_jittor_ops()
    rotation_matrix = operations.precise_reflection_prefix(rotation_wxyz)
    ones = jt.ones_like(scaling_xy[..., :1])
    scales3 = jt.concat((scaling_xy, ones), dim=-1)
    zeros = jt.zeros_like(scales3[..., 0])
    diagonal = jt.stack(
        (
            jt.stack((scales3[..., 0], zeros, zeros), dim=-1),
            jt.stack((zeros, scales3[..., 1], zeros), dim=-1),
            jt.stack((zeros, zeros, scales3[..., 2]), dim=-1),
        ),
        dim=-2,
    )
    # The prefix reproduces the TensorExpr-fused rotation matrix used after
    # PyTorch build_cov's profiling call. The diagonal's third column is
    # [0,0,1], so both Jittor tensor-core modes produce the same normal_raw.
    # Jittor's established slice boundary applies stop_fuse to both the
    # reindex input and output (contrib.py:157-164).
    linear = (rotation_matrix @ diagonal).stop_fuse()
    # Materialize the strided third-column reindex before crossing the custom
    # op boundary. Without this barrier Jittor can fuse the slice into the
    # consumer graph and change the last FP32 bits despite an exact BMM result.
    normal_raw3 = linear[:, :, 2].stop_fuse()
    packed = operations.precise_reflection_suffix(
        xyz,
        camera_center,
        normal_raw3,
    )
    return {
        "view_dir": packed[:, 0:3],
        "normals": packed[:, 3:6],
        "reflected_dir": packed[:, 6:9],
        "trace_ray_d": packed[:, 9:12],
        "trace_ray_o": packed[:, 12:15],
        "positive": packed[:, 15:16] > 0.5,
        "_normal_raw_received": packed[:, 16:19],
        "_normal_flipped": packed[:, 19:22],
        "_normal_denominator": packed[:, 22:23],
        "_reflection_product": packed[:, 23:26],
        "_reflection_dot": packed[:, 26:27],
        "_two_dot": packed[:, 27:28],
        "_reflected_scaled_normal": packed[:, 28:31],
        "_ray_denominator": packed[:, 31:32],
        "_ray_offset": packed[:, 32:35],
        "_flip_dot": packed[:, 35:36],
    }
