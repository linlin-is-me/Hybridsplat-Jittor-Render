from __future__ import annotations

from typing import Any

from .native_runtime import load_jittor_ops, load_native_library


def precise_fuse_maps(base_rgb: Any, reflection_rgb: Any, specular: Any) -> dict[str, Any]:
    if str(base_rgb.dtype) != "float32" or str(reflection_rgb.dtype) != "float32" or str(specular.dtype) != "float32":
        raise TypeError("precise_fuse_maps inputs must be FP32")
    if tuple(base_rgb.shape) != tuple(reflection_rgb.shape):
        raise ValueError("precise_fuse_maps RGB shapes must match")
    if len(base_rgb.shape) != 3 or int(base_rgb.shape[0]) != 1 or int(base_rgb.shape[2]) != 3:
        raise ValueError("precise_fuse_maps RGB inputs must have shape [1,P,3]")
    if tuple(specular.shape) != (1, int(base_rgb.shape[1]), 1):
        raise ValueError("precise_fuse_maps specular must have shape [1,P,1]")
    load_native_library()
    diffuse, final_rgb, stored_reflection = load_jittor_ops().precise_fuse_maps(base_rgb, reflection_rgb, specular)
    return {
        "dif_rgb_map": diffuse,
        "rgb_map": final_rgb,
        "ref_rgb_map": stored_reflection,
    }
