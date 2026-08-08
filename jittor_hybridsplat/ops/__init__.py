"""Jittor 算子及其原生后端。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "NativeRendererContext",
    "FlashgsOutputs",
    "flashgs_forward",
    "build_optix_scene",
    "OptixTraceOutputs",
    "precise_camera_center",
    "precise_camera_inverse",
    "precise_depth_divide",
    "precise_direction",
    "precise_disk_rotation",
    "precise_exp",
    "precise_general_inverse",
    "precise_matmul_nt_n3k3",
    "precise_pixel_normal",
    "precise_pixel_world",
    "precise_reflection_trace",
    "precise_sh_to_rgb",
    "RasterizerOutputs",
    "rasterize_forward",
    "trace_optix",
]


def __getattr__(name: str) -> Any:
    if name == "precise_camera_center":
        from .precise_camera import precise_camera_center

        return precise_camera_center
    if name in ("FlashgsOutputs", "flashgs_forward"):
        from . import flashgs

        return getattr(flashgs, name)
    if name in (
        "precise_camera_inverse",
        "precise_pixel_normal",
        "precise_pixel_world",
    ):
        from . import precise_camera_rays

        return getattr(precise_camera_rays, name)
    if name == "precise_depth_divide":
        from .precise_depth import precise_depth_divide

        return precise_depth_divide
    if name == "precise_exp":
        from .precise_exp import precise_exp

        return precise_exp
    if name in ("precise_general_inverse", "precise_matmul_nt_n3k3"):
        from . import precise_matrix

        return getattr(precise_matrix, name)
    if name == "precise_direction":
        from .precise_direction import precise_direction

        return precise_direction
    if name == "precise_disk_rotation":
        from .precise_disk_rotation import precise_disk_rotation

        return precise_disk_rotation
    if name == "precise_reflection_trace":
        from .precise_reflection_trace import precise_reflection_trace

        return precise_reflection_trace
    if name == "precise_sh_to_rgb":
        from .precise_sh import precise_sh_to_rgb

        return precise_sh_to_rgb
    if name == "NativeRendererContext":
        from . import native_runtime

        return native_runtime.NativeRendererContext
    if name in ("RasterizerOutputs", "rasterize_forward"):
        from . import rasterizer

        return getattr(rasterizer, name)
    if name in ("OptixTraceOutputs", "build_optix_scene", "trace_optix"):
        from . import optix

        return getattr(optix, name)
    raise AttributeError(name)
