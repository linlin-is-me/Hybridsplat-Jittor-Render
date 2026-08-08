"""HybridSplat 的纯 Jittor 推理实现。"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CheckpointContract",
    "CheckpointContractError",
    "GaussianParameters",
    "HybridSplatCheckpoint",
    "infer_checkpoint_contract",
    "validate_checkpoint_archive",
    "load_checkpoint",
    "NativeRendererContext",
    "precise_direction",
    "precise_exp",
    "precise_reflection_trace",
    "precise_sh_to_rgb",
    "HybridSplatSession",
]


def __getattr__(name: str) -> Any:
    if name == "precise_exp":
        from .ops.precise_exp import precise_exp

        return precise_exp
    if name == "precise_direction":
        from .ops.precise_direction import precise_direction

        return precise_direction
    if name == "precise_reflection_trace":
        from .ops.precise_reflection_trace import precise_reflection_trace

        return precise_reflection_trace
    if name == "precise_sh_to_rgb":
        from .ops.precise_sh import precise_sh_to_rgb

        return precise_sh_to_rgb
    if name == "NativeRendererContext":
        from .ops import native_runtime

        return native_runtime.NativeRendererContext
    if name == "HybridSplatSession":
        from .session import HybridSplatSession

        return HybridSplatSession
    if name in (
        "CheckpointContract",
        "CheckpointContractError",
        "GaussianParameters",
        "HybridSplatCheckpoint",
        "infer_checkpoint_contract",
        "validate_checkpoint_archive",
        "load_checkpoint",
    ):
        from . import checkpoint

        return getattr(checkpoint, name)
    raise AttributeError(name)
