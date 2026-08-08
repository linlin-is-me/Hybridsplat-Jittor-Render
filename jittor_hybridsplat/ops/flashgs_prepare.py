"""Exact host activation for the frozen deterministic FlashGS NPZ path."""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import numpy as np

from .native_runtime import _check_status, _error_buffer, load_native_library


class FlashgsPreparedArrays(NamedTuple):
    opacity: np.ndarray
    scaling: np.ndarray
    rotation: np.ndarray
    covariance: np.ndarray
    specular: np.ndarray
    roughness: np.ndarray


def _input(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != np.float32:
        raise TypeError(f"{name} must be float32, got {value.dtype}")
    if not value.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return value


def prepare_flashgs_npz_host(
    opacity_raw: np.ndarray,
    scaling_raw: np.ndarray,
    rotation_raw: np.ndarray,
    specular_raw: np.ndarray,
    roughness_raw: np.ndarray,
) -> FlashgsPreparedArrays:
    count = int(scaling_raw.shape[0])
    arrays = (
        _input("opacity_raw", opacity_raw, (count, 1)),
        _input("scaling_raw", scaling_raw, (count, 2)),
        _input("rotation_raw", rotation_raw, (count, 4)),
        _input("specular_raw", specular_raw, (count, 1)),
        _input("roughness_raw", roughness_raw, (count, 1)),
    )
    outputs = FlashgsPreparedArrays(
        np.empty((count, 1), dtype=np.float32),
        np.empty((count, 2), dtype=np.float32),
        np.empty((count, 4), dtype=np.float32),
        np.empty((count, 6), dtype=np.float32),
        np.empty((count, 1), dtype=np.float32),
        np.empty((count, 1), dtype=np.float32),
    )
    pointer = ctypes.POINTER(ctypes.c_float)
    error = _error_buffer()
    status = load_native_library().hs_flashgs_prepare_npz_host_f32_v1(
        count,
        *[value.ctypes.data_as(pointer) for value in arrays],
        *[value.ctypes.data_as(pointer) for value in outputs],
        error,
        len(error),
    )
    _check_status(status, "hs_flashgs_prepare_npz_host_f32_v1", error)
    if not all(np.isfinite(value).all() for value in outputs):
        raise RuntimeError("native FlashGS NPZ preparation returned NaN or Inf")
    return outputs
