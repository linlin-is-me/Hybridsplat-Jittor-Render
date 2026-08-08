"""Jittor custom op 加载与原生渲染资源生命周期。"""

from __future__ import annotations

import ctypes
import atexit
import hashlib
import os
from pathlib import Path
import threading
from typing import Any

from .build import (
    JITTOR_OPS_GEN_NAME,
    cuda_library_dir,
    native_include_dir,
    native_library_path,
    require_native_build,
)


HS_STATUS_OK = 0
HS_RASTERIZER_ABI_VERSION = 2
_ERROR_CAPACITY = 1024
_BUILD_INFO_CAPACITY = 4096
_LIBRARY: ctypes.CDLL | None = None
_OPS: Any | None = None
_OPS_MODULE: Any | None = None
_LOAD_LOCK = threading.Lock()
_LIVE_CONTEXTS: dict[int, "NativeRendererContext"] = {}
_DEFAULT_MATH_CONTEXT: "NativeRendererContext | None" = None


class NativeRasterizerError(RuntimeError):
    pass


class WorkspaceStatsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("geometry_bytes", ctypes.c_uint64),
        ("binning_bytes", ctypes.c_uint64),
        ("image_bytes", ctypes.c_uint64),
        ("current_bytes", ctypes.c_uint64),
        ("peak_bytes", ctypes.c_uint64),
        ("growth_count", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name, _ctype in self._fields_
        }


class OptixWorkspaceStatsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("scene_ready", ctypes.c_uint32),
        ("params_bytes", ctypes.c_uint64),
        ("gas_output_bytes", ctypes.c_uint64),
        ("gas_scratch_bytes", ctypes.c_uint64),
        ("dist3_bytes", ctypes.c_uint64),
        ("current_bytes", ctypes.c_uint64),
        ("peak_bytes", ctypes.c_uint64),
        ("growth_count", ctypes.c_uint64),
        ("scene_signature", ctypes.c_uint64),
        ("gas_build_count", ctypes.c_uint64),
        ("pipeline_create_count", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name, _ctype in self._fields_
        }


class MathWorkspaceStatsV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("handles_ready", ctypes.c_uint32),
        ("workspace_bytes", ctypes.c_uint64),
        ("current_bytes", ctypes.c_uint64),
        ("peak_bytes", ctypes.c_uint64),
        ("growth_count", ctypes.c_uint64),
        ("handle_create_count", ctypes.c_uint64),
        ("camera_center_calls", ctypes.c_uint64),
        ("general_inverse_calls", ctypes.c_uint64),
        ("matmul_calls", ctypes.c_uint64),
        ("pixel_world_calls", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name, _ctype in self._fields_
        }


class FlashgsWorkspaceStatsV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("device", ctypes.c_int32),
        ("initialized", ctypes.c_uint32),
        ("gaussian_capacity", ctypes.c_uint64),
        ("pair_capacity", ctypes.c_uint64),
        ("range_capacity", ctypes.c_uint64),
        ("output_pixel_capacity", ctypes.c_uint64),
        ("gaussian_bytes", ctypes.c_uint64),
        ("pair_bytes", ctypes.c_uint64),
        ("cub_bytes", ctypes.c_uint64),
        ("ranges_bytes", ctypes.c_uint64),
        ("output_bytes", ctypes.c_uint64),
        ("current_bytes", ctypes.c_uint64),
        ("peak_bytes", ctypes.c_uint64),
        ("growth_count", ctypes.c_uint64),
        ("overflow_retry_count", ctypes.c_uint64),
        ("forward_calls", ctypes.c_uint64),
        ("last_point_count", ctypes.c_uint64),
        ("last_num_rendered", ctypes.c_uint64),
        ("last_tile_count", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name, _ctype in self._fields_
        }


def _error_buffer() -> ctypes.Array[ctypes.c_char]:
    return ctypes.create_string_buffer(_ERROR_CAPACITY)


def _build_info_buffer() -> ctypes.Array[ctypes.c_char]:
    return ctypes.create_string_buffer(_BUILD_INFO_CAPACITY)


def _decode_error(buffer: ctypes.Array[ctypes.c_char]) -> str:
    return buffer.value.decode("utf-8", errors="replace")


def _check_status(
    status: int,
    operation: str,
    error: ctypes.Array[ctypes.c_char],
) -> None:
    if status != HS_STATUS_OK:
        detail = _decode_error(error)
        raise NativeRasterizerError(
            f"{operation} failed with native status {status}: {detail}"
        )


def load_native_library() -> ctypes.CDLL:
    global _LIBRARY
    with _LOAD_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        library_path = require_native_build()
        library = ctypes.CDLL(
            str(library_path),
            mode=os.RTLD_GLOBAL | os.RTLD_NOW,
        )
        error_pointer = ctypes.POINTER(ctypes.c_char)
        handle_pointer = ctypes.POINTER(ctypes.c_void_p)

        library.hs_rasterizer_abi_version.argtypes = []
        library.hs_rasterizer_abi_version.restype = ctypes.c_uint32
        library.hs_rasterizer_build_info.argtypes = [
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_build_info.restype = ctypes.c_int32
        library.hs_rasterizer_create.argtypes = [
            ctypes.c_int32,
            handle_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_create.restype = ctypes.c_int32
        library.hs_rasterizer_synchronize.argtypes = [
            ctypes.c_void_p,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_synchronize.restype = ctypes.c_int32
        library.hs_rasterizer_workspace_stats.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(WorkspaceStatsV1),
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_workspace_stats.restype = ctypes.c_int32
        library.hs_rasterizer_release_workspace.argtypes = [
            ctypes.c_void_p,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_release_workspace.restype = ctypes.c_int32
        library.hs_rasterizer_math_workspace_stats_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(MathWorkspaceStatsV2),
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_math_workspace_stats_v2.restype = ctypes.c_int32
        library.hs_flashgs_workspace_stats_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(FlashgsWorkspaceStatsV1),
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_flashgs_workspace_stats_v1.restype = ctypes.c_int32
        uint64_pointer = ctypes.POINTER(ctypes.c_uint64)
        uint32_pointer = ctypes.POINTER(ctypes.c_uint32)
        int32_pointer = ctypes.POINTER(ctypes.c_int32)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        library.hs_flashgs_debug_copy_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            uint64_pointer,
            uint32_pointer,
            int32_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_flashgs_debug_copy_v1.restype = ctypes.c_int32
        library.hs_flashgs_prepare_npz_host_f32_v1.argtypes = [
            ctypes.c_uint64,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_flashgs_prepare_npz_host_f32_v1.restype = ctypes.c_int32
        library.hs_rasterizer_destroy.argtypes = [
            ctypes.c_void_p,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_destroy.restype = ctypes.c_int32
        library.hs_rasterizer_forward_f32_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            float_pointer,
            ctypes.c_int32,
            ctypes.c_int32,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            ctypes.c_float,
            float_pointer,
            float_pointer,
            float_pointer,
            float_pointer,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_int32,
            float_pointer,
            float_pointer,
            float_pointer,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_forward_f32_v1.restype = ctypes.c_int32
        library.hs_optix_workspace_stats_v1.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(OptixWorkspaceStatsV1),
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_optix_workspace_stats_v1.restype = ctypes.c_int32
        library.hs_optix_release_scene_v1.argtypes = [
            ctypes.c_void_p,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_optix_release_scene_v1.restype = ctypes.c_int32
        library.hs_rasterizer_precise_exp_f32_v1.argtypes = [
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_exp_f32_v1.restype = ctypes.c_int32
        library.hs_rasterizer_precise_sh_to_rgb_f32_v1.argtypes = [
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint32,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_sh_to_rgb_f32_v1.restype = ctypes.c_int32
        library.hs_rasterizer_precise_direction_f32_v1.argtypes = [
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_direction_f32_v1.restype = ctypes.c_int32
        library.hs_rasterizer_precise_camera_center_f32_v2.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            float_pointer,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_camera_center_f32_v2.restype = (
            ctypes.c_int32
        )
        library.hs_rasterizer_precise_general_inverse_f32_v2.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            ctypes.c_uint32,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_general_inverse_f32_v2.restype = (
            ctypes.c_int32
        )
        library.hs_rasterizer_precise_matmul_nt_n3k3_f32_v2.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_matmul_nt_n3k3_f32_v2.restype = (
            ctypes.c_int32
        )
        library.hs_rasterizer_precise_pixel_world_f32_v2.argtypes = [
            ctypes.c_void_p,
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_pixel_world_f32_v2.restype = ctypes.c_int32
        library.hs_rasterizer_precise_disk_rotation_f32_v1.argtypes = [
            float_pointer,
            ctypes.c_uint64,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_disk_rotation_f32_v1.restype = (
            ctypes.c_int32
        )
        library.hs_rasterizer_precise_reflection_prefix_f32_v1.argtypes = [
            float_pointer,
            ctypes.c_uint64,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_reflection_prefix_f32_v1.restype = (
            ctypes.c_int32
        )
        library.hs_rasterizer_precise_reflection_suffix_f32_v1.argtypes = [
            float_pointer,
            float_pointer,
            float_pointer,
            ctypes.c_uint64,
            float_pointer,
            error_pointer,
            ctypes.c_uint64,
        ]
        library.hs_rasterizer_precise_reflection_suffix_f32_v1.restype = (
            ctypes.c_int32
        )

        abi_version = int(library.hs_rasterizer_abi_version())
        if abi_version != HS_RASTERIZER_ABI_VERSION:
            raise NativeRasterizerError(
                f"native ABI {abi_version} does not match "
                f"Python ABI {HS_RASTERIZER_ABI_VERSION}"
            )
        _LIBRARY = library
        return library


def load_jittor_ops() -> Any:
    global _OPS, _OPS_MODULE
    with _LOAD_LOCK:
        if _OPS is not None:
            return _OPS
        import jittor as jt

        jt.flags.use_cuda = 1
        source_dir = Path(__file__).resolve().parent / "jittor"
        include_dir = native_include_dir()
        library_dir = native_library_path().parent
        cuda_lib_dir = cuda_library_dir()
        extra_flags = " ".join(
            [
                f'-I"{include_dir}"',
                f'-L"{library_dir}"',
                "-lhybridsplat_rasterizer",
                f'-L"{cuda_lib_dir}"',
                "-lcublas",
            ]
        )
        module = jt.compile_custom_ops(
            [
                str(source_dir / "rasterize_forward_op.h"),
                str(source_dir / "rasterize_forward_op.cc"),
                str(source_dir / "flashgs_forward_op.h"),
                str(source_dir / "flashgs_forward_op.cc"),
                str(source_dir / "optix_build_scene_op.h"),
                str(source_dir / "optix_build_scene_op.cc"),
                str(source_dir / "optix_trace_op.h"),
                str(source_dir / "optix_trace_op.cc"),
                str(source_dir / "precise_exp_op.h"),
                str(source_dir / "precise_exp_op.cc"),
                str(source_dir / "precise_sigmoid_op.h"),
                str(source_dir / "precise_sigmoid_op.cc"),
                str(source_dir / "precise_normalize4_op.h"),
                str(source_dir / "precise_normalize4_op.cc"),
                str(source_dir / "precise_fuse_maps_op.h"),
                str(source_dir / "precise_fuse_maps_op.cc"),
                str(source_dir / "precise_camera_center_op.h"),
                str(source_dir / "precise_camera_center_op.cc"),
                str(source_dir / "precise_camera_inverse_op.h"),
                str(source_dir / "precise_camera_inverse_op.cc"),
                str(source_dir / "precise_pixel_world_op.h"),
                str(source_dir / "precise_pixel_world_op.cc"),
                str(source_dir / "precise_pixel_normal_op.h"),
                str(source_dir / "precise_pixel_normal_op.cc"),
                str(source_dir / "precise_general_inverse_op.h"),
                str(source_dir / "precise_general_inverse_op.cc"),
                str(source_dir / "precise_matmul_nt_n3k3_op.h"),
                str(source_dir / "precise_matmul_nt_n3k3_op.cc"),
                str(source_dir / "precise_direction_op.h"),
                str(source_dir / "precise_direction_op.cc"),
                str(source_dir / "precise_disk_rotation_op.h"),
                str(source_dir / "precise_disk_rotation_op.cc"),
                str(source_dir / "precise_depth_divide_op.h"),
                str(source_dir / "precise_depth_divide_op.cc"),
                str(source_dir / "precise_reflection_prefix_op.h"),
                str(source_dir / "precise_reflection_prefix_op.cc"),
                str(source_dir / "precise_reflection_suffix_op.h"),
                str(source_dir / "precise_reflection_suffix_op.cc"),
                str(source_dir / "precise_sh_to_rgb_op.h"),
                str(source_dir / "precise_sh_to_rgb_op.cc"),
            ],
            extra_flags=extra_flags,
            return_module=True,
            gen_name_=JITTOR_OPS_GEN_NAME,
        )
        _OPS_MODULE = module
        _OPS = module.ops
        return _OPS


def jittor_ops_identity() -> dict[str, object]:
    load_jittor_ops()
    if _OPS_MODULE is None or not getattr(_OPS_MODULE, "__file__", None):
        raise RuntimeError("Jittor custom-op module has no binary path")
    module_path = Path(_OPS_MODULE.__file__).resolve()
    digest = hashlib.sha256()
    with module_path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "gen_name": JITTOR_OPS_GEN_NAME,
        "module_path": str(module_path),
        "module_sha256": digest.hexdigest(),
    }


class NativeRendererContext:
    """持有原生 CUDA 上下文；关闭前会执行完整设备同步。"""

    def __init__(self, device: int = 0):
        self._library = load_native_library()
        self._device = int(device)
        self._handle = ctypes.c_void_p()
        error = _error_buffer()
        status = self._library.hs_rasterizer_create(
            self._device,
            ctypes.byref(self._handle),
            error,
            len(error),
        )
        _check_status(status, "hs_rasterizer_create", error)
        _LIVE_CONTEXTS[self.handle_value] = self

    @property
    def device(self) -> int:
        return self._device

    @property
    def handle_value(self) -> int:
        if self._handle.value is None:
            raise NativeRasterizerError("native rasterizer context is closed")
        return int(self._handle.value)

    @property
    def build_info(self) -> str:
        output = _build_info_buffer()
        status = self._library.hs_rasterizer_build_info(output, len(output))
        _check_status(status, "hs_rasterizer_build_info", output)
        return _decode_error(output)

    def synchronize(self) -> None:
        import jittor as jt

        jt.sync_all(True)
        error = _error_buffer()
        status = self._library.hs_rasterizer_synchronize(
            self._handle, error, len(error)
        )
        _check_status(status, "hs_rasterizer_synchronize", error)

    def workspace_stats(self) -> dict[str, int]:
        stats = WorkspaceStatsV1()
        stats.struct_size = ctypes.sizeof(WorkspaceStatsV1)
        stats.abi_version = HS_RASTERIZER_ABI_VERSION
        error = _error_buffer()
        status = self._library.hs_rasterizer_workspace_stats(
            self._handle,
            ctypes.byref(stats),
            error,
            len(error),
        )
        _check_status(status, "hs_rasterizer_workspace_stats", error)
        return stats.as_dict()

    def release_workspace(self) -> None:
        import jittor as jt

        jt.sync_all(True)
        error = _error_buffer()
        status = self._library.hs_rasterizer_release_workspace(
            self._handle, error, len(error)
        )
        _check_status(status, "hs_rasterizer_release_workspace", error)

    def math_workspace_stats(self) -> dict[str, int]:
        stats = MathWorkspaceStatsV2()
        stats.struct_size = ctypes.sizeof(MathWorkspaceStatsV2)
        stats.abi_version = HS_RASTERIZER_ABI_VERSION
        error = _error_buffer()
        status = self._library.hs_rasterizer_math_workspace_stats_v2(
            self._handle,
            ctypes.byref(stats),
            error,
            len(error),
        )
        _check_status(status, "hs_rasterizer_math_workspace_stats_v2", error)
        return stats.as_dict()

    def optix_workspace_stats(self) -> dict[str, int]:
        stats = OptixWorkspaceStatsV1()
        stats.struct_size = ctypes.sizeof(OptixWorkspaceStatsV1)
        stats.abi_version = HS_RASTERIZER_ABI_VERSION
        error = _error_buffer()
        status = self._library.hs_optix_workspace_stats_v1(
            self._handle,
            ctypes.byref(stats),
            error,
            len(error),
        )
        _check_status(status, "hs_optix_workspace_stats_v1", error)
        return stats.as_dict()

    def flashgs_workspace_stats(self) -> dict[str, int]:
        stats = FlashgsWorkspaceStatsV1()
        stats.struct_size = ctypes.sizeof(FlashgsWorkspaceStatsV1)
        stats.abi_version = HS_RASTERIZER_ABI_VERSION
        error = _error_buffer()
        status = self._library.hs_flashgs_workspace_stats_v1(
            self._handle,
            ctypes.byref(stats),
            error,
            len(error),
        )
        _check_status(status, "hs_flashgs_workspace_stats_v1", error)
        return stats.as_dict()

    def flashgs_debug_snapshot(self) -> dict[str, Any]:
        import jittor as jt
        import numpy as np

        jt.sync_all(True)
        stats = self.flashgs_workspace_stats()
        point_count = stats["last_point_count"]
        pair_count = stats["last_num_rendered"]
        tile_count = stats["last_tile_count"]
        arrays = {
            "points_xy": np.empty((point_count, 2), dtype=np.float32),
            "rgb": np.empty((point_count, 3), dtype=np.float32),
            "other_features": np.empty((point_count, 2), dtype=np.float32),
            "conic_opacity": np.empty((point_count, 4), dtype=np.float32),
            "normal_opacity": np.empty((point_count, 4), dtype=np.float32),
            "Tu": np.empty((point_count, 3), dtype=np.float32),
            "Tv": np.empty((point_count, 3), dtype=np.float32),
            "Tw": np.empty((point_count, 3), dtype=np.float32),
            "sorted_keys": np.empty(pair_count, dtype=np.uint64),
            "sorted_values": np.empty(pair_count, dtype=np.uint32),
            "tile_ranges": np.empty((tile_count, 2), dtype=np.int32),
        }
        float_pointer = ctypes.POINTER(ctypes.c_float)
        error = _error_buffer()
        status = self._library.hs_flashgs_debug_copy_v1(
            self._handle,
            point_count,
            pair_count,
            tile_count,
            *[
                arrays[name].ctypes.data_as(float_pointer)
                for name in (
                    "points_xy",
                    "rgb",
                    "other_features",
                    "conic_opacity",
                    "normal_opacity",
                    "Tu",
                    "Tv",
                    "Tw",
                )
            ],
            arrays["sorted_keys"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            arrays["sorted_values"].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            arrays["tile_ranges"].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            error,
            len(error),
        )
        _check_status(status, "hs_flashgs_debug_copy_v1", error)
        return arrays

    def release_optix_scene(self) -> None:
        import jittor as jt

        jt.sync_all(True)
        error = _error_buffer()
        status = self._library.hs_optix_release_scene_v1(
            self._handle, error, len(error)
        )
        _check_status(status, "hs_optix_release_scene_v1", error)

    def close(self) -> None:
        if self._handle.value is None:
            return
        import jittor as jt

        jt.sync_all(True)
        handle_value = self.handle_value
        error = _error_buffer()
        status = self._library.hs_rasterizer_destroy(
            self._handle, error, len(error)
        )
        _check_status(status, "hs_rasterizer_destroy", error)
        self._handle = ctypes.c_void_p()
        _LIVE_CONTEXTS.pop(handle_value, None)

    def __enter__(self) -> "NativeRendererContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def default_math_context() -> NativeRendererContext:
    global _DEFAULT_MATH_CONTEXT
    if _DEFAULT_MATH_CONTEXT is None or _DEFAULT_MATH_CONTEXT._handle.value is None:
        _DEFAULT_MATH_CONTEXT = NativeRendererContext(0)
    return _DEFAULT_MATH_CONTEXT


def _close_default_math_context() -> None:
    global _DEFAULT_MATH_CONTEXT
    if _DEFAULT_MATH_CONTEXT is not None:
        try:
            _DEFAULT_MATH_CONTEXT.close()
        except BaseException:
            pass
        _DEFAULT_MATH_CONTEXT = None


atexit.register(_close_default_math_context)

