from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .camera import (
    CameraRecord,
    FlashgsCameraRecord,
    load_camera_records,
    load_flashgs_camera_records,
    materialize_camera,
)
from .checkpoint import CheckpointContract, HybridSplatCheckpoint, load_checkpoint
from .math import gaussian_disks
from .ops.native_runtime import NativeRendererContext
from .ops.optix import build_optix_scene
from .renderer import (
    PreparedFlashgsGaussian,
    PreparedGaussian,
    prepare_flashgs_gaussian,
    prepare_gaussian,
    render_hybridsplat_flashgs_view,
    render_hybridsplat_view,
)


RAW_KEYS = (
    "rgb_map",
    "acc_map",
    "dpt_map",
    "norm_map",
    "surf_norm_map",
    "spec_map",
    "dif_rgb_map",
    "ref_rgb_map",
)
VALIDATED_JITTOR_VERSION = "1.3.11.0"


class HybridSplatSession:
    def __init__(
        self,
        checkpoint: HybridSplatCheckpoint,
        cameras: tuple[CameraRecord, ...],
        context: NativeRendererContext,
        base: PreparedGaussian,
        reflection: PreparedGaussian,
        *,
        backend: str,
        flashgs_cameras: tuple[FlashgsCameraRecord, ...],
        flashgs_base: PreparedFlashgsGaussian | None,
        flashgs_reflection: PreparedFlashgsGaussian | None,
        synchronize_stages: bool,
    ) -> None:
        self.checkpoint = checkpoint
        self.cameras = cameras
        self.context = context
        self.base = base
        self.reflection = reflection
        self.backend = backend
        self.flashgs_cameras = flashgs_cameras
        self.flashgs_base = flashgs_base
        self.flashgs_reflection = flashgs_reflection
        self.synchronize_stages = bool(synchronize_stages)
        self.closed = False
        self.render_count = 0
        self.last_render_metadata: dict[str, Any] | None = None

    @classmethod
    def from_npz(
        cls,
        checkpoint: str | Path,
        dataset_root: str | Path,
        *,
        backend: str = "standard",
        flashgs_camera_path: str | Path | None = None,
        device: int = 0,
        ratio: float = 0.25,
        near: float = 0.2,
        far: float = 20.0,
        synchronize_stages: bool = False,
        checkpoint_contract: CheckpointContract | None = None,
    ) -> "HybridSplatSession":
        import jittor as jt

        if (
            jt.__version__ != VALIDATED_JITTOR_VERSION
            and os.environ.get("HYBRIDSPLAT_ALLOW_UNVERIFIED_JITTOR") != "1"
        ):
            raise RuntimeError(
                f"this renderer is validated with Jittor {VALIDATED_JITTOR_VERSION}, "
                f"but the active version is {jt.__version__}; set "
                "HYBRIDSPLAT_ALLOW_UNVERIFIED_JITTOR=1 only after validating the "
                "current Jittor source and numerical results"
            )
        jt.flags.use_cuda = 1
        # The PyTorch reference forms the 4x4 projection on CUDA with TF32.
        # Jittor's tensor-core path reproduces it while amp_reg=0 keeps FP32 Vars.
        jt.flags.use_tensorcore = 1
        jt.flags.amp_reg = 0
        if backend not in ("standard", "flashgs"):
            raise ValueError("backend must be standard or flashgs")
        if backend == "standard" and flashgs_camera_path is not None:
            raise ValueError("flashgs_camera_path is only valid for the flashgs backend")
        cameras = load_camera_records(
            dataset_root, ratio=ratio, near=near, far=far
        )
        flashgs_cameras = ()
        if backend == "flashgs":
            flashgs_cameras = load_flashgs_camera_records(
                dataset_root, camera_path=flashgs_camera_path
            )
            if len(flashgs_cameras) != len(cameras):
                raise ValueError("standard and FlashGS camera counts differ")
            for standard, flash in zip(cameras, flashgs_cameras):
                if standard.camera_id != flash.camera_id or standard.name != flash.name:
                    raise ValueError("standard and FlashGS camera order differs")
                if (standard.image_height, standard.image_width) != (
                    flash.image_height,
                    flash.image_width,
                ):
                    raise ValueError("standard and FlashGS camera resolution differs")
        model = None
        context = None
        base = None
        reflection = None
        flashgs_base = None
        flashgs_reflection = None
        try:
            model = load_checkpoint(
                checkpoint,
                contract=checkpoint_contract,
                prepare_flashgs=backend == "flashgs",
            )
            context = NativeRendererContext(device)
            base = prepare_gaussian(model.base)
            reflection = prepare_gaussian(
                model.reflection,
                exact_trace_activation=backend == "flashgs",
            )
            if backend == "flashgs":
                flashgs_base = prepare_flashgs_gaussian(model.base)
                flashgs_reflection = prepare_flashgs_gaussian(model.reflection)
            jt.sync_all(True)
            vertices, triangles = gaussian_disks(reflection)
            signature = int.from_bytes(
                hashlib.sha256(model.sha256.encode("ascii")).digest()[:8], "little"
            ) & ((1 << 63) - 1)
            build_optix_scene(
                vertices,
                triangles,
                scene_signature=signature or 1,
                rebuild=True,
                context=context,
            )
            del vertices, triangles
            gc.collect()
            jt.gc()
            return cls(
                model,
                cameras,
                context,
                base,
                reflection,
                backend=backend,
                flashgs_cameras=flashgs_cameras,
                flashgs_base=flashgs_base,
                flashgs_reflection=flashgs_reflection,
                synchronize_stages=synchronize_stages,
            )
        except BaseException:
            if context is not None:
                context.close()
            if base is not None:
                base.close()
            if reflection is not None:
                reflection.close()
            if flashgs_base is not None:
                flashgs_base.close()
            if flashgs_reflection is not None:
                flashgs_reflection.close()
            if model is not None:
                model.close()
            raise

    def _require_open(self) -> None:
        if self.closed:
            raise RuntimeError("HybridSplatSession is closed")

    def _render_view(
        self,
        camera_id: int,
        return_raw: bool,
        stage_timings: dict[str, float] | None,
    ) -> Any:
        import jittor as jt

        self._require_open()
        if isinstance(camera_id, bool) or not isinstance(camera_id, int):
            raise TypeError("camera_id must be an integer")
        if camera_id < 0 or camera_id >= len(self.cameras):
            raise IndexError(f"camera_id {camera_id} is outside the camera set")
        camera = materialize_camera(
            self.cameras[camera_id], context=self.context
        )
        if self.backend == "standard":
            outputs, intermediates = render_hybridsplat_view(
                base=self.base,
                reflection=self.reflection,
                background=self.checkpoint.background_color,
                reflection_background=self.checkpoint.reflection_background_color,
                camera=camera,
                context=self.context,
                synchronize_stages=self.synchronize_stages,
                stage_timings=stage_timings,
            )
            fetch_values = [outputs[name] for name in RAW_KEYS]
        else:
            if self.flashgs_base is None or self.flashgs_reflection is None:
                raise RuntimeError("FlashGS session has no prepared Gaussian data")
            outputs, intermediates = render_hybridsplat_flashgs_view(
                base=self.flashgs_base,
                reflection=self.flashgs_reflection,
                trace_reflection=self.reflection,
                reflection_background=self.checkpoint.reflection_background_color,
                flashgs_camera=self.flashgs_cameras[camera_id],
                viewpoint_camera=camera,
                context=self.context,
                synchronize_stages=self.synchronize_stages,
                stage_timings=stage_timings,
            )
            fetch_values = [outputs[name] for name in RAW_KEYS] + [
                intermediates["base_raster"].rendered,
                intermediates["reflection_raster"].rendered,
            ]
        fetch_started = time.perf_counter()
        fetched = jt.fetch_sync(fetch_values)
        if stage_timings is not None:
            stage_timings["fetch_seconds"] = time.perf_counter() - fetch_started
        raw = {
            name: np.ascontiguousarray(value, dtype=np.float32)
            for name, value in zip(RAW_KEYS, fetched[: len(RAW_KEYS)])
        }
        if self.backend == "flashgs":
            self.last_render_metadata = {
                "camera_id": camera_id,
                "base_num_rendered": int(np.asarray(fetched[-2]).reshape(-1)[0]),
                "reflection_num_rendered": int(np.asarray(fetched[-1]).reshape(-1)[0]),
                "flashgs_workspace": self.context.flashgs_workspace_stats(),
            }
        else:
            self.last_render_metadata = {"camera_id": camera_id}
        expected_pixels = camera.image_height * camera.image_width
        for name, value in raw.items():
            channels = 1 if name in ("acc_map", "dpt_map", "spec_map") else 3
            if value.shape != (1, expected_pixels, channels):
                raise RuntimeError(
                    f"{name} has invalid shape {value.shape}; expected "
                    f"{(1, expected_pixels, channels)}"
                )
            if value.dtype != np.float32 or not np.isfinite(value).all():
                raise RuntimeError(f"{name} is not finite FP32 output")
            raw[name] = np.ascontiguousarray(
                value.reshape(camera.image_height, camera.image_width, channels)
            )
        del outputs, intermediates, camera, fetched
        self.render_count += 1
        if self.render_count % 20 == 0:
            gc.collect()
            jt.gc()
            jt.gc()
        if return_raw:
            return raw
        return raw["rgb_map"]

    def render_view(self, camera_id: int, return_raw: bool = False) -> Any:
        return self._render_view(camera_id, return_raw, None)

    def render_view_profiled(self, camera_id: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        stage_timings: dict[str, float] = {}
        started = time.perf_counter()
        raw = self._render_view(camera_id, True, stage_timings)
        stage_timings["frame_seconds"] = time.perf_counter() - started
        return raw, stage_timings

    def memory_report(self) -> dict[str, Any]:
        self._require_open()
        return {
            "rasterizer": self.context.workspace_stats(),
            "math": self.context.math_workspace_stats(),
            "optix": self.context.optix_workspace_stats(),
            "flashgs": self.context.flashgs_workspace_stats(),
        }

    def close(self) -> None:
        if self.closed:
            return
        import jittor as jt

        first_error = None
        try:
            jt.sync_all(True)
        except BaseException as exc:
            first_error = exc
        try:
            self.context.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self.base.close()
        self.reflection.close()
        if self.flashgs_base is not None:
            self.flashgs_base.close()
        if self.flashgs_reflection is not None:
            self.flashgs_reflection.close()
        try:
            self.checkpoint.close()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        self.cameras = ()
        self.flashgs_cameras = ()
        self.closed = True
        gc.collect()
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "HybridSplatSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
