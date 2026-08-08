from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from .session import HybridSplatSession, RAW_KEYS


CONFIG_KEYS = {
    "dataset_root",
    "ratio",
    "near",
    "far",
    "views",
    "backend",
    "flashgs_camera_path",
    "dump_raw",
    "evaluate",
    "reference",
    "stage_sync",
}


def _config_path(value: object, *, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config {name} must be a nonempty path string")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_render_config(path: Path) -> dict[str, object]:
    """Load the strict JSON-compatible YAML subset used by formal configs."""

    config_path = path.expanduser().resolve()
    try:
        values = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            "render config must use the JSON-compatible YAML subset"
        ) from exc
    if not isinstance(values, dict):
        raise ValueError("render config root must be a mapping")
    unknown = sorted(set(values) - CONFIG_KEYS)
    missing = sorted(CONFIG_KEYS - set(values))
    if unknown or missing:
        raise ValueError(
            f"render config keys mismatch; missing={missing}, unknown={unknown}"
        )
    result: dict[str, object] = dict(values)
    result["dataset_root"] = _config_path(
        values["dataset_root"], name="dataset_root", base=config_path.parent
    )
    reference = values["reference"]
    result["reference"] = (
        None
        if reference is None
        else _config_path(reference, name="reference", base=config_path.parent)
    )
    flashgs_camera_path = values["flashgs_camera_path"]
    result["flashgs_camera_path"] = (
        None
        if flashgs_camera_path is None
        else _config_path(
            flashgs_camera_path,
            name="flashgs_camera_path",
            base=config_path.parent,
        )
    )
    for name in ("ratio", "near", "far"):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"config {name} must be numeric")
        result[name] = float(value)
    if result["ratio"] <= 0.0 or result["near"] <= 0.0 or result["far"] <= result["near"]:
        raise ValueError("config ratio and clipping range are invalid")
    views = values["views"]
    if isinstance(views, str):
        result["views"] = parse_views(views)
    elif isinstance(views, list) and all(isinstance(item, int) and not isinstance(item, bool) for item in views):
        result["views"] = parse_views(",".join(str(item) for item in views))
    else:
        raise ValueError("config views must be a range string or integer list")
    if values["backend"] not in ("standard", "flashgs"):
        raise ValueError("config backend must be standard or flashgs")
    if values["backend"] == "standard" and result["flashgs_camera_path"] is not None:
        raise ValueError("standard backend cannot set flashgs_camera_path")
    if values["backend"] == "flashgs" and result["flashgs_camera_path"] is None:
        default_camera = result["dataset_root"] / "cameras.json"
        if not default_camera.is_file():
            raise ValueError("flashgs backend requires flashgs_camera_path when dataset_root/cameras.json is absent")
    for name in ("dump_raw", "evaluate", "stage_sync"):
        if not isinstance(values[name], bool):
            raise ValueError(f"config {name} must be boolean")
    if values["evaluate"] and result["reference"] is None:
        raise ValueError("config evaluation requires a reference path")
    return result


def parse_views(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("views cannot be empty")
    try:
        if ":" in text:
            fields = text.split(":")
            if len(fields) not in (2, 3):
                raise ValueError
            start, stop = int(fields[0]), int(fields[1])
            step = int(fields[2]) if len(fields) == 3 else 1
            if step == 0:
                raise ValueError
            result = tuple(range(start, stop, step))
        else:
            result = tuple(int(field.strip()) for field in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "views must be comma-separated integers or start:stop[:step]"
        ) from exc
    if not result or len(result) != len(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError(
            "views must be nonempty, unique, and nonnegative"
        )
    return result


def _frame_name(view: int, suffix: str) -> str:
    return f"frame0000_camera{view:04d}{suffix}"


def _save_rgb(path: Path, raw: np.ndarray, height: int, width: int) -> None:
    image = raw.reshape(height, width, 3)
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded, mode="RGB").save(path)


def _find_reference(reference: Path, folder: str, filename: str) -> Path | None:
    direct = reference / folder / filename
    if direct.is_file():
        return direct
    matches = [
        path
        for path in reference.rglob(f"{folder}/{filename}")
        if "smoke" not in path.parts
    ]
    return matches[0] if len(matches) == 1 else None


def _raw_comparison(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    if candidate.shape != reference.shape or candidate.dtype != reference.dtype:
        raise ValueError("reference raw array metadata does not match candidate")
    delta = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    mse = float(np.mean(delta * delta))
    return {
        "mae": float(delta.mean()),
        "p99_abs": float(np.quantile(delta, 0.99)),
        "max_abs": float(delta.max(initial=0.0)),
        "psnr_db": math.inf if mse == 0.0 else float(-10.0 * math.log10(mse)),
    }


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _command_output(command: Iterable[str]) -> str:
    try:
        return subprocess.check_output(tuple(command), text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> str:
    import jittor as jt

    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"jittor={jt.__version__}",
        f"numpy={np.__version__}",
        f"use_cuda={int(jt.flags.use_cuda)}",
        f"use_tensorcore={int(jt.flags.use_tensorcore)}",
        f"amp_reg={int(jt.flags.amp_reg)}",
        f"cache_name={os.environ.get('cache_name', '')}",
        "nvcc=" + _command_output((os.environ.get("nvcc_path", "nvcc"), "--version")),
    ]
    return "\n".join(lines) + "\n"


def render_views(
    *,
    checkpoint: Path,
    dataset_root: Path,
    views: tuple[int, ...],
    output: Path,
    dump_raw: bool,
    reference: Path | None,
    synchronize_stages: bool,
    backend: str = "standard",
    flashgs_camera_path: Path | None = None,
    ratio: float = 0.25,
    near: float = 0.2,
    far: float = 20.0,
    config_path: Path | None = None,
) -> dict[str, object]:
    output = output.expanduser().resolve()
    renders = output / "renders"
    ground_truth = output / "ground_truth"
    raw_dir = output / "raw"
    for folder in (output, renders, ground_truth, raw_dir):
        folder.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    log_lines = []
    started = time.perf_counter()
    memory_report: dict[str, object] = {}
    with HybridSplatSession.from_npz(
        checkpoint,
        dataset_root,
        backend=backend,
        flashgs_camera_path=flashgs_camera_path,
        ratio=ratio,
        near=near,
        far=far,
        synchronize_stages=synchronize_stages,
    ) as session:
        invalid = [view for view in views if view >= len(session.cameras)]
        if invalid:
            raise ValueError(f"views are outside the dataset camera set: {invalid}")
        memory_report["after_session_create"] = session.memory_report()
        for view in views:
            view_started = time.perf_counter()
            filename = _frame_name(view, "")
            try:
                raw = session.render_view(view, return_raw=True)
                record = session.cameras[view]
                if dump_raw:
                    np.savez_compressed(raw_dir / f"{filename}.npz", **raw)
                _save_rgb(
                    renders / f"{filename}.png",
                    raw["rgb_map"],
                    record.image_height,
                    record.image_width,
                )
                item: dict[str, object] = {
                    "camera": view,
                    "seconds": time.perf_counter() - view_started,
                    "finite": bool(all(np.isfinite(value).all() for value in raw.values())),
                }
                if session.last_render_metadata is not None:
                    item["backend_metadata"] = session.last_render_metadata
                if reference is not None:
                    reference_raw = _find_reference(
                        reference, "RAW", f"{filename}.npz"
                    ) or _find_reference(reference, "raw", f"{filename}.npz")
                    if reference_raw is None:
                        raise FileNotFoundError(f"reference raw is missing for view {view}")
                    with np.load(reference_raw, allow_pickle=False) as source:
                        item["rgb_reference"] = _raw_comparison(
                            raw["rgb_map"], source["rgb_map"]
                        )
                    gt = _find_reference(
                        reference, "RENDER", f"{filename}_gt.png"
                    ) or _find_reference(
                        reference, "ground_truth", f"{filename}.png"
                    )
                    if gt is not None:
                        shutil.copy2(gt, ground_truth / f"{filename}.png")
                metrics.append(item)
                log_lines.append(
                    f"camera={view} status=ok seconds={item['seconds']:.6f}"
                )
                del raw
            except BaseException as exc:
                failures.append(
                    {
                        "camera": view,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                log_lines.append(
                    f"camera={view} status=failed type={type(exc).__name__} message={exc}"
                )
        memory_report["before_session_close"] = session.memory_report()

    rgb_items = [item.get("rgb_reference") for item in metrics if "rgb_reference" in item]
    summary: dict[str, object] = {
        "requested_views": len(views),
        "successful_views": len(metrics),
        "failed_views": len(failures),
        "total_seconds": time.perf_counter() - started,
    }
    if rgb_items:
        summary["rgb_reference"] = {
            key: float(np.mean([item[key] for item in rgb_items]))
            for key in ("mae", "p99_abs", "max_abs", "psnr_db")
        }
        summary["rgb_reference"]["psnr_each_min_db"] = float(
            min(item["psnr_db"] for item in rgb_items)
        )
    report = {"summary": summary, "metrics": metrics}
    manifest = {
        "schema_version": 1,
        "status": "complete" if not failures and len(metrics) == len(views) else "failed",
        "framework": "Jittor 1.3.11.0",
        "backend": backend,
        "checkpoint": str(checkpoint.expanduser().resolve()),
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "views": list(views),
        "raw_keys": list(RAW_KEYS),
        "raw_dtype": "float32",
        "synchronize_stages": synchronize_stages,
        "config": None if config_path is None else str(config_path.resolve()),
    }
    if backend == "flashgs":
        resolved_camera = (
            dataset_root.expanduser().resolve() / "cameras.json"
            if flashgs_camera_path is None
            else flashgs_camera_path.expanduser().resolve()
        )
        manifest["flashgs_camera_path"] = str(resolved_camera)
        manifest["flashgs_camera_sha256"] = _sha256(resolved_camera)
        manifest["flashgs_sort_order"] = "tile_depth_then_gaussian_index"
    (output / "metrics.json").write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (output / "failed_views.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    (output / "memory_report.json").write_text(
        json.dumps(memory_report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "environment.txt").write_text(_environment(), encoding="utf-8")
    (output / "render.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"{len(failures)} views failed; see failed_views.json")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render HybridSplat with Jittor")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--views", type=parse_views)
    parser.add_argument("--backend", choices=("standard", "flashgs"))
    parser.add_argument("--flashgs-camera-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-raw", action="store_true", default=None)
    parser.add_argument("--evaluate", action="store_true", default=None)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--stage-sync", action="store_true", default=None)
    parser.add_argument("--no-stage-sync", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    config: dict[str, object] = {
        "dataset_root": None,
        "ratio": 0.25,
        "near": 0.2,
        "far": 20.0,
        "views": parse_views("0:160:8"),
        "backend": "standard",
        "flashgs_camera_path": None,
        "dump_raw": False,
        "evaluate": False,
        "reference": None,
        "stage_sync": False,
    }
    if arguments.config is not None:
        if not arguments.config.is_file():
            raise SystemExit(f"config file does not exist: {arguments.config}")
        config.update(load_render_config(arguments.config))
    if arguments.dataset_root is not None:
        config["dataset_root"] = arguments.dataset_root.resolve()
    if arguments.views is not None:
        config["views"] = arguments.views
    if arguments.backend is not None:
        config["backend"] = arguments.backend
    if arguments.flashgs_camera_path is not None:
        config["flashgs_camera_path"] = arguments.flashgs_camera_path.resolve()
    if arguments.dump_raw is not None:
        config["dump_raw"] = arguments.dump_raw
    if arguments.evaluate is not None:
        config["evaluate"] = arguments.evaluate
    if arguments.reference is not None:
        config["reference"] = arguments.reference.resolve()
    if arguments.stage_sync and arguments.no_stage_sync:
        raise SystemExit("--stage-sync and --no-stage-sync are mutually exclusive")
    if arguments.stage_sync is not None:
        config["stage_sync"] = arguments.stage_sync
    if arguments.no_stage_sync:
        config["stage_sync"] = False
    if config["dataset_root"] is None:
        raise SystemExit("dataset root is required in config or --dataset-root")
    if config["evaluate"] and config["reference"] is None:
        raise SystemExit("evaluation requires a reference path")
    if config["backend"] == "standard" and config["flashgs_camera_path"] is not None:
        raise SystemExit("--flashgs-camera-path is only valid for --backend flashgs")
    report = render_views(
        checkpoint=arguments.checkpoint,
        dataset_root=config["dataset_root"],
        views=config["views"],
        output=arguments.output,
        dump_raw=config["dump_raw"],
        reference=config["reference"] if config["evaluate"] else None,
        synchronize_stages=config["stage_sync"],
        backend=config["backend"],
        flashgs_camera_path=config["flashgs_camera_path"],
        ratio=config["ratio"],
        near=config["near"],
        far=config["far"],
        config_path=arguments.config,
    )
    print(json.dumps(_json_safe(report["summary"]), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
