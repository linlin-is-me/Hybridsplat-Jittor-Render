#!/usr/bin/env python3
"""Validate the Python environment, checkpoint, cameras, and native backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--native", action="store_true")
    arguments = parser.parse_args()

    report: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "python": sys.version.split()[0],
        "checks": {},
        "errors": [],
    }
    checks = report["checks"]
    errors = report["errors"]
    try:
        import jittor as jt

        checks["jittor"] = {
            "version": jt.__version__,
            "verified_baseline": jt.__version__ == "1.3.11.0",
        }
        if jt.__version__ != "1.3.11.0":
            errors.append(
                f"Jittor 1.3.11.0 is required, active version is {jt.__version__}"
            )
    except BaseException as exc:
        errors.append(f"Jittor import failed: {type(exc).__name__}: {exc}")

    if arguments.checkpoint is not None:
        try:
            from jittor_hybridsplat.checkpoint import infer_checkpoint_contract

            contract = infer_checkpoint_contract(arguments.checkpoint)
            checks["checkpoint"] = {
                "path": str(arguments.checkpoint.expanduser().resolve()),
                "epoch": contract.epoch,
                "base_points": contract.base_points,
                "reflection_points": contract.reflection_points,
                "sh_degree": contract.sh_degree,
                "sha256": contract.sha256,
            }
        except BaseException as exc:
            errors.append(f"checkpoint validation failed: {type(exc).__name__}: {exc}")

    if arguments.dataset_root is not None:
        try:
            from jittor_hybridsplat.camera import load_camera_records

            cameras = load_camera_records(arguments.dataset_root)
            checks["cameras"] = {
                "dataset_root": str(arguments.dataset_root.expanduser().resolve()),
                "count": len(cameras),
                "first": cameras[0].name,
            }
        except BaseException as exc:
            errors.append(f"camera validation failed: {type(exc).__name__}: {exc}")

    if arguments.native:
        try:
            from jittor_hybridsplat.ops.native_runtime import NativeRendererContext

            with NativeRendererContext(0) as context:
                checks["native"] = {
                    "build_info": context.build_info,
                    "math_workspace": context.math_workspace_stats(),
                }
        except BaseException as exc:
            errors.append(f"native backend validation failed: {type(exc).__name__}: {exc}")

    if errors:
        report["status"] = "failed"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
