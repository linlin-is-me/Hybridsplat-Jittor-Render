# SPDX-License-Identifier: MIT
"""Convert a trusted HybridSplat PyTorch checkpoint to the strict NPZ format."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from jittor_hybridsplat.checkpoint import validate_checkpoint_archive


ROOT_KEYS = (
    "sampler.bg_color",
    "sampler.ref_bg_color",
)
GAUSSIAN_SUFFIXES = (
    "active_sh_degree",
    "_xyz",
    "_features_dc",
    "_features_rest",
    "_scaling",
    "_rotation",
    "_opacity",
    "_specular",
    "_roughness",
    "max_radii2D",
    "xyz_gradient_accum",
    "denom",
    "xyz_weight_accum",
)
MODEL_KEYS = ROOT_KEYS + tuple(
    f"{prefix}.{suffix}"
    for prefix in ("sampler.pcd", "sampler.ref")
    for suffix in GAUSSIAN_SUFFIXES
)


class ConversionError(ValueError):
    """The PyTorch checkpoint cannot satisfy the renderer checkpoint contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pytorch_checkpoint(path: Path, allow_unsafe_pickle: bool) -> Tuple[Any, str]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required only for checkpoint conversion. Run this command "
            "in the original HybridSplat PyTorch environment."
        ) from exc

    load_kwargs: Dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": not allow_unsafe_pickle,
    }
    try:
        if allow_unsafe_pickle:
            checkpoint = torch.load(path, **load_kwargs)
        else:
            # EasyVolcap stores the epoch and scheduler values as NumPy scalars.
            # Keep weights_only enabled and allow only the NumPy scalar/dtype
            # constructors required by the verified HybridSplat checkpoint.
            from numpy.core.multiarray import scalar as numpy_scalar

            safe_numpy_types = (
                numpy_scalar,
                np.dtype,
                type(np.dtype(np.float64)),
            )
            with torch.serialization.safe_globals(safe_numpy_types):
                checkpoint = torch.load(path, **load_kwargs)
    except TypeError as exc:
        if not allow_unsafe_pickle:
            raise RuntimeError(
                "this PyTorch version does not support safe weights_only loading; "
                "upgrade PyTorch or use --allow-unsafe-pickle only for a trusted file"
            ) from exc
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        if not allow_unsafe_pickle:
            raise RuntimeError(
                "safe PyTorch checkpoint loading failed. If this is your trusted "
                "checkpoint and it requires Python pickle objects, rerun with "
                "--allow-unsafe-pickle"
            ) from exc
        raise
    return checkpoint, str(torch.__version__)


def _scalar_integer(value: Any, name: str) -> int:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ConversionError(f"{name} must be an integer scalar")
    result = int(value)
    if result < 0:
        raise ConversionError(f"{name} must be non-negative")
    return result


def _to_numpy(value: Any, key: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    elif hasattr(value, "detach") and hasattr(value, "cpu"):
        tensor = value.detach().cpu()
        if hasattr(tensor, "contiguous"):
            tensor = tensor.contiguous()
        try:
            array = tensor.numpy()
        except Exception as exc:
            raise ConversionError(f"cannot convert {key} to NumPy") from exc
    else:
        raise ConversionError(f"{key} is not a tensor or ndarray")
    array = np.asarray(array)
    if array.dtype.byteorder == ">":
        raise ConversionError(f"{key} uses unsupported big-endian dtype {array.dtype}")
    return np.ascontiguousarray(array)


def _select_state_dict(checkpoint: Any, model_key: str) -> Mapping[str, Any]:
    if not isinstance(checkpoint, Mapping):
        raise ConversionError("checkpoint root must be a mapping")
    if model_key in checkpoint:
        state = checkpoint[model_key]
        if not isinstance(state, Mapping):
            raise ConversionError(f"checkpoint[{model_key!r}] must be a mapping")
        return state
    if all(key in checkpoint for key in MODEL_KEYS):
        return checkpoint
    raise ConversionError(
        f"checkpoint has no {model_key!r} state dict and is not a top-level "
        "HybridSplat state dict"
    )


def _resolve_prefix(state: Mapping[str, Any], requested: Optional[str]) -> str:
    if requested is not None:
        if not all(f"{requested}{key}" in state for key in MODEL_KEYS):
            missing = [key for key in MODEL_KEYS if f"{requested}{key}" not in state]
            raise ConversionError(
                f"state prefix {requested!r} is missing required keys: {missing}"
            )
        return requested
    for candidate in ("", "module."):
        if all(f"{candidate}{key}" in state for key in MODEL_KEYS):
            return candidate
    raise ConversionError(
        "cannot find the 28 required HybridSplat rendering keys; use "
        "--state-prefix if the state dict has a custom prefix"
    )


def extract_arrays(
    checkpoint: Any,
    *,
    model_key: str = "model",
    state_prefix: Optional[str] = None,
    epoch_override: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if not isinstance(checkpoint, Mapping):
        raise ConversionError("checkpoint root must be a mapping")
    state = _select_state_dict(checkpoint, model_key)
    prefix = _resolve_prefix(state, state_prefix)
    if epoch_override is None:
        if "epoch" not in checkpoint:
            raise ConversionError("checkpoint has no epoch; provide --epoch explicitly")
        epoch = _scalar_integer(checkpoint["epoch"], "epoch")
    else:
        epoch = _scalar_integer(epoch_override, "epoch")

    arrays = {key: _to_numpy(state[f"{prefix}{key}"], key) for key in MODEL_KEYS}
    arrays["epoch"] = np.asarray(epoch, dtype=np.dtype("<i8"))
    selected = {f"{prefix}{key}" for key in MODEL_KEYS}
    ignored = sorted(str(key) for key in state.keys() if str(key) not in selected)
    return arrays, {
        "epoch": epoch,
        "model_key": model_key if model_key in checkpoint else None,
        "state_prefix": prefix,
        "ignored_state_keys": ignored,
    }


def write_npz(
    arrays: Mapping[str, np.ndarray],
    output: Path,
    *,
    overwrite: bool = False,
) -> Any:
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".npz":
        raise ConversionError("output path must end with .npz")
    if output.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".npz", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        contract = validate_checkpoint_archive(temporary)
        os.replace(temporary, output)
        return contract
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a HybridSplat PyTorch checkpoint to strict renderer NPZ"
    )
    parser.add_argument("--input", type=Path, required=True, help="input .pt checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="output .npz file")
    parser.add_argument(
        "--model-key",
        default="model",
        help="checkpoint key containing the model state dict (default: model)",
    )
    parser.add_argument(
        "--state-prefix",
        default=None,
        help="optional prefix before state-dict keys; empty and module. are auto-detected",
    )
    parser.add_argument("--epoch", type=int, help="override a missing checkpoint epoch")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing NPZ")
    parser.add_argument(
        "--allow-unsafe-pickle",
        action="store_true",
        help="allow weights_only=False for a checkpoint that you fully trust",
    )
    parser.add_argument("--report", type=Path, help="optional conversion report JSON")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    source = arguments.input.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"input checkpoint does not exist: {source}")
    checkpoint, torch_version = _load_pytorch_checkpoint(
        source, arguments.allow_unsafe_pickle
    )
    arrays, metadata = extract_arrays(
        checkpoint,
        model_key=arguments.model_key,
        state_prefix=arguments.state_prefix,
        epoch_override=arguments.epoch,
    )
    checkpoint = None
    contract = write_npz(arrays, arguments.output, overwrite=arguments.overwrite)
    output = arguments.output.expanduser().resolve()
    report = {
        "schema_version": 1,
        "status": "complete",
        "input": str(source),
        "input_sha256": _sha256(source),
        "output": str(output),
        "output_sha256": contract.sha256,
        "output_bytes": contract.file_bytes,
        "epoch": contract.epoch,
        "base_points": contract.base_points,
        "reflection_points": contract.reflection_points,
        "sh_degree": contract.sh_degree,
        "array_count": len(arrays),
        "pytorch_version": torch_version,
        "safe_weights_only": not arguments.allow_unsafe_pickle,
        **metadata,
    }
    if arguments.report is not None:
        report_path = arguments.report.expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
