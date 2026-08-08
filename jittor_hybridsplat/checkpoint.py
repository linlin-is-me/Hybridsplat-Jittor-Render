from __future__ import annotations

import gc
import hashlib
import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


class CheckpointContractError(ValueError):
    """Checkpoint does not satisfy the HybridSplat inference contract."""


@dataclass(frozen=True)
class CheckpointContract:
    epoch: int
    base_points: int
    reflection_points: int
    sh_degree: int
    sha256: Optional[str] = None
    file_bytes: Optional[int] = None
    uncompressed_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")
        if self.base_points <= 0 or self.reflection_points <= 0:
            raise ValueError("point counts must be positive")
        if self.sh_degree < 0 or self.sh_degree > 3:
            raise ValueError("SH degree must be in [0, 3]")
        if self.sha256 is not None:
            if len(self.sha256) != 64:
                raise ValueError("SHA-256 must contain 64 hexadecimal characters")
            try:
                int(self.sha256, 16)
            except ValueError as exc:
                raise ValueError("SHA-256 must be hexadecimal") from exc
        for name, value in (
            ("file_bytes", self.file_bytes),
            ("uncompressed_bytes", self.uncompressed_bytes),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def max_sh_channels(self) -> int:
        return (self.sh_degree + 1) ** 2

@dataclass(frozen=True)
class ArraySpec:
    source_key: str
    target_name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    group: Optional[str] = None
    exact_value: Optional[int] = None


def _group_specs(
    prefix: str,
    target_group: str,
    points: int,
    degree: int,
) -> Iterable[ArraySpec]:
    rest = (degree + 1) ** 2 - 1
    yield ArraySpec(
        f"{prefix}.active_sh_degree",
        "active_sh_degree",
        (1,),
        np.dtype("<i8"),
        target_group,
        degree,
    )
    yield ArraySpec(f"{prefix}._xyz", "xyz_raw", (points, 3), np.dtype("<f4"), target_group)
    yield ArraySpec(
        f"{prefix}._features_dc",
        "features_dc_raw",
        (points, 1, 3),
        np.dtype("<f4"),
        target_group,
    )
    yield ArraySpec(
        f"{prefix}._features_rest",
        "features_rest_raw",
        (points, rest, 3),
        np.dtype("<f4"),
        target_group,
    )
    yield ArraySpec(
        f"{prefix}._scaling", "scaling_raw", (points, 2), np.dtype("<f4"), target_group
    )
    yield ArraySpec(
        f"{prefix}._rotation", "rotation_raw", (points, 4), np.dtype("<f4"), target_group
    )
    for source, target in (
        ("_opacity", "opacity_raw"),
        ("_specular", "specular_raw"),
        ("_roughness", "roughness_raw"),
    ):
        yield ArraySpec(
            f"{prefix}.{source}", target, (points, 1), np.dtype("<f4"), target_group
        )
    yield ArraySpec(
        f"{prefix}.max_radii2D", "max_radii2d", (points,), np.dtype("<f4"), target_group
    )
    for source in ("xyz_gradient_accum", "denom", "xyz_weight_accum"):
        yield ArraySpec(
            f"{prefix}.{source}", source, (points, 1), np.dtype("<f4"), target_group
        )


def checkpoint_specs(
    contract: CheckpointContract,
) -> Tuple[ArraySpec, ...]:
    specs = [
        ArraySpec("sampler.bg_color", "background_color", (3,), np.dtype("<f4")),
        ArraySpec(
            "sampler.ref_bg_color",
            "reflection_background_color",
            (3,),
            np.dtype("<f4"),
        ),
    ]
    specs.extend(
        _group_specs("sampler.pcd", "base", contract.base_points, contract.sh_degree)
    )
    specs.extend(
        _group_specs(
            "sampler.ref",
            "reflection",
            contract.reflection_points,
            contract.sh_degree,
        )
    )
    return tuple(specs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_checkpoint_contract(
    path: os.PathLike[str] | str,
    *,
    pin_sha256: bool = True,
) -> CheckpointContract:
    """Infer point counts and SH degree, then pin the checkpoint identity."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise CheckpointContractError(f"checkpoint does not exist: {resolved}")
    required = {
        "epoch",
        "sampler.pcd.active_sh_degree",
        "sampler.pcd._xyz",
        "sampler.pcd._features_rest",
        "sampler.ref.active_sh_degree",
        "sampler.ref._xyz",
        "sampler.ref._features_rest",
    }
    try:
        with np.load(resolved, allow_pickle=False) as archive:
            missing = sorted(required - set(archive.files))
            if missing:
                raise CheckpointContractError(
                    f"checkpoint contract fields are missing: {missing}"
                )
            epoch_array = archive["epoch"]
            if epoch_array.shape != () or epoch_array.dtype.str != "<i8":
                raise CheckpointContractError(
                    "epoch must be a little-endian int64 scalar"
                )
            point_counts = []
            degrees = []
            for prefix in ("sampler.pcd", "sampler.ref"):
                xyz = archive[f"{prefix}._xyz"]
                rest = archive[f"{prefix}._features_rest"]
                active = archive[f"{prefix}.active_sh_degree"]
                if xyz.ndim != 2 or xyz.shape[1] != 3:
                    raise CheckpointContractError(f"{prefix}._xyz must have shape [P,3]")
                if rest.ndim != 3 or rest.shape[0] != xyz.shape[0] or rest.shape[2] != 3:
                    raise CheckpointContractError(
                        f"{prefix}._features_rest must have shape [P,R,3]"
                    )
                if active.shape != (1,) or active.dtype.str != "<i8":
                    raise CheckpointContractError(
                        f"{prefix}.active_sh_degree must be int64 with shape [1]"
                    )
                channel_count = int(rest.shape[1]) + 1
                root = math.isqrt(channel_count)
                if root * root != channel_count:
                    raise CheckpointContractError(
                        f"{prefix} SH channel count is not a perfect square"
                    )
                degree = root - 1
                if int(active[0]) != degree:
                    raise CheckpointContractError(
                        f"{prefix} active SH degree does not match stored features"
                    )
                point_counts.append(int(xyz.shape[0]))
                degrees.append(degree)
            if degrees[0] != degrees[1]:
                raise CheckpointContractError(
                    "base and reflection Gaussians use different SH degrees"
                )
            epoch = int(epoch_array)
    except CheckpointContractError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CheckpointContractError(f"cannot infer checkpoint contract: {exc}") from exc

    try:
        with zipfile.ZipFile(resolved, "r") as archive:
            uncompressed_bytes = sum(info.file_size for info in archive.infolist())
    except (OSError, zipfile.BadZipFile) as exc:
        raise CheckpointContractError(f"invalid checkpoint ZIP archive: {exc}") from exc
    return CheckpointContract(
        epoch=epoch,
        base_points=point_counts[0],
        reflection_points=point_counts[1],
        sh_degree=degrees[0],
        sha256=_sha256(resolved) if pin_sha256 else None,
        file_bytes=resolved.stat().st_size,
        uncompressed_bytes=uncompressed_bytes,
    )


def _validate_path(path: os.PathLike[str] | str, contract: CheckpointContract) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise CheckpointContractError(f"checkpoint does not exist: {resolved}")
    if not resolved.is_file():
        raise CheckpointContractError(f"checkpoint is not a regular file: {resolved}")
    size = resolved.stat().st_size
    if contract.file_bytes is not None and size != contract.file_bytes:
        raise CheckpointContractError(
            f"checkpoint byte size mismatch: expected {contract.file_bytes}, got {size}"
        )
    if contract.sha256 is not None:
        actual = _sha256(resolved)
        if actual != contract.sha256:
            raise CheckpointContractError(
                f"checkpoint SHA-256 mismatch: expected {contract.sha256}, got {actual}"
            )
    return resolved


def _validate_archive(
    path: Path,
    specs: Tuple[ArraySpec, ...],
    contract: CheckpointContract,
) -> None:
    expected = {f"{spec.source_key}.npy" for spec in specs}
    expected.add("epoch.npy")
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                duplicates = sorted({name for name in names if names.count(name) > 1})
                raise CheckpointContractError(
                    f"duplicate ZIP members: {', '.join(duplicates)}"
                )
            if set(names) != expected:
                missing = sorted(expected - set(names))
                extra = sorted(set(names) - expected)
                raise CheckpointContractError(
                    f"ZIP member set mismatch; missing={missing}, extra={extra}"
                )
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    info.is_dir()
                    or member.is_absolute()
                    or len(member.parts) != 1
                    or ".." in member.parts
                ):
                    raise CheckpointContractError(f"unsafe ZIP member: {info.filename}")
                if info.flag_bits & 0x1:
                    raise CheckpointContractError(
                        f"encrypted ZIP member is forbidden: {info.filename}"
                    )
                if info.compress_type != zipfile.ZIP_DEFLATED:
                    raise CheckpointContractError(
                        f"ZIP member is not DEFLATE-compressed: {info.filename}"
                    )
            total = sum(info.file_size for info in infos)
            if (
                contract.uncompressed_bytes is not None
                and total != contract.uncompressed_bytes
            ):
                raise CheckpointContractError(
                    "ZIP uncompressed byte size mismatch: "
                    f"expected {contract.uncompressed_bytes}, got {total}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise CheckpointContractError(f"ZIP CRC failure: {bad_member}")
    except CheckpointContractError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise CheckpointContractError(f"invalid checkpoint ZIP archive: {exc}") from exc


def _validate_array(array: np.ndarray, spec: ArraySpec) -> None:
    if not isinstance(array, np.ndarray):
        raise CheckpointContractError(f"{spec.source_key} is not an ndarray")
    if array.shape != spec.shape:
        raise CheckpointContractError(
            f"{spec.source_key} shape mismatch: expected {spec.shape}, got {array.shape}"
        )
    if array.dtype.str != spec.dtype.str:
        raise CheckpointContractError(
            f"{spec.source_key} dtype mismatch: expected {spec.dtype.str}, "
            f"got {array.dtype.str}"
        )
    if not array.flags.c_contiguous:
        raise CheckpointContractError(f"{spec.source_key} must be C-contiguous")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise CheckpointContractError(f"{spec.source_key} contains NaN or Inf")
    if spec.exact_value is not None and not np.all(array == spec.exact_value):
        raise CheckpointContractError(
            f"{spec.source_key} must equal {spec.exact_value}"
        )


def _read_and_validate_arrays(
    path: Path,
    specs: Tuple[ArraySpec, ...],
    contract: CheckpointContract,
) -> Dict[str, np.ndarray]:
    expected_keys = {spec.source_key for spec in specs}
    expected_keys.add("epoch")
    arrays: Dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            actual_keys = set(archive.files)
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)
                extra = sorted(actual_keys - expected_keys)
                raise CheckpointContractError(
                    f"NPZ key set mismatch; missing={missing}, extra={extra}"
                )
            epoch = archive["epoch"]
            if epoch.shape != () or epoch.dtype.str != "<i8":
                raise CheckpointContractError(
                    "epoch must be a little-endian int64 scalar"
                )
            if int(epoch) != contract.epoch:
                raise CheckpointContractError(
                    f"epoch mismatch: expected {contract.epoch}, got {int(epoch)}"
                )
            arrays["epoch"] = epoch
            for spec in specs:
                array = archive[spec.source_key]
                _validate_array(array, spec)
                arrays[spec.source_key] = array
    except CheckpointContractError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CheckpointContractError(f"failed to read checkpoint arrays: {exc}") from exc
    return arrays


def validate_checkpoint_archive(
    path: os.PathLike[str] | str,
    *,
    pin_sha256: bool = True,
) -> CheckpointContract:
    """Validate an NPZ checkpoint without importing or materializing Jittor Vars."""

    contract = infer_checkpoint_contract(path, pin_sha256=pin_sha256)
    resolved = _validate_path(path, contract)
    specs = checkpoint_specs(contract)
    _validate_archive(resolved, specs, contract)
    arrays = _read_and_validate_arrays(resolved, specs, contract)
    arrays.clear()
    return contract


def _require_jittor():
    try:
        import jittor as jt
    except ImportError as exc:
        raise RuntimeError("Jittor is required to materialize the checkpoint") from exc
    return jt


def _to_var(jt, array: np.ndarray):
    expected_dtype = "int64" if array.dtype.str == "<i8" else "float32"
    var = jt.array(array)
    var.stop_grad()
    if str(var.dtype) != expected_dtype:
        raise CheckpointContractError(
            f"Jittor dtype mismatch: expected {expected_dtype}, got {var.dtype}"
        )
    if list(var.shape) != list(array.shape):
        raise CheckpointContractError(
            f"Jittor shape mismatch: expected {array.shape}, got {var.shape}"
        )
    if not var.is_stop_grad():
        raise CheckpointContractError("Jittor Var is not stop_grad")
    return var


class GaussianParameters:
    def __init__(
        self,
        name: str,
        point_count: int,
        sh_degree: int,
        values: Mapping[str, object],
    ) -> None:
        self.name = name
        self.point_count = point_count
        self.max_sh_degree = sh_degree
        self.active_sh_degree_value = sh_degree
        for target_name, value in values.items():
            setattr(self, target_name, value)

    @property
    def features(self):
        jt = _require_jittor()
        return jt.concat((self.features_dc_raw, self.features_rest_raw), dim=1)

    @property
    def xyz(self):
        return self.xyz_raw

    @property
    def scaling(self):
        from .ops.precise_exp import precise_exp

        return precise_exp(self.scaling_raw)

    @property
    def rotation(self):
        from .math import safe_normalize

        return safe_normalize(self.rotation_raw, eps=1e-12, clamp=True)

    @property
    def opacity(self):
        return self.opacity_raw.sigmoid()

    @property
    def specular(self):
        return self.specular_raw.sigmoid()

    @property
    def roughness(self):
        return self.roughness_raw.sigmoid()

    def close(self) -> None:
        for name in (
            "active_sh_degree",
            "xyz_raw",
            "features_dc_raw",
            "features_rest_raw",
            "scaling_raw",
            "rotation_raw",
            "opacity_raw",
            "specular_raw",
            "roughness_raw",
            "max_radii2d",
            "xyz_gradient_accum",
            "denom",
            "xyz_weight_accum",
            "flashgs_opacity",
            "flashgs_scaling",
            "flashgs_rotation",
            "flashgs_covariance",
            "flashgs_specular",
            "flashgs_roughness",
        ):
            if hasattr(self, name):
                setattr(self, name, None)


class HybridSplatCheckpoint:
    def __init__(
        self,
        path: Path,
        sha256: str,
        epoch: int,
        background_color,
        reflection_background_color,
        base: GaussianParameters,
        reflection: GaussianParameters,
    ) -> None:
        self.path = path
        self.sha256 = sha256
        self.epoch = epoch
        self.background_color = background_color
        self.reflection_background_color = reflection_background_color
        self.base = base
        self.reflection = reflection
        self.closed = False

    def close(self) -> None:
        if self.closed:
            return
        jt = _require_jittor()
        first_error = None
        try:
            jt.sync_all(True)
        except BaseException as exc:
            first_error = exc
        finally:
            self.base.close()
            self.reflection.close()
            self.background_color = None
            self.reflection_background_color = None
            self.closed = True
            gc.collect()
            try:
                jt.clean()
            except BaseException as cleanup_error:
                if first_error is None:
                    first_error = cleanup_error
                elif hasattr(first_error, "add_note"):
                    first_error.add_note(f"checkpoint cleanup also failed: {cleanup_error}")
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "HybridSplatCheckpoint":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def load_checkpoint(
    path: os.PathLike[str] | str,
    *,
    contract: CheckpointContract | None = None,
    require_cuda: bool = True,
    prepare_flashgs: bool = False,
) -> HybridSplatCheckpoint:
    """Validate the complete NPZ contract before creating any Jittor Var."""

    if contract is None:
        contract = infer_checkpoint_contract(path)
    specs = checkpoint_specs(contract)
    resolved = _validate_path(path, contract)
    _validate_archive(resolved, specs, contract)
    arrays = _read_and_validate_arrays(resolved, specs, contract)
    loaded_sha256 = _sha256(resolved)
    if contract.sha256 is not None and loaded_sha256 != contract.sha256:
        arrays.clear()
        raise CheckpointContractError(
            "checkpoint changed while it was being read: "
            f"expected {contract.sha256}, got {loaded_sha256}"
        )
    jt = _require_jittor()
    if require_cuda and int(jt.flags.use_cuda) != 1:
        arrays.clear()
        raise CheckpointContractError(
            "Jittor CUDA is disabled; set jt.flags.use_cuda=1 before loading"
        )
    created = []
    root_values: Dict[str, object] = {}
    group_values: Dict[str, Dict[str, object]] = {"base": {}, "reflection": {}}
    checkpoint = None
    var = None
    flashgs_cpu = []
    try:
        with jt.flag_scope(amp_reg=0, auto_convert_64_to_32=0):
            for spec in specs:
                var = _to_var(jt, arrays[spec.source_key])
                created.append(var)
                if spec.group is None:
                    root_values[spec.target_name] = var
                else:
                    group_values[spec.group][spec.target_name] = var
            if prepare_flashgs:
                from .ops.flashgs_prepare import prepare_flashgs_npz_host

                for group, prefix in (("base", "sampler.pcd"), ("reflection", "sampler.ref")):
                    prepared = prepare_flashgs_npz_host(
                        arrays[f"{prefix}._opacity"],
                        arrays[f"{prefix}._scaling"],
                        arrays[f"{prefix}._rotation"],
                        arrays[f"{prefix}._specular"],
                        arrays[f"{prefix}._roughness"],
                    )
                    flashgs_cpu.extend(prepared)
                    for name, array in zip(
                        (
                            "flashgs_opacity",
                            "flashgs_scaling",
                            "flashgs_rotation",
                            "flashgs_covariance",
                            "flashgs_specular",
                            "flashgs_roughness",
                        ),
                        prepared,
                    ):
                        value = _to_var(jt, array)
                        created.append(value)
                        group_values[group][name] = value
        jt.sync_all(True)
        checkpoint = HybridSplatCheckpoint(
            path=resolved,
            sha256=loaded_sha256,
            epoch=int(arrays["epoch"]),
            background_color=root_values["background_color"],
            reflection_background_color=root_values["reflection_background_color"],
            base=GaussianParameters(
                "base",
                contract.base_points,
                contract.sh_degree,
                group_values["base"],
            ),
            reflection=GaussianParameters(
                "reflection",
                contract.reflection_points,
                contract.sh_degree,
                group_values["reflection"],
            ),
        )
    except BaseException as original_error:
        cleanup_errors = []
        try:
            jt.sync_all(True)
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        checkpoint = None
        var = None
        root_values.clear()
        group_values["base"].clear()
        group_values["reflection"].clear()
        group_values.clear()
        created.clear()
        flashgs_cpu.clear()
        gc.collect()
        try:
            jt.clean()
        except BaseException as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors and hasattr(original_error, "add_note"):
            for cleanup_error in cleanup_errors:
                original_error.add_note(
                    f"checkpoint cleanup also failed: {cleanup_error}"
                )
        raise
    finally:
        arrays.clear()
        flashgs_cpu.clear()
        gc.collect()
        if checkpoint is not None:
            jt.clean()
    if checkpoint is None:
        raise RuntimeError("checkpoint construction ended without a result")
    return checkpoint
