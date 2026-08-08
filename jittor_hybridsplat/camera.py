from __future__ import annotations

import math
import numbers
import re
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _jt():
    import jittor as jt

    return jt


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def _cpu_focal(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a CPU real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def projection_matrix(fov_x: float, fov_y: float, near, far):
    jt = _jt()
    near = near.reshape((1,))
    far = far.reshape((1,))
    tan_x = math.tan(fov_x * 0.5)
    tan_y = math.tan(fov_y * 0.5)
    zero, one = near * 0.0, near * 0.0 + 1.0
    top = tan_y * near
    bottom = -top
    right = tan_x * near
    left = -right
    p00 = 2.0 * near / (right - left)
    p11 = 2.0 * near / (top - bottom)
    p02 = (right + left) / (right - left)
    p12 = (top + bottom) / (top - bottom)
    p22, p23 = far / (far - near), -(far * near) / (far - near)
    row0 = jt.concat((p00, zero, p02, zero), dim=0)
    row1 = jt.concat((zero, p11, p12, zero), dim=0)
    row2 = jt.concat((zero, zero, p22, p23), dim=0)
    row3 = jt.concat((zero, zero, one, zero), dim=0)
    return jt.stack((row0, row1, row2, row3), dim=0)


def world_to_view(rotation, translation):
    jt = _jt()
    upper = jt.concat((rotation, translation), dim=-1)
    scalar = translation[0:1, 0].reshape((1,))
    zero, one = scalar * 0.0, scalar * 0.0 + 1.0
    bottom = jt.concat((zero, zero, zero, one), dim=0).reshape((1, 4))
    return jt.concat((upper, bottom), dim=-2)


@dataclass
class GaussianCamera:
    image_height: int
    image_width: int
    K: object
    R: object
    T: object
    near: object
    far: object
    znear: object
    zfar: object
    FoVx: float
    FoVy: float
    tanfovx: float
    tanfovy: float
    world_view_transform: object
    projection_matrix: object
    full_proj_transform: object
    camera_center: object


@dataclass(frozen=True)
class CameraRecord:
    camera_id: int
    name: str
    image_height: int
    image_width: int
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray
    near: np.ndarray
    far: np.ndarray


@dataclass(frozen=True)
class FlashgsCameraRecord:
    camera_id: int
    name: str
    image_height: int
    image_width: int
    position: np.ndarray
    rotation: np.ndarray
    focal_x: float
    focal_y: float
    znear: float = 0.01
    zfar: float = 100.0


_MATRIX_PATTERN = re.compile(
    r"(?ms)^(?P<kind>K|Rot|T)_(?P<name>[^:\r\n]+):\s*!!opencv-matrix\s*"
    r"rows:\s*(?P<rows>\d+)\s*cols:\s*(?P<cols>\d+)\s*"
    r"dt:\s*[^\r\n]+\s*data:\s*\[(?P<data>[^\]]*)\]"
)
_SCALAR_PATTERN = re.compile(
    r"(?m)^(?P<kind>H|W)_(?P<name>[^:\r\n]+):\s*(?P<value>[-+0-9.eE]+)\s*$"
)


def _read_opencv_yaml(path: Path) -> dict[tuple[str, str], np.ndarray | float]:
    text = path.read_text(encoding="utf-8")
    values: dict[tuple[str, str], np.ndarray | float] = {}
    for match in _MATRIX_PATTERN.finditer(text):
        rows = int(match.group("rows"))
        cols = int(match.group("cols"))
        data = np.fromstring(match.group("data"), sep=",", dtype=np.float64)
        if data.size != rows * cols:
            raise ValueError(
                f"invalid OpenCV matrix {match.group('kind')}_{match.group('name')} "
                f"in {path}: expected {rows * cols} values, got {data.size}"
            )
        values[(match.group("kind"), match.group("name"))] = data.reshape(rows, cols)
    for match in _SCALAR_PATTERN.finditer(text):
        values[(match.group("kind"), match.group("name"))] = float(
            match.group("value")
        )
    return values


def load_camera_records(
    dataset_root: str | Path,
    *,
    ratio: float = 0.25,
    near: float = 0.2,
    far: float = 20.0,
) -> tuple[CameraRecord, ...]:
    """Read EasyVolcap OpenCV YAML cameras without importing PyTorch."""

    root = Path(dataset_root).expanduser().resolve()
    intrinsics_path = root / "intri.yml"
    extrinsics_path = root / "extri.yml"
    if not intrinsics_path.is_file() or not extrinsics_path.is_file():
        raise FileNotFoundError(
            f"camera files are missing below dataset root: {root}"
        )
    if isinstance(ratio, bool) or not isinstance(ratio, numbers.Real):
        raise TypeError("ratio must be a CPU real scalar")
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be finite and positive")
    if not math.isfinite(float(near)) or not math.isfinite(float(far)):
        raise ValueError("near and far must be finite")
    if float(near) <= 0.0 or float(far) <= float(near):
        raise ValueError("camera clipping range must satisfy 0 < near < far")

    intrinsics = _read_opencv_yaml(intrinsics_path)
    extrinsics = _read_opencv_yaml(extrinsics_path)
    names = sorted(
        name
        for kind, name in intrinsics
        if kind == "K" and ("Rot", name) in extrinsics and ("T", name) in extrinsics
    )
    if not names:
        raise ValueError(f"no complete cameras found below {root}")

    records = []
    for camera_id, name in enumerate(names):
        required = (("K", name), ("H", name), ("W", name))
        missing = [f"{kind}_{name}" for kind, key in required if (kind, key) not in intrinsics]
        if missing:
            raise ValueError(f"missing intrinsic entries: {missing}")
        original_k = np.asarray(intrinsics[("K", name)], dtype=np.float64)
        original_h = int(float(intrinsics[("H", name)]))
        original_w = int(float(intrinsics[("W", name)]))
        image_h = int(original_h * ratio)
        image_w = int(original_w * ratio)
        if image_h <= 0 or image_w <= 0:
            raise ValueError(f"scaled camera {name} has nonpositive dimensions")
        scaled_k = np.array(original_k, dtype=np.float64, order="C", copy=True)
        scaled_k[0:1] *= image_w / original_w
        scaled_k[1:2] *= image_h / original_h
        rotation = np.asarray(extrinsics[("Rot", name)], dtype=np.float32)
        translation = np.asarray(extrinsics[("T", name)], dtype=np.float32)
        if scaled_k.shape != (3, 3) or rotation.shape != (3, 3) or translation.shape != (3, 1):
            raise ValueError(f"camera {name} has invalid matrix shapes")
        arrays = (
            np.ascontiguousarray(scaled_k, dtype=np.float32),
            np.ascontiguousarray(rotation, dtype=np.float32),
            np.ascontiguousarray(translation, dtype=np.float32),
            np.asarray([near], dtype=np.float32),
            np.asarray([far], dtype=np.float32),
        )
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError(f"camera {name} contains NaN or Inf")
        records.append(
            CameraRecord(
                camera_id=camera_id,
                name=name,
                image_height=image_h,
                image_width=image_w,
                K=arrays[0],
                R=arrays[1],
                T=arrays[2],
                near=arrays[3],
                far=arrays[4],
            )
        )
    return tuple(records)


def load_flashgs_camera_records(
    dataset_root: str | Path,
    *,
    camera_path: str | Path | None = None,
) -> tuple[FlashgsCameraRecord, ...]:
    """Read the exact six-field camera format used by HybridSplat FlashGS."""

    root = Path(dataset_root).expanduser().resolve()
    path = (
        Path(camera_path).expanduser().resolve()
        if camera_path is not None
        else root / "cameras.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"FlashGS camera file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid FlashGS camera JSON {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("FlashGS camera JSON must contain a nonempty list")
    required = {"id", "img_name", "width", "height", "position", "rotation", "fx", "fy"}
    records = []
    for expected_id, item in enumerate(payload):
        if not isinstance(item, dict) or set(item) != required:
            actual = sorted(item) if isinstance(item, dict) else type(item).__name__
            raise ValueError(
                f"FlashGS camera {expected_id} fields mismatch: expected "
                f"{sorted(required)}, got {actual}"
            )
        camera_id = item["id"]
        if isinstance(camera_id, bool) or not isinstance(camera_id, int):
            raise TypeError(f"FlashGS camera {expected_id} id must be an integer")
        if camera_id != expected_id:
            raise ValueError(
                f"FlashGS camera ids must be consecutive; expected {expected_id}, got {camera_id}"
            )
        name = item["img_name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"FlashGS camera {camera_id} has invalid img_name")
        dimensions = []
        for field in ("width", "height"):
            value = item[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"FlashGS camera {camera_id} {field} must be positive integer")
            dimensions.append(value)
        image_width = int(math.floor(dimensions[0] / 4.0))
        image_height = int(math.floor(dimensions[1] / 4.0))
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"FlashGS camera {camera_id} has invalid quarter resolution")
        position = np.ascontiguousarray(item["position"], dtype=np.float32)
        rotation = np.ascontiguousarray(item["rotation"], dtype=np.float32)
        if position.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError(f"FlashGS camera {camera_id} has invalid pose shape")
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError(f"FlashGS camera {camera_id} pose contains NaN or Inf")
        focal_x = _cpu_focal("fx", item["fx"]) / 4.0
        focal_y = _cpu_focal("fy", item["fy"]) / 4.0
        records.append(
            FlashgsCameraRecord(
                camera_id=camera_id,
                name=name,
                image_height=image_height,
                image_width=image_width,
                position=position,
                rotation=rotation,
                focal_x=focal_x,
                focal_y=focal_y,
            )
        )
    return tuple(records)


def materialize_camera(record: CameraRecord, *, context=None) -> GaussianCamera:
    jt = _jt()
    arrays = [
        jt.array(record.K).stop_grad(),
        jt.array(record.R).stop_grad(),
        jt.array(record.T).stop_grad(),
        jt.array(record.near).stop_grad(),
        jt.array(record.far).stop_grad(),
    ]
    return prepare_camera(
        record.image_height,
        record.image_width,
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[4],
        focal_x=float(record.K[0, 0]),
        focal_y=float(record.K[1, 1]),
        context=context,
    )


def prepare_camera(
    image_height: int,
    image_width: int,
    K,
    R,
    T,
    near,
    far,
    *,
    focal_x: float,
    focal_y: float,
    context=None,
) -> GaussianCamera:
    focal_x = _cpu_focal("focal_x", focal_x)
    focal_y = _cpu_focal("focal_y", focal_y)
    fov_x = focal_to_fov(focal_x, image_width)
    fov_y = focal_to_fov(focal_y, image_height)
    world_view = world_to_view(R, T).transpose(0, 1)
    projection = projection_matrix(fov_x, fov_y, near, far).transpose(0, 1)
    full_projection = world_view @ projection
    from .ops.precise_camera import precise_camera_center

    center = precise_camera_center(R, T, context=context)
    return GaussianCamera(
        image_height=int(image_height),
        image_width=int(image_width),
        K=K,
        R=R,
        T=T,
        near=near,
        far=far,
        znear=near,
        zfar=far,
        FoVx=fov_x,
        FoVy=fov_y,
        tanfovx=math.tan(fov_x * 0.5),
        tanfovy=math.tan(fov_y * 0.5),
        world_view_transform=world_view,
        projection_matrix=projection,
        full_proj_transform=full_projection,
        camera_center=center,
    )


def inverse_3x3(matrix, eps: float = 1.1920928955078125e-7):
    if len(matrix.shape) != 2 or list(matrix.shape) != [3, 3]:
        raise ValueError(f"inverse_3x3 expects [3,3], got {matrix.shape}")
    canonical_eps = 1.1920928955078125e-7
    if float(eps) != canonical_eps:
        raise ValueError(
            "inverse_3x3 only supports the PyTorch FP32 epsilon "
            f"{canonical_eps}, got {eps}"
        )
    from .ops.precise_camera_rays import precise_camera_inverse

    return precise_camera_inverse(matrix)


def camera_rays(camera: GaussianCamera, *, context=None):
    jt = _jt()
    rows = jt.arange(camera.image_height, dtype="float32")
    columns = jt.arange(camera.image_width, dtype="float32")
    i, j = jt.meshgrid(rows, columns)
    i, j = i + 0.5, j + 0.5
    xy1 = jt.stack((j, i, jt.ones_like(i)), dim=-1)[..., None]
    batch_shape = (camera.image_height, camera.image_width)
    inverse_k = inverse_3x3(camera.K).reshape((1, 1, 3, 3)).expand(
        batch_shape + (3, 3)
    )
    translation = camera.T.reshape((1, 1, 3, 1)).expand(
        batch_shape + (3, 1)
    )
    pixel_camera = inverse_k @ xy1
    from .ops.precise_camera_rays import precise_pixel_world

    pixel_world = precise_pixel_world(
        camera.R,
        (pixel_camera - translation)[..., 0],
        context=context,
    )
    origin_single = camera.camera_center
    origin = origin_single.reshape((1, 1, 3)).expand(
        (camera.image_height, camera.image_width, 3)
    )
    direction = pixel_world - origin
    coords = jt.stack((i - 0.5, j - 0.5), dim=-1).int64()
    return (
        origin.reshape((1, -1, 3)),
        direction.reshape((1, -1, 3)),
        coords.reshape((1, -1, 2)),
    )
