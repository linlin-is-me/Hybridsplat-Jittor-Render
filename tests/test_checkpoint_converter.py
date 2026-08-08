from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jittor_hybridsplat.checkpoint import validate_checkpoint_archive


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_pytorch_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_converter", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def synthetic_state(points: int = 3, degree: int = 1):
    rest = (degree + 1) ** 2 - 1
    state = {
        "sampler.bg_color": np.zeros((3,), dtype=np.float32),
        "sampler.ref_bg_color": np.ones((3,), dtype=np.float32),
    }
    for prefix in ("sampler.pcd", "sampler.ref"):
        state.update(
            {
                f"{prefix}.active_sh_degree": np.asarray([degree], dtype=np.int64),
                f"{prefix}._xyz": np.zeros((points, 3), dtype=np.float32),
                f"{prefix}._features_dc": np.zeros((points, 1, 3), dtype=np.float32),
                f"{prefix}._features_rest": np.zeros((points, rest, 3), dtype=np.float32),
                f"{prefix}._scaling": np.zeros((points, 2), dtype=np.float32),
                f"{prefix}._rotation": np.zeros((points, 4), dtype=np.float32),
                f"{prefix}._opacity": np.zeros((points, 1), dtype=np.float32),
                f"{prefix}._specular": np.zeros((points, 1), dtype=np.float32),
                f"{prefix}._roughness": np.zeros((points, 1), dtype=np.float32),
                f"{prefix}.max_radii2D": np.zeros((points,), dtype=np.float32),
                f"{prefix}.xyz_gradient_accum": np.zeros((points, 1), dtype=np.float32),
                f"{prefix}.denom": np.zeros((points, 1), dtype=np.float32),
                f"{prefix}.xyz_weight_accum": np.zeros((points, 1), dtype=np.float32),
            }
        )
    return state


class CheckpointConverterTests(unittest.TestCase):
    def test_extract_and_write_strict_archive(self) -> None:
        state = synthetic_state()
        state["training_only"] = np.ones((1,), dtype=np.float32)
        arrays, metadata = CONVERTER.extract_arrays({"model": state, "epoch": 7})
        self.assertEqual(len(arrays), 29)
        self.assertEqual(metadata["ignored_state_keys"], ["training_only"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.npz"
            contract = CONVERTER.write_npz(arrays, output)
            self.assertEqual(contract.epoch, 7)
            self.assertEqual(contract.base_points, 3)
            self.assertEqual(contract.reflection_points, 3)
            self.assertEqual(contract.sh_degree, 1)
            self.assertEqual(validate_checkpoint_archive(output), contract)

    def test_module_prefix_is_detected(self) -> None:
        state = {f"module.{key}": value for key, value in synthetic_state().items()}
        arrays, metadata = CONVERTER.extract_arrays({"model": state, "epoch": 2})
        self.assertEqual(metadata["state_prefix"], "module.")
        self.assertEqual(int(arrays["epoch"]), 2)

    def test_missing_key_is_rejected(self) -> None:
        state = synthetic_state()
        del state["sampler.ref._xyz"]
        with self.assertRaises(CONVERTER.ConversionError):
            CONVERTER.extract_arrays({"model": state, "epoch": 1})

    def test_existing_output_requires_overwrite(self) -> None:
        arrays, _ = CONVERTER.extract_arrays(
            {"model": synthetic_state(), "epoch": 1}
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "model.npz"
            CONVERTER.write_npz(arrays, output)
            with self.assertRaises(FileExistsError):
                CONVERTER.write_npz(arrays, output)


if __name__ == "__main__":
    unittest.main()
