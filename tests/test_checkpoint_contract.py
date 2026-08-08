from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from jittor_hybridsplat.checkpoint import infer_checkpoint_contract


def _group(prefix: str, points: int, degree: int) -> dict[str, np.ndarray]:
    rest = (degree + 1) ** 2 - 1
    values = {
        f"{prefix}.active_sh_degree": np.asarray([degree], dtype="<i8"),
        f"{prefix}._xyz": np.zeros((points, 3), dtype="<f4"),
        f"{prefix}._features_dc": np.zeros((points, 1, 3), dtype="<f4"),
        f"{prefix}._features_rest": np.zeros((points, rest, 3), dtype="<f4"),
        f"{prefix}._scaling": np.zeros((points, 2), dtype="<f4"),
        f"{prefix}._rotation": np.zeros((points, 4), dtype="<f4"),
        f"{prefix}._opacity": np.zeros((points, 1), dtype="<f4"),
        f"{prefix}._specular": np.zeros((points, 1), dtype="<f4"),
        f"{prefix}._roughness": np.zeros((points, 1), dtype="<f4"),
        f"{prefix}.max_radii2D": np.zeros((points,), dtype="<f4"),
    }
    for name in ("xyz_gradient_accum", "denom", "xyz_weight_accum"):
        values[f"{prefix}.{name}"] = np.zeros((points, 1), dtype="<f4")
    return values


class CheckpointContractTests(unittest.TestCase):
    def test_infers_model_dimensions_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            arrays = {
                "epoch": np.asarray(17, dtype="<i8"),
                "sampler.bg_color": np.zeros((3,), dtype="<f4"),
                "sampler.ref_bg_color": np.zeros((3,), dtype="<f4"),
            }
            arrays.update(_group("sampler.pcd", 5, 3))
            arrays.update(_group("sampler.ref", 2, 3))
            np.savez_compressed(path, **arrays)

            contract = infer_checkpoint_contract(path)
            self.assertEqual(contract.epoch, 17)
            self.assertEqual(contract.base_points, 5)
            self.assertEqual(contract.reflection_points, 2)
            self.assertEqual(contract.sh_degree, 3)
            self.assertEqual(len(contract.sha256 or ""), 64)
            self.assertEqual(contract.file_bytes, path.stat().st_size)
            self.assertGreater(contract.uncompressed_bytes or 0, 0)


if __name__ == "__main__":
    unittest.main()
