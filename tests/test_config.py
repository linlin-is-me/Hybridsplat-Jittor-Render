from __future__ import annotations

import json
import tempfile
from pathlib import Path
import unittest

from jittor_hybridsplat.render import load_render_config, parse_views


class ConfigTests(unittest.TestCase):
    def test_relative_paths_use_config_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            payload = {
                "dataset_root": "../data/scene",
                "ratio": 0.25,
                "near": 0.2,
                "far": 20.0,
                "views": "0:5:2",
                "backend": "standard",
                "flashgs_camera_path": None,
                "dump_raw": True,
                "evaluate": False,
                "reference": None,
                "stage_sync": False,
            }
            path = config_dir / "render.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_render_config(path)
            self.assertEqual(loaded["dataset_root"], (root / "data/scene").resolve())
            self.assertEqual(loaded["views"], (0, 2, 4))

    def test_view_parser_rejects_duplicates(self) -> None:
        with self.assertRaises(Exception):
            parse_views("1,1")


if __name__ == "__main__":
    unittest.main()
