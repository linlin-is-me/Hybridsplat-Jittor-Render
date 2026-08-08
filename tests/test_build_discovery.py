from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from jittor_hybridsplat.ops import build


def executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class BuildDiscoveryTests(unittest.TestCase):
    def test_explicit_cuda_root_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cuda"
            executable(root / "bin" / "nvcc")
            (root / "lib64").mkdir(parents=True)
            resolved_root, nvcc, library = build.resolve_cuda_toolkit(root)
            self.assertEqual(resolved_root, root.resolve())
            self.assertEqual(nvcc, (root / "bin" / "nvcc").resolve())
            self.assertEqual(library, (root / "lib64").resolve())

    def test_nvcc_path_environment_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cuda"
            nvcc = executable(root / "bin" / "nvcc")
            library = root / "targets" / "x86_64-linux" / "lib"
            library.mkdir(parents=True)
            with mock.patch.dict(os.environ, {"nvcc_path": str(nvcc)}, clear=True):
                with mock.patch("shutil.which", return_value=None):
                    resolved = build.resolve_cuda_toolkit()
            self.assertEqual(resolved, (root.resolve(), nvcc.resolve(), library.resolve()))

    def test_cxx_environment_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            compiler = executable(Path(directory) / "bin" / "g++")
            with mock.patch.dict(os.environ, {"CXX": str(compiler)}, clear=True):
                with mock.patch("shutil.which", return_value=None):
                    self.assertEqual(build.resolve_cxx(), compiler.resolve())

    def test_invalid_explicit_compiler_does_not_fall_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-g++"
            with self.assertRaises(RuntimeError):
                build.resolve_cxx(missing)


if __name__ == "__main__":
    unittest.main()
