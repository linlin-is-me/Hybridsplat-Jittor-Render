from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DistributionTests(unittest.TestCase):
    def test_required_sources_and_licenses_exist(self) -> None:
        required = (
            "README.md",
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "pyproject.toml",
            "docs/assets/render-comparison-camera0000.png",
            "scripts/convert_pytorch_checkpoint.py",
            "jittor_hybridsplat/ops/native/CMakeLists.txt",
            "jittor_hybridsplat/ops/native/vendor/standard_rasterizer/forward.cu",
            "jittor_hybridsplat/ops/native/vendor/standard_rasterizer/LICENSE.md",
            "jittor_hybridsplat/ops/native/vendor/flashgs/cuda_rasterizer/preprocess.cu",
            "jittor_hybridsplat/ops/native/vendor/flashgs/LICENSE",
            "jittor_hybridsplat/ops/native/vendor/standard_glm/copying.txt",
            "jittor_hybridsplat/ops/native/vendor/optix_tracer/forward.cu",
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_package_has_no_external_project_paths_or_runtime_torch(self) -> None:
        source_root = ROOT / "jittor_hybridsplat"
        for path in source_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".py", ".h", ".hpp", ".cc", ".cpp", ".cu", ".txt"
            }:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("HybridSplat-src", text, str(path))
            self.assertNotIn("D:\\myproject", text, str(path))
            self.assertNotIn("/mnt/d/", text, str(path))
            if "vendor" not in path.parts:
                self.assertNotIn("torch/extension", text, str(path))
                self.assertNotIn("ATen/", text, str(path))
                self.assertNotIn("c10/", text, str(path))
                self.assertNotIn("pybind11", text, str(path))

    def test_native_binary_does_not_embed_build_tree_ptx_path(self) -> None:
        cmake = (ROOT / "jittor_hybridsplat/ops/native/CMakeLists.txt").read_text(
            encoding="utf-8"
        )
        backend = (
            ROOT / "jittor_hybridsplat/ops/native/optix/optix_backend.cpp"
        ).read_text(encoding="utf-8")
        self.assertNotIn("HS_OPTIX_PTX_PATH", cmake)
        self.assertNotIn("HS_OPTIX_PTX_PATH", backend)
        self.assertIn("dladdr", backend)
        self.assertIn('"share" / "hybridsplat" / "forward.ptx"', backend)

    def test_no_generated_or_model_files(self) -> None:
        forbidden_suffixes = {".so", ".dll", ".dylib", ".pt", ".pth", ".npz"}
        for path in ROOT.rglob("*"):
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), forbidden_suffixes, str(path))


if __name__ == "__main__":
    unittest.main()
