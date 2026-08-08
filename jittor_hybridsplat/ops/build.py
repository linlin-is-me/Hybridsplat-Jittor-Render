"""构建与定位可独立分发的 HybridSplat Jittor 原生后端。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NATIVE_SOURCE_DIR = Path(__file__).resolve().parent / "native"
MATH_REVISION = 11
NATIVE_STAGE_REVISION = "render-v3"
NATIVE_BUILD_ID = (
    "jt1.3.11.0-cu12.6-sm89-abi2-"
    f"math{MATH_REVISION}-{NATIVE_STAGE_REVISION}"
)
JITTOR_OPS_GEN_NAME = (
    f"jittor_hybridsplat_render_ops_abi2_math{MATH_REVISION}_v3"
)
JITTOR_CACHE_NAME = "hybridsplat-render-jt13110-cu126-sm89-abi2-math11-v3"
DEFAULT_NATIVE_ROOT = (
    Path.home() / ".cache" / "hybridsplat_jittor_render" / "native" / NATIVE_BUILD_ID
)

_CONDA_TOOLCHAIN_VARIABLES = (
    "CC",
    "CXX",
    "CFLAGS",
    "CPPFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "CMAKE_ARGS",
    "CMAKE_PREFIX_PATH",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CUDA_HOME",
)


def native_build_dir() -> Path:
    return Path(
        os.environ.get(
            "HYBRIDSPLAT_NATIVE_BUILD_DIR",
            str(DEFAULT_NATIVE_ROOT / "build"),
        )
    ).expanduser()


def native_install_dir() -> Path:
    return Path(
        os.environ.get(
            "HYBRIDSPLAT_NATIVE_INSTALL_DIR",
            str(DEFAULT_NATIVE_ROOT / "install"),
        )
    ).expanduser()


def native_library_path() -> Path:
    return native_install_dir() / "lib" / "libhybridsplat_rasterizer.so"


def native_include_dir() -> Path:
    return native_install_dir() / "include"


def native_build_manifest_path() -> Path:
    return native_install_dir() / "share" / "hybridsplat" / "build_manifest.json"


def cuda_library_dir() -> Path:
    manifest_path = native_build_manifest_path()
    if not manifest_path.is_file():
        raise RuntimeError(f"native build manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("toolchain", {}).get("cuda_library_dir")
    if not isinstance(value, str) or not value:
        raise RuntimeError("native build manifest has no CUDA library directory")
    path = Path(value)
    if not path.is_dir():
        raise RuntimeError(f"CUDA library directory is missing: {path}")
    return path


def _clean_build_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in _CONDA_TOOLCHAIN_VARIABLES:
        environment.pop(name, None)
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = environment.get(
        "CMAKE_BUILD_PARALLEL_LEVEL", "8"
    )
    return environment


def _run(command: Iterable[str], environment: dict[str, str]) -> None:
    subprocess.run(
        [str(item) for item in command],
        check=True,
        env=environment,
    )


def _environment_or_path_executable(environment_name: str, command: str) -> Path:
    configured = os.environ.get(environment_name)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise RuntimeError(f"{environment_name} is not executable: {candidate}")
    environment_candidate = Path(sys.executable).resolve().parent / command
    if environment_candidate.is_file() and os.access(environment_candidate, os.X_OK):
        return environment_candidate
    located = shutil.which(command)
    if located:
        return Path(located).resolve()
    raise RuntimeError(
        f"cannot locate {command}; set {environment_name} or install it in the active environment"
    )


def _resolve_executable(
    explicit: Path | None,
    *,
    environment_names: tuple[str, ...],
    commands: tuple[str, ...],
    label: str,
) -> Path:
    if explicit is not None:
        candidates = (("command line", str(explicit)),)
        allow_fallback = False
    else:
        candidates = tuple(
            (f"environment variable {name}", os.environ[name])
            for name in environment_names
            if os.environ.get(name)
        ) + tuple(("PATH", command) for command in commands)
        allow_fallback = True
    attempted: list[str] = []
    for source, value in candidates:
        expanded = os.path.expanduser(value)
        path = Path(expanded)
        if path.is_absolute() or path.parent != Path("."):
            resolved = path.resolve()
            attempted.append(f"{source}={resolved}")
            if resolved.is_file() and os.access(resolved, os.X_OK):
                return resolved
        else:
            located = shutil.which(expanded)
            attempted.append(f"{source}={expanded}")
            if located:
                return Path(located).resolve()
    detail = ", ".join(attempted) if attempted else "no candidates"
    if not allow_fallback:
        raise RuntimeError(f"{label} from the command line is not executable: {detail}")
    raise RuntimeError(f"cannot locate {label}; checked {detail}")


def resolve_cxx(cxx: Path | None = None) -> Path:
    """Resolve the C++ compiler without binding the source tree to one machine."""

    return _resolve_executable(
        cxx,
        environment_names=("CXX", "cc_path"),
        commands=("g++", "c++"),
        label="C++ compiler",
    )


def resolve_cuda_toolkit(cuda_home: Path | None = None) -> tuple[Path, Path, Path]:
    """Resolve CUDA root, nvcc, and its library directory in priority order."""

    if cuda_home is not None:
        roots = (("command line", cuda_home.expanduser().resolve()),)
        nvcc_candidates: tuple[tuple[str, str], ...] = ()
    else:
        roots = tuple(
            (f"environment variable {name}", Path(os.environ[name]).expanduser().resolve())
            for name in ("CUDA_HOME", "CUDA_PATH")
            if os.environ.get(name)
        )
        nvcc_candidates = tuple(
            (f"environment variable {name}", os.environ[name])
            for name in ("nvcc_path", "CUDACXX", "NVCC")
            if os.environ.get(name)
        ) + (("PATH", "nvcc"),)

    attempted: list[str] = []
    for source, root in roots:
        attempted.append(f"{source}={root}")
        nvcc = root / "bin" / "nvcc"
        if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
            continue
        library = _cuda_library_directory(root)
        if library is not None:
            return root, nvcc.resolve(), library
    if cuda_home is not None:
        raise RuntimeError(
            "CUDA toolkit from the command line is incomplete; checked "
            + ", ".join(attempted)
        )

    for source, value in nvcc_candidates:
        expanded = os.path.expanduser(value)
        candidate = Path(expanded)
        if candidate.is_absolute() or candidate.parent != Path("."):
            nvcc = candidate.resolve()
        else:
            located = shutil.which(expanded)
            nvcc = Path(located).resolve() if located else candidate
        attempted.append(f"{source}={expanded}")
        if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
            continue
        root = nvcc.parent.parent.resolve()
        library = _cuda_library_directory(root)
        if library is not None:
            return root, nvcc, library

    for root in (Path("/usr/local/cuda"), Path("/opt/cuda")):
        attempted.append(f"conventional path={root}")
        nvcc = root / "bin" / "nvcc"
        library = _cuda_library_directory(root)
        if nvcc.is_file() and os.access(nvcc, os.X_OK) and library is not None:
            return root.resolve(), nvcc.resolve(), library
    raise RuntimeError("cannot locate a complete CUDA toolkit; checked " + ", ".join(attempted))


def _cuda_library_directory(cuda_home: Path) -> Path | None:
    for candidate in (
        cuda_home / "lib64",
        cuda_home / "targets" / "x86_64-linux" / "lib",
        cuda_home / "lib",
    ):
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _source_manifest() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(NATIVE_SOURCE_DIR.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = digest
    return result


def build_contract() -> dict[str, object]:
    return {
        "schema_version": 1,
        "math_revision": MATH_REVISION,
        "native_build_id": NATIVE_BUILD_ID,
        "jittor_ops_gen_name": JITTOR_OPS_GEN_NAME,
        "jittor_cache_name": JITTOR_CACHE_NAME,
        "native_build_dir": str(native_build_dir()),
        "native_install_dir": str(native_install_dir()),
        "native_library": str(native_library_path()),
    }


def build_native(
    *,
    cuda_home: Path | None = None,
    arch: int = 89,
    cxx: Path | None = None,
    optix_include: Path | None = None,
    build_type: str = "Release",
) -> Path:
    cmake = _environment_or_path_executable("CMAKE_EXECUTABLE", "cmake")
    ninja = _environment_or_path_executable("NINJA_EXECUTABLE", "ninja")

    build_dir = native_build_dir()
    install_dir = native_install_dir()
    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)
    environment = _clean_build_environment()
    cuda_home, nvcc, cuda_library = resolve_cuda_toolkit(cuda_home)
    cxx = resolve_cxx(cxx)
    if optix_include is None:
        environment_value = os.environ.get("OPTIX_INCLUDE_DIR")
        root_value = os.environ.get("OPTIX_ROOT") or os.environ.get("OPTIX_HOME")
        if environment_value:
            optix_include = Path(environment_value)
        elif root_value:
            optix_include = Path(root_value) / "include"
        else:
            raise RuntimeError(
                "OptiX SDK include directory is required; use --optix-include "
                "or set OPTIX_INCLUDE_DIR"
            )
    optix_include = optix_include.expanduser().resolve()
    if not (optix_include / "optix.h").is_file():
        raise RuntimeError(f"OptiX SDK headers do not exist: {optix_include}")
    if arch <= 0:
        raise ValueError("CUDA architecture must be positive")
    if build_type not in {"Release", "RelWithDebInfo", "Debug"}:
        raise ValueError(f"unsupported build type: {build_type}")

    configure_command = [
        cmake,
        "-S",
        NATIVE_SOURCE_DIR,
        "-B",
        build_dir,
        "-G",
        "Ninja",
        f"-DCMAKE_MAKE_PROGRAM={ninja}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        f"-DCMAKE_CXX_COMPILER={cxx}",
        f"-DCMAKE_CUDA_COMPILER={nvcc}",
        f"-DCMAKE_CUDA_HOST_COMPILER={cxx}",
        f"-DCUDAToolkit_ROOT={cuda_home}",
        f"-DCMAKE_CUDA_ARCHITECTURES={arch}-real",
        f"-DHS_CUDA_ARCH={arch}",
        f"-DHS_OPTIX_INCLUDE_DIR={optix_include}",
        f"-DHS_RASTERIZER_BUILD_ID={NATIVE_BUILD_ID}",
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
    ]
    _run(configure_command, environment)
    _run([cmake, "--build", build_dir], environment)
    _run([cmake, "--install", build_dir], environment)

    library = native_library_path()
    if not library.is_file():
        raise RuntimeError(f"native library was not installed: {library}")

    manifest_dir = install_dir / "share" / "hybridsplat"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 3,
        "math_revision": MATH_REVISION,
        "build_id": NATIVE_BUILD_ID,
        "jittor_ops_gen_name": JITTOR_OPS_GEN_NAME,
        "jittor_cache_name": JITTOR_CACHE_NAME,
        "source_dir": str(NATIVE_SOURCE_DIR),
        "build_dir": str(build_dir),
        "install_dir": str(install_dir),
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "toolchain": {
            "cmake": str(cmake),
            "ninja": str(ninja),
            "gxx": str(cxx),
            "nvcc": str(nvcc),
            "cuda_root": str(cuda_home),
            "cuda_library_dir": str(cuda_library),
            "optix_include_dir": str(optix_include),
            "cuda_architecture": f"{arch}-real",
            "build_type": build_type,
        },
        "configure_command": [str(item) for item in configure_command],
        "sanitized_environment_variables": list(_CONDA_TOOLCHAIN_VARIABLES),
        "source_sha256": _source_manifest(),
    }
    (manifest_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return library


def require_native_build() -> Path:
    library = native_library_path()
    header = native_include_dir() / "hybridsplat_rasterizer_c_api.h"
    if not library.is_file() or not header.is_file():
        raise RuntimeError(
            "HybridSplat native backend is not built; run "
            "python -m jittor_hybridsplat.ops.build"
        )
    return library


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-paths",
        action="store_true",
        help="print the versioned build and install paths",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="print the native build, cache, and ABI identity",
    )
    parser.add_argument(
        "--cuda-home",
        type=Path,
        help="CUDA toolkit root; otherwise inspect CUDA_HOME, nvcc_path, PATH, and common roots",
    )
    parser.add_argument("--arch", type=int, default=89)
    parser.add_argument(
        "--cxx",
        type=Path,
        help="C++ compiler; otherwise inspect CXX, cc_path, g++, and c++",
    )
    parser.add_argument(
        "--optix-include",
        type=Path,
        help="NVIDIA OptiX SDK include directory containing optix.h",
    )
    parser.add_argument(
        "--build-type",
        choices=("Release", "RelWithDebInfo", "Debug"),
        default="Release",
    )
    arguments = parser.parse_args()
    if arguments.print_contract:
        print(json.dumps(build_contract(), indent=2, sort_keys=True))
        return
    if arguments.print_paths:
        print(
            json.dumps(
                {
                    "build_dir": str(native_build_dir()),
                    "install_dir": str(native_install_dir()),
                    "library": str(native_library_path()),
                },
                indent=2,
            )
        )
        return
    print(
        build_native(
            cuda_home=arguments.cuda_home,
            arch=arguments.arch,
            cxx=arguments.cxx,
            optix_include=arguments.optix_include,
            build_type=arguments.build_type,
        )
    )


if __name__ == "__main__":
    main()
