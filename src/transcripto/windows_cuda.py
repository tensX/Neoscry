from __future__ import annotations

import ctypes
import os
import site
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def is_windows() -> bool:
    return os.name == "nt"


def _try_load_dll(name: str) -> bool:
    try:
        ctypes.WinDLL(name)
        return True
    except Exception:
        return False


def _candidate_dll_dirs() -> list[Path]:
    # Common layout for NVIDIA CUDA runtime wheels:
    #   nvidia-cublas-cu12
    #   nvidia-cuda-runtime-cu12
    roots: list[Path] = []
    for p in (site.getusersitepackages(), *site.getsitepackages()):
        if not p:
            continue
        roots.append(Path(p))

    candidates: list[Path] = []
    for r in roots:
        candidates.extend(
            [
                r / "nvidia" / "cublas" / "bin",
                r / "nvidia" / "cublas" / "lib",
                r / "nvidia" / "cuda_runtime" / "bin",
                r / "nvidia" / "cuda_runtime" / "lib",
                r / "nvidia" / "cudnn" / "bin",
                r / "nvidia" / "cudnn" / "lib",
            ]
        )
    return candidates


def add_dll_directories(dirs: Iterable[Path]) -> int:
    if not is_windows():
        return 0
    if not hasattr(os, "add_dll_directory"):
        return 0

    added = 0
    for d in dirs:
        try:
            if not d.exists():
                continue
            os.add_dll_directory(str(d))
            added += 1
        except Exception:
            continue
    return added


def ensure_cublas12_dll_available() -> bool:
    """Try to make cublas64_12.dll loadable on Windows.

    This mirrors a common approach used by apps that rely on CUDA-enabled
    Python wheels (NVIDIA runtime packages) instead of a system-wide CUDA
    Toolkit install.
    """
    if not is_windows():
        return True

    if _try_load_dll("cublas64_12.dll"):
        return True

    add_dll_directories(_candidate_dll_dirs())
    return _try_load_dll("cublas64_12.dll")


def install_cuda_runtime_packages() -> tuple[bool, str]:
    """Install CUDA runtime DLL wheels via pip (Windows only)."""
    if not is_windows():
        return False, "Not Windows"

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "nvidia-cublas-cu12",
        "nvidia-cuda-runtime-cu12",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception as e:
        return False, f"pip failed to start: {e}"

    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        return False, out.strip() or f"pip exited with code {proc.returncode}"

    ok = ensure_cublas12_dll_available()
    if not ok:
        out = (out.strip() + "\n\n") + "Installed runtime wheels, but cublas64_12.dll is still not loadable"
        return False, out
    return True, out.strip()
