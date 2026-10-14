"""Host-safe offscreen graphics probe for isolated simulator interpreters.

Generic preflight executes this module in the configured simulator Python
before the LIBERO import probe.  MuJoCo and PyOpenGL resolve their offscreen
backend from process environment that the Harness only forwards, so a missing
EGL vendor ICD or driver library surfaces downstream as an opaque
``Could not import OffScreenRenderEnv``.  Enumerating devices here names the
actual cause while the graphics stack is still the only thing under test.

The module imports nothing from a simulator backend and is deliberately
dependency-free.  Preflight runs it by file path rather than with ``-m`` so the
check never depends on ``ora.environment`` importing cleanly inside an
external interpreter whose package set the Harness does not own.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
from pathlib import Path
from typing import Any

GRAPHICS_PROBE_SCHEMA = "ora.offscreen-graphics-probe.v1"

DEFAULT_EGL_VENDOR_DIR = "/usr/share/glvnd/egl_vendor.d"

_REPORTED_ENVIRONMENT = (
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "__EGL_VENDOR_LIBRARY_DIRS",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "LIBGL_ALWAYS_SOFTWARE",
    "LP_NUM_THREADS",
    "MUJOCO_EGL_DEVICE_ID",
)

# eglQueryDevicesEXT writes into a caller-owned array; 32 exceeds any host the
# Runtime qualifies and keeps the call a single pass.
_MAX_EGL_DEVICES = 32


def default_libero_backend() -> str:
    """Match MuJoCo's native macOS context while retaining Linux EGL defaults."""
    return "cgl" if platform.system() == "Darwin" else "egl"


def _opengl_platform(backend: str) -> str | None:
    return {"egl": "egl", "osmesa": "osmesa", "cgl": "darwin",
            "glfw": "darwin" if platform.system() == "Darwin" else "glx",
            "glx": "glx", "wgl": "win32"}.get(backend)


def configure_libero_graphics() -> None:
    """Select graphics before importing LIBERO/robosuite; preserve explicit settings.

    MuJoCo's macOS CGL context and robosuite's hidden GLFW window use the
    native Darwin OpenGL loader. They must never inherit the Linux EGL default.
    An explicit conflicting loader is an error rather than an implicit switch.
    """
    backend = os.environ.setdefault("MUJOCO_GL", default_libero_backend())
    expected = _opengl_platform(backend)
    selected = os.environ.get("PYOPENGL_PLATFORM")
    if expected is not None and selected is not None and selected != expected:
        raise ValueError(f"PYOPENGL_PLATFORM={selected!r} contradicts MUJOCO_GL={backend!r}; expected {expected!r}")
    if expected is not None:
        os.environ.setdefault("PYOPENGL_PLATFORM", expected)


def _vendor_icd_files() -> list[str]:
    """List the EGL vendor ICD manifests visible to this process."""

    # libglvnd uses an explicit file list instead of any directory scan.
    explicit = os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES")
    if explicit:
        return [entry for entry in explicit.split(os.pathsep) if entry]
    raw = os.environ.get("__EGL_VENDOR_LIBRARY_DIRS")
    directories = (
        [Path(entry) for entry in raw.split(os.pathsep) if entry]
        if raw else [Path(DEFAULT_EGL_VENDOR_DIR)]
    )
    files: list[str] = []
    for directory in directories:
        try:
            files.extend(sorted(str(path) for path in directory.glob("*.json")))
        except OSError:
            continue
    return files


def query_egl_devices() -> int:
    """Return how many EGL devices the current loader can enumerate."""

    library = ctypes.CDLL("libEGL.so.1")
    library.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    library.eglGetProcAddress.restype = ctypes.c_void_p
    address = library.eglGetProcAddress(b"eglQueryDevicesEXT")
    if not address:
        raise RuntimeError(
            "libEGL.so.1 does not export eglQueryDevicesEXT; the loaded EGL is "
            "not a device-enumerating GLVND/NVIDIA implementation"
        )
    query = ctypes.CFUNCTYPE(
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )(address)
    devices = (ctypes.c_void_p * _MAX_EGL_DEVICES)()
    count = ctypes.c_int(0)
    if not query(_MAX_EGL_DEVICES, devices, ctypes.byref(count)):
        raise RuntimeError("eglQueryDevicesEXT reported failure")
    return int(count.value)


def probe() -> dict[str, Any]:
    """Report the offscreen graphics binding this interpreter would resolve."""

    # Unset bindings follow the same platform selection as the isolated worker.
    backend = os.environ.get("MUJOCO_GL") or default_libero_backend()
    pyopengl = os.environ.get("PYOPENGL_PLATFORM")
    result: dict[str, Any] = {
        "schema": GRAPHICS_PROBE_SCHEMA,
        "backend": backend,
        "environment": {
            name: os.environ.get(name) for name in _REPORTED_ENVIRONMENT
        },
        "ld_library_path_configured": bool(os.environ.get("LD_LIBRARY_PATH")),
        "vendor_icd_files": [],
        "egl_devices": None,
        "usable": False,
        "error": None,
    }
    expected_platform = _opengl_platform(backend)
    if expected_platform is None:
        result["error"] = f"Unsupported offscreen graphics backend: {backend!r}"
        return result
    if pyopengl is not None and pyopengl != expected_platform:
        result["error"] = (
            f"PYOPENGL_PLATFORM={pyopengl!r} contradicts MUJOCO_GL={backend!r}; "
            "MuJoCo and PyOpenGL would bind different offscreen backends"
        )
        return result
    if backend != "egl":
        # osmesa and glfw are legitimate offscreen backends that do not
        # enumerate EGL devices; report the binding without inventing a verdict.
        result["usable"] = True
        return result
    result["vendor_icd_files"] = _vendor_icd_files()
    try:
        devices = query_egl_devices()
    except (OSError, RuntimeError) as exc:
        result["error"] = str(exc)
        return result
    result["egl_devices"] = devices
    result["usable"] = devices > 0
    if devices == 0:
        result["error"] = (
            "eglQueryDevicesEXT enumerated 0 devices; the EGL vendor ICD or the "
            "driver libraries are absent from this process environment"
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description=__doc__)


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    print(json.dumps(probe(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
