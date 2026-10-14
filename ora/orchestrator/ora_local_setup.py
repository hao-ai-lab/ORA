"""Explicit, isolated CPU LIBERO installation and read-only readiness checks.

This module never imports a simulator or contacts a policy provider. Only
``setup`` downloads; a normal CLI invocation and ``doctor`` remain read-only.
"""

from __future__ import annotations

import argparse
import ctypes.util
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Bootstrap before the locked Harness dependencies exist. Importing the
# environment package would import NumPy; a regression guard pairs these pins
# with the canonical host-safe benchmark registry.
CODE_REVISION = "2b910b5b5f53016bef9907632f6f840f1ce2229c"
DATA_REVISION = "c86fc3b8293185a6f373677018ff3e37f8391602"
SCHEMA = "ora.ora-local-setup.v1"
CODE_URL = "https://github.com/Zxy-MLlab/LIBERO-PRO.git"
DATA_URL = "https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LocalSetupError(RuntimeError):
    """An isolated local installation could not be completed."""


def default_local_root() -> Path:
    explicit = os.environ.get("ORA_LOCAL_ROOT")
    if explicit:
        return Path(explicit).expanduser().absolute()
    base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return base.expanduser().absolute() / "ora"


def local_paths(root: Path | str | None = None) -> dict[str, Path]:
    base = Path(root).expanduser().absolute() if root is not None else default_local_root()
    return {
        "root": base,
        "source_root": base / "sources" / f"LIBERO-PRO-{CODE_REVISION}",
        "data_root": base / "sources" / f"LIBERO-Pro-data-{DATA_REVISION}",
        "python": base / "runtime/bin/python",
        "manifest": base / "setup.json",
    }


def _run(argv: list[str], *, timeout: float = 30, env: dict[str, str] | None = None) -> str:
    merged = dict(os.environ)
    merged.update({"GIT_LFS_SKIP_SMUDGE": "1", "GIT_OPTIONAL_LOCKS": "0",
                   "GIT_TERMINAL_PROMPT": "0"})
    if env:
        merged.update(env)
    try:
        result = subprocess.run(argv, check=True, capture_output=True, text=True,
                                timeout=timeout, env=merged)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = str(getattr(exc, "stderr", "") or exc).strip()[-2000:]
        raise LocalSetupError(f"{Path(argv[0]).name} failed: {detail}") from exc
    return result.stdout.strip()


def _git(root: Path, *args: str, timeout: float = 30) -> str:
    return _run(["git", "-c", "core.hooksPath=/dev/null", "-C", str(root), *args],
                timeout=timeout)


def _platform() -> tuple[str, str]:
    system, machine = platform.system(), platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        release = platform.mac_ver()[0]
        try:
            major = int(release.split(".", 1)[0])
        except (ValueError, TypeError) as exc:
            raise LocalSetupError("cannot determine macOS version; local CPU setup requires macOS 13 or newer") from exc
        if major < 13:
            raise LocalSetupError(
                f"local CPU setup requires macOS 13 or newer for the pinned OpenCV wheels; found macOS {release}"
            )
        return system, "arm64"
    if system == "Linux" and machine in {"x86_64", "amd64", "aarch64", "arm64"}:
        return system, "arm64" if machine in {"aarch64", "arm64"} else "x86_64"
    raise LocalSetupError(f"local CPU setup supports Apple Silicon macOS and Linux x86_64/arm64; got {system}/{machine}")


def graphics_environment(graphics: str = "auto") -> dict[str, str]:
    system, _ = _platform()
    selected = ("cgl" if system == "Darwin" else "osmesa") if graphics == "auto" else graphics
    allowed = {"cgl", "glfw"} if system == "Darwin" else {"osmesa", "egl", "glfw"}
    if selected not in allowed:
        raise LocalSetupError(f"graphics backend {selected!r} is unsupported on {system}")
    result = {"MUJOCO_GL": selected}
    if selected == "cgl":
        result["PYOPENGL_PLATFORM"] = "darwin"
    elif selected in {"osmesa", "egl"}:
        result["PYOPENGL_PLATFORM"] = selected
    return result


def environment(root: Path | str | None = None) -> dict[str, str]:
    """Return only the graphics keys from the exact local setup manifest."""
    paths = local_paths(root)
    try:
        manifest = _manifest(paths)
    except (OSError, ValueError, LocalSetupError):
        return graphics_environment()
    settings = manifest.get("environment")
    if not isinstance(settings, dict):
        raise LocalSetupError("local setup graphics settings are invalid; rerun ora setup")
    expected = graphics_environment(settings.get("MUJOCO_GL", "auto"))
    if settings != expected:
        raise LocalSetupError("local setup graphics settings are invalid; rerun ora setup")
    return expected


def _manifest(paths: dict[str, Path]) -> dict[str, Any]:
    path = paths["manifest"]
    if path.stat().st_size > 65536:
        raise LocalSetupError("local setup manifest is too large")
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or not (
        manifest.get("schema") == SCHEMA
        and manifest.get("code_revision") == CODE_REVISION
        and manifest.get("data_revision") == DATA_REVISION
        and manifest.get("paths") == {key: str(value) for key, value in paths.items()}
    ):
        raise LocalSetupError("setup manifest identity/path mismatch; rerun ora setup")
    return manifest


def _graphics_check(environment: dict[str, str]) -> tuple[bool, str]:
    backend = environment.get("MUJOCO_GL")
    if platform.system() == "Darwin":
        return True, "macOS system OpenGL; camera rendering is checked on the first isolated reset"
    if backend == "osmesa":
        found = ctypes.util.find_library("OSMesa")
        return bool(found), found or "OSMesa is missing; install your distribution's Mesa OSMesa runtime, then rerun ora doctor"
    if backend == "egl":
        found = ctypes.util.find_library("EGL")
        return bool(found), found or "EGL is missing; install a Mesa EGL runtime or select --graphics osmesa"
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")), "GLFW requires an active desktop display"


def doctor(root: Path | str | None = None) -> dict[str, Any]:
    """Inspect local files, Git identities and package metadata without a rollout."""
    paths = local_paths(root)
    checks: list[dict[str, Any]] = []
    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    try:
        _platform()
        add("platform", True, f"{platform.system()}/{platform.machine()}")
    except LocalSetupError as exc:
        add("platform", False, str(exc))
    manifest: dict[str, Any] = {}
    try:
        manifest = _manifest(paths)
        add("setup_manifest", True, "matching isolated installation")
    except (OSError, ValueError, LocalSetupError) as exc:
        add("setup_manifest", False, f"run ora setup first: {exc}")
    for name, revision in (("source_root", CODE_REVISION), ("data_root", DATA_REVISION)):
        try:
            current = _git(paths[name], "rev-parse", "HEAD", timeout=10)
            clean = not _git(paths[name], "status", "--porcelain", "--untracked-files=no", timeout=15)
            add(name, current == revision and clean, {"expected": revision, "actual": current, "clean": clean})
        except LocalSetupError as exc:
            add(name, False, str(exc))
    for name, directory, pattern in (
        ("assets", paths["source_root"] / "libero/libero/assets", "**/*.xml"),
        ("base_bddl", paths["source_root"] / "libero/libero/bddl_files", "**/*.bddl"),
        ("base_init_states", paths["source_root"] / "libero/libero/init_files", "**/*.pruned_init"),
        ("pro_bddl", paths["data_root"] / "bddl_files", "**/*.bddl"),
        ("pro_init_states", paths["data_root"] / "init_files", "**/*.pruned_init"),
    ):
        sample = next(directory.glob(pattern), None)
        valid = bool(sample and sample.is_file() and sample.stat().st_size > 0)
        if valid:
            with sample.open("rb") as stream:
                valid = not stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1")
        add(name, valid, str(sample or directory))
    try:
        requirements = runtime_requirements()
        package_data = _run([str(paths["python"]), "-I", "-B", "-c", _METADATA_PROBE], timeout=15)
        observed = json.loads(package_data)
        mismatched = {name: {"expected": version, "actual": observed["packages"].get(name)}
                      for name, version in requirements.items()
                      if observed["packages"].get(name) != version}
        add("runtime_python", observed["python"][:2] == [3, 11], observed["python"])
        add("runtime_packages", not mismatched, mismatched or "all CPU runtime dependencies match frozen pins")
    except (LocalSetupError, ValueError, KeyError) as exc:
        add("runtime_python", False, str(exc))
    try:
        environment = manifest.get("environment", graphics_environment())
        expected = graphics_environment(environment.get("MUJOCO_GL", "auto"))
        add("graphics_configuration", environment == expected, environment)
        passed, detail = _graphics_check(expected)
        add("graphics_runtime", passed, detail)
    except (LocalSetupError, AttributeError) as exc:
        add("graphics_runtime", False, str(exc))
        environment = {}
    return {"ready": all(check["passed"] for check in checks), "checks": checks,
            "paths": {key: str(value) for key, value in paths.items()},
            "code_revision": CODE_REVISION, "data_revision": DATA_REVISION,
            "environment": environment, "manifest": manifest,
            "qualification": "installation checks only; no simulator or provider was started"}


_METADATA_PROBE = """import importlib.metadata as m,json,re,sys
print(json.dumps({'python':list(sys.version_info[:3]),'packages':{
re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version for d in m.distributions()}}))
"""


def runtime_requirements() -> dict[str, str]:
    """Frozen CPU worker closure, separate from the Harness lock.

    Intentionally substitute opencv-python-headless for RoboSuite's declared
    opencv-python dependency at the same cv2 version. The direct LIBERO route
    disables its onscreen renderer and uses MuJoCo for offscreen RGB-D; no cv2
    GUI is needed. Headless cv2 avoids an unrelated libGL.so.1 dependency on
    CPU hosts. Installation uses --no-deps to preserve this reviewed choice.
    """
    system, _ = _platform()
    lines = _COMMON_REQUIREMENTS.splitlines()
    lines += (_MAC_REQUIREMENTS if system == "Darwin" else _LINUX_REQUIREMENTS).splitlines()
    return dict(line.split("==", 1) for line in lines if line)


def _checkout(path: Path, url: str, revision: str) -> None:
    if path.is_symlink():
        raise LocalSetupError(f"refusing a symlinked source directory: {path}")
    if path.exists():
        if not (path / ".git").is_dir():
            raise LocalSetupError(f"refusing to reuse a non-repository directory: {path}")
        try:
            actual = _git(path, "rev-parse", "HEAD")
        except LocalSetupError:
            # A fetch interrupted before the first checkout is safe to resume
            # only in an empty repository already bound to this exact origin.
            if (_git(path, "remote", "get-url", "origin") != url
                    or _git(path, "ls-files")
                    or _git(path, "status", "--porcelain", "--untracked-files=all")):
                raise LocalSetupError(f"refusing to reuse an incomplete foreign checkout: {path}")
        else:
            if actual != revision or _git(path, "status", "--porcelain", "--untracked-files=no"):
                raise LocalSetupError(f"refusing to change an existing modified/wrong-revision checkout: {path}")
            return
    else:
        path.mkdir(parents=True)
        _git(path, "init")
        _git(path, "remote", "add", "origin", url)
    _git(path, "fetch", "--depth", "1", "origin", revision, timeout=1800)
    _git(path, "checkout", "--detach", revision, timeout=300)


def setup(root: Path | str | None = None, *, python: str | Path | None = None,
          graphics: str = "auto") -> dict[str, Any]:
    """Download pinned sources and install an isolated CPU-only Python runtime."""
    _platform()
    environment = graphics_environment(graphics)
    paths = local_paths(root)
    uv = shutil.which("uv")
    if not uv or not shutil.which("git"):
        raise LocalSetupError("ora setup needs uv and Git on PATH")
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        raise LocalSetupError("ora setup needs Node.js 20 or newer and npm; install Node.js from https://nodejs.org/ then rerun ora setup")
    node_version = _run([node, "--version"])
    match = re.fullmatch(r"v(\d+)\.\d+\.\d+", node_version)
    if not match or int(match.group(1)) < 20:
        raise LocalSetupError(f"terminal UI requires Node.js 20 or newer; found {node_version}")
    if not (PROJECT_ROOT / "uv.lock").is_file() or not (PROJECT_ROOT / "ora-tui/package-lock.json").is_file():
        raise LocalSetupError("run ora setup from a source checkout containing uv.lock and ora-tui/package-lock.json")
    interpreter = str(python or sys.executable)
    version = json.loads(_run([interpreter, "-I", "-c", "import json,sys;print(json.dumps(list(sys.version_info[:3])))"]))
    if version[:2] != [3, 11]:
        raise LocalSetupError("isolated LIBERO requires Python 3.11; pass --python /path/to/python3.11")
    _run([uv, "sync", "--project", str(PROJECT_ROOT), "--all-extras", "--locked",
          "--python", interpreter, "--no-python-downloads"], timeout=1800)
    _run([npm, "ci", "--prefix", str(PROJECT_ROOT / "ora-tui"), "--no-audit", "--no-fund"], timeout=600)
    paths["root"].mkdir(parents=True, exist_ok=True)
    _checkout(paths["source_root"], CODE_URL, CODE_REVISION)
    _checkout(paths["data_root"], DATA_URL, DATA_REVISION)
    runtime = paths["python"].parent.parent
    if runtime.is_symlink():
        raise LocalSetupError(f"refusing a symlinked runtime directory: {runtime}")
    if runtime.exists() and not (runtime / "pyvenv.cfg").is_file():
        raise LocalSetupError(f"refusing to reuse a non-virtualenv directory: {runtime}")
    if not runtime.exists():
        _run([uv, "venv", "--python", interpreter, "--no-python-downloads", str(runtime)], timeout=120)
    runtime_version = json.loads(_run([str(paths["python"]), "-I", "-B", "-c",
                                      "import json,sys;print(json.dumps(list(sys.version_info[:3])))"]))
    if runtime_version[:2] != [3, 11]:
        raise LocalSetupError(f"existing simulator virtualenv is not Python 3.11: {runtime}")
    requirements = runtime_requirements()
    # Build helpers are pinned too: Gym/evdev are source distributions. Disabling
    # build isolation prevents an unpinned temporary build environment.
    helpers = [f"{name}=={requirements[name]}" for name in ("setuptools", "wheel", "packaging")]
    _run([uv, "pip", "install", "--python", str(paths["python"]), "--no-deps", *helpers], timeout=600)
    requirements_file = paths["root"] / "runtime-requirements.txt"
    requirements_file.write_text("\n".join(f"{name}=={version}" for name, version in sorted(requirements.items())) + "\n")
    install = [uv, "pip", "install", "--python", str(paths["python"]), "--no-deps", "--no-build-isolation",
               "--torch-backend", "cpu", "-r", str(requirements_file)]
    _run(install, timeout=3600)
    manifest = {"schema": SCHEMA, "code_revision": CODE_REVISION, "data_revision": DATA_REVISION,
                "paths": {key: str(value) for key, value in paths.items()},
                "environment": environment, "runtime_requirements": requirements,
                "requirements_sha256": hashlib.sha256(requirements_file.read_bytes()).hexdigest(),
                "created_unix_s": time.time(), "simulator_started": False}
    temporary = paths["manifest"].with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(paths["manifest"])
    return doctor(paths["root"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ora setup", description="Install the locked Harness and terminal UI, pinned CPU LIBERO sources/assets and a separate simulator runtime (about 2 GB; no provider or simulator starts)")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--python", type=Path, help="existing Python 3.11 interpreter")
    parser.add_argument("--graphics", choices=("auto", "cgl", "glfw", "egl", "osmesa"), default="auto")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.json:
        print("Preparing the locked terminal UI and pinned CPU LIBERO installation. Downloads can take several minutes.", flush=True)
    try:
        result = setup(args.root, python=args.python, graphics=args.graphics)
    except (LocalSetupError, OSError, ValueError) as exc:
        if args.json:
            print(json.dumps({"ready": False, "error": str(exc)}))
        else:
            print(f"Local setup incomplete: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Installed pinned CPU LIBERO in {result['paths']['root']}")
        for check in result["checks"]:
            if not check["passed"]:
                print(f"Needs attention: {check['name']}: {check['detail']}")
        print("Ready for task selection." if result["ready"] else "Run ora doctor after resolving the checks above.")
    return 0 if result["ready"] else 1


_COMMON_REQUIREMENTS = """absl-py==2.5.1
attrs==26.1.0
bddl==3.6.0
click==8.5.0
cloudpickle==3.1.1
contourpy==1.3.3
cycler==0.12.1
defusedxml==0.7.1
easydict==1.13
einops==0.8.1
etils==1.14.0
farama-notifications==0.0.6
fastjsonschema==2.22.2
filelock==4.1.1
fonttools==4.66.1
fsspec==2026.9.0
future==1.0.0
glfw==2.10.2
gym==0.26.2
gym-notices==0.1.0
gymnasium==0.29.1
h5py==3.14.0
imageio==2.37.4
imageio-ffmpeg==0.6.0
iniconfig==2.3.1
jinja2==3.1.6
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-core==5.9.1
jupytext==1.19.6
kiwisolver==1.5.1
llvmlite==0.50.0
markdown-it-py==4.2.0
markupsafe==3.0.4
matplotlib==3.11.1
mdit-py-plugins==0.6.1
mdurl==0.1.2
mpmath==1.3.0
mujoco==3.8.1
nbformat==5.11.1
networkx==3.6.1
nltk==3.10.3
numba==0.68.0
numpy==1.26.4
opencv-python-headless==4.11.0.86
packaging==26.2
pillow==12.2.0
platformdirs==4.13.0
pluggy==1.6.0
pygments==2.21.0
pynput==1.8.2
pyopengl==3.1.10
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
pyyaml==6.0.2
referencing==0.37.0
regex==2026.9.29
robosuite==1.4.1
rpds-py==2026.9.1
scipy==1.16.1
setuptools==84.0.0
six==1.17.0
sympy==1.13.1
termcolor==3.1.0
tqdm==4.70.1
traitlets==5.16.1
typing-extensions==4.16.0
wheel==0.48.0
zipp==4.1.1"""
_MAC_REQUIREMENTS = """pyobjc-core==12.2.2
pyobjc-framework-applicationservices==12.2.2
pyobjc-framework-cocoa==12.2.2
pyobjc-framework-coretext==12.2.2
pyobjc-framework-quartz==12.2.2
torch==2.6.0"""
_LINUX_REQUIREMENTS = """evdev==2.0.0
python-xlib==0.33
torch==2.6.0+cpu"""


if __name__ == "__main__":
    raise SystemExit(main())
