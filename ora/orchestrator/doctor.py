"""Read-only setup checks; never display credential values or start a simulator."""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

SIMULATORS = {
    "robotwin": ("ROBOTWIN_PYTHON", "sapien", ("ROBOTWIN_ROOT",)),
    "libero_pro": ("LIBERO_PRO_PYTHON", "libero", ("LIBERO_PRO_ROOT", "LIBERO_PRO_DATA_ROOT")),
    "robodojo": ("ROBODOJO_PYTHON", "isaacsim", ("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT")),
}


ROBODOJO_GRAPHICS_DOCS = "docs/environments/robodojo.md#vulkan-loader-and-compatibility-layer"
#: Drivers whose Vulkan allocation limit overflows Isaac Sim 5.1 (robodojo_qualification.validate_runtime_driver).
COMPAT_LAYER_DRIVERS = ("595.",)
GRAPHICS_COMPAT_SCRIPT = Path(__file__).resolve().parents[1] / "environment/backends/robodojo/graphics_compat.py"
RECEIPT_CHECK = ("import importlib.util, sys; from pathlib import Path; "
                 "spec = importlib.util.spec_from_file_location('graphics_compat', sys.argv[1]); "
                 "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                 "module.installation(Path(sys.argv[2]))")


def _robodojo_graphics_checks(interpreter: str, nvidia_smi: str | None) -> list[dict]:
    """Vulkan loader and, where the driver needs it, the compatibility-layer receipt.

    The loader is probed in the simulator interpreter (falling back to this one), which
    inherits ``LD_LIBRARY_PATH`` exactly as the isolated worker does. Nothing is started.
    """
    probe = "import ctypes; ctypes.CDLL('libvulkan.so.1')"
    try:
        if interpreter:
            loaded = subprocess.run([interpreter, "-c", probe], capture_output=True, timeout=15, env={
                k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}}).returncode == 0
        else:
            ctypes.CDLL("libvulkan.so.1")
            loaded = True
    except (OSError, subprocess.TimeoutExpired):
        loaded = False
    checks = [{"check": "vulkan_loader", "passed": loaded, "detail": "libvulkan.so.1 loadable" if loaded else
               "libvulkan.so.1 is not loadable: install the Vulkan loader (e.g. libvulkan1) or add its "
               f"directory to LD_LIBRARY_PATH; see {ROBODOJO_GRAPHICS_DOCS}"}]
    driver = ""
    if nvidia_smi:
        try:
            result = subprocess.run([nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=5)
            driver = (result.stdout.strip().splitlines() or [""])[0].strip() if result.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            driver = ""
    if not driver.startswith(COMPAT_LAYER_DRIVERS):
        checks.append({"check": "vulkan_compatibility_layer", "passed": True,
                       "detail": f"not required by driver {driver or 'unknown'}"})
        return checks
    runtime = Path(interpreter or os.environ.get("ROBODOJO_PYTHON", "")).parent.parent.parent
    # The worker's own receipt check, run by path in a child: host layers never import backends.
    try:
        result = subprocess.run([sys.executable, "-c", RECEIPT_CHECK, str(GRAPHICS_COMPAT_SCRIPT), str(runtime)],
                                capture_output=True, text=True, timeout=30)
        ok = result.returncode == 0
        error = (result.stderr.strip().splitlines() or ["receipt check failed"])[-1]
    except (OSError, subprocess.TimeoutExpired) as exc:
        ok, error = False, f"{type(exc).__name__}: {exc}"
    detail = (f"receipt verified under {runtime / 'graphics-compat'} (driver {driver})" if ok else
              f"driver {driver} needs the Vulkan compatibility layer, but "
              f"{runtime / 'graphics-compat/installation.json'} is missing or invalid ({error}); "
              f"build it with `scripts/setup-robodojo graphics-compat`, see {ROBODOJO_GRAPHICS_DOCS}")
    checks.append({"check": "vulkan_compatibility_layer", "passed": ok, "detail": detail})
    return checks


def report(environment: str = "synthetic") -> dict:
    try:
        version = importlib.metadata.version("ora-agent")
    except importlib.metadata.PackageNotFoundError:
        version = "source checkout (not installed)"
    checks = [{"check": "python", "passed": sys.version_info >= (3, 11),
               "detail": platform.python_version()}]
    credentials = {name: bool(os.environ.get(name)) for name in (
        "ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NATIVE_MODEL_API_KEY",
        "DEEPSEEK_LOW_API_KEY", "DEEPSEEK_API_KEY",
    )}
    optional = {"credentials_present": credentials,
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "egl_library": bool(ctypes.util.find_library("EGL")),
                "vulkan_library": bool(ctypes.util.find_library("vulkan"))}
    gpu = shutil.which("nvidia-smi")
    optional["nvidia_smi"] = bool(gpu)
    if environment != "synthetic":
        var, module, asset_vars = SIMULATORS[environment]
        interpreter = os.environ.get(var, "")
        present = bool(interpreter and Path(interpreter).is_file())
        detail = f"Set {var} to the isolated simulator interpreter"
        if present:
            discovery = "import importlib.util; "
            if environment == "libero_pro" and os.environ.get("LIBERO_PRO_ROOT"):
                source = str(Path(os.environ["LIBERO_PRO_ROOT"]) / "libero")
                discovery += "import sys; sys.path.insert(0, " + repr(source) + "); "
            try:
                probe = subprocess.run([interpreter, "-c",
                    discovery + "raise SystemExit(0 if importlib.util.find_spec("
                    + repr(module) + ") else 1)"], capture_output=True, timeout=15,
                    env={k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "PYTHONHOME"}})
                present = probe.returncode == 0
                detail = f"{module}: {'discoverable' if present else 'missing'}; simulator not started"
            except (OSError, subprocess.TimeoutExpired):
                present = False
                detail = f"{var} probe failed or timed out"
        checks.append({"check": var, "passed": present, "detail": detail})
        for var in asset_vars:
            value = os.environ.get(var, "")
            checks.append({"check": var, "passed": bool(value and Path(value).is_dir()),
                           "detail": "directory exists" if value and Path(value).is_dir() else "unset or missing"})
        gpu_ready = False
        if gpu:
            try:
                probe = subprocess.run([gpu, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5)
                optional["gpus"] = probe.stdout.strip().splitlines() if probe.returncode == 0 else []
                gpu_ready = bool(optional["gpus"])
            except (OSError, subprocess.TimeoutExpired):
                optional["gpus"] = []
        checks.append({"check": "nvidia_driver", "passed": gpu_ready,
                       "detail": "visible GPUs discovered" if gpu_ready else "no GPU discovered by nvidia-smi"})
        if environment == "robodojo":
            checks.extend(_robodojo_graphics_checks(interpreter if present else "", gpu))
    return {"ora_version": version, "platform": platform.system(), "environment": environment,
            "passed": all(item["passed"] for item in checks), "checks": checks,
            "optional": optional,
            "note": "Discovery checks only. Launch preflight verifies exact sources, assets, services and qualification."}


def command_doctor(args: argparse.Namespace) -> int:
    result = report(args.env)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"ORA {result['ora_version']} · Python {platform.python_version()} · {result['platform']}")
        for item in result["checks"]:
            print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}: {item['detail']}")
        print("Credentials present: " + ", ".join(k for k, v in result["optional"]["credentials_present"].items() if v)
              if any(result["optional"]["credentials_present"].values()) else "Credentials present: none")
        print(result["note"])
    return 0 if result["passed"] else 1
