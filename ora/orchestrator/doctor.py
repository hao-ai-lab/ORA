"""Read-only setup checks; never display credential values or start a simulator."""
from __future__ import annotations

import argparse
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
