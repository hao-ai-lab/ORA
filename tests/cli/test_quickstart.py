"""Public installation checks and the documented Native Synthetic quickstart."""
from __future__ import annotations

import json
import pytest

from ora.orchestrator import doctor, quickstart


def test_doctor_never_displays_credential_values(monkeypatch):
    monkeypatch.setenv("ORA_API_KEY", "private-value-must-never-appear")
    result = doctor.report()
    assert result["passed"]
    assert result["optional"]["credentials_present"]["ORA_API_KEY"] is True
    assert "private-value-must-never-appear" not in json.dumps(result)


def test_quickstart_requires_a_key_and_an_explicit_anthropic_model(monkeypatch):
    for key in ("ORA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ORA_MODEL", "ORA_PROVIDER", "ORA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="Set ORA_API_KEY"):
        quickstart.synthetic_spec()
    monkeypatch.setenv("ORA_PROVIDER", "anthropic_messages")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder")
    with pytest.raises(ValueError, match="Set ORA_MODEL"):
        quickstart.synthetic_spec()


def test_documented_quickstart_completes_and_seals_an_episode():
    from examples.quickstart_smoke import smoke
    smoke()


def test_libero_doctor_uses_the_declared_source_without_starting_the_simulator(monkeypatch, tmp_path):
    from types import SimpleNamespace
    interpreter = tmp_path / "python"
    interpreter.touch()
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("LIBERO_PRO_PYTHON", str(interpreter))
    monkeypatch.setenv("LIBERO_PRO_ROOT", str(source))
    monkeypatch.setenv("LIBERO_PRO_DATA_ROOT", str(source))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    calls = []
    def probe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="0, test GPU, 8192\n")
    monkeypatch.setattr(doctor.subprocess, "run", probe)
    result = doctor.report("libero_pro")
    assert result["passed"] is True
    assert str(source / "libero") in calls[0][-1]
    assert "find_spec" in calls[0][-1] and "import libero" not in calls[0][-1]
    assert calls[1][1] == "--query-gpu=index,name,memory.total"


def _robodojo_doctor(monkeypatch, tmp_path, *, loader: bool, driver: str):
    """`ora doctor --env robodojo` with a fake interpreter, loader probe and driver."""
    from types import SimpleNamespace
    runtime = tmp_path / "robodojo-runtime"
    interpreter = runtime / "env/bin/python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.touch()
    for name in ("ROBODOJO_ROOT", "ROBODOJO_ASSET_ROOT"):
        (tmp_path / name).mkdir(exist_ok=True)
        monkeypatch.setenv(name, str(tmp_path / name))
    monkeypatch.setenv("ROBODOJO_PYTHON", str(interpreter))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    calls = []
    real_run = doctor.subprocess.run

    def run(command, **kwargs):
        calls.append(command)
        if command[1:3] == ["-c", doctor.RECEIPT_CHECK]:
            return real_run(command, **kwargs)  # the real receipt check, in a child process
        if command[-1] == "import ctypes; ctypes.CDLL('libvulkan.so.1')":
            return SimpleNamespace(returncode=0 if loader else 1, stdout="")
        if "--query-gpu=driver_version" in command:
            return SimpleNamespace(returncode=0, stdout=f"{driver}\n")
        return SimpleNamespace(returncode=0, stdout="0, test GPU, 8192\n")
    monkeypatch.setattr(doctor.subprocess, "run", run)
    result = doctor.report("robodojo")
    return {item["check"]: item for item in result["checks"]}, result, runtime, calls


def _write_compat_installation(runtime):
    import hashlib
    from pathlib import Path
    installation_dir = Path(doctor.__file__).resolve().parents[1] / "environment/backends/robodojo/installation"
    out = runtime / "graphics-compat"
    files = {"layer": out / "lib/libVkLayer_khronos_profiles.so",
             "profile": out / "profiles/VP_ROBODOJO_RTX_ALLOCATION_COMPAT.json",
             "manifest": out / "share/vulkan/implicit_layer.d/VkLayer_khronos_profiles.json",
             "patch": out / "vulkan-profiles-core-id.patch"}
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    files["layer"].write_bytes(b"layer")
    files["profile"].write_bytes((installation_dir / "VP_ROBODOJO_RTX_ALLOCATION_COMPAT.json").read_bytes())
    files["patch"].write_bytes((installation_dir / "vulkan-profiles-core-id.patch").read_bytes())
    files["manifest"].write_text(json.dumps({"layer": {
        "name": "VK_LAYER_KHRONOS_profiles", "type": "GLOBAL", "library_path": str(files["layer"]),
        "enable_environment": {"ROBODOJO_VULKAN_COMPAT_ENABLE": "1"},
        "disable_environment": {"ROBODOJO_VULKAN_COMPAT_DISABLE": ""}}}))
    refs = {role: {"path": path.relative_to(out).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for role, path in files.items()}
    (out / "installation.json").write_text(json.dumps({
        "schema": "ora.robodojo-graphics-compat-installation.v1",
        "source_url": "https://github.com/KhronosGroup/Vulkan-Profiles",
        "source_revision": "a400d0b10059b2cc3cd285475593eee79975a5c4", "patches": [refs["patch"]], **refs}))


def test_robodojo_doctor_reports_a_missing_vulkan_loader_and_compat_layer(monkeypatch, tmp_path):
    checks, result, _, calls = _robodojo_doctor(monkeypatch, tmp_path, loader=False, driver="595.91.07")
    assert result["passed"] is False
    assert checks["vulkan_loader"]["passed"] is False
    assert "LD_LIBRARY_PATH" in checks["vulkan_loader"]["detail"]
    assert "robodojo.md" in checks["vulkan_loader"]["detail"]
    assert checks["vulkan_compatibility_layer"]["passed"] is False
    assert "setup-robodojo graphics-compat" in checks["vulkan_compatibility_layer"]["detail"]
    assert any(command[-1].startswith("import ctypes") and command[0].endswith("env/bin/python") for command in calls)


def test_robodojo_doctor_passes_with_loader_and_verified_receipt(monkeypatch, tmp_path):
    runtime = tmp_path / "robodojo-runtime"
    _write_compat_installation(runtime)
    checks, result, _, _ = _robodojo_doctor(monkeypatch, tmp_path, loader=True, driver="595.91.07")
    assert checks["vulkan_loader"]["passed"] is True
    assert checks["vulkan_compatibility_layer"]["passed"] is True, checks["vulkan_compatibility_layer"]
    assert result["passed"] is True
    receipt = runtime / "graphics-compat/installation.json"
    value = json.loads(receipt.read_text())
    receipt.write_text(json.dumps({**value, "schema": "robot-harness.robodojo-graphics-compat-installation.v1"}))
    checks, result, _, _ = _robodojo_doctor(monkeypatch, tmp_path, loader=True, driver="595.91.07")
    assert checks["vulkan_compatibility_layer"]["passed"] is False
    assert "source drift" in checks["vulkan_compatibility_layer"]["detail"]


def test_robodojo_doctor_skips_the_compat_layer_on_other_drivers(monkeypatch, tmp_path):
    checks, result, _, _ = _robodojo_doctor(monkeypatch, tmp_path, loader=True, driver="570.86.15")
    assert checks["vulkan_compatibility_layer"]["passed"] is True
    assert "not required" in checks["vulkan_compatibility_layer"]["detail"]
    assert result["passed"] is True
