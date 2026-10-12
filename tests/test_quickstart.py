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
