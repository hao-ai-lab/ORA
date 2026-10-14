"""Local entry configuration and authorization without MuJoCo or Provider calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from ora.orchestrator import ora_cli, ora_local
from ora.orchestrator import resources
from ora.orchestrator.resources import HOST_MARKER_ENV


@pytest.fixture
def installation(tmp_path, monkeypatch):
    import ora.orchestrator as package

    module = types.ModuleType("ora.orchestrator.ora_local_setup")
    module.CODE_REVISION, module.DATA_REVISION = "c" * 40, "d" * 40
    installed = {"root": tmp_path, "source_root": tmp_path / "source", "data_root": tmp_path / "data",
                 "python": tmp_path / "runtime/bin/python"}
    module.local_paths = lambda root=None: installed
    module.graphics_environment = lambda graphics="auto": {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}
    module.environment = lambda root=None: module.graphics_environment()
    module.doctor = lambda root=None: {"ready": True, "checks": [
        {"name": "local_installation", "passed": True, "detail": "configured"}], "paths": installed}
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(package, "ora_local_setup", module, raising=False)
    return module, installed


def test_local_defaults_bind_cpu_identity_and_only_alias_missing_credential(installation):
    _, installed = installation
    credential = "private-api-key-do-not-print"
    values = {"DEEPSEEK_API_KEY": credential}
    result = ora_local.local_environment(environ=values)
    assert result[HOST_MARKER_ENV] == "local-cpu"
    assert result["LIBERO_PRO_ROOT"] == str(installed["source_root"])
    assert result["DEEPSEEK_LOW_API_KEY"] == credential
    assert values == {"DEEPSEEK_API_KEY": credential}  # caller is unchanged
    assert "DEEPSEEK_LOW_API_KEY" not in ora_local.local_environment(
        environ={**values, "DEEPSEEK_LOW_API_KEY": "canonical-wins"})


def test_partial_external_source_and_conflicting_host_are_refused(installation):
    with pytest.raises(RuntimeError, match="set all"):
        ora_local.local_environment(environ={"LIBERO_PRO_ROOT": "/external/source"})
    with pytest.raises(RuntimeError, match="existing host marker"):
        ora_local.local_environment(environ={HOST_MARKER_ENV: "qualified-other-host"})
    values = {"LIBERO_PRO_ROOT": "/external/source", "LIBERO_PRO_DATA_ROOT": "/external/data",
              "LIBERO_PRO_PYTHON": "/external/python", HOST_MARKER_ENV: "local-cpu"}
    result = ora_local.local_environment(environ=values)
    assert not (set(result) & set(ora_local.SOURCE_ENV.values()))


def test_doctor_is_read_only_reports_variable_names_and_still_requires_preflight(installation, monkeypatch):
    inspected = []

    def read_source(command, **kwargs):
        inspected.append(command)
        assert command[0] == "git"
        assert kwargs["env"]["GIT_OPTIONAL_LOCKS"] == "0"
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(ora_local.subprocess, "run", read_source)
    monkeypatch.setattr(resources, "require_registered_host", lambda name: name)
    monkeypatch.setattr(ora_local.watch_cli, "_pi_front_end", lambda: ("node", Path("index.mjs")))
    values = {"DEEPSEEK_API_KEY": "never-in-report"}
    result = ora_local.doctor(environ=values)
    assert result["ready"] and result["live_preflight_required"]
    assert result["default"]["provider_profile"] == "deepseek_flash_openai_low_16k_v1"
    credential = next(check for check in result["checks"] if check["name"] == "provider_credential")
    assert credential["detail"]["credential_env"] == "DEEPSEEK_API_KEY"
    assert "never-in-report" not in json.dumps(result)
    assert values == {"DEEPSEEK_API_KEY": "never-in-report"}
    assert len(inspected) == 1


def test_doctor_does_not_treat_marker_conflict_as_ready(installation, monkeypatch):
    monkeypatch.setattr(ora_local.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr(resources, "require_registered_host", lambda name: name)
    result = ora_local.doctor(environ={HOST_MARKER_ENV: "ssh23", "DEEPSEEK_LOW_API_KEY": "secret"},
                              require_frontend=False)
    assert not result["ready"]
    assert any(check["name"] == "local_bindings" and not check["passed"] for check in result["checks"])


@pytest.mark.parametrize("confirmed", [False, True])
def test_local_forwarding_keeps_explicit_launch_confirmation_and_world_options(installation, monkeypatch, confirmed):
    monkeypatch.setattr(ora_local, "doctor", lambda *args, **kwargs: {"ready": True})
    calls = []
    monkeypatch.setattr(ora_local, "prepare_task", lambda **kwargs: calls.append(kwargs) or Path("/prepared/task.json"))
    args = ora_cli.build_parser().parse_args(["local", "--emit", "jsonl", "--world-browser", "off",
                                            "--world-port", "12345", *(["--confirm-launch"] if confirmed else [])])
    environment = {"DEEPSEEK_API_KEY": "credential"}
    forwarded = ora_local.interactive_arguments(args, environment=environment, world_explicit=True)
    assert forwarded[:3] == ["interactive", "--spec", "/prepared/task.json"]
    assert ("--confirm-launch" in forwarded) is confirmed
    assert forwarded[forwarded.index("--world-port") + 1] == "12345"
    assert forwarded[-2:] == ["--emit", "jsonl"]
    assert calls[0]["suite"] == "libero_goal" and calls[0]["task_id"] == calls[0]["seed"] == 0
    assert environment[HOST_MARKER_ENV] == "local-cpu"


def test_incomplete_local_inputs_never_prepare_or_pollute_json_stream(installation, monkeypatch, capsys):
    report = {"ready": False, "checks": [{"name": "missing_input", "passed": False, "detail": "missing"}]}
    monkeypatch.setattr(ora_local, "doctor", lambda *args, **kwargs: report)
    monkeypatch.setattr(ora_local, "prepare_task", lambda **kwargs: pytest.fail("must not prepare incomplete inputs"))
    args = ora_cli.build_parser().parse_args(["local", "--emit", "jsonl"])
    with pytest.raises(RuntimeError, match="incomplete"):
        ora_local.interactive_arguments(args, environment={})
    captured = capsys.readouterr()
    assert captured.out == "" and "setup needs attention" in captured.err


def test_no_argument_entry_uses_local_task_picker_without_implicit_authorization(monkeypatch):
    from ora.orchestrator import ora_backend

    calls = []

    def local(args, **kwargs):
        calls.append(args)
        assert kwargs == {"world_explicit": False}
        return ["interactive", "--spec", "/prepared/task.json", "--emit", "jsonl", "--world-view", "off"]

    monkeypatch.setattr(ora_local, "interactive_arguments", local)
    monkeypatch.setattr(ora_backend, "serve", lambda specs, **kwargs: calls.append(kwargs) or 0)
    assert ora_cli.main([]) == 0
    assert calls[0].command == "local" and not calls[0].confirm_launch
    assert calls[1]["confirm_launch"] is False


@pytest.mark.parametrize("world,enabled", [([], False), (["--world-port", "0"], True),
    (["--world-public-url", "http://127.0.0.1:40201/live"], True),
    (["--world-port", "0", "--world-view", "off"], False)])
def test_headless_local_world_view_requires_the_operators_actual_opt_in(installation, monkeypatch, world, enabled):
    from ora.orchestrator import ora_backend

    calls = []
    monkeypatch.setattr(ora_local, "doctor", lambda *args, **kwargs: {"ready": True})
    monkeypatch.setattr(ora_local, "prepare_task", lambda **kwargs: Path("/prepared/task.json"))
    monkeypatch.setattr(ora_backend, "serve", lambda specs, **kwargs: calls.append(kwargs) or 0)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    with patch.dict(os.environ, {"DEEPSEEK_LOW_API_KEY": "never-reported"}, clear=True):
        assert ora_cli.main(["local", "--emit", "jsonl", *world]) == 0
    assert (calls[0]["observer_factory"] is not None) is enabled
    assert calls[0]["confirm_launch"] is False


def test_setup_delegation_preserves_literal_arguments(installation):
    module, _ = installation
    calls = []
    module.main = lambda values: calls.append(values) or 0
    assert ora_cli.main(["setup", "--root", "/space path/$(literal)", "--graphics", "osmesa"]) == 0
    assert calls == [["--root", "/space path/$(literal)", "--graphics", "osmesa"]]


def test_local_task_input_is_source_bound_content_addressed_and_never_contains_credentials(installation, monkeypatch, tmp_path):
    from ora.agent import composition
    from ora.orchestrator import spec as spec_module

    module = types.ModuleType("ora.agent.composition.ora_local")
    calls = []

    def compose(**kwargs):
        calls.append(kwargs)
        return {"task": kwargs["task"], "source": kwargs["source"], "provider": kwargs["provider"], "seed": kwargs["seed"]}

    module.compose_local_spec = compose
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(composition, "ora_local", module, raising=False)
    monkeypatch.setattr(spec_module, "resolve_run_spec", lambda value: {})
    monkeypatch.setattr(ora_local.paths, "link_logical_workspace", lambda name: tmp_path / name)
    monkeypatch.setenv("DEEPSEEK_LOW_API_KEY", "never-in-spec")
    first = ora_local.prepare_task(suite="libero_goal", task_id=0, seed=0)
    assert first == ora_local.prepare_task(suite="libero_goal", task_id=0, seed=0)
    saved = json.loads(first.read_text())
    assert "never-in-spec" not in first.read_text()
    assert saved["source"]["required_libero_pro_revision"] == "c" * 40
    assert saved["provider"]["provider_profile"] == "deepseek_flash_openai_low_16k_v1"
    assert calls[0]["task"]["task_language"] == "open the middle drawer of the cabinet"
    assert calls[0]["task"]["task_language_divergence_ack"]["bddl_task_language"] == "Open the middle layer of the drawer"
    assert first.parent.name == "specs" and len(list(first.parent.iterdir())) == 1
    first.write_text("changed")
    with pytest.raises(RuntimeError, match="RunSpec changed"):
        ora_local.prepare_task(suite="libero_goal", task_id=0, seed=0)


def test_doctor_json_exit_code_matches_ready(installation, monkeypatch, capsys):
    monkeypatch.setattr(ora_local, "doctor", lambda root: {"ready": False, "checks": []})
    assert ora_cli.main(["doctor", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["ready"] is False


def test_local_entry_names_match_canonical_source_contracts():
    from ora.environment.benchmarks.libero_pro import BASE_SUITES

    assert ora_local.LOCAL_SUITES == BASE_SUITES
    assert ora_local.HOST_MARKER_ENV == resources.HOST_MARKER_ENV


@pytest.mark.parametrize("command,expected", [(["watch", "--help"], 0), ([], 2)])
def test_entry_without_site_packages_has_actionable_bootstrap_or_read_only_help(command, expected):
    root = Path(__file__).parents[1]
    script = "import sys;sys.path.insert(0,sys.argv[1]);from ora.orchestrator.ora_cli import main;sys.exit(main(sys.argv[2:]))"
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script, str(root), *command],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == expected
    assert "Traceback" not in result.stderr
    if expected == 2:
        assert "./scripts/ora setup" in result.stderr
    else:
        assert "Read-only" in result.stdout
