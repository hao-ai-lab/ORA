"""Control plane cli contract regressions."""

from __future__ import annotations
import fnmatch
import json
import subprocess
import tomllib
from pathlib import Path
import pytest
from ora.orchestrator import cli as control_cli
from ora.orchestrator.cli import build_parser
from test_control_plane import (
    ROOT,
)


def test_shared_provider_registry_is_inspectable_without_credentials(capsys) -> None:
    assert control_cli.main(["providers"]) == 0
    output = capsys.readouterr().out
    assert "volcengine_agent_plan_glm_5_2_anthropic_v1" in output
    assert "admission=text_tool_qualified" in output
    assert "usage=agent_plan_api" in output
    assert "ark-" not in output


def test_generic_cli_public_surface_and_reference_are_exact() -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    public = tuple(name for name in choices if name not in control_cli.INTERNAL_COMMANDS)
    assert public == control_cli.PUBLIC_COMMANDS
    assert set(choices) == set(control_cli.PUBLIC_COMMANDS + control_cli.INTERNAL_COMMANDS)

    reference = ROOT / "docs/guides/cli-reference.md"
    documented = tuple(
        line.split("`", 2)[1]
        for line in reference.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `")
    )
    assert documented == control_cli.PUBLIC_COMMANDS

    wrapper = ROOT / "scripts/ora"
    for command in control_cli.PUBLIC_COMMANDS:
        completed = subprocess.run(
            [str(wrapper), command, "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert f"usage: ora {command}" in completed.stdout


def test_generic_cli_dispatches_every_public_command_without_live_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "context-manifest.json").write_text(
        json.dumps({"schema": "context-test"}), encoding="utf-8"
    )
    plan = {"plan_id": "plan-test", "plan_sha256": "sha-test"}
    calls: list[str] = []

    monkeypatch.setattr(control_cli, "load_run_spec", lambda _path: {"spec": True})
    monkeypatch.setattr(control_cli, "resolve_run_spec", lambda _spec: plan)
    monkeypatch.setattr(
        control_cli, "_display_plan", lambda _plan: calls.append("plan")
    )
    monkeypatch.setattr(
        control_cli,
        "command_context",
        lambda _plan: calls.append("context") or 0,
    )
    monkeypatch.setattr(
        control_cli,
        "_preflight",
        lambda _plan: calls.append("preflight") or {"passed": True},
    )
    for name in ("launch", "retry", "status", "watch", "stop"):
        monkeypatch.setattr(
            control_cli,
            f"command_{name}",
            lambda *_args, _name=name, **_kwargs: calls.append(_name) or 0,
        )
    monkeypatch.setattr(
        control_cli.paths,
        "workspace_topology",
        lambda: calls.append("workspace") or {"workspace_root": str(tmp_path)},
    )
    monkeypatch.setattr(
        control_cli,
        "load_builtin_provider_profiles",
        lambda: calls.append("providers") or {
            "test": {
                "protocol": "anthropic_messages",
                "model": "test",
                "admission": "text_tool_qualified",
                "usage_scope": "general_api",
                "supports_images": False,
            },
        },
    )
    monkeypatch.setattr(control_cli, "_run_root", lambda _run_id: run)
    monkeypatch.setattr(
        control_cli,
        "verify_policy_state",
        lambda _root: {"status": "valid", "errors": []},
    )
    monkeypatch.setattr(
        control_cli,
        "load_policy_state",
        lambda _root, _step: calls.append("state") or {"step": 0},
    )
    monkeypatch.setattr(
        control_cli,
        "write_context_audit",
        lambda _root: calls.append("context-audit") or {},
    )
    monkeypatch.setattr(
        control_cli,
        "write_runtime_metrics",
        lambda _root: calls.append("metrics") or {"schema": "metrics-test"},
    )
    monkeypatch.setattr(
        control_cli,
        "write_ownership_audit",
        lambda _root: calls.append("ownership") or {"schema": "ownership-test"},
    )
    monkeypatch.setattr(
        control_cli,
        "evaluate",
        lambda _root: calls.append("evaluate") or {"status": "valid"},
    )
    monkeypatch.setattr(
        control_cli,
        "seal",
        lambda _root, mode="operator": calls.append("seal") or {"status": "sealed"},
    )
    monkeypatch.setattr(
        control_cli,
        "export",
        lambda _root, destination: calls.append("export") or destination,
    )
    monkeypatch.setattr(
        control_cli,
        "verify_agent_trace",
        lambda _root: calls.append("trace-validate") or {"status": "valid"},
    )
    monkeypatch.setattr(
        control_cli,
        "export_sft",
        lambda _root, destination: calls.append("export-sft") or destination,
    )

    from ora.orchestrator import revalidation
    monkeypatch.setattr(
        revalidation, "revalidate_runs",
        lambda *_args, **_kwargs: calls.append("revalidate") or {"runs": []},
    )
    from ora.orchestrator import policy_lock
    monkeypatch.setattr(
        policy_lock, "command_lock",
        lambda args, root: calls.append("lock") or 0,
    )

    invocations = {
        "run": ["run", "--env", "synthetic"],
        "doctor": ["doctor", "--json"],
        "validate": ["validate", str(tmp_path / "spec.yaml")],
        "plan": ["plan", str(tmp_path / "spec.yaml")],
        "context": ["context", str(tmp_path / "spec.yaml")],
        "preflight": ["preflight", str(tmp_path / "spec.yaml")],
        "launch": ["launch", str(tmp_path / "spec.yaml"), "--yes"],
        "retry": ["retry", "run-test", "--yes"],
        "status": ["status"],
        "workspace": ["workspace"],
        "providers": ["providers"],
        "baselines": ["baselines"],
        "experiments": ["experiments"],
        "lock": ["lock", "--check"],
        "revalidate": ["revalidate", str(run), "--output-dir", str(tmp_path / "review")],
        "watch": ["watch", "run-test"],
        "stop": ["stop", "run-test"],
        "metadata": ["metadata", "run-test"],
        "state": ["state", "run-test", "--step", "0"],
        "metrics": ["metrics", "run-test"],
        "ownership": ["ownership", "run-test"],
        "evaluate": ["evaluate", "run-test"],
        "seal": ["seal", "run-test"],
        "export": ["export", "run-test", "--destination", str(tmp_path / "run.tar.gz")],
        "trace-validate": ["trace-validate", "run-test"],
        "export-sft": ["export-sft", "run-test", "--destination", str(tmp_path / "sft")],
    }
    assert tuple(invocations) == control_cli.PUBLIC_COMMANDS
    from ora.orchestrator import quickstart, doctor
    monkeypatch.setattr(quickstart, "command_run", lambda _args: 0)
    monkeypatch.setattr(doctor, "command_doctor", lambda _args: 0)
    for arguments in invocations.values():
        assert control_cli.main(arguments) == 0

    assert {
        "plan", "context", "preflight", "launch", "retry", "status",
        "workspace", "providers", "lock", "watch", "stop", "state", "context-audit", "metrics",
        "evaluate", "seal", "export", "trace-validate", "export-sft",
    } <= set(calls)


def test_runtime_packages_are_in_installed_distribution() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = project["tool"]["setuptools"]["packages"]["find"]["include"]
    for package in (
        "ora.environment.benchmarks",
        "ora.environment.controllers",
    ):
        assert (ROOT / package.replace(".", "/") / "__init__.py").is_file(), package
        assert any(fnmatch.fnmatchcase(package, pattern) for pattern in include), package
    assert project["tool"]["setuptools"]["package-data"]["viewer"] == [
        "templates/*.html", "static/*.css", "provider-pricing.json"
    ]


def test_documented_script_and_test_module_inventory_is_current() -> None:
    scripts_doc = (ROOT / "scripts/README.md").read_text(encoding="utf-8")
    tests_doc = (ROOT / "tests/README.md").read_text(encoding="utf-8")
    for path in sorted((ROOT / "scripts").iterdir()):
        if path.is_file() and path.name != "README.md":
            relative = path.relative_to(ROOT).as_posix()
            assert f"`{relative}`" in scripts_doc, relative
    for relative in (
        "experiments/runtime_v1/run.py",
        "ora/orchestrator/cli.py",
        "ora/environment/backends/robotwin/qualification.py",
        "viewer/app.py",
    ):
        assert f"`{relative}`" in scripts_doc, relative
    manifest_path = ROOT / "tests/test-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests").rglob("test_*.py")
        if path.is_file()
    }
    declared = {item["path"] for item in manifest["modules"]}
    assert declared == actual
    assert "[`test-manifest.json`](test-manifest.json)" in tests_doc


@pytest.mark.parametrize("driver,expected", [
    ("rlinf_v1", "c90951a0c799/2b910b5b5f53/c86fc3b8293b"),
    ("direct_libero_v1", "direct_libero_v1/2b910b5b5f53/c86fc3b8293b"),
])
def test_plan_display_names_the_libero_driver_it_resolved(driver, expected) -> None:
    # The direct driver resolves no RLinf checkout, so it carries no RLinf pin.
    runtime = {
        "kind": "libero_pro", "environment_driver": driver,
        "required_libero_pro_revision": "2b910b5b5f53016bef9907632f6f840f1ce2229c",
        "required_data_revision": "c86fc3b8293b5f6f373677018ff3e37f8391602",
    }
    if driver == "rlinf_v1":
        runtime["required_rlinf_revision"] = "c90951a0c799a750cb5294ed10587c61cc2af8bf"
    assert control_cli._runtime_revision(runtime) == expected
