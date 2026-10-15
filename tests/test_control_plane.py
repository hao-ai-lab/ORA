"""Control-plane, output ownership, experiment and viewer contracts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from ora.orchestrator import paths
from ora.orchestrator import cli as control_cli
from ora.orchestrator.cli import build_parser
from ora.orchestrator.preflight import (
    SERVICE_SET_BINDING_SCHEMA,
    _libero_controller_contract_check,
    _offscreen_graphics_runtime,
    _provider_marker,
    _resident_service_prerequisite,
    run_preflight,
)
from ora.environment import graphics_probe
from ora.environment.backends.libero_pro.session import (
    libero_osc_action_contract,
)
from ora.environment.replay_worker import (
    ReplayWorkerError,
    _FrameWriter,
    _apply_robotwin_control_target,
    _capture_steps,
    _needs_final_capture,
    _uniform_capture_steps,
    _libero_qualified_controller_binding,
    _recorded_control_step_count,
    _robotwin_replay_needs_planner,
    _robotwin_replay_route,
)
from ora.orchestrator.spec import RunSpecError, resolve_run_spec
from ora.orchestrator.context_manifest import context_placeholder
from experiments.runtime_v1 import run as experiment
from viewer import app as viewer
from viewer.libero_pro_catalog import (
    CATALOG_SCHEMA as LIBERO_PRO_CATALOG_SCHEMA,
    LIBERO_PRO_CODE_REVISION,
    LIBERO_PRO_DATA_REVISION,
    MIGRATED_FROM_COMMIT as LIBERO_MIGRATED_FROM_COMMIT,
    ROBUSTNESS_CASES as LIBERO_ROBUSTNESS_CASES,
    TASKS as LIBERO_PRO_TASKS,
)
from viewer.libero_previews import (
    EXPECTED_PREVIEWS as LIBERO_EXPECTED_PREVIEWS,
    PREVIEW_SCHEMA as LIBERO_PREVIEW_SCHEMA,
    RENDER_CONFIG as LIBERO_RENDER_CONFIG,
    load_preview_manifest,
    preview_descriptors,
)
from viewer.robotwin_catalog import (
    CATALOG_SCHEMA,
    MIGRATED_FROM_COMMIT,
    ROBOTWIN_REVISION,
    TASKS,
)
from viewer.registry import discover_run_roots, register_run, visible_runs
from viewer.trajectory import (
    ANALYSIS_SCHEMA,
    _controller_replay_source,
    _continuous_execution_source,
    _load_run_analysis,
    _render_continuous_execution_frame,
    _render_trajectory_frame,
    compile_raw_trajectory,
    compile_trajectory,
    compile_trajectory_video,
    raw_trajectory_event,
)
from viewer.replay import (
    ViewerReplayError,
    ensure_controller_replay,
)


ROOT = Path(__file__).resolve().parents[1]


def init_project(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", tmp_path], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", tmp_path, "config", "user.name", "Test"], check=True)
    (tmp_path / "prompt.md").write_text("prompt v1")
    (tmp_path / "manual.md").write_text("## overview\n\nmanual v1\n")
    (tmp_path / "agents.md").write_text("agent rules v1")
    subprocess.run(["git", "-C", tmp_path, "add", "."], check=True)
    subprocess.run(["git", "-C", tmp_path, "commit", "-qm", "initial"], check=True)
    return tmp_path


def spec() -> dict:
    return {
        "schema": "ora.run-spec.v1",
        "experiment": "runtime-v1", "config": "minimal", "milestone": "M0",
        "output_workspace": "outputs/runtime-v1",
        "task": "synthetic", "task_instruction": "Move once.", "seed": 0,
        "evidence_tier": "diagnostic",
        "policy": {"kind": "scripted", "model": "scripted", "provider": "trusted", "reasoning_effort": "none", "actions": [{"x": 1}]},
        "runtime": {"kind": "synthetic", "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}},
        "budgets": {"max_robot_decisions": 1, "max_model_calls": 1, "max_sim_steps": 1, "wall_time_s": 10},
        "policy_context": {
            "codex_project": ["agents.md"],
            "episode_prefix": ["prompt.md"],
            "tool_manual": ["manual.md"],
        },
        "qualification": {"required": False},
    }


def test_plan_is_deterministic_and_resource_change_rekeys(tmp_path: Path) -> None:
    root = init_project(tmp_path)
    one = resolve_run_spec(spec(), root=root)
    two = resolve_run_spec(spec(), root=root)
    assert one["runtime"]["transport"] == "in_process_direct"
    assert one["runtime"]["environment_signal_policy"] == "lifecycle_only"
    assert one["plan_sha256"] == two["plan_sha256"]
    (root / "prompt.md").write_text("prompt v2")
    changed = resolve_run_spec(spec(), root=root)
    assert changed["plan_sha256"] != one["plan_sha256"]
    assert changed["policy_context"]["composition_sha256"] != one["policy_context"]["composition_sha256"]
    assert changed["source"]["dirty"] is True
    invalid = spec()
    invalid["output_workspace"] = "outputs/shared"
    with pytest.raises(RunSpecError, match="output_workspace"):
        resolve_run_spec(invalid, root=root)
    invalid_transport = spec()
    invalid_transport["runtime"]["transport"] = "remote_rpc"
    with pytest.raises(RunSpecError, match="runtime.transport"):
        resolve_run_spec(invalid_transport, root=root)
    robotwin = spec()
    robotwin["runtime"] = {
        "kind": "robotwin", "transport": "local_rpc",
        "root_env": "ROBOTWIN_ROOT", "python_env": "ROBOTWIN_PYTHON",
        "required_revision": "revision", "task_config": "demo_clean",
        "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
    }
    assert resolve_run_spec(robotwin, root=root)["runtime"]["transport"] == "local_rpc"
    coordinated = spec()
    coordinated["task"] = "adjust_bottle"
    coordinated["runtime"] = {
        **robotwin["runtime"],
        "environment_signal_policy": "task_recoverability_v1",
    }
    assert resolve_run_spec(coordinated, root=root)["runtime"][
        "environment_signal_policy"
    ] == "task_recoverability_v1"
    coordinated["task"] = "open_laptop"
    with pytest.raises(RunSpecError, match="defined only for adjust_bottle"):
        resolve_run_spec(coordinated, root=root)
    robotwin["runtime"].pop("python_env")
    with pytest.raises(RunSpecError, match="runtime.python_env"):
        resolve_run_spec(robotwin, root=root)
    libero = spec()
    libero["runtime"] = {
        "kind": "libero_pro", "transport": "local_rpc",
        "python_env": "LIBERO_PRO_PYTHON",
        "libero_pro_root_env": "LIBERO_PRO_ROOT", "data_root_env": "LIBERO_PRO_DATA_ROOT",
        "required_libero_pro_revision": "libero-revision",
        "required_data_revision": "data-revision", "suite": "libero_object_swap",
        "task_id": 2, "expected_task_language": "place the object",
        "main_camera": "agentview",
        "workspace_bounds_m": {"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.1, 1.3]},
    }
    resolved_libero = resolve_run_spec(libero, root=root)
    assert resolved_libero["runtime"]["task_id"] == 2
    assert context_placeholder(resolved_libero)["main_camera"] == "agentview"
    robustness = spec()
    robustness["runtime"] = {
        **libero["runtime"],
        "suite": "libero_object",
        "required_libero_pro_revision": LIBERO_PRO_CODE_REVISION,
        "robustness_case": "view_occlusion",
    }
    assert resolve_run_spec(robustness, root=root)["runtime"][
        "robustness_case"
    ] == "view_occlusion"
    robustness["runtime"]["required_libero_pro_revision"] = "unpaired"
    with pytest.raises(RunSpecError, match="paired LIBERO-Pro revision"):
        resolve_run_spec(robustness, root=root)
    slotted_libero = json.loads(json.dumps(libero))
    slotted_libero["runtime"].update({
        "required_host": "test-host",
        "resource_profile": "libero-gpu1",
    })
    slotted = resolve_run_spec(slotted_libero, root=root)
    assert "resource_profile" not in slotted["runtime"]
    assert slotted["runtime"]["resource_binding"]["slot_id"] == "gpu1"
    assert slotted["plan_sha256"] != resolved_libero["plan_sha256"]
    invalid_binding = json.loads(json.dumps(libero))
    invalid_binding["runtime"]["resource_binding"] = {}
    with pytest.raises(RunSpecError, match="resolver-owned"):
        resolve_run_spec(invalid_binding, root=root)
    libero["runtime"]["task_id"] = 10
    with pytest.raises(RunSpecError, match="task_id"):
        resolve_run_spec(libero, root=root)


def _graphics_stub(tmp_path: Path, body: str) -> Path:
    """Write an interpreter stand-in that emits one probe line."""

    stub = tmp_path / "python-stub"
    stub.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    stub.chmod(0o755)
    return stub


def _service_set_plan(*, drifted: str | None = None) -> dict:
    """A v2 set binding holding both resident services on one tool surface."""

    def child(schema: str, endpoint: str) -> dict:
        return {
            "schema": schema,
            "tool_surface": "unified_manip_v1",
            "endpoint": endpoint,
            "instance_id": "drifted" if schema == drifted else "bound",
        }

    return {
        "policy": {"tool_surface": {"profile": "unified_manip_v1"}},
        "prerequisite": {
            "schema": SERVICE_SET_BINDING_SCHEMA,
            "capability_scope": [
                "resident_sam3_text_grounding_v1", "resident_contact_graspnet_v1",
            ],
            "services": {
                "segmentation": child(
                    "ora.segmentation-service-binding.v1",
                    "http://127.0.0.1:8114",
                ),
                "grasp_proposals": child(
                    "ora.grasp-proposal-service-binding.v1",
                    "http://127.0.0.1:8115",
                ),
            },
        },
    }


def _install_fake_services(monkeypatch: pytest.MonkeyPatch, seen: list) -> None:
    from ora.agent.tools import grasp

    def make(label: str):
        class Fake:
            def __init__(self, **kwargs):
                self.binding = kwargs["expected_binding"]
                seen.append(label)

            def require_ready(self):
                if self.binding.get("instance_id") == "drifted":
                    raise ValueError(f"{label} service differs from bound instance_id")
                return {
                    "instance_id": "bound", "pid": 4242,
                    "adapter_sha256": "a" * 64, "model": label,
                    "warmup_complete": True,
                }

            def close(self):
                pass

        return Fake

    monkeypatch.setattr(grasp, "Sam3ApiSegmenter", make("SAM3"))
    monkeypatch.setattr(grasp, "ContactGraspNetApiProposer", make("CGN"))


def test_runtime_v1_is_first_class_experiment_package(tmp_path: Path) -> None:
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) >= {
        "validate", "plan", "context", "preflight", "launch", "retry",
        "status", "providers", "watch", "stop", "metadata", "state", "metrics",
        "evaluate", "seal", "export",
    }
    experiment_choices = experiment.build_parser()._subparsers._group_actions[0].choices
    standard_choices = {
        "configs", "explain", "qualify", "context", "plan", "preflight",
        "launch", "status", "watch", "stop", "metadata", "metrics",
        "evaluate", "seal", "export",
    }
    assert set(experiment_choices) == standard_choices | {"control-audit"}
    definition = experiment._definition()
    assert definition["schema"] == "ora.runtime-experiment.v1"
    assert list(definition["milestones"]) == ["M0", "M1", "M2", "M3", "M4", "M5"]
    assert definition["milestones"]["M5"]["seeds"] == [100, 101, 103, 106, 111]
    assert definition["output_workspace"] == "outputs/harness-runtime-v1"
    m0_spec, _ = experiment.compose("M0", policy="scripted", seed=0, preview=True)
    m1_spec, _ = experiment.compose("M1", policy="scripted", seed=100, preview=True)
    assert m0_spec["runtime"]["transport"] == "in_process_direct"
    assert m1_spec["runtime"]["transport"] == "local_rpc"
    package = ROOT / "experiments" / "runtime_v1"
    for required in ("README.md", "experiment.yaml", "run.py", "summary.md"):
        assert (package / required).is_file(), f"runtime_v1/{required}"
    assert (ROOT / "examples/hosts/example-host.yaml").is_file()
    assert (ROOT / "scripts/ora").is_file()
    assert (ROOT / "scripts/open-viewer").is_file()


def test_supervisor_pipeline_writes_runtime_metrics_before_releasing_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"plan_sha256": "plan-1"}))
    lease_root = tmp_path / "leases"
    monkeypatch.setattr(control_cli, "LEASE_ROOT", lease_root)
    monkeypatch.setattr(
        control_cli, "LEASE_LOCK_PATH", lease_root / ".registry.lock",
    )
    monkeypatch.setattr(control_cli, "LEASE_PATH", tmp_path / "active-run.json")
    monkeypatch.setattr(control_cli, "_run_root", lambda _run_id: run)
    plan = {
        "plan_sha256": "plan-1",
        "runtime": {"kind": "synthetic"},
    }
    lease = control_cli._create_lease(plan, "run-1", os.getpid())
    lease = control_cli._assign_lease(lease, os.getpid())

    def fake_episode(*, run_root: Path, plan: dict, run_id: str) -> dict:
        result = {
            "run_id": run_id, "plan_sha256": plan["plan_sha256"],
            "state": "completed", "started_unix_s": 10.0,
            "finished_unix_s": 12.0, "wall_duration_s": 1.8,
        }
        (run_root / "result.json").write_text(json.dumps(result))
        (run_root / "events.jsonl").write_text(
            json.dumps({"type": "run_started", "time": 10.0}) + "\n"
        )
        return result

    monkeypatch.setattr(control_cli, "run_episode", fake_episode)
    assert control_cli.command_supervise(
        "run-1", str(lease["claim_token"]),
    ) == 0
    metrics = json.loads((run / "runtime-metrics.json").read_text())
    assert metrics["run_id"] == "run-1"
    assert metrics["latency"]["episode_wall_s"] == 1.8
    assert (run / "context-audit.json").is_file()
    assert not Path(str(lease["lease_path"])).exists()
    viewer_args = viewer.build_parser().parse_args([])
    assert (viewer_args.host, viewer_args.port) == ("127.0.0.1", 5012)
    assert viewer_args.open_browser is False
    assert not (ROOT / "scripts/run-exp00").exists()
    for milestone in range(6):
        assert not (ROOT / f"scripts/run-harness-m{milestone}").exists()


def test_libero_preview_cache_is_complete_and_safely_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    code_root = tmp_path / "code"
    data_root = tmp_path / "data"
    for task in LIBERO_PRO_TASKS:
        base_suite = task.base_suite
        base_bddl = (
            code_root / "libero/libero/bddl_files" / base_suite
            / f"{task.name}.bddl"
        )
        init_root = (
            data_root / "init_files"
            if task.pro_anchor
            else code_root / "libero/libero/init_files"
        )
        base_init = init_root / base_suite / f"{task.name}.pruned_init"
        base_bddl.parent.mkdir(parents=True, exist_ok=True)
        base_init.parent.mkdir(parents=True, exist_ok=True)
        base_bddl.write_text(
            f"(:language {task.canonical_instruction})\n", encoding="utf-8",
        )
        base_init.write_bytes(b"init")
        for variant in task.core_variants:
            bddl = data_root / "bddl_files" / variant.suite / f"{task.name}.bddl"
            init = data_root / "init_files" / variant.suite / f"{task.name}.pruned_init"
            bddl.parent.mkdir(parents=True, exist_ok=True)
            init.parent.mkdir(parents=True, exist_ok=True)
            bddl.write_text(
                f"(:language {variant.instruction})\n", encoding="utf-8",
            )
            init.write_bytes(b"init")
        for case in LIBERO_ROBUSTNESS_CASES if task.pro_anchor else ():
            bddl = (
                data_root / "bddl_files" / case["id"] / "bddl" / base_suite
                / f"{task.name}.bddl"
            )
            bddl.parent.mkdir(parents=True, exist_ok=True)
            bddl.write_text(
                f"(:language {task.canonical_instruction})\n", encoding="utf-8",
            )
    descriptors = preview_descriptors(
        libero_pro_root=code_root, data_root=data_root,
    )
    assert len(descriptors) == LIBERO_EXPECTED_PREVIEWS == 570
    assert sum(item["layer"] == "base" for item in descriptors) == 130
    assert sum(item["layer"] == "core" for item in descriptors) == 160
    assert sum(item["layer"] == "robustness" for item in descriptors) == 280
    assert next(item for item in descriptors if item["key"] == (
        "base:libero_10:task-00"
    ))["task_name"].startswith("LIVING_ROOM_SCENE2_")
    assert next(item for item in descriptors if item["key"] == (
        "base:libero_90:task-89"
    ))["task_name"].startswith("STUDY_SCENE4_")

    cache = tmp_path / "cache"
    records = []
    for index, descriptor in enumerate(descriptors):
        relative = (
            f"sets/test/{descriptor['layer']}/{descriptor['suite']}/"
            f"task-{descriptor['task_id']:02d}-seed-0-{index:03d}.png"
        )
        image = cache / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"\x89PNG\r\nviewer-test")
        records.append({
            "key": descriptor["key"], "image": relative,
            "layer": descriptor["layer"], "suite": descriptor["suite"],
            "task_id": descriptor["task_id"],
        })
    (cache / "manifest.json").write_text(json.dumps({
        "schema": LIBERO_PREVIEW_SCHEMA,
        "source": {
            "libero_pro_code_revision": "2b910b5b5f53016bef9907632f6f840f1ce2229c",
            "libero_pro_data_revision": "c86fc3b8293185a6f373677018ff3e37f8391602",
        },
        "render": LIBERO_RENDER_CONFIG,
        "records": records,
    }), encoding="utf-8")
    status = load_preview_manifest(cache)
    assert status["ready"] is True
    assert status["available"] == status["expected"] == 570
    monkeypatch.setattr(viewer, "load_preview_manifest", lambda: status)

    client = viewer.app.test_client()
    response = client.get("/gallery/libero-pro/object/0")
    assert response.status_code == 200
    assert b"570 / 570 seed-0 previews loaded" in response.data
    assert b"seed 0 \xc2\xb7 zero action" in response.data
    record = status["index"]["core:libero_object_swap:task-00"]
    image_response = client.get(f"/gallery/libero-pro/preview/{record['image']}")
    assert image_response.status_code == 200
    assert image_response.mimetype == "image/png"
    assert image_response.data == b"\x89PNG\r\nviewer-test"
    assert client.get("/gallery/libero-pro/preview/manifest.json").status_code == 404
    api = client.get("/api/libero-pro/previews").get_json()
    assert api["ready"] is True and api["available"] == 570

    (cache / records[-1]["image"]).unlink()
    incomplete = load_preview_manifest(cache)
    assert incomplete["ready"] is False
    assert incomplete["available"] == 569


def test_runtime_experiment_definition_has_exact_model_and_single_active_runtime() -> None:
    value = yaml.safe_load((ROOT / "experiments/runtime_v1/experiment.yaml").read_text())
    assert value["provider_profile"] == "deepseek_flash_openai_low_16k_v1"
    assert "model" not in value
    # The execution host is the marker's registered host, never a default.
    assert "required_host" not in value["runtime_defaults"]
    assert value["runtime_defaults"]["transport"] == "local_rpc"
    assert value["runtime_defaults"]["python_env"] == "ROBOTWIN_PYTHON"


def test_qualification_processes_run_from_exact_robotwin_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    robotwin_root = tmp_path / "RoboTwin"
    robotwin_root.mkdir()
    monkeypatch.setenv("ROBOTWIN_ROOT", str(robotwin_root))
    monkeypatch.setenv("ROBOTWIN_PYTHON", "/qualified/python")
    monkeypatch.setattr(experiment, "QUALIFICATION_ROOT", tmp_path / "runtime")
    observed: list[Path] = []

    def stop_after_spawn(command, **kwargs):
        observed.append(Path(kwargs["cwd"]))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(experiment.subprocess, "run", stop_after_spawn)
    assert experiment.qualify("M2") == 1
    assert observed == [robotwin_root.resolve()]


def test_runtime_milestone_gate_is_bound_to_current_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(experiment, "GATES_ROOT", tmp_path)
    monkeypatch.setattr(experiment, "_current_source_commit", lambda: "new-commit")
    (tmp_path / "M0-scripted.json").write_text(json.dumps({
        "run_id": "old-run", "source_commit": "old-commit",
    }))
    with pytest.raises(RuntimeError, match="stale for current source"):
        experiment._check_gate("M0", "native_api", None)


def test_runtime_context_preview_does_not_require_execution_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(experiment, "SPECS_ROOT", tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("context preview must not inspect execution gates")

    monkeypatch.setattr(experiment, "_check_gate", forbidden)
    monkeypatch.setattr(experiment, "_qualification_path", forbidden)
    spec, path = experiment.compose("M5", policy="native_api", seed=100, preview=True)
    assert path.is_file()
    assert spec["qualification"] == {"required": False}
    assert spec["policy"]["kind"] == "native_api"


def test_context_audit_distinguishes_native_visibility_from_manifest_only(
    tmp_path: Path,
) -> None:
    from ora.orchestrator.context_audit import policy_delivery_inventory

    def source(path: str) -> dict[str, object]:
        return {"path": path, "bytes": 1, "sha256": "a" * 64}

    plan = {
        "policy": {"kind": "native_api"},
        "policy_context": {
            "schema": "ora.policy-context.v3",
            "harness_system": [source("policy/system.md")],
            "project_instructions": [source("policy/project.md")],
            "episode_task": [source("policy/task.md")],
            "tool_manual": [source("policy/tool-manual.md")],
        },
        "task_instruction": "Do the task.",
    }
    inventory = policy_delivery_inventory(plan)
    by_channel = {
        item["channel"]: item for item in inventory["policy_layers"]
        if item["channel"] != "task"
    }
    assert by_channel["harness_system"]["provider_role"] == "system"
    assert by_channel["project_instructions"]["provider_role"] == "user"
    assert by_channel["episode_task"]["provider_visible"] is True
    assert by_channel["tool_manual"]["provider_visible"] is False
    assert by_channel["tool_manual"]["delivery"] == "manifest_only"
    assert by_channel["tool_manual"]["injection_count"] == 0
    assert inventory["tool_schema"]["provider_visible"] is True
    assert inventory["tool_schema"]["delivery_timing"] == "every_model_call"


def test_snapshot_launch_gate_copies_the_verified_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    gate_rel = "outputs/exp-x/gates/motion-gate.json"
    gate_path = workspace / gate_rel
    gate_path.parent.mkdir(parents=True)
    gate_bytes = b'{"status": "qualified"}'
    gate_path.write_bytes(gate_bytes)
    monkeypatch.setattr(
        control_cli.paths, "resolve_workspace_root", lambda root: workspace,
    )
    plan = {"launch_admission": {"mode": "required", "gate": {
        "path": gate_rel,
        "sha256": hashlib.sha256(gate_bytes).hexdigest(),
        "bytes": len(gate_bytes),
    }}}
    run_root = tmp_path / "run"
    run_root.mkdir()
    control_cli._snapshot_launch_gate(plan, run_root)
    snapshot = run_root / "qualification" / "launch-gate.json"
    assert snapshot.read_bytes() == gate_bytes
    manifest = json.loads(
        (run_root / "qualification" / "launch-gate-snapshot.json").read_text()
    )
    assert manifest["schema"] == (
        "ora.launch-gate-snapshot.v1"
    )
    assert manifest["gate"]["path"] == gate_rel

    # A gate that drifts between verification and staging fails closed.
    gate_path.write_bytes(b'{"status": "changed"}')
    other_run = tmp_path / "run2"
    other_run.mkdir()
    with pytest.raises(RuntimeError, match="changed while staging"):
        control_cli._snapshot_launch_gate(plan, other_run)

    # Disabled or absent admissions stage nothing.
    third_run = tmp_path / "run3"
    third_run.mkdir()
    control_cli._snapshot_launch_gate({"launch_admission": {"mode": "disabled"}}, third_run)
    control_cli._snapshot_launch_gate({}, third_run)
    assert not (third_run / "qualification").exists()


def test_seal_at_landing_seals_once_and_records_refusals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    calls: list[str] = []

    def fake_seal(root: Path, *, mode: str) -> dict[str, str]:
        calls.append(mode)
        (root / "seal.json").write_text("{}")
        return {"mode": mode}

    monkeypatch.setattr(control_cli, "seal", fake_seal)
    control_cli._seal_at_landing(run_root)
    assert calls == ["landing"]
    assert not (run_root / "seal-refusal.json").exists()

    # Already sealed: no second attempt.
    control_cli._seal_at_landing(run_root)
    assert calls == ["landing"]

    refused_root = tmp_path / "run2"
    refused_root.mkdir()

    def refusing_seal(root: Path, *, mode: str) -> dict[str, str]:
        raise control_cli.EvaluationError("invalid runs cannot be sealed as evidence")

    monkeypatch.setattr(control_cli, "seal", refusing_seal)
    control_cli._seal_at_landing(refused_root)
    refusal = json.loads((refused_root / "seal-refusal.json").read_text())
    assert refusal["schema"] == "ora.seal-refusal.v1"
    assert refusal["stage"] == "landing"
    assert "cannot be sealed" in refusal["reason"]


def test_supervisor_waits_while_the_launcher_has_not_yet_named_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # The launcher assigns supervisor_pid only after spawning the supervisor,
    # so a supervisor can read its lease while that field is still null.
    import threading
    run = tmp_path / "run"
    run.mkdir()
    (run / "plan.json").write_text(json.dumps({"plan_sha256": "plan-1"}))
    lease_root = tmp_path / "leases"
    monkeypatch.setattr(control_cli, "LEASE_ROOT", lease_root)
    monkeypatch.setattr(control_cli, "LEASE_LOCK_PATH", lease_root / ".registry.lock")
    monkeypatch.setattr(control_cli, "LEASE_PATH", tmp_path / "active-run.json")
    monkeypatch.setattr(control_cli, "_run_root", lambda _run_id: run)
    plan = {"plan_sha256": "plan-1", "runtime": {"kind": "synthetic"}}
    lease = control_cli._create_lease(plan, "run-1", os.getpid())
    assert lease["supervisor_pid"] is None
    timer = threading.Timer(0.3, control_cli._assign_lease, args=(lease, os.getpid()))
    timer.start()

    def fake_episode(*, run_root: Path, plan: dict, run_id: str) -> dict:
        result = {"run_id": run_id, "plan_sha256": plan["plan_sha256"], "state": "completed",
                  "started_unix_s": 10.0, "finished_unix_s": 12.0, "wall_duration_s": 1.8}
        (run_root / "result.json").write_text(json.dumps(result))
        (run_root / "events.jsonl").write_text(json.dumps({"type": "run_started", "time": 10.0}) + "\n")
        return result

    monkeypatch.setattr(control_cli, "run_episode", fake_episode)
    try:
        assert control_cli.command_supervise("run-1", str(lease["claim_token"])) == 0
    finally:
        timer.join()


def test_runtime_ladder_native_previews_resolve_on_every_milestone(monkeypatch, tmp_path):
    monkeypatch.setattr(experiment, "SPECS_ROOT", tmp_path)
    for milestone, item in experiment._definition()["milestones"].items():
        spec, _ = experiment.compose(milestone, policy="native_api", seed=item["seeds"][0], preview=True)
        plan = resolve_run_spec(spec)
        assert plan["policy"]["kind"] == "native_api"
        assert plan["policy"]["provider_profile"] == "deepseek_flash_openai_low_16k_v1"
        if milestone != "M0":
            assert plan["runtime"]["required_revision"] == "cf4872cc90538fd4bca761c8db912d8f65e49f2b"
    assert experiment._definition()["milestones"]["M5"]["qualification"] == "local"
