"""Known launch failures are checked and explained before consuming an attempt."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.environment import factory
from ora.orchestrator import cli, defaults, preflight
from ora.agent.tools.grasp import (
    ContactGraspNetApiProposer, Sam3ApiSegmenter,
    CONTACT_GRASPNET_MODEL_IDENTITY, SAM3_MODEL_IDENTITY,
)
from ora.agent.tools.runtime import ManipulationToolRuntime
from test_native_baseline import native_spec

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("damage", [None, "revision", "hash", "missing", "outside"])
def test_preflight_and_factory_agree_on_sparse_controller_binding(damage):
    plan = defaults.expand_native_defaults(native_spec("robotwin"), root=ROOT)
    if damage == "revision":
        plan["runtime"]["required_revision"] = "0" * 40
    elif damage == "hash":
        plan["runtime"]["control_profile"]["sha256"] = "0" * 64
    elif damage == "missing":
        del plan["runtime"]["control_profile"]
    elif damage == "outside":
        plan["runtime"]["control_profile"]["path"] = "../foreign-profile.json"
    result = preflight._robotwin_control_binding(plan)
    assert result["passed"] is (damage is None)
    if damage is None:
        assert result["detail"]["required"] is True
    else:
        with pytest.raises((factory.EnvironmentFactoryError, ValueError, OSError)):
            factory.validate_robotwin_control_binding(
                plan["runtime"], action_codec=plan["policy"]["action_codec"],
                project_root=ROOT,
            )
        if damage == "revision":
            assert "source or embodiment" in result["detail"]


def test_runtime_preflight_includes_controller_check_without_starting_worker(monkeypatch):
    plan = defaults.expand_native_defaults(native_spec("robotwin"), root=ROOT)
    plan["runtime"]["required_revision"] = "0" * 40
    monkeypatch.delenv(plan["runtime"]["root_env"], raising=False)
    monkeypatch.delenv(plan["runtime"]["python_env"], raising=False)
    monkeypatch.setattr(preflight, "_execution_resource", lambda _: {"check": "fixture", "passed": True})
    checks = preflight._runtime(plan)
    binding = next(c for c in checks if c["check"] == "robotwin_control_binding")
    assert binding["passed"] is False


def _service(surface, kind):
    return {
        "schema": ("ora.segmentation-service-binding.v1" if kind == "sam"
                   else "ora.grasp-proposal-service-binding.v1"),
        "tool_surface": surface,
        "endpoint": "http://127.0.0.1:" + ("18114" if kind == "sam" else "18115"),
        "instance_id": kind, "pid": 123, "started_unix_s": 1.0,
        "adapter_sha256": "a" * 64,
        "model": dict(SAM3_MODEL_IDENTITY if kind == "sam" else CONTACT_GRASPNET_MODEL_IDENTITY),
        "load_count": 1, "warmup_complete": True,
        "qualification": {"status": "passed", "sha256": "b" * 64, "created_unix_s": 1.0},
    }


def _perception_runtime(tmp_path, surface, sam, grasp):
    return ManipulationToolRuntime(
        control_root=tmp_path / "control",
        manual_path=ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md",
        event_path=tmp_path / "events.jsonl",
        workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
        state_root=None, tool_surface_profile=surface, controlled_end_effector="right",
        prerequisite_binding={
            "schema": "ora.model-service-set-binding.v1",
            "tool_surface": surface,
            "services": {"segmentation": sam, "grasp_proposals": grasp},
        },
    )


@pytest.mark.parametrize("surface", ["libero_placement_direct_overhead_v1", "robotwin_toolchain_b2_closed_noservo_v2"])
@pytest.mark.parametrize("override", [False, True])
def test_preflight_and_real_runtime_clients_use_identical_bound_endpoints(tmp_path, monkeypatch, surface, override):
    sam, grasp = _service(surface, "sam"), _service(surface, "grasp")
    for variable, binding in [("ORA_SEGMENTATION_URL", sam), ("ORA_GRASP_URL", grasp)]:
        monkeypatch.delenv(variable, raising=False)
        if override:
            monkeypatch.setenv(variable, binding["endpoint"])
    seen = []

    def ready(client):
        seen.append(client.endpoint)
        return copy.deepcopy(client.expected_binding)

    # Keep real constructor and exact binding validation; replace only network I/O.
    monkeypatch.setattr(Sam3ApiSegmenter, "require_ready", ready)
    monkeypatch.setattr(ContactGraspNetApiProposer, "require_ready", ready)
    assert preflight._verify_resident_service(sam, surface)["endpoint"] == sam["endpoint"]
    assert preflight._verify_resident_service(grasp, surface)["endpoint"] == grasp["endpoint"]
    runtime = _perception_runtime(tmp_path, surface, sam, grasp)
    contact = getattr(runtime.grasp_backend, "contact_backend", runtime.grasp_backend)
    try:
        assert runtime.segmentation_backend.endpoint == sam["endpoint"]
        assert contact.endpoint == grasp["endpoint"]
        assert runtime.segmentation_backend.expected_binding == sam
        assert contact.expected_binding == grasp
        assert seen == [sam["endpoint"], grasp["endpoint"]] * 2
    finally:
        runtime.segmentation_backend.close()
        contact.close()


def test_mismatched_explicit_endpoint_is_rejected_by_both_consumers(tmp_path, monkeypatch):
    surface = "libero_placement_direct_overhead_v1"
    sam, grasp = _service(surface, "sam"), _service(surface, "grasp")
    monkeypatch.setenv("ORA_SEGMENTATION_URL", "http://127.0.0.1:19999")
    with pytest.raises(ValueError, match="binding is invalid"):
        preflight._verify_resident_service(sam, surface)
    with pytest.raises(ValueError, match="binding is invalid"):
        _perception_runtime(tmp_path, surface, sam, grasp)


@pytest.mark.parametrize("corrupt", [False, True])
def test_duplicate_admission_reports_original_owner_and_cli_failure(tmp_path, monkeypatch, capsys, corrupt):
    plan = {"experiment": "fixture", "config": "cell", "plan_id": "fixture-plan", "plan_sha256": "a" * 64,
            "source": {"commit": "b" * 40}, "run_id": "original-run",
            "launch_admission": {"mode": "single_attempt_cell", "cohort_id": "cohort", "cell_id": "cell"}}
    monkeypatch.setattr(cli.paths, "workspace_from_plan", lambda _: tmp_path)
    claim = cli._claim_launch_admission(plan, "original-run")
    path = Path(claim["path"])
    original = path.read_bytes()
    data = json.loads(original)
    if corrupt:
        path.write_text("broken")
        original = path.read_bytes()
    monkeypatch.setattr(cli, "command_launch", lambda _: cli._claim_launch_admission(plan))
    assert cli.main(["launch", "unused.yaml", "--yes"]) == 2
    message = capsys.readouterr().err
    assert "single-attempt cohort cell was already claimed: cohort/cell" in message
    if corrupt:
        assert "metadata unavailable" in message
    else:
        assert f"run_id={data['run_id']}" in message
        assert f"created_unix_s={data['created_unix_s']}" in message
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.tmp"))

