"""Runtime v1 state contract regressions."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pytest
from ora.orchestrator.evaluation import EvaluationError, assess_learning_eligibility, evaluate, export, inventory, seal
from ora.environment.workspace_profiles import resolve_workspace_profile
from ora.orchestrator.spec import file_sha256
from ora.orchestrator.coordinator import _reconciled_provider_model_calls, run_episode
from ora.environment import ENVIRONMENT_STATUS_SCHEMA, SyntheticSession, validate_environment_status
from ora.agent.tools.robot import RobotToolRuntime, ToolError
from ora.agent.evidence.policy_state import PolicyStateStore, load_policy_state, verify_policy_state
from ora.environment.backends.libero_pro.session import libero_osc_action_contract
from tests.orchestrator.test_runtime_v1 import (
    BOUNDS,
    MANUAL,
    synthetic_plan,
)


def test_policy_state_sanitizes_staged_phase_ledger_in_embedded_receipt(
    tmp_path: Path,
) -> None:
    """EXP06A closing repair A1: staged-approach receipts must stay readable.

    The RoboTwin staged receipt names its controller-tracking phases
    ``global_precontact``/``local_contact``; both key names trip the substring
    privilege scan, which used to make every snapshot embedding such a receipt
    fail ``load_policy_state`` (surfaced by ``grasp.verify_retention`` as
    ``artifact_hash_mismatch`` in 10/10 EXP06A cohort calls).
    """

    observation_root = tmp_path / "policy-observations"
    session = SyntheticSession(tmp_path / "native", "synthetic_tool_boundary", 0)
    session.start()
    store = PolicyStateStore(tmp_path, "phase-ledger-run", "phase-plan")
    first, _ = session.observe(observation_root)
    store.publish(
        observation=first,
        observation_root=observation_root / first["observation_id"],
        cause="reset", task_instruction="Move safely once.",
        previous_public_action=None,
        remaining_budget={"robot_decisions": 1, "model_calls": 2},
    )
    action = {
        "schema": "ora-action.v3", "observation_id": first["observation_id"],
        "action_id": "action-0", "action": "move_pose", "bound_observation_sha256": "bound",
        "arguments": {"arm": "left", "position_m": [0, 0, 1.02],
                      "orientation_xyzw": [0, 0, 0, 1]},
    }
    session.execute(action)
    second, _ = session.observe(observation_root)
    staged_receipt = {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "execution_status": "reached",
        "controller_tracking": {
            "schema": "ora.motion-execution-pipeline.v1",
            "mode": "global_precontact_then_local",
            "completed": True,
            "abort": None,
            "sampled_steps": 12,
            "phases": {
                "global_precontact": {"sampled_steps": 8},
                "local_contact": {"sampled_steps": 4},
                "local_settle": {"sampled_steps": 0},
            },
            "arms": {},
        },
    }
    snapshot = store.publish(
        observation=second,
        observation_root=observation_root / second["observation_id"],
        cause="after_action", task_instruction="Move safely once.",
        previous_public_action=action,
        previous_execution_receipt=staged_receipt,
        remaining_budget={"robot_decisions": 0, "model_calls": 2},
    )
    embedded = snapshot["previous_execution_receipt"]["controller_tracking"]
    assert embedded["phases"] == {"local_settle": {"sampled_steps": 0}}
    assert embedded["phases_redacted"] == ["global_precontact", "local_contact"]
    assert embedded["mode"] == "global_precontact_then_local"
    verification = verify_policy_state(tmp_path)
    assert verification["status"] == "valid"
    loaded = load_policy_state(tmp_path, 1)
    assert loaded["previous_execution_receipt"]["controller_tracking"][
        "phases_redacted"
    ] == ["global_precontact", "local_contact"]


def test_policy_state_history_hash_chain_and_tamper_detection(tmp_path: Path) -> None:
    observation_root = tmp_path / "policy-observations"
    session = SyntheticSession(tmp_path / "native", "synthetic_tool_boundary", 0)
    session.start()
    store = PolicyStateStore(tmp_path, "history-run", "history-plan")
    first, _ = session.observe(observation_root)
    first_state = store.publish(
        observation=first, observation_root=observation_root / first["observation_id"],
        cause="reset", task_instruction="Move safely once.",
        previous_public_action=None,
        remaining_budget={"robot_decisions": 1, "model_calls": 2},
    )
    action = {
        "schema": "ora-action.v3", "observation_id": first["observation_id"],
        "action_id": "action-0", "action": "move_pose", "bound_observation_sha256": "bound",
        "arguments": {"arm": "left", "position_m": [0, 0, 1.02],
                      "orientation_xyzw": [0, 0, 0, 1]},
    }
    session.execute(action)
    second, _ = session.observe(observation_root)
    second_state = store.publish(
        observation=second, observation_root=observation_root / second["observation_id"],
        cause="after_action", task_instruction="Move safely once.",
        previous_public_action=action,
        remaining_budget={"robot_decisions": 0, "model_calls": 2},
    )
    store.finalize(state="completed", termination="trusted_runtime_boundary")
    verification = verify_policy_state(tmp_path)
    assert verification["status"] == "valid"
    assert second_state["previous_state_sha256"] == first_state["state_sha256"]
    assert second_state["previous_public_action"] == action
    assert second_state["previous_execution_receipt"]["receipt"]["state_changed"] is True
    tools = RobotToolRuntime(
        tmp_path / "control", MANUAL, tmp_path / "events.jsonl", BOUNDS,
        allowed_tools=[
            "observation.dump_state", "vision.inspect", "geometry.query",
            "robot.submit_action",
        ],
        state_root=store.root,
    )
    assert tools.invoke("observation.dump_state", {"step": 0})["state"]["step"] == 0
    assert tools.invoke("observation.dump_state", {})["state"]["step"] == 1
    historical_image = tools.invoke(
        "vision.inspect", {"camera": "head_camera", "resolution": "full", "step": 0}
    )
    historical_geometry = tools.invoke(
        "geometry.query", {
            "camera": "head_camera", "points": [[2, 1]], "window_px": 0, "step": 0,
        },
    )
    assert historical_image["step"] == historical_geometry["step"] == 0
    assert historical_image["source_state_sha256"] == first_state["state_sha256"]
    assert historical_geometry["source_state_sha256"] == first_state["state_sha256"]
    assert len(historical_image["source_image_sha256"]) == 64
    with pytest.raises(ToolError, match="unknown policy-state step"):
        tools.invoke("observation.dump_state", {"step": 2})

    step_zero = tmp_path / store.steps[0]["path"]
    step_zero.chmod(0o644)
    tampered = json.loads(step_zero.read_text())
    tampered["remaining_budget"]["robot_decisions"] = 99
    step_zero.write_text(json.dumps(tampered))
    assert verify_policy_state(tmp_path)["status"] == "invalid"
    with pytest.raises(ToolError, match="invalid policy-state evidence"):
        tools.invoke("observation.dump_state", {"step": 1})


def test_synthetic_trace_evaluation_seal_and_export(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    plan = synthetic_plan()
    (run / "plan.json").write_text(json.dumps(plan))
    result = run_episode(run_root=run, plan=plan, run_id="m0-test")
    assert result["state"] == "completed"
    assert result["termination"] == "trusted_runtime_boundary"
    assert result["official_success"] is None
    manifest = json.loads((run / "context-manifest.json").read_text())
    assert manifest["schema"] == "ora.context-manifest.v1"
    assert manifest["planner"]["kind"] == "scripted"
    assert manifest["environment"] == {
        "session_contract": "ora.environment-session.v1",
        "backend": "synthetic",
        "transport": "in_process_direct",
        "process_isolated": False,
        "runtime_python_env": None,
        "transport_protocol": None,
        "session_evidence": None,
        "exclusive_writer": True,
        "public_observation_schema": "ora-observation.v1",
        "state_change_boundary": "execute(action)",
        "official_verdict_owner": "environment",
        "status_contract": "ora.environment-status.v1",
        "status_policy": "lifecycle_only",
        "status_visibility": "control_only",
        "policy_forbidden_inputs": [
            "actor_state", "contact", "expert_program", "object_gt_pose",
            "segmentation", "success_predicate", "task_source",
        ],
    }
    assert manifest["tool_surface"]["only_state_changing_tool"] == "robot.submit_action"
    assert manifest["state_system"]["manifest_path"] == "policy-state/manifest.json"
    state_check = verify_policy_state(run)
    assert state_check["status"] == "valid"
    assert [item["step"] for item in state_check["steps"]] == [0, 1]
    assert [item["cause"] for item in state_check["steps"]] == ["reset", "after_action"]
    assert state_check["steps"][1]["previous_public_action"]["action"] == "move_pose"
    assert state_check["manifest"]["finalized"] is True
    first = evaluate(run)
    second = evaluate(run)
    assert first == second
    assert first["status"] == "valid"
    assert first["context_manifest_sha256"]
    assert first["runtime_metrics_sha256"]
    metrics = json.loads((run / "runtime-metrics.json").read_text())
    assert metrics["schema"] == "ora.runtime-metrics.v1"
    assert metrics["measurement_quality"]["provider_usage"] == "missing"
    assert metrics["latency"]["clock"]["duration_source"] == "monotonic"
    sealed = seal(run)
    assert sealed["evaluation_status"] == "valid"
    assert sealed["schema"] == "ora.seal.v2"
    assert sealed["mode"] == "operator"
    assert sealed["launch_admission_verification"] == "absent"
    assert isinstance(sealed["sealed_source_commit"], str)
    destination = tmp_path / "export.tar.gz"
    assert export(run, destination) == destination
    assert destination.stat().st_size > 0


def test_backfill_seal_records_mode_and_supersedes_the_prior_evaluation(
    tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    plan = synthetic_plan()
    (run / "plan.json").write_text(json.dumps(plan))
    result = run_episode(run_root=run, plan=plan, run_id="m0-backfill")
    assert result["state"] == "completed"
    first_evaluation = evaluate(run)
    assert first_evaluation["status"] == "valid"
    sealed = seal(run, mode="backfill")
    assert sealed["schema"] == "ora.seal.v2"
    assert sealed["mode"] == "backfill"
    assert sealed["evaluation_status"] == "valid"
    superseded = run / "evaluation-superseded-by-backfill.json"
    assert superseded.is_file()
    assert json.loads(superseded.read_text())["status"] == "valid"
    with pytest.raises(EvaluationError, match="already sealed"):
        seal(run, mode="backfill")


def test_verify_launch_admission_snapshot_and_backfill_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.orchestrator import spec as spec_module

    requirement = {
        "gate": "outputs/exp-x/gates/gate.json",
        "resource_profile": "profile", "slot_id": "slot",
    }
    monkeypatch.setattr(
        spec_module, "_launch_requirement", lambda *a, **k: dict(requirement),
    )
    monkeypatch.setattr(
        spec_module, "resolve_workspace_root", lambda root: tmp_path,
    )
    validated = []
    monkeypatch.setattr(
        spec_module, "_validate_launch_gate_record",
        lambda **kw: validated.append(kw),
    )
    gate_bytes = b'{"status": "ok"}'
    plan = {
        "experiment": "exp-x", "config": "cell",
        "source": {"commit": "deadbeef"}, "runtime": {},
        "launch_admission": {"mode": "required", "gate": {
            "path": requirement["gate"],
            "sha256": hashlib.sha256(gate_bytes).hexdigest(),
            "bytes": len(gate_bytes),
        }},
    }
    gate_path = tmp_path / requirement["gate"]
    gate_path.parent.mkdir(parents=True)
    gate_path.write_bytes(gate_bytes)
    assert spec_module.verify_launch_admission(plan) == "live"
    assert validated

    # The live gate is requalified in place; the run-directory snapshot of
    # the launch-time bytes keeps the lineage verifiable without re-running
    # the experiment validator against today's checkout.
    gate_path.write_bytes(b'{"status": "requalified"}')
    run_root = tmp_path / "run"
    (run_root / "qualification").mkdir(parents=True)
    (run_root / "qualification" / "launch-gate.json").write_bytes(gate_bytes)
    assert (
        spec_module.verify_launch_admission(plan, run_root=run_root)
        == "snapshot"
    )

    # Without any surviving byte source a live verification fails closed,
    # while an explicit backfill records the admission as attested only.
    bare_run = tmp_path / "run2"
    bare_run.mkdir()
    with pytest.raises(spec_module.RunSpecError, match="unavailable or changed"):
        spec_module.verify_launch_admission(plan, run_root=bare_run)
    assert (
        spec_module.verify_launch_admission(
            plan, run_root=bare_run, admission="backfill",
        )
        == "recorded_only"
    )

    # Backfill with an intact live gate verifies bytes without re-running
    # the experiment validator, which binds gates to the current HEAD.
    gate_path.write_bytes(gate_bytes)
    validated.clear()
    assert (
        spec_module.verify_launch_admission(plan, admission="backfill")
        == "live"
    )
    assert validated == []


def test_three_step_synthetic_diagnostic_exercises_runtime_state_chain(
    tmp_path: Path,
) -> None:
    run = tmp_path / "state-chain-run"
    run.mkdir()
    plan = synthetic_plan()
    plan.update({
        "plan_sha256": "exp00b-state-chain-plan",
        "plan_id": "exp00b-v1-scripted-test",
        "experiment": "exp00b", "config": "v1",
        "milestone": "context-state-chain",
        "output_workspace": "outputs/exp00b",
        "task": "synthetic_context_state_chain",
        "task_instruction": "Probe public Runtime state and move three times.",
    })
    plan["policy"] = {
        "kind": "scripted", "model": "scripted-policy-v1",
        "provider": "trusted_scripted", "reasoning_effort": "none",
        "state_probe": True,
        "tool_allowlist": ["observation.dump_state", "robot.submit_action"],
        "actions": [
            {
                "action": "move_pose", "arm": "left", "position_m": [0, 0, z],
                "orientation_xyzw": [0, 0, 0, 1],
            }
            for z in (1.01, 1.02, 1.03)
        ],
    }
    plan["budgets"] = {
        "max_robot_decisions": 3, "max_model_calls": 8,
        "max_sim_steps": 3, "wall_time_s": 30,
    }
    plan["runtime"]["terminal_actions"] = 3
    (run / "plan.json").write_text(json.dumps(plan))
    result = run_episode(run_root=run, plan=plan, run_id="exp00b-state-chain")
    assert result["state"] == "completed"
    assert result["decisions"] == 3
    verification = verify_policy_state(run)
    assert verification["status"] == "valid"
    assert [item["step"] for item in verification["steps"]] == [0, 1, 2, 3]
    assert verification["steps"][0]["previous_public_action"] is None
    for step in verification["steps"][1:]:
        assert step["previous_public_action"]["action"] == "move_pose"
        assert step["previous_execution_receipt"]["receipt"]["state_changed"] is True
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    probes = [
        event for event in events
        if event.get("type") == "tool_result"
        and event.get("tool") == "observation.dump_state"
    ]
    assert [event["arguments"]["step"] for event in probes] == [0, 1, 2]
    assert sorted(path.name for path in (run / "native/receipts").iterdir()) == [
        "action-0000.json", "action-0001.json", "action-0002.json"
    ]
    assert evaluate(run)["status"] == "valid"


def test_supervisor_stops_on_control_only_environment_signal(
    tmp_path: Path,
) -> None:
    class SignalingSyntheticSession(SyntheticSession):
        def status(self) -> dict[str, object]:
            if self.action_sequence < 1:
                return super().status()
            return validate_environment_status({
                "schema": ENVIRONMENT_STATUS_SCHEMA,
                "phase": "terminal", "terminal": True,
                "reason": "task_object_out_of_recoverable_workspace",
                "source": "test_environment",
                "policy_visibility": "control_only",
                "signals": [{
                    "code": "task_object_out_of_recoverable_workspace",
                    "severity": "terminal", "source": "test_task_adapter",
                    "policy_visibility": "control_only",
                }],
            })

    session = SignalingSyntheticSession(
        tmp_path / "native", "synthetic_tool_boundary", 0, max_actions=2,
    )
    run = tmp_path / "coordinated-run"
    run.mkdir()
    plan = synthetic_plan()
    plan["policy"]["actions"] = [
        {
            "action": "move_pose", "arm": "left",
            "position_m": [0, 0, height],
            "orientation_xyzw": [0, 0, 0, 1],
        }
        for height in (1.01, 1.02)
    ]
    plan["runtime"]["terminal_actions"] = 2
    plan["budgets"]["max_robot_decisions"] = 2
    result = run_episode(
        run_root=run,
        plan=plan,
        run_id="coordinated-test",
        environment_session_factory=lambda **_kwargs: session,
    )
    assert result["decisions"] == 1
    assert result["termination"] == "trusted_runtime_boundary"
    assert result["environment_status"]["reason"] == (
        "task_object_out_of_recoverable_workspace"
    )
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    statuses = [event["record"] for event in events if event["type"] == "environment_status"]
    assert [status["phase"] for status in statuses] == ["running", "terminal"]
    final_state = (run / "policy-state/steps/step-0001.json").read_text()
    assert "task_object_out_of_recoverable_workspace" not in final_state


def test_evaluation_rejects_tampered_policy_state(tmp_path: Path) -> None:
    run = tmp_path / "tampered-run"
    run.mkdir()
    plan = synthetic_plan()
    (run / "plan.json").write_text(json.dumps(plan))
    run_episode(run_root=run, plan=plan, run_id="tampered-m0")
    step = run / "policy-state/steps/step-0000.json"
    step.chmod(0o644)
    value = json.loads(step.read_text())
    value["remaining_budget"]["robot_decisions"] = 999
    step.write_text(json.dumps(value))
    evaluation = evaluate(run)
    assert evaluation["status"] == "invalid"
    assert any("state_step" in reason for reason in evaluation["reasons"])
    with pytest.raises(EvaluationError, match="invalid runs cannot be sealed"):
        seal(run)


@pytest.mark.parametrize("filename", ["AGENTS.md", "CLAUDE.md"])
def test_seal_inventory_accepts_only_exact_managed_instruction_aliases(
    tmp_path: Path, filename: str,
) -> None:
    run = tmp_path / "run"
    workspace = run / "sandbox/workspace"
    scratch = workspace / "scratch"
    scratch.mkdir(parents=True)
    (workspace / filename).write_text("bound project instructions\n", encoding="utf-8")
    alias = scratch / filename
    alias.symlink_to(f"../{filename}")

    assert inventory(run) == [
        {
            "path": f"sandbox/workspace/{filename}",
            "sha256": file_sha256(workspace / filename),
            "bytes": len("bound project instructions\n"),
        }
    ]

    alias.unlink()
    alias.symlink_to("../../outside-project-instructions")
    with pytest.raises(EvaluationError, match="symbolic links cannot enter sealed evidence"):
        inventory(run)


def test_learning_eligibility_rejects_legacy_and_faulted_libero_evidence(
    tmp_path: Path,
) -> None:
    def artifact(relative: str, value: dict) -> dict[str, str]:
        path = tmp_path / "native" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    workspace_profile = resolve_workspace_profile("libero_pro_panda_tabletop_v1")
    contract = libero_osc_action_contract({
        "control_delta": True,
        "input_min": -1, "input_max": 1,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    })
    plan = {
        "runtime": {
            "kind": "libero_pro", "workspace_profile": workspace_profile,
            "workspace_bounds_m": workspace_profile["bounds_m"],
        },
    }
    healthy = {
        "schema": "ora.libero-control-health.v2",
        "status": "healthy", "codes": [],
        "controller_action_contract_sha256": contract["contract_sha256"],
    }
    qualified_binding = {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": contract["contract_sha256"],
        "binding": {
            "robosuite_module": "/qualified/robosuite/__init__.py",
            "robosuite_module_sha256": "1" * 64,
            "controller_config_file": "/qualified/robosuite/osc_pose.json",
            "controller_config_sha256": "2" * 64,
            "panda_gripper_module": "/qualified/robosuite/panda_gripper.py",
            "panda_gripper_module_sha256": "3" * 64,
            "panda_gripper_file": "/qualified/robosuite/panda_gripper.xml",
            "panda_gripper_sha256": "4" * 64,
        },
    }
    events = [
        {"type": "reset", "record": {
            "controller_action_contract": contract,
            "workspace_profile": workspace_profile,
            "workspace_bounds_m": workspace_profile["bounds_m"],
            "qualified_controller_binding": qualified_binding,
            "source_manifest": {"dependencies": [
                {"path": qualified_binding["binding"][path_field],
                 "sha256": qualified_binding["binding"][digest_field]}
                for path_field, digest_field in (
                    ("robosuite_module", "robosuite_module_sha256"),
                    ("controller_config_file", "controller_config_sha256"),
                    ("panda_gripper_module", "panda_gripper_module_sha256"),
                    ("panda_gripper_file", "panda_gripper_sha256"),
                )
            ]},
        }},
        {"type": "motion_receipt", "record": {
            "schema": "ora.libero-pro-motion-receipt.v2",
            "action": "set_gripper",
            "control_health": healthy,
            "tracking_evidence": artifact("tracking/action-0.json", {
                "schema": "ora.libero-pose-tracking-evidence.v1",
                "policy_visibility": "control_only",
                "compiled_pose_sha256": None,
            }),
        }},
    ]
    eligible = assess_learning_eligibility(
        plan, {"state": "completed"}, events, run_root=tmp_path,
    )
    assert eligible["eligible"] is True
    (tmp_path / "native/tracking/action-0.json").write_text(
        json.dumps({"schema": "tampered"})
    )
    tampered = assess_learning_eligibility(
        plan, {"state": "completed"}, events, run_root=tmp_path,
    )
    assert tampered["eligible"] is False
    assert "missing_or_invalid_pose_tracking:0" in tampered["reasons"]
    composite = assess_learning_eligibility(
        plan, {"state": "completed"}, [
            events[0],
            {"type": "motion_receipt", "record": {
                "schema": "ora.composite-motion-receipt.v1",
                "phase_receipts": [
                    {"phase": "pose", "status": "reached", "receipt": {
                        "schema": "ora.libero-pro-motion-receipt.v2",
                        "action": "move_pose",
                        "pose_contract_sha256": "6" * 64,
                        "compiled_pose_sha256": "7" * 64,
                        "tracking_evidence": artifact("tracking/action-pose.json", {
                            "schema": "ora.libero-pose-tracking-evidence.v1",
                            "policy_visibility": "control_only",
                            "compiled_pose_sha256": "7" * 64,
                        }),
                        "control_health": healthy,
                    }},
                    {"phase": "gripper", "status": "executed", "receipt": {
                        "schema": "ora.libero-pro-motion-receipt.v2",
                        "action": "set_gripper",
                        "tracking_evidence": artifact("tracking/action-gripper.json", {
                            "schema": "ora.libero-pose-tracking-evidence.v1",
                            "policy_visibility": "control_only",
                            "compiled_pose_sha256": None,
                        }),
                        "control_health": healthy,
                    }},
                ],
            }},
        ], run_root=tmp_path,
    )
    assert composite["eligible"] is True
    # A closed manipulation skill wraps those same composite receipts one level
    # deeper, under stage_receipts. Its measurement stages drive nothing and
    # carry no action_id, so they owe no controller evidence; a dispatched
    # stage that lost its receipt still has to fail.
    skill_stage = lambda name, receipt: {
        "stage": name, "status": "executed", "reason": "executed",
        "action_id": f"action-abc:{name}", "measurements": [], "receipt": receipt,
    }
    driven_composite = {
        "schema": "ora.composite-motion-receipt.v1",
        "phase_receipts": [
            {"phase": "pose", "status": "reached", "receipt": {
                "schema": "ora.libero-pro-motion-receipt.v2",
                "action": "move_pose",
                "pose_contract_sha256": "6" * 64,
                "compiled_pose_sha256": "7" * 64,
                "tracking_evidence": artifact("tracking/action-pose.json", {
                    "schema": "ora.libero-pose-tracking-evidence.v1",
                    "policy_visibility": "control_only",
                    "compiled_pose_sha256": "7" * 64,
                }),
                "control_health": healthy,
            }},
        ],
    }
    skill_sequence = assess_learning_eligibility(
        plan, {"state": "completed"}, [
            events[0],
            {"type": "motion_receipt", "record": {
                "schema": "ora.manipulation-skill-sequence-receipt.v1",
                "skill": "grasp_execute",
                "status": "completed",
                "stage_receipts": [
                    skill_stage("approach", driven_composite),
                    skill_stage("close", driven_composite),
                    # Measurement stage: no action_id, no receipt, drives nothing.
                    {"stage": "verify", "status": "executed",
                     "reason": "verified_against_observation-0001",
                     "action_id": None, "measurements": []},
                ],
            }},
        ], run_root=tmp_path,
    )
    assert skill_sequence["eligible"] is True, skill_sequence["reasons"]
    dispatched_without_receipt = assess_learning_eligibility(
        plan, {"state": "completed"}, [
            events[0],
            {"type": "motion_receipt", "record": {
                "schema": "ora.manipulation-skill-sequence-receipt.v1",
                "skill": "grasp_execute",
                "status": "completed",
                "stage_receipts": [
                    {"stage": "approach", "status": "executed",
                     "reason": "executed", "action_id": "action-abc:approach",
                     "measurements": []},
                ],
            }},
        ], run_root=tmp_path,
    )
    assert dispatched_without_receipt["eligible"] is False
    assert "missing_or_invalid_control_health:0" in (
        dispatched_without_receipt["reasons"]
    )
    hybrid_events = [
        events[0],
        {"type": "motion_receipt", "record": {
            "schema": "ora.libero-pro-hybrid-motion-receipt.v1",
            "action": "move_free_space", "motion_status": "reached",
            "state_changed": True,
            "control_health": {
                "schema": "ora.libero-control-health.v2",
                "status": "healthy", "codes": [],
            },
            "planner": {
                "world_model": "none",
                "canonical_artifact": artifact(
                    "curobo/plans/action-0.json", {"schema": "plan"},
                ),
            },
            "execution": {
                "controller": "JOINT_POSITION",
                "canonical_artifact": artifact(
                    "tracking/joint-action-0.json", {"schema": "joint-tracking"},
                ),
            },
            "feedback": {"reachability": True},
        }},
        {"type": "motion_receipt", "record": {
            "schema": "ora.libero-pro-hybrid-motion-receipt.v1",
            "action": "servo_delta", "motion_status": "reached",
            "control_health": healthy,
            "pose_contract_sha256": "c" * 64,
            "compiled_pose_sha256": "d" * 64,
            "tracking_evidence": artifact("tracking/action-servo.json", {
                "schema": "ora.libero-pose-tracking-evidence.v1",
                "policy_visibility": "control_only",
                "compiled_pose_sha256": "d" * 64,
            }),
        }},
        {"type": "motion_receipt", "record": {
            "schema": "ora.libero-pro-hybrid-motion-receipt.v1",
            "action": "recover_motion", "motion_status": "refused",
            "state_changed": False,
            "control_health": {
                "schema": "ora.libero-control-health.v2",
                "status": "degraded", "codes": ["IK_FAIL"],
            },
            "feedback": {"reachability": False},
        }},
    ]
    hybrid = assess_learning_eligibility(
        {**plan, "policy": {"action_codec": "motion_action_v1"}},
        {"state": "completed"}, hybrid_events, run_root=tmp_path,
    )
    assert hybrid["eligible"] is True
    broken_hybrid = assess_learning_eligibility(
        {**plan, "policy": {"action_codec": "motion_action_v1"}},
        {"state": "completed"}, [
            events[0],
            {"type": "motion_receipt", "record": {
                **hybrid_events[1]["record"], "execution": None,
            }},
        ], run_root=tmp_path,
    )
    assert "missing_or_invalid_hybrid_motion_evidence:0" in (
        broken_hybrid["reasons"]
    )
    missing_phase_receipt = assess_learning_eligibility(
        plan, {"state": "completed"}, [
            events[0],
            {"type": "motion_receipt", "record": {
                "schema": "ora.composite-motion-receipt.v1",
                "phase_receipts": [
                    {"phase": "pose", "status": "reached"},
                    {"phase": "gripper", "status": "skipped"},
                ],
            }},
        ], run_root=tmp_path,
    )
    assert "missing_or_invalid_control_health:0" in missing_phase_receipt["reasons"]
    legacy = assess_learning_eligibility(
        {"runtime": {
            "kind": "libero_pro",
            "workspace_bounds_m": workspace_profile["bounds_m"],
        }},
        {"state": "completed"},
        [{"type": "motion_receipt", "record": {}}],
    )
    assert legacy["eligible"] is False
    assert set(legacy["reasons"]) >= {
        "unqualified_action_workspace",
        "missing_or_invalid_controller_action_contract",
        "missing_or_invalid_controller_source_qualification",
        "missing_or_invalid_control_health:0",
    }
    faulted = assess_learning_eligibility(
        plan, {"state": "completed"}, [
            events[0],
            {"type": "motion_receipt", "record": {"control_health": {
                **healthy, "status": "fault", "codes": ["gross_target_divergence"],
            }}},
        ],
    )
    assert faulted["eligible"] is False
    assert "control_health_fault:0" in faulted["reasons"]




def test_learning_eligibility_accepts_sparse_motion_envelopes(
    tmp_path: Path,
) -> None:
    """EXP08A sparse v3 envelopes embed the executor's legacy receipt."""

    def artifact(relative: str, value: dict) -> dict[str, str]:
        path = tmp_path / "native" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))
        return {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    workspace_profile = resolve_workspace_profile("libero_pro_panda_tabletop_v1")
    contract = libero_osc_action_contract({
        "control_delta": True,
        "input_min": -1, "input_max": 1,
        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    })
    plan = {
        "runtime": {
            "kind": "libero_pro", "workspace_profile": workspace_profile,
            "workspace_bounds_m": workspace_profile["bounds_m"],
        },
    }
    healthy = {
        "schema": "ora.libero-control-health.v2",
        "status": "healthy", "codes": [],
        "controller_action_contract_sha256": contract["contract_sha256"],
    }
    qualified_binding = {
        "schema": "ora.libero-controller-qualification.v1",
        "contract_sha256": contract["contract_sha256"],
        "binding": {
            "robosuite_module": "/qualified/robosuite/__init__.py",
            "robosuite_module_sha256": "1" * 64,
            "controller_config_file": "/qualified/robosuite/osc_pose.json",
            "controller_config_sha256": "2" * 64,
            "panda_gripper_module": "/qualified/robosuite/panda_gripper.py",
            "panda_gripper_module_sha256": "3" * 64,
            "panda_gripper_file": "/qualified/robosuite/panda_gripper.xml",
            "panda_gripper_sha256": "4" * 64,
        },
    }
    reset = {"type": "reset", "record": {
        "controller_action_contract": contract,
        "workspace_profile": workspace_profile,
        "workspace_bounds_m": workspace_profile["bounds_m"],
        "qualified_controller_binding": qualified_binding,
        "source_manifest": {"dependencies": [
            {"path": qualified_binding["binding"][path_field],
             "sha256": qualified_binding["binding"][digest_field]}
            for path_field, digest_field in (
                ("robosuite_module", "robosuite_module_sha256"),
                ("controller_config_file", "controller_config_sha256"),
                ("panda_gripper_module", "panda_gripper_module_sha256"),
                ("panda_gripper_file", "panda_gripper_sha256"),
            )
        ]},
    }}
    global_minimal_health = {
        "schema": "ora.libero-control-health.v2",
        "status": "healthy", "codes": [],
    }
    global_envelope = {"type": "motion_receipt", "record": {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "move_pose", "action_id": "action-g",
        "motion_strategy": "global",
        "execution_status": "reached", "state_changed": True,
        "control_health": global_minimal_health,
        "segment": {
            "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
            "action": "move_pose", "action_id": "action-g",
            "motion_status": "reached",
            "control_health": global_minimal_health,
            "planner": {
                "source": "motion_probe_plan_ref",
                "plan_ref": "plan-" + "a" * 32,
                "artifact": artifact("motion-probes/plan-g.json", {
                    "schema": "ora.motion-plan-artifact.v1"
                }),
            },
            "execution": {
                "controller": "JOINT_POSITION",
                "canonical_artifact": artifact("curobo/exec-g.json", {
                    "controller": "JOINT_POSITION"
                }),
            },
        },
    }}
    local_envelope = {"type": "motion_receipt", "record": {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "move_pose", "action_id": "action-l",
        "motion_strategy": "local",
        "execution_status": "reached", "state_changed": True,
        "control_health": healthy,
        "segment": {
            "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
            "action": "servo_delta", "action_id": "action-l",
            "motion_status": "reached",
            "control_health": healthy,
            "pose_contract_sha256": "6" * 64,
            "compiled_pose_sha256": "7" * 64,
            "tracking_evidence": artifact("tracking/action-l.json", {
                "schema": "ora.libero-pose-tracking-evidence.v1",
                "policy_visibility": "control_only",
                "compiled_pose_sha256": "7" * 64,
            }),
        },
    }}
    refusal_envelope = {"type": "motion_receipt", "record": {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "move_pose", "action_id": "action-r",
        "motion_strategy": None,
        "execution_status": "plan_rejected", "state_changed": False,
        "failure_code": "route_contact_approach_exceeds_local_bound",
    }}
    eligible = assess_learning_eligibility(
        plan, {"state": "completed"},
        [reset, global_envelope, local_envelope, refusal_envelope],
        run_root=tmp_path,
    )
    assert eligible["eligible"] is True, eligible["reasons"]

    forged = json.loads(json.dumps(global_envelope))
    forged["record"]["segment"]["action_id"] = "action-other"
    forged_result = assess_learning_eligibility(
        plan, {"state": "completed"}, [reset, forged], run_root=tmp_path,
    )
    assert forged_result["eligible"] is False
    assert any(
        reason.startswith("missing_or_invalid_control_health")
        for reason in forged_result["reasons"]
    )

def test_fault_result_reconciles_native_provider_calls_from_agent_events(
    tmp_path: Path,
) -> None:
    events = [
        {
            "type": "provider_response",
            "model_call_id": "model-call-000000",
            "payload": {"accepted": True},
        },
        {
            "type": "provider_response",
            "model_call_id": "model-call-000001",
            "payload": {"accepted": True},
        },
        {
            "type": "provider_response",
            "model_call_id": "model-call-rejected",
            "payload": {"accepted": False},
        },
        # A duplicate event cannot create an extra billed completion.
        {
            "type": "provider_response",
            "model_call_id": "model-call-000001",
            "payload": {"accepted": True},
        },
    ]
    (tmp_path / "agent-events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    plan = {"policy": {"kind": "native_api"}}
    assert _reconciled_provider_model_calls(tmp_path, plan, 0) == 2
    assert _reconciled_provider_model_calls(tmp_path, plan, 3) == 3
