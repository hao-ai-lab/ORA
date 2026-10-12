"""Simulator-free RoboDojo transport, terminal evidence and sealing checks."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pytest

from ora.environment.rpc import EnvironmentRpcError, LocalRpcEnvironmentSession
from ora.orchestrator import coordinator
from ora.orchestrator.evaluation import (
    EvaluationError, _robodojo_evidence_errors, evaluate, seal,
)


ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {name: str(index) * 40 for index, name in enumerate(
    ("robodojo", "isaaclab", "curobo", "xpolicylab", "assets"), start=1,
)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _worker_project(tmp_path: Path) -> Path:
    """Use the real RPC server with a fake plant in a separate interpreter."""

    project = tmp_path / "fake-worker"
    package = project / "ora"
    environment = package / "environment"
    environment.mkdir(parents=True)
    # The real worker's Kit PID-collision guard reads SHM_ROOT at call time;
    # point it at an empty directory so no test reads or unlinks /dev/shm.
    shm = tmp_path / "fake-shm"
    shm.mkdir()
    (package / "__init__.py").write_text(
        f"__path__.append({str(ROOT / 'ora')!r})\n"
    )
    (environment / "__init__.py").write_text(
        f"__path__.append({str(ROOT / 'ora/environment')!r})\n"
    )
    (environment / "worker.py").write_text(
        f"REAL_WORKER = {str(ROOT / 'ora/environment/worker.py')!r}\n"
        f"FAKE_SHM = {str(shm)!r}\n"
        + r'''
import hashlib
import json
import os
import time
from pathlib import Path
from ora.environment.synthetic import SyntheticSession

def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")

class FixtureSession(SyntheticSession):
    def __init__(self, config):
        super().__init__(Path(config["output_root"]), config["task"], config["seed"],
                         action_codec="runtime_action_v3",
                         max_actions=200 if config["mode"] in (
                             "budget_before_official", "fault_before_official") else 2)
        self.config = config

    def start(self):
        super().start()
        source = self.config["source_record"]
        path = self.output_root / "source-manifest.json"
        write(path, source)
        self.source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "schema": "ora.robodojo-reset.v1",
            "task": self.task, "eval_seed": self.seed, "layout_id": 0,
            "source_manifest": {"path": path.name, "sha256": self.source_hash},
            "physics_steps_before_policy": 0, "reward_registered": True,
        }

    def execute(self, action):
        if self.config["mode"] == "worker_death":
            os._exit(73)
        record = super().execute(action)
        stem = f"action-{self.action_sequence - 1:04d}"
        control_path = self.output_root / f"controller/{stem}.json"
        control_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path = self.output_root / f"controller/{stem}.jsonl"
        rows = [{
            "action_id": action["action_id"] if self.config["mode"] != "misbound_journal" else "another-action",
            "physics_clock": (self.action_sequence - 1) * 10 + index + 1,
            "native_action": self.action_sequence, "command": {"fixture": True}, "measured": {},
        } for index in range(10)]
        if self.config["mode"] == "truncated_journal":
            rows.pop()
        journal_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        write(control_path, {
            "schema": "ora.robodojo-controller-trace.v1",
            "action_id": action["action_id"],
            "samples": [{"native_action": self.action_sequence, "physics_steps": self.action_sequence * 10}],
            "physics_journal": {"path": journal_path.relative_to(self.output_root).as_posix(),
                                "sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest()},
        })
        record["receipt"].update({
            "schema": (
                "ora.robodojo-sparse-motion-receipt.v1"
                if self.config["mode"] == "sparse_receipt"
                else "ora.robodojo-motion-receipt.v1"
            ),
            "action_id": action["action_id"],
            "native_actions": 1, "physics_steps": 10,
            "control_trace": {"path": control_path.relative_to(self.output_root).as_posix(),
                              "sha256": hashlib.sha256(control_path.read_bytes()).hexdigest()},
        })
        if self.config["mode"] == "semantic_count_mismatch":
            record["receipt"].update(native_actions=2, physics_steps=20)
        path = self.output_root / record["path"]
        write(path, record["receipt"])
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return record

    def status(self):
        value = super().status()
        value["source"] = "robodojo_session"
        if value["terminal"] and self.abort_reason is None:
            value["reason"] = "official_task_end"
        return value

    def persist_official_verdict(self):
        value = {
            "schema": "ora.robodojo-official-verdict.v2",
            "official_success": self.config["mode"] != "false",
            "round_index": 0,
            "terminal": self.terminal, "task": self.task, "eval_seed": self.seed,
            "layout_id": 0, "source_manifest_sha256": self.source_hash,
            "verdict_basis": "official_terminal_transition", "fault_reason": None,
            "native_actions": self.action_sequence, "physics_steps": self.action_sequence * 10,
            "reset_physics_steps": 0,
            "abort_reason": self.abort_reason, "source": "RoboDojo EvalEnv.is_episode_end",
        }
        if self.config["mode"] == "missing_verdict":
            value["official_success"] = None
            value["verdict_basis"] = "episode_incomplete"
            return value
        if self.config["mode"] == "budget_before_official":
            # The Supervisor's own budget stopped the round: the simulator
            # never declared a terminal transition, so the standing official
            # predicate is reported instead of a null.
            value["official_success"] = False
            value["terminal"] = False
            value["verdict_basis"] = "standing_official_predicate_at_policy_budget"
        if self.config["mode"] == "fault_before_official":
            value["official_success"] = None
            value["terminal"] = False
            value["verdict_basis"] = "infrastructure_abort"
            value["fault_reason"] = "robodojo_native_execution_failed"
        if self.config["mode"] == "premature_success":
            value["terminal"] = False
        if self.config["mode"] == "verdict_count_mismatch":
            value["native_actions"] += 1
            value["physics_steps"] += 10
        write(self.output_root / "official-verdict.json", value)
        return value

    def close(self):
        if self.config["mode"] == "hang_close":
            time.sleep(60)

namespace = {"__name__": "fixture_rpc_worker", "__package__": "ora.environment"}
exec(compile(Path(REAL_WORKER).read_text(), REAL_WORKER, "exec"), namespace)
namespace["SHM_ROOT"] = Path(FAKE_SHM)
namespace["_build_session"] = FixtureSession
raise SystemExit(namespace["main"]())
'''
    )
    return project


def _source_record() -> dict:
    return {
        "schema": "ora.robodojo-source-manifest.v1",
        "task": "fixture_task", "eval_seed": 0, "layout_id": 0, "num_envs": 1,
        "source_revisions": REVISIONS, "required_revisions": REVISIONS,
        "layout_file": {"path": "/fixture/Assets/Eval_Layout/0/0.json", "sha256": "a" * 64},
        "action_codec": "runtime_action_v3", "sensor_profile": "fixture_rgbd_v1",
        "physics_dt_s": 0.004, "control_hz": 25, "official_native_action_limit": 200,
    }


def _session(tmp_path: Path, *, mode: str, close_timeout_s: float | None = 1.0):
    native = tmp_path / "native"
    return LocalRpcEnvironmentSession(
        python_executable=Path(sys.executable), project_root=_worker_project(tmp_path),
        output_root=native, socket_root=tmp_path / "sockets",
        worker_config={
            "backend": "robodojo", "output_root": str(native), "task": "fixture_task",
            "seed": 0, "mode": mode, "source_record": _source_record(),
        },
        startup_timeout_s=10, request_timeout_s=60, close_timeout_s=close_timeout_s,
    )


def _round_isolation(clock: int) -> dict:
    """The measured restore proof a RoboDojo round-reset record must carry."""

    return {
        "schema": "ora.robodojo-round-isolation.v1",
        "passed": True,
        "checks": {
            "arm_home_restored": True,
            "carried_channels_cleared": True,
            "episode_counters_zeroed": True,
            "failure_latch_cleared": True,
            "layout_identity_unchanged": True,
            "official_lifecycle_rebound": True,
            "official_terminal_cleared": True,
        },
        "measured": {
            "arms": ["left", "right"],
            "cleared_channels": ["official_terminal"],
            "end_flag": False,
            "failure_latch_clear": True,
            "layout_sha256": "a" * 64,
            "layout_sha256_at_start": "a" * 64,
            "max_home_position_error_m": 0.0,
            "max_home_rotation_error_deg": 0.0,
            "native_action_limit": 200,
            "physics_steps_before_policy": clock,
            "reward_registered": True,
            "step_lim": 200,
            "stale_channels": [],
        },
        "tolerances": {"home_position_m": 0.02, "home_rotation_deg": 5.0},
        "failures": [],
    }


def _plan() -> dict:
    paths = {
        "codex_project": "ora/policy/planners/codex/workspace-v1.md",
        "episode_prefix": "ora/policy/core/thin-react-v1.md",
        "tool_manual": "ora/policy/embodiments/aloha-agilex/manual-v1.md",
    }
    return {
        "schema": "ora.run-plan.v1", "plan_sha256": "fixture-plan",
        "plan_id": "robodojo-fixture", "experiment": "runtime-v1", "config": "fixture",
        "milestone": "M0", "output_workspace": "outputs/runtime-v1",
        "task": "fixture_task", "task_instruction": "Move safely once.", "seed": 0,
        "evidence_tier": "diagnostic", "source": {"commit": "fixture", "dirty": False},
        "policy": {
            "kind": "scripted", "model": "scripted-policy-v1", "provider": "trusted_scripted",
            "reasoning_effort": "none", "action_codec": "runtime_action_v3",
            "actions": [{"action": "move_pose", "arm": "left", "position_m": [0, 0, position],
                         "orientation_xyzw": [0, 0, 0, 1]} for position in (1.02, 1.04)],
        },
        "runtime": {
            "kind": "robodojo", "transport": "local_rpc", "eval_seed": 0, "layout_id": 0,
            "layout_file": "Eval_Layout/0/0.json", "layout_sha256": "a" * 64,
            "sensor_profile": "fixture_rgbd_v1", "physics_dt": 0.004,
            "control_hz": 25, "native_action_limit": 200,
            "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
            "required_revision": REVISIONS["robodojo"],
            "required_isaaclab_revision": REVISIONS["isaaclab"],
            "required_curobo_revision": REVISIONS["curobo"],
            "required_xpolicylab_revision": REVISIONS["xpolicylab"],
            "required_asset_revision": REVISIONS["assets"],
        },
        "budgets": {"max_robot_decisions": 2, "max_model_calls": 3, "max_sim_steps": 50,
                    "wall_time_s": 30},
        "policy_context": {
            **{name: [{"path": path, "sha256": _sha(ROOT / path)}] for name, path in paths.items()},
            "composition_sha256": "fixture",
        },
        "qualification": {"required": False},
    }


def test_robodojo_close_has_a_separate_deadline_and_preserves_exit_evidence(tmp_path: Path) -> None:
    session = _session(tmp_path, mode="hang_close", close_timeout_s=0.1)
    session.start()
    assert session.process is not None and session.process.pid != os.getpid()
    started = time.monotonic()
    session.close()
    assert time.monotonic() - started < 20  # The ordinary request deadline is 60 s.
    assert session.process.poll() is not None
    record = json.loads(session.record_path.read_text())
    assert record["close_timeout_s"] == 0.1
    assert "close" in record["close_error"]
    assert record["worker_returncode"] != 0
    assert not session.socket_path.exists()
    first_record = session.record_path.read_bytes()
    session.close()
    assert session.record_path.read_bytes() == first_record


def test_robodojo_close_is_idempotent_after_normal_exit(tmp_path: Path) -> None:
    session = _session(tmp_path, mode="true")
    session.start()
    session.close()
    first_record = session.record_path.read_bytes()
    record = json.loads(first_record)
    assert record["worker_returncode"] == 0 and record["close_error"] is None
    session.close()
    assert session.record_path.read_bytes() == first_record
    with pytest.raises(EnvironmentRpcError, match="session is closed"):
        session.start()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_robodojo_close_rejects_invalid_deadlines(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(EnvironmentRpcError, match="positive and finite"):
        _session(tmp_path, mode="true", close_timeout_s=timeout)


@pytest.mark.parametrize("mode", ["true", "false", "missing_verdict", "premature_success",
                                  "worker_death", "corrupt_receipt", "corrupt_source", "corrupt_control",
                                  "semantic_count_mismatch", "verdict_count_mismatch", "corrupt_journal",
                                  "misbound_journal", "truncated_journal", "budget_before_official",
                                  "fault_before_official", "sparse_receipt"])
def test_robodojo_rpc_supervisor_evidence_and_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    # Admission/source readback is tested by the backend and preflight suites.
    # This fixture exercises the actual Supervisor and RPC after that boundary.
    monkeypatch.setattr(coordinator, "_qualified_runtime_binding", lambda *_args: None)
    run = tmp_path / "run"
    run.mkdir()
    plan = _plan()
    if mode in {"budget_before_official", "fault_before_official"}:
        plan["budgets"]["max_robot_decisions"] = 1
    _write(run / "plan.json", plan)
    session = _session(run, mode=mode)
    result = coordinator.run_episode(
        run_root=run, plan=plan, run_id="robodojo-fixture",
        environment_session_factory=lambda **_kwargs: session,
    )
    if mode == "corrupt_receipt":
        path = run / "native/receipts/action-0000.json"
        payload = json.loads(path.read_text())
        payload["state_changed"] = False
        _write(path, payload)
    elif mode == "corrupt_source":
        path = run / "native/source-manifest.json"
        payload = json.loads(path.read_text())
        payload["layout_id"] = 1
        _write(path, payload)
    elif mode == "corrupt_control":
        path = run / "native/controller/action-0000.json"
        payload = json.loads(path.read_text())
        payload["samples"] = []
        _write(path, payload)
    elif mode == "corrupt_journal":
        path = run / "native/controller/action-0000.jsonl"
        path.write_text(path.read_text() + " ")
    evaluation = evaluate(run)
    if mode in {"true", "false", "sparse_receipt"}:
        assert result["state"] == "completed"
        assert result["official_success"] is (mode != "false")
        assert result["verdict"]["native_actions"] == 2
        assert result["verdict"]["physics_steps"] == 20
        assert result["verdict"]["verdict_basis"] == "official_terminal_transition"
        assert evaluation["status"] == "valid", evaluation["reasons"]
        assert seal(run)["evaluation_status"] == "valid"
    elif mode == "budget_before_official":
        # The policy budget is a behavioral boundary, not a fault: the round
        # completes on the standing official predicate with the simulator's
        # own terminal transition explicitly absent.
        assert result["state"] == "completed"
        assert result["official_success"] is False
        assert result["termination"] == "robot_decision_budget"
        assert result["decisions"] == 1
        assert result["verdict"]["terminal"] is False
        assert result["verdict"]["official_success"] is False
        assert result["verdict"]["verdict_basis"] == (
            "standing_official_predicate_at_policy_budget"
        )
        assert result["verdict"]["fault_reason"] is None
        assert result["verdict"]["abort_reason"] == "robot_decision_budget"
        assert result["verdict"]["native_actions"] == 1 < plan["runtime"]["native_action_limit"]
        assert result["verdict"]["physics_steps"] == 10
        assert evaluation["status"] == "valid", evaluation["reasons"]
        assert seal(run)["evaluation_status"] == "valid"
    elif mode in {"worker_death", "fault_before_official"}:
        assert result["state"] == "incomplete"
        assert result["official_success"] is None
        if mode == "fault_before_official":
            # An infrastructure abort never becomes a task failure.
            assert result["verdict"]["official_success"] is None
            assert result["verdict"]["verdict_basis"] == "infrastructure_abort"
            assert result["verdict"]["fault_reason"] == "robodojo_native_execution_failed"
        assert evaluation["status"] == "incomplete", evaluation["reasons"]
        assert seal(run)["evaluation_status"] == "incomplete"
    else:
        if mode == "missing_verdict":
            assert result["state"] == "incomplete"
            assert result["official_success"] is None
        assert evaluation["status"] == "invalid", evaluation
        if mode == "verdict_count_mismatch":
            assert "robodojo_official_counter_mismatch" in evaluation["reasons"]
        if mode in {"semantic_count_mismatch", "corrupt_journal", "misbound_journal", "truncated_journal"}:
            assert "robodojo_motion_receipt_binding_invalid" in evaluation["reasons"]
        with pytest.raises(EvaluationError, match="invalid runs"):
            seal(run)


def _multi_round_evidence(
    tmp_path: Path, *, rounds: int = 3, perturbation: str | None = None,
) -> tuple[Path, dict, dict, list[dict]]:
    """Lay out one persistent K-round RoboDojo evidence tree and its trace.

    The Supervisor, RPC transport and worker are exercised by the fixture
    above. This builder only reproduces the artifacts and causal trace a
    finished K-round session leaves behind, so one perturbation at a time can
    be joined against the real evaluator. The PhysX clock keeps advancing
    across a round reset while the wrapper's own counters restart, which is
    exactly the join a session-wide sum would get wrong.
    """

    run_root = tmp_path / "run"
    native = run_root / "native"
    _write(native / "source-manifest.json", _source_record())
    source_hash = _sha(native / "source-manifest.json")
    plan = _plan()
    plan["session"] = {
        "cadence": "persistent_agent_k_rounds_v1", "rounds": rounds,
        "budget_scope": "per_round", "reset_owner": "supervisor",
    }
    reset_clocks = [100 + 400 * index for index in range(rounds)]
    events: list[dict] = [{"type": "run_started", "run_id": "robodojo-rounds"}]
    round_records: list[dict] = []
    sequence = 0
    for index in range(rounds):
        clock = reset_clocks[index]
        reset = {
            "schema": (
                "ora.robodojo-reset.v1" if index == 0
                else "ora.robodojo-round-reset.v1"
            ),
            "task": plan["task"], "eval_seed": 0, "seed": 0, "layout_id": 0,
            "round_index": index, "reward_registered": True,
            "physics_steps_before_policy": clock,
            "source_manifest": {
                "path": "source-manifest.json", "sha256": source_hash,
            },
        }
        if index:
            reset["artifact_sequences_preserved"] = {
                "observation_sequence": sequence,
                "action_sequence": sequence,
                "motion_probe_sequence": 0,
            }
            reset["round_isolation"] = _round_isolation(clock)
            _write(native / "round-resets" / f"round-{index:04d}.json", reset)
        events.append({"type": "reset", "record": reset, "round_index": index})
        crossing = perturbation == "receipts_cross_rounds"
        carried_native = 2 * index if crossing else 0
        carried_physics = 20 * index if crossing else 0
        base_clock = (
            reset_clocks[0]
            if crossing
            or (perturbation == "wrong_round_clock" and index == rounds - 1)
            else clock
        )
        for step in range(2):
            action_id = f"round-{index:04d}-action-{step:04d}"
            journal_path = native / "controller" / f"action-{sequence:04d}.jsonl"
            journal_path.parent.mkdir(parents=True, exist_ok=True)
            journal_path.write_text("".join(
                json.dumps({
                    "action_id": action_id, "segment": "command",
                    "physics_clock": (
                        base_clock + carried_physics + step * 10 + offset + 1
                    ),
                    "native_action": carried_native + step + 1,
                }, sort_keys=True) + "\n"
                for offset in range(10)
            ))
            control_path = native / "controller" / f"action-{sequence:04d}.json"
            _write(control_path, {
                "schema": "ora.robodojo-controller-trace.v1",
                "action_id": action_id,
                "samples": [{
                    "native_action": carried_native + step + 1,
                    "physics_steps": carried_physics + (step + 1) * 10,
                }],
                "physics_journal": {
                    "path": journal_path.relative_to(native).as_posix(),
                    "sha256": _sha(journal_path),
                },
            })
            receipt = {
                "schema": "ora.robodojo-sparse-motion-receipt.v1",
                "action_id": action_id, "action": "move_pose",
                "round_index": index, "consumed": True, "state_changed": True,
                "native_actions": 1, "physics_steps": 10,
                "control_trace": {
                    "path": control_path.relative_to(native).as_posix(),
                    "sha256": _sha(control_path),
                },
            }
            receipt_path = native / "receipts" / f"action-{sequence:04d}.json"
            _write(receipt_path, receipt)
            events.append({
                "type": "action_submitted", "action": {"action_id": action_id},
            })
            events.append({"type": "motion_receipt", "record": {
                "path": receipt_path.relative_to(native).as_posix(),
                "sha256": _sha(receipt_path), "receipt": receipt,
            }})
            sequence += 1
        verdict = {
            "schema": "ora.robodojo-official-verdict.v2",
            "source": "RoboDojo EvalEnv.is_episode_end",
            "source_manifest_sha256": source_hash,
            "task": plan["task"], "eval_seed": 0, "seed": 0, "layout_id": 0,
            "round_index": index, "official_success": index == 0,
            "terminal": True, "abort_reason": None,
            "verdict_basis": "official_terminal_transition", "fault_reason": None,
            "native_actions": 2, "physics_steps": 20,
            "reset_physics_steps": clock,
        }
        if perturbation == "verdict_native_count_mismatch" and index == 1:
            verdict["native_actions"] = 3
        if perturbation in {"budget_round", "faulted_round_claims_boolean"} and index == 1:
            # Round 1 ran out of robot decisions: no terminal transition, but
            # the standing official predicate is still a public Boolean.
            verdict.update({
                "official_success": False, "terminal": False,
                "abort_reason": "robot_decision_budget",
                "verdict_basis": "standing_official_predicate_at_policy_budget",
            })
        if perturbation == "faulted_round_claims_boolean" and index == 1:
            verdict["fault_reason"] = "robodojo_native_execution_failed"
        _write(
            native / "rounds" / f"round-{index:04d}" / "official-verdict.json",
            verdict,
        )
        if index == rounds - 1:
            _write(native / "official-verdict.json", verdict)
        authority = (
            "experiment_budget" if verdict["verdict_basis"].startswith("standing")
            else "environment"
        )
        if perturbation == "faulted_authority_claims_boolean" and index == 1:
            authority = "infrastructure"
        round_records.append({
            "schema": "ora.round-result.v1",
            "round_index": index, "state": "completed",
            "termination_authority": authority,
            "official_success": verdict["official_success"], "verdict": verdict,
        })
        events.append({"type": "round_finished", "round_index": index})
    if perturbation == "missing_archived_verdict":
        (native / "rounds/round-0001/official-verdict.json").unlink()
    result = {
        "schema": "ora.session-result.v1",
        "state": "completed", "official_success": None, "verdict": None,
        "termination": "session_round_limit", "rounds": round_records,
        "round_count": rounds,
        "round_success_count": sum(
            record["official_success"] is True for record in round_records
        ),
    }
    return run_root, plan, result, events


@pytest.mark.parametrize(("perturbation", "expected"), [
    (None, []),
    ("budget_round", []),
    ("faulted_round_claims_boolean", ["missing_trusted_boolean_verdict"]),
    ("faulted_authority_claims_boolean", ["missing_trusted_boolean_verdict"]),
    ("wrong_round_clock", [
        "robodojo_motion_receipt_binding_invalid",
        "robodojo_official_counter_mismatch:2",
    ]),
    ("missing_archived_verdict", ["robodojo_official_verdict_missing:1"]),
    ("verdict_native_count_mismatch", [
        "robodojo_official_counter_mismatch:1",
    ]),
    ("receipts_cross_rounds", [
        "robodojo_motion_receipt_binding_invalid",
        "robodojo_official_counter_mismatch:1",
        "robodojo_official_counter_mismatch:2",
    ]),
])
def test_persistent_rounds_join_each_round_to_its_own_reset_and_verdict(
    tmp_path: Path, perturbation: str | None, expected: list[str],
) -> None:
    """A correct K3 session must evaluate clean, and each break must localize.

    Before the per-round join, the valid tree below produced a receipt
    binding failure for every round after the first and a counter mismatch
    for the whole session, because the journal clock was compared against
    round 0's reset and the verdict against the sum of all receipts.
    """

    run_root, plan, result, events = _multi_round_evidence(
        tmp_path, perturbation=perturbation,
    )
    errors = _robodojo_evidence_errors(run_root, plan, result, events)
    assert sorted(set(errors)) == expected


def test_persistent_round_reset_records_must_match_their_archived_file(
    tmp_path: Path,
) -> None:
    run_root, plan, result, events = _multi_round_evidence(tmp_path)
    archived = run_root / "native/round-resets/round-0002.json"
    record = json.loads(archived.read_text())
    record["artifact_sequences_preserved"]["action_sequence"] = 0
    _write(archived, record)
    assert "robodojo_round_reset_invalid:2" in _robodojo_evidence_errors(
        run_root, plan, result, events,
    )


@pytest.mark.parametrize("mutate", [
    pytest.param(lambda record: record.pop("round_isolation"), id="absent"),
    pytest.param(
        lambda record: record["round_isolation"].update(passed=False),
        id="did_not_pass",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["failures"].append("leak"),
        id="carries_failures",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["checks"].update(
            arm_home_restored=False,
        ),
        id="check_did_not_hold",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["checks"].pop(
            "carried_channels_cleared",
        ),
        id="check_not_published",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["measured"].update(
            stale_channels=["last_receipt"],
        ),
        id="stale_channel",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["measured"].update(
            layout_sha256="b" * 64,
        ),
        id="layout_drift",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["measured"].update(
            step_lim=100,
        ),
        id="official_limit_drift",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["measured"].update(
            reward_registered=False,
        ),
        id="evaluator_not_rebuilt",
    ),
    pytest.param(
        lambda record: record["round_isolation"]["measured"].update(
            physics_steps_before_policy=1,
        ),
        id="clock_disagrees_with_the_record",
    ),
])
def test_a_round_that_does_not_prove_its_restore_is_refused(
    tmp_path: Path, mutate,
) -> None:
    """The evidence join refuses a round whose reset record proves nothing.

    RoboDojo publishes a measured restore proof for every round, so an
    absent, failing, incomplete or drifting block localizes to its round
    rather than silently narrowing the isolation guarantee to a claim.
    """

    run_root, plan, result, events = _multi_round_evidence(tmp_path)
    archived = run_root / "native/round-resets/round-0002.json"
    record = json.loads(archived.read_text())
    mutate(record)
    _write(archived, record)
    for event in events:
        if event.get("type") == "reset" and event.get("round_index") == 2:
            event["record"] = record

    assert "robodojo_round_isolation_invalid:2" in _robodojo_evidence_errors(
        run_root, plan, result, events,
    )


@pytest.mark.parametrize(("verdicts", "complete"), [
    ([True, False, True], True),
    ([False, False, False], True),
    ([True, None, False], False),
    ([True, False], False),
])
def test_persistent_session_completion_reads_the_round_verdicts(
    verdicts: list[object], complete: bool,
) -> None:
    """A K-round session keeps its aggregate verdict null by contract.

    Judging RoboDojo completion by that null aggregate marked every finished
    K-round session incomplete, so the rule is applied to the rounds that
    actually carry an official Boolean.
    """

    rounds = [{"official_success": value} for value in verdicts]
    assert coordinator.robodojo_official_complete(
        None, rounds, multi_round=True, round_limit=3,
    ) is complete


@pytest.mark.parametrize(("official", "complete"), [
    (True, True), (False, True), (None, False), (1, False),
])
def test_single_episode_completion_still_reads_the_top_level_verdict(
    official: object, complete: bool,
) -> None:
    assert coordinator.robodojo_official_complete(
        official, [], multi_round=False, round_limit=1,
    ) is complete


def _skill_round_evidence(tmp_path, *, legacy):
    """Two real journal leaves belong to one public skill decision."""
    import copy
    from ora.contracts import MOTION_ACTION_V3_SCHEMA, target_state_phases

    run_root, plan, result, events = _multi_round_evidence(tmp_path)
    native = run_root / "native"
    submitted = [event for event in events if event["type"] == "action_submitted"][:2]
    motions = [event for event in events if event["type"] == "motion_receipt"][:2]
    parent = {"schema": "ora-skill-sequence-action.v1", "action_id": "skill-0",
        "action": "execute_skill_sequence", "observation_id": "obs-0", "bound_observation_sha256": "a" * 64,
        "arguments": {"skill": "grasp_execute", "candidate_id": "grasp-0", "rationale": "test",
                      "sequence": ["approach", "close", "verify"], "resume_from": 0, "stop_before_commit": False}}
    stages = []
    for index, (name, event) in enumerate(zip(("approach", "close"), motions)):
        child = {"schema": MOTION_ACTION_V3_SCHEMA, "action_id": f"skill-0:{index:02d}-{name}",
            "action": "move_pose" if index == 0 else "set_gripper", "observation_id": "obs-0",
            "bound_observation_sha256": "a" * 64,
            "arguments": {"targets": {"right": {"position_m": [.2, .1, .9], "orientation_xyzw": [0., 0., 0., 1.]}}}
                         if index == 0 else {"targets": {"right": "close"}}}
        target = None
        if legacy:
            target = {**child, "schema": "ora-action.v4", "action": "target_state",
                "arguments": {"arm": "right", **({"pose": child["arguments"]["targets"]["right"]} if index == 0
                                                  else {"gripper": 0.0})}}
            child = target_state_phases(target)[0]
        wrapper = copy.deepcopy(event["record"])
        leaf = wrapper["receipt"]
        leaf.update(action_id=child["action_id"], action=child["action"], arguments=copy.deepcopy(child["arguments"]),
                    execution_status="reached")
        control_path = native / leaf["control_trace"]["path"]
        control = json.loads(control_path.read_text())
        control["action_id"] = child["action_id"]
        journal_path = native / control["physics_journal"]["path"]
        rows = [json.loads(line) for line in journal_path.read_text().splitlines()]
        journal_path.write_text("".join(json.dumps({**row, "action_id": child["action_id"]}) + "\n" for row in rows))
        control["physics_journal"]["sha256"] = _sha(journal_path)
        _write(control_path, control)
        leaf["control_trace"]["sha256"] = _sha(control_path)
        receipt_path = native / wrapper["path"]
        _write(receipt_path, leaf)
        wrapper["sha256"] = _sha(receipt_path)
        stage = {"stage": name, "status": "executed", "action_id": f"skill-0:{index:02d}-{name}"}
        if legacy:
            stage["receipt"] = {"schema": "ora.composite-motion-receipt.v1",
                "action_id": target["action_id"], "action": target, "consumed": True,
                "phase_receipts": [{"action": child, "status": "executed", "receipt": wrapper}]}
        else:
            stage.update(action=child, receipt=wrapper)
        stages.append(stage)
    stages.append({"stage": "verify", "status": "executed", "action_id": None})
    submitted[0]["action"] = parent
    motions[0]["record"] = {"schema": "ora.manipulation-skill-sequence-receipt.v1",
        "action_id": parent["action_id"], "action": parent, "consumed": True,
        "committed_sequence": parent["arguments"]["sequence"], "dispatched_stage_actions": 2,
        "status": "completed", "stage_receipts": stages}
    events = [event for event in events if event is not submitted[1] and event is not motions[1]]
    return run_root, plan, result, events, stages


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("damage", [None, "hash", "child_action", "child_observation", "order", "missing", "arguments"])
def test_dojo_evaluation_joins_skill_children_to_native_journals(tmp_path, legacy, damage):
    run_root, plan, result, events, stages = _skill_round_evidence(tmp_path, legacy=legacy)
    stage = stages[1]
    wrapper = stage["receipt"]["phase_receipts"][0]["receipt"] if legacy else stage["receipt"]
    action = stage["receipt"]["action"] if legacy else stage["action"]
    if damage == "hash": wrapper["sha256"] = "0" * 64
    elif damage == "child_action": action["action_id"] = "uncommitted-action"
    elif damage == "child_observation": action["bound_observation_sha256"] = "b" * 64
    elif damage == "order": stages.reverse()
    elif damage == "missing": stage.pop("receipt")
    elif damage == "arguments": wrapper["receipt"]["arguments"] = {"targets": {"right": "open"}}
    errors = _robodojo_evidence_errors(run_root, plan, result, events)
    if damage is None:
        assert errors == []
    else:
        assert "robodojo_motion_receipt_binding_invalid" in errors
