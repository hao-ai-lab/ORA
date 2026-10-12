"""The shared ``wait`` action on LIBERO (``shared_hold_wait_action_v1``, tool ``wait_ticks``).

LIBERO drives the Panda through robosuite OSC_POSE in delta mode, which keeps
no pose target between control steps: each step's goal is the pose measured at
its start plus the commanded delta. The hold is therefore the zero-delta
command -- the contract's input for a zero pose delta (all zeros for the
pinned symmetric ranges), and a zero gripper dimension that leaves the
Panda gripper's stateful directive unchanged -- which is also what a pose
action sends in the gripper dimension.

One LIBERO tick is one control step: one ``env.step``, which robosuite runs as
``control_timestep / model_timestep`` = 0.05 s / 0.002 s = 25 physics steps
(control_freq 20 in the LIBERO BDDL environments). It is the unit of
``max_sim_steps`` and of the environment's own horizon, and it is read from
the live worker, not assumed.

The executor enters from the unpinned execution-capture ``execute`` wrapper:
the session files that compose and persist receipts are hash-pinned by the
current Franka descriptors, and they stay byte-identical.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext, resolve_receipt_effect_projector,
)
from ora.contracts.wait import (
    WAIT_PHYSICS_STEPS_PER_TICK, WAIT_RECEIPT_SCHEMA, wait_receipt_block,
)
from ora.environment.backends.libero_pro.session import LiberoProError, LiberoProSession
from ora.environment.backends.libero_pro.wait import execute_wait, is_wait
from ora.environment.controller_replay import ControllerReplayRecorder
from ora.environment.controllers.libero_osc import (
    inverse_controller_scale, libero_osc_action_contract,
)
from ora.environment.controllers.libero_wait import (
    LIBERO_WAIT_EXECUTOR, libero_hold_command, libero_wait_receipt_issues,
)
from ora.environment.definitions import WAIT_ENVIRONMENTS, resolve_runtime_descriptor

ROOT = Path(__file__).resolve().parents[1]
#: The pinned robosuite OSC_POSE ranges (tests/test_control_plane_preflight.py).
CONTRACT = libero_osc_action_contract({
    "control_delta": True, "input_min": -1, "input_max": 1,
    "output_min": [-0.05] * 3 + [-0.5] * 3, "output_max": [0.05] * 3 + [0.5] * 3,
})
CONTRACT_SHA256 = CONTRACT["contract_sha256"]
OSC_CONTRACT_SCHEMA = "ora.libero-osc-action-contract.v2"


class _Plant:
    """The isolated worker's LIBERO environment port; the session code stays real."""

    def __init__(self, *, terminal_at=None, drift_m=0.0, timing=(0.002, 25), directive_step=None):
        self.commands: list[np.ndarray] = []
        self.position = np.array([0.1, 0.0, 1.0])
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.gripper = 0.4
        self.directive = -1.0
        self.terminal_at = terminal_at
        self.drift_m = drift_m
        self.timing = {"physics_dt_s": timing[0], "native_quantum_physics_steps": timing[1]}
        self.directive_step = directive_step
        self.env = self

    def step(self, command):
        self.commands.append(np.asarray(command).copy())
        self.position = self.position + np.array([0.0, 0.0, -self.drift_m])
        if self.directive_step == len(self.commands):
            self.directive = 1.0
        terminal = self.terminal_at == len(self.commands)
        return [{}], np.zeros(1), np.asarray([terminal]), np.asarray([False]), {}

    def get_env_attr(self, name):
        if name == "harness_servo_timing":
            return [dict(self.timing)]
        if name == "harness_gripper_hold_directive":
            return [self.directive]
        raise AttributeError(name)


def _session(tmp_path: Path, plant: _Plant, *, codec="motion_action_v3", max_sim_steps=100,
             contract=CONTRACT):
    session = object.__new__(LiberoProSession)
    session.execution_capture = {"enabled": False}
    session.environment = plant
    session.output_root = tmp_path
    session.action_codec = codec
    session.motion_profile = json.loads((ROOT / (
        "ora/policy/embodiments/franka-panda/motion-profile-v2.json")).read_text())
    session.controller_action_contract = contract
    session.workspace_bounds_m = {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
    session.max_sim_steps = max_sim_steps
    session.servo_profile = None
    session.robustness_episode = None
    session.action_sequence = session.sim_steps = session.round_index = 0
    session.terminated = session.truncated = False
    session.first_termination = None
    session.abort_reason = session.control_fault_reason = None
    session.world_mutation_sequence = 0
    session.world_mutation_actions = []
    session.command_drift_tracker = None
    session.last_receipt = None
    session.replay_recorder = ControllerReplayRecorder(tmp_path, "libero_pro", {"revision": "r"}, ["agentview"])
    session._current_state = lambda: (plant.position.copy(), plant.orientation.copy(), plant.gripper)
    return session


def _action(ticks=3, *, action="wait", arguments=None):
    return {"schema": "ora-motion-action.v3", "observation_id": "obs-0", "action_id": "act-0",
            "bound_observation_sha256": "0" * 64, "action": action,
            "arguments": {"ticks": ticks} if arguments is None else arguments}


def test_one_tick_is_one_control_step_of_twenty_five_physics_steps(tmp_path):
    plant = _Plant(drift_m=0.001)
    session = _session(tmp_path, plant)
    receipt = execute_wait(session, _action(3))
    # Three env.step calls, each the zero-delta 1x7 OSC command (all zeros under the
    # pinned symmetric contract); nothing else written.
    assert len(plant.commands) == 3
    assert all(command.shape == (1, 7) and command.dtype == np.float32 and not command.any()
               for command in plant.commands)
    assert session.sim_steps == receipt["steps_used"] == 3
    wait = receipt["wait"]
    assert wait["schema"] == WAIT_RECEIPT_SCHEMA and wait["tick_unit"] == "control_step"
    assert (wait["ticks_requested"], wait["ticks_used"], wait["physics_steps"]) == (3, 3, 75)
    assert wait["physics_steps_per_tick"] == 25
    assert wait["tick_period_s"] == pytest.approx(0.05)
    assert wait["terminated_early"] is False
    assert receipt["physics_steps"] == 75
    assert receipt["schema"] == "ora.libero-sparse-motion-receipt.v1"
    assert receipt["action"] == "wait" and receipt["arguments"] == {"ticks": 3}
    assert receipt["execution_status"] == "reached" and receipt["failure_code"] is None
    assert receipt["executor"] == [LIBERO_WAIT_EXECUTOR]
    assert receipt["target_arms"] == [] and receipt["synchronization"]["mode"] == "hold"
    invariant = receipt["command_invariant"]
    assert invariant["controlled_dimension"] == "none"
    assert invariant["held_dimension"] == "eef_pose_target_and_gripper_directive"
    assert invariant["directive_before"] == invariant["directive_after"] == -1.0
    assert invariant["verified"] is True
    right = receipt["arms"]["right"]
    assert right["targeted"] is False and right["pose_error"] is None
    assert right["inactive_arm_drift"]["translation_m"] == pytest.approx(0.003)
    assert receipt["control_health"]["status"] == "healthy"
    assert receipt["control_health"]["controller_action_contract_sha256"] == CONTRACT_SHA256
    assert libero_wait_receipt_issues(receipt) == []


def test_the_hold_is_a_zero_physical_delta_under_the_bound_contract(tmp_path):
    """Asymmetric ranges: a literal zero input would move the arm; the hold does not."""
    skewed = libero_osc_action_contract({
        "control_delta": True, "input_min": -1, "input_max": 1,
        "output_min": [-0.04] * 3 + [-0.5] * 3, "output_max": [0.06] * 3 + [0.5] * 3,
    })
    hold = libero_hold_command(skewed)
    assert hold[:3] == pytest.approx([-0.2] * 3) and hold[3:] == [0.0] * 4
    assert np.allclose(inverse_controller_scale([0.0] * 6, skewed), hold[:6])
    plant = _Plant()
    receipt = execute_wait(_session(tmp_path, plant, contract=skewed), _action(2))
    issued = np.asarray(hold, dtype=np.float32)
    assert all(np.array_equal(command[0], issued) for command in plant.commands)
    assert receipt["command_invariant"]["hold_command"] == issued.astype(float).tolist()
    assert receipt["command_invariant"]["verified"] is True and libero_wait_receipt_issues(receipt) == []
    with pytest.raises(LiberoProError, match="contract"):
        execute_wait(_session(tmp_path / "bad", _Plant(), contract={"contract_sha256": "0" * 64}), _action(1))


def test_the_receipt_is_persisted_counted_and_replayable(tmp_path):
    session = _session(tmp_path, _Plant())
    receipt = execute_wait(session, _action(2))
    path = tmp_path / "receipts/action-0000.json"
    assert json.loads(path.read_text()) == receipt
    assert session.last_receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert session.action_sequence == 1
    assert session.world_mutation_actions == ["wait"] and session.world_mutation_sequence == 1
    entry = session.replay_recorder.entries[0]
    assert entry["controller"] == "libero_osc_pose" and entry["replayable"] is True


def test_a_wait_stops_at_the_official_termination(tmp_path):
    session = _session(tmp_path, _Plant(terminal_at=2))
    receipt = execute_wait(session, _action(5))
    assert receipt["steps_used"] == 2 and session.terminated is True
    assert receipt["wait"]["ticks_used"] == 2 and receipt["wait"]["terminated_early"] is True
    assert receipt["execution_status"] == "not_reached"
    assert receipt["failure_code"] == "episode_terminal_during_execution"
    assert receipt["terminal_after_action"] is True
    assert session.first_termination["controller"] == "OSC_POSE"


def test_a_wait_stops_at_the_harness_step_budget(tmp_path):
    session = _session(tmp_path, _Plant(), max_sim_steps=2)
    receipt = execute_wait(session, _action(5))
    assert receipt["steps_used"] == 2 and session.truncated is True
    assert receipt["wait"]["terminated_early"] is True and session.is_terminal


def test_a_changed_gripper_directive_is_a_control_fault(tmp_path):
    session = _session(tmp_path, _Plant(directive_step=1))
    receipt = execute_wait(session, _action(2))
    assert receipt["command_invariant"]["verified"] is False
    assert receipt["execution_status"] == "execution_fault"
    assert receipt["failure_code"] == "gripper_directive_changed_during_wait"
    assert receipt["control_health"]["status"] == "fault"
    assert session.control_fault_reason == "libero_control_health_fault"
    assert "recorded_command_invariant_violation" in libero_wait_receipt_issues(receipt)


@pytest.mark.parametrize("case", ["other_codec", "terminal", "timing", "not_wait"])
def test_refusals_happen_before_any_step(tmp_path, case):
    plant = _Plant(timing=(0.002, 20) if case == "timing" else (0.002, 25))
    session = _session(tmp_path, plant, codec="runtime_action_v3" if case == "other_codec" else "motion_action_v3")
    if case == "terminal":
        session.terminated = True
    payload = _action(arguments={"targets": {"right": "open"}}, action="set_gripper") if case == "not_wait" else _action(2)
    with pytest.raises((LiberoProError, ValueError), match={
        "other_codec": "motion_action_v3", "terminal": "terminal",
        "timing": "control period", "not_wait": "wait",
    }[case]):
        execute_wait(session, payload)
    assert plant.commands == [] and session.action_sequence == 0


def test_the_capture_wrapper_routes_only_wait(tmp_path, monkeypatch):
    session = _session(tmp_path, _Plant())
    routed = []
    monkeypatch.setattr(LiberoProSession, "_execute_without_capture",
                        lambda self, payload: routed.append(payload["action"]) or {"routed": True})
    assert session.execute(_action(1))["wait"]["ticks_used"] == 1
    move = _action(arguments={"targets": {"right": "open"}}, action="set_gripper")
    assert is_wait(_action(1)) and not is_wait(move)
    assert session.execute(move) == {"routed": True} and routed == ["set_gripper"]


def test_the_wait_block_reaches_the_policy_without_its_schema(tmp_path):
    receipt = execute_wait(_session(tmp_path, _Plant()), _action(4))
    projected = resolve_receipt_effect_projector(receipt["schema"])(receipt, ReceiptProjectionContext(
        last_action_id="act-0", last_action_public_name="wait_ticks", surface_supports=lambda _c: True))
    text = json.dumps(projected)
    assert '"ticks_used": 4' in text and '"physics_steps_per_tick": 25' in text
    assert '"tick_period_s": 0.05' in text and WAIT_RECEIPT_SCHEMA not in text
    assert '"action": "wait_ticks"' in text and '"target_arms": []' in text


def test_libero_advertises_wait_only_on_the_sparse_v3_codec():
    assert set(WAIT_ENVIRONMENTS) == {"libero_pro", "robodojo", "robotwin"}
    runtime = {"kind": "libero_pro", "transport": "local_rpc"}
    assert resolve_runtime_descriptor(runtime, action_codec="motion_action_v3").action_modes[-1] == "wait"
    assert "wait" not in resolve_runtime_descriptor(runtime, action_codec="runtime_action_v3").action_modes


def test_the_shared_wait_block_counts_ticks_in_the_backend_s_own_unit():
    default = wait_receipt_block(ticks_requested=3, physics_steps=30, tick_unit="ten_physics_steps",
                                 tick_period_s=0.04, terminated_early=False)
    assert default["physics_steps_per_tick"] == WAIT_PHYSICS_STEPS_PER_TICK == 10
    assert default["ticks_used"] == 3
    libero = wait_receipt_block(ticks_requested=3, physics_steps=50, tick_unit="control_step",
                                tick_period_s=0.05, terminated_early=True, physics_steps_per_tick=25)
    assert (libero["ticks_used"], libero["physics_steps_per_tick"]) == (2, 25)
    with pytest.raises(ValueError):
        wait_receipt_block(ticks_requested=1, physics_steps=25, tick_unit="control_step",
                           tick_period_s=0.05, terminated_early=False, physics_steps_per_tick=0)


def test_learning_eligibility_judges_a_wait_under_its_own_contract(tmp_path):
    from ora.orchestrator.evaluation_learning import _libero_receipt_reasons

    receipt = execute_wait(_session(tmp_path, _Plant()), _action(2))

    def reasons(record, sha=CONTRACT_SHA256):
        return _libero_receipt_reasons(0, record, 1, tmp_path, sha, OSC_CONTRACT_SCHEMA)

    assert reasons(receipt) == []
    assert reasons(receipt, sha="e" * 64) == ["missing_or_invalid_control_health:0"]
    broken = json.loads(json.dumps(receipt))
    broken["wait"]["ticks_used"] = 3
    assert reasons(broken) == ["invalid_wait_receipt:0"]
    faulted = execute_wait(_session(tmp_path / "fault", _Plant(directive_step=1)), _action(2))
    assert reasons(faulted) == ["control_health_fault:0", "invalid_wait_receipt:0"]


def test_the_control_audit_accepts_a_wait_and_names_a_violation(tmp_path):
    from ora.orchestrator.control_audit import _validate_sparse_receipt

    receipt = execute_wait(_session(tmp_path, _Plant()), _action(2))
    path = tmp_path / "receipts/action-0000.json"
    assert _validate_sparse_receipt(tmp_path, path, receipt) == []
    faulted = execute_wait(_session(tmp_path / "fault", _Plant(directive_step=1)), _action(2))
    issues = _validate_sparse_receipt(tmp_path, path, faulted)
    assert "recorded_command_invariant_violation" in issues and "recorded_control_health_fault" in issues
