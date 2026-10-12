"""The trusted stage loop: one decision in, one receipt out, never a wedge.

`execute_skill_sequence` is the half of a closed skill that no tool handler can
be. A tool call writes one action and returns, so the several Supervisor
executions a skill needs happen here, between the submitted action and the
receipt the policy reads on its next turn -- with no Agent turn in between.

Two properties are load-bearing and everything else is detail. A mutating stage
that did not happen must stop the sequence, because descending from an
unconfirmed approach produces a confident receipt about something that did not
occur. And the sequence must report **however** it ends -- refused, terminal,
interrupted, or with the backend raising -- because a skill that can end without
releasing its ledger is worse than the three explicit calls it replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from ora.contracts import (
    RUNTIME_ACTION_SCHEMA, SKILL_SEQUENCE_ACTION_SCHEMA,
)
from ora.environment import SyntheticSession
from ora.orchestrator.action_executor import (
    execute_runtime_action, execute_skill_sequence,
)


POSE = [0.3, 0.1, 0.9, 0.0, 0.0, 0.0, 1.0]


def _session(tmp_path: Path, *, max_actions: int = 8) -> SyntheticSession:
    session = SyntheticSession(
        output_root=tmp_path / "native", task="synthetic-0", seed=0,
        max_actions=max_actions,
    )
    session.start()
    return session


def _skill_action(*, skill="grasp_execute", sequence=None, resume_from=0,
                  stop_before_commit=False) -> dict[str, Any]:
    return {
        "schema": SKILL_SEQUENCE_ACTION_SCHEMA,
        "observation_id": "observation-0001",
        "action_id": "action-1",
        "action": "execute_skill_sequence",
        "arguments": {
            "skill": skill,
            "candidate_id": "grasp-abc123",
            "rationale": "the only candidate clear of the neighbouring mug",
            "stop_before_commit": stop_before_commit,
            "sequence": sequence or ["approach", "close", "verify"],
            "resume_from": resume_from,
        },
        "bound_observation_sha256": "a" * 64,
    }


class _Driver:
    """A minimal stand-in for the tool runtime's half of the protocol."""

    def __init__(self, *, plan: Mapping[str, Mapping[str, Any]] | None = None):
        self.plan = dict(plan or {})
        self.receipts: list[dict[str, Any]] = []
        self.seen: list[str] = []

    def drive(self, action, *, stage, stage_index, observation,
              observation_root, completed, capture_observation):
        del observation, observation_root, completed
        self.seen.append(stage)
        override = self.plan.get(stage)
        if override is not None:
            if override.get("kind") == "measurement_after_capture":
                capture_observation()
                return {"kind": "measurement", **{
                    key: value for key, value in override.items() if key != "kind"
                }}
            return dict(override)
        if stage in {"stop", "verify"}:
            capture_observation()
            return {
                "kind": "measurement",
                "status": "stopped" if stage == "stop" else "executed",
                "reason": f"{stage}_measured",
                "measurements": [{"quantity": stage, "value": 1.0}],
                **({"verification": {"strength": "strong_negative"}}
                   if stage == "verify" else {}),
            }
        arm_action = (
            {"arm": "right", "gripper": 0.0} if stage in {"close", "release"}
            else {
                "arm": "right", "position_m": POSE[:3],
                "orientation_xyzw": POSE[3:],
            }
        )
        return {
            "kind": "mutation",
            "action": {
                "schema": RUNTIME_ACTION_SCHEMA,
                "observation_id": action["observation_id"],
                "action_id": f"{action['action_id']}:{stage_index:02d}-{stage}",
                "action": (
                    "set_gripper" if stage in {"close", "release"} else "move_pose"
                ),
                "arguments": arm_action,
                "bound_observation_sha256": action["bound_observation_sha256"],
            },
        }

    def build(self, *, action, status, stages, observation, stop_reason):
        record = {
            "schema": "ora.skill-receipt.v1",
            "skill": action["arguments"]["skill"],
            "status": status,
            "stop_reason": stop_reason,
            "stages": [dict(item) for item in stages],
            "observation_id": observation.get("observation_id"),
        }
        self.receipts.append(record)
        return record


def _run(session, action, driver, **kwargs):
    observation, _images = session.observe(session.output_root / "observations")

    def capture():
        fresh, _ = session.observe(session.output_root / "observations")
        return fresh, session.output_root / "observations" / fresh["observation_id"]

    return execute_skill_sequence(
        session, action,
        initial_observation=observation,
        initial_observation_root=(
            session.output_root / "observations" / observation["observation_id"]
        ),
        drive_stage=driver.drive, build_receipt=driver.build,
        capture_observation=capture, **kwargs,
    )


def test_a_grasp_runs_its_whole_sequence_as_one_coordinator_execution(tmp_path):
    """Two dispatched stage actions, one robot decision, one receipt."""

    session = _session(tmp_path)
    driver = _Driver()

    receipt, (observation, _root) = _run(session, _skill_action(), driver)

    assert driver.seen == ["approach", "close", "verify"]
    assert receipt["status"] == "completed"
    assert receipt["robot_decisions"] == 1
    assert receipt["dispatched_stage_actions"] == 2
    assert receipt["action_id"] == "action-1"
    assert [item["stage"] for item in receipt["stage_receipts"]] == [
        "approach", "close", "verify",
    ]
    assert receipt["fresh_observation_id"] == observation["observation_id"]
    assert len(driver.receipts) == 1


def test_a_verifying_stage_hands_its_evidence_to_the_receipt(tmp_path):
    """The verification block is the one part not reducible to measurements."""

    session = _session(tmp_path)
    driver = _Driver()

    receipt, _ = _run(session, _skill_action(), driver)

    verify = receipt["stage_receipts"][-1]
    assert verify["stage"] == "verify"
    assert verify["verification"] == {"strength": "strong_negative"}
    assert driver.receipts[0]["stages"][-1]["verification"] == {
        "strength": "strong_negative",
    }


def test_the_receipt_action_id_is_the_submitted_one_not_a_stage_id(tmp_path):
    """The AgentSession rejects a receipt that does not match its last action."""

    session = _session(tmp_path)
    receipt, _ = _run(session, _skill_action(), _Driver())

    assert receipt["action_id"] == "action-1"
    dispatched = [
        item["action_id"] for item in receipt["stage_receipts"]
        if item["action_id"] is not None
    ]
    assert dispatched == ["action-1:00-approach", "action-1:01-close"]


def test_a_pose_that_did_not_arrive_ends_the_sequence(tmp_path):
    """Closing from an unconfirmed approach would report a grasp that never was."""

    session = _session(tmp_path)
    driver = _Driver(plan={"approach": {
        "kind": "mutation",
        "action": {
            "schema": RUNTIME_ACTION_SCHEMA,
            "observation_id": "observation-0001",
            "action_id": "action-1:00-approach",
            "action": "move_pose",
            # SyntheticSession reaches whatever it is commanded, so the
            # not-reached path is exercised through missing pose evidence
            # rather than by pretending a synthetic arm missed.
            "arguments": {
                "arm": "right", "position_m": POSE[:3],
                "orientation_xyzw": POSE[3:],
            },
            "bound_observation_sha256": "a" * 64,
        },
    }})
    original = session.execute

    def blind(payload):
        record = original(payload)
        record["receipt"].pop("commanded_pose_xyz_xyzw", None)
        return record

    session.execute = blind  # type: ignore[method-assign]

    receipt, _ = _run(session, _skill_action(), driver)

    assert driver.seen == ["approach"]
    assert receipt["status"] == "aborted"
    assert receipt["stop_reason"] == "approach_not_reached"
    assert receipt["stage_receipts"][0]["status"] == "not_reached"
    assert len(driver.receipts) == 1


def test_a_refused_stage_still_reports_and_releases(tmp_path):
    session = _session(tmp_path)
    driver = _Driver(plan={"close": {
        "kind": "refusal", "status": "refused",
        "reason": "candidate_observation_is_no_longer_current",
    }})

    receipt, _ = _run(session, _skill_action(), driver)

    assert receipt["status"] == "aborted"
    assert receipt["stop_reason"] == "candidate_observation_is_no_longer_current"
    assert receipt["stage_receipts"][-1]["status"] == "refused"
    assert driver.receipts[-1]["status"] == "aborted"


def test_a_terminal_environment_stops_the_sequence_and_reports(tmp_path):
    session = _session(tmp_path, max_actions=1)
    driver = _Driver()

    receipt, _ = _run(session, _skill_action(), driver)

    assert receipt["status"] == "terminated"
    assert receipt["stop_reason"] == "environment_terminal"
    assert receipt["stage_receipts"][-1]["status"] == "terminated"
    assert len(driver.receipts) == 1


def test_a_raised_backend_fault_reports_before_it_propagates(tmp_path):
    """The ledger must release even when the sequence cannot finish."""

    session = _session(tmp_path)
    driver = _Driver()

    def fault(payload):
        raise RuntimeError("simulator step budget is exhausted")

    session.execute = fault  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        _run(session, _skill_action(), driver)

    assert len(driver.receipts) == 1
    assert driver.receipts[0]["status"] == "aborted"
    assert driver.receipts[0]["stop_reason"] == "trusted_execution_fault"
    assert driver.receipts[0]["stages"][-1]["status"] == "unavailable"


def test_a_raised_driver_fault_reports_before_it_propagates(tmp_path):
    """A driver is expected to return its refusals; a raise is still a fault."""

    session = _session(tmp_path)
    driver = _Driver()

    def fault(*args, **kwargs):
        raise RuntimeError("the committed geometry could not be read")

    driver.drive = fault  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        _run(session, _skill_action(), driver)

    assert len(driver.receipts) == 1
    assert driver.receipts[0]["stop_reason"] == "stage_driver_fault"


def test_an_operator_interrupt_stops_the_sequence_and_reports(tmp_path):
    session = _session(tmp_path)
    driver = _Driver()

    receipt, _ = _run(
        session, _skill_action(), driver, should_stop=lambda: True,
    )

    assert receipt["status"] == "aborted"
    assert receipt["stop_reason"] == "operator_interrupt"
    assert driver.seen == []
    assert len(driver.receipts) == 1


def test_the_place_stop_ends_the_sequence_with_the_remainder_uncommitted(tmp_path):
    session = _session(tmp_path)
    driver = _Driver()
    action = _skill_action(
        skill="place_execute", stop_before_commit=True,
        sequence=["transport", "descend", "stop", "release", "settle", "verify"],
    )

    receipt, _ = _run(session, action, driver)

    assert driver.seen == ["transport", "descend", "stop"]
    assert receipt["status"] == "stopped"
    assert receipt["stop_reason"] == "stopped_for_close_view_before_release"
    assert receipt["dispatched_stage_actions"] == 2


def test_a_resume_skips_the_completed_prefix_without_re_executing_it(tmp_path):
    session = _session(tmp_path)
    driver = _Driver()
    action = _skill_action(
        skill="place_execute", stop_before_commit=True, resume_from=3,
        sequence=["transport", "descend", "stop", "release", "settle", "verify"],
    )

    receipt, _ = _run(session, action, driver)

    assert driver.seen == ["release", "settle", "verify"]
    assert receipt["resumed_from"] == 3
    assert receipt["status"] == "completed"


def test_a_stage_action_that_drifted_from_the_commitment_is_refused(tmp_path):
    """A driven stage may not silently retarget a different observation."""

    session = _session(tmp_path)
    driver = _Driver(plan={"approach": {
        "kind": "mutation",
        "action": {
            "schema": RUNTIME_ACTION_SCHEMA,
            "observation_id": "observation-9999",
            "action_id": "action-1:00-approach",
            "action": "move_pose",
            "arguments": {
                "arm": "right", "position_m": POSE[:3],
                "orientation_xyzw": POSE[3:],
            },
            "bound_observation_sha256": "a" * 64,
        },
    }})

    with pytest.raises(ValueError, match="drifted from its committed action"):
        _run(session, _skill_action(), driver)


def test_the_plain_executor_refuses_a_skill_sequence(tmp_path):
    """It names the owner of the loop instead of executing half of one."""

    session = _session(tmp_path)

    with pytest.raises(ValueError, match="coordinator-owned stage loop"):
        execute_runtime_action(session, _skill_action())


@pytest.mark.parametrize('arrives', [True, False])
def test_target_state_stage_arrival_controls_the_following_close(tmp_path, arrives):
    """Real driver envelopes must stop at a failed composite pose phase."""
    from ora.contracts import TARGET_STATE_ACTION_SCHEMA

    class CompositeDriver(_Driver):
        def drive(self, *args, **kwargs):
            driven = super().drive(*args, **kwargs)
            if driven['kind'] == 'mutation':
                action = driven['action']
                old = action['arguments']
                action.update(schema=TARGET_STATE_ACTION_SCHEMA, action='target_state', arguments={
                    'arm': old['arm'], **({'gripper': old['gripper']} if 'gripper' in old else {
                        'pose': {'position_m': old['position_m'], 'orientation_xyzw': old['orientation_xyzw']}}),
                })
            return driven

    session = _session(tmp_path)
    original = session.execute
    dispatched = []

    def measured(action):
        dispatched.append(action['action'])
        receipt = original(action)
        if action['action'] == 'move_pose' and not arrives:
            receipt['receipt']['reached_pose_xyz_xyzw'] = list(receipt['receipt']['reached_pose_xyz_xyzw'])
            receipt['receipt']['reached_pose_xyz_xyzw'][2] += 0.026691
        return receipt

    session.execute = measured
    receipt, _ = _run(session, _skill_action(), CompositeDriver())
    assert dispatched == (['move_pose', 'set_gripper'] if arrives else ['move_pose'])
    assert receipt['status'] == ('completed' if arrives else 'aborted')
    assert receipt['skill_receipt']['stages'][0]['status'] == ('executed' if arrives else 'not_reached')
    if not arrives:
        assert receipt['stop_reason'] == 'approach_not_reached'
        from ora.agent.native.receipt_projection import (
            resolve_receipt_effect_projector, ReceiptProjectionContext,
        )
        effect = resolve_receipt_effect_projector(receipt['schema'])(receipt,
            ReceiptProjectionContext(last_action_id=receipt['action_id'],
                                     last_action_public_name='grasp_execute'))
        assert effect['status'] == 'aborted'
        assert effect['failure_stage'] == 'approach'
        assert effect['detail']['stages'][0]['status'] == 'not_reached'
        assert effect['detail']['stages_not_reached'] == ['close', 'verify']


@pytest.mark.parametrize('status', ['reached', 'not_reached', 'planner_failed', 'plan_rejected', 'execution_fault'])
@pytest.mark.parametrize('schema', [
    'ora.robotwin-sparse-motion-receipt.v7',
    'ora.libero-sparse-motion-receipt.v1',
])
def test_sparse_stage_uses_controller_execution_status(tmp_path, status, schema):
    from ora.contracts import MOTION_ACTION_V3_SCHEMA

    class SparseDriver(_Driver):
        def drive(self, *args, **kwargs):
            driven = super().drive(*args, **kwargs)
            if driven['kind'] == 'mutation':
                action = driven['action']
                pose = action['action'] == 'move_pose'
                action.update(schema=MOTION_ACTION_V3_SCHEMA, arguments={
                    'targets': {'right': {'position_m': POSE[:3], 'orientation_xyzw': POSE[3:]} if pose else 'close'},
                    **({'interactions': {'right': 'contact_approach'}} if pose else {}),
                })
            return driven

    session = _session(tmp_path)
    dispatched = []

    def execute(action):
        dispatched.append(action)
        return {'schema': schema,
                'action_id': action['action_id'], 'consumed': True,
                'arms': {'right': {'pose_error': {'position_m': .00314, 'orientation_rad': .026}}},
                'execution_status': status if action['action'] == 'move_pose' else 'executed'}

    session.execute = execute
    receipt, _ = _run(session, _skill_action(), SparseDriver())
    assert len(dispatched) == (2 if status == 'reached' else 1)
    assert receipt['status'] == ('completed' if status == 'reached' else 'aborted')
    assert receipt['stage_receipts'][0]['action'] == dispatched[0]
    measurement = receipt['stage_receipts'][0]['measurements'][-1]
    assert measurement['quantity'] == 'endpoint_residual'
    assert measurement['value']['errors_by_arm']['right']['position_m'] == .00314


@pytest.mark.parametrize(("operation", "status", "code", "ticks", "verified", "expected"), [
    ("close", "reached", None, 2, True, "completed"),
    ("close", "not_reached", "gripper_not_reached", 2, True, "completed"),
    ("open", "not_reached", "gripper_not_reached", 2, True, "aborted"),
    ("open", "reached", None, 2, True, "completed"),
    ("close", "not_reached", "gripper_not_reached", 0, True, "aborted"),
    ("close", "not_reached", "gripper_not_reached", 2, False, "aborted"),
    ("close", "not_reached", "episode_terminal_during_execution", 2, True, "aborted"),
    ("close", "not_reached", "gripper_stalled_before_target", 4, True, "completed"),
    ("open", "not_reached", "gripper_stalled_before_target", 4, True, "aborted"),
    ("close", "not_reached", "gripper_stalled_before_target", 4, False, "aborted"),
])
def test_dojo_sparse_skills_verify_applied_close_but_stop_before_exit_on_incomplete_open(
    tmp_path, operation, status, code, ticks, verified, expected,
):
    from ora.contracts import MOTION_ACTION_V3_SCHEMA

    class SparseDriver(_Driver):
        def drive(self, *args, **kwargs):
            driven = super().drive(*args, **kwargs)
            if driven["kind"] == "mutation":
                action = driven["action"]
                pose = action["action"] == "move_pose"
                action.update(schema=MOTION_ACTION_V3_SCHEMA, arguments={
                    "targets": {"right": {"position_m": POSE[:3], "orientation_xyzw": POSE[3:]} if pose else operation},
                    **({"interactions": {"right": "contact_approach"}} if pose else {}),
                })
            return driven

    session = _session(tmp_path)
    driver = SparseDriver()
    def execute(action):
        pose = action["action"] == "move_pose"
        return {"schema": "ora.robodojo-sparse-motion-receipt.v1",
            "action_id": action["action_id"], "consumed": True,
            "execution_status": "reached" if pose else status,
            "native_actions": ticks, "failure_code": None if pose else code,
            "command_invariant": {"verified": verified},
            "gripper_directives": {"right": float(operation == "open")}}
    session.execute = execute
    action = (_skill_action() if operation == "close" else _skill_action(skill="place_execute",
        sequence=["transport", "descend", "release", "settle", "verify"]))
    receipt, _ = _run(session, action, driver)
    assert receipt["status"] == expected
    if code == "gripper_stalled_before_target" and expected == "completed":
        close = next(stage for stage in receipt["stage_receipts"] if stage["stage"] == "close")
        assert close["status"] == "executed" and close["reason"] == code
    if expected == "completed":
        assert "verify" in driver.seen
    elif operation == "open":
        assert "settle" not in driver.seen
    else:
        assert "verify" not in driver.seen


@pytest.mark.parametrize("schema", [
    "ora.robotwin-sparse-motion-receipt.v7",
    "ora.libero-sparse-motion-receipt.v1",
    "ora.robodojo-sparse-motion-receipt.v1",
])
@pytest.mark.parametrize("terminal_stage", ["approach", "close", "verify"])
@pytest.mark.parametrize("terminal_reason", ["official_success", "step_budget_exhausted"])
def _dojo_close_stages(tmp_path, *, code, arms):
    from ora.contracts import MOTION_ACTION_V3_SCHEMA

    class SparseDriver(_Driver):
        def drive(self, *args, **kwargs):
            driven = super().drive(*args, **kwargs)
            if driven["kind"] == "mutation":
                action = driven["action"]
                pose = action["action"] == "move_pose"
                action.update(schema=MOTION_ACTION_V3_SCHEMA, arguments={
                    "targets": {"right": {"position_m": POSE[:3], "orientation_xyzw": POSE[3:]} if pose else "close"},
                    **({"interactions": {"right": "contact_approach"}} if pose else {}),
                })
            return driven

    session = _session(tmp_path)

    def execute(action):
        pose = action["action"] == "move_pose"
        return {"schema": "ora.robodojo-sparse-motion-receipt.v1",
                "action_id": action["action_id"], "consumed": True,
                "execution_status": "reached" if pose else "not_reached",
                "native_actions": 4, "failure_code": None if pose else code,
                "command_invariant": {"verified": True},
                "gripper_directives": {"right": 0.0}, "arms": {} if pose else arms}
    session.execute = execute
    receipt, _ = _run(session, _skill_action(), SparseDriver())
    return receipt["stage_receipts"]


def test_a_dojo_close_stage_carries_each_arm_gripper_outcome(tmp_path):
    outcome = {"state": "stalled_before_target", "commanded_opening_normalized": 0.0,
               "measured_opening_normalized": 0.5}
    stages = _dojo_close_stages(tmp_path, code="gripper_stalled_before_target",
                                arms={"right": {"gripper_outcome": outcome}, "left": {"targeted": False}})
    close = next(stage for stage in stages if stage["stage"] == "close")
    assert close["status"] == "executed" and close["reason"] == "gripper_stalled_before_target"
    assert close["measurements"][-1] == {"quantity": "gripper_outcome", "value": {"right": outcome}}


@pytest.mark.parametrize("schema", [
    "ora.robotwin-sparse-motion-receipt.v7",
    "ora.libero-sparse-motion-receipt.v1",
    "ora.robodojo-sparse-motion-receipt.v1",
])
@pytest.mark.parametrize("terminal_stage", ["approach", "close", "verify"])
@pytest.mark.parametrize("terminal_reason", ["official_success", "step_budget_exhausted"])
def test_trusted_terminal_ends_skill_before_arrival_or_next_stage(
    tmp_path, schema, terminal_stage, terminal_reason,
):
    """A world boundary is neither failed arrival nor sequence completion."""
    session = _session(tmp_path)
    driver = _Driver()
    terminal = False
    raw_receipts = []
    original_drive = driver.drive

    def drive(*args, **kwargs):
        nonlocal terminal
        result = original_drive(*args, **kwargs)
        if kwargs["stage"] == terminal_stage == "verify":
            terminal = True
        return result

    def execute(action):
        nonlocal terminal
        terminal = action["action_id"].endswith(terminal_stage)
        # Reproduce G003: executed, no motion failure, terminal before endpoint.
        receipt = {
            "schema": schema, "action_id": action["action_id"],
            "consumed": True, "state_changed": True,
            "execution_status": "executed" if terminal else "reached",
            "failure_code": None, "not_reached_reason": None,
            "terminal_after_action": terminal,
            "arms": {"right": {"pose_error": {"translation_m": .06, "orientation_rad": .2}}},
            "command_invariant": {"verified": True},
        }
        if action["action"] == "set_gripper" and not terminal:
            receipt["execution_status"] = "reached" if "robodojo" in schema else "executed"
        raw_receipts.append(dict(receipt))
        return receipt

    driver.drive = drive
    session.execute = execute
    session.status = lambda: {"terminal": terminal, "reason": terminal_reason}
    receipt, _ = _run(session, _skill_action(), driver)
    expected = ["approach", "close", "verify"][:["approach", "close", "verify"].index(terminal_stage) + 1]
    assert driver.seen == expected
    assert receipt["status"] == "terminated"
    assert receipt["terminal_after_action"] is True
    assert receipt["stop_reason"] == "environment_terminal"
    assert receipt["stage_receipts"][-1]["status"] == "terminated"
    assert receipt["skill_receipt"]["status"] == "terminated"
    assert len(driver.receipts) == 1
    for stage, raw in zip((s for s in receipt["stage_receipts"] if s.get("receipt")), raw_receipts):
        assert stage["receipt"] == raw
    from ora.agent.native.receipt_projection import (
        resolve_receipt_effect_projector, ReceiptProjectionContext,
    )
    effect = resolve_receipt_effect_projector(receipt["schema"])(
        receipt, ReceiptProjectionContext(last_action_id=receipt["action_id"],
                                         last_action_public_name="grasp_execute"))
    assert effect["status"] == "terminated"
    assert effect["round_terminated"] is True
    assert effect["failure_stage"] is None
    assert effect["failure_code"] is None
    assert effect["detail"]["resumable"] is False


def test_preterminal_skill_drives_nothing_and_is_not_resumable(tmp_path):
    session = _session(tmp_path)
    session.status = lambda: {"terminal": True, "reason": "official_success"}
    driver = _Driver()
    receipt, _ = _run(session, _skill_action(), driver)
    assert driver.seen == []
    assert receipt["status"] == "terminated"
    assert receipt["stage_receipts"][0]["status"] == "skipped"
    assert receipt["stage_receipts"][0]["reason"] == "environment_terminal"
    assert receipt["dispatched_stage_actions"] == 0
    assert len(driver.receipts) == 1


def test_terminated_skill_still_exposes_dispatched_controller_evidence(tmp_path):
    """Changing the envelope status must not hide a child from evaluation."""
    from ora.orchestrator.evaluation import _physical_motion_receipts
    session = _session(tmp_path, max_actions=1)
    receipt, _ = _run(session, _skill_action(), _Driver())
    assert receipt["status"] == "terminated"
    children = _physical_motion_receipts([{"type": "motion_receipt", "record": receipt}])
    assert children == [receipt["stage_receipts"][0]["receipt"]]
