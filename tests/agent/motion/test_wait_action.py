"""The shared ``wait`` action: contract, hold semantics and receipts (EXP11A g004, B5).

``wait`` names no arm, pose or gripper. Every applied command is held and
``ticks`` control ticks (10 physics steps each) elapse; RoboDojo counts them
as native actions, which is how an interactive scene's scripted demonstrator
advances without the robot moving (``imitate_sorting_sequence``).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import validate_schema_instance
from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext, resolve_receipt_effect_projector,
)
from ora.contracts import validate_runtime_action
from ora.contracts.action import ActionValidationError
from ora.contracts.wait import WAIT_MAX_TICKS, WAIT_RECEIPT_SCHEMA
from ora.environment.backends.robotwin.wait import WAIT_CONTROLLER, hold_current_targets
from ora.environment.definitions import WAIT_ENVIRONMENTS
from tests.environments.robodojo.test_robodojo_backend import fake_sdk_candidate_boundary as _fake_boundary
from tests.environments.robodojo.test_robodojo_backend import pose, sparse_session, v3

fake_sdk_candidate_boundary = pytest.fixture(autouse=True)(_fake_boundary.__wrapped__)

LIVE = ("robodojo_placement_closed_skills_v2", "robotwin_toolchain_b2_closed_noservo_v2",
        "libero_placement_direct_overhead_v1")


def _action(arguments):
    return {"schema": "ora-motion-action.v3", "observation_id": "o", "action_id": "a",
            "action": "wait", "arguments": arguments, "bound_observation_sha256": "0" * 64}


@pytest.mark.parametrize("ticks", [1, 7, WAIT_MAX_TICKS])
def test_the_contract_accepts_one_to_twenty_ticks(ticks):
    assert validate_runtime_action(_action({"ticks": ticks}))["arguments"] == {"ticks": ticks}


@pytest.mark.parametrize("arguments", [{"ticks": 0}, {"ticks": 21}, {"ticks": 1.5}, {"ticks": True},
                                       {}, {"ticks": 2, "targets": {"left": "open"}}, {"targets": {}}])
def test_the_contract_refuses_everything_else_before_any_write(arguments):
    with pytest.raises(ActionValidationError, match="wait"):
        validate_runtime_action(_action(arguments))


def _wait(session, tmp_path, ticks):
    return session.execute(v3(session, tmp_path, kind="wait", targets=None) | {"arguments": {"ticks": ticks}})


def test_robodojo_holds_every_command_and_advances_exactly_the_requested_native_actions(tmp_path):
    session = sparse_session(tmp_path)
    try:
        session.execute(v3(session, tmp_path, targets={"left": pose([0.05, 0.0, 0.8])}))
        session.execute(v3(session, tmp_path, kind="set_gripper", targets={"right": "close"}))
        plant = session.environment
        commands_before = json.dumps(plant.robot_manager.control_manager.prev_control[0], sort_keys=True)
        native_before, sent = plant.take_action_cnt[0], len(plant.commands)
        receipt = _wait(session, tmp_path, 5)["receipt"]
        assert plant.take_action_cnt[0] - native_before == 5  # a scripted scene advances 5 native actions
        assert plant.commands[sent:] == [{}] * 5  # no drive, gripper or limiter target was written
        assert json.dumps(plant.robot_manager.control_manager.prev_control[0], sort_keys=True) == commands_before
        assert receipt["execution_status"] == "reached" and receipt["failure_code"] is None
        assert receipt["native_actions"] == 5 and receipt["physics_steps"] == 50
        assert receipt["target_arms"] == [] and receipt["synchronization"]["mode"] == "hold"
        assert receipt["command_invariant"]["controlled_dimension"] == "none"
        assert receipt["command_invariant"]["verified"] is True
        assert receipt["gripper_directives"]["right"] == 0.0
        wait = receipt["wait"]
        assert wait["schema"] == WAIT_RECEIPT_SCHEMA and wait["tick_unit"] == "native_action"
        assert (wait["ticks_requested"], wait["ticks_used"], wait["physics_steps"]) == (5, 5, 50)
        assert wait["tick_period_s"] == pytest.approx(0.04) and wait["terminated_early"] is False
        assert all(receipt["arms"][arm]["inactive_arm_drift"] is not None for arm in ("left", "right"))
        projected = resolve_receipt_effect_projector(receipt["schema"])(receipt, ReceiptProjectionContext(
            last_action_id=receipt["action_id"], last_action_public_name="wait_ticks",
            surface_supports=lambda capability: True))
        text = json.dumps(projected)
        assert '"ticks_used": 5' in text and WAIT_RECEIPT_SCHEMA not in text  # schema stays Control-only
    finally:
        session.close()


def test_robodojo_wait_stops_at_an_official_terminal(tmp_path):
    session = sparse_session(tmp_path)
    try:
        session.environment.terminal_at = session.environment.take_action_cnt[0] + 3
        receipt = _wait(session, tmp_path, 10)["receipt"]
        assert receipt["native_actions"] == 3 and receipt["wait"]["terminated_early"] is True
        assert receipt["failure_code"] == "episode_terminal_during_execution"
    finally:
        session.close()


class _Scene:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    @staticmethod
    def get_timestep():
        return 0.004


def _robotwin(terminal_at=None):
    scene = _Scene()
    records = []
    session = SimpleNamespace(
        action_sequence=4,
        _after_physics_step=lambda environment, *, action, phase: (
            phase == "wait" and terminal_at is not None and scene.steps >= terminal_at),
        replay_recorder=SimpleNamespace(record=lambda **kwargs: records.append(kwargs)),
    )
    state = SimpleNamespace(arguments={"ticks": 3}, action={"action_id": "a", "action": "wait"},
                            environment=SimpleNamespace(scene=scene), execution_attempted=False,
                            executed=False, wait=None, replay_recorded=False)
    return session, state, scene, records


def test_robotwin_steps_ten_physics_steps_per_tick_without_writing_a_target():
    session, state, scene, records = _robotwin()
    hold_current_targets(session, state)
    assert scene.steps == 30 and state.executed and state.replay_recorded
    assert (state.wait["ticks_used"], state.wait["physics_steps"]) == (3, 30)
    assert state.wait["tick_unit"] == "ten_physics_steps"
    assert state.wait["tick_period_s"] == pytest.approx(0.04)
    assert records == [{"action_sequence": 4, "action": state.action, "controller": WAIT_CONTROLLER,
                        "arrays": {}, "physics_steps": 30, "replayable": False,
                        "metadata": {"reason": "hold_without_new_drive_targets", "ticks_requested": 3}}]


def test_robotwin_wait_stops_at_the_official_success_latch():
    session, state, scene, _records = _robotwin(terminal_at=12)
    hold_current_targets(session, state)
    assert scene.steps == 12 and state.wait["ticks_used"] == 2 and state.wait["terminated_early"] is True


@pytest.mark.parametrize("profile", LIVE)
def test_every_live_surface_publishes_the_same_wait_contract(profile):
    surface = resolve_tool_surface(profile)
    definition = next(item for item in surface.definitions if item["name"] == "wait_ticks")
    assert "wait_ticks" in surface.mutation_tools and surface.supports("shared_hold_wait_action_v1")
    assert definition["inputSchema"]["properties"]["ticks"]["maximum"] == WAIT_MAX_TICKS
    validate_schema_instance({"ticks": 20}, definition["inputSchema"])
    with pytest.raises(Exception):
        validate_schema_instance({"ticks": 21}, definition["inputSchema"])
    validate_schema_instance({"status": "submitted", "action": "wait_ticks"}, surface.result_schemas["wait_ticks"])
    for section in ("Use when:", "Effect:", "Returns:", "Does not:", "On error:"):
        assert section in definition["description"]


def test_only_the_backends_that_execute_wait_advertise_it():
    # LIBERO joined in 0059 (environment/backends/libero_pro/wait.py; tests/environments/libero/test_libero_wait_action.py).
    assert set(WAIT_ENVIRONMENTS) == {"libero_pro", "robodojo", "robotwin"}


def test_the_tool_submits_one_motion_v3_wait_and_names_the_bound_on_refusal(tmp_path):
    from ora.agent.tools.robot import ToolError
    from tests.environments.robodojo.test_robodojo_observe_result import _runtime

    runtime = _runtime(tmp_path)
    pointer, source, observation = runtime._current()
    pointer = {**pointer, "generation": 7, "observation_sha256": "a" * 64}
    runtime._current = lambda: (pointer, source, observation)
    with pytest.raises(ToolError, match="maximum 20"):  # the published schema gate
        runtime.invoke("wait_ticks", {"ticks": 21})
    from ora.agent.tools import wait_tool
    with pytest.raises(ToolError) as refused:  # the handler's own gate names the bound too
        wait_tool.submit(runtime, {"ticks": 0})
    assert refused.value.as_dict()["field"] == "ticks" and refused.value.as_dict()["received"] == 0
    assert refused.value.as_dict()["constraints"]["maximum"] == WAIT_MAX_TICKS
    assert not (runtime.control_root / "submitted-action.json").exists()
    result = runtime.invoke("wait_ticks", {"ticks": 3})
    assert {key: value for key, value in result.items() if key != "_runtime"} == {
        "status": "submitted", "action": "wait_ticks"}
    submitted = json.loads((runtime.control_root / "submitted-action.json").read_text())
    action = submitted if "schema" in submitted else submitted["action"]
    assert action["schema"] == "ora-motion-action.v3"
    assert action["action"] == "wait" and action["arguments"] == {"ticks": 3}


def test_robotwin_wait_runs_through_the_session_s_own_sparse_helpers(tmp_path, monkeypatch):
    """No branch in the descriptor-pinned session.py: admission, count, classify and compose are its own."""
    from ora.environment.backends.robotwin import session as session_module
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession
    from ora.environment.backends.robotwin.wait import execute_wait
    from tests.environments.robotwin.test_robotwin_contact_recovery import Plant, binding

    plant, recorder = Plant(), SimpleNamespace(records=[], updates=[])
    recorder.record = lambda **kwargs: recorder.records.append(kwargs)
    recorder.update = lambda *args, **kwargs: recorder.updates.append((args, kwargs))
    scene = SimpleNamespace(steps=0, get_timestep=lambda: 0.004)

    def step():
        scene.steps += 1
        plant.step_physics()

    scene.step = step
    live = object.__new__(RoboTwinLiveSession)
    for name, value in {
        "control_profile": binding(), "operator_execution_video": False, "last_receipt": None,
        "output_root": tmp_path, "action_sequence": 0, "servo_profile": None, "max_sim_steps": None,
        "world_mutation_sequence": 0, "world_mutation_actions": [], "replay_recorder": recorder,
        "gripper_command_targets": {"left": "open", "right": "close"}, "official_terminal": None,
        "abort_reason": None, "workspace_bounds_m": {"x": [-.6, .6], "y": [-.4, .4], "z": [.7, 1.45]},
        "environment": SimpleNamespace(scene=scene, robot=object(), eval_success=False,
                                       take_action_cnt=0, step_lim=100),
        "_measured_arm_joints": lambda _env, arm: plant.read_joint_positions(arm),
        "_public_pose_wxyz": lambda _env, arm: [*plant.read_joint_positions(arm), 1., 0., 0., 0.],
        "_after_physics_step": lambda *_args, **_kwargs: False,
        "_drive_target_digest": lambda _env, arm: "arm-" + arm,
        "_gripper_drive_target_digest": lambda _env, arm: "grip-" + arm,
        "_capture_operator_frame": lambda **_kwargs: None,
    }.items():
        setattr(live, name, value)
    monkeypatch.setattr(session_module, "RoboTwinMotionBackendAdapter", lambda **_kwargs: plant)
    (tmp_path / "receipts").mkdir()
    record = execute_wait(live, _action({"ticks": 3}))
    receipt = record["receipt"]
    assert scene.steps == 30 and plant.writes == [] and live.environment.take_action_cnt == 1
    assert json.loads((tmp_path / record["path"]).read_text()) == receipt  # the file is the receipt
    assert receipt["execution_status"] == "executed" and receipt["target_arms"] == []
    assert receipt["motion_backend"] == "hold_current_drive_targets"
    assert (receipt["wait"]["ticks_used"], receipt["wait"]["physics_steps"]) == (3, 30)
    for arm in ("left", "right"):
        preservation = receipt["arms"][arm]["command_preservation"]
        assert preservation["arm_drive_target_unchanged"] and preservation["gripper_drive_target_unchanged"]
    assert live.world_mutation_actions == ["wait"] and recorder.records[0]["replayable"] is False
    projected = resolve_receipt_effect_projector(receipt["schema"])(receipt, ReceiptProjectionContext(
        last_action_id="a", last_action_public_name="wait_ticks", surface_supports=lambda capability: True))
    assert '"ticks_used": 3' in json.dumps(projected)


def test_a_wait_between_release_and_placement_checks_keeps_the_robotwin_stage_history():
    """A hold names no arm and changes no command: it neither breaks nor extends the gripper history."""
    from ora.agent.tools.placement_runtime import stage_evidence
    from tests.environments.robotwin.test_robotwin_placement_adapter import history

    def waited(step):
        state = history({}, name="wait", step=step, status="executed")
        state["previous_public_action"]["arguments"] = {"ticks": 5}
        state["previous_execution_receipt"]["arguments"] = {"ticks": 5}
        return state

    states = [history({"left": "close"}), waited(1)]
    assert stage_evidence(states, "pre_release", arm="left")["status"] == "confirmed"
    states += [history({"left": "open"}, step=2), waited(3)]
    stage = stage_evidence(states, "post_release", arm="left")
    assert stage["status"] == "confirmed" and stage["post_open_pose_actions"] == 0
