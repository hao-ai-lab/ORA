"""The ARX jaw reading publishes both fingers (EXP11A g012 E11G12-001).

The released public opening normalized ``fingers[0]`` only. Under a lateral
load the two prismatic fingers (joint7/joint8, 0-0.044 m each) shift together
to one side, so that one reading reported 0.95 "nearly open" while the two-finger
gap was about half open (by_language seq 61: fingers 41.4 / 9.5 mm), or read low
while the gap was wide (seq 404: 5.1 / 37.9 mm). The same value decided the
gripper arrival/stall rule and jaw_hold. These tests pin the two-finger mean on
the unchanged command scale, the signed lateral offset beside it, the
consumers that now read it, and that a one-sided reading never becomes a
confident held/released verdict.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from ora.agent.tools import jaw_hold
from ora.contracts.gripper_feedback import (
    GRIPPER_FEEDBACK_SCHEMA, GRIPPER_FEEDBACK_SCHEMA_V2, TWO_FINGER_MEASUREMENT_SOURCE, gripper_feedback,
    validate_gripper_feedback,
)
from ora.environment.backends.robodojo import common, feedback, sparse_motion
from ora.environment.backends.robodojo.motion_checks import derive_jaw_step

SCALE = (-0.01, 0.044)  # robot_config.yml gripper_scale (command scale), metres


def norm(value):
    return (value - SCALE[0]) / (SCALE[1] - SCALE[0])


def robot():
    return SimpleNamespace(gripper_scale=list(SCALE), gripper_move={"sign": 1, "mimic": ["joint8", 1, 0]})


def _feedback(**overrides):
    values = dict(requested_opening_normalized=0.0, drive_target_opening_normalized=0.3,
                  measured_opening_normalized=0.66, effective_target_opening_normalized=0.185,
                  tolerance_normalized=0.03, drive_target_source="resolved_sdk_joint_position_target",
                  measurement_source=TWO_FINGER_MEASUREMENT_SOURCE,
                  pose_hold_semantics="preserve_last_applied_drive_target", finger_lateral_offset_m=0.016)
    values.update(overrides)
    return gripper_feedback(**values)


# --- contract ----------------------------------------------------------------

def test_the_two_finger_source_is_a_versioned_feedback_with_the_lateral_offset():
    value = _feedback()
    assert value["schema"] == GRIPPER_FEEDBACK_SCHEMA_V2 == "ora.gripper-feedback.v2"
    assert value["measurement_source"] == "two_finger_encoder_mean_v1"
    assert value["finger_lateral_offset_m"] == 0.016
    assert validate_gripper_feedback(value) == value


def test_released_sources_keep_the_v1_shape_unchanged():
    value = _feedback(measurement_source="primary_finger_encoder", finger_lateral_offset_m=None)
    assert value["schema"] == GRIPPER_FEEDBACK_SCHEMA and "finger_lateral_offset_m" not in value
    assert validate_gripper_feedback(value) == value


@pytest.mark.parametrize("overrides", [
    {"finger_lateral_offset_m": None},                                       # two-finger without its offset
    {"finger_lateral_offset_m": math.nan},
    {"measurement_source": "primary_finger_encoder"},                         # an offset on a one-finger source
    {"measurement_source": "normalized_measured_aperture"},
])
def test_the_offset_belongs_to_the_two_finger_source_only(overrides):
    with pytest.raises(ValueError):
        _feedback(**overrides)


def test_validation_rejects_a_dropped_or_relabelled_offset():
    value = _feedback()
    with pytest.raises(ValueError):
        validate_gripper_feedback({k: v for k, v in value.items() if k != "finger_lateral_offset_m"})
    with pytest.raises(ValueError):
        validate_gripper_feedback({**value, "schema": GRIPPER_FEEDBACK_SCHEMA})


# --- the published reading ---------------------------------------------------

@pytest.mark.parametrize("fingers,old,new,offset_mm", [
    # by_language seq 61 (controller action-0006 end): published 0.952
    ([0.0414094925, 0.0095378673], 0.952, 0.6569, 15.94),
    # by_language seq 404: published 0.28 "closed" while the gap was wide
    ([0.0051, 0.0379], 0.280, 0.5833, -16.4),
    # classify_objects seq 835 close stage: primary pinned open
    ([0.044, 0.035], 1.0, 0.9167, 4.5),
])
def test_the_public_opening_is_the_two_finger_mean_on_the_command_scale(fingers, old, new, offset_mm):
    assert common.gripper_opening(robot(), fingers[0]) == pytest.approx(old, abs=1e-3)
    assert common.two_finger_opening(robot(), fingers) == pytest.approx(new, abs=1e-3)
    assert common.finger_lateral_offset_m(fingers) * 1000 == pytest.approx(offset_mm, abs=0.05)


@pytest.mark.parametrize("value", [0.0, 0.0172, 0.044])
def test_symmetric_fingers_publish_exactly_the_released_value(value):
    assert common.two_finger_opening(robot(), [value, value]) == common.gripper_opening(robot(), value)
    assert common.finger_lateral_offset_m([value, value]) == 0.0


def test_a_mimic_offset_maps_the_second_finger_back_to_the_primary_coordinate():
    arm = SimpleNamespace(gripper_scale=list(SCALE), gripper_move={"sign": 1, "mimic": ["joint8", 2.0, 0.001]})
    # joint8 = 2 * joint7 + 0.001 when the pair is symmetric in command terms.
    assert common.two_finger_opening(arm, [0.02, 0.041]) == pytest.approx(norm(0.02))


def test_gripper_state_publishes_one_reading_everywhere(monkeypatch):
    import numpy as np

    monkeypatch.setattr(sparse_motion, "gripper_closed_floor", lambda session, arm: 0.185)
    session = SimpleNamespace(robots={"left": robot()}, gripper_directives={"left": 0.0})
    state = feedback.gripper_state(session, "left", np.array([0.0414094925, 0.0095378673]), 0.3)
    assert state["gripper"] == pytest.approx(0.6569, abs=1e-4)
    assert state["gripper_finger_qpos"] == pytest.approx([0.0414094925, 0.0095378673])
    typed = state["gripper_feedback"]
    assert typed["measured_opening_normalized"] == state["gripper"]
    assert typed["measurement_source"] == "two_finger_encoder_mean_v1"
    assert typed["finger_lateral_offset_m"] == pytest.approx(0.01594, abs=1e-5)
    assert validate_gripper_feedback(typed) == typed


# --- the SDK primary-relative limiter in the gripper outcome -------------------

def _after(fingers, drive, effective=0.185, requested=0.0):
    mean = common.two_finger_opening(robot(), fingers)
    typed = _feedback(measured_opening_normalized=mean, drive_target_opening_normalized=drive,
                      effective_target_opening_normalized=effective, requested_opening_normalized=requested,
                      finger_lateral_offset_m=common.finger_lateral_offset_m(fingers))
    return {"gripper": mean, "gripper_finger_qpos": list(fingers), "gripper_feedback": typed}


def test_a_close_held_by_the_primary_relative_limit_is_named_with_both_fingers():
    # classify_objects seq 835 after the secondary settles: primary pinned at
    # 0.044 (1.0), drive target 0.8 = 1.0 - 0.2, secondary at that target.
    secondary = SCALE[0] + 0.8 * (SCALE[1] - SCALE[0])
    outcome = sparse_motion._gripper_outcome(state="stalled_before_target", budget=20, used=8,
                                             after=_after([0.044, secondary], drive=0.8), robot=robot())
    assert outcome["primary_finger_limit_holds_close"] is True
    assert outcome["measured_finger_displacements_m"] == pytest.approx([0.044, secondary])
    assert outcome["finger_lateral_offset_m"] == pytest.approx((0.044 - secondary) / 2)
    assert outcome["measured_opening_normalized"] == pytest.approx(0.9)
    assert outcome["sdk_limited_drive_target_normalized"] == 0.8


@pytest.mark.parametrize("fingers,drive,requested,effective", [
    ([0.02, 0.02], 0.35, 0.0, 0.185),          # symmetric stall on an object: both fingers stopped
    ([0.0, 0.0], 0.0, 0.0, 0.185),             # reached the floor
    ([0.044, 0.044], 1.0, 1.0, 1.0),           # an open
])
def test_other_jaw_outcomes_do_not_claim_the_limiter(fingers, drive, requested, effective):
    outcome = sparse_motion._gripper_outcome(state="stalled_before_target", budget=20, used=5,
                                             after=_after(fingers, drive, effective, requested), robot=robot())
    assert outcome["primary_finger_limit_holds_close"] is False


# --- the stall rule reads both fingers ------------------------------------------

def test_a_close_ends_only_when_both_fingers_have_stopped(tmp_path):
    from test_robodojo_backend import sparse_session, v3

    session = sparse_session(tmp_path)
    try:
        plant, manager = session.environment, session.environment.robot_manager
        original = plant.native_tick
        # Recorded shape (classify_objects seq 835): the primary finger stays at
        # its 44 mm stop while the other closes 3 mm per native action to 33 mm.
        script = iter([[0.044, 0.041], [0.044, 0.038], [0.044, 0.035], [0.044, 0.0332]] + [[0.044, 0.0332]] * 16)

        def tick(control):
            result = original(control)
            manager.fingers["left_arm"] = list(next(script))
            return result

        plant.native_tick = tick
        receipt = session.execute(v3(session, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    finally:
        session.close()
    outcome = receipt["arms"]["left"]["gripper_outcome"]
    # The primary-only reading stood at 1.0 from the first action and would have
    # stalled after three; the two-finger reading moves until the fourth.
    assert outcome["state"] == "stalled_before_target" and outcome["ticks_used"] == 7
    assert outcome["measured_opening_normalized"] == pytest.approx(norm((0.044 + 0.0332) / 2))
    assert receipt["after"]["left"]["gripper_feedback"]["finger_lateral_offset_m"] == pytest.approx(0.0054)


# --- qualification ---------------------------------------------------------------

def test_the_jaw_probe_compares_the_public_reading_with_both_fingers():
    reading = {"gripper_finger_qpos": [0.02, 0.01], "applied_gripper_joint_targets": [0.02, 0.02],
               "gripper": norm(0.015), "gripper_command": 0.5,
               "controlled_link_pose_xyz_xyzw": [0, 0, 0, 0, 0, 0, 1],
               "finger_body_position_w_m": {"link7": [0, 0.07, 0], "link8": [0, -0.06, 0]}}
    assert derive_jaw_step(command=0.5, reading=reading)["normalized_reading_error"] == pytest.approx(0.0)
    primary_only = {**reading, "gripper": norm(0.02)}
    assert derive_jaw_step(command=0.5, reading=primary_only)["normalized_reading_error"] == pytest.approx(
        norm(0.02) - norm(0.015))


# --- jaw_hold: a one-sided reading is not hold evidence --------------------------

def _hold(close, after, *, close_offset=0.0, after_offset=0.0, drive=None, state="stalled_before_target"):
    outcome = {"state": state, "measured_opening_normalized": close, "effective_target_opening_normalized": 0.185,
               "sdk_limited_drive_target_normalized": close - 0.2 if drive is None else drive,
               "finger_lateral_offset_m": close_offset}
    return jaw_hold.hold_evidence(outcome=outcome, after_lift=after, after_lift_offset=after_offset,
                                  tolerance=0.03, opened=1.0)


def test_symmetric_readings_keep_their_verdicts():
    assert _hold(0.5, 0.5)["held"] == "likely"
    assert _hold(0.5, 0.185)["reason"] == "jaws_closed_to_the_floor_during_lift"
    assert _hold(0.99999999, 0.99999999)["reason"] == "jaws_never_closed"


@pytest.mark.parametrize("close_offset,after_offset", [(0.0045, 0.0), (0.0, 0.0160), (-0.0164, -0.0164)])
def test_strongly_asymmetric_fingers_leave_the_verdict_unknown(close_offset, after_offset):
    evidence = _hold(0.6, 0.6, close_offset=close_offset, after_offset=after_offset)
    assert evidence["held"] == "unknown" and evidence["reason"] == "finger_offset_exceeds_symmetric_bound"
    assert evidence["evidence"]["close_finger_lateral_offset_m"] == close_offset
    assert evidence["evidence"]["after_lift_finger_lateral_offset_m"] == after_offset
    assert evidence["evidence"]["symmetric_offset_bound_m"] == jaw_hold.SYMMETRIC_OFFSET_BOUND_M == 0.004


def test_the_recorded_pinned_primary_close_is_no_longer_called_never_closed():
    # classify_objects seq 835: released reading 1.0 -> unlikely (jaws_never_closed);
    # both fingers give 0.917 with a 4.5 mm offset, which is not evidence either way.
    evidence = _hold(0.9167, 0.9167, close_offset=0.0045, after_offset=0.0045, drive=0.8)
    assert evidence["held"] == "unknown" and evidence["reason"] == "finger_offset_exceeds_symmetric_bound"


def test_a_missing_offset_never_invents_asymmetry():
    outcome = {"state": "stalled_before_target", "measured_opening_normalized": 0.5,
               "effective_target_opening_normalized": 0.185, "sdk_limited_drive_target_normalized": 0.3}
    evidence = jaw_hold.hold_evidence(outcome=outcome, after_lift=0.5, tolerance=0.03)
    assert evidence["held"] == "likely" and evidence["evidence"]["close_finger_lateral_offset_m"] is None


def test_the_verify_stage_reads_the_after_lift_offset_from_the_typed_feedback():
    close = {"stage": "close", "measurements": [{"quantity": "gripper_outcome", "value": {"left": {
        "state": "stalled_before_target", "measured_opening_normalized": 0.6,
        "effective_target_opening_normalized": 0.185, "sdk_limited_drive_target_normalized": 0.4,
        "finger_lateral_offset_m": 0.0}}}]}
    record = {"gripper_feedback": _feedback(measured_opening_normalized=0.6, finger_lateral_offset_m=0.012)}
    evidence = jaw_hold.verify_stage_evidence([close], "left", record)
    assert evidence["held"] == "unknown" and evidence["evidence"]["after_lift_finger_lateral_offset_m"] == 0.012


# --- delivery ----------------------------------------------------------------

def test_observe_delivers_the_two_finger_reading_through_the_live_result_schema(tmp_path):
    from test_robodojo_observe_result import _runtime

    import json

    runtime = _runtime(tmp_path)
    _, source, observation = runtime._current()
    fingers = [0.044, 0.0332]
    mean = common.two_finger_opening(robot(), fingers)
    observation["proprioception"]["end_effectors"]["right"].update(
        gripper=mean, gripper_finger_qpos=fingers,
        gripper_feedback=_feedback(measured_opening_normalized=mean, drive_target_opening_normalized=0.8,
                                   finger_lateral_offset_m=common.finger_lateral_offset_m(fingers)))
    (source / "observation.json").write_text(json.dumps(observation, sort_keys=True, indent=2) + "\n")
    gripper = runtime.invoke("observe", {})["public_robot_facts"]["end_effectors"]["right"]["gripper"]
    assert gripper["measured_opening_normalized"] == pytest.approx(0.9)
    # The per-finger readings keep their one public name; the feedback adds only the offset.
    assert gripper["measured_finger_displacements_m"] == fingers and gripper["measured_aperture_mm"] is None
    assert gripper["feedback"]["measurement_source"] == "two_finger_encoder_mean_v1"
    assert gripper["feedback"]["finger_lateral_offset_m"] == pytest.approx(0.0054)
    assert "finger_positions_m" not in gripper["feedback"]


def test_a_control_replay_compares_jaw_readings_only_under_one_measurement_source():
    """A run recorded before the two-finger reading replays with finger joints compared, not the reading."""
    from ora.environment.backends.robodojo.control_replay import state_divergence

    def state(fingers, gripper, source):
        return {"arm_joint_qpos": [0.0] * 6, "gripper_finger_qpos": list(fingers), "gripper": gripper,
                "gripper_feedback": {"measurement_source": source}}

    recorded = {"left": state([0.044, 0.0332], 1.0, "primary_finger_encoder"),
                "right": state([0.02, 0.02], 0.55, "two_finger_encoder_mean_v1")}
    replayed = {"left": state([0.044, 0.0332], 0.9, "two_finger_encoder_mean_v1"),
                "right": state([0.02, 0.02], 0.56, "two_finger_encoder_mean_v1")}
    divergence = state_divergence(recorded, replayed)
    assert divergence["finger_m"] == 0.0
    # Left changed derivation, not physics; right is compared as before.
    assert divergence["gripper"] == pytest.approx(0.01)


def test_a_journal_replay_compares_jaw_readings_only_under_one_measurement_source():
    from ora.orchestrator.robodojo_motion_replay import line_divergence

    def line(gripper, source):
        arm = {"arm_joint_qpos": [0.0] * 6, "gripper_finger_qpos": [0.044, 0.0332], "gripper": gripper,
               "pose_xyz_xyzw": [0, 0, 0, 0, 0, 0, 1], "gripper_feedback": {"measurement_source": source}}
        return {"measured": {"left": arm, "right": arm}, "applied_position_targets": {}}

    old, new = line(1.0, "primary_finger_encoder"), line(0.9, "two_finger_encoder_mean_v1")
    assert line_divergence(old, new)["gripper"] == 0.0 and line_divergence(old, new)["finger_m"] == 0.0
    assert line_divergence(line(0.95, "two_finger_encoder_mean_v1"), new)["gripper"] == pytest.approx(0.05)


def test_never_closed_is_judged_on_the_close_reading_alone():
    # A partial open stage (0.96) lets a one-sided close pass the never-closed
    # band; the close reading it compares is then not evidence either way.
    outcome = {"state": "stalled_before_target", "measured_opening_normalized": 0.94,
               "effective_target_opening_normalized": 0.185, "sdk_limited_drive_target_normalized": 0.74,
               "finger_lateral_offset_m": 0.006}
    one_sided = jaw_hold.hold_evidence(outcome=outcome, after_lift=0.94, tolerance=0.03, opened=0.96)
    assert one_sided["held"] == "unknown" and one_sided["reason"] == "finger_offset_exceeds_symmetric_bound"
    # The after-lift reading does not enter a never-closed verdict.
    symmetric = {**outcome, "finger_lateral_offset_m": 0.0}
    kept = jaw_hold.hold_evidence(outcome=symmetric, after_lift=0.6, after_lift_offset=0.01, tolerance=0.03,
                                  opened=0.96)
    assert kept["held"] == "unlikely" and kept["reason"] == "jaws_never_closed"
