"""RoboDojo gripper feedback on the live ARX surface.

The traces below are measured ARX X5 normalized openings, one sample at
command start and one after every native action (40 ms), taken read-only from
recorded sparse set_gripper receipts:

- FREE_AIR: exp11a-robodojo 20260926T231016Z action-0031, a close in free air;
- ON_OBJECT: the same run's action-0025, a close already stopped on an object
  (encoder 0.887, SDK-limited drive target 0.687 for all 20 native actions);
- CONTACT: toolchain-b2-robodojo 20260912T193946Z action-0035, a close that
  meets an object on the fourth native action;
- CREEP: inspect-robots-b2-robodojo 20260914T071901Z action-0038, a close that
  keeps creeping a few thousandths per native action against an object.
"""
from __future__ import annotations

import pytest

from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext,
    resolve_receipt_effect_projector,
)
from ora.environment.backends.robodojo.common import ARX_GRIPPER_STALL
from ora.environment.backends.robodojo.sparse_motion import gripper_failure_code
from ora.environment.controllers.gripper_stall import (
    GRIPPER_STALL_REASON,
    gripper_stalled,
)
from test_robodojo_backend import sparse, v3  # noqa: F401  (fixture and action helper)

LIVE = "robodojo_placement_closed_skills_v2"
FLOOR = 0.01 / 0.054

FREE_AIR = [1.0, 0.8341, 0.6703, 0.5111, 0.3605, 0.2269, 0.1852]
ON_OBJECT = [0.887] * 21
CONTACT = [0.9996, 0.8346, 0.6707, 0.5114, 0.4998, 0.5001, 0.5001, 0.5001, 0.5001,
           0.5001, 0.5001, 0.5001, 0.5001, 0.5001, 0.5001, 0.5001, 0.5001, 0.5001,
           0.5001, 0.5001, 0.5001]
CREEP = [0.4122, 0.365, 0.3624, 0.3598, 0.3535, 0.3471, 0.3423, 0.3366, 0.3318, 0.3295,
         0.3297, 0.3279, 0.3256, 0.3236, 0.323, 0.3232, 0.3224, 0.3221, 0.3222, 0.3214,
         0.3213]


def _first_stall(trace: list[float], target: float = FLOOR) -> int | None:
    """Native actions executed when the rule first ends the command."""
    for samples in range(1, len(trace) + 1):
        if ARX_GRIPPER_STALL.arrived(trace[samples - 1], target):
            return None
        if gripper_stalled(trace[:samples], target=target, rule=ARX_GRIPPER_STALL):
            return samples - 1
    return None


def test_free_air_close_arrives_and_never_stalls() -> None:
    assert _first_stall(FREE_AIR) is None
    assert ARX_GRIPPER_STALL.arrived(FREE_AIR[-1], FLOOR)


@pytest.mark.parametrize("trace,expected", [(ON_OBJECT, 3), (CONTACT, 7)])
def test_a_close_on_an_object_ends_three_native_actions_after_the_jaws_stop(trace, expected) -> None:
    # The recorded commands ran all 20 native actions (0.8 s).
    assert _first_stall(trace) == expected


def test_a_creeping_close_on_an_object_still_ends_before_the_budget() -> None:
    stall = _first_stall(CREEP)
    assert stall is not None and stall < 20


def test_the_rule_speaks_native_actions() -> None:
    assert ARX_GRIPPER_STALL.tick_period_s == pytest.approx(0.04)
    assert ARX_GRIPPER_STALL.tick_unit == "native_action"
    assert ARX_GRIPPER_STALL.arrival_tolerance == pytest.approx(0.03)


@pytest.mark.parametrize("states,terminal,code", [
    ({"left": "reached_target"}, False, None),
    ({"left": "stalled_before_target"}, False, GRIPPER_STALL_REASON),
    ({"left": "reached_target", "right": "stalled_before_target"}, False, GRIPPER_STALL_REASON),
    ({"left": "stalled_before_target", "right": "moving_at_budget"}, False, "gripper_not_reached"),
    ({"left": "interrupted"}, True, "episode_terminal_during_execution"),
])
def test_one_public_code_per_jaw_command(states, terminal, code) -> None:
    assert gripper_failure_code(states, terminal=terminal) == code


def _obstructed(sparse, *, opening: float) -> None:
    """Stop the fake plant's left jaw at ``opening`` as an object would."""
    plant = sparse.environment
    manager = plant.robot_manager
    stop_m = -0.01 + opening * 0.054
    native_tick = plant.native_tick

    def tick(control):
        result = native_tick(control)
        manager.fingers["left_arm"] = [max(stop_m, value) for value in manager.fingers["left_arm"]]
        return result

    plant.native_tick = tick


def test_a_blocked_close_ends_early_with_a_measured_outcome(sparse, tmp_path) -> None:
    _obstructed(sparse, opening=0.74)
    receipt = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    assert receipt["execution_status"] == "not_reached"
    assert receipt["failure_stage"] == "execution"
    assert receipt["failure_code"] == GRIPPER_STALL_REASON
    # One native action to meet the object, three still ones, not 20.
    assert receipt["native_actions"] == 4
    outcome = receipt["arms"]["left"]["gripper_outcome"]
    assert outcome["state"] == "stalled_before_target"
    assert outcome["commanded_opening_normalized"] == 0.0
    assert outcome["measured_opening_normalized"] == pytest.approx(0.74)
    assert outcome["effective_target_opening_normalized"] == pytest.approx(
        receipt["gripper_targets_effective"]["left"])
    assert outcome["sdk_limited_drive_target_normalized"] == pytest.approx(
        receipt["after"]["left"]["gripper_feedback"]["drive_target_opening_normalized"])
    assert outcome["sdk_drive_limit_normalized"] == 0.2
    assert (outcome["ticks_used"], outcome["ticks_budget"]) == (4, 20)
    assert outcome["seconds_used"] == pytest.approx(0.16)
    assert "not proof of a grasp" in outcome["claim_boundary"]
    assert "gripper_outcome" not in receipt["arms"]["right"]


def test_a_free_close_still_reaches_and_reports_its_outcome(sparse, tmp_path) -> None:
    receipt = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    assert receipt["execution_status"] == "reached" and receipt["failure_code"] is None
    assert receipt["arms"]["left"]["gripper_outcome"]["state"] == "reached_target"


def _projected(record):
    context = ReceiptProjectionContext(
        last_action_id=record["action_id"], last_action_public_name="set_gripper",
        surface_supports=resolve_tool_surface(LIVE).supports,
    )
    return resolve_receipt_effect_projector(record["schema"])(record, context)


def test_a_stalled_close_reaches_the_policy_with_its_outcome(sparse, tmp_path) -> None:
    _obstructed(sparse, opening=0.74)
    receipt = sparse.execute(v3(sparse, tmp_path, kind="set_gripper", targets={"left": "close"}))["receipt"]
    projected = _projected(receipt)
    assert projected["status"] == "not_reached"
    assert projected["failure_code"] == GRIPPER_STALL_REASON
    outcome = projected["detail"]["arms"]["left"]["gripper_outcome"]
    assert outcome["state"] == "stalled_before_target"
    for number in ("measured_opening_normalized", "commanded_opening_normalized",
                   "sdk_limited_drive_target_normalized"):
        assert isinstance(outcome[number], float)


def test_the_live_set_gripper_help_states_timing_limiter_and_stall() -> None:
    surface = resolve_tool_surface(LIVE)
    tool = next(d for d in surface.definitions if d["name"] == "set_gripper")
    text = tool["description"]
    for phrase in ("25 Hz", "40 ms", "20 native actions last at most 0.8 s",
                   "within 0.2 normalized units of the current FIRST finger encoder",
                   "0.005", GRIPPER_STALL_REASON, "arms.<arm>.gripper_outcome",
                   "sdk_limited_drive_target_normalized", "moving_at_budget"):
        assert phrase in text, phrase
    # The source-owned controller contract still closes the help, after the layer.
    assert text.rstrip().endswith("}")
    grasp = next(d for d in surface.definitions if d["name"] == "grasp_execute")
    assert GRIPPER_STALL_REASON in grasp["description"]


def test_the_help_numbers_are_the_executed_numbers() -> None:
    from ora.contracts.robodojo_arx import ARX_X5_CONTRACT

    text = next(d for d in resolve_tool_surface(LIVE).definitions if d["name"] == "set_gripper")["description"]
    period_ms = round(1000 * ARX_GRIPPER_STALL.tick_period_s)
    budget = ARX_X5_CONTRACT["native_actions_per_command"]
    assert f"{period_ms} ms" in text
    assert f"{budget} native actions last at most {budget * ARX_GRIPPER_STALL.tick_period_s:g} s" in text
    assert f"less than {ARX_GRIPPER_STALL.still_step:g}" in text
    assert f"within {ARX_GRIPPER_STALL.arrival_tolerance:g} of its effective target" in text
    assert f"{ARX_GRIPPER_STALL.window} native actions ({ARX_GRIPPER_STALL.window * ARX_GRIPPER_STALL.tick_period_s:g} s)" in text
    limit = ARX_X5_CONTRACT["gripper_drive_limit"]["maximum_normalized_offset"]
    assert f"within {limit:g} normalized units of the current FIRST finger encoder" in text
