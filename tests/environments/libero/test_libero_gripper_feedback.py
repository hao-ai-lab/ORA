"""LIBERO gripper feedback on the live direct surface.

The two traces below are measured LIBERO jaw openings, one per 20 Hz control
tick: a close in free air from the start pose, and a close on the bowl after
replaying the approach of the 2026-09-23 settle smoke episode.
"""
from __future__ import annotations

from typing import Any, Mapping

import pytest

from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.context.public_facts import gripper_command_delivery
from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext,
    resolve_receipt_effect_projector,
)
from ora.environment.backends.libero_pro.motion_settle import (
    GRIPPER_STALL_REASON,
    gripper_outcome,
    gripper_stalled,
)

LIVE = "libero_placement_direct_overhead_v1"
#: A kept surface that does not declare commanded-gripper delivery (no LIBERO one is left).
FROZEN = "robotwin_toolchain_b2_closed_noservo_v1"

FREE_AIR = [0.9708, 0.9245, 0.844, 0.7439, 0.633, 0.5159, 0.3953, 0.2728, 0.1675,
            0.1049, 0.0697, 0.05, 0.039, 0.0328, 0.0294]
ON_OBJECT = [0.97, 0.9236, 0.8435, 0.7437, 0.6328, 0.5158, 0.3952, 0.273, 0.1685,
             0.1077, 0.0914, 0.0911, 0.0908, 0.0905, 0.0903, 0.0902, 0.0902, 0.09]


def _first_stall(trace: list[float], target: float) -> int | None:
    for tick in range(1, len(trace) + 1):
        if gripper_stalled(trace[:tick], target=target):
            return tick
    return None


def test_a_close_on_an_object_stalls_early_and_free_air_never_does() -> None:
    assert _first_stall(FREE_AIR, 0.0) is None
    stall = _first_stall(ON_OBJECT, 0.0)
    # The jaws stop at tick 11; three still ticks later the close is over,
    # instead of waiting out the 80-tick budget.
    assert stall is not None and 12 <= stall <= 15


def test_a_command_that_reached_its_target_is_not_a_stall() -> None:
    assert not gripper_stalled([0.0294, 0.0293, 0.0293, 0.0293, 0.0293], target=0.0)


def test_the_outcome_names_the_state_in_physical_units() -> None:
    outcome = gripper_outcome(
        state="stalled_before_target", target=0.0, measured=0.0864,
        ticks_used=14, ticks_budget=80, maximum_aperture_m=0.08,
    )
    assert outcome["state"] == "stalled_before_target"
    assert outcome["commanded_opening_normalized"] == 0.0
    assert outcome["measured_aperture_mm"] == pytest.approx(6.912)
    assert outcome["seconds_used"] == pytest.approx(0.7)
    assert outcome["ticks_budget"] == 80
    assert "not proof of a grasp" in outcome["claim_boundary"]


def _projected(record: Mapping[str, Any]) -> dict[str, Any]:
    context = ReceiptProjectionContext(
        last_action_id=record["action_id"], last_action_public_name="set_gripper",
        surface_supports=resolve_tool_surface(LIVE).supports,
    )
    return resolve_receipt_effect_projector(record["schema"])(record, context)


def test_a_stalled_close_reaches_the_policy_as_a_gripper_result() -> None:
    record = {
        "schema": "ora.libero-pro-hybrid-motion-receipt.v2",
        "action_id": "action-3", "action": "set_gripper", "motion_status": "stopped",
        "feedback": {"stop_reason": GRIPPER_STALL_REASON},
        "gripper_outcome": gripper_outcome(
            state="stalled_before_target", target=0.0, measured=0.0864,
            ticks_used=14, ticks_budget=80, maximum_aperture_m=0.08),
        "terminal_after_action": False,
    }
    projected = _projected(record)
    assert projected["status"] == "not_reached"
    assert projected["failure_code"] == GRIPPER_STALL_REASON
    assert projected["detail"]["gripper_outcome"]["state"] == "stalled_before_target"
    assert projected["failure_code"] != "endpoint_residual_exceeded"


def test_the_live_surface_constrains_the_segmentation_camera() -> None:
    tool = next(d for d in resolve_tool_surface(LIVE).definitions if d["name"] == "vision_segment")
    assert tool["inputSchema"]["properties"]["camera"]["enum"] == ["agentview", "wrist"]


def test_the_live_set_gripper_help_states_timing_and_the_stall() -> None:
    tool = next(d for d in resolve_tool_surface(LIVE).definitions if d["name"] == "set_gripper")
    text = tool["description"]
    for phrase in ("80 control ticks", "20 Hz", "15 ticks", GRIPPER_STALL_REASON,
                   "measured_aperture_mm"):
        assert phrase in text


def _gripper_branches(node: Any) -> list[Mapping[str, Any]]:
    found = []
    if isinstance(node, Mapping):
        properties = node.get("properties")
        if isinstance(properties, Mapping) and "measurement_semantics" in properties and (
            properties["measurement_semantics"].get("const")
            == "two_prismatic_finger_displacements_not_command"
        ):
            found.append(node)
        for value in node.values():
            found.extend(_gripper_branches(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_gripper_branches(value))
    return found


@pytest.mark.parametrize("profile,delivered", [(LIVE, True), (FROZEN, False)])
def test_the_commanded_gripper_is_delivered_only_where_declared(profile, delivered) -> None:
    surface = resolve_tool_surface(profile)
    assert gripper_command_delivery(surface.view_config) is delivered
    branches = _gripper_branches(dict(surface.result_schemas))
    assert branches
    for branch in branches:
        declared = {"commanded_opening_normalized", "command_target"} <= set(branch["properties"])
        assert declared is delivered
