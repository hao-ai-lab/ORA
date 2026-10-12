"""D3b: static facts once per round in a protected slot (``round_static_facts_slot_v1``).

``native_slots_v4`` states every ``current_actual`` field once per Session
round in a protected message; each observation carries inline only what
differs byte for byte from that slot and lists the rest under
``static_facts.unchanged``. These tests cover the pure split and the Native
session that applies it; the Context lifecycle (aging, reset, emergency
checkpoint) is in ``test_round_static_facts_context.py``.
"""

from __future__ import annotations

import copy
import json

import pytest

from ora.agent.context import round_static_facts as slot
from ora.agent.contracts import PolicySessionError

MATRIX = [[1.0, 0.0, 0.0, 0.25], [0.0, 1.0, 0.0, -0.5], [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]


def _camera(offset: float = 0.0) -> dict:
    extrinsic = copy.deepcopy(MATRIX)
    extrinsic[0][3] += offset
    return {"intrinsic_cv": [[397.04, 0.0, 320.0], [0.0, 397.04, 240.0], [0.0, 0.0, 1.0]],
            "extrinsic_cv": extrinsic, "cam2world_cv": extrinsic,
            "camera_origin_world_m": [extrinsic[0][3], -0.5, 1.5], "depth_unit": "metres"}


def _actual(sequence: int, *, wrist: float = 0.0, pose_x: float = 0.3) -> dict:
    return {
        "actual_sequence": sequence, "observation_step": sequence,
        "camera_geometry": {"head": _camera(), "left_wrist": _camera(wrist)},
        "conventions": {"camera_geometry.<camera>": {"pixel_coordinates": "origin top-left"}},
        "delivery": "complete_current_rgb_and_calibration", "depth_delivery": "on_demand",
        "public_robot_facts": {
            "observation_id": f"robodojo:o-{sequence}", "snapshot_role": "current",
            "controller_feedback": {"position_tolerance_m": 0.01, "arrival_comparison": "<=",
                                    "gripper_drive_limit": {"kind": "measured_primary_relative"}},
            "end_effectors": {"left": {"pose_xyz_xyzw": [pose_x, -0.35, 0.92, 0, 0, 0, 1],
                                       "controlled_frame": "arx_x5_link6"}},
        },
        "view_dimensions": {"head": {"width": 640, "height": 480},
                            "left_wrist": {"width": 640, "height": 480}},
        "views": ["head", "left_wrist"],
    }


def _dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def test_the_slot_holds_every_field_but_the_identity_keys():
    fields = slot.capture(_actual(0))
    assert "actual_sequence" not in fields and "observation_step" not in fields
    assert fields["camera_geometry"] == _actual(0)["camera_geometry"]
    message = slot.slot_message(round_index=2, actual_sequence=7, fields=fields)
    body = json.loads(message["content"][0]["text"])["round_static_facts"]
    assert body == {"round": 2, "captured_at_actual_sequence": 7, "rule": slot.RULE,
                    "current_actual": fields}
    assert message["role"] == "user" and message["_protected_context"] is True
    assert message["_round_static_facts"] == {"round": 2}
    assert message["content"][0]["text"] == _dumps(json.loads(message["content"][0]["text"]))


def test_the_round_first_observation_points_everything_at_the_slot():
    first = _actual(0)
    reduced = slot.reduce(first, slot.capture(first), round_index=1)
    assert reduced["actual_sequence"] == 0 and reduced["observation_step"] == 0
    assert reduced["static_facts"] == {"round_slot": 1, "unchanged": sorted([
        "camera_geometry", "conventions", "delivery", "depth_delivery",
        "public_robot_facts", "view_dimensions", "views"])}
    assert set(reduced) == {"actual_sequence", "observation_step", "static_facts"}
    assert slot.restore(reduced, slot.capture(first)) == first


def test_a_changed_value_is_always_inline():
    fields = slot.capture(_actual(0))
    later = _actual(5, wrist=0.031, pose_x=0.41)
    reduced = slot.reduce(later, fields, round_index=1)
    unchanged = reduced["static_facts"]["unchanged"]
    # The fixed camera is stated by the slot; the moving wrist camera is inline
    # except the intrinsics, which did not change.
    assert "camera_geometry.head" in unchanged
    assert reduced["camera_geometry"]["left_wrist"]["extrinsic_cv"] == later["camera_geometry"][
        "left_wrist"]["extrinsic_cv"]
    assert "camera_geometry.left_wrist.intrinsic_cv" in unchanged
    assert reduced["public_robot_facts"]["end_effectors"]["left"]["pose_xyz_xyzw"][0] == 0.41
    assert reduced["public_robot_facts"]["observation_id"] == "robodojo:o-5"
    assert "public_robot_facts.controller_feedback" in unchanged
    # Every inline value differs from the slot; every listed one is identical.
    for path in unchanged:
        value, base = later, fields
        for step in path.split("."):
            value, base = value[step], base[step]
        assert _dumps(value) == _dumps(base)
    assert slot.restore(reduced, fields) == later


def test_a_field_too_short_to_name_stays_inline():
    fields = slot.capture(_actual(0))
    later = _actual(1, wrist=0.01)
    reduced = slot.reduce(later, fields, round_index=1)
    # "metres" (8 bytes) is shorter than its path, so it is restated, not named.
    assert reduced["camera_geometry"]["left_wrist"]["depth_unit"] == "metres"
    assert "camera_geometry.left_wrist.depth_unit" not in reduced["static_facts"]["unchanged"]
    assert reduced["public_robot_facts"]["snapshot_role"] == "current"


def test_a_key_that_is_not_an_identifier_is_never_split():
    fields = slot.capture(_actual(0))
    later = _actual(1)
    later["conventions"]["camera_geometry.<camera>"]["depth_unit"] = "metres"
    reduced = slot.reduce(later, fields, round_index=1)
    assert reduced["conventions"] == later["conventions"]
    assert not any("<" in path for path in reduced["static_facts"]["unchanged"])
    assert slot.restore(reduced, fields) == later


def test_new_and_vanished_fields():
    fields = slot.capture(_actual(0))
    later = _actual(1)
    later["held_object_offset"] = {"frame": "world", "translation_m": [0, 0, 0.1]}
    del later["views"]
    reduced = slot.reduce(later, fields, round_index=1)
    assert reduced["held_object_offset"] == later["held_object_offset"]
    assert "views" not in reduced and "views" not in reduced["static_facts"]["unchanged"]
    assert slot.restore(reduced, fields) == later


def test_slot_text_survives_materialization_and_aging_markers(tmp_path):
    from ora.agent.evidence.trajectory import ArtifactStore
    from ora.agent.memory.artifacts import ContextArtifactCatalog

    artifacts = ArtifactStore(tmp_path)
    catalog = ContextArtifactCatalog(tmp_path, "policy-session-1", artifacts)
    message = slot.slot_message(round_index=1, actual_sequence=0, fields=slot.capture(_actual(0)))
    observation = {"role": "user", "content": [{"type": "text", "text": "{}"}],
                   "_observation": {"step": 0}, "_session_round": 2}
    ref = slot.materialize(message, catalog=catalog, round_index=1, action_step=0)
    assert ref["bytes"] == len(message["content"][0]["text"].encode())
    assert slot.invariant_content(message) == message["content"]
    aged, changed, refs = slot.externalize_prior_rounds([message, observation], through_action_steps=1)
    # The slot precedes any observation of round 2 here, so it is not inside a step.
    assert changed == [] and aged[0] is message
    inside = [dict(observation, _session_round=1), message,
              {"role": "user", "content": [], "_observation": {"step": 1}, "_session_round": 2}]
    aged, changed, refs = slot.externalize_prior_rounds(inside, through_action_steps=1)
    assert changed == [1] and refs == [ref]
    marker = json.loads(aged[1]["content"][0]["text"])
    assert marker["round_static_facts"] == {"round": 1}
    assert marker["recoverable_round_static_facts"]["ref"] == ref["ref"]
    assert slot.invariant_content(aged[1]) == message["content"]
    assert inside[1]["content"] == message["content"]  # copy-on-write
    tampered = dict(aged[1], content=[{"type": "text", "text": "{}"}])
    with pytest.raises(PolicySessionError):
        slot.invariant_content(tampered)
    # The current round's slot is never externalized, whatever the frontier.
    current = [dict(observation, _session_round=1), message,
               {"role": "user", "content": [], "_observation": {"step": 1}, "_session_round": 1}]
    assert slot.externalize_prior_rounds(current, through_action_steps=2)[1] == []


# -- The Native session -------------------------------------------------------


def _planner(tmp_path, backend, composition):
    from ora.agent.native.engine import NativeSessionEngine
    from test_current_visual_delivery import planner_for, published

    context, _, _, surface, _ = published(tmp_path, backend)
    planner = planner_for(tmp_path, context, surface)
    planner.engine = NativeSessionEngine()
    planner.actual_sequence = 0
    planner.context_composition = composition
    return planner, context


def _in_round(context, round_index):
    return {**context, "remaining_budget": {"session_round": round_index, "session_rounds": 3}}


def _current(message):
    return json.loads(message["content"][0]["text"])["current_actual"]


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
def test_native_slots_v4_states_each_round_once(tmp_path, backend):
    legacy_planner, context = _planner(tmp_path / "v3", backend, "native_slots_v3")
    legacy = _current(legacy_planner._causal_observation_message(
        context=_in_round(context, 1), terminal_lifecycle=False))
    assert "static_facts" not in legacy
    assert legacy_planner.engine.pending_round_static_facts is None

    planner, context = _planner(tmp_path / "v4", backend, slot.COMPOSITION)
    first = planner._causal_observation_message(context=_in_round(context, 1),
                                                terminal_lifecycle=False)
    pending = planner.engine.pending_round_static_facts
    fields = json.loads(pending["content"][0]["text"])["round_static_facts"]["current_actual"]
    assert fields == slot.capture(legacy)
    assert slot.restore(_current(first), fields) == legacy
    assert _current(first)["static_facts"]["round_slot"] == 1

    planner._append_round_static_facts({"event_id": "agent-event-000007"})
    assert planner.messages[-1] is pending
    assert pending["_source_event_ids"] == ["agent-event-000007"]
    assert planner.engine.pending_round_static_facts is None

    again = planner._causal_observation_message(context=_in_round(context, 1),
                                                terminal_lifecycle=False)
    assert planner.engine.pending_round_static_facts is None  # same round: no new slot
    restored = slot.restore(_current(again), fields)
    assert restored == {**legacy, "actual_sequence": restored["actual_sequence"]}

    planner._causal_observation_message(context=_in_round(context, 2), terminal_lifecycle=False)
    second = planner.engine.pending_round_static_facts
    assert second["_round_static_facts"] == {"round": 2}
    planner._append_round_static_facts({"event_id": "agent-event-000009"})
    planner._append_round_static_facts({"event_id": "agent-event-000010"})  # nothing pending
    assert [message.get("_round_static_facts") for message in planner.messages] == [
        {"round": 1}, {"round": 2}]


def test_native_slots_v4_needs_the_session_round(tmp_path):
    planner, context = _planner(tmp_path, "robotwin", slot.COMPOSITION)
    with pytest.raises(PolicySessionError, match="Session round"):
        planner._causal_observation_message(context=context, terminal_lifecycle=False)


def test_native_slots_v4_composes_with_once_per_payload(tmp_path):
    """D3a's conventions and D3b's slot together still restore the D3a delivery."""
    from delivery_lessons_fixtures import DEDUP, with_capabilities

    legacy_planner, context = _planner(tmp_path / "v3", "robotwin", "native_slots_v3")
    legacy_planner.tools.surface = with_capabilities(legacy_planner.tools.surface, DEDUP)
    legacy = _current(legacy_planner._causal_observation_message(
        context=_in_round(context, 1), terminal_lifecycle=False))
    assert "conventions" in legacy
    planner, context = _planner(tmp_path / "v4", "robotwin", slot.COMPOSITION)
    planner.tools.surface = with_capabilities(planner.tools.surface, DEDUP)
    first = _current(planner._causal_observation_message(context=_in_round(context, 1),
                                                         terminal_lifecycle=False))
    fields = json.loads(planner.engine.pending_round_static_facts["content"][0]["text"])[
        "round_static_facts"]["current_actual"]
    assert slot.restore(first, fields) == legacy
