"""``observation_delivery_trim_v2``: the policy's copy leaves out what carries no information.

Each rule removes one named field at one place, only at the stated value, and
the information stays in the same request (the facts beside the effect, the
round's static-facts slot, the embodiment card or the surface's own
description layer). The canonical observation, receipt and tool-result event
keep every byte; a surface that does not declare the capability receives the
very object it passed in.
"""

from __future__ import annotations

import json
from pathlib import Path

from ora.agent.context import delivery_trim as trim
from ora.agent.context import delivery_trim_native as native
from ora.agent.context.policy_delivery import deliver_public_facts

DATA = Path(__file__).parent / "data" / "delivery_trim"


def recorded(backend: str) -> dict:
    return json.loads((DATA / f"{backend}-observation.json").read_text(encoding="utf-8"))


def _camera(name, **projections):
    return {"width": 640, "height": 480, "image_view": name,
            "native_pixel_to_image_offset_xy": [0, 0], **projections}


def test_identity_defaults_go_and_every_other_value_stays():
    facts = {"cameras": {"head": _camera("head"),
                         "front_camera": {**_camera("front_camera"), "native_pixel_to_image_offset_xy": [12, 24],
                                          "image_view": "contact_sheet"}}}
    cameras = trim.trim_public_facts(facts)["cameras"]
    assert cameras["head"] == {"width": 640, "height": 480}
    assert cameras["front_camera"]["native_pixel_to_image_offset_xy"] == [12, 24]
    assert cameras["front_camera"]["image_view"] == "contact_sheet"


def test_frame_labels_leave_per_arm_and_shared_conventions():
    single = {"end_effectors": {"right": {"orientation_order": "xyzw", "controlled_frame": "panda_grip_site",
                                          "position_frame": "world", "pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1]}}}
    assert trim.trim_public_facts(single)["end_effectors"]["right"] == {
        "position_frame": "world", "pose_xyz_xyzw": [0, 0, 1, 0, 0, 0, 1]}
    shared = {"conventions": {"end_effectors.<arm>": {"orientation_order": "xyzw", "controlled_frame": "arx_x5_link6",
                                                      "position_unit": "metre"},
                              "cameras.<camera>": {"resampling": "none"}}}
    assert trim.trim_public_facts(shared)["conventions"] == {
        "end_effectors.<arm>": {"position_unit": "metre"}, "cameras.<camera>": {"resampling": "none"}}
    only_labels = {"conventions": {"end_effectors.<arm>": {"orientation_order": "xyzw"}}}
    assert "conventions" not in trim.trim_public_facts(only_labels)


def test_the_arrival_tolerance_goes_only_when_the_controller_states_the_same_value():
    def facts(own, controller):
        return {"controller_feedback": {"gripper_tolerance_normalized": controller},
                "end_effectors": {"right": {"gripper": {"feedback": {"arrival_check": {
                    "residual_normalized": 0.0002, "tolerance_normalized": own, "within_tolerance": True}}}}}}

    same = trim.trim_public_facts(facts(0.03, 0.03))
    check = same["end_effectors"]["right"]["gripper"]["feedback"]["arrival_check"]
    assert check == {"residual_normalized": 0.0002, "within_tolerance": True}
    assert same["controller_feedback"]["gripper_tolerance_normalized"] == 0.03
    differs = trim.trim_public_facts(facts(0.05, 0.03))
    assert differs["end_effectors"]["right"]["gripper"]["feedback"]["arrival_check"]["tolerance_normalized"] == 0.05
    no_controller = trim.trim_public_facts({"end_effectors": facts(0.03, 0.03)["end_effectors"]})
    assert "tolerance_normalized" in no_controller["end_effectors"]["right"]["gripper"]["feedback"]["arrival_check"]


def _effect(**detail):
    return {"action": "move_pose", "status": "reached", "detail": detail}


def test_an_untargeted_arm_loses_its_nulls_and_its_noise_drift():
    arms = {"right": {"targeted": True, "reached": None},
            "left": {"targeted": False, "reached": None,
                     "inactive_arm_drift": {"translation_m": 0.00005, "orientation_deg": 0.005}},
            "third": {"targeted": False, "inactive_arm_drift": {"translation_m": 0.01, "orientation_deg": 0.0}}}
    detail = trim.trim_action_effect(_effect(arms=arms, target_arms=["right"]))["detail"]
    assert detail["arms"] == {"right": {"targeted": True, "reached": None},
                              "third": {"targeted": False,
                                        "inactive_arm_drift": {"translation_m": 0.01, "orientation_deg": 0.0}}}


def _description(surface_key: str, tool: str) -> str:
    import test_published_limits as limits

    return next(item["description"] for item in limits._surface(surface_key).definitions if item["name"] == tool)


def test_the_default_help_states_none_of_the_v1_omissions():
    # v2 (the default surfaces) restores those fields, so its help states none of them.
    for surface in ("robotwin_v2", "libero_overhead", "robodojo_v2"):
        observe = _description(surface, "observe")
        assert "omits claim" not in observe and "Delivered measurements are rounded" not in observe
        assert "is the string behind_camera or outside_image" not in observe


def test_a_surface_without_the_capability_receives_the_very_object():
    facts = recorded("robodojo")["round_static_facts"]["current_actual"]["public_robot_facts"]
    assert deliver_public_facts(facts, capabilities=()) is facts
    assert trim.CAPABILITY == "observation_delivery_trim_v2"


def test_the_capability_runs_last_on_the_delivered_form():
    facts = recorded("robotwin")["round_static_facts"]["current_actual"]["public_robot_facts"]
    delivered = deliver_public_facts(facts, capabilities=(trim.CAPABILITY,))
    assert delivered == trim.trim_public_facts(facts)


def test_the_core_prompt_states_the_comparison_and_the_defaults_once():
    root = Path(__file__).resolve().parents[1]
    core = (root / "ora/policy/core/evidence-action-loop-v8.md").read_text(encoding="utf-8")
    assert "compare that ACTION INPUT ACTUAL with this ACTION RESULT ACTUAL before choosing the next tool" in core
    assert "terminal_lifecycle only when it is true" in core
    assert "fresh_for_this_action only when that differs from comparison.available" in core
    assert "static_facts.absent" in core and "static_facts.unchanged" not in core


def test_trim_observation_envelope_leaves_out_only_the_fixed_values():
    comparison = {"available": True, "action_input_actual_sequence": 10,
                  "action_result_actual_sequence": 11, "instruction": "Use this initial actual evidence."}
    envelope = {"comparison": comparison, "terminal_lifecycle": False, "current_actual": {"actual_sequence": 11}}
    assert native.trim_observation_envelope(envelope) == {
        "comparison": comparison, "current_actual": {"actual_sequence": 11}}
    kept = {"comparison": {"instruction": "something else"}, "terminal_lifecycle": True}
    assert native.trim_observation_envelope(kept) == kept


def test_view_freshness_leaves_only_when_it_equals_the_pairing():
    paired = {"current_views": {"fresh_for_this_action": True, "status": "complete"}}
    assert native.trim_view_freshness(paired, paired=True) == {"current_views": {"status": "complete"}}
    initial = {"current_views": {"fresh_for_this_action": False}}
    assert native.trim_view_freshness(initial, paired=False) == {"current_views": {}}
    stale = {"current_views": {"fresh_for_this_action": False, "status": "complete"}}
    assert native.trim_view_freshness(stale, paired=True) == stale
