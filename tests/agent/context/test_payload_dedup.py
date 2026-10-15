"""D3a: one observation or result payload states each repeated item once.

``within_payload_deduplication_v1`` keeps the current public robot facts as the
owner of the controller rule and of each arm's gripper feedback; the action
effect points there instead of repeating ~1.6 K characters. Convention strings
repeated per camera and per arm are stated once, and RoboDojo's per-segment IK
journal (Control evidence) leaves the effect. No reference crosses an
observation, so Context tail aging cannot leave one dangling.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

import pytest

from tests.support.delivery_lessons_fixtures import (
    DEDUP, NEW_CAPABILITIES, dojo_gripper_feedback, project, robodojo_receipt, with_capabilities,
)
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.context.delivery_trim import CAPABILITY as TRIM
from ora.agent.context.payload_dedup import (
    CAPABILITY, CONVENTIONS_KEY, FACTS_FIELD, REPORTED_IN_FACTS, hoist_fact_conventions,
)
from ora.contracts.robodojo_arx import public_feedback_contract

FACTS = ("public_motion_effect_v1", "current_public_robot_facts_v1", "contact_motion_feedback_v1")

#: The maintained defaults declare every delivery capability; the legacy
#: projection is the same resolved surface with all of them withdrawn
#: (runtime projection only), as the frozen direct_v3 surfaces delivered.
LIVE_PROFILES = {
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "libero_pro": "libero_placement_direct_overhead_v1",
    "robodojo": "robodojo_placement_closed_skills_v2",
}
DELIVERY = frozenset({*NEW_CAPABILITIES, TRIM})


def _legacy(backend):
    surface = resolve_tool_surface(LIVE_PROFILES[backend])
    assert surface.supports(DEDUP)
    return dataclasses.replace(
        surface, capabilities=tuple(item for item in surface.capabilities if item not in DELIVERY))


def test_capability_name_is_the_one_the_fixtures_declare():
    assert CAPABILITY == DEDUP


def test_effect_points_to_the_current_facts_and_drops_the_segment_journal():
    legacy = project(robodojo_receipt(), FACTS)
    delivered = project(robodojo_receipt(), (DEDUP, *FACTS))
    detail = delivered["detail"]
    assert detail["controller_feedback"] == {
        "status": REPORTED_IN_FACTS, "field": f"{FACTS_FIELD}.controller_feedback"}
    assert detail["gripper_feedback"] == {
        "status": REPORTED_IN_FACTS, "field": f"{FACTS_FIELD}.end_effectors.<arm>.gripper.feedback",
        "arms": ["left", "right"]}
    assert all("segments" not in arm for arm in detail["arms"].values())
    # Nothing else in the effect changes.
    for arm in legacy["detail"]["arms"].values():
        arm.pop("segments")
    for key in ("controller_feedback", "gripper_feedback"):
        legacy["detail"].pop(key)
        detail.pop(key)
    assert delivered == legacy


def _facts(arms=("left", "right")):
    def camera(name):
        return {"width": 640, "height": 480, "image_view": name, "resampling": "none",
                "pixel_coordinates": "opencv_top_left_xy", "native_pixel_to_image_offset_xy": [0, 0],
                "calibration": {"status": "reported_in_camera_geometry",
                                "pixel_basis": "native_rgb_before_sheet_translation"}}

    def arm(side, opening):
        feedback = dojo_gripper_feedback(measured=opening)
        feedback.pop("schema")
        return {"pose_xyz_xyzw": [0.1, 0.2, 0.9, 0, 0, 0, 1], "position_frame": "world",
                "position_unit": "metre", "orientation_order": "xyzw", "controlled_frame": "arx_x5_link6",
                "axes": {"approach_world": [0, 0, -1], "closing_world": [0, 1, 0]},
                "gripper": {"command_target": "open", "commanded_opening_normalized": 1.0,
                            "measured_opening_normalized": opening, "measured_aperture_mm": None,
                            "grasp_evidence": "not_established_by_joint_measurement",
                            "measurement_semantics": "prismatic_encoders_nominal_geometry",
                            "feedback": feedback}}

    return {"observation_id": "robodojo:o-5", "snapshot_role": "current",
            "controller_feedback": public_feedback_contract(),
            "cameras": {name: camera(name) for name in ("head", "left_wrist", "right_wrist")},
            "end_effectors": {side: arm(side, 0.97 if side == "left" else 0.99) for side in arms}}


def _restore(facts):
    """Put every stated-once convention back into each entry it describes."""
    def merge(target, shared):
        for key, value in shared.items():
            if isinstance(value, dict):
                merge(target.setdefault(key, {}), value)
            else:
                target[key] = value

    restored = copy.deepcopy(facts)
    for pattern, shared in restored.pop(CONVENTIONS_KEY, {}).items():
        collection = pattern.split(".")[0]
        for entry in restored[collection].values():
            merge(entry, shared)
    return restored


def test_conventions_are_stated_once_and_nothing_is_lost():
    facts = _facts()
    hoisted = hoist_fact_conventions(facts)
    assert facts == _facts()
    assert _restore(hoisted) == facts
    conventions = hoisted[CONVENTIONS_KEY]
    assert conventions["cameras.<camera>"] == {
        "pixel_coordinates": "opencv_top_left_xy", "resampling": "none",
        "calibration": {"status": "reported_in_camera_geometry",
                        "pixel_basis": "native_rgb_before_sheet_translation"}}
    shared_arm = conventions["end_effectors.<arm>"]
    assert shared_arm["controlled_frame"] == "arx_x5_link6"
    assert shared_arm["gripper"]["feedback"]["arrival_check"] == {"comparison": "<="}
    assert set(hoisted["cameras"]["head"]) == {
        "width", "height", "image_view", "native_pixel_to_image_offset_xy"}
    # State that merely coincides is not a convention and stays with its arm.
    assert hoisted["end_effectors"]["left"]["gripper"]["command_target"] == "open"
    assert hoisted["observation_id"] == "robodojo:o-5"
    assert len(json.dumps(hoisted)) < len(json.dumps(facts))


def test_a_differing_value_or_a_single_entry_stays_in_place():
    facts = _facts()
    facts["end_effectors"]["right"]["controlled_frame"] = "other_frame"
    hoisted = hoist_fact_conventions(facts)
    assert "controlled_frame" not in hoisted[CONVENTIONS_KEY]["end_effectors.<arm>"]
    assert hoisted["end_effectors"]["right"]["controlled_frame"] == "other_frame"
    assert _restore(hoisted) == facts
    single = _facts(arms=("right",))
    assert "end_effectors.<arm>" not in hoist_fact_conventions(single)[CONVENTIONS_KEY]


def test_an_observe_result_carrying_facts_is_deduplicated_too():
    from ora.agent.context.policy_delivery import deliver_tool_result

    surface = _legacy("robodojo")
    result = {"status": "actual_current_observation", "public_robot_facts": _facts()}
    assert deliver_tool_result(surface, "observe", result) is result
    delivered = deliver_tool_result(with_capabilities(surface, DEDUP), "observe", result)
    assert CONVENTIONS_KEY in delivered["public_robot_facts"]
    assert _restore(delivered["public_robot_facts"]) == result["public_robot_facts"]


def _native_observation(tmp_path, capabilities):
    """One RoboDojo action result as the Native session assembles it."""
    from ora.agent.context.codex import ContextProjector
    from ora.agent.evidence.policy_state import PolicyStateStore, REQUIRED_ABSENT
    from ora.contracts.robot_geometry import GEOMETRY_VERIFICATION_SCHEMA
    from tests.agent.context.test_current_visual_delivery import planner_for
    from tests.agent.native.test_native_causal_runtime import MANUAL, PROJECT, PROMPT
    from tests.environments.robodojo.test_robodojo_perception import _synthetic_robodojo_observation

    surface = with_capabilities(_legacy("robodojo"), *capabilities)
    source, observation = _synthetic_robodojo_observation(tmp_path / "obs")
    feedback = {arm: dojo_gripper_feedback(measured=1.0) for arm in ("left", "right")}
    for arm, eef in observation["proprioception"]["end_effectors"].items():
        eef["gripper_feedback"] = feedback[arm]
    observation.update(
        controller_feedback=public_feedback_contract(),
        forbidden_modalities_absent=sorted(REQUIRED_ABSENT),
        robot_geometry_verification={
            "schema": GEOMETRY_VERIFICATION_SCHEMA, "scope": "loaded_robot_geometry_only",
            "descriptor_sha256": surface.view_config["public_geometry_descriptor"]["sha256"],
            "passed": True})
    (source / "observation.json").write_text(json.dumps(observation, sort_keys=True, indent=2) + "\n")
    store = PolicyStateStore(tmp_path, "dedup-test", "plan-test")
    store.publish(observation=observation, observation_root=source, cause="initial",
                  task_instruction="Inspect", previous_public_action=None, remaining_budget={})
    projector = ContextProjector(tmp_path / "workspace", tmp_path / "control", PROMPT, MANUAL, PROJECT,
                                 run_root=tmp_path, main_camera="head", view_config=surface.view_config,
                                 capabilities=surface.capabilities)
    context = projector.publish(observation=observation, observation_root=source,
                                task_instruction="Inspect", remaining_budget={},
                                policy_state=store.steps[-1])
    planner = planner_for(tmp_path, context, surface)
    planner._causal_observation_message(context=context, terminal_lifecycle=False)
    planner.pending_receipt_body = robodojo_receipt(feedback=feedback)
    message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
    return message["content"][0]["text"]


def _resolve(envelope, field, arm=None):
    value = envelope
    for step in field.replace("<arm>", arm or "<arm>").split("."):
        value = value[step]
    return value


def test_pointers_resolve_inside_the_same_native_observation(tmp_path):
    legacy_text = _native_observation(tmp_path / "legacy", ())
    text = _native_observation(tmp_path / "dedup", (DEDUP,))
    legacy, envelope = json.loads(legacy_text), json.loads(text)
    detail = envelope["last_action_effect"]["detail"]
    owner = _resolve(envelope, detail["controller_feedback"]["field"])
    assert owner == legacy["last_action_effect"]["detail"]["controller_feedback"]
    assert text.count(json.dumps(owner, ensure_ascii=False, sort_keys=True)) == 1
    facts = _restore(envelope["current_actual"]["public_robot_facts"])
    for arm in detail["gripper_feedback"]["arms"]:
        field = detail["gripper_feedback"]["field"].replace(f"{FACTS_FIELD}.", "")
        resolved = _resolve(facts, field, arm)
        assert resolved == legacy["last_action_effect"]["detail"]["gripper_feedback"][arm]
    # The restored facts are exactly what the legacy observation delivered.
    assert facts == legacy["current_actual"]["public_robot_facts"]
    assert len(text.encode()) < len(legacy_text.encode())
    assert hashlib.sha256(legacy_text.encode()).hexdigest() != hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize("backend", sorted(LIVE_PROFILES))
def test_without_the_capability_the_projections_are_untouched(backend):
    from ora.agent.context.policy_delivery import deliver_public_facts

    facts = _facts()
    surface = _legacy(backend)
    assert deliver_public_facts(facts, capabilities=surface.capabilities) is facts


CAMERA_GEOMETRY_CONVENTIONS = {
    "pixel_coordinates", "camera_coordinates", "extrinsic_convention", "depth_semantics",
    "depth_unit",
}


def test_native_camera_geometry_states_its_five_conventions_once(tmp_path):
    legacy = json.loads(_native_observation(tmp_path / "legacy", ()))["current_actual"]
    current = json.loads(_native_observation(tmp_path / "dedup", (DEDUP,)))["current_actual"]
    shared = current[CONVENTIONS_KEY]["camera_geometry.<camera>"]
    assert set(shared) == CAMERA_GEOMETRY_CONVENTIONS
    assert set(current["camera_geometry"]) == set(legacy["camera_geometry"]) == {
        "head", "left_wrist", "right_wrist"}
    for name, entry in current["camera_geometry"].items():
        assert not CAMERA_GEOMETRY_CONVENTIONS & set(entry)
        assert {**entry, **shared} == legacy["camera_geometry"][name]
    # Everything else in current_actual is what the legacy observation carried.
    restored = {key: value for key, value in current.items() if key != CONVENTIONS_KEY}
    restored["camera_geometry"] = {name: {**entry, **shared}
                                   for name, entry in current["camera_geometry"].items()}
    restored["public_robot_facts"] = _restore(restored["public_robot_facts"])
    assert restored == legacy


def test_camera_geometry_hoisting_keeps_a_differing_or_single_camera_in_place():
    from ora.agent.context.payload_dedup import hoist_camera_geometry_conventions

    def geometry(depth_unit="metres"):
        return {"intrinsic_cv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "depth_unit": depth_unit,
                "pixel_coordinates": "origin top-left; u right, v down"}

    actual = {"actual_sequence": 3, "camera_geometry": {"a": geometry(), "b": geometry("mm")}}
    hoisted = hoist_camera_geometry_conventions(actual)
    assert hoisted[CONVENTIONS_KEY] == {
        "camera_geometry.<camera>": {"pixel_coordinates": "origin top-left; u right, v down"}}
    assert hoisted["camera_geometry"]["b"]["depth_unit"] == "mm"
    assert actual["camera_geometry"]["a"]["pixel_coordinates"]  # the input is not mutated
    single = {"camera_geometry": {"a": geometry()}}
    assert hoist_camera_geometry_conventions(single) == single
    assert hoist_camera_geometry_conventions({"actual_sequence": 1}) == {"actual_sequence": 1}
