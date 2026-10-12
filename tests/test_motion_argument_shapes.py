"""Both published spellings of a move_pose target, and the unions removed.

The companion of `test_placement_argument_shapes.py`. `place.*` was flattened
first; the same union stayed on `move_pose`, which the sampled corpus calls
heavily -- the most expensive place in the surface to keep a shape whose
refusal names no field. Its flat schema is a `schema_order` component and is
visible after `resolve_tool_surface`.
"""

from __future__ import annotations

import json

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import PROFILE_CATALOG
from ora.agent.tools import motion_arguments
from ora.agent.tools.robot import ToolError


UNION_KEYWORDS = ("oneOf", "anyOf", "allOf", "$ref", "$defs")


def _flat_motion_profiles() -> list[str]:
    """Derived from the capability, never listed by name.

    A name list is a second copy of the manifests' own answer, and a sibling
    that hash-reuses these components inherits the flat form without appearing
    in it.
    """
    return sorted(
        profile for profile in PROFILE_CATALOG
        if resolve_tool_surface(profile).supports(motion_arguments.CAPABILITY)
    )


FLAT_PROFILES = _flat_motion_profiles()


def _move_pose(profile: str) -> dict:
    surface = resolve_tool_surface(profile)
    return next(
        item for item in surface.definitions if item["name"] == "move_pose"
    )


def test_the_capability_reaches_the_surfaces_that_publish_the_flat_form():
    """Neither half of the pair may land without the other."""
    assert FLAT_PROFILES, "no profile declares the flat motion arguments"
    for profile in FLAT_PROFILES:
        assert motion_arguments.declares_flat_targets(_move_pose(profile)["inputSchema"])


@pytest.mark.parametrize("profile", FLAT_PROFILES)
def test_a_flat_move_pose_carries_no_union_and_names_its_new_fields(profile: str):
    """Both repaired defects, on the delivered shape rather than asserted.

    `anyOf` is listed beside `oneOf` deliberately: expressing "delta also needs
    translation_m" as a union would reproduce H6-013 while looking like a schema
    improvement.
    """
    definition = _move_pose(profile)
    encoded = json.dumps(definition["inputSchema"])
    for forbidden in UNION_KEYWORDS:
        assert forbidden not in encoded, f"move_pose still delivers {forbidden}"
    # Field-level descriptions are delivered to the provider with the schema,
    # including on Dojo's frozen descriptions. Check every advertised arm.
    schema = definition["inputSchema"]
    targets = schema["properties"].get("targets", {}).get("properties") or {"right": schema}
    for field in ("position_mode", "orientation_mode", "rotation_angle_deg"):
        for arm, target in targets.items():
            parameter = target["properties"][field]
            assert field in definition["description"] or parameter.get("description"), (arm, field)


@pytest.mark.parametrize("target,expected", [
    ({"position_mode": "hold_current", "orientation_mode": "hold_current"},
     {"position": {"mode": "hold_current"}, "orientation": {"mode": "hold_current"}}),
    ({"position_mode": "absolute", "position_m": [0.1, 0.2, 0.3],
      "orientation_mode": "hold_current"},
     {"position": {"mode": "absolute", "position_m": [0.1, 0.2, 0.3]},
      "orientation": {"mode": "hold_current"}}),
    ({"position_mode": "delta", "translation_m": [0.0, 0.0, -0.05],
      "position_frame": "end_effector", "orientation_mode": "hold_current",
      "interaction": "contact_approach"},
     {"position": {"mode": "delta", "translation_m": [0.0, 0.0, -0.05],
                   "frame": "end_effector"},
      "orientation": {"mode": "hold_current"}, "interaction": "contact_approach"}),
    ({"position_mode": "hold_current", "orientation_mode": "axis_angle_delta",
      "rotation_axis": "z", "rotation_angle_deg": 15.0, "orientation_frame": "world"},
     {"position": {"mode": "hold_current"},
      "orientation": {"mode": "axis_angle_delta", "axis": "z", "angle_deg": 15.0,
                      "frame": "world"}}),
])
def test_the_flat_spelling_compiles_to_the_shape_the_compiler_already_took(
    target: dict, expected: dict,
):
    assert motion_arguments.arm_target(target) == expected


@pytest.mark.parametrize("target", [
    {"position": {"mode": "hold_current"}, "orientation": {"mode": "hold_current"}},
    {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
])
def test_the_nested_and_canonical_spellings_still_read_unchanged(target: dict):
    """A frozen surface's ergonomic form, and the pose every producer hands over."""
    assert motion_arguments.arm_target(target) == target


@pytest.mark.parametrize("target,field", [
    ({"position_mode": "delta", "position_frame": "world",
      "orientation_mode": "hold_current"}, "targets.left.translation_m"),
    ({"position_mode": "absolute", "orientation_mode": "hold_current"},
     "targets.left.position_m"),
    ({"position_mode": "hold_current", "position_m": [0, 0, 0],
      "orientation_mode": "hold_current"}, "targets.left.position_m"),
    ({"position_mode": "sideways", "orientation_mode": "hold_current"},
     "targets.left.position_mode"),
    ({"position_mode": "hold_current", "orientation_mode": "axis_angle_delta",
      "rotation_axis": "z", "orientation_frame": "world"},
     "targets.left.rotation_angle_deg"),
    ({"position_mode": "hold_current", "orientation_mode": "hold_current",
      "pose": "up"}, "targets.left.pose"),
    ({"position_mode": "hold_current", "orientation_mode": "hold_current",
      "interaction": "curobo"}, "targets.left.interaction"),
])
def test_a_refusal_names_the_field_rather_than_a_branch_index(target: dict, field: str):
    """The whole point of the flattening.

    The nested union answered a malformed position with "$input must match
    exactly one input branch; oneOf[0]: $input.targets.left must match exactly
    one input branch" and empty constraints, naming neither field nor bound.
    """
    with pytest.raises(ToolError) as raised:
        motion_arguments.arm_target(target, arm="targets.left")
    assert raised.value.field == field
    assert raised.value.code == "invalid_arguments"
    assert raised.value.retryable is True


@pytest.mark.parametrize("profile", FLAT_PROFILES)
def test_the_copyable_grasp_target_is_published_in_the_form_its_surface_accepts(
    profile: str,
):
    """A target published to be copied unchanged must survive being copied.

    `grasp.plan` publishes `approach_delta` for the policy to paste into
    `move_pose`. Flattening the input without flattening what the producer
    publishes would have left a field whose help says "copy unchanged" and whose
    own surface refuses it.
    """
    schema = _move_pose(profile)["inputSchema"]
    published = motion_arguments.arm_target_fields(
        schema,
        position={"mode": "delta", "translation_m": [0.0, 0.0, -0.05],
                  "frame": "end_effector"},
        orientation={"mode": "hold_current"},
        interaction="contact_approach",
    )
    assert published == {
        "position_mode": "delta", "translation_m": [0.0, 0.0, -0.05],
        "position_frame": "end_effector", "orientation_mode": "hold_current",
        "interaction": "contact_approach",
    }
    # And it reads back through the tool that receives it.
    assert motion_arguments.arm_target(published)["position"]["mode"] == "delta"


def test_a_nested_surface_still_gets_the_nested_publication():
    """The write half follows the schema, so one producer serves both spellings."""
    nested = motion_arguments.arm_target_fields(
        {"properties": {"targets": {}}},
        position={"mode": "delta", "translation_m": [0.0, 0.0, -0.05],
                  "frame": "end_effector"},
        orientation={"mode": "hold_current"},
        interaction="contact_approach",
    )
    assert nested == {
        "position": {"mode": "delta", "translation_m": [0.0, 0.0, -0.05],
                     "frame": "end_effector"},
        "orientation": {"mode": "hold_current"},
        "interaction": "contact_approach",
    }


@pytest.mark.parametrize("profile", FLAT_PROFILES)
def test_every_published_copyable_bundle_survives_its_own_move_pose(profile: str):
    """The defect class flattening an input creates, guarded once for all producers.

    Two tools publish arguments a policy is told to copy into ``move_pose``
    unchanged: ``grasp.plan``'s ``approach_delta`` and each ``place.plan``
    candidate's ``move_pose``. Flattening the consumer without flattening both
    producers leaves a field whose help says "copy unchanged" and whose own
    surface rejects it at the schema gate, before the handler's dual-read is
    ever reached. Checked against the published result schemas rather than
    against the producers, because the schema is what a reader is promised.
    """
    from ora.agent.composition import validate_schema_instance

    surface = resolve_tool_surface(profile)
    move_pose = _move_pose(profile)["inputSchema"]
    dual = "targets" in move_pose.get("properties", {})

    approach = motion_arguments.arm_target_fields(
        move_pose,
        position={"mode": "delta", "translation_m": [0.0, 0.0, -0.05],
                  "frame": "end_effector"},
        orientation={"mode": "hold_current"},
        interaction="contact_approach",
    )
    validate_schema_instance(
        {"targets": {"right": approach}} if dual else approach, move_pose,
    )

    candidate = motion_arguments.arm_target_fields(
        move_pose,
        position={"mode": "absolute", "position_m": [0.1, 0.2, 0.3]},
        orientation={"mode": "absolute_quaternion",
                     "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
        interaction="free_space",
    )
    published = {"targets": {"right": candidate}} if dual else candidate
    validate_schema_instance(published, move_pose)

    schemas = surface.result_schemas
    place_plan = schemas.get("place.plan") or schemas.get("place_plan")
    if place_plan is None:
        # The live closed-skills surfaces retired place_plan by undeclaration
        # (2026-09-29), so they publish no placement candidate to copy.
        assert {item["name"] for item in surface.definitions}.isdisjoint({"place.plan", "place_plan"})
        return
    declared = place_plan["properties"]["candidates"]["items"]["properties"]["move_pose"]
    validate_schema_instance(published, declared)
