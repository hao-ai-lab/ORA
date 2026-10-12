"""The live Dojo surface carries the EXP11A g013 capabilities at once.

Revisions 14-17 of the retired ``robodojo_placement_closed_skills_v1`` were
developed on three parallel branches, each against revision 13:

- 14 ``policy_abort_requires_infrastructure_evidence_v1`` (finish_episode abort admission);
- 15 ``reach_nearest_orientation_v1`` (unsolved ``reach_along_request`` gains
  ``nearest_reachable_orientation``);
- 16 ``observed_clearance_reference_v1`` (``observed_clearance`` per pose target);
- 17 ``pose_orientation_from_world_axes_v1`` (``move_pose`` ``world_axes`` mode
  and ``target_axes_world`` per pose target).

Its successor ``robodojo_placement_closed_skills_v2`` (the RoboDojo default)
carries 15 and 17 with the v2 observed clearance and the v2 finish contract.
Each capability's own tests pin it alone. These pin the combination: every
pose tool's compiled target carries all three result fields, the three
``move_pose`` help sentences coexist in revision order, the binders run in one
deterministic order, and the Tool Runtime reaches the two per-target
annotations through one forwarder that refuses to overwrite a field.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.finish_outcomes_schema import (
    ABORT_CAPABILITY as ABORT_ROUND, CAPABILITY as FINISH_OUTCOMES,
)
from ora.agent.composition.observed_clearance_v2_schema import CAPABILITY as OBSERVED_CLEARANCE
from ora.agent.composition.reach_orientation_schema import CAPABILITY as NEAREST_ORIENTATION
from ora.agent.composition.world_axes_schema import CAPABILITY as WORLD_AXES, COMPANIONS, MODE
from ora.agent.tools import observed_clearance_runtime, pose_target_annotations, world_axes

DOJO = "robodojo_placement_closed_skills_v2"
ROOT = Path(__file__).resolve().parents[1]
POSE_TOOLS = ("move_pose", "move_to", "rotate_wrist", "rotate_pitch")
RESULT_FIELDS = {"reach_along_request", "observed_clearance", "target_axes_world"}
HELP_MARKERS = (  # revision order: 15, 16, 17
    "reach_along_request also carries nearest_reachable_orientation",
    "compiled_pose.targets.<arm>.observed_clearance (also on move_to",
    "world_axes compiles approach_world",
)


@pytest.fixture(scope="module")
def dojo():
    return resolve_tool_surface(DOJO)


def _definition(surface, name):
    return next(item for item in surface.definitions if item["name"] == name)


def _target_fields(surface, name):
    compiled = surface.result_schemas[name]["properties"]["compiled_pose"]
    return compiled["properties"]["targets"]["additionalProperties"]["properties"]


def test_revision_18_carries_the_kernel_units_policy_visible_components(dojo):
    refs = {item["slot"]: item["ref"] for item in dojo.components}
    assert refs["robodojo_wording_help"].endswith("robodojo-wording-help-v2.json")
    assert refs["robodojo_recovery_help"].endswith("robodojo-recovery-help-v3.json")
    assert refs["robodojo_grasp_hold_help"].endswith("robodojo-grasp-hold-help-v3.json")
    assert refs["robodojo_live_observe_results"].endswith("robodojo-live-observe-results-v4.json")
    assert dojo.view_config["public_geometry_descriptor"]["path"].endswith(
        "arx-x5/descriptor-public-geometry-v5.json")
    # The kernel text and the g013 appends coexist on move_pose, each once.
    description = _definition(dojo, "move_pose")["description"]
    assert description.count("and against the other arm (both arms together when one call moves both;") == 1
    assert all(description.count(marker) == 1 for marker in HELP_MARKERS)


@pytest.mark.parametrize("tool", POSE_TOOLS)
def test_every_pose_target_carries_all_three_result_fields(dojo, tool):
    fields = _target_fields(dojo, tool)
    assert RESULT_FIELDS <= set(fields)
    assert set(fields) == RESULT_FIELDS | ({"jaw_symmetry"} if tool == "move_pose" else set())
    unsolved = next(item for item in fields["reach_along_request"]["oneOf"]
                    if item.get("properties", {}).get("target_reachable") == {"const": False})
    assert "nearest_reachable_orientation" in unsolved["properties"]
    assert "nearest_reachable_orientation" in unsolved["required"]
    assert {variant["properties"]["status"]["const"] for variant in fields["observed_clearance"]["oneOf"]} == {
        "measured", "unknown"}
    assert set(fields["target_axes_world"]["required"]) == {"approach", "jaw_line"}


def test_move_pose_carries_the_world_axes_mode_and_three_help_sentences_in_revision_order(dojo):
    definition = _definition(dojo, "move_pose")
    for arm in ("left", "right"):
        target = definition["inputSchema"]["properties"]["targets"]["properties"][arm]["properties"]
        assert MODE in target["orientation_mode"]["enum"]
        assert all(name in target for name in COMPANIONS)
    description = definition["description"]
    positions = [description.find(marker) for marker in HELP_MARKERS]
    assert all(description.count(marker) == 1 for marker in HELP_MARKERS)
    assert -1 not in positions and positions == sorted(positions)
    assert f"orientation_mode (hold_current, absolute_quaternion, axis_angle_delta, {MODE})" in description


def test_the_binders_run_in_one_deterministic_order(dojo):
    derived = [item["slot"].removeprefix("derived:") for item in dojo.components
               if item["slot"].startswith("derived:")]
    order = ["pose_primitive_reach_along_request_v1", NEAREST_ORIENTATION, OBSERVED_CLEARANCE,
             FINISH_OUTCOMES, ABORT_ROUND, WORLD_AXES]
    assert [slot for slot in derived if slot in order] == order
    again = resolve_tool_surface(DOJO)
    assert json.dumps(again.definitions, sort_keys=True) == json.dumps(dojo.definitions, sort_keys=True)
    assert json.dumps(dict(again.result_schemas), sort_keys=True) == json.dumps(
        dict(dojo.result_schemas), sort_keys=True)


class _Host:
    surface = None


def test_the_runtime_forwarder_merges_the_reference_then_axes_then_clearance(monkeypatch):
    calls = []
    monkeypatch.setattr(world_axes, "echo", lambda surface, pose: calls.append(("axes", pose)) or {
        "target_axes_world": {"approach": [0.0, 0.0, -1.0], "jaw_line": [0.0, 1.0, 0.0]}})
    monkeypatch.setattr(observed_clearance_runtime, "reference",
                        lambda host, arm, target, *, observation, root: calls.append(("clearance", arm, target))
                        or {"observed_clearance": {"status": "unknown"}})
    compiled = {"target_pose_xyz_xyzw": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]}
    reference = {"reach_along_request": {"target_reachable": True}}
    merged = pose_target_annotations.merged(_Host(), "left", compiled, reference, {}, ROOT)
    assert list(merged) == ["reach_along_request", "target_axes_world", "observed_clearance"]
    assert calls == [("axes", {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}),
                     ("clearance", "left", compiled["target_pose_xyz_xyzw"])]
    assert reference == {"reach_along_request": {"target_reachable": True}}  # never mutated


def test_the_runtime_forwarder_refuses_to_overwrite_a_field(monkeypatch):
    monkeypatch.setattr(world_axes, "echo", lambda surface, pose: {"reach_along_request": {}})
    monkeypatch.setattr(observed_clearance_runtime, "reference", lambda *args, **kwargs: {})
    compiled = {"target_pose_xyz_xyzw": [0.0] * 6 + [1.0]}
    with pytest.raises(RuntimeError, match="reach_along_request"):
        pose_target_annotations.merged(_Host(), "left", compiled, {"reach_along_request": {}}, {}, ROOT)


def test_the_tool_runtime_reaches_the_pose_annotations_only_through_the_forwarder():
    import sys

    from ora.agent.tools.runtime import ManipulationToolRuntime

    # The runtime is runtime/core.py plus its tool-family mixin modules.
    files = sorted({
        sys.modules[klass.__module__].__file__ for klass in ManipulationToolRuntime.__mro__
        if klass.__module__.startswith("ora.agent.tools.runtime.")
    })
    source = "\n".join(Path(name).read_text(encoding="utf-8") for name in files)
    assert source.count("pose_target_annotations.merged(") == 2  # move_pose and the pose primitives
    assert "world_axes.echo(" not in source
    assert "observed_clearance_runtime.reference(" not in source
