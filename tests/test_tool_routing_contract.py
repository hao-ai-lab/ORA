"""Validate model-visible producer/consumer handoffs on selected surfaces."""
from __future__ import annotations

import copy
import json

import pytest

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.composition.tool_surface import ToolSurfaceError

PROFILES = [
    "libero_placement_direct_overhead_v1",
    "robotwin_toolchain_b2_closed_noservo_v1", "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo_placement_closed_skills_v2",
]


@pytest.mark.parametrize("profile", PROFILES)
def test_actual_producer_help_examples_validate_against_selected_consumer(profile):
    definitions = {item["name"]: item for item in resolve_tool_surface(profile).definitions}
    for producer in ("grasp_plan", "place_plan", "align_plan"):
        if producer not in definitions:
            continue
        examples = [json.loads(line) for line in definitions[producer]["description"].splitlines()
                    if line.startswith('{"tool":')]
        assert examples, f"{profile}/{producer} has no complete handoff example"
        for example in examples:
            consumer = definitions[example["tool"]]
            validate_schema_instance(example["arguments"], consumer["inputSchema"])
            # Correct examples must not be achieved by relaxing the action gate.
            old_shape = copy.deepcopy(example["arguments"])
            pose = next(iter(old_shape["targets"].values())) if "targets" in old_shape else old_shape
            pose.pop("position_mode")
            pose.pop("orientation_mode")
            with pytest.raises(ToolSurfaceError):
                validate_schema_instance(old_shape, consumer["inputSchema"])


@pytest.mark.parametrize("profile", PROFILES)
def test_execution_routes_only_recommend_capabilities_on_that_surface(profile):
    definitions = {item["name"]: item for item in resolve_tool_surface(profile).definitions}
    grasp = definitions["grasp_plan"]["description"]
    if "place_plan" not in definitions:
        # The live closed-skills line retired place_plan, place_execute and
        # place_resume by undeclaration (2026-09-29): no remaining help may
        # route a policy to them.
        if "grasp_execute" in definitions:
            assert "grasp_execute(candidate_id, rationale)" in grasp
            assert "intermediate decisions" in grasp
        else:
            # The atomic LIBERO direct line declares no closed skill at all.
            for name, item in definitions.items():
                assert "grasp_execute" not in item["description"], name
        assert {"place_execute", "place_resume"}.isdisjoint(definitions)
        for name, item in definitions.items():
            for retired in ("place_plan", "place_execute", "place_resume"):
                assert retired not in item["description"], (name, retired)
        return
    place = definitions["place_plan"]["description"]
    if "grasp_execute" in definitions:
        assert "grasp_execute(candidate_id, rationale)" in grasp
        assert "place_execute(candidate_id, rationale)" in place
        assert "place_resume" in place
        assert "intermediate decisions" in grasp
    else:
        assert "grasp_execute" not in grasp
        assert "grasp_execute" not in definitions["set_gripper"]["description"]
        assert "place_execute" not in place
        assert "place_resume" not in place
        assert "place_verify" in definitions
