"""Default surfaces reach both provider wires with complete help and closed schemas.

The positive H-admission checks, bound to the exp08h package code, moved to
``test_exp08h_placement_bundle.py`` with relocation U7 (2026-09-28). The
correctness and unified surfaces these checks were written against are retired;
the same projection, grounding and acknowledgement mechanisms are pinned on the
maintained defaults.
"""

from __future__ import annotations

import copy
import json

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import validate_tool_help_contract
from ora.agent.native.gateway import openai_tool_definitions
from ora.agent.providers.native import translate_anthropic_request


DEFAULTS = {
    "libero": "libero_placement_direct_overhead_v1",
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo": "robodojo_placement_closed_skills_v2",
}


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
@pytest.mark.parametrize("candidate_count", [0, 1, 2])
def test_unified_grounding_composes_preview_and_perception_results_through_real_dispatch(
    tmp_path, monkeypatch, backend, candidate_count,
):
    from types import SimpleNamespace
    from tests.agent.perception.test_grounding_grasp_runtime import _runtime, _select_correctness_surface
    from ora.agent.native.session import NativePlanner

    runtime, _ = _runtime(tmp_path)
    _select_correctness_surface(runtime, DEFAULTS[backend])
    camera = "agentview"
    if backend == "robotwin":
        control, root, observation, pose, gripper = runtime._current_grip_state()
        observation["cameras"] = {"head_camera": observation["cameras"]["agentview"]}
        runtime._current_grip_state = lambda: (control, root, observation, pose, gripper)
        camera = "head_camera"
    ground = runtime.grounding_backend.ground

    def bounded_ground(**arguments):
        candidates, diagnostics = ground(**arguments)
        candidates = candidates[:candidate_count]
        return candidates, {**diagnostics, "returned_candidate_count": len(candidates)}

    monkeypatch.setattr(runtime.grounding_backend, "ground", bounded_ground)
    result = runtime.invoke("vision_ground", {"camera": camera, "text": "red object"})
    assert len(result["candidates"]) == candidate_count
    assert len(result["candidate_previews"]) == candidate_count
    assert len(result["images_base64"]) == candidate_count + 1
    assert result["perception_evidence"]["semantic_identity_verified"] is False
    planner = object.__new__(NativePlanner)
    planner.tools = SimpleNamespace(surface=runtime.surface)
    visible = planner._causal_tool_result("vision_ground", result)
    assert visible["candidate_previews"] == [
        {key: value for key, value in item.items() if key not in {"sha256", "path"}}
        for item in result["candidate_previews"]
    ]
    assert visible["perception_evidence"]["semantic_identity_verified"] is False
    # diagnostics is a producer-built block the result schema declares for the
    # policy (grasp.plan keeps its candidate ledger there), so a declaration
    # now makes it promised evidence. backend stays provenance and stays out.
    assert "backend" not in visible


@pytest.mark.parametrize("backend", ["libero", "robotwin"])
@pytest.mark.parametrize("tool", ["move_to", "move_pose", "rotate_wrist", "rotate_pitch"])
def test_unified_motion_ack_is_not_a_consumed_execution_receipt(tmp_path, monkeypatch, backend, tool):
    from tests.support.holding_fixtures import PublicEpisode
    from tests.environments.robotwin.test_robotwin_placement_adapter import RoboTwinPublicRun
    from ora.agent.native.session import NativePlanner

    if backend == "libero":
        episode = PublicEpisode(tmp_path, monkeypatch, profile=DEFAULTS["libero"])
        runtime, pose = episode.runtime, episode.pose
        args = {"angle_deg": 2} if tool.startswith("rotate") else {"position_m": [pose[0], pose[1], pose[2] + 0.005]}
        if tool == "move_pose":
            args.update(position_mode="absolute", orientation_mode="absolute_quaternion",
                        orientation_xyzw=pose[3:])
    else:
        episode = RoboTwinPublicRun(tmp_path, profile=DEFAULTS["robotwin"], bind_submitted=True)
        runtime, pose = episode.runtime, episode.frames[-1]["poses"]["left"]
        target = 2 if tool.startswith("rotate") else {"position_m": [pose[0], pose[1], pose[2] + 0.005]}
        if tool == "move_pose":
            target.update(position_mode="absolute", orientation_mode="absolute_quaternion",
                          orientation_xyzw=pose[3:])
        args = {"targets": {"left": target}}
    result = runtime.invoke(tool, args)
    planner = object.__new__(NativePlanner)
    planner.tools = runtime
    visible = planner._causal_tool_result(tool, result)
    assert visible["status"] == "submitted"
    assert visible["compiled_pose"]
    assert "action" in visible
    assert not {"execution_status", "motion_status", "measured_endpoint", "consumed"} & visible.keys()


@pytest.mark.parametrize("profile", sorted(DEFAULTS.values()))
def test_default_complete_help_and_closed_schema_survive_real_provider_projection(profile):
    surface = resolve_tool_surface(profile)
    original = copy.deepcopy(surface.definitions)
    validate_tool_help_contract(surface.definitions, surface.result_schemas)
    names = [item["name"] for item in surface.definitions]
    assert len(names) == len(set(names))
    assert set(surface.mutation_tools) < set(names)
    assert surface.lifecycle_tools == ("finish_episode",)
    assert set(surface.result_schemas) == set(names)

    projected = openai_tool_definitions(names, definitions=surface.definitions)
    assert [item["function"]["name"] for item in projected] == names
    body = translate_anthropic_request({"model": "cpu-fixture", "max_tokens": 64,
        "messages": [{"role": "user", "content": "Schema projection only."}],
        "tools": projected})
    assert len(body["tools"]) == len(names)
    assert len({item["name"] for item in body["tools"]}) == len(names)
    aliases = {}
    for block in body.get("system", []):
        text = block.get("text", "")
        if text.startswith("<wire_tool_aliases>\n"):
            aliases = json.loads(text.split("\n", 1)[1].rsplit("\n", 1)[0])
    for original_tool, native, wire in zip(surface.definitions, projected, body["tools"], strict=True):
        name = original_tool["name"]
        assert wire["name"] == aliases.get(name, name)
        assert wire["description"] == native["function"]["description"] == original_tool["description"]
        schema = native["function"]["parameters"]
        expected_wire_schema = ({"type": "object", **schema}
                                if "type" not in schema and "oneOf" in schema else schema)
        assert wire["input_schema"] == expected_wire_schema
        assert wire["input_schema"]["type"] == "object"
        branches = schema.get("oneOf", [schema])
        assert all(branch["type"] == "object" and branch["additionalProperties"] is False
                   for branch in branches)
    assert surface.definitions == original
