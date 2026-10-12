"""Independent acceptance of the complete public-data successor."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.baseline import load_native_baseline
from ora.agent.contracts import PolicySessionError
from ora.orchestrator.defaults import expand_native_defaults
from test_native_baseline import native_spec
from test_current_visual_delivery import published, planner_for

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
@pytest.mark.parametrize("kind", ["native_api"])
def test_fresh_consumers_select_all_public_data(backend, kind):
    spec = native_spec(backend)
    spec["policy"]["kind"] = kind
    expanded = expand_native_defaults(spec, root=ROOT)
    surface = resolve_tool_surface(expanded["policy"]["tool_surface"]["profile"])
    # direct_v4 (integration 0059 unit A0): the v2 generation of each default; LIBERO's
    # is direct_v2's overhead successor since integration 0084.
    assert surface.profile_id == ("libero_placement_direct_overhead_v1" if backend == "libero_pro" else "robotwin_toolchain_b2_closed_noservo_v2")
    assert surface.supports("current_public_robot_facts_v1")
    assert surface.supports("full_current_visual_v1")
    assert surface.view_config["current_observation_delivery"] == "all_public_v1"
    assert surface.view_config["current_visual_delivery"] == "full_rgb_v1"
    descriptor_version = (
        "direct-v1-controller-feedback-v1-session-split-v1" if backend == "libero_pro" else "v5-session-split-v1"
    )
    assert surface.view_config["public_geometry_descriptor"]["path"].endswith(f"-{descriptor_version}.json")
    cameras = ["agentview", "wrist"] if backend == "libero_pro" else [
        "head_camera", "front_camera", "left_camera", "right_camera",
    ]
    assert surface.view_config["automatic_actual_views"] == cameras
    assert expanded["runtime"]["policy_cameras"] == cameras
    assert "observer_camera" not in json.dumps(surface.result_schemas)
    profile = expanded["runtime"]["environment_prompt_profile"]
    expected_profile = "libero_pro_public_direct_v1" if backend == "libero_pro" else "robotwin_public_explicit_v6"
    assert profile["profile_id"] == expected_profile
    # The current descriptors refresh source provenance while preserving the
    # reviewed v3 public cards; render equality guards that unchanged meaning.
    embodiment = "franka-panda" if backend == "libero_pro" else "aloha-agilex"
    card_path = f"ora/policy/embodiments/{embodiment}/manual-public-geometry-v3.md"
    from ora.agent.composition.embodiment import load_embodiment_descriptor, render_embodiment_card
    assert (ROOT / card_path).read_text() == render_embodiment_card(load_embodiment_descriptor(
        ROOT / surface.view_config["public_geometry_descriptor"]["path"]))
    assert card_path in expanded["policy_context"]["project_instructions"]
    card = (ROOT / card_path).read_text()
    assert "finger collision extent" in card
    assert "commands already target it" in card
    assert "observer_camera" not in card
    assert expanded["policy"]["provider_recovery"] == "same_credential_deadline_v1"


@pytest.mark.parametrize("backend", ["libero_pro", "robotwin"])
def test_missing_public_facts_cannot_silently_become_a_valid_current_message(tmp_path, backend):
    context, _, _, surface, _ = published(tmp_path, backend)
    planner = planner_for(tmp_path, context, surface)
    context = copy.deepcopy(context)
    del context["public_robot_facts"]
    with pytest.raises(PolicySessionError, match="public robot facts"):
        planner._causal_observation_message(context=context, terminal_lifecycle=False)


def test_new_geometry_binding_cannot_pair_with_old_manual(tmp_path):
    """A path-only default update must not leave the model reading the old card."""
    from ora.agent.composition.prompt_bundle import audit_prompt_bundle, load_prompt_bundle
    bundle = load_prompt_bundle(ROOT / load_native_baseline(ROOT)["instances"]["robotwin"]["prompt_bundle"])
    bundle = copy.deepcopy(bundle)
    bundle["layers"]["embodiment"]["source"] = "ora/policy/embodiments/aloha-agilex/manual-v1.md"
    report = audit_prompt_bundle(bundle, project_root=ROOT)
    assert not report["passed"]


ALIGN_SURFACES = {"robotwin": True, "libero": False}
#: The default surface of each backend.
ALIGN_PROFILES = {"robotwin": "robotwin_toolchain_b2_closed_noservo_v2", "libero": "libero_placement_direct_overhead_v1"}


def _align_definition(surface):
    return next(item for item in surface.definitions if item["name"] == "align_plan")


@pytest.mark.parametrize("backend,advertised", sorted(ALIGN_SURFACES.items()))
def test_align_plan_is_advertised_exactly_where_it_can_answer(backend, advertised):
    """RoboTwin gains the read-only reachability tool; LIBERO's gap is a decision.

    LIBERO is excluded on two measured grounds, not on a policy preference:
    ``LiberoProSession`` has no ``align_plan`` method, so every call would end
    as a non-retryable internal fault, and ``synthesise_approach_poses`` puts
    the approach direction on local +X, which the ALOHA descriptor declares as
    its approach axis and the Panda grip site declares as its closing axis.
    """
    surface = resolve_tool_surface(ALIGN_PROFILES[backend])
    names = {item["name"] for item in surface.definitions}
    assert ("align_plan" in names) is advertised
    assert surface.supports("read_only_ik_reachability_v1") is advertised
    if not advertised:
        assert "align_plan" not in surface.result_schemas
        return
    assert "align_plan" not in surface.mutation_tools
    assert "align_plan" not in surface.lifecycle_tools
    assert "align_plan" not in surface.observation_tools
    assert set(surface.result_schemas) == names


@pytest.mark.parametrize("kind", ["native_api"])
def test_fresh_robotwin_consumers_inherit_align_plan_without_an_overlay(kind):
    spec = native_spec("robotwin")
    spec["policy"]["kind"] = kind
    expanded = expand_native_defaults(spec, root=ROOT)
    surface = resolve_tool_surface(expanded["policy"]["tool_surface"]["profile"])
    assert "align_plan" in {item["name"] for item in surface.definitions}
    assert surface.supports("read_only_ik_reachability_v1")


def test_align_plan_help_makes_the_adjacent_tool_boundary_executable():
    """The help must separate it from grasp.plan and deny the arm reading."""
    surface = resolve_tool_surface(ALIGN_PROFILES["robotwin"])
    text = _align_definition(surface)["description"]
    # A policy_angles_degrees_v1 surface states its angles in degrees.
    angles = "in degrees" if surface.supports("policy_angles_degrees_v1") else "in radians"
    for marker in ("Use when:", "Effect:", "Returns:", "Does not:", "On error:"):
        assert marker in text
    # The clause names the adjacent tool as this surface spells it. A near miss
    # pointing at a name the surface does not advertise is worse than none.
    assert "use grasp_plan for contact geometry and candidate finger poses" in text
    assert "grasp_plan takes no approach_side or align candidate input" in text
    assert 'approach_side "left" does not mean the left arm' in text
    assert "directions toward the object and never toward an arm" in text
    for quantity in ("0.02-0.30 m", "0.10 m", "in metres", angles):
        assert quantity in text
    for non_guarantee in ("collision-free", "has not been tried", "visible-surface centroid"):
        assert non_guarantee in text
    for recovery in ("target_ref_unavailable", "stale_observation",
                     "target_geometry_unavailable", "When candidates are unreachable"):
        assert recovery in text


def test_align_plan_input_schema_survives_a_lossy_delivery_layer():
    """H6-012: a dangling $ref reaches the model as a definition it cannot see."""
    schema = _align_definition(resolve_tool_surface(ALIGN_PROFILES["robotwin"]))["inputSchema"]
    encoded = json.dumps(schema)
    assert "$ref" not in encoded and "$defs" not in encoded
    assert "oneOf" not in schema and schema["additionalProperties"] is False
    for name, child in schema["properties"].items():
        assert child["description"].strip(), name
        types = child["type"] if isinstance(child["type"], list) else [child["type"]]
        if "null" in types:  # optional_argument_null_as_omitted_v1: an optional field may be null
            assert name not in schema["required"] and len(types) == 2, name
        [kind] = [item for item in types if item != "null"]
        assert kind in {"string", "number", "integer"}, name
        if kind in {"number", "integer"}:
            assert {"minimum", "maximum", "default"} <= set(child), name
    assert schema["required"] == ["target_ref", "approach_side", "arm"]


def test_align_plan_schema_and_runtime_accept_the_same_calls():
    """Tightening only one side of the pair is the contract defect."""
    from ora.agent.composition.tool_surface import ToolSurfaceError, validate_schema_instance
    from ora.agent.tools.alignment import validate_arguments
    from ora.contracts.refusal import EnvironmentRefusal

    schema = _align_definition(resolve_tool_surface(ALIGN_PROFILES["robotwin"]))["inputSchema"]
    base = {"target_ref": "T0", "approach_side": "top", "arm": "left"}
    cases = [
        base,
        {**base, "standoff_m": 0.02, "max_candidates": 1},
        {**base, "arm": "both", "standoff_m": 0.3, "max_candidates": 6},
        {**base, "standoff_m": 0.019},
        {**base, "standoff_m": 0.31},
        {**base, "approach_side": "above"},
        {**base, "max_candidates": 0},
        {**base, "max_candidates": 7},
        {"target_ref": "T0", "approach_side": "top"},
        {**base, "unknown": 1},
    ]
    for case in cases:
        try:
            validate_schema_instance(dict(case), schema, root_path="$input")
            accepted_by_schema = True
        except ToolSurfaceError:
            accepted_by_schema = False
        try:
            validate_arguments(case)
            accepted_by_runtime = True
        except EnvironmentRefusal:
            accepted_by_runtime = False
        assert accepted_by_schema is accepted_by_runtime, case
    assert validate_arguments(base) == {
        **base, "standoff_m": 0.10, "max_candidates": 3,
    }, "the schema defaults must be the values the runtime actually applies"


def test_align_plan_result_schema_accepts_a_real_batch_and_rejects_drift():
    from ora.agent.composition.tool_surface import ToolSurfaceError, validate_schema_instance
    from ora.agent.tools.alignment import (
        build_request, frame_local_axes, validate_arguments,
    )
    from ora.environment.controllers.alignment import plan_alignment

    surface = resolve_tool_surface(ALIGN_PROFILES["robotwin"])
    schema = surface.result_schemas["align_plan"]

    class Adapter:
        """Answer every pose without a planner, alternating the three reasons."""

        def solve_batch_reachability(self, arm, poses, **tolerances):
            reasons = [None, "joint_limit", "singular", "not_converged"]
            answers = []
            for index, _pose in enumerate(poses):
                reason = reasons[index % len(reasons)]
                answer = {
                    "reachable": reason is None, "position_error_m": 0.0012,
                    "orientation_error_rad": 0.004, "iterations": 7,
                    "claim_boundary": "Inverse kinematics for this arm model only.",
                }
                if reason is not None:
                    answer["reason"] = reason
                answers.append(answer)
            return answers

    # arrival_check is what plan_alignment reads: local_cartesian is the local
    # servo controller's convergence bound, 2 mm tighter than what the backend
    # planner accepts, and reading it refused poses the planner would have solved.
    profile = {"profile": {"arrival_check": {
        "position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05,
    }, "local_cartesian": {
        "position_tolerance_m": 0.003, "orientation_tolerance_rad": 0.03,
    }}}
    request = build_request(
        validate_arguments({"target_ref": "T0", "approach_side": "top",
                            "arm": "both", "max_candidates": 6}),
        centre_xyz_m=[0.1, 0.0, 0.9],
        # Named rather than defaulted: align_plan is advertised on RoboTwin
        # here, and the module no longer assumes any one embodiment's frame.
        local_axes=frame_local_axes("aloha_grip_center_tcp"),
    )
    result = plan_alignment(request, adapter=Adapter(), control_profile=profile,
                            available_arms=("left", "right"))
    validate_schema_instance(result, schema)
    assert len(result["candidates"]) == 12
    assert result["motion_planner_called"] is False
    assert [c["rejection_reason"] for c in result["candidates"][:4]] == [
        None, "joint_limit", "singular", "not_converged",
    ]
    for drift in ({"motion_planner_called": True}, {"approach_side": "above"},
                  {"requested_arm": "either"}):
        with pytest.raises(ToolSurfaceError):
            validate_schema_instance({**result, **drift}, schema)
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance(
            {**result, "candidates": [{**result["candidates"][0], "reason": "joint_limit"}]},
            schema,
        )
