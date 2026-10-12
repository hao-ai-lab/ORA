"""The RoboDojo closed-skills default carries Dojo backend facts, not RoboTwin ones.

`robodojo_placement_closed_skills_v2` is the RoboTwin B2 closed arm re-bound to
the ARX X5 (`test_robodojo_closed_skill_surface_contract_v2.py` holds its tool
set, effects, capabilities and definition bytes to the RoboTwin v2 sibling).
The byte derivation of its v1 predecessor from the RoboTwin B2 v1 arm left the
tree with that predecessor. These checks keep what a Dojo result, camera set,
controlled frame and help text must say on the surface this tree composes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface

ROOT = Path(__file__).resolve().parents[1]
SIBLING = "robotwin_toolchain_b2_closed_noservo_v2"
ROBODOJO = "robodojo_placement_closed_skills_v2"
ROBODOJO_CAMERAS = ["head", "left_wrist", "right_wrist"]


def _without_null_rule(profile: str):
    """The surface as composed before the two last-bound EXP11B binders run.

    optional_argument_null_as_omitted_v1 and compute_grammar_from_evaluator_v1
    bind after every layer; modelling the Dojo layers on the already-bound
    sibling would hide the null sentence behind later mode rewrites and put the
    generated compute section before the Dojo appends.
    """
    import ora.agent.composition.compute_grammar_schema as grammar
    import ora.agent.composition.null_optional_arguments as rule

    originals = (rule.bind_null_optional_arguments, grammar.bind_compute_grammar)
    rule.bind_null_optional_arguments = lambda definitions: {"slot": "-", "ref": "-", "sha256": "0" * 64}
    grammar.bind_compute_grammar = lambda definitions, capabilities: {"slot": "-", "ref": "-", "sha256": "0" * 64}
    try:
        return resolve_tool_surface(profile)
    finally:
        rule.bind_null_optional_arguments, grammar.bind_compute_grammar = originals


@pytest.fixture(scope="module")
def robodojo():
    return resolve_tool_surface(ROBODOJO)


@pytest.fixture(scope="module")
def sibling():
    return resolve_tool_surface(SIBLING)


def test_the_grasp_ik_annotation_admits_exactly_what_the_dojo_check_returns(robodojo):
    from ora.contracts.reach_check import ROBODOJO_REASONS, SHARED_REASONS

    stage = robodojo.result_schemas["grasp_plan"]["$defs"]["candidate"]["properties"][
        "ik_reachability"]["properties"]["pregrasp"]["properties"]
    assert set(stage["reason"]["enum"]) == {None, *SHARED_REASONS, *ROBODOJO_REASONS}


def test_the_align_result_admits_exactly_what_the_dojo_adapter_returns(robodojo):
    enum = robodojo.result_schemas["align_plan"]["properties"]["candidates"]["items"][
        "properties"]["rejection_reason"]["enum"]
    assert set(enum) == {None, "ik_failed", "ik_branch_jump", "fk_residual_exceeded"}


def test_view_config_binds_the_dojo_cameras_and_controlled_frame(robodojo, sibling):
    assert robodojo.view_config["automatic_actual_views"] == ROBODOJO_CAMERAS
    assert robodojo.view_config["default_observe_views"] == ROBODOJO_CAMERAS
    assert robodojo.view_config["controlled_frame_convention"] == "arx_x5_link6"
    shared = {key for key in sibling.view_config if key not in {
        "automatic_actual_views", "default_observe_views", "controlled_frame_convention",
        "public_geometry_descriptor"}}
    for key in shared:
        assert robodojo.view_config[key] == sibling.view_config[key], key


def test_no_robotwin_or_aloha_fact_survives_outside_the_shared_legacy_names(robodojo):
    """The substitutions are complete: no RoboTwin camera, frame, planner or evidence fact remains."""
    for definition in robodojo.definitions:
        text = definition["description"] + json.dumps(definition["inputSchema"])
        for term in ("RoboTwin", "ALOHA", "head_camera", "front_camera", "left_camera",
                     "right_camera", "CuRobo", "Top-K", "grip-center", "public TCP", "measured TCP",
                     "force/progress", "recovery.baseline_step", "execution_gate", "128-seed",
                     "not collision-checked", "consumes one robot decision."):
            assert term not in text, (definition["name"], term)


#: The controller-feedback requirement the activation contract checked while
#: RoboDojo was a developing surface (removed by integration 0059 unit A0, when
#: robodojo_placement_closed_skills_v2 became the RoboDojo default instance).
DEVELOPING_REQUIREMENT = (
    "robodojo_controller_feedback_v1", "current_public_robot_facts_v1", "motion_mode_companion_help_v1",
    "optional_argument_null_as_omitted_v1", "compute_grammar_from_evaluator_v1",
)


def test_the_dojo_surface_satisfies_the_developing_robodojo_requirement(robodojo):
    assert all(robodojo.supports(cap) for cap in DEVELOPING_REQUIREMENT)
    default = json.loads((ROOT / "tests/data/default-expectations.json").read_text())["instances"]["robodojo"]
    assert set(DEVELOPING_REQUIREMENT) <= set(default["required_capabilities"])
