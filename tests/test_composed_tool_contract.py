"""Composed tool headers, SDK, tunable table and compiler (layer 1 inputs and outputs)."""

from __future__ import annotations

import json

import pytest

from ora.agent.composition.tool_surface import validate_tool_help_contract
from ora.agent.tools.composed import sdk
from ora.agent.tools.composed.compiler import (
    HELP_LABELS, CompileError, compile_tool, compiled_identity, help_sections,
)
from ora.agent.tools.composed.declaration import DeclarationError, parse_declaration
from ora.agent.tools.composed.library import load_source
from ora.agent.tools.composed.starlark_runtime import StarlarkProgramError
from ora.agent.tools.composed.tunables import TunableError, load_tunables, parse_tunables
from ora.contracts.action import MOTION_ACTION_V3_SCHEMA

HEADER = """
TOOL = composed_tool(
    name = "nudge_and_grip",
    version = "c1",
    family = "gripper",
    summary = "Move a short way, then close the gripper.",
    use_when = "a short straight approach precedes a close.",
    does_not = ["establish a grasp"],
    inputs = {"arm": arm_input(), "rationale": rationale_input()},
    stages = [
        stage("approach", primitive = "move", summary = "move a short way"),
        stage("close", primitive = "gripper", summary = "close the gripper"),
        stage("verify", primitive = "check", summary = "record the outcome"),
    ],
    budgets = budgets(primitive_actions = 4, queries = 2),
    arm_profiles = ["single_arm", "dual_arm"],
    parameters = {"lift_m": tunable("lift.height_m")},
    refusals = [refusal("target_out_of_reach", "the requested offset leaves the workspace")],
    {extra}
)
def plan(ctx, inputs, params):
    return sdk.commit(arm = inputs["arm"])
def execute(ctx, c):
    return sdk.complete()
"""


def _header(extra: str = "", **replace: str) -> str:
    source = HEADER.replace("{extra}", extra)
    for old, new in replace.items():
        source = source.replace(old, new)
    return source


def _declaration(source: str):
    return load_source(source, filename="fixture.star")


# ---- declaration ----------------------------------------------------------------------------------


def test_a_valid_header_parses_into_its_declaration():
    _, declaration = _declaration(_header())
    assert declaration.name == "nudge_and_grip"
    assert [item.name for item in declaration.stages] == ["approach", "close", "verify"]
    assert [item.name for item in declaration.action_stages] == ["approach", "close"]
    assert declaration.stage_primitives == {"approach": "move", "close": "gripper", "verify": "check"}
    assert declaration.budgets.primitive_actions == 4 and declaration.budgets.queries == 2
    assert declaration.parameters[0].tunable == "lift.height_m" and not declaration.parameters[0].expose
    assert declaration.refusals[0].code == "target_out_of_reach"


@pytest.mark.parametrize(
    ("replace", "message"),
    [
        ({'"rationale": rationale_input()': '"why": arm_input()'}, "arm input must be named arm"),
        ({', "rationale": rationale_input()': ""}, "exactly one rationale input"),
        ({'"arm": arm_input()': '"object_gt": annotation_input("x")'}, "privileged"),
        ({'"arm": arm_input(), ': ""}, "dual_arm tool needs arm_input"),
        ({'primitive = "check"': 'primitive = "teleport"'}, "primitive must be one of"),
        ({'stage("verify"': 'stage("close"'}, "stage names must be unique"),
        ({"primitive_actions = 4": "primitive_actions = 1"}, "smaller than the number of action stages"),
        ({"then close the gripper.": "then close. It grips."}, "one sentence"),
        ({'"a short straight approach': '"Use when a short straight approach'}, "must not repeat its own label"),
        ({'arm_profiles = ["single_arm", "dual_arm"]': 'arm_profiles = ["tri_arm"]'}, "arm_profiles"),
    ],
)
def test_malformed_headers_are_refused_with_the_field_named(replace, message):
    with pytest.raises(DeclarationError, match=message):
        _declaration(_header(**replace))


def test_a_header_without_an_action_stage_is_refused():
    source = _header(**{
        'stage("approach", primitive = "move", summary = "move a short way"),': "",
        'stage("close", primitive = "gripper", summary = "close the gripper"),': "",
    })
    with pytest.raises(DeclarationError, match="at least one action stage"):
        _declaration(source)


def test_more_than_six_stages_is_refused():
    many = ", ".join(f'stage("s{index}", primitive = "gripper", summary = "s")' for index in range(7))
    source = _header(**{
        'stage("approach", primitive = "move", summary = "move a short way"),': many + ",",
        "primitive_actions = 4": "primitive_actions = 9",
    })
    with pytest.raises(DeclarationError, match="1..6 stages"):
        _declaration(source)


def test_a_constructor_refuses_unknown_keywords_at_load():
    with pytest.raises(StarlarkProgramError) as raised:
        _declaration(_header(extra="teleport = True,"))
    assert raised.value.code == "host_error" and "composed_tool" in raised.value.detail


def test_parse_declaration_refuses_a_hand_built_value():
    with pytest.raises(DeclarationError, match="composed_tool"):
        parse_declaration({"name": "x"})


# ---- tunables -------------------------------------------------------------------------------------


def test_the_committed_tunable_table_loads_unqualified():
    table = load_tunables()
    assert table.sha256.startswith("sha256:")
    lift = table["lift.height_m"]
    assert lift.default_for(None) == 0.03 and lift.default_for("aloha-agilex") == 0.03
    assert not lift.qualified and not table["probe.variant_policy"].qualified
    assert table["probe.variant_policy"].admit("exact") == "exact"
    with pytest.raises(TunableError):
        table["probe.variant_policy"].admit("anything")


def _table(**lift):
    entry = {
        "id": "lift.height_m", "primitive": "move", "type": "number", "unit": "m", "frame": "world +z",
        "description": "Lift height.", "defaults": {"*": 0.03}, "range": None, "qualification": None,
        "source": "fixture",
    }
    entry.update(lift)
    return parse_tunables({"schema": "ora.primitive-tunables.v1", "entries": [entry]},
                          path="fixture.json", sha256="sha256:fixture")


def test_a_qualified_range_bounds_the_values_it_admits():
    table = _table(range=[0.01, 0.06], qualification={"evidence": "fixture-qualification", "date": "2026-10-08"})
    lift = table["lift.height_m"]
    assert lift.qualified and lift.admit(0.05) == 0.05
    for bad in (0.07, float("nan"), True, "0.05"):
        with pytest.raises(TunableError):
            lift.admit(bad)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"defaults": {"aloha": 0.03}}, "'\\*' entry"),
        ({"range": [0.05, 0.01]}, "range"),
        ({"range": [0.01, 0.02]}, "default for"),
        ({"id": "lift"}, "dotted"),
        ({"qualification": {"evidence": "x"}}, "qualification"),
    ],
)
def test_malformed_tunable_entries_are_refused(change, message):
    with pytest.raises(TunableError, match=message):
        _table(**change)


# ---- SDK ------------------------------------------------------------------------------------------


STAGES = {"approach": "move", "close": "gripper", "verify": "check"}
POSE = {"position_m": [0.3, 0.0, 0.2], "orientation_xyzw": [0.0, 1.0, 0.0, 0.0]}


def test_action_requests_normalize_positional_and_keyword_spellings():
    a = sdk.validate_action_request("sdk.gripper", ["right", "close"], {"stage": "close"}, arms=("right",), stages=STAGES)
    b = sdk.validate_action_request("sdk.gripper", [], {"arm": "right", "directive": "close", "stage": "close"},
                                    arms=("right",), stages=STAGES)
    assert a == b == {"primitive": "gripper", "stage": "close", "arm": "right", "directive": "close"}
    move = sdk.validate_action_request("sdk.move", ["left", POSE], {"interaction": "contact_approach", "stage": "approach"},
                                       arms=("left", "right"), stages=STAGES)
    assert move["pose"] == POSE and move["interaction"] == "contact_approach"


@pytest.mark.parametrize(
    ("symbol", "args", "kwargs", "message"),
    [
        ("sdk.gripper", ["right", "close"], {"stage": "lift"}, "not declared"),
        ("sdk.gripper", ["right", "close"], {"stage": "approach"}, "declared for move"),
        ("sdk.gripper", ["left", "close"], {"stage": "close"}, "arm 'left'"),
        ("sdk.gripper", ["right", "half"], {"stage": "close"}, "directive"),
        ("sdk.gripper", ["right"], {"stage": "close"}, "missing"),
        ("sdk.move", ["right", POSE], {"interaction": "teleport", "stage": "approach"}, "interaction"),
        ("sdk.move", ["right", {"position_m": [0, 0, 0], "orientation_xyzw": [0, 0, 0, 2]}],
         {"interaction": "precision", "stage": "approach"}, "unit quaternion"),
        ("sdk.move", ["right", {"position_m": [0, 0]}], {"interaction": "precision", "stage": "approach"}, "position_m"),
        ("sdk.observe", [], {}, "not an SDK action"),
    ],
)
def test_malformed_action_requests_are_refused(symbol, args, kwargs, message):
    with pytest.raises(sdk.SdkCallError, match=message):
        sdk.validate_action_request(symbol, args, kwargs, arms=("right",), stages=STAGES)


def test_requests_lower_to_the_motion_v3_envelopes_atomic_tools_submit():
    gripper = sdk.lower_motion_v3(
        {"primitive": "gripper", "stage": "close", "arm": "left", "directive": "close"},
        action_id="a-1", observation_id="obs-1", observation_sha256="f" * 64,
    )
    assert gripper["schema"] == MOTION_ACTION_V3_SCHEMA
    assert (gripper["action"], gripper["arguments"]) == ("set_gripper", {"targets": {"left": "close"}})
    move = sdk.lower_motion_v3(
        {"primitive": "move", "stage": "approach", "arm": "right", "pose": POSE, "interaction": "precision"},
        action_id="a-2", observation_id="obs-1", observation_sha256="f" * 64,
    )
    assert move["action"] == "move_pose"
    assert move["arguments"] == {"targets": {"right": POSE}, "interactions": {"right": "precision"}}


def test_result_and_measurement_constructors_check_their_arguments():
    assert sdk.commit(arm="right") == {"kind": "commit", "data": {"arm": "right"}, "evidence": {}}
    assert sdk.refuse("x_code", "m", constraints={"n": 1})["constraints"] == {"n": 1}
    assert sdk.refuse("target_out_of_reach", " too far ")["message"] == "too far"
    assert sdk.measure("verify", "lift_m", 0.03, "m") == {
        "kind": "measurement", "stage": "verify", "quantity": "lift_m", "value": 0.03, "unit": "m",
    }
    for bad in (
        lambda: sdk.refuse("Bad-Code", "x"),
        lambda: sdk.abort("1code"),
        lambda: sdk.measure("verify", "q", [0.0] * (sdk.MAX_MEASUREMENT_VECTOR + 1)),
        lambda: sdk.measure("verify", "q", {"nested": 1}),
        lambda: sdk.complete(measurements=[{"quantity": "q"}]),
    ):
        with pytest.raises(sdk.SdkCallError):
            bad()


def test_vector_and_pose_arithmetic():
    assert sdk.vec_add([1, 2, 3], [1, 1, 1]) == [2.0, 3.0, 4.0]
    assert sdk.vec_sub([1, 2, 3], [1, 1, 1]) == [0.0, 1.0, 2.0]
    assert sdk.vec_scale([1, 2], 0.5) == [0.5, 1.0]
    assert sdk.vec_norm([3, 4]) == 5.0
    lifted = sdk.pose_translate(POSE, [0.0, 0.0, 0.03])
    assert lifted["position_m"] == pytest.approx([0.3, 0.0, 0.23])
    assert lifted["orientation_xyzw"] == POSE["orientation_xyzw"]
    with pytest.raises(sdk.SdkCallError):
        sdk.vec_add([1, 2], [1])
    with pytest.raises(sdk.SdkCallError):
        sdk.vec_scale([1], float("inf"))


def test_the_host_table_partitions_actions_queries_and_pure_symbols():
    assert {symbol for symbol, kind in sdk.HOST.items() if kind == "action"} == {
        "sdk.gripper", "sdk.move", "sdk.call",
    }
    assert {symbol for symbol, kind in sdk.HOST.items() if kind == "query"} == {
        "sdk.robot_state", "sdk.grasp.target", "sdk.grasp.precommit_probe", "sdk.capture", "sdk.check.grasp",
        "sdk.ask", "sdk.read",
    }
    assert "sdk.check.hold" in sdk.PURE
    assert sdk.CHECK_QUERIES <= sdk.EXECUTE_QUERIES and not sdk.PLAN_QUERIES & sdk.EXECUTE_QUERIES
    assert set(sdk.PURE) == {symbol for symbol, kind in sdk.HOST.items() if kind == "pure"}
    assert set(sdk.WORST_CASE_OUTCOME_MEASUREMENTS) == set(sdk.STAGE_PRIMITIVES)


# ---- compiler -------------------------------------------------------------------------------------


def _compiled(source=None, arm_profile="dual_arm", tunables=None):
    program, declaration = _declaration(source or _header())
    return compile_tool(declaration, source_sha256=program.digest, tunables=tunables or load_tunables(),
                        arm_profile=arm_profile)


def test_help_is_six_labelled_lines_generated_from_the_header():
    compiled = _compiled()
    lines = help_sections(compiled["definition"]["description"])
    assert [line.split(":")[0] + ":" for line in lines[1:]] == list(HELP_LABELS)
    assert lines[0] == "Move a short way, then close the gripper."
    effect = lines[2]
    assert "approach (move a short way), then close (close the gripper), then verify (record the outcome)" in effect
    assert "at most 4 primitive actions" in effect
    assert "may repeat its action" in effect and "remaining stages are skipped" in effect
    assert "Check stages only read public state." in effect
    assert "pose target is held" not in effect
    assert "the requested offset leaves the workspace (target_out_of_reach)" in lines[5]
    assert "(episode_terminal)" in lines[5] and lines[5].startswith("On error: Refused: ")
    validate_tool_help_contract([compiled["definition"]], {"nudge_and_grip": compiled["result_schema"]})


def test_arm_profiles_compile_different_schemas():
    dual = _compiled(arm_profile="dual_arm")["definition"]["inputSchema"]
    single = _compiled(arm_profile="single_arm")["definition"]["inputSchema"]
    assert dual["properties"]["arm"]["enum"] == ["left", "right"] and "arm" in dual["required"]
    assert "arm" not in single["properties"] and single["required"] == ["rationale"]
    assert single["additionalProperties"] is False


def test_exposing_an_unqualified_tunable_is_a_compile_error():
    source = _header(**{'tunable("lift.height_m")': 'tunable("lift.height_m", expose = True)'})
    with pytest.raises(CompileError, match="no qualification evidence"):
        _compiled(source)
    qualified = _table(range=[0.01, 0.06], qualification={"evidence": "fixture", "date": "2026-10-08"})
    schema = _compiled(source, tunables=qualified)["definition"]["inputSchema"]
    assert schema["properties"]["lift_m"]["minimum"] == 0.01 and schema["properties"]["lift_m"]["maximum"] == 0.06
    assert "lift_m" not in schema["required"]


def test_an_unknown_tunable_is_a_compile_error():
    with pytest.raises(CompileError, match="unknown tunable"):
        _compiled(_header(**{'tunable("lift.height_m")': 'tunable("lift.speed")'}))


def test_the_compiled_identity_binds_source_and_contract():
    first = _compiled()
    assert compiled_identity(first) == first["compiled_sha256"]
    assert _compiled()["compiled_sha256"] == first["compiled_sha256"]
    changed = _compiled(_header() + "\n# a comment changes the source bytes\n")
    assert changed["compiled_sha256"] != first["compiled_sha256"]
    tampered = json.loads(json.dumps(first))
    tampered["definition"]["description"] += " extra"
    assert compiled_identity(tampered) != tampered["compiled_sha256"]


def test_annotation_inputs_are_optional_and_echoed_in_the_result_schema():
    source = _header(**{'"rationale": rationale_input()': '"rationale": rationale_input(), '
                                                         '"target_ref": annotation_input("What you aim at.")'})
    compiled = _compiled(source)
    schema = compiled["definition"]["inputSchema"]
    assert schema["properties"]["target_ref"]["type"] == ["string", "null"]
    assert "target_ref" not in schema["required"]
    intent = compiled["result_schema"]["properties"]["intent"]
    assert intent["required"] == ["target_ref", "role"]
    assert "echoed under intent labelled agent_annotation_only" in compiled["definition"]["description"]
