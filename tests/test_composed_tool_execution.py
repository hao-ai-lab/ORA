"""Composed tool execution, layer-1 checks and the layer-2 contract over the library."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ora.agent.tools.composed import contract
from ora.agent.tools.composed.checks import check_tool, known_tool_names
from ora.agent.tools.composed.compiler import compile_tool
from ora.agent.tools.composed.executor import (
    STOP_BUDGET_EXHAUSTED, STOP_DRIVER_FAULT, ActionRequest, Execution, Finished, submit,
)
from ora.agent.tools.composed.library import LIBRARY_ROOT, LibraryError, load_library, load_tool
from ora.agent.tools.composed.tunables import load_tunables
from ora.agent.tools.tool_errors import ToolError

TUNABLES = load_tunables()
LIBRARY = load_library()
NAMES = known_tool_names()
TOOLS = sorted(LIBRARY)
CASES = [(key, profile) for key in TOOLS for profile in LIBRARY[key].declaration.arm_profiles]


def _arguments(tool, arm_profile, arm="left"):
    arguments = {"rationale": "the jaws are where they need to be"}
    if arm_profile == "dual_arm" and tool.declaration.input("arm_input"):
        arguments["arm"] = arm
    if tool.declaration.input("candidate_input"):
        arguments["candidate_id"] = "grasp-a1"
    return arguments


def _fixture(tmp_path: Path, source: str, name: str = "fixture_tool", version: str = "c1"):
    directory = tmp_path / name / version
    (directory / "compiled").mkdir(parents=True)
    (directory / "tool.star").write_text(source, encoding="utf-8")
    tool = load_tool(directory)
    for arm_profile in tool.declaration.arm_profiles:
        document = compile_tool(tool.declaration, source_sha256=tool.program.digest, tunables=TUNABLES,
                                arm_profile=arm_profile)
        tool.compiled_path(arm_profile).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return tool


RETRYING = """
TOOL = composed_tool(
    name = "fixture_tool", version = "c1", family = "gripper",
    summary = "Close, re-open and close again until the close executes.",
    use_when = "a fixture needs a bounded retry.",
    does_not = ["establish a grasp"],
    inputs = {"arm": arm_input(), "rationale": rationale_input()},
    stages = [
        stage("close", primitive = "gripper", summary = "close the gripper"),
        stage("verify", primitive = "check", summary = "record the arm state"),
    ],
    budgets = budgets(primitive_actions = 3, queries = 1),
    arm_profiles = ["single_arm", "dual_arm"],
    refusals = [refusal("already_closed", "the gripper already holds a close directive")],
)

def plan(ctx, inputs, params):
    state = sdk.robot_state(inputs["arm"])
    if state["gripper_directive"] == "close":
        return sdk.refuse("already_closed", "the gripper already holds a close directive")
    return sdk.commit(arm = inputs["arm"])

def execute(ctx, c):
    for attempt in range(3):
        r = sdk.gripper(c["arm"], "close", stage = "close")
        if r["status"] == "executed":
            return sdk.complete(measurements = [sdk.measure("verify", "close_attempts", attempt + 1)])
    return sdk.abort("close_never_executed")
"""


# ---- the committed library ------------------------------------------------------------------------


def test_the_library_holds_the_gripper_tools_and_the_grasp():
    assert TOOLS == [("close_gripper", "c1"), ("grasp_execute", "c1"), ("open_gripper", "c1")]
    for tool in LIBRARY.values():
        assert tool.declaration.arm_profiles == ("single_arm", "dual_arm")
        assert tool.directory.parent.parent == LIBRARY_ROOT


@pytest.mark.parametrize("key", TOOLS)
def test_every_library_tool_passes_the_static_checks(key):
    assert check_tool(LIBRARY[key], tunables=TUNABLES, known_names=NAMES) == []


@pytest.mark.parametrize(("key", "arm_profile"), CASES)
def test_every_library_tool_keeps_its_contract_in_every_scenario(key, arm_profile):
    tool = LIBRARY[key]
    arguments = _arguments(tool, arm_profile)
    nominal = contract.run(tool, arm_profile=arm_profile, arguments=arguments, tunables=TUNABLES)
    assert nominal.finished.status == "completed"
    for scenario in contract.scenarios(nominal):
        done = contract.run(tool, arm_profile=arm_profile, arguments=arguments, tunables=TUNABLES, scenario=scenario)
        assert contract.violations(tool, done) == []


# ---- open_gripper / close_gripper against the atomic tools they recompose ------------------------


def _atomic_definitions():
    root = LIBRARY_ROOT.parent / "tool_surfaces"
    single = json.loads((root / "explicit_motion_v1" / "libero-definitions.json").read_text())["tools"]
    dual = json.loads((root / "explicit_motion_v2" / "robotwin-definitions.json").read_text())["tools"]
    return {item["name"]: item for item in single}, {item["name"]: item for item in dual}


@pytest.mark.parametrize(("name", "directive"), [("open_gripper", "open"), ("close_gripper", "close")])
def test_the_gripper_tools_submit_what_the_atomic_tools_submit(name, directive):
    tool = LIBRARY[(name, "c1")]
    single, dual = _atomic_definitions()
    for arm_profile, atomic, arm in (("single_arm", single[name], "right"), ("dual_arm", dual[name], "left")):
        done = contract.run(tool, arm_profile=arm_profile, arguments=_arguments(tool, arm_profile, arm),
                            tunables=TUNABLES)
        # _gripper_atomic_v3 submits exactly this motion v3 action for the same call.
        assert [(item["action"], item["arguments"]) for item in done.envelopes] == [
            ("set_gripper", {"targets": {arm: directive}}),
        ]
        assert done.result["arm"] == arm and done.result["committed_sequence"] == [directive]
        assert done.finished.receipt["stages"][0]["status"] == "executed"
        schema = tool.compiled(arm_profile)["definition"]["inputSchema"]
        # Same arm and annotation arguments as the atomic tool; the composed protocol adds a rationale.
        assert set(schema["properties"]) == set(atomic["inputSchema"]["properties"]) | {"rationale"}
        if "arm" in atomic["inputSchema"]["properties"]:
            assert schema["properties"]["arm"]["enum"] == atomic["inputSchema"]["properties"]["arm"]["enum"]
        if "target_ref" in schema["properties"]:
            assert schema["properties"]["target_ref"]["type"] == atomic["inputSchema"]["properties"]["target_ref"]["type"]
        compiled_help = tool.compiled(arm_profile)["definition"]["description"]
        assert compiled_help.split("\n")[0] == atomic["description"].split(" Use when:")[0]


def test_close_gripper_echoes_target_ref_as_an_annotation():
    tool = LIBRARY[("close_gripper", "c1")]
    arguments = {**_arguments(tool, "dual_arm"), "target_ref": "red mug"}
    done = contract.run(tool, arm_profile="dual_arm", arguments=arguments, tunables=TUNABLES)
    assert done.result["intent"] == {"target_ref": "red mug", "role": "agent_annotation_only"}
    plain = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES)
    assert plain.result["intent"] == {"target_ref": None, "role": "agent_annotation_only"}


@pytest.mark.parametrize(
    ("arguments", "field"),
    [
        ({"rationale": "why", "arm": "middle"}, "arm"),
        ({"rationale": "why"}, None),
        ({"rationale": "", "arm": "left"}, "rationale"),
        ({"rationale": "x" * 201, "arm": "left"}, "rationale"),
        ({"rationale": "why", "arm": "left", "speed": 1}, None),
    ],
)
def test_schema_refusals_name_the_field_and_submit_nothing(arguments, field):
    world = contract.FakeWorld()
    before = json.dumps(world.state, sort_keys=True)
    with pytest.raises(ToolError) as raised:
        submit(LIBRARY[("open_gripper", "c1")], arm_profile="dual_arm", arguments=arguments, port=world.port(),
               tunables=TUNABLES, call_id="a", observation_id="o", observation_sha256="f" * 64)
    assert raised.value.code == "invalid_arguments"
    if field is not None:
        assert raised.value.field == field
    assert json.dumps(world.state, sort_keys=True) == before


def test_single_arm_tools_refuse_an_arm_argument():
    with pytest.raises(ToolError) as raised:
        submit(LIBRARY[("open_gripper", "c1")], arm_profile="single_arm",
               arguments={"rationale": "why", "arm": "right"}, port=contract.FakeWorld().port(),
               tunables=TUNABLES, call_id="a", observation_id="o", observation_sha256="f" * 64)
    assert raised.value.code == "invalid_arguments"


# ---- the executor on a retrying fixture ------------------------------------------------------------


def test_a_bounded_retry_records_attempts_in_one_stage(tmp_path):
    tool = _fixture(tmp_path, RETRYING)
    outcomes = iter(["not_reached", "unavailable", "executed"])

    def script(step, world):
        return {"status": next(outcomes), "reason": "scripted", "measurements": [{"quantity": "q", "value": 1}]}

    done = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES,
                        outcome=script)
    assert done.finished.status == "completed"
    close, verify = done.finished.stages
    assert close["status"] == "executed" and close["action_id"] == "action-contract:00-close.3"
    assert {"quantity": "attempts", "value": 3} in close["measurements"]
    assert verify["status"] == "executed" and verify["measurements"] == [{"quantity": "close_attempts", "value": 3}]
    assert done.finished.receipt["cost"]["dispatched_stage_actions"] == 3
    assert contract.violations(tool, done) == []


def test_a_spent_action_budget_aborts_after_the_last_allowed_action(tmp_path):
    tool = _fixture(tmp_path, RETRYING.replace("primitive_actions = 3", "primitive_actions = 2"))
    done = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES,
                        outcome=lambda step, world: {"status": "not_reached", "reason": "scripted", "measurements": []})
    assert done.finished.status == "aborted" and done.finished.stop_reason == STOP_BUDGET_EXHAUSTED
    assert len(done.finished.dispatched) == 2
    assert [item["stage"] for item in done.finished.stages] == ["close"]


def test_plan_refusals_must_be_declared_and_submit_nothing(tmp_path):
    tool = _fixture(tmp_path, RETRYING)
    world = contract.FakeWorld()
    world.state["left"]["gripper_directive"] = "close"
    with pytest.raises(ToolError) as raised:
        submit(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), port=world.port(),
               tunables=TUNABLES, call_id="a", observation_id="o", observation_sha256="f" * 64)
    assert raised.value.code == "already_closed" and raised.value.retryable is True
    undeclared = _fixture(tmp_path / "u", RETRYING.replace('sdk.refuse("already_closed"', 'sdk.refuse("other_code"'))
    with pytest.raises(ToolError) as raised:
        submit(undeclared, arm_profile="dual_arm", arguments=_arguments(undeclared, "dual_arm"), port=world.port(),
               tunables=TUNABLES, call_id="a", observation_id="o", observation_sha256="f" * 64)
    assert raised.value.code == "internal_tool_error"


def test_the_query_budget_covers_plan_and_execute(tmp_path):
    source = RETRYING.replace(
        'return sdk.complete(measurements = [sdk.measure("verify", "close_attempts", attempt + 1)])',
        'sdk.robot_state(c["arm"])\n            return sdk.complete()',
    )
    tool = _fixture(tmp_path, source)
    done = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES)
    assert done.finished.status == "aborted" and done.finished.stop_reason == STOP_BUDGET_EXHAUSTED
    roomy = _fixture(tmp_path / "r", source.replace("queries = 1", "queries = 2"))
    done = contract.run(roomy, arm_profile="dual_arm", arguments=_arguments(roomy, "dual_arm"), tunables=TUNABLES)
    assert done.finished.status == "completed"


@pytest.mark.parametrize(
    ("change", "fault"),
    [
        ('stage = "close")', 'stage = "verify")'),
        ('sdk.measure("verify"', 'sdk.measure("nowhere"'),
        ("return sdk.abort(\"close_never_executed\")", "return {\"kind\": \"done\"}"),
    ],
)
def test_body_faults_end_the_sequence_with_a_receipt(tmp_path, change, fault):
    tool = _fixture(tmp_path, RETRYING.replace(change, fault, 1))
    outcome = (lambda step, world: {"status": "not_reached", "reason": "s", "measurements": []}) \
        if "kind" in fault else None
    done = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES,
                        outcome=outcome)
    assert done.finished.status == "aborted" and done.finished.stop_reason == STOP_DRIVER_FAULT
    assert done.finished.fault and done.finished.receipt["stop_reason"] == STOP_DRIVER_FAULT


def test_the_wall_budget_stops_between_actions(tmp_path):
    tool = _fixture(tmp_path, RETRYING)
    world = contract.FakeWorld()
    commitment, _ = submit(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), port=world.port(),
                           tunables=TUNABLES, call_id="a", observation_id="o", observation_sha256="f" * 64)
    now = [0.0]
    execution = Execution(tool, commitment, world.port(), clock=lambda: now[0])
    step = execution.next()
    assert isinstance(step, ActionRequest)
    execution.resolve({"status": "not_reached", "reason": "slow", "measurements": []})
    now[0] = tool.declaration.budgets.wall_s + 1.0
    finished = execution.next()
    assert isinstance(finished, Finished) and finished.stop_reason == STOP_BUDGET_EXHAUSTED


def test_replay_is_deterministic_across_executions(tmp_path):
    tool = _fixture(tmp_path, RETRYING)

    def script(step, world):
        return {"status": "not_reached" if step.index == 0 else "executed", "reason": "s", "measurements": []}

    first = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES,
                         outcome=script)
    second = contract.run(tool, arm_profile="dual_arm", arguments=_arguments(tool, "dual_arm"), tunables=TUNABLES,
                          outcome=script)
    assert first.finished.receipt == second.finished.receipt
    assert first.envelopes == second.envelopes


# ---- layer-1 findings ------------------------------------------------------------------------------


def test_static_checks_report_drift_attribution_and_tool_names(tmp_path):
    tool = _fixture(tmp_path, RETRYING)
    assert check_tool(tool, tunables=TUNABLES, known_names=NAMES) == []
    path = tool.compiled_path("dual_arm")
    document = json.loads(path.read_text())
    document["definition"]["description"] += " Hand-edited."
    path.write_text(json.dumps(document))
    findings = {item.check for item in check_tool(tool, tunables=TUNABLES, known_names=NAMES)}
    assert findings == {"compiled_drift"}
    with pytest.raises(LibraryError, match="compiled_sha256"):
        tool.compiled("dual_arm")

    named = _fixture(tmp_path / "n", RETRYING.replace("a fixture needs a bounded retry", "move_pose is too much"))
    assert {item.check for item in check_tool(named, tunables=TUNABLES, known_names=NAMES)} == {"undeclared_tool_name"}

    stray = _fixture(tmp_path / "s", RETRYING.replace(
        'sdk.measure("verify", "close_attempts", attempt + 1)', 'sdk.measure("lift", "close_attempts", attempt + 1)'))
    assert [item.check for item in check_tool(stray, tunables=TUNABLES, known_names=NAMES)] == ["stage_attribution"]


def test_a_tool_must_live_at_its_declared_name_and_version(tmp_path):
    directory = tmp_path / "open_gripper" / "c2"
    directory.mkdir(parents=True)
    shutil.copy(LIBRARY_ROOT / "open_gripper" / "c1" / "tool.star", directory / "tool.star")
    with pytest.raises(LibraryError, match="declares open_gripper/c1"):
        load_tool(directory)
