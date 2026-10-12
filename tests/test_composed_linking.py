"""Linking a composed tool to the rest of the Harness (composed spec §20).

A task tool is code plus a description that may call any library tool:
``sdk.read`` reaches the policy's eight read tools, and a ``tool`` stage runs another composed tool
(``sdk.call``) whose actions are dispatched as that stage's attempts. Tools
may live outside the harness library (a practice agent's workspace) and bind
by absolute path; every tool a binding reaches is pinned by sha256. Covered
here with the contract fake world and the surface resolver; no simulator,
model service or Provider.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import ToolSurfaceError
from ora.agent.tools.composed import contract, sdk
from ora.agent.tools.composed.checks import check_tool, known_tool_names
from ora.agent.tools.composed.compiler import compile_tool
from ora.agent.tools.composed.declaration import DeclarationError
from ora.agent.tools.composed.executor import Execution, Finished
from ora.agent.tools.composed.library import LibraryError, load_tool, nested_closure
from ora.agent.tools.composed.starlark_runtime import StarlarkProgramError
from ora.agent.tools.composed.tunables import load_tunables
from ora.agent.tools.tool_errors import ToolError

TUNABLES = load_tunables()
COMPOSED = "robodojo_placement_closed_composed_v1"


def _tool(root: Path, source: str, name: str, version: str = "c1"):
    directory = root / name / version
    (directory / "compiled").mkdir(parents=True, exist_ok=True)
    (directory / "tool.star").write_text(source, encoding="utf-8")
    tool = load_tool(directory)
    for arm_profile in tool.declaration.arm_profiles:
        document = compile_tool(tool.declaration, source_sha256=tool.program.digest, tunables=TUNABLES,
                                arm_profile=arm_profile)
        tool.compiled_path(arm_profile).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return tool


def _header(name, stages, inputs='{"arm": arm_input(), "rationale": rationale_input()}', budget="primitive_actions = 4, queries = 4"):
    return f"""
TOOL = composed_tool(
    name = "{name}", version = "c1", family = "fixture",
    summary = "A fixture for linked composed tools.",
    use_when = "a test needs it.",
    does_not = ["certify task success"],
    inputs = {inputs},
    stages = [{stages}],
    budgets = budgets({budget}),
    arm_profiles = ["dual_arm"],
    refusals = [refusal("not_found", "the reference was not measured")],
)
"""


LIFT = _header("lift_arm", 'stage("lift", primitive = "move", summary = "lift 3 cm")') + """
def plan(ctx, inputs, params):
    state = sdk.robot_state(inputs["arm"])
    return sdk.commit(arm = inputs["arm"], pose = sdk.pose.translate(
        {"position_m": state["tcp_position_m"], "orientation_xyzw": state["tcp_orientation_xyzw"]}, [0.0, 0.0, 0.03]))

def execute(ctx, c):
    r = sdk.move(c["arm"], c["pose"], interaction = "free_space", stage = "lift")
    if r["status"] != "executed":
        return sdk.abort("lift_" + r["status"])
    return sdk.complete()
"""

ABOVE = _header(
    "above_object",
    'stage("hover", primitive = "move", summary = "hover above the grounded object"), '
    'stage("lift", primitive = "tool", summary = "lift again", tool = "lift_arm/c1")',
    inputs='{"arm": arm_input(), "object": text_input("what to hover above"), '
           '"height_m": number_input("hover height", 0.02, 0.3, unit = "m", required = False), '
           '"rationale": rationale_input()}',
) + """
def plan(ctx, inputs, params):
    found = sdk.read("vision_ground", {"query": inputs["object"]})
    if not found["ok"]:
        return sdk.refuse("not_found", "vision_ground did not localize " + inputs["object"])
    center = found["result"]["center_m"]
    height = inputs["height_m"] if inputs["height_m"] != None else 0.1
    state = sdk.robot_state(inputs["arm"])
    pose = {"position_m": sdk.vec.add(center, [0.0, 0.0, height]), "orientation_xyzw": state["tcp_orientation_xyzw"]}
    return sdk.commit(arm = inputs["arm"], pose = pose)

def execute(ctx, c):
    r = sdk.move(c["arm"], c["pose"], interaction = "free_space", stage = "hover")
    if r["status"] != "executed":
        return sdk.abort("hover_" + r["status"])
    again = sdk.read("vision_ground", {"query": "after hover"})
    lifted = sdk.call(stage = "lift", arguments = {"arm": c["arm"]})
    if lifted["status"] != "executed":
        return sdk.abort("lift_" + lifted["status"])
    return sdk.complete(measurements = [sdk.measure("hover", "regrounded", again["ok"])])
"""


@pytest.fixture
def library(tmp_path):
    root = tmp_path / "workspace"
    lift = _tool(root, LIFT, "lift_arm")
    above = _tool(root, ABOVE, "above_object")
    return root, lift, above


def _world(**reads):
    return contract.FakeWorld(reads={"vision_ground": lambda arguments: {"center_m": [0.1, 0.0, 0.8],
                                                                         "query": arguments.get("query")}, **reads})


# ---- inputs -----------------------------------------------------------------------------------------------


def test_value_inputs_compile_to_a_closed_schema(library):
    _root, _lift, above = library
    schema = above.compiled("dual_arm")["definition"]["inputSchema"]
    assert schema["properties"]["object"]["type"] == ["string"]
    assert schema["properties"]["height_m"]["type"] == ["number", "null"]
    assert schema["properties"]["height_m"]["maximum"] == 0.3 and "Unit: m." in schema["properties"]["height_m"]["description"]
    assert set(schema["required"]) == {"arm", "object", "rationale"}
    assert "A tool stage runs the composed tool it names" in above.compiled("dual_arm")["definition"]["description"]


@pytest.mark.parametrize("inputs, message", [
    ('{"arm": arm_input(), "pick": choice_input("which", ["only"]), "rationale": rationale_input()}', "options"),
    ('{"arm": arm_input(), "h": number_input("h", 1, 0), "rationale": rationale_input()}', "minimum < maximum"),
    ('{"arm": arm_input(), "rationale": text_input("why"), "x": rationale_input()}', "reserved name"),
])
def test_value_inputs_are_validated(inputs, message, tmp_path):
    with pytest.raises((DeclarationError, LibraryError, StarlarkProgramError), match=message):
        _tool(tmp_path, _header("bad_input", 'stage("lift", primitive = "move", summary = "lift")', inputs=inputs)
              + LIFT.split(")\n", 1)[1].split("\n", 1)[1], "bad_input")


# ---- sdk.read and nested tools -------------------------------------------------------------------------


def test_a_body_reads_plans_and_runs_a_nested_tool_as_attempts_of_its_stage(library):
    _root, _lift, above = library
    done = contract.run(above, arm_profile="dual_arm", tunables=TUNABLES, world=_world(),
                        arguments={"arm": "left", "object": "the board", "height_m": None, "rationale": "hover"})
    assert done.finished.status == "completed"
    assert [item.action_id for item in done.finished.dispatched] == ["action-contract:00-hover",
                                                                    "action-contract:01-lift"]
    assert done.requests[1].request["nested_stage"] == "lift"
    assert done.envelopes[0]["arguments"]["targets"]["left"]["position_m"] == pytest.approx([0.1, 0.0, 0.9])
    lift = next(item for item in done.finished.stages if item["stage"] == "lift")
    assert lift["status"] == "executed"
    summary = next(item["value"] for item in lift["measurements"] if item["quantity"] == "nested_tool")
    assert summary["tool"] == "lift_arm" and summary["dispatched_actions"] == 1 and summary["stages"] == {"lift": "executed"}
    assert done.finished.receipt["cost"]["dispatched_stage_actions"] == 2
    hover = next(item for item in done.finished.stages if item["stage"] == "hover")
    assert {"quantity": "regrounded", "value": True} in hover["measurements"]
    for scenario in contract.scenarios(done):
        run = contract.run(above, arm_profile="dual_arm", tunables=TUNABLES, world=_world(), scenario=scenario,
                           arguments={"arm": "left", "object": "the board", "height_m": 0.05, "rationale": "hover"})
        assert contract.violations(above, run) == []


def test_an_unmeasured_read_lets_plan_refuse_and_a_non_read_tool_is_a_fault(library):
    _root, _lift, above = library
    with pytest.raises(ToolError) as refused:
        contract.run(above, arm_profile="dual_arm", tunables=TUNABLES, world=contract.FakeWorld(),
                     arguments={"arm": "left", "object": "the board", "height_m": None, "rationale": "hover"})
    assert refused.value.code == "not_found"
    bad = _tool(library[0], ABOVE.replace('sdk.read("vision_ground", {"query": inputs["object"]})',
                                          'sdk.read("move_pose", {})').replace('"above_object"', '"bad_read"'),
                "bad_read")
    with pytest.raises(ToolError, match="read tools"):
        contract.run(bad, arm_profile="dual_arm", tunables=TUNABLES, world=_world(),
                     arguments={"arm": "left", "object": "x", "height_m": None, "rationale": "hover"})
    assert any(item.check == "read_tool" for item in check_tool(bad, tunables=TUNABLES, known_names=known_tool_names()))


def test_a_nested_refusal_reaches_the_body_as_a_refused_outcome(library):
    root, _lift, above = library
    _tool(root, LIFT.replace("    state = sdk.robot_state", '    return sdk.refuse("not_found", "no")\n    state = sdk.robot_state')
          .replace('stage("lift"', 'stage("lift"').replace("budgets(primitive_actions", "budgets(primitive_actions"),
          "lift_arm", "c1")
    done = contract.run(load_tool(root / "above_object" / "c1"), arm_profile="dual_arm", tunables=TUNABLES,
                        world=_world(), arguments={"arm": "left", "object": "b", "height_m": None, "rationale": "r"})
    assert done.finished.status == "aborted" and done.finished.stop_reason == "lift_refused"
    assert len(done.finished.dispatched) == 1


def test_nesting_depth_is_capped_at_run_time(library):
    _root, _lift, above = library
    world = _world()
    world.near = above.directory.parent.parent
    commitment, _result = contract.submit(
        above, arm_profile="dual_arm", arguments={"arm": "left", "object": "b", "height_m": None, "rationale": "r"},
        port=world.port(), tunables=TUNABLES, call_id="action-1", observation_id=contract.OBSERVATION_ID,
        observation_sha256=contract.OBSERVATION_SHA256)
    execution = Execution(above, commitment, world.port(), depth=2)
    step = execution.next()
    execution.resolve(world.apply(step.request))
    finished = execution.next()
    assert isinstance(finished, Finished) and finished.stop_reason == "lift_refused"
    lift = next(item for item in finished.stages if item["stage"] == "lift")
    assert lift["reason"] == "nesting_too_deep"


def test_tool_stages_resolve_without_cycles(library, tmp_path):
    root, lift, above = library
    assert set(nested_closure(above)) == {"lift_arm/c1"} and nested_closure(lift) == {}
    loop_a = _header("loop_a", 'stage("next", primitive = "tool", summary = "b", tool = "loop_b/c1")') + \
        'def plan(ctx, inputs, params):\n    return sdk.commit()\n\ndef execute(ctx, c):\n' \
        '    sdk.call(stage = "next")\n    return sdk.complete()\n'
    _tool(root, loop_a, "loop_a")
    looped = _tool(root, loop_a.replace('"loop_a"', '"loop_b"').replace("loop_b/c1", "loop_a/c1"), "loop_b")
    with pytest.raises(LibraryError, match="cycle"):
        nested_closure(looped)
    missing = _tool(root, loop_a.replace('"loop_a"', '"loop_c"').replace("loop_b/c1", "absent/c1"), "loop_c")
    assert any(item.check == "nested_tool" for item in check_tool(missing, tunables=TUNABLES,
                                                                  known_names=known_tool_names()))


# ---- binding tools from a workspace -------------------------------------------------------------------


def test_a_workspace_tool_binds_by_absolute_path_with_its_closure_pinned(library):
    root, lift, above = library
    surface = resolve_tool_surface(COMPOSED, component_overrides={"composed:above_object": str(above.directory)})
    binding = surface.composed_tools["above_object"]
    assert binding["root"] == str(root) and binding["source_sha256"] == above.program.digest
    assert binding["scope"] == "added"
    assert binding["nested"] == {"lift_arm/c1": {
        "directory": str(lift.directory), "source_sha256": lift.program.digest,
        "compiled_sha256": lift.compiled("dual_arm")["compiled_sha256"]}}
    assert surface.definitions[-1] == above.compiled("dual_arm")["definition"]
    with pytest.raises(ToolSurfaceError, match="same name"):
        resolve_tool_surface(COMPOSED, component_overrides={"composed:other": str(above.directory)})


# ---- SDK arithmetic -----------------------------------------------------------------------------------------


def test_quaternion_helpers_compose_rotations():
    quarter = sdk.quat_axis_angle([0, 0, 1], 3.141592653589793 / 2)
    assert sdk.quat_rotate(quarter, [1, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-9)
    half = sdk.quat_mul(quarter, quarter)
    assert sdk.quat_rotate(half, [1, 0, 0]) == pytest.approx([-1, 0, 0], abs=1e-9)
    assert sdk.vec_cross([1, 0, 0], [0, 1, 0]) == [0, 0, 1] and sdk.vec_dot([1, 2, 3], [1, 1, 1]) == 6
    with pytest.raises(sdk.SdkCallError):
        sdk.math_sqrt(-1)


# ---- the live path: a workspace tool on a real tool runtime ----------------------------------------------


OUTER = """
TOOL = composed_tool(
    name = "lift_then_open", version = "c1", family = "fixture",
    summary = "Lift the arm, re-measure, then open the gripper through the library tool.",
    use_when = "a test needs a nested library tool after a move.",
    does_not = ["certify task success"],
    inputs = {"arm": arm_input(), "rationale": rationale_input()},
    stages = [
        stage("lift", primitive = "move", summary = "lift 2 cm"),
        stage("open", primitive = "tool", summary = "open the gripper", tool = "open_gripper/c1"),
    ],
    budgets = budgets(primitive_actions = 3, queries = 4),
    arm_profiles = ["single_arm"],
)

def plan(ctx, inputs, params):
    state = sdk.robot_state(inputs["arm"])
    pose = {"position_m": state["tcp_position_m"], "orientation_xyzw": state["tcp_orientation_xyzw"]}
    return sdk.commit(arm = inputs["arm"], pose = sdk.pose.translate(pose, [0.0, 0.0, 0.02]))

def execute(ctx, c):
    if sdk.move(c["arm"], c["pose"], interaction = "free_space", stage = "lift")["status"] != "executed":
        return sdk.abort("lift_not_reached")
    seen = sdk.read("observe", {})
    opened = sdk.call(stage = "open")
    if opened["status"] != "executed":
        return sdk.abort("open_" + opened["status"])
    return sdk.complete(measurements = [sdk.measure("lift", "reread", seen["ok"])])
"""


def test_a_workspace_tool_reads_and_nests_on_the_live_path(tmp_path):
    from dataclasses import replace

    from ora.agent.tools.composed.library import LIBRARY_ROOT
    from ora.orchestrator.composed_tool_job import execute_composed_tool
    from test_composed_grasp_equivalence import _backend, _host
    from test_manipulation_skill_driver import _submitted
    from test_composed_grasp_equivalence import _observe

    outer = _tool(tmp_path / "workspace", OUTER, "lift_then_open")
    opener = load_tool(LIBRARY_ROOT / "open_gripper" / "c1")
    host = _host(tmp_path / "host", composed=True)
    bindings = {**host.surface.composed_tools, "lift_then_open": {
        "tool": "lift_then_open", "version": "c1", "arm_profile": "single_arm",
        "compiled_sha256": outer.compiled("single_arm")["compiled_sha256"], "scope": "added",
        "root": str(outer.directory.parent.parent), "source_sha256": outer.program.digest,
        "nested": {"open_gripper/c1": {"directory": str(opener.directory), "source_sha256": opener.program.digest,
                                       "compiled_sha256": opener.compiled("single_arm")["compiled_sha256"]}},
    }}
    host.surface = replace(host.surface, composed_tools=bindings)
    host.allowed_tools = tuple(dict.fromkeys((*host.allowed_tools, "lift_then_open", "observe")))
    host.invoke("lift_then_open", {"rationale": "lift then release"})
    action = _submitted(host)
    backend = _backend(tmp_path / "backend", faults={}, max_actions=16)
    root = tmp_path / "observations"
    box = {"latest": _observe(backend, root), "captured": False}
    captures = []

    def capture():
        box["latest"] = _observe(backend, root)
        box["captured"] = True
        box["stale"] = False
        captures.append(box["latest"][0]["observation_id"])
        return box["latest"]

    def fresh():
        return capture() if box.get("stale") else box["latest"]

    execution = host.start_composed_tool(action, capture=capture, latest=lambda: box["latest"], fresh=fresh)
    outer_receipt, _latest = execute_composed_tool(
        backend, action, execution=execution, initial_observation=box["latest"][0],
        initial_observation_root=box["latest"][1], capture_observation=capture, latest_box=box,
    )
    assert outer_receipt["status"] == "completed", (outer_receipt["stop_reason"], [(i["stage"], i["status"], i["reason"]) for i in outer_receipt["stage_receipts"]], outer_receipt["skill_receipt"].get("stop_reason"))
    stages = {item["stage"]: item for item in outer_receipt["stage_receipts"]}
    assert stages["open"]["action_id"].endswith(":01-open")
    assert stages["open"]["action"]["action"] == "set_gripper"
    summary = next(item["value"] for item in stages["open"]["measurements"] if item["quantity"] == "nested_tool")
    assert summary["tool"] == "open_gripper" and summary["status"] == "completed"
    assert len(captures) >= 1, "the read after the lift saw a fresh capture"
    assert host.__dict__.get("_current_override") is None
