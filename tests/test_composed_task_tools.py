"""Task tools: a composed tool a harness change adds on one task's episodes only (composed spec §18, T1).

A ``composed:<tool>`` override may add a library tool the surface does not
declare: its compiled definition is appended, it becomes a mutation tool and
its binding says ``scope: added``. A harness-change record's ``task_tools``
selects such tools per task, so the default expansion binds them on that task
and on no other. No simulator or Provider.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace

import pytest
pytestmark = pytest.mark.usefixtures("ready_model_services")


from ora.agent.composition.tool_surface import ToolSurfaceError, bind_composed_tools, resolve_tool_surface
from ora.agent.tools.composed.library import LIBRARY_ROOT, load_tool
from ora.orchestrator import defaults, harness_change
from ora.orchestrator.paths import PROJECT_ROOT
from test_composed_grasp_equivalence import _host
from test_manipulation_skill_driver import _submitted

COMPOSED = "robotwin_toolchain_b2_closed_composed_v1"
DEFAULT_ROBOTWIN = "robotwin_toolchain_b2_closed_noservo_v2"
OPEN = load_tool(LIBRARY_ROOT / "open_gripper" / "c1")
ADD = {"composed:open_gripper": "open_gripper/c1"}


def test_an_override_adds_a_task_tool_the_surface_does_not_declare():
    plain = resolve_tool_surface(COMPOSED)
    added = resolve_tool_surface(COMPOSED, component_overrides=ADD)
    names = [item["name"] for item in added.definitions]
    assert "open_gripper" not in [item["name"] for item in plain.definitions]
    assert names[:-1] == [item["name"] for item in plain.definitions] and names[-1] == "open_gripper"
    assert added.definitions[-1] == OPEN.compiled("dual_arm")["definition"]
    assert added.result_schemas["open_gripper"] == OPEN.compiled("dual_arm")["result_schema"]
    assert added.mutation_tools == (*plain.mutation_tools, "open_gripper")
    assert added.composed_tools["open_gripper"]["scope"] == "added"
    assert added.composed_tools["grasp_execute"] == plain.composed_tools["grasp_execute"]
    assert added.composition_sha256 != plain.composition_sha256
    with pytest.raises(ToolSurfaceError, match="composed_tools_v1"):
        resolve_tool_surface(DEFAULT_ROBOTWIN, component_overrides=ADD)


def test_a_manifest_still_binds_only_declared_tools():
    definitions = [{"name": "observe", "description": "x", "inputSchema": {"type": "object"}}]
    with pytest.raises(ToolSurfaceError, match="does not declare"):
        bind_composed_tools({"composed_tools": {"open_gripper": "open_gripper/c1"}}, definitions, {},
                            ["composed_tools_v1"])


# --- harness-change task tools -------------------------------------------------------------------------

def _record(**overlay):
    return {"schema": harness_change.SCHEMA, "id": "rt-task-tool", "baseline": defaults.BASELINE_ID,
            "parent": None, "overlay": {"robotwin": {"tool_surface": COMPOSED, **overlay}},
            "summary": "add open_gripper on move_can_pot only", "sources": []}


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    tree = tmp_path_factory.mktemp("tree")
    shutil.copytree(PROJECT_ROOT / "ora", tree / "ora")
    shutil.copy2(PROJECT_ROOT / "policy.lock", tree / "policy.lock")
    return tree


def _binding(root, record, name):
    raw = json.dumps(record).encode()
    (root / harness_change.CHANGES_ROOT / name).write_bytes(raw)
    return {"path": f"{harness_change.CHANGES_ROOT}{name}", "sha256": hashlib.sha256(raw).hexdigest()}


def _expand(root, binding, task):
    spec = {"policy": {"kind": "native_api", "supports_images": True, "harness_change": binding},
            "runtime": {"kind": "robotwin", "task": task}}
    return defaults.expand_native_defaults(spec, root=root)["policy"]["tool_surface"]


def test_task_tools_bind_on_their_task_and_no_other(root):
    binding = _binding(root, _record(task_tools={"move_can_pot": {"open_gripper": "open_gripper/c1"}}), "task.json")
    on_task = _expand(root, binding, "move_can_pot")
    assert on_task == {"profile": COMPOSED, "component_overrides": ADD}
    assert _expand(root, binding, "adjust_bottle") == {"profile": COMPOSED, "component_overrides": {}}
    assert harness_change.task_key({"suite": "libero_10", "task_id": 3}) == "libero_10-t03"
    assert harness_change.task_key({"task": "move_can_pot"}) == "move_can_pot"
    unbindable = _binding(root, _record(task_tools={"move_can_pot": {"open_gripper": "open_gripper/c9"}}), "bad.json")
    with pytest.raises(harness_change.HarnessChangeError, match="open_gripper/c9"):
        _expand(root, unbindable, "move_can_pot")
    _expand(root, unbindable, "adjust_bottle")   # another task never resolves the missing version


@pytest.mark.parametrize("task_tools, message", [
    ({"Move Can": {"open_gripper": "open_gripper/c1"}}, "task name or <suite>-tNN"),
    ({"move_can_pot": {"open_gripper": "close_gripper/c1"}}, "<tool>/c<n>"),
    ({"move_can_pot": {}}, "<tool>/c<n>"),
    ({"move_can_pot": {"grasp_execute": "grasp_execute/c1"}}, "already binds"),
])
def test_task_tools_are_validated(task_tools, message):
    record = _record(task_tools=task_tools, composed_tools={"grasp_execute": "grasp_execute/c1"})
    with pytest.raises(harness_change.HarnessChangeError, match=message):
        harness_change.parse_change(record)


# --- dispatch ------------------------------------------------------------------------------------------

def test_the_runtime_serves_an_added_task_tool_through_the_composed_path(tmp_path):
    host = _host(tmp_path, composed=True)
    bindings = {**host.surface.composed_tools, "open_gripper": {
        "tool": "open_gripper", "version": "c1", "arm_profile": "single_arm", "scope": "added",
        "compiled_sha256": OPEN.compiled("single_arm")["compiled_sha256"]}}
    host.surface = replace(host.surface, composed_tools=bindings)
    host.allowed_tools = (*host.allowed_tools, "open_gripper")
    host.invoke("open_gripper", {"rationale": "release the can beside the pot"})
    action = _submitted(host)
    assert action["action"] == "execute_composed_tool"
    assert (action["arguments"]["tool"], action["arguments"]["version"]) == ("open_gripper", "c1")
    assert action["arguments"]["committed_sequence"] == ["open"]
