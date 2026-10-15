"""Every tool a registered surface advertises can actually be dispatched.

Nothing else in the suite joins these two halves. A surface manifest lists the
tools the model is shown; the runtime decides what to do with a name when the
model calls it. The loader checks the manifest against itself -- order, schema
references, mutation subsets -- and never against the runtime, so a tool can be
advertised to a policy by a profile no test invokes and fail only in a live
episode, as `internal_tool_error` from a bare `KeyError`.

That gap is why this file exists. Deleting a handler is safe exactly when no
registered profile still declares the tool, and the only way to know is to ask
every profile in the catalog.

Dispatch is read statically rather than by constructing runtimes, because
building one needs a control root, a manual, an event log and an environment
bridge, and a test that needs all four would be testing the fixture. The two
runtimes dispatch differently -- `ManipulationToolRuntime.invoke` builds a
name-to-method dict and subscripts it, `RobotToolRuntime.invoke` is an if/elif
chain over `name` -- so each is read in its own shape. `ManipulationToolRuntime`
subclasses `RobotToolRuntime` but overrides `invoke` without delegating, so its
dict is the complete set and the parent's chain is not a fallback.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from ora.agent.composition.tool_surface import (
    PROFILE_CATALOG,
    resolve_tool_surface,
)

ROOT = Path(__file__).resolve().parents[3]
RUNTIME_CORE = ROOT / "ora/agent/tools/runtime/core.py"
ROBOT = ROOT / "ora/agent/tools/robot.py"

#: `runtime_id` to the class whose `invoke` serves it, mirroring the dispatch in
#: `ora/agent/tools/factory.py`.
RUNTIME_CLASS = {
    "robot_runtime": "RobotToolRuntime",
    "native_atomic_causal": "ManipulationToolRuntime",
}


def _invoke_body(path: Path, class_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "invoke":
                    return item
    raise AssertionError(f"{class_name}.invoke not found in {path}")


def _dict_dispatch_names(path: Path, class_name: str) -> frozenset[str]:
    """Keys of the name-to-method mapping `invoke` subscripts."""

    names: set[str] = set()
    for node in ast.walk(_invoke_body(path, class_name)):
        if isinstance(node, ast.Dict):
            keys = [
                key.value for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            if len(keys) == len(node.keys) and keys:
                names.update(keys)
    return frozenset(names)


def _branch_dispatch_names(path: Path, class_name: str) -> frozenset[str]:
    """String literals `invoke` compares the tool name against."""

    names: set[str] = set()
    for node in ast.walk(_invoke_body(path, class_name)):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "name"):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                names.add(comparator.value)
            elif isinstance(comparator, (ast.Set, ast.Tuple, ast.List)):
                names.update(
                    element.value for element in comparator.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                )
    return frozenset(names)


def _reaches_a_handler(name: str, handlers: frozenset[str]) -> bool:
    """A surface's advertised name, resolved the way `invoke` resolves it.

    Handlers are registered once, under the canonical spelling. `invoke`
    normalises whatever a surface advertised onto that before subscripting, so a
    frozen surface's dotted name and an active surface's snake_case name reach
    the same method. Asserting the advertised name appears literally as a key
    would be the weaker claim: it would pass for a tool registered twice and
    fail for one registered once and resolved correctly.
    """

    from ora.agent.tools import tool_names

    return name in handlers or tool_names.canonical(name) in handlers


def _handlers(class_name: str) -> frozenset[str]:
    if class_name == "ManipulationToolRuntime":
        return _dict_dispatch_names(RUNTIME_CORE, class_name)
    return _branch_dispatch_names(ROBOT, class_name)


def test_the_dispatch_readers_find_a_plausible_surface() -> None:
    """A silent extraction failure would make every join below vacuous."""

    perception = _handlers("ManipulationToolRuntime")
    robot = _handlers("RobotToolRuntime")
    assert len(perception) > 20, perception
    assert {"observe", "move_pose"} <= perception
    # Both spellings of a renamed tool resolve, through the same rule invoke
    # uses. Asserting the dotted one appears literally as a key would pin the
    # dispatch table's shape rather than the property that matters.
    assert _reaches_a_handler("grasp.plan", perception)
    assert _reaches_a_handler("grasp_plan", perception)
    assert {"observation.dump_state", "robot.submit_action"} <= robot


@pytest.mark.parametrize("profile_id", sorted(PROFILE_CATALOG))
def test_every_declared_tool_of_a_registered_profile_has_a_handler(
    profile_id: str,
) -> None:
    """A profile may not advertise a tool the runtime cannot dispatch."""

    surface = resolve_tool_surface(profile_id)
    class_name = RUNTIME_CLASS.get(surface.runtime_id)
    assert class_name is not None, (
        f"{profile_id} names runtime {surface.runtime_id!r}, which "
        f"factory.build_tool_runtime cannot build"
    )
    declared = {item["name"] for item in surface.definitions}
    available = _handlers(class_name)
    missing = sorted(n for n in declared if not _reaches_a_handler(n, available))
    assert not missing, (
        f"{profile_id} advertises {missing} to the policy, but "
        f"{class_name}.invoke has no handler for them. A call would reach the "
        f"model as a non-retryable internal_tool_error. Either keep a handler "
        f"or stop declaring the tool."
    )



def test_geometry_query_is_dispatchable_on_the_perception_runtime() -> None:
    """The projection capability exists; only three of 31 surfaces could reach it.

    `RobotToolRuntime._geometry` serves world->pixel projection, pixel->world
    back-projection and region depth, and `geometry.query` is declared by
    `robot_runtime_v1`, `robot_runtime_v2` and `robotwin_dual_layer_v1`. Those
    are served by the if/elif chain in `robot.py`.

    `ManipulationToolRuntime` inherits the method but overrides `invoke`
    without delegating, so the surfaces built on it -- `native_atomic_causal`
    and `openeta_light`, which is most of them -- could not dispatch the name
    however it was spelled. Sealed traces show the consequence: the policy
    derives `u = cx + fx * x / z` by hand with intrinsics it read off
    calibration, in runs whose surface omits the tool.

    The name is the snake_case one here. `geometry.query` stays spelled that way
    on the three surfaces that already froze it; nothing re-spells a frozen
    surface.
    """

    handlers = _handlers("ManipulationToolRuntime")
    assert "geometry_query" in handlers, (
        "the perception runtime cannot dispatch geometry_query, so a surface "
        "that declared it would reach the model as internal_tool_error"
    )
    assert "geometry.query" in _handlers("RobotToolRuntime"), (
        "the three surfaces that declare geometry.query are served by "
        "RobotToolRuntime.invoke and must keep their frozen spelling"
    )

    source = ast.parse(ROBOT.read_text(encoding="utf-8"))
    owning = {
        item.name
        for node in source.body
        if isinstance(node, ast.ClassDef) and node.name == "RobotToolRuntime"
        for item in node.body
        if isinstance(item, ast.FunctionDef)
    }
    assert "_geometry" in owning, (
        "geometry_query dispatches to an inherited _geometry; if it moved, the "
        "perception runtime's entry is now a bare AttributeError"
    )
