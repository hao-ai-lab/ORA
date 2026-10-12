"""Every declared tool has a handler; retired verifiers keep their measurements.

The tool-surface redesign collapsed ``grasp.plan``/``place.plan`` plus a
separate read-only verifier into two closed skills, and retired
``grasp.verify_retention``, ``place.verify`` and ``vision.offset`` from the
surface a policy sees. ``place.verify`` and ``vision.offset`` are declared by
no kept profile, so their handlers are gone; the measurement functions moved
into the closed skills' stages and stay importable.

If a later change removes a handler while any registered profile still
declares its tool, these tests go red.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path

import pytest

from ora.agent.composition.tool_surface import (
    PROFILE_CATALOG, resolve_tool_surface,
)
import ora.agent.tools.runtime.core as runtime_core
import ora.agent.tools.robot as robot_tools


@functools.lru_cache(maxsize=1)
def _dispatched_tool_names() -> frozenset[str]:
    """Read the dispatch sites themselves rather than guessing from bytecode.

    Two runtimes serve the registered catalog and they dispatch differently:
    ``ManipulationToolRuntime.invoke`` builds one ``handlers`` dict literal,
    while ``robot.py``'s runtime uses an ``if name == ...`` chain. Parsing both
    keeps this capture readable and stable — a bytecode probe silently reports a
    partial set when CPython splits a large ``BUILD_CONST_KEY_MAP``.
    """

    names: set[str] = set()
    core_tree = ast.parse(Path(runtime_core.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(core_tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "handlers"
                and isinstance(node.value, ast.Dict)):
            for key in node.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    names.add(key.value)
    assert names, "the runtime dispatch table literal was not found"

    robot_tree = ast.parse(Path(robot_tools.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(robot_tree):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "name"):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                names.add(comparator.value)
            elif isinstance(comparator, ast.Set):
                for element in comparator.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        names.add(element.value)
    return frozenset(names)


#: The three tools the redesign retires from the *new* surface.
def _resolvable_tool_names() -> frozenset[str]:
    """Every name `invoke` can dispatch, both spellings of each."""

    from ora.agent.tools import tool_names

    dispatched = _dispatched_tool_names()
    return frozenset(dispatched) | {
        dotted for dotted, canon in tool_names.DOTTED_TOOL_NAMES.items()
        if canon in dispatched
    }


def test_every_tool_declared_by_any_registered_profile_has_a_handler():
    """A declared tool without a handler raises ``internal_tool_error`` at runtime."""

    handled = _resolvable_tool_names()
    missing: dict[str, set[str]] = {}
    for profile_id in PROFILE_CATALOG:
        surface = resolve_tool_surface(profile_id)
        for item in surface.definitions:
            if item["name"] not in handled:
                missing.setdefault(item["name"], set()).add(profile_id)
    assert not missing, f"declared tools without a dispatch handler: {missing}"


def test_the_surviving_measurement_functions_are_importable():
    """Each retired tool's measurement must survive its tool's retirement.

    ``vision.offset`` becomes a subtraction of two canonical localizations,
    ``grasp.verify_retention`` becomes the closed grasp skill's verify stage,
    and ``place.verify`` becomes the closed place skill's verify stage. If any
    of these imports breaks, a number left the surface without a successor.
    """

    from ora.agent.tools.localization import localize, offset_between
    from ora.agent.tools.placement import assess_geometry
    from ora.agent.tools.retention import verify_retention

    assert callable(offset_between) and callable(localize)
    assert callable(verify_retention)
    assert callable(assess_geometry)
