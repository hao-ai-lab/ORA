"""Model-visible ToolSurface help and public-result authoring contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.surface_copies import scratch_profile_root
from ora.agent.composition import tool_surface as surface_module
from ora.agent.composition.tool_surface import (
    TOOL_HELP_DESCRIPTION_MARKERS,
    ToolSurfaceError,
    resolve_tool_surface,
    validate_tool_help_contract,
    validate_schema_instance,
)


def _definition() -> dict[str, object]:
    return {
        "name": "inspect",
        "description": (
            "Inspect one value. Use when: evidence is needed. Effect: read-only. "
            "Returns: success and value. Does not: change the world. "
            "On error: correct the named field and retry."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the public value to inspect.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    }


def _result_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "success": {"type": "boolean", "const": True},
            "value": {"type": "string"},
        },
        "required": ["success", "value"],
        "additionalProperties": False,
    }


def test_motion_surface_help_covers_every_advertised_tool() -> None:
    surface = resolve_tool_surface("libero_placement_direct_overhead_v1")
    names = {item["name"] for item in surface.definitions}
    assert surface.supports("policy_tool_help_v1")
    assert set(surface.result_schemas) == names
    for definition in surface.definitions:
        assert all(
            marker in definition["description"]
            for marker in TOOL_HELP_DESCRIPTION_MARKERS
        )
    observe = next(
        item for item in surface.definitions if item["name"] == "observe"
    )
    assert observe["inputSchema"]["properties"]["views"]["items"]["enum"] == [
        "agentview", "wrist", "pointcloud_top", "pointcloud_front",
        "pointcloud_side",
    ]


@pytest.mark.parametrize("missing_marker", TOOL_HELP_DESCRIPTION_MARKERS)
def test_help_contract_rejects_a_missing_description_section(
    missing_marker: str,
) -> None:
    definition = _definition()
    definition["description"] = str(definition["description"]).replace(
        missing_marker, "Missing:"
    )
    with pytest.raises(ToolSurfaceError, match="help lacks sections"):
        validate_tool_help_contract(
            [definition], {"inspect": _result_schema()},
        )


def test_help_contract_rejects_undocumented_or_open_parameters() -> None:
    undocumented = _definition()
    del undocumented["inputSchema"]["properties"]["name"]["description"]
    with pytest.raises(ToolSurfaceError, match="lacks a parameter description"):
        validate_tool_help_contract(
            [undocumented], {"inspect": _result_schema()},
        )

    open_schema = _definition()
    del open_schema["inputSchema"]["additionalProperties"]
    with pytest.raises(ToolSurfaceError, match="additionalProperties=false"):
        validate_tool_help_contract(
            [open_schema], {"inspect": _result_schema()},
        )


def test_help_contract_requires_one_public_result_schema_per_tool() -> None:
    with pytest.raises(ToolSurfaceError, match="must cover exactly"):
        validate_tool_help_contract([_definition()], {})


def test_help_contract_result_schema_rejects_public_shape_drift() -> None:
    schema = _result_schema()
    validate_schema_instance({"success": True, "value": "ok"}, schema)
    with pytest.raises(ToolSurfaceError, match="unknown result fields"):
        validate_schema_instance(
            {"success": True, "value": "ok", "hidden": "leak"}, schema,
        )


def test_new_catalog_profile_cannot_omit_the_help_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_root = scratch_profile_root(monkeypatch, tmp_path, {
        "new_surface/base.json": {"tools": [_definition()]},
        "new_surface/manifest.json": {
            "schema": "ora.tool-surface-profile.v1",
            "profile_id": "new_surface",
            "revision": 1,
            "runtime_id": "native_atomic_causal",
            "base_definitions": "base.json",
            "tool_order": ["inspect"],
            "components": {},
            "description_order": [],
            "schema_order": [],
            "result_order": [],
            "mutation_tools": [],
            "lifecycle_tools": [],
            "capabilities": [],
        },
    })
    monkeypatch.setitem(
        surface_module.PROFILE_CATALOG, "new_surface", profile_root / "new_surface/manifest.json",
    )
    with pytest.raises(ToolSurfaceError, match="must declare policy_tool_help_v1"):
        resolve_tool_surface("new_surface")
