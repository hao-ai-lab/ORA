"""Placement depth-edge help and its delivery to both provider wires.

The opt-in ``placement_support`` v1/v2 surfaces, their capability manifests and
the released content identities these tests compared are retired; the depth-edge
help they introduced is pinned on the kept surface that declares it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ora.agent.composition.tool_surface import (
    TOOL_HELP_DESCRIPTION_MARKERS,
    resolve_tool_surface,
    validate_tool_help_contract,
)
from ora.agent.native.gateway import NativeToolGateway, openai_tool_definitions
from ora.agent.providers.native import _anthropic_tool_alias, translate_anthropic_request
from ora.agent.tools.runtime import ManipulationToolRuntime


CAPABILITY = "public_rgbd_placement_depth_edges_v1"
PLACEMENT_TOOLS = {"place_plan"}
#: The kept surface declaring the depth-edge placement help (runtime_v1 RoboTwin
#: prepared-grasp qualification).
PROFILES = ("robotwin_toolchain_b2_closed_noservo_v1",)


@pytest.mark.parametrize("profile", PROFILES)
def test_placement_depth_help_exposes_uncertainty_not_background_or_full_fit(profile):
    surface = resolve_tool_surface(profile)
    assert surface.supports(CAPABILITY)
    validate_tool_help_contract(surface.definitions, surface.result_schemas)
    for definition in surface.definitions:
        if definition["name"] not in PLACEMENT_TOOLS:
            continue
        help_text = definition["description"]
        positions = [help_text.index(marker) for marker in TOOL_HELP_DESCRIPTION_MARKERS]
        assert positions == sorted(positions)
        for phrase in (
            "original mask and depth", "preserved", "discontinuity pixels", "uncertain",
            "not certified background", "all retained points", "unchanged support/opening tests",
            "conditional on unresolved silhouette", "full-object fit", "passage", "release",
        ):
            assert phrase in help_text


@pytest.mark.parametrize("profile", PROFILES)
def test_placement_depth_native_receives_identical_placement_help_and_shapes(profile):
    surface = resolve_tool_surface(profile)
    names = [item["name"] for item in surface.definitions]
    runtime = SimpleNamespace(surface=surface, allowed_tools=names)
    runtime.definitions = lambda: ManipulationToolRuntime.definitions(runtime)
    assert NativeToolGateway(runtime).delivered() == list(surface.definitions)
    openai = openai_tool_definitions(names, definitions=runtime.definitions())
    anthropic = translate_anthropic_request({"model": "offline-contract", "max_tokens": 128,
        "messages": [{"role": "user", "content": "Inspect current public evidence."}], "tools": openai})
    openai_by_name = {item["function"]["name"]: item["function"] for item in openai}
    anthropic_by_name = {item["name"]: item for item in anthropic["tools"]}
    for definition in surface.definitions:
        name = definition["name"]
        if name not in PLACEMENT_TOOLS:
            continue  # Inherited provider-specific motion projections are unchanged.
        assert openai_by_name[name]["description"] == definition["description"]
        assert openai_by_name[name]["parameters"] == definition["inputSchema"]
        wire = anthropic_by_name[_anthropic_tool_alias(name)]
        assert wire["description"] == definition["description"]
        assert wire["input_schema"] == definition["inputSchema"]
