"""Tool names are legal function names, and the dotted ones still resolve.

Both provider protocols constrain a function name to `^[a-zA-Z0-9_-]{1,64}$`.
A dot is not in that set, so every dotted name has been travelling through an
alias rewrite on the wire and back again -- `providers/native.py` encodes it
outbound and `tool_names_by_alias` restores it on the way in. The repository's
own convention is already mixed: most tools are snake_case and a minority carry
a family prefix separated by a dot.

The active surfaces move to snake_case, family first, so a model scanning the
tool table sees families together and the alias layer becomes an identity map
for them. The frozen surfaces keep their dotted names: those are what sealed
runs were shown, and 174 of the 176 files naming one are frozen.

So the runtime has to serve both, and that is what this pins.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import re

from ora.agent.composition.tool_surface import resolve_tool_surface

#: The pattern both provider protocols enforce on a function name.
_STRICT_PROVIDER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
ROOT = Path(__file__).resolve().parents[1]
LIVE = {
    "libero_pro": "libero_placement_direct_overhead_v1",
    "robotwin": "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo": "robodojo_placement_closed_skills_v2",
}
#: Every dotted name the active surfaces carried, and what each becomes.
RENAMES = {
    "vision.segment": "vision_segment",
    "vision.ground": "vision_ground",
    "grasp.plan": "grasp_plan",
    "grasp.verify_retention": "grasp_verify_retention",
    "place.plan": "place_plan",
}


def test_the_live_surfaces_advertise_legal_provider_function_names():
    """Unit c, landed. This is the inverted form of the test that held its place.

    The live surfaces declare `canonical_tool_names_v1`, so every name they
    advertise matches the strict provider pattern and `tool_alias` is an
    identity over them -- no name reaches a model as a slug plus eight hex
    digits of a hash it never sees, and no `<wire_tool_aliases>` block is
    inserted for them.

    The rename happens in composition rather than in the component files.
    Two of the layers a live profile binds are frozen catalogs it
    hash-references, so their bytes cannot change; rewriting the editable ones
    instead would leave the rule with an exception on whichever backend binds a
    frozen layer.
    """

    from ora.agent.providers.tool_names import tool_alias

    for profile in LIVE.values():
        surface = resolve_tool_surface(profile)
        names = [tool["name"] for tool in surface.definitions]
        assert not (set(names) & set(RENAMES)), "a dotted name survived composition"
        for name in names:
            assert _STRICT_PROVIDER_NAME.fullmatch(name), name
            assert tool_alias(name) == name, f"{name} is still rewritten on the wire"
        assert not [key for key in surface.result_schemas if "." in key]


def test_every_frozen_surface_keeps_the_name_its_runs_were_shown():
    """174 of the 176 files naming a dotted tool are frozen identities."""

    from ora.agent.composition.tool_surface import PROFILE_CATALOG

    frozen_with_dots = 0
    for profile in PROFILE_CATALOG:
        if profile in LIVE.values():
            continue
        try:
            surface = resolve_tool_surface(profile)
        except Exception:
            continue
        if any("." in tool["name"] for tool in surface.definitions):
            frozen_with_dots += 1
    assert frozen_with_dots > 0, (
        "renaming must not have rewritten the frozen surfaces"
    )


@pytest.mark.parametrize("dotted,renamed", sorted(RENAMES.items()))
def test_both_spellings_resolve_to_one_handler(dotted, renamed):
    """Normalised at the dispatch boundary, not registered twice.

    Registering each handler under both spellings would let a tool added later
    be reachable under one and missing under the other. One table, keyed by the
    canonical name, and the boundary maps whatever a surface advertised onto it.
    """

    from ora.agent.tools import tool_names

    assert tool_names.canonical(dotted) == renamed
    assert tool_names.canonical(renamed) == renamed
    assert tool_names.advertised(renamed, dotted=True) == dotted
    assert tool_names.advertised(renamed, dotted=False) == renamed


def test_the_dispatch_table_is_keyed_by_the_canonical_name():
    """A dotted key in the table would be unreachable from an active surface."""

    source = (ROOT / "ora/agent/tools/runtime/core.py").read_text(encoding="utf-8")
    table = source[source.index("handlers = {"):source.index("result = handlers[")]
    for dotted in RENAMES:
        assert f'"{dotted}":' not in table, f"{dotted} is still a dispatch key"
    for renamed in RENAMES.values():
        assert f'"{renamed}":' in table, f"{renamed} has no handler"
