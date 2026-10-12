"""Bind public operational limits from the same values used for admission."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Mapping

from ora.contracts.workspace import normalize_workspace_bounds
from .tool_surface import ResolvedToolSurface, canonical_sha256


def bind_public_workspace(surface: ResolvedToolSurface,
                          bounds: Mapping[str, list[float]]) -> ResolvedToolSurface:
    if not surface.supports("current_public_robot_facts_v1"):
        return surface
    public = {"frame": "world", "unit": "metre", "bounds": normalize_workspace_bounds(bounds),
              "semantics": "command_envelope_not_reachability",
              "outside_measured_state": "only_monotonic_recovery_toward_envelope"}
    definitions = copy.deepcopy(list(surface.definitions))
    # One home for the complete numeric box. Every relevant tool points here,
    # rather than duplicating the three ranges across all motion tools.
    for item in definitions:
        if item["name"] == "move_pose":
            item["description"] += "\nBound action workspace: " + json.dumps(public, sort_keys=True, separators=(",", ":"))
        elif item["name"] in surface.mutation_tools:
            item["description"] += " The world-frame command workspace is documented in move_pose help; it is not a reachability guarantee."
    return replace(surface, definitions=tuple(definitions), composition_sha256=canonical_sha256({
        "base": surface.composition_sha256, "public_workspace": public, "definitions": definitions,
    }))
