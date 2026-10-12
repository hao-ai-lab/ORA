"""Maintained closed-skill surfaces describe what the driver actually does.

Two published sentences had drifted from execution: the grasp precommit help
described one exact pregrasp probe while the check probes pregrasp and contact
approach with parallel-jaw equivalents, and the place_resume help promised
three dispatched stage actions while continuing dispatches two. On the
development line the default closed-skill surfaces deliver corrected text
through their generated published_limits layers. Frozen surfaces keep the text
their sealed runs were shown.
"""

from __future__ import annotations

import pytest

from ora.agent.composition import resolve_tool_surface

#: The default closed-skill surfaces: grasp_execute only, help from published_limits layers.
DEFAULT_CLOSED = ("robotwin_toolchain_b2_closed_noservo_v2", "robodojo_placement_closed_skills_v2")
FROZEN = ("robotwin_toolchain_b2_closed_noservo_v1",)


def _help(profile_id: str, tool: str) -> str:
    surface = resolve_tool_surface(profile_id)
    definitions = surface.definitions if hasattr(surface, "definitions") else surface["definitions"]
    return next(item["description"] for item in definitions if item["name"] == tool)


@pytest.mark.parametrize("profile_id", DEFAULT_CLOSED)
def test_default_closed_surfaces_already_state_both_probed_poses(profile_id):
    text = _help(profile_id, "grasp_execute")
    assert "probe of the selected pregrasp and contact poses" in text
    assert "selected exact pregrasp" not in text


@pytest.mark.parametrize("profile_id", FROZEN)
def test_frozen_surfaces_keep_their_published_text(profile_id):
    assert "three dispatched stage actions" in _help(profile_id, "place_resume")
    assert "selected exact pregrasp" in _help(profile_id, "grasp_execute")
