"""A hardcoded palette must not silently decide how many candidates are shown.

``grasp.query`` validates ``max_candidates`` up to four. The camera overlay held
three colours and consumed them with ``zip(..., strict=False)``, so a fourth
candidate drew nothing at all — no wireframe, no label, no error — while the
global review panel, holding four colours, showed it. Two renderings of one
candidate list disagreed and nothing announced it.

Both renderers now read one table, and the drawable bound is derived from that
table rather than duplicated beside it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import ora.agent.tools.grasp as grasp
import ora.agent.tools.runtime.grasp as runtime_grasp


def test_the_palette_covers_every_candidate_the_runtime_will_admit():
    """The validated ceiling and the drawable ceiling are the same number."""

    # grasp_plan admission lives in the runtime's grasp mixin.
    source = Path(runtime_grasp.__file__).read_text(encoding="utf-8")
    assert "    maximum = 3\n" in source, (
        "the admitted candidate ceiling moved; re-check it against the palette"
    )
    assert grasp.GRASP_MAX_DRAWABLE_CANDIDATES >= 3, (
        f"the overlay can attribute only {grasp.GRASP_MAX_DRAWABLE_CANDIDATES} "
        "candidates while the runtime admits 3; the last would draw nothing"
    )


def test_the_drawable_bound_is_derived_from_the_table_not_duplicated():
    assert grasp.GRASP_MAX_DRAWABLE_CANDIDATES == len(grasp.GRASP_CANDIDATE_COLOURS)


def test_every_candidate_colour_is_distinguishable():
    colours = grasp.GRASP_CANDIDATE_COLOURS
    assert len(set(colours)) == len(colours), "two candidates would share a colour"
    for colour in colours:
        assert len(colour) == 3
        assert all(0 <= channel <= 255 for channel in colour)


GRASP_RENDERERS = ("_render_grasp_overlay", "_render_grasp_global_review")


def _colour_bindings(function_name):
    tree = ast.parse(Path(grasp.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        return [
            item.value for item in ast.walk(node)
            if isinstance(item, ast.Assign) and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == "colours"
        ]
    raise AssertionError(f"{function_name} was not found")


def test_both_grasp_renderers_read_the_same_table():
    """The defect was two literals drifting apart; one table is the fix."""

    for name in GRASP_RENDERERS:
        bindings = _colour_bindings(name)
        assert bindings, f"{name} no longer binds a `colours` name"
        for value in bindings:
            assert isinstance(value, ast.Name) and value.id == "GRASP_CANDIDATE_COLOURS", (
                f"{name} reintroduced its own candidate palette literal; that is "
                "how the camera overlay and the global review drifted apart"
            )


def test_grounding_request_and_rendering_bounds_are_independent():
    """A palette extension must not advertise unsupported service requests."""
    assert grasp.GROUNDING_MAX_CANDIDATES == len(grasp.GROUNDING_CANDIDATE_COLOURS)
    assert grasp.GROUNDING_MAX_CANDIDATES >= grasp.GROUNDING_REQUEST_MAX_CANDIDATES
    source = Path(grasp.__file__).read_text(encoding="utf-8")
    assert "1 <= max_candidates <= GROUNDING_REQUEST_MAX_CANDIDATES" in source
