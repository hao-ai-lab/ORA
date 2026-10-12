"""One palette bounds how many grounding candidates can be drawn.

Two hardcoded three-colour tuples sat inline in `grasp.py`, and they disagreed
about what happens past the third candidate. The overlay zipped
`strict=False`, so a fourth candidate was drawn in no colour at all and raised
nothing. The enlarged preview indexed `colours[rank % 3]`, so a fourth was
painted the first one's colour and became unattributable against the overview.
Neither is an error the model can see: the call succeeds and the evidence is
quietly wrong.

The candidate ceiling was therefore never a product decision. It was the length
of a tuple nobody read as a bound. This pins the bound to the table, so raising
one cannot leave the other behind.
"""
from __future__ import annotations

import numpy as np
import pytest

from ora.agent.tools import grasp


def test_one_table_bounds_both_renderings():
    """A single table, and a ceiling derived from it rather than restated."""

    assert grasp.GROUNDING_MAX_CANDIDATES == len(grasp.GROUNDING_CANDIDATE_COLOURS)
    assert grasp.GROUNDING_MAX_CANDIDATES >= 32


def test_the_released_first_three_are_preserved_byte_for_byte():
    """Widening this table may append. It may not restate.

    The frozen point-segmentation v1 path renders through these same two
    helpers, and more than twenty profiles publish that contract, so the first
    three colours are the identity of sealed runs. Changing one rewrites images
    a released run already produced.
    """

    assert grasp.GROUNDING_CANDIDATE_COLOURS[:3] == (
        (48, 220, 96), (64, 180, 255), (255, 190, 48),
    )


def test_every_colour_is_distinguishable_from_every_other():
    """A palette whose entries collide is a table that silently aliases.

    The bound is 55 rather than a rounder number because the three released
    colours are fixed and the rest are packed around them: farthest-point
    sampling of the remaining space reaches 59.7, against 86.7 when all
    thirty-two are free to move. That is the price of not rewriting sealed
    images, and it is worth paying.
    """

    table = [np.asarray(c, dtype=float) for c in grasp.GROUNDING_CANDIDATE_COLOURS]
    worst = min(
        float(np.linalg.norm(a - b))
        for i, a in enumerate(table) for b in table[i + 1:]
    )
    assert worst >= 55.0, f"closest pair separated by only {worst:.1f}"


def test_every_colour_reads_against_a_photograph():
    """Too dark and a mask reads as shadow; too pale and it reads as glare."""

    for colour in grasp.GROUNDING_CANDIDATE_COLOURS:
        assert max(colour) >= 140, colour
        assert min(colour) <= 210, colour


def test_a_candidate_past_the_table_is_refused_rather_than_aliased():
    """The failure the modulo hid: rank 3 was painted rank 0's colour."""

    assert grasp.grounding_candidate_colour(0) != grasp.grounding_candidate_colour(3)
    with pytest.raises(ValueError, match="grounding candidate"):
        grasp.grounding_candidate_colour(grasp.GROUNDING_MAX_CANDIDATES)


def test_the_runtime_ceiling_is_the_table_and_not_a_literal():
    """Both admission points read the bound instead of restating three."""

    from ora.agent.tools.runtime import core as runtime_core, perception as runtime_perception

    # vision_ground's admission point lives in the runtime's perception mixin.
    for source in (
        (grasp.__file__), (runtime_core.__file__), (runtime_perception.__file__),
    ):
        text = open(source, encoding="utf-8").read()
        assert "max_candidates must be in [1,3]" not in text
        assert "1 <= max_candidates <= 3" not in text
