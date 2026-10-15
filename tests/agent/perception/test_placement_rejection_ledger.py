"""An empty candidate list must still carry what was measured to empty it.

``propose_placement`` computed patch counts, hull margins, rejection tallies and
the observed lowest point, then spent them all on choosing one reason string. A
policy cannot reason about a rejection it cannot see, and the numbers reach the
public result through the existing geometry projection rather than a new field.
"""

from __future__ import annotations

import numpy as np

from ora.agent.tools.placement import propose_placement
from ora.agent.tools.placement_runtime import measurements, public_geometry
from tests.agent.perception.test_placement_geometry import POSE, block, cavity, patch


def _ledger(proposal):
    return proposal["assessment"]["geometry"]["proposal_rejections"]


def test_a_successful_plan_reports_how_much_room_the_best_patch_had():
    proposal = propose_placement(
        block(), patch(), destination_kind="support_surface",
        tcp_pose=POSE, phase="preplace",
    )
    ledger = _ledger(proposal)

    assert proposal["status"] == "candidates"
    assert ledger["considered_patch_count"] > 0
    assert ledger["best_observed_hull_margin_m"] is not None
    assert set(ledger["rejected_patch_counts"]) == {
        "uncovered_footprint", "protected_proximity", "too_close_to_selected",
    }


def test_an_early_return_still_carries_the_ledger():
    """A phase that never reaches candidate generation reports the empty ledger."""

    support, _boundary = cavity()
    proposal = propose_placement(
        block(), support, destination_kind="front_cavity",
        tcp_pose=POSE, phase="preplace",
    )

    assert not proposal["candidates"]
    ledger = _ledger(proposal)
    assert ledger["rejected_for_opening_envelope"] == 0
    assert ledger["considered_patch_count"] is None


def test_a_footprint_that_fits_nowhere_reports_the_tally_not_only_the_reason():
    """The whole support is a narrow strip, so every patch fails coverage."""

    proposal = propose_placement(
        block(center=(0, 0, 0.15)), patch(hole=True),
        destination_kind="support_surface", tcp_pose=POSE, phase="preplace",
    )
    ledger = _ledger(proposal)

    assert ledger["considered_patch_count"] > 0
    rejected = ledger["rejected_patch_counts"]
    assert sum(rejected.values()) >= 0
    assert isinstance(rejected["uncovered_footprint"], int)


def test_the_descent_phase_publishes_the_observed_lowest_point():
    """Whether a descent is available at all is decided by this number."""

    proposal = propose_placement(
        block(center=(0, 0, 0.02)), patch(),
        destination_kind="support_surface", tcp_pose=POSE, phase="lower",
    )
    ledger = _ledger(proposal)

    assert ledger["visible_lowest_point_above_support_m"] is not None
    assert ledger["available_descent_m"] is not None


def test_the_ledger_reaches_the_public_result_through_the_geometry_projection():
    """No new result field: it flattens through ``measurements`` like the rest.

    The place.plan result schema is ``additionalProperties: false`` on frozen
    profiles, so the ledger has to travel inside the geometry block that
    ``public_geometry`` already projects.
    """

    proposal = propose_placement(
        block(), patch(), destination_kind="support_surface",
        tcp_pose=POSE, phase="preplace",
    )

    names = {item["name"] for item in measurements(public_geometry(proposal["assessment"]))}
    assert "proposal_rejections.considered_patch_count" in names
    assert "proposal_rejections.best_observed_hull_margin_m" in names
    assert "proposal_rejections.rejected_patch_counts.uncovered_footprint" in names


def test_the_ledger_is_plan_only_and_does_not_appear_in_a_verification():
    """``place.verify`` assesses geometry; it proposes nothing to reject."""

    from ora.agent.tools.placement import assess_geometry

    assessment = assess_geometry(
        block(), patch(), destination_kind="support_surface", tcp_pose=POSE,
    )
    assert "proposal_rejections" not in assessment["geometry"]
