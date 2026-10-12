"""The canonical localization contract and the two measurements it absorbs.

``vision.offset`` is a subtraction of two canonical centroids, so it survives
its tool's retirement as geometry. ``closure_evidence`` puts the measured jaw
separation beside the target's measured extents — two quantities the Harness
already computed independently, neither of which knew the other existed.
"""

from __future__ import annotations

import math

import numpy
import pytest

from ora.agent.tools.localization import (
    CLOSURE_SCHEMA, NEAR_CLOSED_M, OFFSET_SCHEMA, SCHEMA,
    closure_evidence, localize, offset_between,
)
from ora.agent.tools.robot import ToolError


def _geometry(**overrides):
    value = {
        "schema": "ora.observation-bound-target-geometry.v1",
        "observation_id": "obs-1",
        "target_ref": "target-a",
        "source_camera": "agentview",
        "centroid_world_m": [0.4, 0.1, 0.9],
        # Ordered by descending principal variance: largest first.
        "principal_axes_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "principal_extents_m": [0.12, 0.05, 0.032],
        "visible_point_count": 900,
    }
    value.update(overrides)
    return value


def test_localize_names_the_extent_ordering_instead_of_leaving_it_to_each_caller():
    result = localize(_geometry())

    assert result["schema"] == SCHEMA
    assert result["largest_extent_m"] == 0.12
    assert result["middle_extent_m"] == 0.05
    assert result["smallest_extent_m"] == 0.032
    assert result["principal_extents_m"] == [0.12, 0.05, 0.032]
    assert "visible surface" in result["claim_boundary"]


def test_localize_carries_measured_quality_only_when_the_source_recorded_it():
    """v1 geometry has no quality block; its absence must not fail the call."""

    assert "status" not in localize(_geometry())

    graded = localize(_geometry(
        schema="ora.observation-bound-target-geometry.v2",
        status="unknown", quality_failures=["depth_inlier_fraction"],
    ))
    assert graded["status"] == "unknown"
    assert graded["quality_failures"] == ["depth_inlier_fraction"]


@pytest.mark.parametrize("field", ["centroid_world_m", "principal_extents_m"])
def test_localize_refuses_a_non_finite_measurement(field):
    with pytest.raises(ToolError) as refused:
        localize(_geometry(**{field: [0.1, float("nan"), 0.3]}))
    assert refused.value.code == "target_geometry_unavailable"


def test_the_offset_is_the_subtraction_vision_offset_performed():
    a = localize(_geometry(target_ref="target-a", centroid_world_m=[0.4, 0.1, 0.9]))
    b = localize(_geometry(target_ref="target-b", centroid_world_m=[0.5, 0.1, 0.95]))

    offset = offset_between(a, b)

    assert offset["schema"] == OFFSET_SCHEMA
    assert offset["from_target_ref"] == "target-a"
    assert offset["to_target_ref"] == "target-b"
    assert offset["translation_world_m"] == pytest.approx([0.1, 0.0, 0.05])
    assert offset["distance_m"] == pytest.approx(math.dist([0.4, 0.1, 0.9], [0.5, 0.1, 0.95]))
    assert "not material points" in offset["claim_boundary"]


def test_two_centroids_from_different_cameras_are_still_refused():
    """The retired tool's shared-frame requirement is preserved, not re-derived.

    Two centroids measured through different cameras describe different visible
    halves of their objects, so their difference is not a physical offset.
    """

    a = localize(_geometry(target_ref="target-a", source_camera="agentview"))
    b = localize(_geometry(target_ref="target-b", source_camera="wrist"))

    with pytest.raises(ToolError) as refused:
        offset_between(a, b)
    assert refused.value.code == "cross_camera_targets"
    assert refused.value.field == "b"


def test_an_offset_needs_two_current_and_different_targets():
    a = localize(_geometry(target_ref="target-a"))

    with pytest.raises(ToolError) as same:
        offset_between(a, localize(_geometry(target_ref="target-a")))
    assert same.value.code == "invalid_arguments"

    with pytest.raises(ToolError) as stale:
        offset_between(a, localize(_geometry(target_ref="target-b", observation_id="obs-0")))
    assert stale.value.code == "stale_observation"


def test_closure_evidence_calls_a_met_jaw_a_strong_negative():
    """Opening at or below the near-closed bound is the one near-conclusive read."""

    evidence = closure_evidence(
        gripper_opening_m=0.0009, localization=localize(_geometry()),
    )

    assert evidence["schema"] == CLOSURE_SCHEMA
    assert evidence["strength"] == "strong_negative"
    assert evidence["near_closed_m"] == NEAR_CLOSED_M
    assert evidence["gripper_opening_m"] == 0.0009
    assert evidence["target_extents_m"] == [0.12, 0.05, 0.032]


def test_closure_evidence_refuses_to_turn_a_matching_width_into_a_positive():
    """A separation inside the measured extent range proves nothing, and says so."""

    evidence = closure_evidence(
        gripper_opening_m=0.030, localization=localize(_geometry()),
    )

    assert evidence["strength"] == "not_contradictory"
    assert "proves_nothing" in evidence["reason"]
    # The point of the pair is that it is evidence, not a verdict.
    assert not any(isinstance(value, bool) for value in evidence.values())


def test_a_separation_wider_than_the_visible_object_is_only_weak_evidence():
    """Visible extents under-measure hidden extent, so this cannot be strong."""

    evidence = closure_evidence(
        gripper_opening_m=0.20, localization=localize(_geometry()),
    )

    assert evidence["strength"] == "weak_negative"
    assert "under_measures" in evidence["reason"]


def test_a_missing_jaw_measurement_is_reported_rather_than_silently_omitted():
    """LIBERO publishes no jaw separation at all; a silent gap reads as a pass."""

    evidence = closure_evidence(
        gripper_opening_m=None, localization=localize(_geometry()),
    )

    assert evidence["strength"] == "unavailable"
    assert evidence["missing"] == ["gripper_opening_m"]
    assert evidence["gripper_opening_m"] is None
    assert evidence["target_extents_m"] == [0.12, 0.05, 0.032]


def test_an_absent_localization_is_also_reported():
    evidence = closure_evidence(gripper_opening_m=0.01, localization=None)

    assert evidence["strength"] == "unavailable"
    assert evidence["missing"] == ["target_extents_m"]


def test_the_jaw_axis_is_declared_unknown_rather_than_assumed():
    """Nothing measures which principal axis the jaws close along."""

    evidence = closure_evidence(
        gripper_opening_m=0.030, localization=localize(_geometry()),
    )
    assert evidence["compared_against"] == "full_visible_extent_range_jaw_axis_unknown"
    assert evidence["target_largest_extent_m"] == 0.12
    assert evidence["target_smallest_extent_m"] == 0.032


def test_the_named_extents_come_from_the_sizes_not_from_the_variance_order():
    """Principal variance order is not size order, and a real target proved it.

    The first B2 run measured [0.1034, 0.0704, 0.0751] on a sauce bottle. The
    axis order is the eigenvalue order and stays paired with
    principal_axes_world, but reading smallest_extent_m off index 2 reported
    75.1 mm when the smallest measured dimension was 70.4 mm -- 4.7 mm out, on
    the quantity a jaw-separation comparison is judged against.
    """

    measured = [0.10342800325280663, 0.07041063477776481, 0.07514093969840292]
    result = localize(_geometry(principal_extents_m=list(measured)))

    # The pairing with the axes is what the ordering is for, so it survives.
    assert result["principal_extents_m"] == measured
    assert result["largest_extent_m"] == pytest.approx(0.10342800325280663)
    assert result["middle_extent_m"] == pytest.approx(0.07514093969840292)
    assert result["smallest_extent_m"] == pytest.approx(0.07041063477776481)


def test_a_separation_wider_than_the_widest_extent_is_weak_whatever_the_axis_order():
    """The weak-negative boundary is the largest extent, not the first one."""

    localization = {
        "schema": SCHEMA,
        # Variance order puts the widest dimension last.
        "principal_extents_m": [0.06, 0.05, 0.11],
    }
    inside = closure_evidence(gripper_opening_m=0.08, localization=localization)
    assert inside["strength"] == "not_contradictory"
    assert inside["target_largest_extent_m"] == pytest.approx(0.11)
    assert inside["target_smallest_extent_m"] == pytest.approx(0.05)

    wider = closure_evidence(gripper_opening_m=0.12, localization=localization)
    assert wider["strength"] == "weak_negative"


def test_a_measured_target_publishes_every_axis_paired_with_its_own_extent():
    """One axis of three was published, and three tasks need the other two.

    The segmentation localizer runs a full eigendecomposition and then keeps
    only the eigenvector of the largest eigenvalue. The other two are measured
    and discarded, so the policy cannot read a target's orientation: a QR board
    faces the way its smallest-extent axis points, a fan's facing needs two
    axes to disambiguate, and an upright bottle is its largest-extent axis. All
    three tasks stand at 0 across every cohort to date.

    Publishing the frame is not enough on its own, because eigenvalue order is
    variance order and a real target measured [0.1034, 0.0704, 0.0751]. Each
    axis therefore carries the extent measured along it, and the named
    largest/middle/smallest carry the axis that goes with them.
    """

    from ora.agent.tools.grasp import _principal_frame

    # A flat board: wide in x, tall in y, thin in z. Its normal is the axis of
    # least extent, which is the quantity "make the code face you" turns on.
    rng = numpy.random.default_rng(7)
    points = rng.uniform(-1.0, 1.0, size=(4000, 3)) * numpy.array([0.20, 0.12, 0.004])
    frame = _principal_frame(points, extended=True)

    extents = frame["principal_extents_m"]
    axes = frame["principal_axes_world"]
    assert len(axes) == 3 and all(len(axis) == 3 for axis in axes)
    assert len(extents) == 3

    # Every axis is a unit vector, and they are mutually orthogonal.
    for axis in axes:
        assert numpy.linalg.norm(axis) == pytest.approx(1.0, abs=1e-9)
    for i in range(3):
        for j in range(i + 1, 3):
            assert numpy.dot(axes[i], axes[j]) == pytest.approx(0.0, abs=1e-9)

    # The named scalars come from the sizes, and each names its own axis.
    assert frame["largest_extent_m"] == pytest.approx(max(extents))
    assert frame["smallest_extent_m"] == pytest.approx(min(extents))
    assert frame["middle_extent_m"] == pytest.approx(sorted(extents)[1])
    assert abs(numpy.dot(frame["largest_extent_axis_world"], [1.0, 0.0, 0.0])) == pytest.approx(1.0, abs=1e-3)
    assert abs(numpy.dot(frame["smallest_extent_axis_world"], [0.0, 0.0, 1.0])) == pytest.approx(1.0, abs=1e-3)
    # The board is 0.4 x 0.24 x 0.008 across its full spread; the extents are
    # measured on the same inlier percentiles the centre and AABB already use.
    assert frame["largest_extent_m"] > frame["middle_extent_m"] > frame["smallest_extent_m"]

    # The single axis older consumers read stays, and stays the largest.
    assert frame["principal_axis_world"] == pytest.approx(
        frame["largest_extent_axis_world"]
    )


def test_the_published_frame_survives_a_target_whose_variance_order_is_not_size_order():
    """The ordering the array keeps is the eigen order; the names are sizes."""

    from ora.agent.tools.grasp import _principal_frame

    rng = numpy.random.default_rng(11)
    # Deliberately near-degenerate in two directions so eigen order is unstable.
    points = rng.uniform(-1.0, 1.0, size=(4000, 3)) * numpy.array([0.10, 0.0705, 0.0751])
    frame = _principal_frame(points, extended=True)
    extents = frame["principal_extents_m"]
    assert frame["smallest_extent_m"] == pytest.approx(min(extents))
    assert frame["largest_extent_m"] == pytest.approx(max(extents))
    # Whatever the eigen order, the named axis and its extent belong together.
    index = extents.index(min(extents))
    assert frame["smallest_extent_axis_world"] == pytest.approx(
        frame["principal_axes_world"][index]
    )


def test_legacy_principal_axis_keeps_variance_order_when_extent_order_differs():
    """Sparse tails must not change the released single-axis meaning."""
    from itertools import product
    from ora.agent.tools.grasp import _principal_frame

    # Four percent distant x points dominate variance but fall outside the
    # 2.5/97.5 percentile extent. The y axis has the greatest retained extent.
    xs = [0.0] * 96 + [-5.0, -5.0, 5.0, 5.0]
    points = numpy.array(list(product(xs, [-1.0, 0.0, 1.0], [-0.1, 0.1])))
    legacy = _principal_frame(points, extended=False)
    extended = _principal_frame(points, extended=True)
    assert set(legacy) == {"principal_axis_world"}
    assert legacy["principal_axis_world"] == pytest.approx([1.0, 0.0, 0.0])
    assert extended["principal_axis_world"] == legacy["principal_axis_world"]
    assert extended["largest_extent_axis_world"] == pytest.approx([0.0, 1.0, 0.0])
