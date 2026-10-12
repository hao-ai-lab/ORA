"""Pure synthetic public-geometry contracts for placement; no simulator/GPU."""

import copy
import json

import numpy as np
import pytest

from ora.agent.tools.placement import (
    assess_geometry, fit_rim_geometry, fit_support_geometry, propose_placement,
)
from ora.agent.tools.placement_runtime import measurements, public_geometry


POSE = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]


def geometry(points, camera="agentview"):
    return {"voxel_points_world_m": np.asarray(points).tolist(), "source_camera": camera}


def patch(*, z=0.0, hole=False):
    points = np.asarray([[x, y, z] for x in np.arange(-.15, .151, .01)
                         for y in np.arange(-.15, .151, .01)])
    if hole:
        points = points[(np.abs(points[:, 0]) > .045) | (np.abs(points[:, 1]) > .045)]
    return geometry(points)


def block(center=(0, 0, .04)):
    points = np.asarray([[x, y, z] for x in (-.012, 0, .012)
                         for y in (-.012, 0, .012) for z in (-.02, .02)])
    return geometry(points + np.asarray(center))


def ring(*, side=False, arc=2*np.pi):
    theta = np.linspace(0, arc, 12, endpoint=False)
    points = np.column_stack([.04*np.cos(theta), .04*np.sin(theta), np.zeros(12)])
    if side:
        points = points[:, [2, 1, 0]]
    return points


def cavity():
    support = geometry([[x, y, 0] for x in np.arange(-.12, .121, .01)
                        for y in np.arange(.04, .201, .01)])
    boundary = geometry([[x, 0, z] for x in np.arange(-.12, .121, .01)
                         for z in np.arange(.01, .201, .01)])
    return support, boundary


def assess(obj, support=None, **kwargs):
    return assess_geometry(obj, patch() if support is None else support,
                           destination_kind="support_surface", tcp_pose=POSE, **kwargs)


def test_rim_circle_distinguishes_upright_sideways_without_object_pca():
    up = fit_rim_geometry(ring())
    side = fit_rim_geometry(ring(side=True))
    assert up["status"] == "consistent"
    assert side["status"] == "contradicted"
    assert up["metrics"]["tilt_deg"] == pytest.approx(0)
    assert side["metrics"]["tilt_deg"] == pytest.approx(90)
    assert up["metrics"]["radius_m"] == pytest.approx(.04)


def test_rim_degenerate_small_arc_missing_or_non_circle_is_unknown():
    assert fit_rim_geometry(None)["status"] == "unknown"
    assert fit_rim_geometry(ring(arc=np.pi/2))["status"] == "unknown"
    assert fit_rim_geometry([[i*.01, 0, 0] for i in range(6)])["status"] == "unknown"
    ellipse = ring() * [2, 1, 1]
    assert fit_rim_geometry(ellipse)["status"] == "unknown"


def test_occluded_base_never_becomes_supported_or_suspended_claim():
    result = assess(block(center=(0, 0, .15)))
    assert result["checks"]["support"]["status"] == "unknown"
    assert result["checks"]["support"]["reason"] == "base_support_not_observable"
    assert result["checks"]["orientation"]["status"] == "unknown"
    assert "stability" in result["claim_boundary"]


def test_support_hole_is_unknown_and_is_not_filled_by_convex_hull():
    result = assess(block(), patch(hole=True))
    assert result["geometry"]["support"]["status"] == "consistent"
    assert result["checks"]["region"]["status"] == "unknown"
    proposal = propose_placement(block(), patch(hole=True), destination_kind="support_surface",
                                 tcp_pose=POSE, phase="preplace")
    assert proposal["status"] == "candidates"
    for candidate in proposal["candidates"]:
        point = candidate["target_pose"]["position_m"]
        assert abs(point[0]) > .045 or abs(point[1]) > .045


def test_visible_object_outside_support_is_a_scoped_contradiction():
    result = assess(block(center=(.30, 0, .04)))
    assert result["checks"]["region"]["status"] == "contradicted"
    assert "observed_support_envelope" in result["checks"]["region"]["reason"]


def test_open_container_needs_separate_rim_and_interior_support():
    result = assess_geometry(block(center=(.3, 0, .08)), patch(), patch(z=.15),
                             destination_kind="open_container", tcp_pose=POSE)
    assert result["geometry"]["opening"]["status"] == "consistent"
    assert result["checks"]["region"]["status"] == "contradicted"
    missing = assess_geometry(block(), patch(), destination_kind="open_container", tcp_pose=POSE)
    assert missing["checks"]["region"]["status"] == "unknown"


def test_cavity_outside_before_release_is_not_mislabelled_as_exit_failure():
    support, opening = cavity()
    result = assess_geometry(block(center=(0, -.06, .06)), support, opening,
                             destination_kind="front_cavity", tcp_pose=POSE)
    assert result["geometry"]["opening"]["status"] == "consistent"
    assert result["checks"]["region"]["status"] == "contradicted"
    assert result["checks"]["region"]["reason"] == "visible_object_outside_cavity_opening"
    assert result["geometry"]["opening"]["metrics"]["inward_normal_world"] == pytest.approx([0,1,0])


def test_cavity_without_observable_opening_or_interior_side_is_unknown():
    support, _ = cavity()
    result = assess_geometry(block(), support, destination_kind="front_cavity", tcp_pose=POSE)
    assert result["checks"]["region"]["status"] == "unknown"
    proposal = propose_placement(block(), support, destination_kind="front_cavity", tcp_pose=POSE,
                                 phase="preplace")
    assert proposal["status"] == "unknown"
    assert not proposal["candidates"]


def test_offset_changes_tcp_goal_without_moving_target_patch_or_orientation():
    original = block(center=(0, 0, .15))
    shifted = block(center=(.07, -.02, .15))
    p1 = propose_placement(original, patch(), destination_kind="support_surface", tcp_pose=POSE,
                           phase="preplace")
    p2 = propose_placement(shifted, patch(), destination_kind="support_surface", tcp_pose=POSE,
                           phase="preplace")
    assert p1["status"] == p2["status"] == "candidates"
    a = np.asarray(p1["candidates"][0]["target_pose"]["position_m"])
    b = np.asarray(p2["candidates"][0]["target_pose"]["position_m"])
    assert b-a == pytest.approx([-.07, .02, 0])
    assert p1["candidates"][0]["target_pose"]["orientation_xyzw"] == POSE[3:]


def test_exit_requires_open_and_returns_short_outward_goals_not_execution():
    support, opening = cavity()
    closed = propose_placement(block(), support, opening, destination_kind="front_cavity",
                               tcp_pose=POSE, phase="exit")
    assert closed["candidates"] == []
    result = propose_placement(block(), support, opening, destination_kind="front_cavity",
                               tcp_pose=POSE, phase="exit", gripper_open=True)
    assert len(result["candidates"]) == 3
    for candidate, distance in zip(result["candidates"], (.02, .03, .04)):
        assert candidate["target_pose"]["position_m"] == pytest.approx([0,-distance,.3])
        assert "plan_ref" not in candidate
        assert candidate["metrics"]["disengagement_established"] is False


def test_protected_geometry_changed_and_view_change_remain_distinct():
    pairs = [{"before_geometry": block(), "current_geometry": block(center=(.06,0,.04))}]
    result = assess(block(), protected_geometries=pairs)
    assert result["checks"]["protected_objects"]["status"] == "contradicted"
    pairs[0]["current_geometry"]["source_camera"] = "wrist"
    assert assess(block(), protected_geometries=pairs)["checks"]["protected_objects"]["status"] == "unknown"


def test_protected_points_exclude_candidate_footprints():
    protected = block(center=(0,0,.04))
    result = propose_placement(block(), patch(), destination_kind="support_surface", tcp_pose=POSE,
                               phase="preplace", protected_geometries=[{
                                   "before_geometry": protected, "current_geometry": protected}])
    assert result["candidates"]
    for candidate in result["candidates"]:
        assert np.linalg.norm(candidate["target_pose"]["position_m"][:2]) > .03


def test_exit_rejects_observed_protected_object_on_tcp_segment():
    protected = block(center=(0, 0, .33))
    result = propose_placement(block(), patch(), destination_kind="support_surface", tcp_pose=POSE,
                               phase="exit", gripper_open=True, protected_geometries=[{
                                   "before_geometry": protected, "current_geometry": protected}])
    assert result["status"] == "no_candidate"
    assert result["reason"] == "protected_visible_geometry_near_tcp_segment"


def test_lower_requires_actual_observed_footprint_and_remains_bounded():
    result = propose_placement(block(center=(0, 0, .15)), patch(),
                               destination_kind="support_surface", tcp_pose=POSE, phase="lower")
    assert result["status"] == "candidates"
    candidate = result["candidates"][0]
    assert candidate["target_pose"]["position_m"] == pytest.approx([0,0,.27])
    assert candidate["metrics"]["release_height_established"] is False
    outside = propose_placement(block(center=(.3, 0, .15)), patch(),
                                destination_kind="support_surface", tcp_pose=POSE, phase="lower")
    assert outside["candidates"] == []


@pytest.mark.parametrize("phase", ["preplace", "lower"])
def test_sideways_requested_rim_blocks_each_closed_gripper_placement_phase(phase):
    result = propose_placement(block(center=(0, 0, .15)), patch(),
                               destination_kind="support_surface", tcp_pose=POSE,
                               phase=phase, rim_points_world=ring(side=True))
    assert result["assessment"]["checks"]["orientation"]["status"] == "contradicted"
    assert result["candidates"] == []
    assert result["reason"] == "requested_upright_rim_contradicted_no_implicit_rotation"


@pytest.mark.parametrize("kind", ["open_container", "front_cavity"])
@pytest.mark.parametrize("phase", ["preplace", "lower"])
def test_small_observed_opening_rejects_wide_visible_object_despite_large_support(kind, phase):
    obj = geometry([[x, y, z] for x in (-.045, 0, .045)
                    for y in (.055, .10, .145) for z in (.10, .14)])
    if kind == "open_container":
        support = patch()
        opening = geometry([[x, y, .08] for x in np.arange(-.025, .026, .005)
                            for y in np.arange(.075, .126, .005)])
    else:
        support, _ = cavity()
        opening = geometry([[x, 0, z] for x in np.arange(-.025, .026, .005)
                            for z in np.arange(.01, .201, .005)])
    result = propose_placement(obj, support, opening, destination_kind=kind,
                               tcp_pose=POSE, phase=phase)
    assert result["assessment"]["geometry"]["support"]["status"] == "consistent"
    assert result["assessment"]["geometry"]["opening"]["status"] == "consistent"
    region = result["assessment"]["checks"]["region"]
    assert region["status"] == "contradicted"
    assert region["reason"] == "visible_object_exceeds_observed_opening_envelope"
    assert result["status"] in {"unknown", "no_candidate"}
    assert result["candidates"] == []


def test_front_aperture_height_constrains_current_relation_and_preplace_candidate():
    support, _ = cavity()
    opening = geometry([[x, 0, z] for x in np.arange(-.12, .121, .01)
                        for z in np.arange(.01, .056, .005)])
    obj = block(center=(0, .10, .10))
    result = propose_placement(obj, support, opening, destination_kind="front_cavity",
                               tcp_pose=POSE, phase="preplace")
    assert result["assessment"]["checks"]["region"]["status"] == "contradicted"
    assert result["candidates"] == []
    assert result["reason"] == "visible_object_exceeds_observed_opening_outer_envelope"


@pytest.mark.parametrize("opening_top, expected_standoff", [
    (.125, .08), (.105, .06), (.085, .04), (.065, .02),
])
def test_front_cavity_preplace_uses_highest_bounded_standoff_that_fits_all_points(
    opening_top, expected_standoff,
):
    support, _ = cavity()
    opening = geometry([[x, 0, z] for x in np.arange(-.12, .121, .01)
                        for z in np.arange(.01, opening_top + .001, .005)])
    obj = block(center=(0, -.06, .15))
    original = copy.deepcopy((obj, support, opening))
    result = propose_placement(obj, support, opening, destination_kind="front_cavity",
                               tcp_pose=POSE, phase="preplace")
    assert result["status"] == "candidates"
    assert 1 <= len(result["candidates"]) <= 3
    source = np.asarray(obj["voxel_points_world_m"])
    for candidate in result["candidates"]:
        assert candidate["metrics"]["visible_lower_surface_standoff_m"] == expected_standoff
        shifted = source + np.asarray(candidate["target_pose"]["position_m"]) - POSE[:3]
        assert shifted[:, 2].min() == pytest.approx(expected_standoff)
        assert shifted[:, 2].max() <= opening_top + 1e-9
        assert candidate["metrics"]["minimum_observed_opening_envelope_margin_m"] >= -1e-9
        assert candidate["metrics"]["opening_outer_envelope_is_passage_certificate"] is False
        assert candidate["metrics"]["hidden_base_extent_known"] is False
        assert candidate["target_pose"]["orientation_xyzw"] == POSE[3:]
    assert (obj, support, opening) == original


def test_front_cavity_standoff_cannot_drop_a_single_tall_visible_point():
    support, _ = cavity()
    opening = geometry([[x, 0, z] for x in np.arange(-.12, .121, .01)
                        for z in np.arange(.01, .106, .005)])
    dense = np.repeat(np.asarray(block(center=(0, -.06, .15))["voxel_points_world_m"]),
                      100, axis=0)
    dense[123] = [0, -.06, .23]  # 100 mm observed height leaves less than 20 mm.
    result = propose_placement(geometry(dense), support, opening,
                               destination_kind="front_cavity", tcp_pose=POSE, phase="preplace")
    assert result["status"] == "no_candidate"
    assert result["reason"] == "visible_object_exceeds_observed_opening_outer_envelope"
    assert result["candidates"] == []


@pytest.mark.parametrize("kind, expected_standoff", [("support_surface", .08),
                                                     ("open_container", .16)])
def test_front_cavity_standoff_repair_leaves_other_destination_heights_unchanged(
    kind, expected_standoff,
):
    result = propose_placement(block(), patch(), patch(z=.08) if kind == "open_container" else None,
                               destination_kind=kind, tcp_pose=POSE, phase="preplace")
    assert result["status"] == "candidates"
    assert all(candidate["metrics"]["visible_lower_surface_standoff_m"] == pytest.approx(expected_standoff)
               for candidate in result["candidates"])


@pytest.mark.parametrize("kind", ["open_container", "front_cavity"])
def test_preplace_observed_footprint_stays_in_opening_outer_envelope_without_passage_claim(kind):
    obj = block(center=(.10, .10, .15))
    if kind == "open_container":
        support = patch()
        opening = geometry([[x, y, .08] for x in np.arange(-.065, .066, .005)
                            for y in np.arange(-.065, .066, .005)])
    else:
        support, _ = cavity()
        opening = geometry([[x, 0, z] for x in np.arange(-.065, .066, .005)
                            for z in np.arange(.01, .201, .005)])
    result = propose_placement(obj, support, opening, destination_kind=kind,
                               tcp_pose=POSE, phase="preplace")
    assert result["candidates"]
    source = np.asarray(obj["voxel_points_world_m"])
    for candidate in result["candidates"]:
        translated = source + np.asarray(candidate["target_pose"]["position_m"]) - POSE[:3]
        assert np.min(translated[:, 0]) >= -.065 - 1e-9
        assert np.max(translated[:, 0]) <= .065 + 1e-9
        if kind == "open_container":
            assert np.min(translated[:, 1]) >= -.065 - 1e-9
            assert np.max(translated[:, 1]) <= .065 + 1e-9
        else:
            assert np.min(translated[:, 2]) >= .01 - 1e-9
            assert np.max(translated[:, 2]) <= .20 + 1e-9
        assert candidate["metrics"]["opening_outer_envelope_is_passage_certificate"] is False


@pytest.mark.parametrize("kind", ["open_container", "front_cavity"])
def test_unrelated_support_and_opening_planes_do_not_establish_an_interior(kind):
    support, boundary = (patch(), patch(z=.08)) if kind == "open_container" else cavity()
    boundary = geometry(np.asarray(boundary["voxel_points_world_m"]) + [1.0, 0, 0])
    result = assess_geometry(block(), support, boundary, destination_kind=kind, tcp_pose=POSE)
    assert result["geometry"]["opening"]["status"] == "unknown"
    assert result["geometry"]["opening"]["reason"] == "selected_support_does_not_overlap_observed_opening"
    assert result["checks"]["region"]["status"] == "unknown"


def test_opening_diagonal_sliver_is_not_a_resolved_two_dimensional_envelope():
    # Axis-aligned spans alone are large for this thin diagonal strip.
    along = np.arange(-.06, .061, .005)
    across = np.asarray([-.00025, 0, .00025])
    boundary = geometry([[(u-v)/np.sqrt(2), (u+v)/np.sqrt(2), .08]
                         for u in along for v in across])
    result = assess_geometry(block(), patch(), boundary,
                             destination_kind="open_container", tcp_pose=POSE)
    assert result["geometry"]["opening"]["status"] == "unknown"
    assert result["geometry"]["opening"]["reason"] == "opening_outer_envelope_degenerate"
    assert result["checks"]["region"]["status"] == "unknown"


def test_parallel_support_planes_and_line_are_not_unique_support():
    first = np.asarray(patch()["voxel_points_world_m"])
    # Repeat enough to exercise the bounded plane-fit sampling path too.
    first = np.repeat(first, 3, axis=0)
    second = first[:int(len(first)*.80)] + [0,0,.08]
    result = fit_support_geometry(geometry(np.concatenate([first,second])))
    assert result["status"] == "unknown"
    assert result["reason"] == "multiple_parallel_planes"
    assert fit_support_geometry(geometry([[i*.01,0,0] for i in range(20)]))["status"] == "unknown"


def test_every_observed_object_corner_is_used_for_candidate_fit():
    small = np.asarray(block()["voxel_points_world_m"])
    dense = np.repeat(small, 100, axis=0)
    dense[123] = [1.0,0,.04]
    result = propose_placement(geometry(dense), patch(), destination_kind="support_surface",
                               tcp_pose=POSE, phase="preplace")
    assert result["status"] == "no_candidate"


@pytest.mark.parametrize("bad", [[], [[float("nan"),0,0]], [[0,0]], [[0,0,float("inf")]]])
def test_malformed_geometry_fails_closed(bad):
    with pytest.raises(ValueError):
        assess(geometry(bad))


def test_invalid_pose_nonfinite_rim_and_modes_fail_closed():
    with pytest.raises(ValueError):
        assess_geometry(block(), patch(), destination_kind="support_surface", tcp_pose=[0]*7)
    with pytest.raises(ValueError):
        fit_rim_geometry([[0,0,float("nan")]]*6)
    with pytest.raises(ValueError):
        propose_placement(block(), patch(), destination_kind="support_surface", tcp_pose=POSE,
                           phase="execute")


def test_pure_repeatable_finite_json_and_no_input_mutation():
    obj, surface = block(), patch()
    originals = copy.deepcopy((obj,surface))
    a = propose_placement(obj, surface, destination_kind="support_surface", tcp_pose=POSE,
                           phase="preplace")
    b = propose_placement(obj, surface, destination_kind="support_surface", tcp_pose=POSE,
                           phase="preplace")
    assert a == b
    assert (obj,surface) == originals
    json.dumps(a, allow_nan=False)


def test_held_object_offset_reaches_the_projected_measurements():
    """The offset that explains a release residual must leave the audit record.

    ``assess_geometry`` returns it as a sibling of ``geometry``, so projecting
    ``geometry`` alone kept the only delivered statement of where the visible held
    object sits relative to the controlled point out of every result.
    """

    center = (0.03, -0.02, 0.04)
    result = assess(block(center=center))
    expected = np.asarray(center) - np.asarray(POSE[:3])
    assert result["object_visible_offset_from_tcp_world_m"] == pytest.approx(expected, abs=1e-9)

    projected = measurements(public_geometry(result))
    named = {entry["name"]: entry for entry in projected}
    assert "object_visible_offset_from_tcp_world_m" in named
    assert named["object_visible_offset_from_tcp_world_m"]["vector"] == pytest.approx(
        expected, abs=1e-9
    )

    # The measurement shape stays closed: one named vector, no scalar or text.
    entry = named["object_visible_offset_from_tcp_world_m"]
    assert set(entry) == {"name", "scalar", "vector", "text"}
    assert entry["scalar"] is None and entry["text"] is None


def test_assessment_written_before_the_projection_still_projects():
    """A replayed assessment carries no offset key; its absence is not a failure."""

    legacy = {"geometry": assess(block())["geometry"]}
    projected = measurements(public_geometry(legacy))
    assert projected
    assert all(
        entry["name"] != "object_visible_offset_from_tcp_world_m" for entry in projected
    )
