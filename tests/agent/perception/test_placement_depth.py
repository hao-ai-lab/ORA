"""Placement-only depth uncertainty; no semantics, simulator or model calls."""

import numpy as np
import pytest

from ora.agent.tools.placement_depth import classify_boundary_depth, project_geometry


def scene():
    mask = np.zeros((24, 24), dtype=bool)
    mask[6:18, 6:18] = True
    depth = np.full(mask.shape, 0.9)
    depth[mask] = 0.6
    return mask, depth


def test_only_far_boundary_with_observed_outside_continuity_becomes_uncertain():
    mask, depth = scene()
    depth[6, 10] = .9  # Segmented boundary includes the adjacent farther plane.
    original = (mask.copy(), depth.copy())
    result = classify_boundary_depth(mask, depth)
    assert result["status"] == "usable"
    assert np.argwhere(result["uncertain_mask"]).tolist() == [[6, 10]]
    assert result["measurements"]["uncertain_pixel_count"] == 1
    assert result["retained_mask"][6, 11]
    assert np.array_equal(mask, original[0]) and np.array_equal(depth, original[1])


def test_thin_handle_and_real_foreground_corner_are_not_blanket_eroded():
    mask, depth = scene()
    mask[11, 18:22] = True
    depth[11, 18:22] = .65  # Thin, farther geometry, not the outside .9 m plane.
    depth[6, 11] = .64  # A real farther corner is not background-continuous.
    result = classify_boundary_depth(mask, depth)
    assert result["uncertain_mask"].sum() == 0
    assert result["retained_mask"][11, 18:22].all()
    assert result["retained_mask"][6, 11]


def test_far_interior_and_smooth_slopes_remain_evidence_even_if_inconvenient():
    mask, depth = scene()
    depth[10, 10] = .9  # Interior discontinuity is not boundary contamination.
    depth[6, 9:14] = np.linspace(.61, .65, 5)
    result = classify_boundary_depth(mask, depth)
    assert result["retained_mask"][10, 10]
    assert result["retained_mask"][6, 9:14].all()


def test_foreground_side_of_depth_edge_is_never_discarded():
    mask, depth = scene()
    result = classify_boundary_depth(mask, depth)
    assert np.array_equal(result["retained_mask"], mask)


def test_real_far_tip_matching_background_is_uncertain_not_proven_removed_background():
    mask, depth = scene()
    # Identical public evidence also permits a real far corner beside the
    # background. No geometric classifier may label this semantic background.
    depth[6, 10] = .9
    result = project_geometry(mask, depth, [[100., 0, 12], [0, 100., 12], [0, 0, 1]],
        np.eye(4), observation_id="obs", target_ref="tip", camera="wrist")
    evidence = result["depth_boundary_evidence"]
    assert evidence["uncertain_pixels_uv"] == [[10, 6]]
    assert evidence["uncertain_pixels_are_certified_background"] is False
    assert evidence["full_silhouette_established"] is False


def test_excessive_boundary_ambiguity_fails_closed():
    mask, depth = scene()
    depth[6, 6:18] = .9
    depth[17, 6:18] = .9
    result = classify_boundary_depth(mask, depth)
    assert result["status"] == "unknown"
    assert result["reason"] == "too_many_uncertain_boundary_pixels"
    assert result["measurements"]["uncertain_fraction"] > .10


def test_invalid_depth_is_explicit_and_never_reconstructed():
    mask, depth = scene()
    depth[7, 7] = np.nan
    depth[8, 8] = 0
    result = classify_boundary_depth(mask, depth)
    assert result["measurements"]["invalid_depth_pixel_count"] == 2
    assert not result["retained_mask"][7, 7] and not result["retained_mask"][8, 8]
    assert not result["uncertain_mask"][7, 7]


def test_projection_keeps_all_retained_points_and_separate_uncertainty():
    mask, depth = scene()
    depth[6, 10] = .9
    k = np.array([[100., 0, 12], [0, 100., 12], [0, 0, 1.]])
    result = project_geometry(mask, depth, k, np.eye(4),
        observation_id="obs", target_ref="target", camera="wrist")
    assert result["visible_point_count"] == int(mask.sum()) - 1
    # No voxel representative may silently discard a thin but retained corner.
    assert len(result["voxel_points_world_m"]) == result["visible_point_count"]
    evidence = result["depth_boundary_evidence"]
    assert evidence["uncertain_pixels_uv"] == [[10, 6]]
    assert len(evidence["uncertain_points_world_m"]) == 1
    assert evidence["uncertain_pixels_are_certified_background"] is False
    assert evidence["full_silhouette_established"] is False
    assert result["voxel_size_m"] == 0.0


@pytest.mark.parametrize("kind", ["shape", "nonboolean", "intrinsic", "extrinsic"])
def test_bad_projection_inputs_are_refused(kind):
    mask, depth = scene()
    k, e = np.eye(3), np.eye(4)
    if kind == "shape":
        depth = depth[:-1]
    elif kind == "nonboolean":
        mask = mask.astype(float) * 2
    elif kind == "intrinsic":
        k[0, 0] = 0
    else:
        e[0] = 0
    with pytest.raises(ValueError):
        project_geometry(mask, depth, k, e, observation_id="obs", target_ref="target", camera="wrist")


def placement_fixture(*, uncertain=True, usable=True):
    from ora.agent.tools.placement_depth import SCHEMA
    obj = {"voxel_points_world_m": [[x, y, z] for x in (-.01, 0, .01)
        for y in (-.01, 0, .01) for z in (.04, .08)]}
    support = {"voxel_points_world_m": [[x, y, 0] for x in np.linspace(-.15, .15, 31)
        for y in np.linspace(-.15, .15, 31)]}
    obj["depth_boundary_evidence"] = {"schema": SCHEMA, "status": "usable" if usable else "unknown",
        "reason": "local_far_side_boundary_uncertainty",
        "measurements": {"uncertain_pixel_count": 1 if uncertain else 0,
            "uncertain_fraction": .01 if uncertain else 0.0},
        "thresholds": {}, "uncertain_points_world_m": [[.8, .8, .04]] if uncertain else []}
    return obj, support


def test_conditional_candidate_does_not_upgrade_raw_object_unknown_to_fit():
    from ora.agent.tools.placement import propose_placement
    obj, support = placement_fixture()
    result = propose_placement(obj, support, destination_kind="support_surface",
        tcp_pose=[0, 0, .3, 0, 0, 0, 1], phase="preplace")
    assert result["status"] == "candidates"
    region = result["assessment"]["checks"]["region"]
    assert region["status"] == "unknown"
    assert region["reason"] == "uncertain_depth_boundary_full_object_fit_unresolved"
    quality = result["assessment"]["geometry"]["depth_boundary"]
    assert quality["raw_object_region_status"] == "unknown"
    assert quality["retained_object_region_status"] == "consistent"
    for candidate in result["candidates"]:
        assert candidate["metrics"]["conditional_on_unresolved_silhouette"] is True
        assert candidate["metrics"]["full_object_fit"] == "unknown"


def test_excessive_uncertainty_cannot_produce_a_candidate():
    from ora.agent.tools.placement import propose_placement
    obj, support = placement_fixture(usable=False)
    result = propose_placement(obj, support, destination_kind="support_surface",
        tcp_pose=[0, 0, .3, 0, 0, 0, 1], phase="preplace")
    assert result["candidates"] == []
    assert result["status"] == "unknown"
    assert result["reason"] == "public_depth_boundary_evidence_unusable"


def test_reliable_visible_contradiction_survives_uncertainty_annotation():
    from ora.agent.tools.placement import assess_geometry
    obj, support = placement_fixture()
    obj["voxel_points_world_m"] = (np.asarray(obj["voxel_points_world_m"]) + [.5, 0, 0]).tolist()
    result = assess_geometry(obj, support, destination_kind="support_surface", tcp_pose=[0, 0, .3, 0, 0, 0, 1])
    assert result["checks"]["region"]["status"] == "contradicted"


def test_opt_in_plane_refinement_rechecks_original_six_mm_inliers_after_refitting():
    from ora.agent.tools.placement import fit_support_geometry
    from ora.agent.tools.placement_depth import SCHEMA
    x, y = np.meshgrid(np.linspace(-.065, .065, 41), np.linspace(-.065, .065, 41))
    radius = np.sqrt(x*x + y*y) / .065
    selected = radius <= 1
    points = np.column_stack((x[selected], y[selected], .012 * radius[selected] ** 4))
    old = {"voxel_points_world_m": points.tolist()}
    assert fit_support_geometry(old)["status"] == "unknown"
    current = {**old, "depth_boundary_evidence": {"schema": SCHEMA}}
    result = fit_support_geometry(current)
    assert result["status"] == "consistent"
    assert result["thresholds"]["maximum_residual_m"] == .006
    assert result["thresholds"]["minimum_inlier_fraction"] == .55
    assert result["metrics"]["plane_residual_p95_m"] <= .006
    assert result["metrics"]["inlier_fraction"] >= .55
    assert result["metrics"]["refinement_iterations"] <= 4
    assert result["observed_points_world_m"]
    assert all(point in old["voxel_points_world_m"] for point in result["observed_points_world_m"])
