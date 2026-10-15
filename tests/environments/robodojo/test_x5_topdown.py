"""The ARX X5 public-RGB-D top-down grasp family, compiled to link6.

EXP11A g003: every object at most 25 mm tall got zero Contact-GraspNet
candidates in 8 of 8 calls, and manual top-down closes on them stalled 58-100%
of the time, slipping more when the jaw centre sat over 10 mm above the
object's mid-height. The live RoboDojo surface now fuses the deterministic
top-down family with the learned one, as RoboTwin's ALOHA surfaces do, with an
X5 height rule: support-aware mid-height, fingertips a clearance above the
support, robust top minus a clearance for tall objects, never below mid-height.

All geometry is synthetic and offline: identity camera-to-world, a table plane
at z = 0 and box-shaped targets.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.tools import x5_topdown
from ora.agent.tools.arx_x5_gripper import (
    ARX_X5_GRIPPER_GEOMETRY, ARX_X5_LINK6_COMPILER_IDENTITIES,
)
from ora.agent.tools.grasp import (
    GRASP_COMPILER_IDENTITIES, AlohaHybridGraspProposer, PcaTopDownGraspProposer,
    _compile_grasp_candidate, _compiled_grasp_geometry,
)

BOUNDS = {"x": [-0.3, 0.3], "y": [-0.3, 0.3], "z": [-0.05, 0.5]}
LEFT = x5_topdown.X5_TOPDOWN_COMPILER_IDENTITIES["left"]
LINK6_TO_CENTRE = float(ARX_X5_GRIPPER_GEOMETRY["contact_baseline_link6_x_m"])


def _box(centre_xy, size_xy, height, *, step=0.004):
    """Top face and four sides of an axis-aligned box standing on z = 0."""
    (cx, cy), (sx, sy) = centre_xy, size_xy
    xs = np.arange(cx - sx / 2, cx + sx / 2 + 1e-9, step)
    ys = np.arange(cy - sy / 2, cy + sy / 2 + 1e-9, step)
    zs = np.arange(0.0, height + 1e-9, step)
    top = [(x, y, height) for x in xs for y in ys]
    sides = ([(x, cy - sy / 2, z) for x in xs for z in zs] + [(x, cy + sy / 2, z) for x in xs for z in zs]
             + [(cx - sx / 2, y, z) for y in ys for z in zs] + [(cx + sx / 2, y, z) for y in ys for z in zs])
    return np.asarray(top + sides, dtype=np.float64)


def _table(centre_xy, *, half=0.12, step=0.005, z=0.0):
    xs = np.arange(centre_xy[0] - half, centre_xy[0] + half, step)
    ys = np.arange(centre_xy[1] - half, centre_xy[1] + half, step)
    return np.asarray([(x, y, z) for x in xs for y in ys], dtype=np.float64)


def _propose(target, scene, *, bounds=BOUNDS, identity=LEFT):
    return PcaTopDownGraspProposer().propose(
        scene_points_camera_m=scene, target_points_camera_m=target, camera_to_world=np.eye(4),
        target_ref="target-1", workspace_bounds_m=bounds, max_candidates=3, compiler_identity=identity,
    )


def _jaw_centre_z(candidate):
    return candidate["grasp_center_xyz_m"][2] - LINK6_TO_CENTRE


def test_the_identity_is_the_released_v1_plus_a_topdown_profile():
    for arm, identity in x5_topdown.X5_TOPDOWN_COMPILER_IDENTITIES.items():
        released = ARX_X5_LINK6_COMPILER_IDENTITIES[arm]
        assert GRASP_COMPILER_IDENTITIES[identity["id"]] is identity
        assert identity["id"] == f"contact_graspnet_to_robodojo_x5_{arm}_link6_v2" != released["id"]
        extra = set(identity) - set(released)
        assert extra == {"public_rgbd_topdown_profile"}
        same = {key for key in released if key not in {"id", "predicted_opening_source"}}
        assert {key: identity[key] for key in same} == {key: released[key] for key in same}
        assert identity["maximum_opening_m"] == 0.08
    assert x5_topdown.FINGERTIP_BEYOND_CONTACT_CENTRE_M == pytest.approx(0.0075)


def test_a_flat_15_mm_object_is_grasped_low_with_fingertips_clear_of_the_support():
    target = _box((0.05, 0.0), (0.04, 0.02), 0.015)
    candidates, diagnostics = _propose(target, np.vstack([_table((0.05, 0.0)), target]))
    assert 1 <= len(candidates) <= 2  # the profile's two-candidate bound
    for candidate in candidates:
        height = candidate["topdown_height"]
        assert height["support_z_m"] == pytest.approx(0.0, abs=1e-6)
        assert height["support_source"] == "scene_depth_ring"
        # mid-height 7.5 mm would put the fingertips at 0 mm; the floor lifts
        # the jaw centre to 7.5 mm fingertip reach + 3 mm clearance.
        assert _jaw_centre_z(candidate) == pytest.approx(0.0105, abs=1e-6)
        assert height["fingertip_z_m"] == pytest.approx(0.003, abs=1e-6)
        assert height["reaches_object"] is True
        assert candidate["approach_world"] == [0.0, 0.0, -1.0]
        assert candidate["pregrasp_center_xyz_m"][2] == pytest.approx(
            candidate["grasp_center_xyz_m"][2] + 0.04)
    best = candidates[0]
    assert best["gripper_opening_m"] == pytest.approx(0.02 + 0.006, abs=0.004)  # across the short side
    assert best["jaw_world"][1] == pytest.approx(1.0, abs=0.05) or best["jaw_world"][1] == pytest.approx(-1.0, abs=0.05)
    assert diagnostics["rejected"] == [] or all("reasons" in item for item in diagnostics["rejected"])


def test_a_34_mm_cube_is_grasped_at_mid_height():
    target = _box((0.0, 0.05), (0.034, 0.034), 0.034)
    candidates, _ = _propose(target, np.vstack([_table((0.0, 0.05)), target]))
    assert candidates
    for candidate in candidates:
        assert _jaw_centre_z(candidate) == pytest.approx(0.017, abs=0.002)
        assert candidate["topdown_height"]["mid_height_z_m"] == pytest.approx(0.017, abs=0.002)


def test_a_tall_object_is_grasped_below_its_top_and_never_below_mid_height():
    target = _box((0.0, 0.0), (0.03, 0.03), 0.12)
    candidates, _ = _propose(target, np.vstack([_table((0.0, 0.0)), target]))
    assert candidates
    height = candidates[0]["topdown_height"]
    assert _jaw_centre_z(candidates[0]) == pytest.approx(0.12 - 0.02, abs=0.002)
    assert _jaw_centre_z(candidates[0]) > height["mid_height_z_m"]


def test_an_object_wider_than_the_nominal_aperture_is_rejected_with_its_numbers():
    target = _box((0.0, 0.0), (0.10, 0.10), 0.03)
    candidates, diagnostics = _propose(target, np.vstack([_table((0.0, 0.0)), target]))
    assert candidates == []
    assert diagnostics["rejected"]
    for item in diagnostics["rejected"]:
        assert "aperture_exceeds_embodiment_limit" in item["reasons"]
        assert item["maximum_opening_m"] == 0.08 and item["required_opening_m"] > 0.08
        assert item["limit_semantics"] == "compiler_admission_not_measured_physical_aperture"


@pytest.mark.parametrize(("bounds", "reason"), [
    ({**BOUNDS, "x": [-0.3, 0.2]}, "contact_pose_outside_workspace"),
    ({**BOUNDS, "z": [-0.05, 0.18]}, "hover_pose_outside_workspace"),
])
def test_a_candidate_at_the_workspace_edge_is_rejected_by_its_link6_pose(bounds, reason):
    target = _box((0.21, 0.0), (0.04, 0.02), 0.015)
    candidates, diagnostics = _propose(target, np.vstack([_table((0.21, 0.0)), target]), bounds=bounds)
    assert candidates == []
    assert all(reason in item["reasons"] for item in diagnostics["rejected"])


def test_without_visible_support_the_rule_falls_back_to_the_target_extent():
    target = _box((0.0, 0.0), (0.04, 0.02), 0.015)
    candidates, _ = _propose(target, target)
    height = candidates[0]["topdown_height"]
    assert height["support_z_m"] is None and height["support_source"] == "unavailable_target_extent_only"
    assert height["reaches_object"] is True


def test_a_topdown_candidate_compiles_to_link6_with_its_contact_at_the_jaw_centre():
    target = _box((0.05, 0.0), (0.04, 0.02), 0.015)
    candidates, _ = _propose(target, np.vstack([_table((0.05, 0.0)), target]))
    raw = {**candidates[0], "rank": 0}
    compiled = _compile_grasp_candidate(raw, compiler_identity=LEFT, public_contact_pose=True)
    assert compiled["contact_pose"]["position_m"] == pytest.approx(raw["grasp_center_xyz_m"])
    geometry = _compiled_grasp_geometry(raw, compiler_identity=LEFT)
    assert geometry["contact_baseline"][2] == pytest.approx(0.0105, abs=1e-6)
    assert geometry["approach_distance"] == pytest.approx(0.04)
    # link6 local +X (the approach) points straight down.
    x, y, z, w = compiled["pregrasp_pose"]["orientation_xyzw"]
    approach = [1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)]
    assert approach == pytest.approx([0.0, 0.0, -1.0], abs=1e-6)


def test_the_aloha_profile_is_unchanged():
    identity = GRASP_COMPILER_IDENTITIES["contact_graspnet_to_robotwin_aloha_left_grip_center_v3"]
    target = _box((0.0, 0.0), (0.04, 0.02), 0.015)
    candidates, _ = _propose(target, np.vstack([_table((0.0, 0.0)), target]), identity=identity)
    assert candidates and "topdown_height" not in candidates[0]
    top = float(np.percentile(np.asarray(
        [p for p in target if np.all((p >= np.percentile(target, 2, axis=0)) & (p <= np.percentile(target, 98, axis=0)))]
    )[:, 2], 95))
    assert candidates[0]["grasp_center_xyz_m"][2] == pytest.approx(top + 0.008, abs=1e-6)


class _NoLearned:
    backend_id = "contact_graspnet_test_double"
    backend_version = "v1"

    def configuration(self):
        return {"pregrasp_offset_m": 0.04}

    def propose(self, **kwargs):
        return [], {"request_id": "cgn-empty", "rejected": [], "model": {
            "family": "contact_graspnet", "source_commit": "x", "checkpoint_sha256": "y",
            "config_sha256": "0" * 64}}


def test_the_fused_backend_supplies_topdown_when_the_learned_family_is_empty():
    backend = AlohaHybridGraspProposer(contact_backend=_NoLearned(), backend_id=x5_topdown.HYBRID_BACKEND_ID)
    target = _box((0.05, 0.0), (0.04, 0.02), 0.015)
    candidates, diagnostics = backend.propose(
        scene_points_camera_m=np.vstack([_table((0.05, 0.0)), target]), target_points_camera_m=target,
        camera_to_world=np.eye(4), target_ref="target-1", workspace_bounds_m=BOUNDS,
        max_candidates=32, compiler_identity=LEFT)
    assert candidates and all(item["pose_convention"] == "pca_topdown_parallel_jaw_center_v1"
                              for item in candidates)
    assert diagnostics["candidate_family_counts"]["learned_available"] == 0
    assert backend.configuration()["selection"] == "up_to_two_topdown_then_contact_fill_v1"
    assert backend.backend_id == "contact_graspnet_plus_x5_rgbd_topdown"


def test_only_the_live_dojo_surface_selects_the_x5_family():
    dojo = resolve_tool_surface("robodojo_placement_closed_skills_v2")
    assert dojo.supports(x5_topdown.CAPABILITY)
    assert x5_topdown.hybrid_backend_arguments(dojo) == {"backend_id": x5_topdown.HYBRID_BACKEND_ID}
    twin = resolve_tool_surface("robotwin_toolchain_b2_closed_noservo_v2")
    assert x5_topdown.hybrid_backend_arguments(twin) == {}
    libero = resolve_tool_surface("libero_placement_direct_overhead_v1")
    assert not libero.supports(x5_topdown.CAPABILITY)
    assert x5_topdown.hybrid_backend_arguments(libero) is None
    help_text = next(item["description"] for item in dojo.definitions if item["name"] == "grasp_plan")
    assert "Top-down candidates from the target mask" in help_text
    assert "learned candidates only" not in help_text


def test_the_dojo_runtime_compiles_with_the_v2_identity_and_admits_the_fused_result(tmp_path):
    """The real dispatcher: v2 compiler, fused backend id, IK selection, result admission.

    The synthetic Dojo head view's mask is 0.2 m wide, so the top-down family
    is refused on aperture (and says so) while the learned candidate is shown.
    """
    import copy

    import tests.environments.robodojo.test_robodojo_closed_skills_default as parity
    from tests.agent.motion.test_reach_check import _stub_bridge

    runtime = parity.tool_runtime(tmp_path)
    runtime.surface = resolve_tool_surface("robodojo_placement_closed_skills_v2")
    runtime.allowed_tools = tuple(item["name"] for item in runtime.surface.definitions)
    assert runtime._grasp_compiler_id("left") == "contact_graspnet_to_robodojo_x5_left_link6_v2"
    learned = {
        "candidate_id": "grasp-0123456789abcdef01234567", "source_index": 0, "rank": 0,
        "source_gripper_origin_xyz_m": [0.0, 0.0, 0.6966], "source_pregrasp_origin_xyz_m": [0.0, 0.0, 0.6566],
        "approach_world": [0.0, 0.0, 1.0], "jaw_world": [1.0, 0.0, 0.0], "gripper_opening_m": 0.05,
        "backend_score": 0.7, "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "checks": {"workspace": "passed", "aperture": "passed", "ik": "not_checked", "collision": "not_checked"},
        "execution_ready": False,
    }
    seen = []

    class Learned(_NoLearned):
        def propose(self, **kwargs):
            seen.append(kwargs["compiler_identity"]["id"])
            _, diagnostics = super().propose(**kwargs)
            return [copy.deepcopy(learned)], diagnostics

    runtime.grasp_backend = AlohaHybridGraspProposer(contact_backend=Learned(),
                                                     backend_id=x5_topdown.HYBRID_BACKEND_ID)
    requests = []
    runtime.align_plan_callback = _stub_bridge(requests)
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    result = runtime.invoke("grasp_plan", {"target_ref": grounded["candidates"][0]["target_ref"], "arm": "left"})
    assert seen == ["contact_graspnet_to_robodojo_x5_left_link6_v2"]
    assert result["compiler"]["id"] == "contact_graspnet_to_robodojo_x5_left_link6_v2"
    assert result["backend"]["id"] == x5_topdown.HYBRID_BACKEND_ID
    assert [item["candidate_id"] for item in result["candidates"]] == [learned["candidate_id"]]
    assert result["ik_reachability"]["checked_candidate_count"] == 1
    assert result["diagnostics"]["candidate_family_counts"] == {
        "learned_available": 1, "public_rgbd_topdown_available": 0, "selected": 1}
    assert {reason for item in result["diagnostics"]["rejected"] for reason in item["reasons"]} == {
        "aperture_exceeds_embodiment_limit"}
    assert [query["interaction"] for query in requests[0]["queries"]] == ["free_space", "contact_approach"]


def test_both_families_enter_one_ik_ordered_pool():
    """A top-down and a learned candidate are ordered by the same IK and tilt rule."""
    from ora.agent.tools import reach_check
    from tests.agent.motion.test_reach_check import _stub_bridge

    target = _box((0.05, 0.0), (0.04, 0.02), 0.015)
    topdown, _ = _propose(target, np.vstack([_table((0.05, 0.0)), target]))
    tilted = math.radians(50.0)
    approach = [math.sin(tilted), 0.0, -math.cos(tilted)]
    origin = [0.05, 0.0, 0.12]
    learned = {
        "candidate_id": "grasp-learned000000000000000", "source_index": 0,
        "source_gripper_origin_xyz_m": origin,
        "source_pregrasp_origin_xyz_m": [o - a * 0.04 for o, a in zip(origin, approach)],
        "approach_world": approach, "jaw_world": [0.0, 1.0, 0.0], "gripper_opening_m": 0.03,
        "backend_score": 0.9, "pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
        "checks": {"workspace": "passed", "aperture": "passed", "ik": "not_checked", "collision": "not_checked"},
        "execution_ready": False,
    }
    raws = [learned, *topdown]  # learned first: the backend order
    compiled = [{**_compile_grasp_candidate({**raw, "rank": index}, compiler_identity=LEFT,
                                            public_contact_pose=True), "rank": index}
                for index, raw in enumerate(raws)]
    chosen = reach_check.select_grasp_candidates(
        compiled, arm="left", bridge=_stub_bridge([]), workspace_bounds_m=BOUNDS,
        approach_axis_local=[1.0, 0.0, 0.0])
    order = [item["candidate_id"] for item in chosen["ordered"]]
    # Both reachable; the vertical top-down candidate outranks the 50-degree learned one.
    assert order[0] == topdown[0]["candidate_id"] and order[-1] == learned["candidate_id"]
    tilt = {item["candidate_id"]: item["ik_reachability"]["approach_tilt_from_vertical_deg"]
            for item in chosen["ordered"]}
    assert tilt[learned["candidate_id"]] == pytest.approx(50.0, abs=0.2)
    assert tilt[topdown[0]["candidate_id"]] == pytest.approx(0.0, abs=0.2)
    assert chosen["summary"]["checked_candidate_count"] == len(raws)
