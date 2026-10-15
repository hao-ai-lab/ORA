"""Public-RGB-D top-down grasp candidates for the ARX X5, compiled to link6.

RoboTwin's ALOHA surfaces fuse Contact-GraspNet with a deterministic top-down
family (``PcaTopDownGraspProposer`` under ``AlohaHybridGraspProposer``). The
RoboDojo X5 had the learned family only, and a recorded batch measured the cost:
every object at most 25 mm tall (number pieces, cloth, watch, pen) received
zero Contact-GraspNet candidates in 8 of 8 calls, while manual top-down closes
on those objects stalled 58-100% of the time and slipped more often when the
jaw centre sat more than 10 mm above the object's mid-height.

This module adds what the X5 needs and nothing ALOHA uses:

* a compiler identity family ``contact_graspnet_to_robodojo_x5_{arm}_link6_v2``
  -- the released v1 identity plus a top-down profile -- so v1 queries and
  every frozen Dojo surface keep their exact identity;
* a support-aware grasp height: the jaw contact centre goes to the object's
  mid-height between the robust top and the support plane (read from scene
  depth around the mask), no lower than keeps the fingertips a clearance above
  the support, and for tall objects to the robust top minus a clearance,
  never below mid-height;
* the link6 conversion the proposer needs: top-down candidates name the
  controlled point, which on the X5 is 150.1 mm behind the jaw contact centre
  along the approach axis.

The X5 geometry is read from ``arx_x5_gripper`` (a pinned embodiment source
this module deliberately does not edit). Aperture admission stays the nominal
0.08 m with the ``compiler_admission_not_measured_physical_aperture``
semantics. The support plane is computed here from public depth; when the
segmented-localization support fields land (``support_z_m``/``mid_height_z_m``)
this estimate should read them instead.
"""
from __future__ import annotations

from typing import Any, Mapping

from .arx_x5_gripper import ARX_X5_GRIPPER_GEOMETRY, ARX_X5_LINK6_COMPILER_IDENTITIES

CAPABILITY = "robodojo_x5_public_rgbd_topdown_grasp_v1"
HYBRID_BACKEND_ID = "contact_graspnet_plus_x5_rgbd_topdown"
HEIGHT_RULE = "support_aware_mid_height_v1"
#: The pad inner faces span link6 x 0.1426-0.1576 m around the 0.1501 m contact
#: baseline, so the fingertip reaches 7.5 mm beyond the contact centre.
FINGERTIP_BEYOND_CONTACT_CENTRE_M = round(
    float(ARX_X5_GRIPPER_GEOMETRY["pad_inner_face_span_link6_x_m"][1])
    - float(ARX_X5_GRIPPER_GEOMETRY["contact_baseline_link6_x_m"]), 4,
)
PROFILE = {
    "height_rule": HEIGHT_RULE,
    #: Robust visible top of the target mask, as ALOHA reads it.
    "surface_percentile": 95.0,
    #: For a tall object the contact centre sits this far below the robust
    #: top, so the whole 15 mm pad lies on the object; never below mid-height.
    "top_clearance_m": 0.02,
    #: Fingertip clearance above the support plane.
    "table_clearance_m": 0.003,
    "fingertip_beyond_contact_centre_m": FINGERTIP_BEYOND_CONTACT_CENTRE_M,
    #: The same 40 mm pre-grasp standoff Contact-GraspNet candidates use on
    #: RoboDojo, inside the 50 mm local contact envelope.
    "pregrasp_standoff_m": 0.04,
    "maximum_candidates": 2,
}
#: Scene points whose horizontal distance to the target footprint lies in this
#: band are read as the support around the object.
SUPPORT_RING_M = (0.01, 0.06)
SUPPORT_MINIMUM_POINTS = 16
SUPPORT_PERCENTILE = 30.0

X5_TOPDOWN_COMPILER_IDENTITIES = {
    arm: {
        **identity,
        "id": f"contact_graspnet_to_robodojo_x5_{arm}_link6_v2",
        "public_rgbd_topdown_profile": dict(PROFILE),
        # Two families share one query identity: learned candidates publish the
        # Contact-GraspNet width head, top-down ones the mask width plus
        # clearance. Neither is a measured X5 aperture.
        "predicted_opening_source": (
            "learned:contact_graspnet_panda_width_head_bin;"
            "topdown:public_rgbd_mask_width_plus_clearance"
        ),
    }
    for arm, identity in ARX_X5_LINK6_COMPILER_IDENTITIES.items()
}


def compiler_id(arm: str) -> str:
    return f"contact_graspnet_to_robodojo_x5_{arm}_link6_v2"


def hybrid_backend_arguments(surface: Any) -> dict[str, Any] | None:
    """Constructor overrides for the fused backend a surface selects, else None."""

    if surface.supports("robotwin_aloha_public_rgbd_topdown_grasp_v1"):
        return {}
    if surface.supports(CAPABILITY):
        return {"backend_id": HYBRID_BACKEND_ID}
    return None


def support_plane_z(target_world: Any, scene_world: Any) -> float | None:
    """Support height around the target footprint from public scene depth, or None."""
    import numpy as np

    target = np.asarray(target_world, dtype=np.float64)
    scene = np.asarray(scene_world, dtype=np.float64).reshape(-1, 3)
    scene = scene[np.isfinite(scene).all(axis=1)]
    if len(target) < 3 or len(scene) < SUPPORT_MINIMUM_POINTS:
        return None
    low = np.percentile(target[:, :2], 2, axis=0)
    high = np.percentile(target[:, :2], 98, axis=0)
    outside = np.maximum(np.maximum(low - scene[:, :2], 0.0), np.maximum(scene[:, :2] - high, 0.0))
    distance = np.linalg.norm(outside, axis=1)
    top = float(np.percentile(target[:, 2], 95))
    ring = (distance > SUPPORT_RING_M[0]) & (distance <= SUPPORT_RING_M[1]) & (scene[:, 2] <= top)
    if int(np.count_nonzero(ring)) < SUPPORT_MINIMUM_POINTS:
        return None
    # The support cannot lie above the lowest visible part of the target.
    return min(float(np.percentile(scene[ring, 2], SUPPORT_PERCENTILE)),
               float(np.percentile(target[:, 2], 5)))


def topdown_contact_height(target_world: Any, scene_points_camera_m: Any, camera_to_world: Any,
                           profile: Mapping[str, Any]) -> dict[str, Any]:
    """The jaw contact-centre height and the evidence it was derived from.

    ``target_world`` is the proposer's filtered target cloud in world metres;
    the scene cloud arrives in the target camera frame with its transform.
    """
    import numpy as np

    if (not isinstance(profile, Mapping) or set(profile) != set(PROFILE)
            or profile["height_rule"] != HEIGHT_RULE):
        raise ValueError("X5 public RGB-D top-down profile is invalid")
    scene = np.asarray(scene_points_camera_m, dtype=np.float64).reshape(-1, 3)
    transform = np.asarray(camera_to_world, dtype=np.float64)
    scene_world = (transform @ np.column_stack([scene, np.ones(len(scene))]).T).T[:, :3]
    target = np.asarray(target_world, dtype=np.float64)
    top = float(np.percentile(target[:, 2], float(profile["surface_percentile"])))
    support = support_plane_z(target, scene_world)
    if support is None:
        mid = float((np.percentile(target[:, 2], 5) + np.percentile(target[:, 2], 95)) / 2)
        floor = None
    else:
        mid = (support + top) / 2
        floor = support + float(profile["fingertip_beyond_contact_centre_m"]) + float(
            profile["table_clearance_m"])
    centre = max(mid, top - float(profile["top_clearance_m"]))
    if floor is not None:
        centre = max(centre, floor)
    fingertip = centre - float(profile["fingertip_beyond_contact_centre_m"])
    return {
        "height_rule": HEIGHT_RULE,
        "contact_centre_z_m": centre,
        "robust_top_z_m": top,
        "support_z_m": support,
        "support_source": "scene_depth_ring" if support is not None else "unavailable_target_extent_only",
        "mid_height_z_m": mid,
        "fingertip_z_m": fingertip,
        # Fingertips at or above the robust top close on air.
        "reaches_object": fingertip < top,
    }
