"""Deterministic Contact-GraspNet to Panda grip-site compilation."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

from ora.contracts.pose import matrix_quaternion


PANDA_GRIP_SITE_COMPILER_IDENTITY = {
    "id": "contact_graspnet_to_robosuite_panda_grip_site_v2",
    "embodiment_descriptor_id": "franka_panda_libero_pro_v1",
    "embodiment_descriptor_sha256": (
        "a7b66c77978ea23f8c92b136fd575326a2e82bcd0b4e5f232219ae7532afa033"
    ),
    "source_pose_convention": "contact_graspnet_panda_gripper_origin_opencv_v2",
    "controlled_point": "robosuite_grip_site",
    "controlled_point_semantics": "grip_center",
    "contact_graspnet_gripper_depth_m": 0.1034,
    "source_origin_to_controlled_point_m": 0.097,
    "controlled_point_to_contact_baseline_m": 0.0064,
    "axis_mapping": {
        "model_jaw_x": "grip_site_local_positive_x",
        "model_approach_z": "grip_site_local_positive_z",
    },
}


def compiled_grasp_geometry(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve one learned proposal frame into Panda controlled points."""

    import numpy as np

    convention = candidate.get("pose_convention")
    if convention == PANDA_GRIP_SITE_COMPILER_IDENTITY["source_pose_convention"]:
        source_origin = np.asarray(
            candidate["source_gripper_origin_xyz_m"], dtype=np.float64,
        )
        source_pregrasp = np.asarray(
            candidate["source_pregrasp_origin_xyz_m"], dtype=np.float64,
        )
        direct_grip_site = direct_pregrasp_grip_site = None
    elif convention == "pca_topdown_parallel_jaw_center_v1":
        direct_grip_site = np.asarray(
            candidate["grasp_center_xyz_m"], dtype=np.float64,
        )
        direct_pregrasp_grip_site = np.asarray(
            candidate["pregrasp_center_xyz_m"], dtype=np.float64,
        )
        source_origin = source_pregrasp = np.zeros(3, dtype=np.float64)
    else:
        raise ValueError("grasp candidate pose convention is not compiler-qualified")
    approach = np.asarray(candidate["approach_world"], dtype=np.float64)
    jaw = np.asarray(candidate["jaw_world"], dtype=np.float64)
    if any(
        value.shape != (3,)
        for value in (source_origin, source_pregrasp, approach, jaw)
    ):
        raise ValueError("grasp candidate vectors must each contain three values")
    if not all(
        np.isfinite(value).all()
        for value in (source_origin, source_pregrasp, approach, jaw)
    ):
        raise ValueError("grasp candidate contains non-finite geometry")
    approach_norm = float(np.linalg.norm(approach))
    if not math.isfinite(approach_norm) or approach_norm <= 1e-6:
        raise ValueError("grasp approach axis is degenerate")
    z_axis = approach / approach_norm
    x_axis = jaw - z_axis * float(np.dot(jaw, z_axis))
    x_norm = float(np.linalg.norm(x_axis))
    if not math.isfinite(x_norm) or x_norm <= 1e-6:
        raise ValueError("grasp jaw and approach axes are degenerate")
    x_axis /= x_norm
    y_axis = np.cross(z_axis, x_axis)
    y_norm = float(np.linalg.norm(y_axis))
    if not math.isfinite(y_norm) or y_norm <= 1e-6:
        raise ValueError("grasp axes cannot define a lateral direction")
    y_axis /= y_norm
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    if float(np.linalg.det(rotation)) < 0.999:
        raise ValueError("grasp axes do not form a right-handed orientation")
    source_to_grip_site = float(
        PANDA_GRIP_SITE_COMPILER_IDENTITY[
            "source_origin_to_controlled_point_m"
        ]
    )
    if direct_grip_site is not None and direct_pregrasp_grip_site is not None:
        source_origin = direct_grip_site - z_axis * source_to_grip_site
        source_pregrasp = (
            direct_pregrasp_grip_site - z_axis * source_to_grip_site
        )
    approach_distance = float(np.dot(source_origin - source_pregrasp, z_axis))
    lateral_error = float(np.linalg.norm(
        (source_origin - source_pregrasp) - z_axis * approach_distance,
    ))
    if (
        not math.isfinite(approach_distance)
        or not 0.005 <= approach_distance <= 0.10
        or lateral_error > 1e-4
    ):
        raise ValueError("grasp pregrasp is not one bounded axial servo away")
    gripper_depth = float(
        PANDA_GRIP_SITE_COMPILER_IDENTITY[
            "contact_graspnet_gripper_depth_m"
        ]
    )
    if direct_grip_site is not None and direct_pregrasp_grip_site is not None:
        grip_site = direct_grip_site
        pregrasp_grip_site = direct_pregrasp_grip_site
    else:
        grip_site = source_origin + z_axis * source_to_grip_site
        pregrasp_grip_site = grip_site - z_axis * approach_distance
    contact_baseline = source_origin + z_axis * gripper_depth
    opening = float(candidate["gripper_opening_m"])
    contact_negative_jaw = contact_baseline - x_axis * opening / 2.0
    contact_positive_jaw = contact_baseline + x_axis * opening / 2.0
    return {
        "source_gripper_origin": source_origin,
        "source_pregrasp_origin": source_pregrasp,
        "grip_site": grip_site,
        "pregrasp_grip_site": pregrasp_grip_site,
        "contact_baseline": contact_baseline,
        "contact_negative_jaw": contact_negative_jaw,
        "contact_positive_jaw": contact_positive_jaw,
        "approach_distance": approach_distance,
        "rotation": rotation,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    }


def compile_grasp_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one valid proposal into the exact policy motion handoff."""

    geometry = compiled_grasp_geometry(candidate)
    if float(geometry["approach_distance"]) > 0.05:
        raise ValueError("grasp approach exceeds bounded servo distance")
    orientation_xyzw = matrix_quaternion(geometry["rotation"].tolist())
    return {
        "candidate_id": candidate["candidate_id"],
        "rank": int(candidate["rank"]),
        "pregrasp_pose": {
            "position_m": [
                float(item) for item in geometry["pregrasp_grip_site"]
            ],
            "orientation_xyzw": orientation_xyzw,
        },
        "contact_pose": {
            "position_m": [float(item) for item in geometry["grip_site"]],
            "orientation_xyzw": orientation_xyzw,
        },
        "approach_delta": {
            "frame": "grip_site",
            "translation_m": [
                0.0, 0.0, float(geometry["approach_distance"]),
            ],
            "rotation_rotvec_rad": [0.0, 0.0, 0.0],
        },
        "gripper_opening_m": float(candidate["gripper_opening_m"]),
        "backend_score": float(candidate["backend_score"]),
        "compiler_checks": {
            "finite_geometry": "passed",
            "right_handed_grip_site_axes": "passed",
            "bounded_local_approach": "passed",
            "source_origin_to_grip_site": "passed",
            "contact_baseline_reconstruction": "passed",
            "embodiment_mapping": PANDA_GRIP_SITE_COMPILER_IDENTITY["id"],
            "workspace": candidate["checks"]["workspace"],
            "aperture": candidate["checks"]["aperture"],
            "reachability": "delegated_to_move_free_space",
        },
    }


def prepare_compiled_grasp_candidates(
    candidates: list[dict[str, Any]], *, compile_for_execution: bool,
    compile_candidate: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
    check_geometry: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop invalid candidates once, then bind raw and compiled rank order.

    ``compile_candidate`` and ``check_geometry`` default to the Panda
    grip-site compiler; registered non-Panda embodiment compilers inject
    their own identity-bound callables and reuse this admission and
    re-ranking contract without duplicating it.
    """

    compile_one = compile_candidate or compile_grasp_candidate
    check_one = check_geometry or compiled_grasp_geometry
    accepted: list[tuple[dict[str, Any], dict[str, Any] | None]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        raw = dict(candidate)
        try:
            compiled = (
                compile_one(raw) if compile_for_execution else None
            )
            if compiled is None:
                check_one(raw)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({
                "candidate_id": str(raw.get("candidate_id", "unknown")),
                "reason": str(exc),
            })
            continue
        accepted.append((raw, compiled))
    raw_candidates: list[dict[str, Any]] = []
    compiled_candidates: list[dict[str, Any]] = []
    for rank, (raw, compiled) in enumerate(accepted):
        raw["rank"] = rank
        raw_candidates.append(raw)
        if compiled is not None:
            compiled["rank"] = rank
            compiled_candidates.append(compiled)
    return raw_candidates, compiled_candidates, rejected


__all__ = [
    "PANDA_GRIP_SITE_COMPILER_IDENTITY",
    "compile_grasp_candidate",
    "compiled_grasp_geometry",
    "prepare_compiled_grasp_candidates",
]
