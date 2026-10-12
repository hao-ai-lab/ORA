"""Runtime half of ``reach_nearest_orientation_v1`` (see ``composition/reach_orientation_schema.py``).

For an unsolved target the requested orientation is turned about axes
perpendicular to its approach axis, holding the nominal jaw centre (the
controlled point plus the descriptor's jaw offset) where the request put it,
and each rotation's candidates go to the read-only reach check in one request
on the target's own route. The first rotation with a reachable candidate ends
the sweep; among its reachable candidates the one with the larger joint-limit
margin, then the smaller joint move, is reported. Nothing here changes the
submitted action, and any failure reports the reference unavailable.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.agent.composition.reach_orientation_schema import (
    CAPABILITY, CLAIM, ROTATIONS_DEG, TURN_DIRECTIONS,
)
from ora.contracts.pose import matrix_quaternion, quaternion_matrix

from .reach_check import check_queries
from .robot import ToolError

Bridge = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def jaw_geometry(surface: Any, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """The controlled point's approach axis and jaw-centre offset, in its own frame."""
    from ora.agent.context.contact_point_projection import jaw_contact_center_offsets

    from .alignment import frame_local_axes

    view = surface.view_config or {}
    approach = np.asarray(frame_local_axes(str(view.get("controlled_frame_convention")))["approach"], float)
    offset = np.asarray(jaw_contact_center_offsets(view)[arm], float)
    return approach / np.linalg.norm(approach), offset


def _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * skew @ skew


def turn_axes(approach_world: np.ndarray, directions: int = TURN_DIRECTIONS) -> list[np.ndarray]:
    """Evenly spaced unit axes perpendicular to the approach, the first horizontal."""
    first = np.cross([0.0, 0.0, 1.0], approach_world)
    if np.linalg.norm(first) < 1e-9:
        first = np.array([1.0, 0.0, 0.0])
    first = first / np.linalg.norm(first)
    second = np.cross(approach_world, first)
    angles = [2.0 * math.pi * index / directions for index in range(directions)]
    return [math.cos(angle) * first + math.sin(angle) * second for angle in angles]


def turned_candidates(
    target_pose: Sequence[float], *, approach_local: np.ndarray, offset_local: np.ndarray, rotation_deg: float,
    directions: int = TURN_DIRECTIONS,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Each turn of the request by ``rotation_deg`` with the jaw centre held, and that centre."""
    rotation = np.asarray(quaternion_matrix(list(target_pose[3:])), float)
    center = np.asarray(target_pose[:3], float) + rotation @ offset_local
    candidates = []
    for axis in turn_axes(rotation @ approach_local, directions):
        turned = _axis_rotation(axis, math.radians(rotation_deg)) @ rotation
        position = center - turned @ offset_local
        candidates.append({"pose_xyzw": [*map(float, position), *matrix_quaternion(turned.tolist())],
                           "approach_world": turned @ approach_local})
    return candidates, center


def _preference(row: Mapping[str, Any]) -> tuple[float, float]:
    margin, delta = row.get("joint_limit_margin_rad"), row.get("joint_delta_rad")
    return (-1.0 if margin is None else float(margin), -math.inf if delta is None else -float(delta))


def _found(angle: float, candidate: Mapping[str, Any], *, reachable: int, center: np.ndarray,
           queries: int, claim: str = CLAIM) -> dict[str, Any]:
    approach = candidate["approach_world"]
    pose = candidate["pose_xyzw"]
    return {
        "status": "found", "rotation_from_request_deg": angle,
        "tilt_from_straight_down_deg": round(math.degrees(math.acos(max(-1.0, min(1.0, -approach[2])))), 1),
        "approach_world": [round(float(item), 4) for item in approach],
        "pose": {"position_m": [round(item, 5) for item in pose[:3]],
                 "orientation_xyzw": [round(item, 5) for item in pose[3:]]},
        "jaw_center_m": [round(float(item), 5) for item in center],
        "reachable_turns_at_rotation": reachable, "queries": queries, "claim_boundary": claim,
    }


def nearest_orientation(
    arm: str, *, target_pose: Sequence[float], interaction: str, approach_local: np.ndarray,
    offset_local: np.ndarray, bridge: Bridge | None, workspace_bounds_m: Mapping[str, Any],
    directions: int = TURN_DIRECTIONS, claim: str = CLAIM,
) -> dict[str, Any]:
    """The smallest reachable turn of an unsolved target, or that the sweep found none.

    ``claim`` is the published claim boundary; the v2 sweep
    (``reach_reference.nearest``) names the reach check it shares with
    ``reach_along_request`` instead of v1's route claim.
    """
    queries, center = 0, None
    for angle in ROTATIONS_DEG:
        candidates, center = turned_candidates(target_pose, approach_local=approach_local, offset_local=offset_local,
                                               rotation_deg=angle, directions=directions)
        rows = check_queries(arm, [{"pose_xyzw": item["pose_xyzw"], "interaction": interaction, "from_index": None}
                                   for item in candidates],
                             bridge=bridge, workspace_bounds_m=workspace_bounds_m)["results"]
        queries += len(candidates)
        reachable = [(row, item) for row, item in zip(rows, candidates) if row["reachable"]]
        if reachable:
            best = max(reachable, key=lambda pair: _preference(pair[0]))[1]
            return _found(angle, best, reachable=len(reachable), center=center, queries=queries, claim=claim)
    return {"status": "none_within_sweep", "largest_rotation_deg": ROTATIONS_DEG[-1],
            "jaw_center_m": [round(float(item), 5) for item in center], "queries": queries,
            "claim_boundary": claim}


def reference(host: Any, arm: str, *, target_pose: Sequence[float], interaction: str) -> dict[str, Any]:
    """``nearest_reachable_orientation`` for one unsolved target on a declaring surface, else nothing."""
    if not host.surface.supports(CAPABILITY):
        return {}
    try:
        approach, offset = jaw_geometry(host.surface, arm)
        nearest = nearest_orientation(
            arm, target_pose=[float(item) for item in target_pose], interaction=interaction, approach_local=approach,
            offset_local=offset, bridge=getattr(host, "align_plan_callback", None),
            workspace_bounds_m=host.workspace_bounds_m)
    except Exception as error:  # an IK reference never blocks a move
        nearest = {"status": "unavailable", "failure": error.code if isinstance(error, ToolError)
                   else type(error).__name__}
    return {"nearest_reachable_orientation": nearest}


__all__ = ["jaw_geometry", "nearest_orientation", "reference", "turn_axes", "turned_candidates"]
