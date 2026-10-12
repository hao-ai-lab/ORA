"""Finite pose validation and bounded lifted targets for closed skills.

A candidate whose committed geometry is unusable is refused as that candidate
(``field=candidate_id``): another candidate can still succeed, so the refusal
is retryable and only a changed ``candidate_id`` can (EXP11B census D7).
"""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
from .robot import ToolError


def _unusable(message: str, candidate_id: Any, **constraints: Any) -> ToolError:
    return ToolError(message, code="candidate_geometry_unavailable", retryable=True,
                     field="candidate_id", received=candidate_id,
                     retry_requires_changed_arguments=True, constraints=constraints or None)


def lift_outside_workspace(contact: Mapping[str, Any], height_m: float,
                           bounds: Mapping[str, Any]) -> dict[str, Any] | None:
    """The bound a lift of ``height_m`` above ``contact`` crosses, or ``None``."""
    z = float(contact["position_m"][2]) + float(height_m)
    maximum = float(bounds["z"][1])
    return None if z <= maximum else {"lift_target_z_m": z, "workspace_z_max_m": maximum}


def split_liftable(
    candidates: list[dict[str, Any]], compiled: list[dict[str, Any]], height_m: float,
    bounds: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Move every candidate whose lift target leaves the workspace to the compiler ledger.

    ``grasp_execute`` lifts ``height_m`` above the contact and refused such a
    candidate only at commit; ``grasp_plan`` now rejects it where it publishes
    rejections, in the ``{candidate_id, reason}`` shape that ledger already has.
    """
    rejected, kept = [], set()
    for item in compiled:
        contact = item.get("contact_pose")
        crossed = (lift_outside_workspace(contact, height_m, bounds)
                   if isinstance(contact, Mapping) and isinstance(contact.get("position_m"), Sequence)
                   else None)
        if crossed is None:
            kept.add(str(item.get("candidate_id")))
            continue
        rejected.append({"candidate_id": str(item.get("candidate_id")), "reason": (
            f"lift target z {crossed['lift_target_z_m']:.4f} m exceeds the workspace z maximum "
            f"{crossed['workspace_z_max_m']:.4f} m")})
    return ([item for item in candidates if str(item.get("candidate_id")) in kept],
            [item for item in compiled if str(item.get("candidate_id")) in kept], rejected)


def _pose(value: Any, what: str, candidate_id: Any = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _unusable(f"{what} is unavailable on the selected candidate", candidate_id)
    position = value.get("position_m")
    orientation = value.get("orientation_xyzw")
    if not (
        isinstance(position, Sequence) and len(position) == 3
        and isinstance(orientation, Sequence) and len(orientation) == 4
        and all(
            not isinstance(item, bool) and isinstance(item, (int, float))
            and math.isfinite(float(item))
            for item in [*position, *orientation]
        )
    ):
        raise _unusable(f"{what} is not a finite pose", candidate_id)
    return {
        "position_m": [float(item) for item in position],
        "orientation_xyzw": [float(item) for item in orientation],
    }


def _lifted(pose: Mapping[str, Any], height_m: float) -> dict[str, Any]:
    """The same orientation, raised along world +Z.

    Transport and retreat differ from the place pose only in standoff, and the
    standoff is the surface's own pre-contact distance rather than a second
    number invented here.
    """

    position = list(pose["position_m"])
    return {
        "position_m": [position[0], position[1], position[2] + float(height_m)],
        "orientation_xyzw": list(pose["orientation_xyzw"]),
    }


def validate_prepared_targets(poses: Mapping[str, Mapping[str, Any]], bounds: Mapping[str, Any],
                              candidate_id: Any = None) -> None:
    """Reject unusable committed geometry before the opening can release anything."""
    for stage, pose in poses.items():
        norm = math.sqrt(sum(v * v for v in pose["orientation_xyzw"]))
        if abs(norm - 1.0) > 1e-3:
            raise _unusable(f"{stage} orientation is not a unit quaternion", candidate_id,
                            stage=stage, quaternion_norm=norm)
        if any(not bounds[axis][0] <= value <= bounds[axis][1]
               for axis, value in zip("xyz", pose["position_m"])):
            raise _unusable(f"{stage} target is outside the public workspace", candidate_id,
                            stage=stage, position_m=list(pose["position_m"]),
                            workspace_bounds_m={axis: list(bounds[axis]) for axis in "xyz"})
    pregrasp, contact = poses["pregrasp"], poses["approach"]
    dot = abs(sum(a * b for a, b in zip(pregrasp["orientation_xyzw"], contact["orientation_xyzw"])))
    distance = math.dist(pregrasp["position_m"], contact["position_m"])
    if dot < 1.0 - 1e-5 or distance <= 1e-6:
        raise _unusable("pregrasp must supply a nonzero approach at the contact orientation",
                        candidate_id)
