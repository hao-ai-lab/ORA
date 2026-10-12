"""Placement refusals that carry the measured quantity beside their bound.

EXP11A g003 called ``place_plan`` five times on RoboDojo and never received a
candidate. None of the five was a camera-geometry defect -- the table plane in
the same head view measured 2.5 degrees of tilt and 3.6 mm residual. They were
selection outcomes the policy could only correct if it saw why:

- a basket's front handle slot selected as support fitted at 65.1 degrees tilt
  against the 15 degree bound, but the ``support`` and ``region`` checks said
  only ``support_plane_unresolved`` with no measurement;
- a bowl rim selected as boundary had too little plane coverage, reported only
  as ``opening_plane_unresolved``;
- a held object selected from its own wrist camera retained 117,088 depth
  points, and the refusal said ``requires 16..100000 retained depth points``
  without the number, the region or the camera;
- one call named the same region as support and boundary and was refused
  without naming the field.

The two structured errors below name the measured quantity, because an error
that omits it is the defect. (The opt-in plane-fit measurements attached to
unresolved checks, ``placement_refusal_measurements_v1``, are declared by no
kept surface and left this tree.)
"""

from __future__ import annotations

from typing import Any, Mapping

from .robot import ToolError

ROLE_FIELDS = {"object": "object_ref", "support": "destination_support_ref",
               "boundary": "destination_boundary_ref"}


class RetainedPointBoundsError(ValueError):
    """A selected region retained a depth-point count outside the projection bounds."""

    def __init__(self, retained: int, *, minimum: int, maximum: int, measurements: Mapping[str, Any]) -> None:
        super().__init__(
            f"placement projection retained {retained} depth points; it requires {minimum}..{maximum}")
        self.retained, self.minimum, self.maximum = int(retained), int(minimum), int(maximum)
        self.measurements = dict(measurements)


def retained_point_error(error: RetainedPointBoundsError, *, role: str, camera: str) -> ToolError:
    too_many = error.retained > error.maximum
    return ToolError(
        f"{error} for the {role} region from camera {camera}"
        + ("; a region that fills a close wrist view is usually better selected from the head camera"
           if too_many else "; select a region with more visible depth"),
        code="placement_evidence_unavailable", retryable=True, field=ROLE_FIELDS.get(role, role),
        received=error.retained,
        constraints={"minimum_retained_points": error.minimum, "maximum_retained_points": error.maximum,
                     "role": role, "source_camera": camera,
                     "mask_pixel_count": error.measurements.get("mask_pixel_count"),
                     "invalid_depth_pixel_count": error.measurements.get("invalid_depth_pixel_count")},
    )


def duplicate_reference_error(specs: list[tuple[str, str]]) -> ToolError | None:
    """Name the first field whose region reference an earlier role already uses."""
    seen: dict[str, str] = {}
    for role, ref in specs:
        if ref in seen:
            return ToolError(
                "object, support and boundary need distinct selected region references",
                code="invalid_arguments", field=ROLE_FIELDS.get(role, role), received=ref,
                constraints={"same_reference_as": ROLE_FIELDS.get(seen[ref], seen[ref])})
        seen[ref] = role
    return None


__all__ = ["ROLE_FIELDS", "RetainedPointBoundsError", "duplicate_reference_error", "retained_point_error"]
