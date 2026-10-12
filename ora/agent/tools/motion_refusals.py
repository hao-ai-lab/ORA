"""Motion refusals in the public argument names, with the bounds they enforce.

Three defects of the pose tools' refusals (EXP11B census 1-3):

* ``move_pose`` attached ``constraints.examples`` in the nested spelling
  ``{"position": {...}, "orientation": {...}}``. Every surface that publishes
  the flat ``position_mode``/``orientation_mode`` targets refuses that shape, so
  each refusal taught a request its own schema rejects (17 EXP11A refusals).
  ``move_pose_examples`` writes them in the bound schema's spelling, through
  ``motion_arguments.arm_target_fields``, and keeps only examples that schema
  admits.
* The pose compiler (``contracts/motion.py``, a RoboDojo worker-kernel file)
  names ``move_pose`` for ``move_to``/``rotate_*`` and internal fields
  (``angle_deg``, ``target_orientation_xyzw``), and states no bounds. The
  public bounds are checked here first, under the public field names, and a
  compiler message is re-raised under the primitive that was called.
* The workspace refusal named ``xyz_m``, which is not an argument of the
  sparse pose tools; ``target_field`` names the argument the target came from.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from ora.agent.composition.schema_validation import (
    ToolSurfaceError, validate_schema_instance,
)

from . import motion_arguments
from .robot import ToolError

#: ``contracts/motion.py``: ``angle_deg must be finite and within [-180, 180]``.
ANGLE_BOUND_DEG = 180.0
#: ``contracts/pose.py`` ``canonicalize_quaternion``: a norm in [0.98, 1.02].
QUATERNION_NORM_BOUNDS = (0.98, 1.02)


def _finite_number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(float(value)))


def check_rotation_angle(value: Any, field: str) -> None:
    """A signed angle in degrees inside the compiler's bound."""

    if not _finite_number(value) or abs(float(value)) > ANGLE_BOUND_DEG:
        raise ToolError(
            f"{field} must be a finite angle in degrees within "
            f"[-{ANGLE_BOUND_DEG:g}, {ANGLE_BOUND_DEG:g}]",
            code="invalid_motion_arguments", field=field, received=value,
            constraints={"minimum": -ANGLE_BOUND_DEG, "maximum": ANGLE_BOUND_DEG,
                         "unit": "degrees"},
        )


def check_quaternion(value: Any, field: str) -> None:
    """Four finite numbers whose norm the compiler accepts."""

    low, high = QUATERNION_NORM_BOUNDS
    norm = (math.sqrt(sum(float(item) ** 2 for item in value))
            if isinstance(value, list) and len(value) == 4 and all(map(_finite_number, value))
            else None)
    if norm is None or not low <= norm <= high:
        raise ToolError(
            f"{field} must be a unit quaternion [x,y,z,w] with norm in [{low}, {high}]",
            code="invalid_motion_arguments", field=field, received=value,
            constraints={"length": 4, "norm_minimum": low, "norm_maximum": high,
                         "received_norm": norm},
        )


def check_flat_request(request: Mapping[str, Any], prefix: str) -> None:
    """The public bounds of one flat arm target, read from its compiled choice."""

    orientation = request.get("orientation", {})
    if orientation.get("mode") == "axis_angle_delta":
        check_rotation_angle(orientation.get("angle_deg"), f"{prefix}.rotation_angle_deg")
    elif orientation.get("mode") == "absolute_quaternion":
        check_quaternion(orientation.get("quaternion_xyzw"), f"{prefix}.orientation_xyzw")


def target_field(surface: Any, arm: str, name: str | None = None) -> str | None:
    """The public path of one arm target (or one of its fields) on this surface."""

    if surface.supports("single_arm_flat_targets_v1"):
        return name
    return f"targets.{arm}" + (f".{name}" if name else "")


def primitive_field(surface: Any, arm: str, primitive: str) -> str | None:
    """The public path of one ``move_to``/``rotate_*`` arm target.

    A ``rotate_*`` target is the signed angle itself (``angle_deg`` on a
    single-arm flat surface); a ``move_to`` target is an object.
    """

    if primitive == "move_to" or not surface.supports("single_arm_flat_targets_v1"):
        return target_field(surface, arm)
    return "angle_deg"


def primitive_refusal(error: Exception, *, primitive: str, field: str | None) -> ToolError:
    """A compiler or request error re-raised under the primitive that was called."""

    message = str(error).replace("move_pose", primitive)
    constraints = dict(getattr(error, "constraints", None) or {})
    return ToolError(message, code="invalid_motion_arguments", field=field,
                     constraints=constraints)


class PrimitiveAngleError(ValueError):
    """A ``rotate_*`` angle outside the compiler's bound, with that bound."""

    constraints = {"minimum": -ANGLE_BOUND_DEG, "maximum": ANGLE_BOUND_DEG, "unit": "degrees"}


def _flat_target(schema: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    if "position" in target:
        return motion_arguments.arm_target_fields(
            schema, position=target["position"], orientation=target["orientation"],
            interaction=target.get("interaction"),
        )
    return motion_arguments.arm_target_fields(
        schema, position={"mode": "absolute", "position_m": target["position_m"]},
        orientation={"mode": "absolute_quaternion", "quaternion_xyzw": target["orientation_xyzw"]},
    )


def move_pose_examples(surface: Any, examples: Any) -> list[dict[str, Any]]:
    """``examples`` in the spelling ``surface``'s own ``move_pose`` schema admits."""

    schema = motion_arguments.surface_move_pose_schema(surface)
    flat = motion_arguments.declares_flat_targets(schema)
    single_arm = surface.supports("single_arm_flat_targets_v1")
    written = []
    for example in examples:
        if "plan_ref" in example:
            if surface.supports("direct_pose_submission_v1"):
                continue
            candidate = copy.deepcopy(dict(example))
        else:
            ((arm, target),) = example["targets"].items()
            target = _flat_target(schema, target) if flat else copy.deepcopy(dict(target))
            candidate = target if single_arm else {"targets": {arm: target}}
        admitted = _admitted(candidate, schema, single_arm)
        if admitted is not None:
            written.append(admitted)
    return written


def _admitted(candidate: dict[str, Any], schema: Mapping[str, Any],
              single_arm: bool) -> dict[str, Any] | None:
    """``candidate``, or it without ``interaction`` on a schema that has none, or ``None``."""

    if not schema:
        return candidate
    try:
        validate_schema_instance(candidate, schema, root_path="$input")
        return candidate
    except ToolSurfaceError:
        pass
    target = candidate if single_arm else next(iter(candidate.get("targets", {}).values()), {})
    if "interaction" not in target:
        return None
    target.pop("interaction")
    return _admitted(candidate, schema, single_arm)


__all__ = [
    "ANGLE_BOUND_DEG", "PrimitiveAngleError", "QUATERNION_NORM_BOUNDS",
    "check_flat_request", "check_quaternion", "check_rotation_angle",
    "move_pose_examples", "primitive_field", "primitive_refusal", "target_field",
]
