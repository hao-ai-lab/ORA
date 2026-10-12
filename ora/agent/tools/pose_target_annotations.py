"""The read-only annotations a compiled pose target carries after its IK reference.

``move_pose`` and the semantic pose primitives that compile through it
(``move_to``, ``rotate_wrist``, ``rotate_pitch``) call :func:`merged` once per
arm with the compiled target actually submitted (after any jaw turn) and the
planning-time reach reference (``move_pose_reach.apply_reference`` or
``primitive_reference``). Each declared capability contributes its own key, in
this fixed order:

1. the reach reference as given (``jaw_symmetry``, ``reach_along_request`` and,
   under ``reach_nearest_orientation_v1`` or ``_v2``, its unsolved form's
   ``nearest_reachable_orientation``);
2. ``target_axes_world`` -- ``pose_orientation_from_world_axes_v1``
   (``world_axes.echo``);
3. ``observed_clearance`` -- ``observed_clearance_reference_v2``
   (``observed_clearance_runtime.reference``).

A key that a later contribution would overwrite is a programming error, never a
silent replacement; the composition binders refuse the same overlap in the
result schemas. A surface that declares none of them gets the reference alone,
and no annotation blocks or changes the move. Keeping this here leaves the
Tool Runtime (``agent/tools/runtime``) one forwarder per pose tool.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import observed_clearance_runtime, world_axes


def merged(
    host: Any, arm: str, compiled: Mapping[str, Any], reference: Mapping[str, Any],
    observation: Mapping[str, Any], root: Path,
) -> dict[str, Any]:
    """The reach reference plus every declared pose-target annotation for one arm."""
    target = compiled["target_pose_xyz_xyzw"]
    pose = {"position_m": list(target[:3]), "orientation_xyzw": list(target[3:])}
    result = dict(reference)
    for part in (
        world_axes.echo(host.surface, pose),
        observed_clearance_runtime.reference(host, arm, target, observation=observation, root=root),
    ):
        overlap = sorted(set(result) & set(part))
        if overlap:
            raise RuntimeError(f"pose-target annotations would overwrite {overlap}")
        result.update(part)
    return result


__all__ = ["merged"]
