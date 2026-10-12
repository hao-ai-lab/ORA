"""Policy-facing alignment planning: the protocol edge of a read-only tool.

The computation is the environment's (`environment/controllers/reachability.py`
and `controllers/alignment.py`). This file owns only what the policy sees: the
argument contract, the approach-pose synthesis the policy should not have to do
by hand, and the translation of a backend refusal into a tool error that names
the field.

Kept out of the tool runtime (`runtime/core.py`) on purpose. That file was
once 5,686 lines, and every tool that grew there did so twenty lines at a time.
The repository's own split -- `retention.py`, `grasp.py`, `placement.py`,
`compute.py` -- puts the protocol boundary in the runtime and the work in a
module of its own.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

from ora.contracts.refusal import EnvironmentRefusal
from .robot import ToolError
from ora.contracts.alignment import (
    APPROACH_SIDES, MAXIMUM_POSES, REQUEST_SCHEMA,
)

ARGUMENT_FIELDS = frozenset({"target_ref", "approach_side", "arm"})
OPTIONAL_FIELDS = frozenset({"standoff_m", "max_candidates"})

DEFAULT_STANDOFF_M = 0.10
MINIMUM_STANDOFF_M = 0.02
MAXIMUM_STANDOFF_M = 0.30
DEFAULT_CANDIDATES = 3

# World-frame unit vector pointing *from* the approach position *toward* the
# target, per side. Named directions toward the object, never an arm.
_APPROACH_DIRECTIONS = {
    "top": (0.0, 0.0, -1.0),
    "front": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, -1.0, 0.0),
    "right": (0.0, 1.0, 0.0),
}


#: There is deliberately no default controlled-frame convention. The two
#: maintained embodiments genuinely disagree: aloha-agilex approaches along
#: local +X and separates along +Y, franka-panda approaches along +Z and
#: separates along +X. Synthesising against the wrong one points the
#: jaw-separation line at the object and rolls every candidate about the closing
#: axis -- a well-formed answer to a question nobody asked, which no error
#: message would ever report. A default was correct only while this tool was
#: advertised on RoboTwin alone, which made it the thing standing between the
#: tool and every other embodiment. Callers name the convention their surface
#: declares in `view_config.controlled_frame_convention`; axis values are
#: resolved through the one reader that owns them, never copied.


def frame_local_axes(convention: str) -> dict[str, list[float]]:
    """The declared approach and closing axes of one controlled-frame convention."""

    from ora.agent.composition.embodiment_geometry import public_local_axes

    return public_local_axes(convention)


def _quaternion_from_approach(
    direction: Sequence[float], roll_rad: float,
    *, local_axes: Mapping[str, Sequence[float]],
) -> list[float]:
    """A gripper orientation whose *declared* approach axis points along `direction`.

    The jaw line is rolled about that axis so several candidates differ only in
    how the fingers straddle the object.

    The rotation is built from the embodiment's own axis declaration rather than
    from a fixed column order. Writing the approach into column 0 assumes local
    +X is the approach axis, which is true of aloha-agilex and false of
    franka-panda; the columns are placed where the descriptor says they belong.
    """

    import numpy as np

    forward = np.asarray(direction, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    reference = np.array([0.0, 0.0, 1.0]) if abs(forward[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(reference, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    cos_roll, sin_roll = math.cos(roll_rad), math.sin(roll_rad)
    rolled_right = right * cos_roll + up * sin_roll
    rolled_up = up * cos_roll - right * sin_roll
    # World columns in (approach, closing, third) order, and the local frame in
    # the same order. R = world @ local.T then maps each declared local axis to
    # its world counterpart, whichever columns the descriptor put them in.
    approach_local = np.asarray(local_axes["approach"], dtype=np.float64)
    closing_local = np.asarray(local_axes["closing"], dtype=np.float64)
    approach_local /= np.linalg.norm(approach_local)
    closing_local /= np.linalg.norm(closing_local)
    world = np.stack([forward, rolled_right, rolled_up], axis=1)
    local = np.stack(
        [approach_local, closing_local, np.cross(approach_local, closing_local)], axis=1
    )
    matrix = world @ local.T
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return [
            float((matrix[2, 1] - matrix[1, 2]) / scale),
            float((matrix[0, 2] - matrix[2, 0]) / scale),
            float((matrix[1, 0] - matrix[0, 1]) / scale),
            float(0.25 * scale),
        ]
    index = int(np.argmax(np.diag(matrix)))
    i, j, k = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][index]
    scale = math.sqrt(1.0 + matrix[i, i] - matrix[j, j] - matrix[k, k]) * 2.0
    parts = [0.0, 0.0, 0.0]
    parts[i] = 0.25 * scale
    parts[j] = float((matrix[j, i] + matrix[i, j]) / scale)
    parts[k] = float((matrix[k, i] + matrix[i, k]) / scale)
    return [*parts, float((matrix[k, j] - matrix[j, k]) / scale)]


def synthesise_approach_poses(
    centre_xyz_m: Sequence[float], *, approach_side: str, standoff_m: float,
    candidate_count: int, local_axes: Mapping[str, Sequence[float]],
) -> list[list[float]]:
    """Poses standing off the target along one side, differing only in jaw roll.

    The policy should not have to compose a quaternion by hand to ask whether an
    approach is reachable; that is exactly the arithmetic the tool surface exists
    to remove.
    """

    direction = _APPROACH_DIRECTIONS[approach_side]
    centre = [float(value) for value in centre_xyz_m]
    position = [centre[axis] - direction[axis] * standoff_m for axis in range(3)]
    poses: list[list[float]] = []
    for index in range(candidate_count):
        roll = math.pi * index / max(candidate_count, 1)
        poses.append([
            *position, *_quaternion_from_approach(direction, roll, local_axes=local_axes),
        ])
    return poses


def validate_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a malformed request by naming the field, never by a bare message."""

    unknown = set(arguments) - ARGUMENT_FIELDS - OPTIONAL_FIELDS
    missing = ARGUMENT_FIELDS - set(arguments)
    if unknown or missing:
        raise EnvironmentRefusal(
            "align_plan requires target_ref, approach_side and arm",
            field=sorted(unknown or missing)[0],
            received=sorted(arguments),
            allowed=sorted(ARGUMENT_FIELDS | OPTIONAL_FIELDS),
        )
    if arguments["approach_side"] not in APPROACH_SIDES:
        raise EnvironmentRefusal(
            "approach side is not a declared direction", field="approach_side",
            received=arguments["approach_side"], allowed=list(APPROACH_SIDES),
        )
    standoff = arguments.get("standoff_m", DEFAULT_STANDOFF_M)
    if not isinstance(standoff, (int, float)) or isinstance(standoff, bool) or not (
        MINIMUM_STANDOFF_M <= float(standoff) <= MAXIMUM_STANDOFF_M
    ):
        raise EnvironmentRefusal(
            "standoff is outside the declared range", field="standoff_m",
            received=standoff,
            allowed=[MINIMUM_STANDOFF_M, MAXIMUM_STANDOFF_M],
        )
    count = arguments.get("max_candidates", DEFAULT_CANDIDATES)
    if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 6:
        raise EnvironmentRefusal(
            "max_candidates is outside the declared range", field="max_candidates",
            received=count, allowed=[1, 6],
        )
    return {
        "target_ref": arguments["target_ref"],
        "approach_side": arguments["approach_side"],
        "arm": arguments["arm"],
        "standoff_m": float(standoff),
        "max_candidates": int(count),
    }


def build_request(
    validated: Mapping[str, Any], *, centre_xyz_m: Sequence[float],
    local_axes: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """The closed backend request; its field set is checked on arrival."""

    poses = synthesise_approach_poses(
        centre_xyz_m,
        approach_side=validated["approach_side"],
        standoff_m=validated["standoff_m"],
        candidate_count=min(validated["max_candidates"], MAXIMUM_POSES),
        local_axes=local_axes,
    )
    return {
        "schema": REQUEST_SCHEMA,
        "target_ref": validated["target_ref"],
        "approach_side": validated["approach_side"],
        "arm": validated["arm"],
        "target_poses_xyzw": poses,
    }


def plan(
    arguments: Mapping[str, Any],
    *,
    bridge: Callable[[Mapping[str, Any]], dict[str, Any]] | None,
    centroid_of: Callable[[str], Sequence[float]],
    frame_convention: str,
) -> dict[str, Any]:
    """Validate, synthesise, call the read-only bridge, and shape every error.

    Takes callables rather than the tool runtime, for the same reason
    ``solve_reachability`` does: a module that cannot reach a session cannot
    accidentally acquire a side effect. ``bridge`` is the read-only environment
    callback and ``centroid_of`` resolves a target reference to its measured
    world centroid.

    A refusal from either side becomes a tool error naming the field, so the
    policy is told which argument to change rather than being handed an
    internal fault it cannot act on.
    """

    if bridge is None:
        raise ToolError(
            "align_plan requires the environment read-only bridge",
            code="tool_unavailable", retryable=False,
        )
    try:
        validated = validate_arguments(arguments)
        centre = centroid_of(validated["target_ref"])
        return bridge(build_request(
            validated, centre_xyz_m=centre,
            local_axes=frame_local_axes(frame_convention),
        ))
    except EnvironmentRefusal as error:
        raise ToolError(
            str(error), code=error.code, retryable=error.retryable,
            field=error.field, received=error.received,
            constraints={"allowed": error.allowed},
        ) from error
