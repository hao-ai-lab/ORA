"""Runtime half of the v2 reach pre-check (see ``composition/reach_reference_schema.py``).

``move_pose_reach`` asks the internal read-only reach check about each
submitted target, as under the v1 capabilities; on a surface declaring a v2
generation this module shapes what the policy is told. Every annotation names
the IK that answered (``ik_model``), whether that IK is ``move_pose``'s own
admission (``matches_move_pose_admission``, from the backend), the measured
disagreement when it is not, and the backend's claim boundary. On RoboTwin
that is one damped-least-squares descent with joint limits only, not the
128-seed collision-checked CuRobo IK ``move_pose`` admits through, so a solved
target is an approximate pre-check and an unsolved one is not proof of
anything. The submitted action never changes and a reference that cannot be
taken never blocks the move.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from ora.agent.composition.reach_reference_schema import (
    MEASURED_DISAGREEMENT, MOVE_POSE_CAPABILITY, NEAREST_CAPABILITY, NEAREST_CLAIM, PRIMITIVE_CAPABILITY,
)

from . import jaw_center, reach_orientation
from .robot import ToolError

#: The sampling half of an unsolved target's claim; the backend's own claim
#: boundary follows it.
SAMPLING_CLAIM = (
    "Read-only IK at sampled points of the straight interpolation from the measured pose to the target; "
    "reachability is not proven monotone between samples, and no path, collision, payload or arrival claim."
)


def move_pose_declared(surface: Any) -> bool:
    return bool(surface.supports(MOVE_POSE_CAPABILITY))


def primitive_declared(surface: Any) -> bool:
    return bool(surface.supports(PRIMITIVE_CAPABILITY))


def basis(answer: Mapping[str, Any]) -> dict[str, Any]:
    """Which IK answered, whether it is move_pose's admission, and its measured error when not."""
    model = answer.get("ik_model")
    matches = answer.get("matches_move_pose_admission")
    measured = MEASURED_DISAGREEMENT.get(str(model)) if matches is False else None
    return {"ik_model": model, "matches_move_pose_admission": matches,
            "measured_disagreement": None if measured is None else copy.deepcopy(measured)}


def solved(answer: Mapping[str, Any]) -> dict[str, Any]:
    """A target the reach check solved: still only that check's answer."""
    return {"target_reachable": True, **basis(answer), "claim_boundary": str(answer["claim_boundary"])}


def unsolved(host: Any, arm: str, reach: Mapping[str, Any], *, answer: Mapping[str, Any],
             target_pose: Sequence[float], interaction: str) -> dict[str, Any]:
    """The v1 reach-along fields of an unsolved target, re-claimed, plus the basis and nearest turn."""
    return {**reach, "claim_boundary": f"{SAMPLING_CLAIM} {answer['claim_boundary']}", **basis(answer),
            **nearest(host, arm, target_pose=target_pose, interaction=interaction)}


def nearest(host: Any, arm: str, *, target_pose: Sequence[float], interaction: str) -> dict[str, Any]:
    """``nearest_reachable_orientation`` on a surface declaring the v2 sweep, else nothing."""
    if not host.surface.supports(NEAREST_CAPABILITY):
        return {}
    try:
        approach, offset = jaw_center.approach_and_offset(host.surface, arm)
        value = reach_orientation.nearest_orientation(
            arm, target_pose=[float(item) for item in target_pose], interaction=interaction,
            approach_local=approach, offset_local=offset, bridge=getattr(host, "align_plan_callback", None),
            workspace_bounds_m=host.workspace_bounds_m, claim=NEAREST_CLAIM)
    except Exception as error:  # an IK reference never blocks a move
        value = {"status": "unavailable",
                 "failure": error.code if isinstance(error, ToolError) else type(error).__name__}
    return {"nearest_reachable_orientation": value}


__all__ = ["SAMPLING_CLAIM", "basis", "move_pose_declared", "nearest", "primitive_declared", "solved", "unsolved"]
