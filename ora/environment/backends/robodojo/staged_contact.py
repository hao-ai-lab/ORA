"""Staged contact approach on the RoboDojo sparse v3 route.

A ``contact_approach`` larger than the local envelope (50 mm / 15 degrees) is
refused on the released Dojo route. With ``runtime.contact_approach_route``
set to ``global_precontact_then_local`` -- which the RunSpec derives from a
surface declaring ``robodojo_staged_contact_approach_v1`` -- it is staged the
way RoboTwin and LIBERO stage it: one global SDK-IK plus joint-interpolation
move to a pre-contact pose, then the bounded local step to the target.

The global segment stops at measured arrival within the ARX endpoint
tolerance, so the pre-contact pose leaves the local step exactly that much
headroom: wherever the global segment lands inside its tolerance, the rest
still fits one local step. Each arm's receipt discloses ``precontact`` (the
pose, whether the local contact step was attempted, and why not); a stop in
the pre-contact stage reports that stage, never an arrival at the target.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.contracts.action import SERVO_MAX_ROTATION_RAD, SERVO_MAX_TRANSLATION_M
from ora.environment.controllers.motion_transaction import _precontact_pose

from .common import ORIENTATION_TOLERANCE_RAD, POSITION_TOLERANCE_M, pose_error, pose_reached
from .sparse_motion import (
    GLOBAL_EXECUTOR, LOCAL_EXECUTOR, STAGED_CONTACT_STRATEGY, _local_step, _run_global,
    within_local_envelope,
)

STAGED_CONTACT_CAPABILITY = "robodojo_staged_contact_approach_v1"
PRECONTACT_LOCAL_TRANSLATION_M = SERVO_MAX_TRANSLATION_M - POSITION_TOLERANCE_M
PRECONTACT_LOCAL_ROTATION_RAD = SERVO_MAX_ROTATION_RAD - ORIENTATION_TOLERANCE_RAD


def precontact_poses(
    before: Mapping[str, Any], targets: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Per-arm pre-contact poses: each target less the final local step."""
    return {arm: _precontact_pose(
        [float(item) for item in before[arm]["pose_xyz_xyzw"]], list(target),
        local_translation_m=PRECONTACT_LOCAL_TRANSLATION_M, local_rotation_rad=PRECONTACT_LOCAL_ROTATION_RAD,
    ) for arm, target in targets.items()}


def handoff_refusal(
    *, terminal: bool, precontact: Mapping[str, Sequence[float]], targets: Mapping[str, Sequence[float]],
    landed: Mapping[str, Any], remaining: int,
) -> str | None:
    """Why the local contact step must not run after the pre-contact segment, or ``None``."""
    if terminal:
        return "episode_terminated_before_contact_approach"
    if not all(pose_reached(pose, landed[arm]["pose_xyz_xyzw"]) for arm, pose in precontact.items()):
        return "precontact_segment_not_reached"
    if not all(within_local_envelope(*pose_error(target, landed[arm]["pose_xyz_xyzw"]))
               for arm, target in targets.items()):
        return "residual_outside_local_servo_envelope"
    if remaining <= 0:
        return "native_action_budget_exhausted"
    return None


#: Where a refused hand-off is reported: termination and budget are execution
#: outcomes with their shared public codes; the other two stop at pre-contact.
_REFUSAL_STAGE = {
    "episode_terminated_before_contact_approach": ("execution", "episode_terminal_during_execution"),
    "native_action_budget_exhausted": ("execution", "native_action_budget_exhausted"),
}


def staged_contact_approach(
    execution: Any, targets: Mapping[str, Sequence[float]], interactions: Mapping[str, str],
) -> dict[str, Any]:
    """Global move to the pre-contact pose, then the bounded local step."""
    precontact = precontact_poses(execution.before, targets)
    # Until the hand-off is decided the reason names the stage: a stop inside
    # the pre-contact segment carries its own failure stage and code.
    disclosure = {arm: {"pose_xyz_xyzw": list(pose), "attempted": False, "reason": "precontact_segment_stopped"}
                  for arm, pose in precontact.items()}
    common = {"strategy": STAGED_CONTACT_STRATEGY, "interactions": interactions, "target_poses": targets,
              "precontact": disclosure}
    stopped = _run_global(execution, [("precontact", precontact)], common)
    if stopped is not None:
        return stopped
    landed = execution.session._proprioception()["end_effectors"]
    reason = handoff_refusal(terminal=bool(execution.session.is_terminal), precontact=precontact,
                             targets=targets, landed=landed, remaining=execution.remaining())
    if reason is not None:
        for value in disclosure.values():
            value["reason"] = reason
        stage, code = _REFUSAL_STAGE.get(reason, ("precontact", reason))
        return execution.finish(status="not_reached", stage=stage, code=code, executor=[GLOBAL_EXECUTOR],
                                **common)
    for value in disclosure.values():
        value.update(attempted=True, reason=None)
    return _local_step(execution, targets, interactions, start=landed, strategy=STAGED_CONTACT_STRATEGY,
                       executor=(GLOBAL_EXECUTOR, LOCAL_EXECUTOR), precontact=disclosure)


__all__ = [
    "PRECONTACT_LOCAL_ROTATION_RAD", "PRECONTACT_LOCAL_TRANSLATION_M", "STAGED_CONTACT_CAPABILITY",
    "handoff_refusal", "precontact_poses", "staged_contact_approach",
]
