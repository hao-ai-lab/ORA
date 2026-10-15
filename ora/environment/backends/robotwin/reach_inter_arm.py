"""Other-arm screening for RoboTwin's read-only reach check.

RoboTwin admission already screens the other arm: ``planning.validate_inter_arm_clearance``
refuses a plan whose link spheres (the control profile's ``dual_arm_collision_check``:
45 mm spheres at the pinocchio chain links, 5 mm surface clearance) come too close
to the other arm held at its measured joints, through
``controllers/inter_arm_clearance.InterArmClearancePolicy``. Unscreened, the
read-only reach check (``motion_adapter.reach_check_batch``) has no collision
model at all, so a pose it calls reachable can be one admission refuses for the
other arm, and the nearest-orientation sweep can propose such a turn.

:func:`bind_screen` builds the admission's own test for one arm's reach-check
solutions: the configuration the descent reached, alone, screened against the
other arm held at its measured joints and assessed as an endpoint with the
admission policy's planning thresholds (and its escape rule when the arms start
closer than that). A blocked solution becomes ``inter_arm_collision_endpoint``;
the screened check names its own ``ik_model`` (``contracts/reach_check.py``),
capability ``robotwin_reach_check_inter_arm_screen_v1``.

Wired per surface: the RunSpec derives ``runtime.reach_check_inter_arm_screen``
from that capability, the worker passes it to ``RoboTwinLiveSession``, and
``reach_check_batch`` binds the screen for the queried arm when the session flag
is set. The screen does not claim the configuration ``move_pose``'s 128-seed
CuRobo IK would choose.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ora.contracts.reach_check import (
    INTER_ARM_COLLISION_ENDPOINT, ROBOTWIN_INTER_ARM_IK_MODELS,
)
from ora.environment.controllers.inter_arm_clearance import (
    InterArmClearancePolicy, minimum_inter_arm_distance,
)

ENDPOINT = "endpoint"
#: The reach-check reason a blocked solution reports.
REASONS = {ENDPOINT: INTER_ARM_COLLISION_ENDPOINT}
#: The ``ik_model`` a screened reach check names: a different answer set from
#: ``robotwin_single_seed_damped_least_squares_v1``, so no measured
#: disagreement carries over.
IK_MODELS = {ENDPOINT: ROBOTWIN_INTER_ARM_IK_MODELS[ENDPOINT]}
#: The screened reach check's claim boundary (policy-visible through
#: ``reach_along_request.claim_boundary``; Chinese reference in
#: docs/reference/public-interfaces.md).
CLAIM_BOUNDARIES = {
    ENDPOINT: (
        "One damped-least-squares IK descent from the measured joints (or a chained query's earlier "
        "solution) with joint limits, then the configuration it reached screened against the other arm "
        "held at its measured joints with move_pose's link-sphere inter-arm clearance, at that "
        "configuration only and not along any path to it: no other collision model, no route model, and "
        "not the 128-seed collision-checked CuRobo IK move_pose plans through, which may reach the pose in "
        "another configuration, so move_pose can still refuse a reachable pose or reach one reported "
        "unreachable."
    ),
}
ARMS = ("left", "right")
ChainPoints = Callable[[str, Sequence[float]], Sequence[Sequence[float]]]


def screen_mode(*, screen: bool) -> str | None:
    """The mode the session flag selects: the endpoint screen, or none."""
    if not isinstance(screen, bool):
        raise ValueError("reach-check inter-arm screen flag must be Boolean")
    return ENDPOINT if screen else None


def batch_basis(mode: str) -> dict[str, Any]:
    """The batch fields a screened reach check names instead of the unscreened ones."""
    return {"ik_model": IK_MODELS[mode], "matches_move_pose_admission": False,
            "claim_boundary": CLAIM_BOUNDARIES[mode]}


def bind_screen(
    arm: str, *, measured_joints: Mapping[str, Sequence[float]], fk_chain_points: ChainPoints,
    profile: Mapping[str, Any],
) -> Callable[[Sequence[float]], dict[str, Any] | None]:
    """A screen for one arm's solutions: ``screen(end)`` is ``None`` when clear.

    ``measured_joints`` holds both arms' measured joints at the action start,
    ``fk_chain_points`` the session's pinocchio chain points and ``profile``
    the control profile's ``dual_arm_collision_check``.
    """
    if arm not in ARMS:
        raise ValueError("reach screen arm must be left or right")
    other = "right" if arm == "left" else "left"
    held = fk_chain_points(other, measured_joints[other])
    initial, _pair = minimum_inter_arm_distance(fk_chain_points("left", measured_joints["left"]),
                                                fk_chain_points("right", measured_joints["right"]))
    policy = InterArmClearancePolicy.from_profile(profile, initial)

    def screen(end: Sequence[float]) -> dict[str, Any] | None:
        moving = fk_chain_points(arm, [float(item) for item in end])
        distance, pair = minimum_inter_arm_distance(*((moving, held) if arm == "left" else (held, moving)))
        decision = policy.assess(distance, planning=True, endpoint=True)
        if not decision["allowed"]:
            return {"reason": REASONS[ENDPOINT], "stage_index": 0, "stages": 1,
                    "link_pair": pair, **{key: value for key, value in decision.items() if key != "allowed"}}
        return None

    return screen


__all__ = [
    "CLAIM_BOUNDARIES", "ENDPOINT", "IK_MODELS", "REASONS",
    "batch_basis", "bind_screen", "screen_mode",
]
