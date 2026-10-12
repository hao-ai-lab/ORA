"""Versioned read-only reach-check envelope shared by Agent and Environment.

The reach check asks one arm whether it can hold each of a batch of exact
world-frame poses, answered by the backend's own read-only inverse kinematics.
It is an internal Harness check -- ``grasp_plan``'s pooled IK selection is its
consumer -- and no ToolSurface declares it as a policy tool.
Each query names the move_pose ``interaction`` it models and may start from the
configuration an earlier query in the same batch reached (``from_index``), so a
pregrasp-then-approach chain is answered the way it would execute. It travels
over the same read-only IK bridge as ``align_plan`` (``plan_alignment`` routes
on the request ``schema``), so no new worker method, RPC verb or coordinator
callback exists for it. RoboTwin and RoboDojo route it through their existing
``align_plan`` session methods; LIBERO, which serves no ``align_plan``, answers
only this schema on the bridge (``backends/libero_pro/reach_check.py``). The
shared half lives here because ``agent`` may not import
``ora.environment``.

Each backend owns its closed rejection vocabulary; the published result schema
of the surface that serves it names the exact list. The union below is what the
controller accepts from any backend; anything else is contract drift, not a
policy error.
"""

from __future__ import annotations

from .alignment import AlignmentRequestDrift

REQUEST_SCHEMA = "ora.reach-check-request.v1"
RESULT_SCHEMA = "ora.reach-check-result.v1"

#: One call answers at most this many poses. It is a cost bound (one IK solve
#: per pose, two for a staged contact approach), and it is what a grasp_plan
#: annotation needs: three candidates x two jaw-equivalent members x pregrasp
#: and approach = 12.
MAXIMUM_POSES = 16

REQUEST_FIELDS = frozenset({"schema", "arm", "queries"})
QUERY_FIELDS = frozenset({"pose_xyzw", "interaction", "from_index"})
#: The move_pose interactions, whose route a backend may model.
INTERACTIONS = ("free_space", "precision", "contact_approach")

#: Assigned by the policy-side edge before the backend is asked: a pose the
#: action workspace refuses is refused by every motion tool too.
OUTSIDE_WORKSPACE = "outside_workspace"
#: A query chained (``from_index``) onto a pose that was not reachable has no
#: start configuration to solve from.
START_UNREACHABLE = "start_unreachable"
SHARED_REASONS = (OUTSIDE_WORKSPACE, START_UNREACHABLE)

#: RoboTwin damped-least-squares reachability (``controllers/reachability.py``).
ROBOTWIN_REASONS = ("joint_limit", "singular", "not_converged")
#: RoboTwin's other-arm screen (``backends/robotwin/reach_inter_arm.py``), which a
#: surface opts into per capability (integration 0059, M6). ``endpoint`` screens the
#: configuration the descent reached against the other arm held at its measured
#: joints, refuses with its own reason and names its own ``ik_model`` because it
#: is a different answer set from the unscreened one. ``ROBOTWIN_REASONS`` stays
#: the unscreened set, so a surface without the screen publishes the same reason
#: enum as before.
INTER_ARM_COLLISION_ENDPOINT = "inter_arm_collision_endpoint"
#: A path that meets the other arm held at its measured joints (RoboDojo below).
INTER_ARM_COLLISION_PATH = "inter_arm_collision_path"
ROBOTWIN_INTER_ARM_REASONS = {
    "endpoint": (*ROBOTWIN_REASONS, INTER_ARM_COLLISION_ENDPOINT),
}
ROBOTWIN_INTER_ARM_IK_MODELS = {
    "endpoint": "robotwin_single_seed_damped_least_squares_inter_arm_endpoint_v1",
}
#: RoboDojo SDK IK plus endpoint selection on the modelled move_pose route
#: (``backends/robodojo/ik.py`` and ``sparse_motion.py``); ``inter_arm_collision_path``
#: is a path that meets the other arm held at its measured joints. The last one means
#: the measured link6 pose disagrees with FK of the measured joints (the arm is
#: still settling or the model is off); asking again after it settles is valid.
ROBODOJO_REASONS = (
    "ik_failed", "joint_limit", "ik_branch_jump", "local_ik_branch_jump",
    "self_collision_path", INTER_ARM_COLLISION_PATH, "fk_residual_exceeded",
    "route_contact_approach_exceeds_local_bound", "fk_calibration_mismatch",
)
#: LIBERO CuRobo v0.7.8 IK (``backends/libero_pro/reach_check.py``): no seed met
#: the solver's thresholds within its joint-limit and self-collision constraints;
#: a solution whose grip site lands outside the arrival tolerance; and, as on
#: RoboDojo, CuRobo's forward kinematics of the measured joints disagreeing with
#: the measured panda_hand pose, which voids the frame conversion for every query.
LIBERO_REASONS = ("ik_failed", "fk_residual_exceeded", "fk_calibration_mismatch")
BACKEND_REASONS = (
    frozenset(ROBOTWIN_REASONS) | frozenset(ROBODOJO_REASONS) | frozenset(LIBERO_REASONS)
    | {START_UNREACHABLE}
    | frozenset(reason for reasons in ROBOTWIN_INTER_ARM_REASONS.values() for reason in reasons)
)
#: RoboDojo names the move_pose route it modelled; the RoboTwin and LIBERO checks model none.
ROBODOJO_ROUTES = ("global", "local", "global_precontact_then_local")

ROW_FIELDS = frozenset({
    "reachable", "reason", "route", "position_error_m", "orientation_error_rad",
    "joint_limit_margin_rad", "joint_delta_rad",
})
BATCH_FIELDS = frozenset({
    "ik_model", "matches_move_pose_admission", "claim_boundary", "results",
})
RESULT_FIELDS = frozenset({
    "schema", "arm", "ik_model", "matches_move_pose_admission",
    "motion_planner_called", "claim_boundary", "results",
})


class ReachCheckRequestDrift(AlignmentRequestDrift):
    """The request or backend row shape is not the one both sides agreed on.

    A correctable input is an ``EnvironmentRefusal``; drift means the two sides
    were built from different revisions of this file, which no policy action
    repairs. It subclasses ``AlignmentRequestDrift`` because both travel over
    the one read-only IK bridge, so a backend that already maps alignment drift
    to its live error maps this one the same way.
    """
