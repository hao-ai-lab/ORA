"""Read-only reach check for LIBERO: one CuRobo IK solve per exact grip-site pose.

A mixin of ``session.LiberoProSession`` plus the backend's ``reach_check_batch``
(the controller contract in ``controllers/reach_check.py``). It holds methods
only, no dataclass fields, and edits no pinned motion module.

Frame, units and controlled point (environment-compatibility audit):

* A query is a world-frame pose of the robosuite Panda ``grip_site`` -- the
  point and frame ``move_pose`` commands on LIBERO (``panda_grip_site_axes_v1``:
  approach local +Z, jaw line local +X) -- in metres with an ``xyzw``
  quaternion, already canonicalised by the agent edge.
* It is compiled exactly as the global ``move_pose`` segment compiles its
  target: ``planned_motion._grip_target_planner_pose`` turns it into the
  ``panda_hand`` pose in the robot base frame, using the measured world poses
  of the base, hand and grip site (one owner of that conversion, reused here).
* The isolated CuRobo worker (``curobo_worker.Worker.solve_ik``) solves it with
  the IK solver of the MotionGen planner ``move_pose`` uses -- ``franka.yml``,
  ``panda_joint1..7`` in radians, ``world_model: none`` -- seeded from the
  measured joints or, for a chained query, from the earlier query's solution.
* The answer is judged at the controlled point: CuRobo's forward kinematics of
  the solution, mapped back through the same measured base and hand-to-grip
  transforms, against the requested grip-site pose with the arrival bound
  ``libero_local_endpoint_tolerances`` (the bound every LIBERO ``move_pose``
  route ends on, since each ends with the bounded OSC segment).
* Before any row, CuRobo's forward kinematics of the measured joints must agree
  with the measured ``panda_hand`` within 5 mm / 0.05 rad -- the cross-check the
  planned route applies -- or every row is ``fk_calibration_mismatch``.

What it covers: joint position limits and Panda self-collision as CuRobo's
robot model encodes them, with the solver's own seed count and thresholds, in
one IK attempt per query. What it does not: the table, objects or any scene
geometry (CuRobo has no world model here); a trajectory (``plan_single`` also
optimises one, over up to four attempts); the bounded OSC route ``move_pose``
takes within 50 mm and 15 degrees, which no IK screens; payload and contact.
So it is not ``move_pose``'s admission (``matches_move_pose_admission:
false``), and a finite seed search can miss a solution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.contracts.reach_check import START_UNREACHABLE
from ora.environment.controllers.curobo_motion import CUROBO_REQUIRED_VERSION
from ora.environment.controllers.hybrid_motion import libero_local_endpoint_tolerances

from .common import LiberoProError

IK_MODEL = "libero_curobo_ik_robot_only_v1"
ARMS = ("right",)
#: The planned route's own FK cross-check (``planned_motion._validate_planned_start``).
CALIBRATION_TOLERANCE = (0.005, 0.05)
_POSITION_M, _ORIENTATION_RAD = libero_local_endpoint_tolerances()
CLAIM_BOUNDARY = (
    f"One CuRobo {CUROBO_REQUIRED_VERSION} IK solve per pose by the IK solver of the planner that move_pose's "
    "global segment uses, seeded from the measured joints or a chained query's solution, with joint limits "
    "and Panda self-collision only: no table, object or scene model. The solution's grip site is checked "
    f"against the {_POSITION_M:g} m / {_ORIENTATION_RAD:g} rad arrival bound. It plans no trajectory and does "
    "not model the bounded OSC route, so move_pose can still refuse or fail to reach a pose reported "
    "reachable, and a pose reported unreachable may still be reachable: the seed search is finite."
)


def _blank(reason: str) -> dict[str, Any]:
    return {"reachable": False, "reason": reason, "route": None, "position_error_m": None,
            "orientation_error_rad": None, "joint_limit_margin_rad": None, "joint_delta_rad": None}


def _matrix(pose: Sequence[float]) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat([float(item) for item in pose[3:]]).as_matrix()
    transform[:3, 3] = [float(item) for item in pose[:3]]
    return transform


def _residual(reached: Any, target: Sequence[float]) -> tuple[float, float]:
    """Translation (m) and rotation angle (rad) from a reached transform to a target pose."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    wanted = _matrix(target)
    position = float(np.linalg.norm(reached[:3, 3] - wanted[:3, 3]))
    rotation = float(Rotation.from_matrix(wanted[:3, :3].T @ reached[:3, :3]).magnitude())
    return position, rotation


def _clock(session: Any, environment: Any, state: Mapping[str, Any]) -> tuple[Any, ...]:
    return (session.world_mutation_sequence, getattr(environment, "_elapsed_steps", None),
            tuple(float(item) for item in state["joint_positions"]))


def reach_check_batch(
    session: Any, arm: str, queries: Sequence[Mapping[str, Any]], *,
    position_tolerance_m: float, orientation_tolerance_rad: float,
) -> dict[str, Any]:
    """One row per query, in order; nothing is commanded and no physics advances."""
    import numpy as np

    from ora.environment.controllers.reach_check import joint_delta_rad, joint_limit_margin_rad

    if arm not in ARMS:
        raise LiberoProError("LIBERO reach check exposes only the right arm")
    planner = session.motion_planner
    if planner is None:
        raise LiberoProError("CuRobo planner is not active")
    environment = session._require_environment()
    state = session._motion_state()
    before = _clock(session, environment, state)
    measured = [float(item) for item in state["joint_positions"]]
    answer = planner.solve_ik({
        "joint_names": list(state["joint_names"]), "joint_positions": measured,
        "queries": [{"target_pose_base_xyzw": session._grip_target_planner_pose(
                        state, [float(item) for item in query["pose_xyzw"]])[0],
                     "from_index": query["from_index"]} for query in queries],
    })
    if _clock(session, environment, session._motion_state()) != before:
        raise LiberoProError("read-only reach check observed the simulation advance")
    if answer.get("status") != "solved":
        raise LiberoProError(f"CuRobo IK failed: {answer.get('error', {}).get('type', 'unknown')}")
    world_from_base = _matrix(state["base_pose_world_xyzw"])
    hand_from_grip = np.linalg.inv(_matrix(state["hand_pose_world_xyzw"])) @ _matrix(
        state["grip_site_pose_world_xyzw"])
    measured_hand = np.linalg.inv(world_from_base) @ _matrix(state["hand_pose_world_xyzw"])
    drift = _residual(_matrix(answer["start_ee_pose_base_xyzw"]),
                      [*measured_hand[:3, 3], *_quaternion(measured_hand)])
    calibrated = drift[0] <= CALIBRATION_TOLERANCE[0] and drift[1] <= CALIBRATION_TOLERANCE[1]
    limits = list(zip(answer["joint_limits"][0], answer["joint_limits"][1]))
    rows: list[dict[str, Any]] = []
    for query, solved in zip(queries, answer["results"]):
        start = query["from_index"]
        if not calibrated:
            rows.append(_blank("fk_calibration_mismatch"))
            continue
        if start is not None and not rows[start]["reachable"]:
            rows.append(_blank(START_UNREACHABLE))
            continue
        if not solved["solved"]:
            rows.append(_blank("ik_failed"))
            continue
        joints = solved["joint_positions"]
        seed = measured if start is None else answer["results"][start]["joint_positions"]
        grip = world_from_base @ _matrix(solved["ee_pose_base_xyzw"]) @ hand_from_grip
        position, rotation = _residual(grip, query["pose_xyzw"])
        within = position <= position_tolerance_m and rotation <= orientation_tolerance_rad
        rows.append({"reachable": within, "reason": None if within else "fk_residual_exceeded", "route": None,
                     "position_error_m": position, "orientation_error_rad": rotation,
                     "joint_limit_margin_rad": joint_limit_margin_rad(joints, limits),
                     "joint_delta_rad": joint_delta_rad(joints, seed)})
    return {"ik_model": IK_MODEL, "matches_move_pose_admission": False,
            "claim_boundary": CLAIM_BOUNDARY, "results": rows}


def _quaternion(transform: Any) -> list[float]:
    from scipy.spatial.transform import Rotation

    return [float(item) for item in Rotation.from_matrix(transform[:3, :3]).as_quat()]


class _Adapter:
    """The one controller-contract method, bound to this session."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def reach_check_batch(self, arm: str, queries: Sequence[Mapping[str, Any]], **tolerances: float):
        return reach_check_batch(self._session, arm, queries, **tolerances)


class LiberoReachCheckMixin:
    """Serve the internal reach check on the read-only IK bridge, and nothing else."""

    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Answer ``ora.reach-check-request.v1`` over the ``align_plan`` bridge.

        LIBERO declares no ``align_plan`` tool and serves no alignment request:
        any other schema is refused as a backend fault, exactly as before this
        method existed, so serving the reach check exposes nothing new.
        """
        from ora.contracts.reach_check import REQUEST_SCHEMA, ReachCheckRequestDrift
        from ora.environment.controllers.reach_check import check_reach

        if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA:
            raise LiberoProError(
                "LIBERO serves only the internal reach check on the read-only IK bridge, not align_plan"
            )
        try:
            return check_reach(
                request, adapter=_Adapter(self),
                control_profile={"profile": {"arrival_check": {
                    "position_tolerance_m": _POSITION_M, "orientation_tolerance_rad": _ORIENTATION_RAD}}},
                available_arms=ARMS,
            )
        except ReachCheckRequestDrift as error:
            raise LiberoProError(str(error)) from error


__all__ = ["CLAIM_BOUNDARY", "IK_MODEL", "LiberoReachCheckMixin", "reach_check_batch"]
