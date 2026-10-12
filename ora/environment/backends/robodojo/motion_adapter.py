"""RoboDojo plant adapter for the backend-neutral motion transaction (v1).

Global planning here is one SDK CuRobo IK solve followed by the original
joint-target interpolation, not CuRobo trajectory planning: it certifies a
kinematic target, never a collision-free path, and ``global_joint_planning``
is therefore a non-parity claim against the LIBERO and RoboTwin CuRobo
routes.  Local Cartesian steps compile to a joint target through the same
IK.  The session's policy action path drives the SDK directly with journaled
native ticks; this bridge exists for shared-executor conformance and probes.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ora.environment.controllers.motion_transaction import (
    MotionBackendCapabilities,
    MotionTransactionError,
)

from .common import array, native_pose
from .sparse_motion import GLOBAL_PLANNING_NON_PARITY, solve_ik
from .ik import articulation_limits


PLAN_SCHEMA = "robodojo.ik-joint-interpolation-plan.v1"
COLLISION_VALIDATION_SCHEMA = "ora.robodojo-collision-validation.v1"
NATIVE_ACTION_DT_S = 0.04


class RobodojoMotionBackendAdapter:
    """Expose ARX X5 measured state, IK targets and native ticks to shared control."""

    capabilities = MotionBackendCapabilities(
        backend_id="robodojo",
        embodiment_id="arx_x5",
        arms=("left", "right"),
        controlled_point="link6",
        planner_frames={"left": "link6", "right": "link6"},
        synchronized_groups=(("left", "right"),),
        global_joint_planning=True,
        local_cartesian=True,
    )
    global_planning_parity = GLOBAL_PLANNING_NON_PARITY

    def __init__(
        self, *, read_joints: Callable[[str], Sequence[float]],
        read_link6_pose: Callable[[str], Sequence[float]],
        solve_ik: Callable[[str, Sequence[float]], Mapping[str, Any]],
        joint_limits: Callable[[str], Sequence[Sequence[float]]],
        workspace_bounds_m: Mapping[str, Sequence[float]],
        step_physics: Callable[[Mapping[str, Any]], None],
        arm_keys: Mapping[str, str],
        reachability: Callable[..., Any] | None = None,
        local_solve_ik: Callable[[str, Sequence[float]], Mapping[str, Any]] | None = None,
        reach_check: Callable[..., Any] | None = None,
    ) -> None:
        self._read_joints = read_joints
        self._read_link6_pose = read_link6_pose
        self._solve_ik = solve_ik
        self._local_solve_ik = local_solve_ik or solve_ik
        self._joint_limits = joint_limits
        self._workspace_bounds_m = dict(workspace_bounds_m)
        self._step_physics = step_physics
        self._arm_keys = dict(arm_keys)
        self._reachability = reachability
        self._reach_check = reach_check
        self._pending: dict[str, list[float]] = {}

    @classmethod
    def from_session(cls, session: Any) -> "RobodojoMotionBackendAdapter":
        manager = session.environment.robot_manager
        from .reach_check import reach_check_batch
        from .reachability import solve_batch

        def limits(arm: str) -> list[list[float]]:
            return articulation_limits(session, arm)

        return cls(
            read_joints=lambda arm: session._proprioception()["end_effectors"][arm]["arm_joint_qpos"],
            read_link6_pose=lambda arm: session._proprioception()["end_effectors"][arm]["pose_xyz_xyzw"],
            solve_ik=lambda arm, pose: solve_ik(session, arm, pose),
            local_solve_ik=lambda arm, pose: solve_ik(session, arm, pose, jump_limit_rad=session.local_joint_jump_limit_rad),
            joint_limits=limits,
            workspace_bounds_m=session.config["workspace_bounds_m"],
            step_physics=session._conformance_tick,
            arm_keys={arm: manager.process_name(robot.arm_name) for arm, robot in session.robots.items()},
            reachability=lambda arm, poses, **kwargs: solve_batch(session, arm, poses, **kwargs),
            reach_check=lambda arm, queries, **kwargs: reach_check_batch(session, arm, queries, **kwargs),
        )

    def _require_arm(self, arm: str) -> None:
        if arm not in self.capabilities.arms:
            raise MotionTransactionError(f"RoboDojo motion adapter has no arm {arm!r}")

    def read_joint_positions(self, arm: str) -> Sequence[float]:
        self._require_arm(arm)
        return [float(item) for item in self._read_joints(arm)]

    def read_public_tcp_pose(self, arm: str) -> Sequence[float]:
        self._require_arm(arm)
        return [float(item) for item in self._read_link6_pose(arm)]

    def capture_action_start(self) -> Mapping[str, Any]:
        return {
            "joints": {arm: list(self.read_joint_positions(arm)) for arm in self.capabilities.arms},
            "public_tcp_pose_xyzw": {arm: list(self.read_public_tcp_pose(arm)) for arm in self.capabilities.arms},
        }

    def compile_public_target(self, arm: str, public_tcp_pose_xyzw: Sequence[float]) -> Any:
        self._require_arm(arm)
        return native_pose(public_tcp_pose_xyzw)

    @staticmethod
    def _public_pose(planner_target: Sequence[float]) -> list[float]:
        pose = [float(item) for item in planner_target]
        if len(pose) != 7:
            raise MotionTransactionError("RoboDojo planner target must be one wxyz pose")
        return pose[:3] + pose[4:] + [pose[3]]

    def plan_global(self, arm: str, planner_target: Any, action_start: Mapping[str, Any]) -> Mapping[str, Any]:
        self._require_arm(arm)
        solved = self._solve_ik(arm, self._public_pose(planner_target))
        joints = solved.get("joint_target")
        if solved.get("ik_selection", {}).get("outcome") == "joint_jump":
            joints = None
        trajectory = None
        if joints is not None:
            joints = [float(item) for item in joints]
            trajectory = {"positions": [joints], "velocities": [[0.0] * len(joints)], "dt_s": NATIVE_ACTION_DT_S}
        return {
            "schema": PLAN_SCHEMA, "reachable": joints is not None,
            "ik_status": str(solved.get("ik_status")), "joint_target": joints,
            "joint_delta_rad": solved.get("joint_delta_rad"), "trajectory": trajectory,
            "ik_selection": solved.get("ik_selection"),
            "parity": GLOBAL_PLANNING_NON_PARITY,
        }

    def plan_batch_global(
        self, arm: str, planner_targets: Sequence[Any], action_start: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._require_arm(arm)
        return {
            "schema": "robodojo.motion-plan-batch-adapter.v1",
            "results": [self.plan_global(arm, target, action_start) for target in planner_targets],
        }

    def solve_batch_reachability(self, arm: str, target_poses_xyzw: Sequence[Sequence[float]], **kwargs: Any):
        self._require_arm(arm)
        if self._reachability is None:
            raise MotionTransactionError("RoboDojo read-only kinematics is unavailable")
        return self._reachability(arm, target_poses_xyzw, **kwargs)

    def reach_check_batch(self, arm: str, queries: Sequence[Mapping[str, Any]], **kwargs: Any):
        self._require_arm(arm)
        if self._reach_check is None:
            raise MotionTransactionError("RoboDojo read-only reach check is unavailable")
        return self._reach_check(arm, queries, **kwargs)

    def fk_control_pose(self, arm: str, joints: Sequence[float]) -> Sequence[float]:
        self._require_arm(arm)
        raise MotionTransactionError(
            "RoboDojo v1 exposes no forward kinematics; planned poses are IK targets, not FK-verified"
        )

    def fk_public_tcp_pose(self, arm: str, joints: Sequence[float]) -> Sequence[float]:
        return self.fk_control_pose(arm, joints)

    def joint_limits(self, arm: str) -> Sequence[Sequence[float]]:
        self._require_arm(arm)
        return self._joint_limits(arm)

    def workspace_bounds_m(self) -> Mapping[str, Sequence[float]]:
        return dict(self._workspace_bounds_m)

    def validate_collision(self, plans: Mapping[str, Any], action_start: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "schema": COLLISION_VALIDATION_SCHEMA, "status": "not_validated",
            "reason": "independent_per_arm_ik_without_dual_arm_collision_model",
            "arms": sorted(str(arm) for arm in plans),
        }

    def write_joint_target(self, arm: str, position: Sequence[float], velocity: Sequence[float]) -> None:
        self._require_arm(arm)
        values = np.asarray(position, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise MotionTransactionError("RoboDojo joint target must be one finite vector")
        self._pending[arm] = values.tolist()

    def step_physics(self) -> None:
        control = {self._arm_keys[arm]: {"position": list(joints)} for arm, joints in self._pending.items()}
        self._step_physics(control)

    def compute_local_joint_target(
        self, arm: str, current_joints: Sequence[float], current_public_tcp_pose_xyzw: Sequence[float],
        target_public_tcp_pose_xyzw: Sequence[float], profile: Mapping[str, Any],
    ) -> Mapping[str, Sequence[float]]:
        self._require_arm(arm)
        solved = self._local_solve_ik(arm, [float(item) for item in target_public_tcp_pose_xyzw])
        joints = solved.get("joint_target")
        if joints is None or solved.get("ik_selection", {}).get("outcome") == "joint_jump":
            raise MotionTransactionError(f"RoboDojo local IK failed: {solved.get('ik_status')}")
        joints = [float(item) for item in joints]
        return {"position": joints, "velocity": [0.0] * len(joints)}


__all__ = ["COLLISION_VALIDATION_SCHEMA", "PLAN_SCHEMA", "RobodojoMotionBackendAdapter"]
