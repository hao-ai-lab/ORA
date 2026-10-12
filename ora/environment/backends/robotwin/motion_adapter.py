"""RoboTwin plant adapter for the backend-neutral motion transaction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ora.environment.controllers.motion_transaction import (
    MotionBackendCapabilities,
)


class RoboTwinMotionBackendAdapter:
    """Expose ALOHA measured state and sparse joint targets to shared control."""

    capabilities = MotionBackendCapabilities(
        backend_id="robotwin",
        embodiment_id="aloha_agilex",
        arms=("left", "right"),
        controlled_point="parallel_jaw_grip_center_tcp",
        planner_frames={"left": "fl_link6", "right": "fr_link6"},
        synchronized_groups=(("left", "right"),),
        global_joint_planning=True,
        local_cartesian=True,
    )

    def __init__(self, *, session: Any, environment: Any):
        self._session = session
        self._environment = environment
        scene = getattr(environment, "scene", None)
        robot = getattr(environment, "robot", None)
        if scene is None or not callable(getattr(scene, "step", None)) or robot is None:
            raise RuntimeError("RoboTwin motion adapter requires scene.step and robot")
        self._scene = scene
        self._robot = robot

    def read_joint_positions(self, arm: str) -> Sequence[float]:
        return self._session._measured_arm_joints(self._environment, arm)

    def capture_action_start(self) -> Mapping[str, Any]:
        return {
            "joints": {
                arm: list(self.read_joint_positions(arm))
                for arm in self.capabilities.arms
            },
            "public_tcp_pose_xyzw": {
                arm: list(self.read_public_tcp_pose(arm))
                for arm in self.capabilities.arms
            },
            "planner_qpos": {
                arm: list(getattr(self._robot, f"{arm}_entity").get_qpos())
                for arm in self.capabilities.arms
            },
        }

    def compile_public_target(
        self, arm: str, public_tcp_pose_xyzw: Sequence[float],
    ) -> Any:
        if arm not in self.capabilities.arms or len(public_tcp_pose_xyzw) != 7:
            raise RuntimeError("RoboTwin public TCP target is invalid")
        from .session import _xyzw_to_wxyz

        public_wxyz = _xyzw_to_wxyz(
            list(public_tcp_pose_xyzw[:3]), list(public_tcp_pose_xyzw[3:]),
        )
        return {
            "kind": "pose",
            "pose_wxyz": self._session._public_to_planner_wxyz(
                arm, public_wxyz,
            ),
        }

    def plan_global(
        self, arm: str, planner_target: Any, action_start: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if arm not in self.capabilities.arms:
            raise RuntimeError("RoboTwin planner arm is invalid")
        qpos = action_start.get("planner_qpos", {}).get(arm)
        if not isinstance(qpos, list):
            raise RuntimeError("RoboTwin action-start planner qpos is unavailable")
        if not isinstance(planner_target, Mapping):
            raise RuntimeError("RoboTwin planner target is malformed")
        kind = planner_target.get("kind")
        if kind == "pose":
            planner = getattr(self._robot, f"{arm}_plan_path")
            return planner(planner_target["pose_wxyz"], last_qpos=qpos)
        if kind == "joint":
            planner = getattr(self._robot, f"{arm}_plan_joint_path")
            return planner(planner_target["joints"], last_qpos=qpos)
        raise RuntimeError("RoboTwin planner target kind is invalid")

    def plan_batch_global(
        self, arm: str, planner_targets: Sequence[Any],
        action_start: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if arm not in self.capabilities.arms or not planner_targets:
            raise RuntimeError("RoboTwin batch planner request is invalid")
        if any(
            not isinstance(target, Mapping) or target.get("kind") != "pose"
            for target in planner_targets
        ):
            raise RuntimeError("RoboTwin batch planner accepts explicit poses only")
        qpos = action_start.get("planner_qpos", {}).get(arm)
        if not isinstance(qpos, list):
            raise RuntimeError("RoboTwin action-start planner qpos is unavailable")
        planner = getattr(self._robot, f"{arm}_plan_path")
        return {
            "schema": "robotwin.curobo-topk-batch-result.v1",
            "results": [
                planner(target["pose_wxyz"], last_qpos=qpos)
                for target in planner_targets
            ],
        }

    def fk_control_pose(self, arm: str, joints: Sequence[float]) -> Sequence[float]:
        return self._session._pinocchio_link6_pose_xyzw(
            self._environment, arm, joints,
        )

    def fk_public_tcp_pose(self, arm: str, joints: Sequence[float]) -> Sequence[float]:
        return self._session._pinocchio_public_pose_xyzw(
            self._environment, arm, joints,
        )

    def joint_limits(self, arm: str) -> Sequence[Sequence[float]]:
        return self._session._arm_joint_limits(self._environment, arm)

    def workspace_bounds_m(self) -> Mapping[str, Sequence[float]]:
        return self._session.workspace_bounds_m

    def validate_collision(
        self, plans: Mapping[str, Any], action_start: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        from .planning import validate_inter_arm_clearance

        return validate_inter_arm_clearance(
            plans,
            measured_joints=action_start["joints"],
            fk_chain_points=lambda selected, joints: (
                self._session._pinocchio_chain_points(
                    self._environment, selected, joints,
                )
            ),
            link_names={arm: spec["pinocchio_chain_links"] for arm, spec in self._session.control_profile["profile"]["arms"].items()},
            profile=self._session.control_profile["profile"][
                "dual_arm_collision_check"
            ],
        )

    def solve_batch_reachability(
        self, arm: str, target_poses_xyzw: Sequence[Sequence[float]], *,
        position_tolerance_m: float, orientation_tolerance_rad: float,
    ) -> list[Mapping[str, Any]]:
        """Decide reachability for many poses with inverse kinematics alone.

        ``plan_batch_global`` is the only other batch entry and it always calls
        CuRobo, so asking "can the arm reach these poses" used to cost one motion
        plan per pose. This composes the forward kinematics, joint limits and
        damped-least-squares step that were already here, and touches no planner.

        Every candidate starts from the same measured seed, so the ranking
        compares poses rather than the order they happened to be solved in.
        """

        from .planning import solve_reachability

        seed = [float(value) for value in self.read_joint_positions(arm)]
        limits = self.joint_limits(arm)
        results: list[Mapping[str, Any]] = []
        for pose in target_poses_xyzw:
            solved = solve_reachability(
                target_pose_xyzw=pose, seed_joints=seed,
                fk_public_pose=lambda joints: self.fk_public_tcp_pose(arm, joints),
                joint_limits=limits,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
            results.append({**solved, "arm": arm, "seed_joints": list(seed)})
        return results

    def reach_check_batch(
        self, arm: str, queries: Sequence[Mapping[str, Any]], *,
        position_tolerance_m: float, orientation_tolerance_rad: float,
    ) -> Mapping[str, Any]:
        """Answer exact poses with the same read-only IK ``align_plan`` uses.

        Each query is solved by one damped-least-squares descent seeded from
        the measured joints, or -- for a chained query -- from the solution the
        earlier query reached. Rows carry the residual, joint-limit margin and
        joint move at the configuration the descent ended on. This solver models
        no move_pose route, so ``interaction`` does not change the answer and
        ``route`` is null. It is not the 128-seed collision-checked CuRobo IK
        ``move_pose`` plans through, and the result says so twice:
        ``matches_move_pose_admission`` is false and the claim boundary names it.

        When the session's surface asked for the other-arm screen
        (``reach_inter_arm``, integration 0059 M6), every solved row is then
        screened -- the reached configuration alone -- against the other arm
        held at its measured joints; a blocked row becomes unreachable with
        the screen's reason, no chained query starts from it, and the batch
        names the screened ``ik_model`` and claim boundary.
        """

        from ora.contracts.reach_check import START_UNREACHABLE
        from ora.environment.controllers.reach_check import (
            joint_delta_rad, joint_limit_margin_rad,
        )
        from .planning import solve_reachability
        from .reach_inter_arm import batch_basis

        measured = [float(value) for value in self.read_joint_positions(arm)]
        limits = self.joint_limits(arm)
        mode, screen = self._reach_check_screen(arm)
        rows: list[dict[str, Any]] = []
        ends: list[list[float] | None] = []
        for query in queries:
            seed = measured if query["from_index"] is None else ends[query["from_index"]]
            if seed is None:
                rows.append({
                    "reachable": False, "reason": START_UNREACHABLE, "route": None,
                    "position_error_m": None, "orientation_error_rad": None,
                    "joint_limit_margin_rad": None, "joint_delta_rad": None,
                })
                ends.append(None)
                continue
            solved = solve_reachability(
                target_pose_xyzw=query["pose_xyzw"], seed_joints=seed,
                fk_public_pose=lambda joints: self.fk_public_tcp_pose(arm, joints),
                joint_limits=limits,
                position_tolerance_m=position_tolerance_m,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
            joints = solved.get("joints")
            reachable = bool(solved["reachable"])
            reason = None if reachable else solved.get("reason", "not_converged")
            end = [float(value) for value in joints] if reachable else None
            blocked = None if end is None or screen is None else screen(end)
            if blocked is not None:
                reachable, reason, end = False, blocked["reason"], None
            rows.append({
                "reachable": reachable,
                "reason": reason,
                "route": None,
                "position_error_m": float(solved["position_error_m"]),
                "orientation_error_rad": float(solved["orientation_error_rad"]),
                "joint_limit_margin_rad": joint_limit_margin_rad(joints, limits),
                "joint_delta_rad": joint_delta_rad(joints, seed),
            })
            ends.append(end)
        if mode is not None:
            return {**batch_basis(mode), "results": rows}
        return {
            "ik_model": "robotwin_single_seed_damped_least_squares_v1",
            "matches_move_pose_admission": False,
            "claim_boundary": (
                "One damped-least-squares IK descent from the measured joints (or a "
                "chained query's earlier solution) with joint limits only: no collision "
                "model, no route model, and not the 128-seed collision-checked CuRobo IK "
                "move_pose plans through, so move_pose can still refuse a reachable pose "
                "or reach one reported unreachable."
            ),
            "results": rows,
        }

    def _reach_check_screen(self, arm: str) -> tuple[str | None, Any]:
        """The session's reach-check screen mode and the screen bound for ``arm``."""

        from .reach_inter_arm import bind_screen, screen_mode

        mode = screen_mode(screen=self._session.reach_check_inter_arm_screen)
        if mode is None:
            return None, None
        return mode, bind_screen(
            arm,
            measured_joints={
                name: [float(value) for value in self.read_joint_positions(name)]
                for name in self.capabilities.arms
            },
            fk_chain_points=lambda selected, joints: (
                self._session._pinocchio_chain_points(self._environment, selected, joints)
            ),
            profile=self._session.control_profile["profile"]["dual_arm_collision_check"],
        )

    def compute_local_joint_target(
        self, arm: str, current_joints: Sequence[float],
        current_public_tcp_pose_xyzw: Sequence[float],
        target_public_tcp_pose_xyzw: Sequence[float], profile: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        import numpy as np

        from .planning import (
            RoboTwinPlanRejected,
            cartesian_pose_error_twist,
            weighted_svd_dls_joint_delta,
        )

        error = cartesian_pose_error_twist(
            target_public_tcp_pose_xyzw, current_public_tcp_pose_xyzw,
        )
        jacobian = self.public_tcp_jacobian(arm, current_joints)
        limits = np.asarray(self.joint_limits(arm), dtype=np.float64)
        delta, diagnostics = weighted_svd_dls_joint_delta(
            error, jacobian,
            current_joints=current_joints,
            joint_limits=limits,
            base_damping=float(profile["base_damping"]),
            singular_value_threshold=float(profile["singular_value_threshold"]),
            maximum_damping=float(profile["maximum_damping"]),
            joint_limit_margin_rad=float(profile["joint_limit_margin_rad"]),
            minimum_joint_mobility=float(profile["minimum_joint_mobility"]),
            maximum_translation_step_m=float(profile["maximum_translation_step_m"]),
            maximum_rotation_step_rad=float(profile["maximum_rotation_step_rad"]),
            maximum_joint_step_rad=float(profile["maximum_joint_step_rad"]),
        )
        current = np.asarray(current_joints, dtype=np.float64)

        def score(pose: Sequence[float]) -> float:
            residual = cartesian_pose_error_twist(
                target_public_tcp_pose_xyzw, pose,
            )
            return (
                float(np.linalg.norm(residual[:3]))
                / float(profile["maximum_translation_step_m"])
                + float(np.linalg.norm(residual[3:]))
                / float(profile["maximum_rotation_step_rad"])
            )

        current_score = score(current_public_tcp_pose_xyzw)
        accepted_target = None
        predicted_pose = None
        accepted_scale = None
        for backtrack in range(int(profile["maximum_line_search_halvings"]) + 1):
            scale = 0.5 ** backtrack
            candidate = current + scale * delta
            if np.any(candidate < limits[:, 0]) or np.any(candidate > limits[:, 1]):
                continue
            pose = self.fk_public_tcp_pose(arm, candidate)
            if score(pose) < current_score - float(profile["minimum_predicted_progress"]):
                accepted_target = candidate
                predicted_pose = pose
                accepted_scale = scale
                break
        if accepted_target is None or predicted_pose is None or accepted_scale is None:
            raise RoboTwinPlanRejected(
                "local_servo_stalled",
                f"RoboTwin analytic local proposal for {arm} has no predicted progress",
                details={
                    "arm": arm,
                    "current_score": current_score,
                    "maximum_line_search_halvings": int(
                        profile["maximum_line_search_halvings"]
                    ),
                },
            )
        applied_delta = accepted_target - current
        velocity = np.clip(
            applied_delta / float(profile["velocity_target_horizon_s"]),
            -float(profile["maximum_velocity_target_rad_s"]),
            float(profile["maximum_velocity_target_rad_s"]),
        )
        return {
            "position": accepted_target.tolist(),
            "velocity": velocity.tolist(),
            "predicted_public_tcp_pose_xyzw": [
                float(value) for value in predicted_pose
            ],
            "predicted_score_before": current_score,
            "predicted_score_after": score(predicted_pose),
            "line_search_scale": accepted_scale,
            "solver": diagnostics,
        }

    def public_tcp_jacobian(
        self, arm: str, current_joints: Sequence[float],
    ) -> Any:
        """Return the analytic world-frame Jacobian at the public grip center."""

        import numpy as np
        import transforms3d as t3d

        from .planning import public_tcp_point_jacobian

        context = self._session._pinocchio_context(self._environment, arm)
        entity = context["entity"]
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64).copy()
        joints = np.asarray(current_joints, dtype=np.float64)
        if joints.shape != (len(context["planner_joint_indices"]),):
            raise RuntimeError(f"RoboTwin analytic Jacobian joints are invalid for {arm}")
        for index, value in zip(context["planner_joint_indices"], joints):
            qpos[int(index)] = float(value)
        model = context["model"]
        model.compute_full_jacobian(qpos)
        # In the pinned SAPIEN 3.0b1 build, local=False exposes the spatial
        # Jacobian in the articulation-root frame.  Its linear rows are the
        # spatial velocity about the articulation-root origin, not the
        # velocity of link6.  Rotate both row blocks into simulator world and
        # then shift that spatial velocity directly to the public TCP.
        #
        # This convention is source-qualified against finite-difference FK.
        # Treating local=True as a link-origin world Jacobian produces columns
        # that are nearly orthogonal to measured FK for ALOHA joints 1 and 6.
        raw = np.asarray(
            model.get_link_jacobian(int(context["tcp_link_index"]), local=False),
            dtype=np.float64,
        )
        if raw.ndim != 2 or raw.shape[0] != 6:
            raise RuntimeError("RoboTwin analytic link Jacobian shape drifted")
        selected = raw[:, [int(index) for index in context["planner_joint_indices"]]]
        root = self._session._entity_root_pose(entity)
        root_rotation = t3d.quaternions.quat2mat(
            np.asarray(root.q, dtype=np.float64)
        )
        selected = np.vstack((
            root_rotation @ selected[:3],
            root_rotation @ selected[3:],
        ))
        public_pose = np.asarray(
            self.fk_public_tcp_pose(arm, current_joints), dtype=np.float64,
        )
        root_position = np.asarray(root.p, dtype=np.float64)
        offset_world = public_pose[:3] - root_position
        return public_tcp_point_jacobian(selected, offset_world)

    def read_public_tcp_pose(self, arm: str) -> Sequence[float]:
        # RoboTwin exposes wxyz internally; shared motion uses public xyzw.
        from .session import _wxyz_to_xyzw

        return _wxyz_to_xyzw(
            self._session._public_pose_wxyz(self._environment, arm)
        )

    def write_joint_target(
        self, arm: str, position: Sequence[float], velocity: Sequence[float],
    ) -> None:
        self._robot.set_arm_joints(position, velocity, arm)

    def step_physics(self) -> None:
        self._scene.step()


__all__ = ["RoboTwinMotionBackendAdapter"]
