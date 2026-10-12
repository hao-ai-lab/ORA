# Harness-vendored RoboTwin planner adapter.
#
# Provenance: RoboTwin upstream c3ddfa8b97d5519efa828b075999bd0006778e5e plus
# the reviewed EXP06A planner chain (external commit dee8ee9062165bd3f921f05e
# b0f96aaaa6ba1b06, CaP-X MIT-derived; see CAPX_MIT_ATTRIBUTION.md), now owned
# by robot_harness and installed onto a vanilla upstream checkout by
# vendored_install.  Harness-side corrections over the external chain:
#   1. joint-segment fallbacks first fold the exact IK goal by whole
#      revolutions toward the start (inside position limits), then admit on
#      the UNWRAPPED delta with a per-joint |goal - start| <= min(pi, limit
#      span) bound, so remote elbow/wrist branches can no longer fold into
#      apparent nearby goals;
#   2. the executed segment endpoint is exactly the reported ik_joint_goal
#      (the normalized configuration; no 2*pi endpoint/report divergence);
#   3. every fallback trajectory that bypasses CuRobo time parameterization
#      (linear segments and graph-only paths) is densified to the execution
#      joint step and retimed against the robot's joint velocity limits, so
#      interpolation_dt_s and velocity are honest instead of zero/fabricated.
import mplib.planner
import mplib
import numpy as np
import pdb
import traceback
import numpy as np
import toppra as ta
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
import transforms3d as t3d
import envs._GLOBAL_CONFIGS as CONFIGS


try:
    # ********************** CuroboPlanner (optional) **********************
    from curobo.types.math import Pose as CuroboPose
    import time
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import (
        MotionGen,
        MotionGenConfig,
        MotionGenPlanConfig,
        PoseCostMetric,
    )
    from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
    from curobo.util import logger
    import torch
    import yaml
    from curobo.util import logger
    logger.setup_logger(level="error", logger_name="curobo")

    def _planner_value(value):
        """Convert a CuRobo scalar/tensor into a process-safe Python value."""
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().to("cpu")
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "name"):
            return value.name
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)


    def _motion_result(result, *, joint_names, preserve_batch=False, ik=None):
        """Preserve CuRobo diagnostics without returning private debug buffers."""
        success_value = _planner_value(result.success)
        if isinstance(success_value, list):
            successes = [bool(item) for item in np.asarray(success_value).reshape(-1)]
            status_values = ["Success" if item else "Fail" for item in successes]
            status = (
                status_values
                if preserve_batch else
                status_values[0] if len(status_values) == 1 else status_values
            )
        else:
            successes = [bool(success_value)]
            status = "Success" if successes[0] else "Fail"
        record = {
            "schema": "robotwin.curobo-plan-result.v1",
            "status": status,
            "planner_status": _planner_value(result.status),
            "valid_query": bool(result.valid_query),
            "attempts": int(result.attempts),
            "trajopt_attempts": int(result.trajopt_attempts),
            "used_graph": bool(result.used_graph),
            "goalset_index": _planner_value(result.goalset_index),
            "path_buffer_last_tstep": _planner_value(result.path_buffer_last_tstep),
            "errors": {
                "position": _planner_value(result.position_error),
                "rotation": _planner_value(result.rotation_error),
                "cspace": _planner_value(result.cspace_error),
            },
            "timings_s": {
                "solve": float(result.solve_time),
                "ik": float(result.ik_time),
                "graph": float(result.graph_time),
                "trajopt": float(result.trajopt_time),
                "finetune": float(result.finetune_time),
                "total": float(result.total_time),
            },
            "interpolation_dt_s": float(result.interpolation_dt),
            "joint_names": list(joint_names),
            "ik": ik,
        }
        if result.interpolated_plan is not None and any(successes):
            if preserve_batch:
                # CuRobo stores batched interpolations in a fixed-size padded
                # buffer.  get_paths() applies each row's authoritative
                # path_buffer_last_tstep; returning the raw tensor would turn
                # padding into thousands of fake controller waypoints.
                paths = (
                    [result.get_interpolated_plan()]
                    if len(successes) == 1 else result.get_paths()
                )
                record["position"] = [
                    np.array(paths[index].position.detach().to("cpu"))
                    if successes[index] else None
                    for index in range(len(successes))
                ]
                record["velocity"] = [
                    np.array(paths[index].velocity.detach().to("cpu"))
                    if successes[index] else None
                    for index in range(len(successes))
                ]
            else:
                path = result.get_interpolated_plan()
                record["position"] = np.array(path.position.detach().to("cpu"))
                record["velocity"] = np.array(path.velocity.detach().to("cpu"))
        return record


    def _batch_motion_records(records):
        """Combine independently exact candidate plans behind one batch RPC."""
        if not records:
            raise ValueError("batch planning requires at least one target")
        scalar_fields = (
            "status", "planner_status", "valid_query", "attempts",
            "trajopt_attempts", "used_graph", "goalset_index",
            "path_buffer_last_tstep", "interpolation_dt_s", "ik", "error",
        )
        return {
            "schema": "robotwin.curobo-plan-result.v1",
            **{field: [record.get(field) for record in records] for field in scalar_fields},
            "joint_names": list(records[0]["joint_names"]),
            "position": [record.get("position") for record in records],
            "velocity": [record.get("velocity") for record in records],
            "errors": {
                name: [record.get("errors", {}).get(name) for record in records]
                for name in ("position", "rotation", "cspace")
            },
            "timings_s": {
                name: [record.get("timings_s", {}).get(name) for record in records]
                for name in ("solve", "ik", "graph", "trajopt", "finetune", "total")
            },
        }

    class CuroboPlanner:

        def __init__(
            self,
            robot_origion_pose,
            active_joints_name,
            all_joints,
            yml_path=None,
        ):
            super().__init__()
            ta.setup_logging("CRITICAL")  # hide logging
            logger.setup_logger(level="error", logger_name="'curobo")

            if yml_path != None:
                self.yml_path = yml_path
            else:
                raise ValueError("[Planner.py]: CuroboPlanner yml_path is None!")
            self.robot_origion_pose = robot_origion_pose
            self.active_joints_name = active_joints_name
            self.all_joints = all_joints

            # translate from baselink to arm's base
            with open(self.yml_path, "r") as f:
                yml_data = yaml.safe_load(f)
            self.frame_bias = yml_data["planner"]["frame_bias"]

            # motion generation
            if True:
                world_config = {
                    "cuboid": {
                        "table": {
                            "dims": [0.7, 2, 0.04],  # x, y, z
                            "pose": [
                                self.robot_origion_pose.p[1],
                                0.0,
                                0.74 - self.robot_origion_pose.p[2],
                                1,
                                0,
                                0,
                                0.0,
                            ],  # x, y, z, qw, qx, qy, qz
                        },
                    }
                }
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                # Match the source-bound CaP-X CuRobo search width (MIT,
                # commit 53e9966).  A single trajectory seed produced an
                # intermittent all-candidate failure for an otherwise
                # repeatably reachable ALOHA pre-contact goal.
                num_trajopt_seeds=4,
                num_graph_seeds=4,
            )

            self.motion_gen_config = motion_gen_config
            self.world_config = world_config

            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen.warmup()
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.yml_path,
                world_config,
                interpolation_dt=1 / 250,
                num_trajopt_seeds=4,
                num_graph_seeds=4,
                # Harness reachability probes intentionally vary the number of
                # current goal poses (one exact target through a bounded batch
                # of grasp variants).  This pinned CUDA stack cannot reset a
                # captured graph when the batch shape changes, so the read-only
                # batch planner must remain shape-dynamic instead of inheriting
                # RoboTwin's fixed ROTATE_NUM graph.
                use_cuda_graph=False,
            )
            self.motion_gen_batch = MotionGen(motion_gen_config)
            self.motion_gen_batch.warmup(batch=CONFIGS.ROTATE_NUM)

        @staticmethod
        def _wrapped_joint_distance(solutions, reference):
            """Squared wrapped distance used by CaP-X (MIT, commit 53e9966)."""
            values = np.asarray(solutions, dtype=np.float64)
            current = np.asarray(reference, dtype=np.float64)
            delta = (values - current[None, :] + np.pi) % (2.0 * np.pi) - np.pi
            return np.sum(delta * delta, axis=-1)

        @classmethod
        def _deduplicated_ik_candidates(
            cls, solutions, reference, *, threshold_rad=0.05,
            nearest_count=8, diverse_count=8,
        ):
            """Return deterministic near and far-point-sampled IK candidates.

            CuRobo may return many seeds that collapse onto the same wrapped
            joint configuration.  Keeping all of them only repeats the same
            failed joint-space query.  We first deduplicate in wrapped joint
            space, retain the nearest configurations, then greedily add remote
            elbow/wrist configurations by farthest-point sampling.
            """

            values = np.asarray(solutions, dtype=np.float64)
            reference = np.asarray(reference, dtype=np.float64)
            if values.ndim != 2 or reference.shape != (values.shape[1],):
                raise ValueError("IK candidate dimensions are invalid")
            distances = cls._wrapped_joint_distance(values, reference)
            ordered = sorted(range(len(values)), key=lambda index: (distances[index], index))
            unique = []
            unique_source_indices = []
            for index in ordered:
                candidate = values[index]
                if any(
                    np.sqrt(cls._wrapped_joint_distance([candidate], existing)[0])
                    < threshold_rad
                    for existing in unique
                ):
                    continue
                unique.append(candidate)
                unique_source_indices.append(index)
            if not unique:
                return [], [], [], 0

            selected = list(range(min(nearest_count, len(unique))))
            remaining = [index for index in range(len(unique)) if index not in selected]
            while remaining and len(selected) < nearest_count + diverse_count:
                def separation(index):
                    candidate = unique[index]
                    return min(
                        cls._wrapped_joint_distance([candidate], unique[other])[0]
                        for other in selected
                    )

                chosen = max(
                    remaining,
                    key=lambda index: (separation(index), -distances[unique_source_indices[index]], -index),
                )
                selected.append(chosen)
                remaining.remove(chosen)
            return (
                [unique[index] for index in selected],
                [unique_source_indices[index] for index in selected],
                [float(np.sqrt(distances[unique_source_indices[index]])) for index in selected],
                len(unique),
            )

        def _joint_limit_arrays(self):
            """Return (position_limits[2,dof] or None, per-joint vmax or None)."""
            try:
                limits = self.motion_gen.kinematics.get_joint_limits()
                position = np.asarray(
                    _planner_value(limits.position), dtype=np.float64,
                )
                velocity = np.asarray(
                    _planner_value(limits.velocity), dtype=np.float64,
                )
            except (AttributeError, TypeError, ValueError):
                return None, None
            dof = len(self.active_joints_name)
            if position.shape != (2, dof) or not np.isfinite(position).all():
                position = None
            if velocity.shape == (2, dof):
                velocity = np.minimum(np.abs(velocity[0]), np.abs(velocity[1]))
            elif velocity.shape == (dof,):
                velocity = np.abs(velocity)
            else:
                velocity = None
            if velocity is not None and (
                not np.isfinite(velocity).all() or bool(np.any(velocity <= 0.0))
            ):
                velocity = None
            return position, velocity

        @staticmethod
        def _densify_joint_path(position, *, maximum_joint_step_rad):
            """Linearly subdivide segments so no joint moves more per step."""
            position = np.asarray(position, dtype=np.float64)
            if position.ndim != 2 or len(position) < 2:
                return position
            rows = [position[0]]
            for index in range(1, len(position)):
                previous = position[index - 1]
                current = position[index]
                peak = float(np.max(np.abs(current - previous)))
                segments = max(1, int(np.ceil(peak / float(maximum_joint_step_rad))))
                for sub in range(1, segments + 1):
                    rows.append(previous + (current - previous) * (sub / segments))
            return np.asarray(rows, dtype=np.float64)

        @staticmethod
        def _normalize_joint_goal(start, goal, position_limits):
            """Fold each goal joint by 2*pi toward the start, inside limits.

            The embodiment limit file may declare placeholder joint ranges far
            wider than one revolution, so exact IK legitimately returns
            2*pi-equivalent configurations.  Executing the raw value would
            command a full extra revolution; rejecting it would refuse a goal
            the arm can reach directly.  The honest resolution is to execute
            the wrapped-equivalent configuration nearest the start and REPORT
            exactly that configuration as the executed joint goal.
            """

            start = np.asarray(start, dtype=np.float64)
            goal = np.asarray(goal, dtype=np.float64)
            turns = np.round((goal - start) / (2.0 * np.pi))
            normalized = goal - 2.0 * np.pi * turns
            if position_limits is not None:
                below = normalized < position_limits[0]
                normalized[below] += 2.0 * np.pi
                above = normalized > position_limits[1]
                normalized[above] -= 2.0 * np.pi
                if bool(np.any(normalized < position_limits[0] - 1e-9)) or bool(
                    np.any(normalized > position_limits[1] + 1e-9)
                ):
                    return None, turns
            return normalized, turns

        @staticmethod
        def _retime_joint_path(position, *, velocity_limits, velocity_utilization=0.5):
            """Return (dt_s, velocity, peak_step_rad) bounded by joint velocity limits."""
            position = np.asarray(position, dtype=np.float64)
            steps = np.abs(np.diff(position, axis=0))
            peak_step = float(np.max(steps)) if steps.size else 0.0
            # Placeholder limit files can declare absurd joint speeds; cap the
            # budget at a physical arm bound so the declared dt stays honest.
            budget = np.minimum(
                np.asarray(velocity_limits, dtype=np.float64), 3.0
            ) * float(velocity_utilization)
            per_step = (
                np.max(steps / budget[None, :], axis=1) if steps.size else np.zeros(0)
            )
            dt = max(float(np.max(per_step)) if per_step.size else 0.0, 1e-3)
            velocity = np.zeros_like(position)
            if len(position) >= 2:
                velocity[1:-1] = (position[2:] - position[:-2]) / (2.0 * dt)
            return dt, velocity, peak_step

        def _fallback_precondition_failure(
            self, *, reason, detail, elapsed_s,
        ):
            """Record a skipped joint-segment fallback instead of silence."""
            return {
                "schema": "robotwin.curobo-plan-result.v1",
                "status": "Fail",
                "planner_status": "VALIDATED_SEGMENT_PRECONDITION_FAIL",
                "valid_query": True,
                "attempts": 1,
                "trajopt_attempts": 0,
                "used_graph": False,
                "goalset_index": None,
                "path_buffer_last_tstep": None,
                "errors": {"position": None, "rotation": None, "cspace": None},
                "timings_s": {
                    "solve": elapsed_s, "ik": 0.0, "graph": 0.0,
                    "trajopt": 0.0, "finetune": 0.0, "total": elapsed_s,
                },
                "interpolation_dt_s": None,
                "joint_names": list(self.active_joints_name),
                "constraint_check": {
                    "backend": "harness_vendored_precondition",
                    "precondition_failure": reason,
                    "detail": detail,
                },
            }

        def _validated_linear_joint_fallback(
            self, start_joint_states, goal, *, distance_rad,
            nearby_distance_rad=0.60, maximum_distance_rad=3.00,
            validation_joint_step_rad=0.005,
            execution_joint_step_rad=0.025,
        ):
            """Admit an exact-IK joint segment through dense CuRobo constraints.

            CuRobo's trajectory optimizer and graph planner can both reject a
            valid exact joint goal even when the complete direct joint segment
            is collision free.  This fallback is deliberately bounded and, in
            the harness-vendored revision, honest about its geometry: it is
            admitted on the UNWRAPPED joint delta (per joint
            |goal - start| <= min(pi, position limit span) and unwrapped L2
            <= 3.00 rad), so a remote elbow/wrist branch can never fold into
            an apparently nearby goal, and the executed endpoint is exactly
            the reported ik_joint_goal.  The complete segment is sampled at no
            more than 0.005 rad per joint through the same CuRobo
            world/self-collision and joint-limit rollout.  The controller
            receives a trajectory with no more than 0.025 rad per joint step,
            retimed against the robot's joint velocity limits so
            interpolation_dt_s and velocity are truthful.  The Harness still
            applies independent FK, workspace, dual-arm and
            measured-convergence admission to every returned waypoint.
            """

            started = time.time()
            start = np.asarray(
                start_joint_states.position.detach().to("cpu").numpy(),
                dtype=np.float64,
            ).reshape(-1)
            goal = np.asarray(goal, dtype=np.float64).reshape(-1)
            if (
                not np.isfinite(start).all()
                or not np.isfinite(goal).all()
                or start.shape != goal.shape
            ):
                return self._fallback_precondition_failure(
                    reason="non_finite_or_shape_mismatch",
                    detail={
                        "start_shape": list(start.shape),
                        "goal_shape": list(goal.shape),
                    },
                    elapsed_s=time.time() - started,
                )
            # Honest admission: fold the exact IK goal by whole revolutions
            # toward the start (staying inside position limits), then admit on
            # the UNWRAPPED delta to that normalized goal. The executed and
            # reported goal are the same normalized configuration, so remote
            # elbow/wrist branches cannot fold into apparently nearby goals
            # and the endpoint can never drift by 2*pi from its report.
            position_limits, velocity_limits = self._joint_limit_arrays()
            normalized_goal, wrap_turns = self._normalize_joint_goal(
                start, goal, position_limits,
            )
            if normalized_goal is None:
                return self._fallback_precondition_failure(
                    reason="wrapped_equivalent_goal_outside_position_limits",
                    detail={
                        "wrap_turns": wrap_turns.tolist(),
                        "wrapped_candidate_distance_rad": float(distance_rad),
                    },
                    elapsed_s=time.time() - started,
                )
            goal = normalized_goal
            delta = goal - start
            per_joint_cap = np.full(delta.shape, np.pi, dtype=np.float64)
            if position_limits is not None:
                per_joint_cap = np.minimum(
                    per_joint_cap, position_limits[1] - position_limits[0],
                )
            unwrapped_l2 = float(np.sqrt(np.sum(delta * delta)))
            if bool(np.any(np.abs(delta) > per_joint_cap)) or (
                unwrapped_l2 > float(maximum_distance_rad)
            ):
                return self._fallback_precondition_failure(
                    reason="unwrapped_delta_exceeds_envelope",
                    detail={
                        "unwrapped_l2_rad": unwrapped_l2,
                        "maximum_distance_rad": float(maximum_distance_rad),
                        "maximum_per_joint_delta_rad": float(
                            np.max(np.abs(delta)) if delta.size else 0.0
                        ),
                        "wrapped_candidate_distance_rad": float(distance_rad),
                        "wrap_turns": wrap_turns.tolist(),
                    },
                    elapsed_s=time.time() - started,
                )
            if velocity_limits is None:
                return self._fallback_precondition_failure(
                    reason="joint_velocity_limits_unavailable",
                    detail={"required_for": "honest_time_parameterization"},
                    elapsed_s=time.time() - started,
                )
            maximum_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
            validation_steps = max(
                2,
                int(np.ceil(maximum_delta / float(validation_joint_step_rad))) + 1,
            )
            validation_alpha = np.linspace(
                0.0, 1.0, validation_steps, dtype=np.float64,
            )[:, None]
            validation_position = (
                start[None, :] + validation_alpha * delta[None, :]
            )
            state = JointState.from_position(
                torch.tensor(validation_position, dtype=torch.float32).cuda(),
                joint_names=self.active_joints_name,
            )
            metrics = self.motion_gen.check_constraints(state)
            feasible = np.asarray(
                _planner_value(metrics.feasible), dtype=bool,
            ).reshape(-1)
            elapsed = time.time() - started
            segment_kind = (
                "nearby_fallback"
                if unwrapped_l2 <= float(nearby_distance_rad)
                else "bounded_exact_ik_segment"
            )
            failure_status = (
                "VALIDATED_LINEAR_CONSTRAINT_FAIL"
                if segment_kind == "nearby_fallback"
                else "VALIDATED_DENSE_JOINT_SEGMENT_CONSTRAINT_FAIL"
            )
            success_status = (
                "VALIDATED_LINEAR_JOINT_FALLBACK"
                if segment_kind == "nearby_fallback"
                else "VALIDATED_DENSE_JOINT_SEGMENT"
            )
            if (
                len(feasible) != validation_steps
                or not bool(np.all(feasible))
            ):
                return {
                    "schema": "robotwin.curobo-plan-result.v1",
                    "status": "Fail",
                    "planner_status": failure_status,
                    "valid_query": True,
                    "attempts": 1,
                    "trajopt_attempts": 0,
                    "used_graph": False,
                    "goalset_index": None,
                    "path_buffer_last_tstep": None,
                    "errors": {
                        "position": None, "rotation": None, "cspace": None,
                    },
                    "timings_s": {
                        "solve": elapsed, "ik": 0.0, "graph": 0.0,
                        "trajopt": 0.0, "finetune": 0.0, "total": elapsed,
                    },
                    "interpolation_dt_s": None,
                    "joint_names": list(self.active_joints_name),
                    "constraint_check": {
                        "backend": "curobo_motion_gen_check_constraints",
                        "segment_kind": segment_kind,
                        "samples": validation_steps,
                        "feasible_samples": int(np.count_nonzero(feasible)),
                        "nearby_distance_rad": float(nearby_distance_rad),
                        "maximum_distance_rad": float(maximum_distance_rad),
                        "validation_joint_step_rad": float(
                            validation_joint_step_rad
                        ),
                        "execution_joint_step_rad": float(
                            execution_joint_step_rad
                        ),
                    },
                }
            execution_steps = max(
                2,
                int(np.ceil(maximum_delta / float(execution_joint_step_rad))) + 1,
            )
            execution_alpha = np.linspace(
                0.0, 1.0, execution_steps, dtype=np.float64,
            )[:, None]
            position = start[None, :] + execution_alpha * delta[None, :]
            # The executed endpoint is exactly the admitted IK goal; the
            # unwrapped delta guarantees this algebraically and the explicit
            # assignment removes any floating-point endpoint drift.
            position[-1] = goal
            interpolation_dt_s, velocity, peak_step_rad = self._retime_joint_path(
                position, velocity_limits=velocity_limits,
            )
            return {
                "schema": "robotwin.curobo-plan-result.v1",
                "status": "Success",
                "planner_status": success_status,
                "valid_query": True,
                "attempts": 1,
                "trajopt_attempts": 0,
                "used_graph": False,
                "goalset_index": None,
                "path_buffer_last_tstep": execution_steps - 1,
                "errors": {
                    "position": None, "rotation": None, "cspace": 0.0,
                },
                "timings_s": {
                    "solve": elapsed, "ik": 0.0, "graph": 0.0,
                    "trajopt": 0.0, "finetune": 0.0, "total": elapsed,
                },
                "interpolation_dt_s": interpolation_dt_s,
                "joint_names": list(self.active_joints_name),
                "position": position,
                "velocity": velocity,
                "executed_joint_goal": goal.copy(),
                "constraint_check": {
                    "backend": "curobo_motion_gen_check_constraints",
                    "segment_kind": segment_kind,
                    "wrap_turns": wrap_turns.tolist(),
                    "samples": validation_steps,
                    "feasible_samples": validation_steps,
                    "execution_waypoints": execution_steps,
                    "nearby_distance_rad": float(nearby_distance_rad),
                    "maximum_distance_rad": float(maximum_distance_rad),
                    "unwrapped_l2_rad": unwrapped_l2,
                    "wrapped_candidate_distance_rad": float(distance_rad),
                    "endpoint_is_ik_joint_goal": True,
                    "validation_joint_step_rad": float(
                        validation_joint_step_rad
                    ),
                    "execution_joint_step_rad": float(
                        execution_joint_step_rad
                    ),
                    "time_parameterization": {
                        "source": "joint_velocity_limits",
                        "velocity_utilization": 0.5,
                        "peak_joint_step_rad": peak_step_rad,
                        "interpolation_dt_s": interpolation_dt_s,
                    },
                },
            }

        def _plan_pose_via_nearest_ik(self, start_joint_states, joint_angles, goal_pose):
            """Solve exact IK and expose every admitted top-K joint plan.

            This is a source-bound adaptation of CaP-X's public nearest-IK
            planning pattern (MIT, commit 53e9966).  It does not use CaP-X at
            runtime and does not fall back to a relaxed pose target.
            """
            seed_count = 128
            ik_config = IKSolverConfig.load_from_robot_config(
                self.motion_gen_config.robot_cfg,
                world_model=self.world_config,
                tensor_args=self.motion_gen.tensor_args,
                num_seeds=seed_count,
                position_threshold=0.005,
                rotation_threshold=0.05,
                self_collision_check=True,
            )
            ik_solver = IKSolver(ik_config)
            reference_tensor = torch.tensor(
                joint_angles, dtype=torch.float32,
            ).cuda().reshape(1, -1)
            ik_started = time.time()
            ik_result = ik_solver.solve_single(
                goal_pose,
                retract_config=reference_tensor,
                # CuRobo's near-state IK contract requires the measured
                # action-start joints as both the regularization target and
                # an explicit optimizer seed.  Random-only seeds occasionally
                # returned only remote elbow/wrist branches for a 4 cm
                # pre-contact return, even though the nearby branch was
                # reachable.  The remaining 127 seeds stay solver-generated,
                # so global configuration diversity is preserved.
                seed_config=reference_tensor.unsqueeze(0),
                return_seeds=seed_count,
            )
            ik_time_s = time.time() - ik_started
            success_values = np.asarray(
                _planner_value(ik_result.success), dtype=bool,
            ).reshape(-1)
            solutions = np.asarray(
                ik_result.solution.detach().to("cpu").numpy(), dtype=np.float64,
            ).reshape(-1, len(self.active_joints_name))
            finite = np.isfinite(solutions).all(axis=1)
            # Keep seed identities before any destructive filtering. The SDK
            # success mask combines convergence and constraints; a false bit
            # alone cannot identify collision, limits or pose residual.
            candidate_validation = [{
                "candidate_index": index,
                "finite": bool(finite[index]),
                "solver_success": (bool(success_values[index])
                    if len(success_values) == len(solutions) else None),
                "joint_limit_check": "unknown",
                "violating_joints": [],
                "rejection": "non_finite_solution" if not finite[index] else None,
                "solver_failure_detail": "unknown",
            } for index in range(len(solutions))]
            # Count the actual successive filters without changing admission.
            # The solver's success flag combines internal constraints; it does
            # not identify a particular collision or pose-tolerance cause.
            constraint_rejection_counts = {
                "non_finite_solution": int(np.count_nonzero(~finite)),
                "solver_unsuccessful": None,
                "joint_limit": None,
            }
            if len(success_values) == len(solutions):
                for index in np.flatnonzero(finite & ~success_values):
                    candidate_validation[int(index)]["rejection"] = "solver_unsuccessful"
                constraint_rejection_counts["solver_unsuccessful"] = int(
                    np.count_nonzero(finite & ~success_values)
                )
                finite &= success_values
            position_limits = None
            try:
                joint_limits = np.asarray(
                    _planner_value(
                        ik_solver.kinematics.get_joint_limits().position
                    ),
                    dtype=np.float64,
                )
                if joint_limits.shape == (2, solutions.shape[1]):
                    within_limits = np.all(
                        (solutions >= joint_limits[0] - 1e-6)
                        & (solutions <= joint_limits[1] + 1e-6),
                        axis=1,
                    )
                    constraint_rejection_counts["joint_limit"] = int(
                        np.count_nonzero(finite & ~within_limits)
                    )
                    for index in np.flatnonzero(finite):
                        row = candidate_validation[int(index)]
                        row["joint_limit_check"] = "passed" if within_limits[index] else "rejected"
                        if not within_limits[index]:
                            row["rejection"] = "joint_limit"
                            for joint, name in enumerate(self.active_joints_name):
                                value = float(solutions[index, joint])
                                lower, upper = map(float, joint_limits[:, joint])
                                if value < lower - 1e-6 or value > upper + 1e-6:
                                    row["violating_joints"].append({
                                        "joint": name, "value_rad": value,
                                        "lower_rad": lower, "upper_rad": upper,
                                        "excess_rad": max(lower - value, value - upper),
                                    })
                    finite &= within_limits
                    position_limits = joint_limits
            except (AttributeError, TypeError, ValueError):
                # IKSolver.success is already collision/limit checked.  The
                # explicit bound is retained when this pinned CuRobo build
                # exposes named limits, without inventing a fallback limit.
                pass
            valid = solutions[finite]
            for index, row in enumerate(candidate_validation):
                row["filter_outcome"] = "retained" if finite[index] else "rejected"
            valid_source_indices = np.flatnonzero(finite)
            # Placeholder-wide joint limits let exact IK legitimately return
            # 2*pi-aliased configurations; planning toward the raw alias either
            # fails trajopt spuriously or executes a physical extra revolution.
            # Fold every candidate toward the current configuration before
            # deduplication so the wrapped-distance clusters keep a near-branch
            # representative instead of whichever alias the seed order won.
            alias_folded_count = 0
            for row_index in range(len(valid)):
                normalized, turns = self._normalize_joint_goal(
                    joint_angles, valid[row_index], position_limits
                )
                if normalized is not None and bool(np.any(turns != 0)):
                    valid[row_index] = normalized
                    alias_folded_count += 1
            (
                candidates, source_indices, distances, deduplicated_count,
            ) = self._deduplicated_ik_candidates(valid, joint_angles)
            ik_summary = {
                "status": "Success" if len(candidates) else "Fail",
                "seed_count": seed_count,
                "returned_solution_count": int(len(solutions)),
                "constraint_rejection_counts": constraint_rejection_counts,
                "candidate_validation": candidate_validation,
                "planned_source_indices": [int(valid_source_indices[index]) for index in source_indices],
                "valid_solution_count": int(len(valid)),
                "deduplicated_solution_count": int(deduplicated_count),
                "planned_candidate_count": int(len(candidates)),
                "alias_folded_solution_count": int(alias_folded_count),
                "deduplication_threshold_rad": 0.05,
                "nearest_candidate_limit": 8,
                "diverse_candidate_limit": 8,
                "position_error": _planner_value(ik_result.position_error),
                "rotation_error": _planner_value(ik_result.rotation_error),
                "time_s": ik_time_s,
            }
            if not len(candidates):
                return {
                    "schema": "robotwin.curobo-plan-result.v1",
                    "status": "Fail",
                    "planner_status": "IK_FAIL",
                    "valid_query": True,
                    "attempts": 0,
                    "trajopt_attempts": 0,
                    "used_graph": False,
                    "goalset_index": None,
                    "path_buffer_last_tstep": None,
                    "errors": {
                        "position": _planner_value(ik_result.position_error),
                        "rotation": _planner_value(ik_result.rotation_error),
                        "cspace": None,
                    },
                    "timings_s": {
                        "solve": 0.0, "ik": ik_time_s, "graph": 0.0,
                        "trajopt": 0.0, "finetune": 0.0, "total": ik_time_s,
                    },
                    "interpolation_dt_s": None,
                    "joint_names": list(self.active_joints_name),
                    "ik": ik_summary,
                    "error": {
                        "stage": "ik", "type": "IKFailure",
                        "message": "multi-seed exact IK produced no legal solution",
                    },
                }
            records = []
            for candidate_index, (goal, source_index, distance) in enumerate(
                zip(candidates, source_indices, distances)
            ):
                try:
                    goal_state = JointState.from_position(
                        torch.tensor(goal, dtype=torch.float32).cuda().reshape(1, -1),
                        joint_names=self.active_joints_name,
                    )
                    result = self.motion_gen.plan_single_js(
                        start_joint_states,
                        goal_state,
                        MotionGenPlanConfig(
                            enable_graph=False, enable_opt=True, max_attempts=4,
                        ),
                    )
                    record = _motion_result(
                        result, joint_names=self.active_joints_name,
                    )
                    record["planning_attempt"] = "trajopt"
                    failed_attempts = []
                    if record["status"] != "Success":
                        failed_attempts.append({
                            "planning_attempt": record["planning_attempt"],
                            "planner_status": record.get("planner_status"),
                            "errors": record.get("errors"),
                            "timings_s": record.get("timings_s"),
                        })
                        fallback = self.motion_gen.plan_single_js(
                            start_joint_states,
                            goal_state,
                            MotionGenPlanConfig(
                                enable_graph=True, enable_opt=True,
                                enable_graph_attempt=2, max_attempts=10,
                            ),
                        )
                        fallback_record = _motion_result(
                            fallback, joint_names=self.active_joints_name,
                        )
                        fallback_record["planning_attempt"] = "graph_fallback"
                        record = fallback_record
                    if record["status"] != "Success":
                        failed_attempts.append({
                            "planning_attempt": record["planning_attempt"],
                            "planner_status": record.get("planner_status"),
                            "errors": record.get("errors"),
                            "timings_s": record.get("timings_s"),
                        })
                        # CuRobo can reject a geometrically valid very-short
                        # joint path only during time-optimal finetuning.  The
                        # non-finetuned trajectory remains collision and
                        # joint-limit checked, and the Harness still applies its
                        # independent FK, workspace and convergence admission.
                        fallback = self.motion_gen.plan_single_js(
                            start_joint_states,
                            goal_state,
                            MotionGenPlanConfig(
                                enable_graph=True, enable_opt=True,
                                enable_graph_attempt=2, max_attempts=10,
                                enable_finetune_trajopt=False,
                            ),
                        )
                        fallback_record = _motion_result(
                            fallback, joint_names=self.active_joints_name,
                        )
                        fallback_record["planning_attempt"] = (
                            "graph_fallback_without_finetune"
                        )
                        record = fallback_record
                    if record["status"] != "Success":
                        failed_attempts.append({
                            "planning_attempt": record["planning_attempt"],
                            "planner_status": record.get("planner_status"),
                            "errors": record.get("errors"),
                            "timings_s": record.get("timings_s"),
                        })
                        # A short exact joint goal may be graph-reachable even
                        # when both time-parameterized optimization passes reject
                        # it.  Keep the graph's collision-checked path instead of
                        # turning that optimizer limitation into a false
                        # reachability failure.  The Harness still independently
                        # admits every waypoint and verifies measured convergence.
                        fallback = self.motion_gen.plan_single_js(
                            start_joint_states,
                            goal_state,
                            MotionGenPlanConfig(
                                enable_graph=True,
                                enable_opt=False,
                                need_graph_success=True,
                                max_attempts=10,
                            ),
                        )
                        fallback_record = _motion_result(
                            fallback, joint_names=self.active_joints_name,
                        )
                        fallback_record["planning_attempt"] = "graph_only_fallback"
                        if fallback_record["status"] == "Success":
                            # A graph-only path is purely geometric: CuRobo stamps
                            # a fabricated optimized_dt and adjacent waypoints may
                            # jump arbitrarily far.  Densify to the execution step
                            # and retime against joint velocity limits so the
                            # returned trajectory is dynamically honest, or fail
                            # closed when limits are unavailable.
                            _, graph_velocity_limits = self._joint_limit_arrays()
                            if graph_velocity_limits is None:
                                fallback_record["status"] = "Fail"
                                fallback_record["planner_status"] = (
                                    "GRAPH_ONLY_RETIME_UNAVAILABLE"
                                )
                                fallback_record.pop("position", None)
                                fallback_record.pop("velocity", None)
                            else:
                                dense = self._densify_joint_path(
                                    fallback_record["position"],
                                    maximum_joint_step_rad=0.025,
                                )
                                (
                                    graph_dt, graph_velocity, graph_peak,
                                ) = self._retime_joint_path(
                                    dense, velocity_limits=graph_velocity_limits,
                                )
                                fallback_record["position"] = dense
                                fallback_record["velocity"] = graph_velocity
                                fallback_record["interpolation_dt_s"] = graph_dt
                                fallback_record["path_buffer_last_tstep"] = (
                                    len(dense) - 1
                                )
                                fallback_record["retimed"] = {
                                    "source": "joint_velocity_limits",
                                    "velocity_utilization": 0.5,
                                    "maximum_joint_step_rad": 0.025,
                                    "peak_joint_step_rad": graph_peak,
                                    "waypoints": int(len(dense)),
                                }
                        record = fallback_record
                    if record["status"] != "Success":
                        failed_attempts.append({
                            "planning_attempt": record["planning_attempt"],
                            "planner_status": record.get("planner_status"),
                            "errors": record.get("errors"),
                            "timings_s": record.get("timings_s"),
                        })
                        fallback_record = self._validated_linear_joint_fallback(
                            start_joint_states,
                            goal,
                            distance_rad=distance,
                        )
                        if fallback_record is not None:
                            fallback_status = fallback_record.get("planner_status")
                            if fallback_status in {
                                "VALIDATED_LINEAR_JOINT_FALLBACK",
                                "VALIDATED_LINEAR_CONSTRAINT_FAIL",
                            }:
                                fallback_record["planning_attempt"] = (
                                    "validated_linear_joint_fallback"
                                )
                            elif fallback_status == (
                                "VALIDATED_SEGMENT_PRECONDITION_FAIL"
                            ):
                                fallback_record["planning_attempt"] = (
                                    "validated_segment_precondition"
                                )
                            else:
                                fallback_record["planning_attempt"] = (
                                    "validated_dense_joint_segment"
                                )
                            record = fallback_record
                    if failed_attempts:
                        record["failed_attempts"] = failed_attempts
                except (AttributeError, ValueError, IndexError, KeyError, ArithmeticError) as exc:
                    # A malformed result for one independently seeded goal
                    # must not erase earlier completed trajectories. Device,
                    # allocator and transport RuntimeErrors are deliberately
                    # not recoverable here: their planner instance may be bad.
                    record = {
                        "schema": "robotwin.curobo-plan-result.v1",
                        "status": "Fail",
                        "planner_status": "PLANNER_ERROR",
                        "planning_attempt": "candidate_exception",
                        "valid_query": False,
                        "timings_s": {},
                        "error": {
                            "stage": "joint_planning",
                            "type": type(exc).__name__,
                            "message": str(exc)[:500],
                        },
                    }
                # A joint-segment fallback may have folded the raw IK goal by
                # whole revolutions; the reported goal must be the executed
                # configuration, never a 2*pi-offset alias of it.
                executed_goal = record.pop("executed_joint_goal", None)
                record.update({
                    "ik_candidate_index": candidate_index,
                    "ik_source_solution_index": int(source_index),
                    "ik_distance_rad": float(distance),
                    "ik_joint_goal": np.asarray(
                        goal if executed_goal is None else executed_goal,
                        dtype=np.float64,
                    ),
                })
                records.append(record)

            successful = [record for record in records if record["status"] == "Success"]
            candidate_errors = [
                record for record in records
                if record.get("planning_attempt") == "candidate_exception"
            ]
            # Native RoboTwin consumers (expert play_once, plan_batch length
            # comparison) read a single top-level trajectory. Expose the
            # shortest successful candidate's trajectory at the top level so
            # the installed checkout still runs the vanilla expert/oracle
            # paths; the Harness ignores these keys and performs its own
            # validated candidate selection over `candidates`.
            native_best = min(
                successful,
                key=lambda record: (
                    len(record["position"])
                    if record.get("position") is not None else float("inf"),
                    record["ik_candidate_index"],
                ),
            ) if successful else None
            native_projection = {
                "position": None, "velocity": None,
                "interpolation_dt_s": None, "path_buffer_last_tstep": None,
                "native_selected_candidate_index": None,
            }
            if native_best is not None and native_best.get("position") is not None:
                native_projection = {
                    "position": native_best["position"],
                    "velocity": native_best.get("velocity"),
                    "interpolation_dt_s": native_best.get("interpolation_dt_s"),
                    "path_buffer_last_tstep": native_best.get(
                        "path_buffer_last_tstep"
                    ),
                    "native_selected_candidate_index": native_best[
                        "ik_candidate_index"
                    ],
                }
            return {
                "schema": "robotwin.curobo-plan-result.v2",
                **native_projection,
                "status": "Success" if successful else "Fail",
                "planner_status": (
                    "TOP_K_PLAN_CANDIDATES" if successful
                    else "PLANNER_ERROR" if candidate_errors
                    else "TOP_K_PLANNING_FAILED"
                ),
                # Success provenance is surfaced at the top level so a
                # consumer reading only the summary can see whether success
                # came from time-parameterized planning or a bounded fallback.
                "success_planning_attempts": sorted({
                    str(record.get("planning_attempt", "unknown"))
                    for record in successful
                }),
                "valid_query": bool(successful) or not candidate_errors,
                "candidate_error_count": len(candidate_errors),
                "joint_names": list(self.active_joints_name),
                "ik": ik_summary,
                "candidates": records,
                "error": (
                    None if successful else candidate_errors[0]["error"] if candidate_errors else {
                        "stage": "joint_planning", "type": "PlanningFailure",
                        "message": "all exact top-K IK joint goals failed",
                    }
                ),
                "timings_s": {
                    "ik": ik_time_s,
                    "candidate_total": float(sum(
                        float(record.get("timings_s", {}).get("total", 0.0))
                        for record in records
                    )),
                },
            }

        def plan_path(
            self,
            curr_joint_pos,
            target_gripper_pose,
            constraint_pose=None,
            arms_tag=None,
        ):
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
            target_pose_p, target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)
            if not ("aloha-agilex" in self.yml_path):
                target_pose_p[0] += self.frame_bias[0]
                target_pose_p[1] += self.frame_bias[1]
                target_pose_p[2] += self.frame_bias[2]
            else: # patch for aloha-agilex
                T_target = t3d.affines.compose(target_pose_p, t3d.quaternions.quat2mat(target_pose_q), [1, 1, 1])
                T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                if arms_tag == "left":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                elif arms_tag == "right":
                    rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                else:
                    raise ValueError(f"Invalid arms_tag: {arms_tag}")

                T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                T_new = T_rot @ T_bias @ T_target
                target_pose_p = T_new[:3, 3]
                target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

            goal_pose_of_ee = CuroboPose.from_list(list(target_pose_p) + list(target_pose_q))
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            start_joint_states = JointState.from_position(
                torch.tensor(joint_angles).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            if constraint_pose is not None:
                # Native expert paths (grasp_actor and friends) request a
                # held partial pose along the motion. Joint-space nearest-IK
                # planning cannot express that path constraint, so constrained
                # queries use the upstream pose-goal flow with CuRobo's
                # PoseCostMetric, exactly as vanilla c3ddfa8 did. The Harness
                # never issues constrained queries; this branch exists only so
                # the installed checkout still runs the native expert/oracle.
                plan_config = MotionGenPlanConfig(max_attempts=10)
                plan_config.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(
                        constraint_pose
                    ),
                )
                result = self.motion_gen.plan_single(
                    start_joint_states, goal_pose_of_ee, plan_config,
                )
                record = _motion_result(
                    result, joint_names=self.active_joints_name,
                )
                record["planning_attempt"] = "constrained_pose_goal"
                return record
            return self._plan_pose_via_nearest_ik(
                start_joint_states, joint_angles, goal_pose_of_ee,
            )

        def plan_joint_path(self, curr_joint_pos, target_joint_pos):
            """Plan to one named active-joint configuration with full diagnostics."""
            joint_indices = [
                self.all_joints.index(name)
                for name in self.active_joints_name
                if name in self.all_joints
            ]
            start = [round(float(curr_joint_pos[index]), 5) for index in joint_indices]
            target = [float(value) for value in target_joint_pos]
            if len(target) != len(self.active_joints_name):
                raise ValueError("target_joint_pos does not match active joint order")
            start_state = JointState.from_position(
                torch.tensor(start, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            goal_state = JointState.from_position(
                torch.tensor(target, dtype=torch.float32).cuda().reshape(1, -1),
                joint_names=self.active_joints_name,
            )
            result = self.motion_gen.plan_single_js(
                start_state, goal_state, MotionGenPlanConfig(max_attempts=10)
            )
            return _motion_result(result, joint_names=self.active_joints_name)

        def _plan_batch_pose_direct_legacy(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """
            Plan a batch of trajectories for multiple target poses.

            Input:
                - curr_joint_pos: List of current joint angles (1 x n)
                - target_gripper_pose_list: List of target poses [sapien.Pose, sapien.Pose, ...]

            Output:
                - result['status']: numpy array of string values indicating "Success"/"Fail" for each pose
                - result['position']: numpy array of joint positions with shape (n x m x l)
                  where n is number of target poses, m is number of waypoints, l is number of joints
                - result['velocity']: numpy array of joint velocities with same shape as position
            """

            num_poses = len(target_gripper_pose_list)
            # transformation from world to arm's base
            world_base_pose = np.concatenate([
                np.array(self.robot_origion_pose.p),
                np.array(self.robot_origion_pose.q),
            ])
            poses_list = []
            for target_gripper_pose in target_gripper_pose_list:
                world_target_pose = np.concatenate([np.array(target_gripper_pose.p), np.array(target_gripper_pose.q)])
                base_target_pose_p, base_target_pose_q = self._trans_from_world_to_base(world_base_pose, world_target_pose)

                if not ("aloha-agilex" in self.yml_path):
                    base_target_pose_p[0] += self.frame_bias[0]
                    base_target_pose_p[1] += self.frame_bias[1]
                    base_target_pose_p[2] += self.frame_bias[2]
                else: # patch for aloha-agilex
                    T_target = t3d.affines.compose(base_target_pose_p, t3d.quaternions.quat2mat(base_target_pose_q), [1, 1, 1])
                    T_bias = t3d.affines.compose(self.frame_bias, np.eye(3), [1, 1, 1])

                    if arms_tag == "left":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.02)
                    elif arms_tag == "right":
                        rot = t3d.axangles.axangle2mat([0, 0, 1], -0.01)
                    else:
                        raise ValueError(f"Invalid arms_tag: {arms_tag}")

                    T_rot = t3d.affines.compose([0, 0, 0], rot, [1, 1, 1])
                    T_new = T_rot @ T_bias @ T_target
                    base_target_pose_p = T_new[:3, 3]
                    base_target_pose_q = t3d.quaternions.mat2quat(T_new[:3, :3])

                base_target_pose_list = list(base_target_pose_p) + list(base_target_pose_q)
                poses_list.append(base_target_pose_list)

            poses_cuda = torch.tensor(poses_list, dtype=torch.float32).cuda()
            goal_pose_of_ee = CuroboPose(poses_cuda[:, :3], poses_cuda[:, 3:])
            joint_indices = [self.all_joints.index(name) for name in self.active_joints_name if name in self.all_joints]
            joint_angles = [curr_joint_pos[index] for index in joint_indices]
            joint_angles = [round(angle, 5) for angle in joint_angles]  # avoid the precision problem
            joint_angles_cuda = (torch.tensor(joint_angles, dtype=torch.float32).cuda().reshape(1, -1))
            joint_angles_cuda = torch.cat([joint_angles_cuda] * num_poses, dim=0)
            start_joint_states = JointState.from_position(joint_angles_cuda, joint_names=self.active_joints_name)
            # plan
            plan_config = MotionGenPlanConfig(max_attempts=10)
            if constraint_pose is not None:
                pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(constraint_pose),
                )
                plan_config.pose_cost_metric = pose_cost_metric

            result = self.motion_gen_batch.plan_batch(
                start_joint_states, goal_pose_of_ee, plan_config
            )
            return _motion_result(
                result, joint_names=self.active_joints_name, preserve_batch=True,
            )

        def plan_batch(
            self,
            curr_joint_pos,
            target_gripper_pose_list,
            constraint_pose=None,
            arms_tag=None,
        ):
            """Plan all explicit candidates through exact nearest-IK joint goals.

            Candidate ordering is preserved.  This is a single source RPC and
            does not invent orientation slack or parallel-jaw variants; those
            must be explicit in the caller's candidate list.
            """
            records = [
                self.plan_path(
                    curr_joint_pos,
                    target,
                    constraint_pose=constraint_pose,
                    arms_tag=arms_tag,
                )
                for target in target_gripper_pose_list
            ]
            return _batch_motion_records(records)

        def plan_grippers(self, now_val, target_val):
            num_step = 200
            dis_val = target_val - now_val
            step = dis_val / num_step
            res = {}
            vals = np.linspace(now_val, target_val, num_step)
            res["num_step"] = num_step
            res["per_step"] = step
            res["result"] = vals
            return res

        def _trans_from_world_to_base(self, base_pose, target_pose):
            '''
                transform target pose from world frame to base frame
                base_pose: np.array([x, y, z, qw, qx, qy, qz])
                target_pose: np.array([x, y, z, qw, qx, qy, qz])
            '''
            base_p, base_q = base_pose[0:3], base_pose[3:]
            target_p, target_q = target_pose[0:3], target_pose[3:]
            rel_p = target_p - base_p
            wRb = t3d.quaternions.quat2mat(base_q)
            wRt = t3d.quaternions.quat2mat(target_q)
            result_p = wRb.T @ rel_p
            result_q = t3d.quaternions.mat2quat(wRb.T @ wRt)
            return result_p, result_q

except Exception as e:
    print('[planner.py]: Something wrong happened when importing CuroboPlanner! Please check if Curobo is installed correctly. If the problem still exists, you can install Curobo from https://github.com/NVlabs/curobo manually.')
    print('Exception traceback:')
    traceback.print_exc()


# ********************** MplibPlanner **********************
class MplibPlanner:
    # links=None, joints=None
    def __init__(
        self,
        urdf_path,
        srdf_path,
        move_group,
        robot_origion_pose,
        robot_entity,
        planner_type="mplib_RRT",
        scene=None,
    ):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging

        links = [link.get_name() for link in robot_entity.get_links()]
        joints = [joint.get_name() for joint in robot_entity.get_active_joints()]

        if scene is None:
            self.planner = mplib.Planner(
                urdf=urdf_path,
                srdf=srdf_path,
                move_group=move_group,
                user_link_names=links,
                user_joint_names=joints,
                use_convex=False,
            )
            self.planner.set_base_pose(robot_origion_pose)
        else:
            planning_world = SapienPlanningWorld(scene, [robot_entity])
            self.planner = SapienPlanner(planning_world, move_group)

        self.planner_type = planner_type
        self.plan_step_lim = 2500
        self.TOPP = self.planner.TOPP

    def show_info(self):
        print("joint_limits", self.planner.joint_limits)
        print("joint_acc_limits", self.planner.joint_acc_limits)

    def plan_pose(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        try_times=2,
        log=True,
    ):
        result = {}
        result["status"] = "Fail"

        now_try_times = 1
        while result["status"] != "Success" and now_try_times < try_times:
            result = self.planner.plan_pose(
                goal_pose=target_pose,
                current_qpos=np.array(now_qpos),
                time_step=1 / 250,
                planning_time=5,
                # rrt_range=0.05
                # =================== mplib 0.1.1 ===================
                # use_point_cloud=use_point_cloud,
                # use_attach=use_attach,
                # planner_name="RRTConnect"
            )
            now_try_times += 1

        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
        else:
            n_step = result["position"].shape[0]
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_screw(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=False,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        result = self.planner.plan_screw(
            goal_pose=target_pose,
            current_qpos=now_qpos,
            time_step=1 / 250,
            # =================== mplib 0.1.1 ===================
            # use_point_cloud=use_point_cloud,
            # use_attach=use_attach,
        )

        # plan fail
        if result["status"] != "Success":
            if log:
                print(f"\n {arms_tag} arm planning failed ({result['status']}) !")
            # return result
        else:
            n_step = result["position"].shape[0]
            # plan step lim
            if n_step > self.plan_step_lim:
                if log:
                    print(f"\n {arms_tag} arm planning wrong! (step = {n_step})")
                result["status"] = "Fail"

        return result

    def plan_path(
        self,
        now_qpos,
        target_pose,
        use_point_cloud=False,
        use_attach=False,
        arms_tag=None,
        log=True,
    ):
        """
        Interpolative planning with screw motion.
        Will not avoid collision and will fail if the path contains collision.
        """
        if self.planner_type == "mplib_RRT":
            result = self.plan_pose(
                now_qpos,
                target_pose,
                use_point_cloud,
                use_attach,
                arms_tag,
                try_times=10,
                log=log,
            )
        elif self.planner_type == "mplib_screw":
            result = self.plan_screw(now_qpos, target_pose, use_point_cloud, use_attach, arms_tag, log)

        return result

    def plan_grippers(self, now_val, target_val):
        num_step = 200  # TODO
        dis_val = target_val - now_val
        per_step = dis_val / num_step
        res = {}
        vals = np.linspace(now_val, target_val, num_step)
        res["num_step"] = num_step
        res["per_step"] = per_step  # dis per step
        res["result"] = vals
        return res
