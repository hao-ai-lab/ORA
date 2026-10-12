"""The worker-side environment wrapper behind RLinf's getattr channel.

``_CameraMetadataEnvironment`` wraps the constructed LIBERO environment inside
the isolated worker and exposes controller-owned public geometry, sensor state
and the bounded joint-trajectory, replay and Servo controls as ``harness_*``
attributes. ``session`` re-exports both names.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

from ora.environment.controllers.hybrid_motion import (
    bounded_joint_position_command,
    libero_global_endpoint_tolerances,
    motion_stop_reason,
    precision_force_stop_enabled,
    validate_hybrid_motion_profile,
)
from ora.environment.controllers.force_feedback import (
    sensor_reaction_feedback,
)

from .execution_capture import LiberoWorkerCaptureMixin
from .common import LiberoProError, _quaternion_error_rotvec


class _CameraMetadataEnvironment(LiberoWorkerCaptureMixin):
    """Expose controller-owned public geometry through RLinf's getattr channel."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._hybrid_motion_profile: dict[str, Any] | None = None
        self._replay_controller_mode = "OSC_POSE"
        self._replay_step_result: dict[str, Any] | None = None
        self._joint_execution_observation: dict[str, Any] | None = None
        self._joint_execution_status: dict[str, Any] | None = None
        self._init_execution_capture()


    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._capture_active = False
        result = self._delegate.reset(*args, **kwargs)
        # A trusted round reset ends the old replay trajectory. Keep the
        # actual selected controller mode, but never carry its old terminal,
        # observation or joint-step result into the fresh world.
        self._replay_step_result = None
        self._replay_observation = None
        self._joint_execution_observation = None
        self._joint_execution_status = None
        return result

    @property
    def unwrapped(self) -> "_CameraMetadataEnvironment":
        # RLinf's worker applies set_env_attr to ``env.unwrapped``.  Keeping the
        # trusted wrapper as that target lets the Harness synchronize controller
        # state without exposing an arbitrary backend mutation surface.
        return self

    @property
    def harness_camera_meta(self) -> dict[str, Any]:
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
        )

        sim = self._delegate.sim
        extent = float(sim.model.stat.extent)
        near = float(sim.model.vis.map.znear) * extent
        far = float(sim.model.vis.map.zfar) * extent
        height = 256
        width = 256
        result: dict[str, Any] = {}
        for name in ("agentview", "robot0_eye_in_hand"):
            result[name] = {
                "intrinsic_K": get_camera_intrinsic_matrix(
                    sim, name, height, width,
                ),
                "extrinsic_cam2world": get_camera_extrinsic_matrix(sim, name),
                "depth_near": near,
                "depth_far": far,
                "height": height,
                "width": width,
            }
        return result

    @property
    def harness_robot_geometry(self) -> dict[str, Any]:
        from .robot_asset_geometry import panda_geometry_from_model

        robots = getattr(self._delegate, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) != 1:
            raise LiberoProError("public geometry requires one loaded Panda")
        prefix = robots[0].gripper.naming_prefix
        sim = self._delegate.sim
        return {"right_gripper": panda_geometry_from_model(
            getattr(sim.model, "_model", sim.model),
            getattr(sim.data, "_data", sim.data), prefix=prefix,
        )}

    @property
    def harness_grip_site_pose(self) -> dict[str, Any]:
        """Return the exact site pose used by RoboSuite's OSC controller.

        ``robot0_eef_pos`` is a grip-site position while ``robot0_eef_quat`` is
        the gripper hand-body orientation in RoboSuite 1.4.  Combining them
        creates a pose that belongs to no rigid body.  Reading both components
        from the controller's ``grip_site`` keeps the public pose and local
        action frames on one physical frame.
        """

        from robosuite.utils.transform_utils import mat2quat

        robots = getattr(self._delegate, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) != 1:
            raise LiberoProError("LIBERO environment must expose one robot")
        robot = robots[0]
        gripper = getattr(robot, "gripper", None)
        sites = getattr(gripper, "important_sites", None)
        site_name = sites.get("grip_site") if isinstance(sites, Mapping) else None
        if not isinstance(site_name, str) or not site_name:
            raise LiberoProError("Panda gripper does not expose its grip_site")
        sim = self._delegate.sim
        site_id = int(sim.model.site_name2id(site_name))
        position = sim.data.site_xpos[site_id].copy()
        rotation = sim.data.site_xmat[site_id].reshape(3, 3).copy()
        orientation = mat2quat(rotation)
        return {
            "frame": "robosuite_grip_site",
            "site_name": site_name,
            "position_m": position,
            "orientation_xyzw": orientation,
        }

    @property
    def harness_gripper_hold_directive(self) -> float:
        robots = getattr(self._delegate, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) != 1:
            raise LiberoProError("LIBERO environment must expose one robot")
        current = getattr(robots[0].gripper, "current_action", None)
        if current is None or len(current) != 2:
            raise LiberoProError("Panda gripper directive is unavailable")
        return float((float(current[1]) - float(current[0])) / 2.0)

    @property
    def harness_servo_timing(self) -> dict[str, Any]:
        # LIBERO's ControlEnv exposes the robot/sim but keeps RoboSuite's
        # timing on its inner ``env``. Read the actual controller owner;
        # otherwise RLinf's hasattr/getattr channel turns AttributeError
        # into a misleading None result.
        owner = self._delegate
        seen: set[int] = set()
        while not hasattr(owner, "model_timestep"):
            if id(owner) in seen or not hasattr(owner, "env"):
                raise LiberoProError("native Servo timing owner is unavailable")
            seen.add(id(owner))
            owner = owner.env
        dt = float(owner.model_timestep)
        control_dt = float(owner.control_timestep)
        if not math.isfinite(dt) or not math.isfinite(control_dt) or dt <= 0 or control_dt <= 0:
            raise LiberoProError("invalid native Servo timing")
        quantum = int(control_dt / dt)
        if quantum < 1:
            raise LiberoProError("invalid native Servo timing")
        return {"physics_dt_s": dt, "native_quantum_physics_steps": quantum}

    @property
    def harness_motion_state(self) -> dict[str, Any]:
        return self._sensor_motion_state(
            force_feedback_enabled=precision_force_stop_enabled(self._hybrid_motion_profile),
        )

    def _sensor_motion_state(self, *, force_feedback_enabled: bool) -> dict[str, Any]:
        """Return only sensor-equivalent robot state and rigid-frame calibration."""

        from robosuite.utils.transform_utils import mat2quat
        import numpy as np

        robots = getattr(self._delegate, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) != 1:
            raise LiberoProError("LIBERO environment must expose one robot")
        robot = robots[0]
        sim = self._delegate.sim
        root_name = str(robot.robot_model.root_body)
        hand_name = str(robot.robot_model.eef_name)
        root_id = int(sim.model.body_name2id(root_name))
        hand_id = int(sim.model.body_name2id(hand_name))
        force_torque = getattr(robot, "ee_force", None), getattr(robot, "ee_torque", None)
        wrench = None
        force_magnitude_n = None
        force_projection: dict[str, Any] = {}
        if force_feedback_enabled:
            force_projection = sensor_reaction_feedback(None, None)
        if all(item is not None for item in force_torque):
            candidate = np.concatenate(force_torque).astype(np.float64)
            if candidate.shape == (6,) and np.isfinite(candidate).all():
                force_magnitude_n = float(np.linalg.norm(candidate[:3]))
                profile = self._hybrid_motion_profile
                force_clip = (
                    float(profile["sensing"]["wrench_clip_force_n"])
                    if profile is not None else 100.0
                )
                torque_clip = (
                    float(profile["sensing"]["wrench_clip_torque_nm"])
                    if profile is not None else 100.0
                )
                wrench = np.concatenate([
                    np.clip(candidate[:3], -force_clip, force_clip),
                    np.clip(candidate[3:], -torque_clip, torque_clip),
                ]).tolist()
                if force_feedback_enabled:
                    # RoboSuite ee_force is the named force_ee sensor reading.
                    # MuJoCo defines it in that sensor's site frame, child to
                    # parent. Do not substitute the hand or camera rotation.
                    try:
                        sensor_name = robot.gripper.important_sensors["force_ee"]
                        sensor_id = int(sim.model.sensor_name2id(sensor_name))
                        site_id = int(sim.model.sensor_objid[sensor_id])
                        rotation = sim.data.site_xmat[site_id].reshape(3, 3)
                        source = {
                            "sensor_name": sensor_name,
                            "sensor_site_id": site_id,
                            "sensor_type": int(sim.model.sensor_type[sensor_id]),
                            "raw_force_sensor_n": candidate[:3].tolist(),
                            "world_from_sensor_rotation": rotation.tolist(),
                            "raw_convention": "mujoco_site_child_to_parent",
                        }
                        # mjSENS_FORCE=4 and mjOBJ_SITE=6 in the pinned MuJoCo
                        # enum; ask the installed enum instead of assuming IDs.
                        from mujoco import mjtObj, mjtSensor
                        if (source["sensor_type"] != int(mjtSensor.mjSENS_FORCE)
                                or int(sim.model.sensor_objtype[sensor_id]) != int(mjtObj.mjOBJ_SITE)):
                            raise ValueError("force sensor is not bound to a site")
                        force_projection = sensor_reaction_feedback(candidate[:3], rotation)
                        source["sha256"] = hashlib.sha256(json.dumps(
                            source, sort_keys=True, separators=(",", ":"),
                        ).encode()).hexdigest()
                        force_projection["force_source_binding"] = source
                    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
                        force_projection = sensor_reaction_feedback(None, None)
        return {
            "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
            "joint_positions": np.asarray(robot._joint_positions, dtype=np.float64).tolist(),
            "joint_velocities": np.asarray(robot._joint_velocities, dtype=np.float64).tolist(),
            "robot_joint_names_source": list(robot.robot_joints),
            "base_pose_world_xyzw": (
                sim.data.body_xpos[root_id].tolist()
                + mat2quat(sim.data.body_xmat[root_id].reshape(3, 3)).tolist()
            ),
            "hand_pose_world_xyzw": (
                sim.data.body_xpos[hand_id].tolist()
                + mat2quat(sim.data.body_xmat[hand_id].reshape(3, 3)).tolist()
            ),
            "grip_site_pose_world_xyzw": (
                self.harness_grip_site_pose["position_m"].tolist()
                + self.harness_grip_site_pose["orientation_xyzw"].tolist()
            ),
            "external_wrench_clipped": wrench,
            "external_force_magnitude_n": force_magnitude_n,
            **force_projection,
        }

    @property
    def harness_joint_execution_result(self) -> dict[str, Any]:
        result = getattr(self, "_joint_execution_result", None)
        if not isinstance(result, Mapping):
            raise LiberoProError("joint trajectory has not been executed")
        return dict(result)

    @property
    def harness_joint_execution_status(self) -> dict[str, Any]:
        """Return termination even if later observation/controller work failed."""

        if self._joint_execution_status is None:
            raise LiberoProError("joint trajectory has not been dispatched")
        return dict(self._joint_execution_status)

    @property
    def harness_joint_execution_observation(self) -> dict[str, Any]:
        """Transport the last actual step observation without advancing physics."""

        if self._joint_execution_observation is None:
            raise LiberoProError("joint trajectory has no executed observation")
        return dict(self._joint_execution_observation)

    @property
    def harness_motion_profile(self) -> dict[str, Any] | None:
        return (
            None if self._hybrid_motion_profile is None
            else dict(self._hybrid_motion_profile)
        )

    @harness_motion_profile.setter
    def harness_motion_profile(self, value: Any) -> None:
        self._hybrid_motion_profile = validate_hybrid_motion_profile(value)

    @property
    def harness_replay_controller_mode(self) -> str:
        return self._replay_controller_mode

    @harness_replay_controller_mode.setter
    def harness_replay_controller_mode(self, value: Any) -> None:
        """Switch only the isolated derived-replay controller boundary."""

        from robosuite.controllers import load_controller_config

        if value not in {"OSC_POSE", "JOINT_POSITION"}:
            raise LiberoProError("replay controller mode is invalid")
        # Ordinary live OSC actions retain their controller between actions.
        # Reloading the same mode here resets hidden controller goal state.
        if value == self._replay_controller_mode:
            return
        robots = getattr(self._delegate, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) != 1:
            raise LiberoProError("LIBERO environment must expose one robot")
        robot = robots[0]
        robot.controller_config = load_controller_config(default_controller=value)
        robot._load_controller()
        owner = self._delegate
        while not hasattr(owner, "_action_dim") and hasattr(owner, "env"):
            owner = owner.env
        if not hasattr(owner, "_action_dim"):
            raise LiberoProError("RoboSuite action dimension owner is unavailable")
        owner._action_dim = int(robot.action_dim)
        self._replay_controller_mode = str(value)

    @property
    def harness_replay_step_result(self) -> dict[str, Any] | None:
        return (
            None if self._replay_step_result is None
            else dict(self._replay_step_result)
        )

    @property
    def harness_replay_observation(self) -> dict[str, Any]:
        """Return the last replay step's camera/state sample without stepping."""
        observation = getattr(self, "_replay_observation", None)
        if not isinstance(observation, Mapping):
            raise LiberoProError("replay observation is unavailable")
        return dict(observation)

    @property
    def harness_replay_command(self) -> None:
        return None

    @harness_replay_command.setter
    def harness_replay_command(self, value: Any) -> None:
        """Apply one recorded command without invoking a planner or policy."""

        import numpy as np

        command = np.asarray(value, dtype=np.float64)
        expected = 8 if self._replay_controller_mode == "JOINT_POSITION" else 7
        if command.shape != (expected,) or not np.isfinite(command).all():
            raise LiberoProError("recorded replay command is malformed")
        result = self.step(command)
        if not isinstance(result, tuple) or len(result) not in {4, 5}:
            raise LiberoProError("recorded replay step result is malformed")
        if len(result) == 5:
            _obs, _reward, terminated, truncated, _info = result
        else:
            _obs, _reward, terminated, _info = result
            truncated = False
        self._replay_step_result = {
            "terminated": bool(np.asarray(terminated).any()),
            "truncated": bool(np.asarray(truncated).any()),
        }
        if not isinstance(_obs, Mapping):
            raise LiberoProError("recorded replay observation is malformed")
        self._replay_observation = dict(_obs)

    @property
    def harness_joint_trajectory_command(self) -> None:
        return None

    @harness_joint_trajectory_command.setter
    def harness_joint_trajectory_command(self, value: Any) -> None:
        """Execute a frozen CuRobo trajectory through RoboSuite JOINT_POSITION."""

        import numpy as np
        from robosuite.controllers import load_controller_config

        if not isinstance(value, Mapping):
            raise LiberoProError("joint trajectory command must be an object")
        execution_id = value.get("execution_id")
        self._joint_execution_result = None
        self._joint_execution_observation = None
        self._joint_execution_status = {
            "execution_id": execution_id,
            "steps_used": 0, "terminated": False, "truncated": False,
        }
        positions = np.asarray(value.get("positions"), dtype=np.float64)
        profile = self._hybrid_motion_profile
        if profile is None:
            raise LiberoProError("joint execution requires a bound motion profile")
        maximum_waypoints = int(profile["global_motion"]["max_waypoints"])
        if (
            positions.ndim != 2 or positions.shape[1] != 7
            or not 1 <= len(positions) <= maximum_waypoints
            or not np.isfinite(positions).all()
        ):
            raise LiberoProError("joint trajectory positions are malformed")
        stop_request_path = Path(str(value.get("stop_request_path", ""))).resolve()
        robots = getattr(self._delegate, "robots", None)
        if not isinstance(robots, (list, tuple)) or len(robots) != 1:
            raise LiberoProError("LIBERO environment must expose one robot")
        robot = robots[0]
        # Active controller configs have already been enriched with MuJoCo
        # pointers and cannot be copied or sent across a process boundary.
        # Reload both immutable source configs for each bounded switch.
        osc_config = load_controller_config(default_controller="OSC_POSE")
        measured = []
        issued_commands = []
        waypoint_ticks: list[int] = []
        terminal = False
        truncated = False
        stop_reason = None
        reached_waypoints = 0
        active_goal_index = 0
        global_motion = profile["global_motion"]
        waypoint_tolerance = float(
            global_motion["waypoint_tracking_tolerance_rad"]
        )
        command_step = float(global_motion["joint_command_step_rad"])
        max_ticks_per_waypoint = int(
            global_motion["max_ticks_per_waypoint"]
        )
        max_terminal_ticks = int(
            global_motion["max_terminal_settle_ticks"]
        )
        max_execution_ticks = int(global_motion["max_execution_ticks"])
        terminal_tracking_limit = float(
            global_motion["joint_tracking_error_limit_rad"]
        )
        settle_joint_tolerance = global_motion.get("terminal_settle_joint_tolerance_rad")
        target_pose = value.get("target_pose_world_xyzw")
        if settle_joint_tolerance is not None:
            target_pose = np.asarray(target_pose, dtype=np.float64)
            if (target_pose.shape != (7,) or not np.isfinite(target_pose).all()
                    or not math.isclose(float(np.linalg.norm(target_pose[3:])), 1.0, abs_tol=1e-6)):
                raise LiberoProError("v2 joint settling requires its source-bound TCP target")
        execution_phase = "waypoint_tracking"
        settle_ticks = 0
        settle_endpoint_error = None
        settle_exhausted = False
        action_owner = self._delegate
        while not hasattr(action_owner, "_action_dim") and hasattr(action_owner, "env"):
            action_owner = action_owner.env
        if not hasattr(action_owner, "_action_dim"):
            raise LiberoProError("RoboSuite action dimension owner is unavailable")
        previous_action_dim = int(action_owner._action_dim)
        try:
            robot.controller_config = load_controller_config(
                default_controller="JOINT_POSITION"
            )
            robot._load_controller()
            action_owner._action_dim = int(robot.action_dim)

            def execute_tick(target: Any) -> bool:
                nonlocal terminal, truncated, stop_reason
                stop_reason = motion_stop_reason(
                    stop_request_path=stop_request_path,
                )
                if stop_reason is not None:
                    return False
                if len(measured) >= max_execution_ticks:
                    stop_reason = "execution_tick_budget_exhausted"
                    return False
                current = np.asarray(robot._joint_positions, dtype=np.float64)
                arm = bounded_joint_position_command(
                    current.tolist(), np.asarray(target).tolist(),
                    maximum_step_rad=command_step,
                )
                command = np.concatenate([
                    np.asarray(arm, dtype=np.float64), np.asarray([0.0]),
                ])
                issued_commands.append(command.tolist())
                result = self.step(command)
                if not isinstance(result, tuple) or len(result) not in {4, 5}:
                    raise LiberoProError("RoboSuite joint step result is malformed")
                if len(result) == 5:
                    _obs, _reward, term, trunc, _info = result
                else:
                    _obs, _reward, term, _info = result
                    trunc = False
                terminal = terminal or bool(np.asarray(term).any())
                truncated = truncated or bool(np.asarray(trunc).any())
                # The official result is authoritative before any subsequent
                # measurement, image transport or controller restoration.
                self._joint_execution_status = {
                    "execution_id": execution_id,
                    "steps_used": len(issued_commands),
                    "terminated": terminal, "truncated": truncated,
                }
                if not isinstance(_obs, Mapping):
                    raise LiberoProError("joint step observation is malformed")
                self._joint_execution_observation = dict(_obs)
                measured.append(
                    np.asarray(robot._joint_positions, dtype=np.float64).tolist()
                )
                if terminal or truncated:
                    stop_reason = (
                        "environment_truncated" if truncated
                        else "environment_terminated"
                    )
                    return False
                return True

            for index, target in enumerate(positions):
                active_goal_index = index
                # Poll trusted interruption even when the next frozen waypoint
                # is already inside tolerance and would require zero plant ticks.
                stop_reason = motion_stop_reason(
                    stop_request_path=stop_request_path,
                )
                if stop_reason is not None:
                    break
                ticks = 0
                while float(np.max(np.abs(
                    target - np.asarray(robot._joint_positions, dtype=np.float64)
                ))) > waypoint_tolerance:
                    if ticks >= max_ticks_per_waypoint:
                        stop_reason = "waypoint_tracking_timeout"
                        break
                    before_ticks = len(measured)
                    continuing = execute_tick(target)
                    ticks += len(measured) - before_ticks
                    if not continuing:
                        break
                waypoint_ticks.append(ticks)
                if stop_reason is not None:
                    break
                reached_waypoints += 1

            if stop_reason is None and reached_waypoints == len(positions):
                execution_phase = "terminal_settle"
                final_goal = positions[-1]

                def terminal_settled() -> bool:
                    nonlocal settle_endpoint_error
                    joint_delta = final_goal - np.asarray(robot._joint_positions, dtype=np.float64)
                    if settle_joint_tolerance is None:
                        return float(np.linalg.norm(joint_delta)) <= terminal_tracking_limit
                    actual = self.harness_grip_site_pose
                    position_error = float(np.linalg.norm(target_pose[:3] - actual["position_m"]))
                    orientation_error = float(np.linalg.norm(_quaternion_error_rotvec(
                        target_pose[3:], actual["orientation_xyzw"],
                    )))
                    position_bound, orientation_bound = libero_global_endpoint_tolerances()
                    settle_endpoint_error = {"position_m": position_error, "orientation_rad": orientation_error}
                    return (float(np.max(np.abs(joint_delta))) <= settle_joint_tolerance
                            and position_error <= position_bound and orientation_error <= orientation_bound)

                while not terminal_settled():
                    if settle_ticks >= max_terminal_ticks:
                        # The settle loop owns the last millimetres of the
                        # global route. Exhausting it is the measured capability
                        # limit the policy has to be told about, not a silent
                        # timeout.
                        settle_exhausted = True
                        stop_reason = "terminal_tracking_timeout"
                        break
                    before_ticks = len(measured)
                    continuing = execute_tick(final_goal)
                    settle_ticks += len(measured) - before_ticks
                    if not continuing:
                        break
                if waypoint_ticks:
                    waypoint_ticks[-1] += settle_ticks
        finally:
            action_owner._action_dim = previous_action_dim
            robot.controller_config = osc_config
            robot._load_controller()
        final = np.asarray(robot._joint_positions, dtype=np.float64)
        goal = positions[active_goal_index]
        self._joint_execution_result = {
            "execution_id": execution_id,
            "controller": "JOINT_POSITION",
            "steps_used": len(measured),
            "joint_positions": final.tolist(),
            "tracking_error_l2_rad": float(np.linalg.norm(final - goal)),
            "tracking_error_max_abs_rad": float(np.max(np.abs(final - goal))),
            "execution_phase": execution_phase,
            "waypoints_total": len(positions),
            "waypoints_reached": reached_waypoints,
            "waypoint_ticks": waypoint_ticks,
            "terminated": terminal,
            "truncated": truncated,
            "stop_reason": stop_reason,
            "terminal_tick": len(measured) if terminal or truncated else None,
            "samples": measured,
            "commands": issued_commands,
            **({"terminal_settle": {
                "ticks": settle_ticks, "joint_tolerance_rad": settle_joint_tolerance,
                "max_ticks": max_terminal_ticks, "exhausted": settle_exhausted,
                "endpoint_error": settle_endpoint_error,
                "endpoint_tolerance": dict(zip(("position_m", "orientation_rad"), libero_global_endpoint_tolerances())),
            }} if settle_joint_tolerance is not None else {}),
        }

    @property
    def harness_robustness_state(self) -> dict[str, Any]:
        state = getattr(self._delegate, "harness_robustness_state", None)
        if not isinstance(state, Mapping):
            raise LiberoProError("LIBERO robustness state is unavailable")
        return dict(state)

    @property
    def harness_robustness_armed(self) -> bool:
        return bool(getattr(self._delegate, "harness_robustness_armed", False))

    @harness_robustness_armed.setter
    def harness_robustness_armed(self, value: Any) -> None:
        if not hasattr(type(self._delegate), "harness_robustness_armed"):
            raise LiberoProError("LIBERO robustness runtime is unavailable")
        self._delegate.harness_robustness_armed = value


def _camera_metadata_env_factory(factory: Callable[[], Any]) -> Callable[[], Any]:
    def build() -> _CameraMetadataEnvironment:
        return _CameraMetadataEnvironment(factory())

    return build
