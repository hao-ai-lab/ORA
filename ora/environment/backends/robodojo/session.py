"""One isolated, source-bound RoboDojo episode over runtime v3 and sparse motion v3."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from ora.contracts.action import MOTION_ACTION_V3_SCHEMA, RUNTIME_ACTION_SCHEMA, validate_runtime_action
from ora.environment.base import ENVIRONMENT_STATUS_SCHEMA, validate_environment_status
from ora.environment.benchmarks.robodojo import (
    ACTION_CODECS, NATIVE_ACTION_LIMITS, SENSOR_PROFILE,
    UPSTREAM_ENDING_CONDITIONS, classify_round_ending, configuration_binding,
    official_terminal_reason, public_context, validate_scene_registration,
)
from ora.environment.controllers.robodojo_arx import controller_binding, validate_robot_facts

from .round_lifecycle import (
    RobodojoRoundMixin, PER_ROUND_CLEARED_CHANNELS, ROUND_HOME_POSITION_TOLERANCE_M,
    ROUND_HOME_ROTATION_TOLERANCE_DEG, ROUND_ISOLATION_SCHEMA, ROUND_RESET_SCHEMA,
)
from .observation import RobodojoObservationMixin
from .execution_capture import RobodojoExecutionCaptureMixin
from .feedback import gripper_state, add_receipt_feedback
from . import layout_rounds, recovery, sparse_motion
from .bridge import RobodojoError, build_environment
from .sequence_admission import SEQUENCE_PARENT_SCHEMAS, admit_sequence_child
from .common import (
    GRIPPER_TOLERANCE,
    array, file_hash, gripper_opening, json_default,
    jsonable, measured_pose,
    native_pose, physical_gripper, pose_error, pose_reached, workspace_violation, write_json,
)


SOURCE_SCHEMA = "ora.robodojo-source-manifest.v1"
RESET_SCHEMA = "ora.robodojo-reset.v1"
RECEIPT_SCHEMA = "ora.robodojo-motion-receipt.v1"
CONTROL_TRACE_SCHEMA = "ora.robodojo-controller-trace.v1"
VERDICT_SCHEMA = "ora.robodojo-official-verdict.v2"
TERMINAL_BASIS = "official_terminal_transition"
POLICY_BUDGET_BASIS = "standing_official_predicate_at_policy_budget"
INFRASTRUCTURE_BASIS = "infrastructure_abort"
RUNNING_BASIS = "episode_incomplete"
CAMERAS = {"head": "cam_head", "left_wrist": "cam_left_wrist", "right_wrist": "cam_right_wrist"}
FORBIDDEN_ABSENT = ["actor_state", "object_gt_pose", "segmentation", "contact", "success_predicate", "expert_program", "task_source"]

_json_default = json_default
_array = array
_pose = measured_pose
_opening = gripper_opening
_physical_gripper = physical_gripper


class RobodojoSession(RobodojoRoundMixin, RobodojoObservationMixin, RobodojoExecutionCaptureMixin):
    """The session never exposes the upstream environment or reward state."""

    def __init__(
        self, *, config: Mapping[str, Any], output_root: Path,
        environment_factory: Callable[..., tuple[Any, Any, dict[str, Any]]] = build_environment,
    ) -> None:
        self.config = deepcopy(dict(config))
        self.task = self.config.get("task", "arrange_largest_number")
        if not isinstance(self.task, str) or self.task not in NATIVE_ACTION_LIMITS:
            raise RobodojoError("unsupported RoboDojo task")
        self.native_action_limit = NATIVE_ACTION_LIMITS[self.task]
        if self.config.get("native_action_limit", self.native_action_limit) != self.native_action_limit:
            raise RobodojoError("RoboDojo native action limit differs from the selected task")
        if self.config.get("task_variant", self.task) != self.task:
            raise RobodojoError("RoboDojo task and runtime selection differ")
        self.action_codec = self.config.get("action_codec") or ACTION_CODECS[0]
        if self.action_codec not in ACTION_CODECS:
            raise RobodojoError("unsupported RoboDojo first-stage binding: action_codec")
        self.config["action_codec"] = self.action_codec
        fixed = {"task": self.task, "eval_seed": 0, "num_envs": 1, "sensor_profile": SENSOR_PROFILE}
        for name, expected in fixed.items():
            if self.config.get(name, expected) != expected:
                raise RobodojoError(f"unsupported RoboDojo first-stage binding: {name}")
            self.config[name] = expected
        self._init_layout_schedule()
        self.max_sim_steps = int(self.config.get("max_sim_steps", self.native_action_limit * 10))
        self.native_actions_per_command = int(self.config.get("native_actions_per_command", 20))
        if not 10 <= self.max_sim_steps <= self.native_action_limit * 10 or self.max_sim_steps % 10:
            raise RobodojoError(f"max_sim_steps must be a multiple of 10 in [10, {self.native_action_limit * 10}]")
        if self.native_actions_per_command != 20:
            raise RobodojoError("native_actions_per_command is fixed at 20")
        self.free_space_clearance_m = float(self.config.get("free_space_clearance_m", sparse_motion.FREE_SPACE_CLEARANCE_M))
        self.local_joint_jump_limit_rad = float(
            self.config.get("local_joint_jump_limit_rad", sparse_motion.LOCAL_JOINT_JUMP_LIMIT_RAD)
        )
        self.global_joint_jump_limit_rad = float(
            self.config.get("global_joint_jump_limit_rad", sparse_motion.GLOBAL_JOINT_JUMP_LIMIT_RAD)
        )
        if (not 0.0 < self.free_space_clearance_m <= 0.5
                or not 0.0 < self.local_joint_jump_limit_rad <= self.global_joint_jump_limit_rad <= 2 * np.pi):
            raise RobodojoError("sparse motion bounds are outside their supported ranges")
        # Absent means the released route: a contact approach beyond the local
        # envelope is refused. Only a surface that declares staging selects it.
        self.contact_approach_route = self.config.get("contact_approach_route")
        if self.contact_approach_route not in (None, sparse_motion.STAGED_CONTACT_STRATEGY):
            raise RobodojoError("contact_approach_route names no supported route")
        self.recovery_route = recovery.configured_route(self.config)
        self.output_root = Path(output_root).resolve()
        self._init_execution_capture()
        self.environment_factory = environment_factory
        self.environment: Any = None
        self.app: Any = None
        self.robots: dict[str, Any] = {}
        self.source: dict[str, Any] = {}
        self.source_hash: str | None = None
        self.calibration_readbacks: dict[str, Any] = {}
        self.started = False
        self.closed = False
        self.abort_reason: str | None = None
        self.fault_reason: str | None = None
        self.boundary_reason: str | None = None
        self.official_terminal: dict[str, Any] | None = None
        self.sim_steps = 0
        self.reset_physics_steps = 0
        self.round_index = 0
        self.observation_sequence = 0
        self.action_sequence = 0
        self.motion_probe_sequence = 0
        self.world_mutation_sequence = 0
        self.world_mutation_actions: list[str] = []
        self.plan_references: dict[str, dict[str, Any]] = {}
        self.action_sequences: list[dict[str, Any]] = []
        self.last_observation_id: str | None = None
        self.last_receipt: dict[str, Any] | None = None
        self.pose_targets: dict[str, list[float]] = {}
        self.gripper_directives: dict[str, float] = {}
        self.home_joints: dict[str, list[float]] = {}
        self.home_poses: dict[str, list[float]] = {}
        self.last_safe_joints: dict[str, list[float]] = {}
        self.last_safe_poses: dict[str, list[float]] = {}
        # The pose the session's *first* reset measured, kept for the life of
        # the session: a later round is compared against the world the run
        # started from, not against the previous round's own reset.
        self.session_home_poses: dict[str, list[float]] = {}
        self.session_home_joints: dict[str, list[float]] = {}

    def _clock(self) -> int:
        # This counter is advanced by PhysX callbacks, including accidental
        # Kit/app updates outside RoboDojo.sim_step. The wrapper's own counter
        # alone cannot establish that observe() did not advance physics.
        return int(self.environment.sim.sim.current_time_step_index)

    def start(self) -> dict[str, Any]:
        if self.started or self.closed:
            raise RobodojoError("a RoboDojo session starts exactly once")
        self.output_root.mkdir(parents=True, exist_ok=True)
        try:
            self.environment, self.app, self.source = self.environment_factory(self.config)
            self._reset_environment("reset-initialization-stack.txt")
            self.robots = {
                robot.arm_name.split("_")[0]: robot
                for robot in self.environment.robot_manager.robot_list if robot.type == "target"
            }
            if set(self.robots) != {"left", "right"}:
                raise RobodojoError("ARX X5 must have exactly left and right target arms")
            for robot in self.robots.values():
                validate_robot_facts({
                    "joint_dof": len(robot.arm_joints_name), "ee_link": robot.ee_link_name, "ee_type": robot.ee_type,
                    "gripper_scale": list(robot.gripper_scale), "gripper_sign": robot.gripper_move["sign"],
                    "gripper_mimic_multiplier": robot.gripper_move["mimic"][1],
                    "gripper_mimic_offset": robot.gripper_move["mimic"][2],
                })
            runtime_embodiment = self._runtime_embodiment()
            self.robot_geometry_verification = None
            if self.config.get("public_robot_geometry") is not None:
                from ora.contracts.robot_geometry import verify_loaded_robot_geometry
                from .robot_asset_geometry import loaded_geometry
                self.robot_geometry_verification = verify_loaded_robot_geometry(
                    self.config["public_robot_geometry"], loaded_geometry(self),
                )
            self._bind_reset_state()
            manifest_path = self.output_root / "source-manifest.json"
            if manifest_path.exists():
                raise RobodojoError("refusing to overwrite a source manifest")
            write_json(manifest_path, self.source)
            self.source_hash = file_hash(manifest_path)
            self.started = True
            instruction = self._task_instruction()
            fingerprint = {
                "loaded_files": self.source.get("loaded_files", {}),
                "installed_packages": self.source.get("installed_packages", {}),
                "resolved_environment_config_sha256": hashlib.sha256(json.dumps(
                    self.source.get("resolved_environment_config", {}), sort_keys=True, allow_nan=False,
                ).encode()).hexdigest(),
            }
            binding = self.config.get("qualification_binding")
            if binding is not None:
                if (binding.get("passed") is not True
                        or binding.get("source_revisions") != self.source["source_revisions"]
                        or binding.get("layout_sha256") != self.source["layout_file"]["sha256"]
                        or binding.get("configuration") != configuration_binding(self.config)
                        or binding.get("actual_source_fingerprint") != fingerprint
                        or binding.get("device_binding") != {
                            key: value for key, value in self.source.get("device_binding", {}).items()
                            if key != "startup_log_prefix"
                        }
                        or binding.get("task_instruction") != instruction):
                    raise RobodojoError("live source/control/instruction readback differs from infrastructure qualification")
            if self.config.get("expected_task_language", instruction) != instruction:
                raise RobodojoError("upstream public instruction differs from the selected policy task language")
            reset = {
                "schema": RESET_SCHEMA, "task": self.task, "eval_seed": 0, "seed": 0, "layout_id": self.layout_id,
                "source_manifest": {"path": "source-manifest.json", "sha256": self.source_hash},
                "source_binding": {"source_manifest_sha256": self.source_hash, "source_revisions": self.source["source_revisions"]},
                "control_binding": controller_binding(),
                "runtime_embodiment": runtime_embodiment,
                "device_binding": self.source.get("device_binding"),
                "physics_steps_before_policy": self.reset_physics_steps,
                "reward_registered": bool(self.environment.reward_registered),
                "task_instruction": instruction, "public_environment_context": public_context(),
                "scene_registration": self._scene_registration(),
                "actual_source_fingerprint": fingerprint,
                "action_codec": self.action_codec, "round_index": 0,
            }
            reset.update(layout_rounds.start_reset_fields(self))
            write_json(self.output_root / "reset.json", reset)
            self._start_execution_capture()
            return reset
        except BaseException:
            import traceback

            (self.output_root / "session-start-error.txt").write_text(traceback.format_exc())
            self._fault("robodojo_start_failed")
            raise

    def _task_instruction(self) -> str:
        instruction = str(self.environment.obs_manager.instruction[0])
        if not instruction.strip():
            raise RobodojoError("upstream public task instruction is empty")
        return instruction

    def _require_started(self) -> None:
        if not self.started or self.closed or self.environment is None:
            raise RobodojoError("RoboDojo session is not active")

    def _proprioception(self) -> dict[str, Any]:
        """Publish the requested directive separately from the SDK drive target.

        The SDK can limit an explicit close relative to the primary encoder.
        Its resolved buffer is therefore not the requested directive. Keeping
        that distinction here makes observation, retention and receipts agree.
        """
        manager = self.environment.robot_manager
        result = {}
        for arm, robot in self.robots.items():
            joints = array(manager.get_joint(robot, env_idx_list=[0])[0])
            fingers = array(manager.get_end_effector_real_val(robot, env_idx_list=[0])[0])
            if not np.isfinite(joints).all() or not np.isfinite(fingers).all():
                raise RobodojoError("robot joint measurement is not finite")
            key = manager.process_name(robot.gripper_name)
            command = gripper_opening(
                robot, float(manager.control_manager.prev_control[0][key]["position"][0]),
            )
            result[arm] = {
                "pose_xyz_xyzw": measured_pose(manager, robot), "pose_frame": "robodojo_episode_local_ee_link",
                "arm_joint_qpos": joints.tolist(),
                **self._gripper_state(arm, fingers, command),
            }
        return {"end_effectors": result, "single_arm": False}

    def _gripper_state(self, arm: str, fingers: Any, applied: float) -> dict[str, Any]:
        return gripper_state(self, arm, fingers, applied)

    def _runtime_embodiment(self) -> dict[str, Any]:
        """Collect bounded robot/sensor facts from the initialized simulator."""
        manager = self.environment.robot_manager
        arms = {}
        for arm, robot in self.robots.items():
            articulation = manager.robot_key[manager.robot_list.index(robot)]
            joint_names, body_names = list(articulation.joint_names), list(articulation.body_names)
            if (articulation.is_fixed_base is not True or len(robot.arm_joints_name) != 6
                    or not set(robot.arm_joints_name).issubset(joint_names)
                    or not set(robot.gripper_joints_name).issubset(joint_names)
                    or robot.ee_link_name not in body_names):
                raise RobodojoError("compiled ARX articulation differs from the declared robot")
            arms[arm] = {
                "fixed_base": True, "arm_joint_names": list(robot.arm_joints_name),
                "gripper_joint_names": list(robot.gripper_joints_name),
                "compiled_joint_names": joint_names, "compiled_body_names": body_names,
                "controlled_link": robot.ee_link_name,
            }
        camera_manager = self.environment.camera_manager
        cameras = {}
        for public, source in CAMERAS.items():
            index = camera_manager.camera_names[0].index(source)
            camera = camera_manager.cameras[0][index]
            mount = camera_manager.camera_config[source].camera.get("mount_link")
            expected = None if public == "head" else f"robot{0 if public == 'left_wrist' else 1}/camera"
            prim_path = str(camera.prim_path)
            prefix = "/World/envs/env_0/" + ("" if expected is None else expected + "/")
            if mount != expected or not prim_path.startswith(prefix):
                raise RobodojoError("compiled camera attachment differs from the public sensor binding")
            cameras[public] = {"source_name": source, "mount_link": mount, "optical_prim_path": prim_path}
        return {"schema": "ora.robodojo-runtime-embodiment.v1",
                "collector": "initialized_isaac_articulation_and_optical_camera_prim",
                "arms": arms, "cameras": cameras,
                "position_frame": "episode_local_world", "quaternion_order": "xyzw"}

    def _applied_position_targets(self) -> dict[str, list[float]]:
        """Read the SDK buffers actually submitted by RobotManager.control_robot."""
        manager = self.environment.robot_manager
        result = {}
        for robot in self.robots.values():
            articulation = manager.robot_key[manager.robot_list.index(robot)]
            targets = array(articulation.data.joint_pos_target[0])
            for name, indices in ((robot.arm_name, robot.arm_joint_indices),
                                  (robot.gripper_name, robot.gripper_joint_indices)):
                values = targets[indices]
                if not np.isfinite(values).all():
                    raise RobodojoError("SDK applied an invalid joint target")
                result[manager.process_name(name)] = values.tolist()
        if getattr(self.environment, "interact", False):
            from .interaction_control import support_position_targets
            result.update(support_position_targets(self))
        return result

    def measure_jaw_state(self) -> dict[str, Any]:
        """Read the raw private jaw evidence the aperture probe judges.

        Qualification-only. This is never reachable from a policy tool: the
        worker exposes it under its own RPC method, no ToolSurface names it,
        and nothing in ``execute``/``observe`` calls it. It returns raw
        readings and derives nothing — the pad-face separation, the nominal
        comparison and the pass criteria all live in the scripted judge in
        ``motion_qualification`` so they can be unit tested without a plant.

        Finger link world positions are private state: the policy sees a
        normalized ``gripper`` reading, never a body pose.
        """

        manager = self.environment.robot_manager
        arms: dict[str, Any] = {}
        for arm, robot in self.robots.items():
            articulation = manager.robot_key[manager.robot_list.index(robot)]
            data = articulation.data
            body_names = list(articulation.body_names)
            finger_bodies: dict[str, list[float]] = {}
            for name in robot.gripper_joints_name:
                # The X5 finger joints are named jointN and drive linkN; the
                # articulation indexes bodies by name, so resolve the driven
                # body from the joint name rather than assuming an order.
                body = name.replace("joint", "link")
                if body not in body_names:
                    raise RobodojoError(
                        f"ARX articulation has no finger body {body!r}"
                    )
                position = array(data.body_pos_w[0][body_names.index(body)])
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise RobodojoError("finger body position is not finite")
                finger_bodies[body] = position.tolist()
            targets = array(data.joint_pos_target[0])[robot.gripper_joint_indices]
            fingers = array(
                manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]
            )
            key = manager.process_name(robot.gripper_name)
            command = gripper_opening(
                robot,
                float(manager.control_manager.prev_control[0][key]["position"][0]),
            )
            if not np.isfinite(targets).all() or not np.isfinite(fingers).all():
                raise RobodojoError("gripper measurement is not finite")
            arms[arm] = {
                **self._gripper_state(arm, fingers, command),
                "applied_gripper_joint_targets": targets.tolist(),
                "gripper_joint_names": list(robot.gripper_joints_name),
                "controlled_link_pose_xyz_xyzw": measured_pose(manager, robot),
                "finger_body_position_w_m": finger_bodies,
            }
        return {
            "schema": "ora.robodojo-jaw-state.v1",
            "arms": arms,
            "physics_steps": self.sim_steps,
            "native_actions": (
                int(self.environment.take_action_cnt[0])
                if self.environment is not None else 0
            ),
            "round_index": self.round_index,
        }

    def _open_journal(self) -> Path:
        journal = self.output_root / "controller" / f"action-{self.action_sequence:04d}.jsonl"
        journal.parent.mkdir(parents=True, exist_ok=True)
        if journal.exists():
            raise RobodojoError("refusing to overwrite a controller journal")
        journal.touch()
        return journal

    def _run_native_control(
        self, *, action_id: str, control: Mapping[str, Any], held_targets: Mapping[str, Any],
        journal: Path, tracking: list[dict[str, Any]], segment: str,
        reached: Callable[[Mapping[str, Any]], bool], budget: int,
    ) -> int:
        """Drive one SDK command dict for up to ``budget`` native actions.

        Every physics step is journaled before its successor so a later fault
        cannot lose the executed prefix; the applied SDK target buffers must
        equal the recorded command on controlled dimensions and the pre-action
        targets on held dimensions.
        """
        def record_physics(command: Mapping[str, Any]) -> None:
            applied = self._applied_position_targets()
            sample = {"action_id": action_id, "segment": segment,
                      "physics_clock": self._clock(), "native_action": int(self.environment.take_action_cnt[0]),
                      "command": command, "applied_position_targets": applied,
                      "measured": self._proprioception()["end_effectors"]}
            with journal.open("a") as stream:
                stream.write(json.dumps(sample, default=json_default, allow_nan=False, sort_keys=True) + "\n")
            for key, target in command.items():
                expected = held_targets.get(key, target)["position"]
                if (not np.allclose(applied[key], target["position"], rtol=0, atol=1e-6)
                        or not np.allclose(applied[key], expected, rtol=0, atol=1e-6)):
                    self._fault("robodojo_applied_command_mismatch")
                    raise RobodojoError(f"SDK applied a changed held target or an unrecorded command: {key}")

        used = 0
        while used < budget and not self.is_terminal:
            clock_before = self._clock()
            self.environment.controller_trace_callback = record_physics
            try:
                terminal = self.environment.native_tick(control)
            except BaseException:
                self._fault("robodojo_native_execution_failed")
                raise
            finally:
                self.sim_steps += self._clock() - clock_before
                self.environment.controller_trace_callback = None
            if self._clock() - clock_before != 10:
                self._fault("native_cadence_mismatch")
                raise RobodojoError("native action did not advance exactly 10 physics steps")
            used += 1
            current = self._proprioception()["end_effectors"]
            tracking.append({"native_action": self.environment.take_action_cnt[0], "physics_steps": self.sim_steps,
                             "segment": segment, "control": deepcopy(control), "measured": current})
            if terminal is not None:
                self.official_terminal = deepcopy(terminal)
            if reached(current) or self.is_terminal:
                break
        return used

    def _conformance_tick(self, control: Mapping[str, Any]) -> None:
        """One unjournaled native tick for adapter conformance, never a policy action path."""
        self._require_started()
        if self.is_terminal:
            raise RobodojoError("terminated or aborted episode cannot accept actions")
        clock_before = self._clock()
        terminal = self.environment.native_tick(control)
        self.sim_steps += self._clock() - clock_before
        if self._clock() - clock_before != 10:
            self._fault("native_cadence_mismatch")
            raise RobodojoError("native action did not advance exactly 10 physics steps")
        if terminal is not None:
            self.official_terminal = deepcopy(terminal)

    def _held_targets_unchanged(self, held_targets: Mapping[str, Any]) -> bool:
        command_after = self.environment.robot_manager.control_manager.prev_control[0]
        return all(
            json.dumps(command_after[key], default=json_default, sort_keys=True)
            == json.dumps(target, default=json_default, sort_keys=True)
            for key, target in held_targets.items()
        )

    def _persist_receipt(self, receipt: dict[str, Any], *, tracking: list[dict[str, Any]], journal: Path) -> dict[str, Any]:
        add_receipt_feedback(receipt)
        private = self.output_root / "controller" / f"action-{self.action_sequence:04d}.json"
        write_json(private, {"schema": CONTROL_TRACE_SCHEMA, "action_id": receipt["action_id"],
                             "samples": tracking, "physics_journal": {"path": journal.relative_to(self.output_root).as_posix(), "sha256": file_hash(journal)}})
        receipt["control_trace"] = {"path": private.relative_to(self.output_root).as_posix(), "sha256": file_hash(private)}
        # Upstream initial targets can be numpy arrays; the RPC response itself
        # must contain only JSON values, not just the on-disk serialization.
        receipt = jsonable(receipt)
        receipt_path = self.output_root / "receipts" / f"action-{self.action_sequence:04d}.json"
        if receipt_path.exists():
            raise RobodojoError("refusing to overwrite a controller receipt")
        write_json(receipt_path, receipt)
        if receipt.get("state_changed"):
            self.world_mutation_sequence += 1
            self.world_mutation_actions.append(str(receipt.get("action")))
        self.last_receipt = {"path": receipt_path.relative_to(self.output_root).as_posix(),
                             "sha256": file_hash(receipt_path), "receipt": deepcopy(receipt)}
        self.last_observation_id = None
        self.action_sequence += 1
        return deepcopy(self.last_receipt)

    def begin_action_sequence(self, action: Mapping[str, Any]) -> None:
        """Trusted executor extension; no policy tool exposes this admission."""
        from ora.contracts.action import target_state_phases
        self._require_started()
        parent = validate_runtime_action(action)
        if self.is_terminal:
            raise RobodojoError("terminated episode cannot begin a sequence")
        schema = parent["schema"]
        if schema not in SEQUENCE_PARENT_SCHEMAS:
            raise RobodojoError("unsupported sequence parent")
        if self.action_sequences:
            if len(self.action_sequences) != 1 or schema != "ora-action.v4":
                raise RobodojoError("sequence nesting exceeds the committed stage boundary")
            admit_sequence_child(self.action_sequences[-1], parent, action_codec=self.action_codec)
        elif parent["observation_id"] != self.last_observation_id:
            raise RobodojoError("sequence must bind the latest unconsumed observation")
        if schema == "ora-action.v4":
            phases = target_state_phases(parent)
            exact = {phase["action_id"]: phase for phase in phases}
            children = list(exact)
        else:
            args = parent["arguments"]
            children = [f"{parent['action_id']}:{index:02d}-{stage}"
                        for index, stage in enumerate(args["sequence"])
                        if index >= args["resume_from"] and stage not in {"stop", "verify"}]
            exact = {}
        self.action_sequences.append({"parent": parent, "children": children, "exact": exact, "next": 0})
        self.last_observation_id = None

    def end_action_sequence(self, action_id: str) -> None:
        if not self.action_sequences or self.action_sequences[-1]["parent"]["action_id"] != action_id:
            raise RobodojoError("sequence cleanup does not match its active parent")
        self.action_sequences.pop()

    def _execute_without_capture(self, action: Mapping[str, Any]) -> dict[str, Any]:
        self._require_started()
        if self.is_terminal:
            raise RobodojoError("terminated or aborted episode cannot accept actions")
        normalized = validate_runtime_action(action)
        if self.action_sequences:
            admit_sequence_child(self.action_sequences[-1], normalized, action_codec=self.action_codec)
        elif normalized["observation_id"] != self.last_observation_id:
            raise RobodojoError("action must bind the latest unconsumed observation")
        if normalized["schema"] == MOTION_ACTION_V3_SCHEMA:
            if self.action_codec != "motion_action_v3":
                raise RobodojoError("sparse motion v3 requires the motion_action_v3 codec")
            return sparse_motion.execute_sparse_motion_v3(self, normalized)
        if normalized["schema"] != RUNTIME_ACTION_SCHEMA:
            raise RobodojoError("only runtime action v3 and motion action v3 are supported")
        arguments = normalized["arguments"]
        arm, kind = arguments["arm"], normalized["action"]
        manager, robot = self.environment.robot_manager, self.robots[arm]
        before = self._proprioception()["end_effectors"]
        command_before = deepcopy(manager.control_manager.prev_control[0])
        if getattr(self.environment, "interact", False):
            from .interaction_control import policy_targets
            command_before = policy_targets(self, command_before)
        pose_targets_before = deepcopy(self.pose_targets)
        gripper_directives_before = deepcopy(self.gripper_directives)
        control: dict[str, Any] = {}
        ik_status = "not_requested"
        ik_result = None
        target_pose = deepcopy(self.pose_targets[arm])
        if kind == "move_pose":
            target_pose = arguments["position_m"] + arguments["orientation_xyzw"]
            if workspace_violation(target_pose, self.config.get("workspace_bounds_m")) is not None:
                raise RobodojoError("pose target exceeds the qualified workspace")
            ik_result = sparse_motion.solve_ik(self, arm, target_pose)
            ik_status = str(ik_result["ik_status"])
            if ik_result["ik_selection"]["outcome"] == "joint_jump":
                ik_status = "JointJump"
            if ik_status == "Success":
                joints = array(ik_result["joint_target"])
                if joints.shape != np.asarray(before[arm]["arm_joint_qpos"]).shape or not np.isfinite(joints).all():
                    raise RobodojoError("IK returned invalid joint targets")
                control[manager.process_name(robot.arm_name)] = {"position": joints.tolist()}
                self.pose_targets[arm] = target_pose
        else:
            self.gripper_directives[arm] = arguments["gripper"]
            control[manager.process_name(robot.gripper_name)] = {"position": physical_gripper(robot, arguments["gripper"])}
        effective_gripper_target = max(self.gripper_directives[arm], sparse_motion.gripper_closed_floor(self, arm))
        held_targets = {key: value for key, value in command_before.items() if key not in control}
        native_before, steps_before = int(self.environment.take_action_cnt[0]), self.sim_steps
        tracking: list[dict[str, Any]] = []
        journal = self._open_journal()

        def reached(current: Mapping[str, Any]) -> bool:
            if kind == "move_pose":
                return pose_reached(target_pose, current[arm]["pose_xyz_xyzw"])
            return abs(current[arm]["gripper"] - effective_gripper_target) <= GRIPPER_TOLERANCE

        if control:
            self._run_native_control(
                action_id=normalized["action_id"], control=control, held_targets=held_targets,
                journal=journal, tracking=tracking, segment="command", reached=reached,
                budget=self.native_actions_per_command,
            )
        after = self._proprioception()["end_effectors"]
        distance, angle = pose_error(target_pose, after[arm]["pose_xyz_xyzw"])
        motion_status = "ik_failed" if kind == "move_pose" and ik_status != "Success" else "reached" if reached(after) else "not_reached"
        command_after = deepcopy(manager.control_manager.prev_control[0])
        if getattr(self.environment, "interact", False):
            from .interaction_control import policy_targets
            command_after = policy_targets(self, command_after)
        held_verified = self._held_targets_unchanged(held_targets)
        if not held_verified:
            self._fault("robodojo_held_command_changed")
            raise RobodojoError("action changed a held SDK command")
        receipt = {
            "schema": RECEIPT_SCHEMA, "action_id": normalized["action_id"], "action": kind,
            "target_arms": [arm],
            "arguments": arguments, "consumed": True, "state_changed": self.sim_steps > steps_before,
            "motion_status": motion_status, "ik_status": ik_status,
            "ik_selection": None if ik_result is None else ik_result["ik_selection"],
            "planning_diagnostics": None if ik_result is None else ik_result["planning_diagnostics"],
            "ik_attempts": [] if ik_result is None else [ik_result],
            "native_actions": int(self.environment.take_action_cnt[0]) - native_before,
            "physics_steps": self.sim_steps - steps_before,
            "commanded_pose_xyz_xyzw": target_pose, "reached_pose_xyz_xyzw": after[arm]["pose_xyz_xyzw"],
            "reached_gripper": after[arm]["gripper"], "gripper_directive": self.gripper_directives[arm],
            "before_pose_xyz_xyzw": before[arm]["pose_xyz_xyzw"],
            "position_error_m": distance, "orientation_error_rad": angle,
            "before": before, "after": after,
            "command_invariant": {
                "schema": "ora.command-invariant.v2",
                "controlled_dimension": "eef_pose_target" if kind == "move_pose" else "gripper",
                "held_dimension": "gripper_directive" if kind == "move_pose" else "eef_pose_target",
                "verified": held_verified and (
                    self.gripper_directives == gripper_directives_before if kind == "move_pose"
                    else self.pose_targets == pose_targets_before),
                "applied_targets": "per_physics_step_sdk_joint_position_buffer",
            },
            "command_targets_before": command_before,
            "command_targets_after": command_after,
            "pose_targets_before": pose_targets_before, "pose_targets_after": deepcopy(self.pose_targets),
        }
        return self._persist_receipt(receipt, tracking=tracking, journal=journal)

    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._require_started()
        from ora.environment.controllers.alignment import plan_alignment
        from .common import POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD
        return plan_alignment(request, adapter=self.motion_backend_adapter(),
            control_profile={"profile": {"arrival_check": {
                "position_tolerance_m": POSITION_TOLERANCE_M,
                "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD}}},
            available_arms=("left", "right"))

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self._require_started()
        if self.action_codec != "motion_action_v3":
            raise RobodojoError("motion probing requires the motion_action_v3 codec")
        return sparse_motion.probe_motion(self, request)

    def motion_backend_adapter(self) -> Any:
        from .motion_adapter import RobodojoMotionBackendAdapter

        self._require_started()
        return RobodojoMotionBackendAdapter.from_session(self)

    @property
    def is_terminal(self) -> bool:
        return bool(self.status()["terminal"])

    def status(self) -> dict[str, Any]:
        reason = self.abort_reason
        if reason is None and self.official_terminal is not None:
            reason = official_terminal_reason(self.official_terminal, self.native_action_limit)
        if reason is None and self.sim_steps >= self.max_sim_steps:
            reason = "max_sim_steps_per_episode_exhausted"
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA, "phase": "terminal" if reason else "running",
            "terminal": reason is not None, "reason": reason, "source": "robodojo_session",
            "policy_visibility": "control_only", "signals": [],
        })

    def _fault(self, reason: str) -> None:
        """Record an infrastructure fault, never a policy-side stop.

        Only the Harness itself calls this. A faulted episode keeps a null
        official verdict, so a broken worker can never be published as a task
        failure.
        """
        if not reason:
            raise RobodojoError("fault reason must be nonempty")
        if self.fault_reason is None:
            self.fault_reason = reason
        if self.official_terminal is None and self.abort_reason is None:
            self.abort_reason = reason

    def abort(self, reason: str) -> None:
        """Stop the episode on the Supervisor's own policy-side boundary.

        The Supervisor calls this only for a decision, wall-time, model-call
        or teacher-horizon budget, or a policy lifecycle request; an
        infrastructure fault never reaches this path.
        """
        if not reason:
            raise RobodojoError("abort reason must be nonempty")
        if self.official_terminal is not None or self.abort_reason is not None:
            return
        self.boundary_reason = reason
        self.abort_reason = reason

    def _official_outcome(self) -> tuple[bool | None, bool, str]:
        """Resolve the official Boolean, its terminal claim and its provenance.

        RoboDojo latches success inside ``RewardManager.step``, which the
        wrapper runs after every native action, and no physics advances
        outside a native tick. So when a policy-side budget stops a round
        without an official terminal transition, the standing official
        predicate was already evaluated after the last physics step and was
        not a success. Reporting it is the same rule LIBERO applies, and it
        avoids a forced ``get_reward(final_check=True)``, which writes
        ``_gated_score_lst`` and can latch ``env.success`` false, is called
        two or three times inside the upstream step loop rather than once at
        the end of an episode, and for every shipped task evaluates nothing
        new.
        """
        if self.official_terminal is not None:
            return bool(self.official_terminal["official_success"]), True, TERMINAL_BASIS
        if self.fault_reason is not None:
            return None, False, INFRASTRUCTURE_BASIS
        if self.boundary_reason is not None or self.sim_steps >= self.max_sim_steps:
            return False, False, POLICY_BUDGET_BASIS
        return None, False, RUNNING_BASIS

    def persist_official_verdict(self) -> dict[str, Any]:
        official_success, terminal, basis = self._official_outcome()
        native_actions = (
            int(self.environment.take_action_cnt[0])
            if self.environment is not None else 0
        )
        # ``verdict_basis`` says where the Boolean came from. It does not say
        # whether the round is comparable with an upstream rollout: a policy
        # ``finish_episode`` and an exhausted decision budget both arrive as
        # ``standing_official_predicate_at_policy_budget`` and only the first
        # is an ending upstream has no counterpart for. The comparability
        # fields answer that separately, so no reader has to infer it from
        # ``abort_reason``.
        ending = classify_round_ending(
            official_terminal=self.official_terminal is not None,
            fault_reason=self.fault_reason,
            abort_reason=self.abort_reason,
            official_success=official_success,
            native_actions=native_actions,
            native_action_limit=self.native_action_limit,
        )
        verdict = {
            "schema": VERDICT_SCHEMA,
            "official_success": official_success,
            "terminal": terminal, "verdict_basis": basis, "fault_reason": self.fault_reason,
            **ending,
            "upstream_ending_conditions": list(UPSTREAM_ENDING_CONDITIONS),
            "native_action_limit": self.native_action_limit,
            "task": self.task, "eval_seed": 0, "seed": 0, "layout_id": self.layout_id, "round_index": self.round_index,
            "native_actions": native_actions,
            "physics_steps": self.sim_steps, "reset_physics_steps": self.reset_physics_steps,
            "abort_reason": self.abort_reason, "source_manifest_sha256": self.source_hash,
            "source": "RoboDojo EvalEnv.is_episode_end", "created_unix_s": time.time(),
            **layout_rounds.verdict_fields(self),
        }
        self._capture_execution_boundary("terminal")
        write_json(self.output_root / "official-verdict.json", verdict)
        return verdict

    def close(self) -> None:
        if self.closed:
            return
        self.finalize_operator_video()
        self.closed = True
        try:
            if self.environment is not None:
                self.environment.close()
        finally:
            if self.app is not None:
                self.app.close()


__all__ = ["RECEIPT_SCHEMA", "ROUND_RESET_SCHEMA", "RobodojoError", "RobodojoSession"]
