"""Direct, model-free environment session for the pinned RoboTwin API.

This module deliberately exposes only RGB-D, calibration, proprioception and
execution receipts.  Task source, actor/object state, contacts, segmentation,
expert programs and the success predicate never enter the policy workspace.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


from ora.contracts.action import (
    MOTION_ACTION_V3_SCHEMA,
    validate_runtime_action,
)
from ora.contracts.pose import canonicalize_quaternion
from ora.contracts.acceptance import motion_acceptance_check
from ora.contracts.refusal import EnvironmentRefusal
from ora.environment.base import (
    ENVIRONMENT_STATUS_SCHEMA,
    validate_environment_status,
)
from ora.environment.controller_replay import ControllerReplayRecorder
from ora.environment.execution_capture import resolve_execution_capture
from .execution_capture import RoboTwinExecutionCaptureMixin
from ora.environment.controllers.profile_capabilities import (
    CONTACT_RECOVERY_SINCE,
    CONTROLLER_TRACKING_SINCE,
    GRIP_CENTER_SINCE,
    IK_CONSTRAINT_FEEDBACK_SINCE,
    LOCAL_SERVO_DIAGNOSTICS_SINCE,
    MOTION_INTENT_SINCE,
    MOTION_TRANSACTION_SINCE,
    OLDEST_COMPOSED_CONTROL_PROFILE,
    RGBD_GRASP_STAGING_SINCE,
    VALIDATED_PLANNER_SINCE,
    loaded_control_profile_version,
    profile_has,
)
from ora.environment.controllers.inter_arm_clearance import (
    InterArmClearancePolicy, minimum_inter_arm_distance as minimum_inter_arm_distance_value,
)
from ora.environment.controllers.motion_transaction import (
    MotionExecutionFault,
    MotionTransactionError,
    classify_motion_status,
    execute_measured_convergence,
    require_motion_capabilities,
    select_post_global_phase,
    select_motion_strategy,
    validate_motion_backend_adapter,
    write_motion_plan_artifact,
)
from ora.environment.backends.robotwin.motion_adapter import (
    RoboTwinMotionBackendAdapter,
)
from ora.environment.backends.robotwin.planning import (
    RoboTwinPlanRejected,
    ValidatedRoboTwinPlan,
    bounded_dls_joint_delta,
    cartesian_pose_error_twist,
    numerical_cartesian_jacobian,
    planner_public_summary,
    select_topk_validated_plan,
    select_topk_validated_plan_set,
    split_batch_plan_result,
    validate_inter_arm_clearance,
    validate_single_arm_plan,
    write_plan_artifact,
)

# Every name that was importable from this module before the 2026-10-01
# split stays importable: the imports above and the re-exports below. The
# mixins import what they use themselves.
from .common import (
    OBSERVATION_SCHEMA,
    RECEIPT_SCHEMA,
    SPARSE_RECEIPT_SCHEMA,
    SPARSE_RECEIPT_V3_SCHEMA,
    SPARSE_RECEIPT_V4_SCHEMA,
    SPARSE_RECEIPT_V5_SCHEMA,
    SPARSE_RECEIPT_V6_SCHEMA,
    SPARSE_RECEIPT_V7_SCHEMA,
    VERDICT_SCHEMA,
    OPERATOR_EXECUTION_FRAME_SCHEMA,
    OPERATOR_EXECUTION_VIDEO_SCHEMA,
    OPERATOR_CAMERA_ORDER,
    DEFAULT_REVISION,
    SUPPORTED_TASKS,
    DEFERRED_SESSION_TASKS,
    DEFAULT_WORKSPACE_BOUNDS_M,
    ENVIRONMENT_SIGNAL_POLICIES,
    TASK_RECOVERABILITY_ACTORS,
    RoboTwinLiveError,
    RoboTwinControllerTrackingError,
    _canonical_bytes,
    _json_hash,
    _file_hash,
    _atomic_json,
    _jsonable,
    _git,
    _load_task_args,
    _task_factory,
    _wxyz_to_xyzw,
    _xyzw_to_wxyz,
    _translate_pose_local_wxyz,
    _pose_error,
    _validate_workspace_target,
    probe_variant_sort_key,
    _SparseMotionState,
)
from .round_lifecycle import RoboTwinRoundMixin
from .control_profile import RoboTwinControlProfileMixin
from .kinematics import RoboTwinKinematicsMixin
from .observation import RoboTwinObservationMixin
from .probe import RoboTwinProbeMixin
from .dense_motion import RoboTwinDenseMotionMixin
from .motion_transaction import RoboTwinMotionTransactionMixin
from .local_cartesian import RoboTwinLocalCartesianMixin
from .sparse_admission import RoboTwinSparseAdmissionMixin
from .sparse_planning import RoboTwinSparsePlanningMixin
from .sparse_execution import RoboTwinSparseExecutionMixin
from .sparse_settlement import RoboTwinSparseSettlementMixin
from .operator_capture import RoboTwinOperatorCaptureMixin


#: Tasks whose ``check_success`` reads attributes that upstream assigns only
#: inside ``play_once``. An ORA rollout never runs the expert, so
#: the predicate would raise AttributeError at the reset probe and surface as
#: an infrastructure fault rather than an official verdict. Each entry
#: restores exactly what the expert's first lines compute from the reset
#: scene, transcribed from the pinned task source, so the predicate evaluates
#: the same quantity it would upstream. These are deterministic functions of
#: the reset state, not expert behavior: nothing here plans, moves or scores.
_EXPERT_ONLY_EVALUATOR_STATE = {
    # envs/open_laptop.py:33-35
    "open_laptop": ("arm_tag",),
    # envs/place_object_scale.py:102
    "place_object_scale": ("arm_tag",),
    # envs/put_object_cabinet.py:85-87
    "put_object_cabinet": ("arm_tag", "origin_z"),
}


def _restore_expert_only_evaluator_state(environment: Any, task: str) -> None:
    """Assign evaluator state upstream initializes inside ``play_once``."""

    required = _EXPERT_ONLY_EVALUATOR_STATE.get(task)
    if not required:
        return
    from envs.utils import get_face_prod  # type: ignore[import-not-found]
    from envs._base_task import ArmTag  # type: ignore[import-not-found]

    if task == "open_laptop":
        face_prod = get_face_prod(
            environment.laptop.get_pose().q, [1, 0, 0], [1, 0, 0],
        )
        environment.arm_tag = ArmTag("left" if face_prod > 0 else "right")
    elif task == "place_object_scale":
        environment.arm_tag = ArmTag(
            "right" if environment.object.get_pose().p[0] > 0 else "left"
        )
    elif task == "put_object_cabinet":
        environment.arm_tag = ArmTag(
            "right" if environment.object.get_pose().p[0] > 0 else "left"
        )
        environment.origin_z = environment.object.get_pose().p[2]
    missing = [name for name in required if not hasattr(environment, name)]
    if missing:
        raise RoboTwinLiveError(
            f"expert-only evaluator state for {task} is incomplete: "
            + ", ".join(missing)
        )


@dataclass
class RoboTwinLiveSession(
    RoboTwinRoundMixin,
    RoboTwinControlProfileMixin,
    RoboTwinKinematicsMixin,
    RoboTwinObservationMixin,
    RoboTwinProbeMixin,
    RoboTwinDenseMotionMixin,
    RoboTwinMotionTransactionMixin,
    RoboTwinLocalCartesianMixin,
    RoboTwinSparseAdmissionMixin,
    RoboTwinSparsePlanningMixin,
    RoboTwinSparseExecutionMixin,
    RoboTwinSparseSettlementMixin,
    RoboTwinOperatorCaptureMixin,
    RoboTwinExecutionCaptureMixin,
):
    """One trusted RoboTwin environment writer for one episode."""

    robotwin_root: Path
    output_root: Path
    task: str
    actual_seed: int
    episode: int
    task_config: str = "demo_randomized"
    required_revision: str = DEFAULT_REVISION
    workspace_bounds_m: Mapping[str, Any] = field(
        default_factory=lambda: dict(DEFAULT_WORKSPACE_BOUNDS_M)
    )
    max_sim_steps: int | None = None
    operator_world_video: bool = False
    operator_execution_video: bool = False
    execution_capture: Mapping[str, Any] | None = None
    _execution_recorder: Any = field(init=False, default=None, repr=False)
    _execution_step_tap: Any = field(init=False, default=None, repr=False)
    _execution_physics_step: int = field(init=False, default=0)
    _execution_action: Any = field(init=False, default=None, repr=False)
    _execution_capture_error: str | None = field(init=False, default=None)
    environment_signal_policy: str = "lifecycle_only"
    action_codec: str = "runtime_action_v3"
    control_profile: Mapping[str, Any] | None = None
    public_robot_geometry: Mapping[str, Any] | None = None
    robot_geometry_verification: dict[str, Any] | None = field(init=False, default=None)
    #: The read-only reach check's other-arm screen (integration 0059 M6),
    #: which the RunSpec derives from the ToolSurface: it screens the
    #: configuration each solved query reached (``reach_inter_arm``). Off is
    #: the unscreened check.
    reach_check_inter_arm_screen: bool = False
    operator_video_fps: int = 2
    task_factory: Callable[[str], Any] = _task_factory
    task_args_factory: Callable[..., dict[str, Any]] = _load_task_args
    environment: Any = field(init=False, default=None, repr=False)
    official_terminal: dict[str, Any] | None = field(init=False, default=None)
    terminal_observation: dict[str, Any] | None = field(init=False, default=None, repr=False)
    terminal_public_poses: dict[str, list[float]] | None = field(init=False, default=None, repr=False)
    terminal_gripper_targets: dict[str, str] | None = field(init=False, default=None, repr=False)
    physics_step_sequence: int = field(init=False, default=0)
    round_index: int = field(init=False, default=0)
    observation_sequence: int = field(init=False, default=0)
    action_sequence: int = field(init=False, default=0)
    abort_reason: str | None = field(init=False, default=None)
    last_receipt: dict[str, Any] | None = field(init=False, default=None)
    source_manifest: dict[str, Any] = field(init=False)
    operator_frame_sequence: int = field(init=False, default=0)
    operator_execution_frame_sequence: int = field(init=False, default=0)
    operator_capture_failure: str | None = field(init=False, default=None)
    operator_video_record: dict[str, Any] | None = field(init=False, default=None)
    replay_recorder: ControllerReplayRecorder = field(init=False, repr=False)
    previous_working_directory: Path | None = field(init=False, default=None)
    gripper_command_targets: dict[str, str] = field(
        init=False, default_factory=lambda: {"left": "unknown", "right": "unknown"}
    )
    last_safe_pose: dict[str, list[float]] = field(init=False, default_factory=dict)
    last_safe_joints: dict[str, list[float]] = field(init=False, default_factory=dict)
    home_joints: dict[str, list[float]] = field(init=False, default_factory=dict)
    pinocchio_models: dict[str, Any] = field(init=False, default_factory=dict, repr=False)
    world_mutation_sequence: int = field(init=False, default=0)
    world_mutation_actions: list[str] = field(init=False, default_factory=list)
    motion_probe_sequence: int = field(init=False, default=0)
    plan_references: dict[str, dict[str, Any]] = field(
        init=False, default_factory=dict, repr=False,
    )

    def __post_init__(self) -> None:
        self.execution_capture = resolve_execution_capture(self.execution_capture)
        self.robotwin_root = self.robotwin_root.expanduser().resolve()
        self.output_root = self.output_root.expanduser().resolve()
        if self.task not in SUPPORTED_TASKS:
            raise RoboTwinLiveError(
                f"task is outside the registered RoboTwin session set: {self.task}"
            )
        if isinstance(self.actual_seed, bool) or not isinstance(self.actual_seed, int):
            raise RoboTwinLiveError("actual seed must be an integer")
        if self.max_sim_steps is not None and (
            isinstance(self.max_sim_steps, bool)
            or not isinstance(self.max_sim_steps, int)
            or self.max_sim_steps <= 0
        ):
            raise RoboTwinLiveError("max_sim_steps must be a positive integer")
        if isinstance(self.episode, bool) or not isinstance(self.episode, int) or self.episode < 0:
            raise RoboTwinLiveError("episode must be non-negative")
        if (
            isinstance(self.operator_world_video, bool) is False
            or isinstance(self.operator_execution_video, bool) is False
            or isinstance(self.operator_video_fps, bool)
            or not isinstance(self.operator_video_fps, int)
            or self.operator_video_fps <= 0
        ):
            raise RoboTwinLiveError("operator video configuration is invalid")
        if self.environment_signal_policy not in ENVIRONMENT_SIGNAL_POLICIES:
            raise RoboTwinLiveError("environment signal policy is invalid")
        if self.action_codec not in {
            "runtime_action_v3", "target_state_v4", "motion_action_v3",
        }:
            raise RoboTwinLiveError("RoboTwin action codec is invalid")
        if self.action_codec == "motion_action_v3":
            version = loaded_control_profile_version(self.control_profile)
            if version is None:
                raise RoboTwinLiveError(
                    "sparse motion action requires the qualified ALOHA control profile"
                )
            if version < OLDEST_COMPOSED_CONTROL_PROFILE:
                raise RoboTwinLiveError(
                    "sparse motion action requires a control profile of version >= "
                    f"{OLDEST_COMPOSED_CONTROL_PROFILE}; v{version} executes only at the "
                    "commits that composed it"
                )
        elif self.control_profile is not None:
            raise RoboTwinLiveError(
                "RoboTwin control profile is valid only for a sparse motion action"
            )
        self._validate_reach_check_screen()
        if (
            self.environment_signal_policy == "task_recoverability_v1"
            and self.task not in TASK_RECOVERABILITY_ACTORS
        ):
            raise RoboTwinLiveError(
                f"task recoverability monitor is unavailable for {self.task}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.source_manifest = self._verify_source() if self.task_factory is _task_factory else {
            "robotwin_revision": self.required_revision,
            "test_double": True,
        }
        self.replay_recorder = ControllerReplayRecorder(
            output_root=self.output_root,
            backend="robotwin",
            environment_identity={
                "task": self.task,
                "seed": self.actual_seed,
                "episode": self.episode,
                "task_config": self.task_config,
                "action_codec": self.action_codec,
                "source_manifest": self.source_manifest,
            },
            camera_panels=list(OPERATOR_CAMERA_ORDER),
        )

    def _validate_reach_check_screen(self) -> None:
        """The screen flags are a valid mode, and a screen has the admission's sphere model."""

        from .reach_inter_arm import screen_mode

        try:
            mode = screen_mode(screen=self.reach_check_inter_arm_screen)
        except ValueError as error:
            raise RoboTwinLiveError(str(error)) from error
        profile = (self.control_profile or {}).get("profile") or {}
        if mode is not None and not isinstance(profile.get("dual_arm_collision_check"), Mapping):
            raise RoboTwinLiveError(
                "the reach-check inter-arm screen requires the control profile's "
                "dual_arm_collision_check"
            )

    def _verify_source(self) -> dict[str, Any]:
        if not (self.robotwin_root / "envs").is_dir():
            raise RoboTwinLiveError(f"invalid RoboTwin root: {self.robotwin_root}")
        revision = _git(self.robotwin_root, "rev-parse", "HEAD")
        if revision != self.required_revision:
            raise RoboTwinLiveError(
                f"RoboTwin revision mismatch: expected {self.required_revision}, got {revision}"
            )
        relative_paths = [
            "envs/_base_task.py",
            "envs/camera/camera.py",
            "envs/robot/robot.py",
            "envs/robot/planner.py",
            f"envs/{self.task}.py",
            f"task_config/{self.task_config}.yml",
            "task_config/_eval_step_limit.yml",
        ]
        if self._uses_motion_transaction_profile():
            relative_paths.append("CAPX_MIT_ATTRIBUTION.md")
        modified = _git(
            self.robotwin_root,
            "diff",
            "--name-only",
            self.required_revision,
            "--",
            *relative_paths,
        )
        if modified:
            raise RoboTwinLiveError(f"selected RoboTwin runtime files are modified: {modified}")
        files = [
            {
                "path": relative,
                "sha256": _file_hash(self.robotwin_root / relative),
            }
            for relative in relative_paths
        ]
        manifest = {
            "robotwin_revision": revision,
            "task": self.task,
            "task_config": self.task_config,
            "files": files,
        }
        return {**manifest, "manifest_sha256": _json_hash(manifest)}

    def start(self) -> dict[str, Any]:
        if self.environment is not None:
            raise RoboTwinLiveError("RoboTwin episode already started")
        root_text = str(self.robotwin_root)
        curobo_text = str(self.robotwin_root / "envs/curobo/src")
        for path in (root_text, curobo_text):
            if path not in sys.path:
                sys.path.insert(0, path)
        if self.task_factory is _task_factory:
            self.previous_working_directory = Path.cwd().resolve()
            os.chdir(self.robotwin_root)
        try:
            environment = self._construct_environment(self.actual_seed)
        except Exception:
            self._restore_working_directory()
            raise
        self.motion_probe_sequence = 0
        self._bind_episode_state(environment)
        self._verify_loaded_robot_geometry(environment)
        self._start_execution_capture()
        self._capture_operator_reset_frames(label="reset")
        reset = {
            "schema": "ora.robotwin-live-reset.v1",
            "task": self.task,
            "episode": self.episode,
            "actual_seed": self.actual_seed,
            # The Supervisor validates every round's public experiment seed
            # under the shared ``reset["seed"]`` key, including round 0.
            "seed": self.actual_seed,
            "round_index": self.round_index,
            "official_predicate_at_reset": False,
            "source_manifest": self.source_manifest,
            "workspace_bounds_m": _jsonable(self.workspace_bounds_m),
            "operator_world_video_enabled": self.operator_world_video,
            "operator_execution_video_enabled": self.operator_execution_video,
            "execution_capture": self.execution_capture,
            "execution_capture_error": self._execution_capture_error,
            # Round zero records the same control bindings every later round
            # does, so both can be audited under one rule instead of the
            # first round being structurally poorer evidence than the rest.
            "task_config": self.task_config,
            "action_codec": self.action_codec,
            "controller_action_contract": (
                None if self.control_profile is None
                else {
                    "control_profile_sha256": self.control_profile["sha256"],
                    "profile_version": loaded_control_profile_version(
                        self.control_profile
                    ),
                }
            ),
            # RoboTwin's session owns no public environment descriptor: the
            # RunPlan supplies it, so the Supervisor's post-reset descriptor
            # drift check has nothing backend-observed to compare here.
        }
        _atomic_json(self.output_root / "reset.json", reset)
        return reset

    def _construct_environment(self, seed: int) -> Any:
        """Create and reset one fresh RoboTwin task environment at ``seed``.

        RoboTwin's ``Base_Task`` exposes no in-place episode reset. Its
        ``setup_demo`` seeds NumPy and Torch and then rebuilds the SAPIEN
        engine, scene, table, robot, cameras and randomized actor poses from
        scratch, so a later round at the same seed reproduces the same initial
        layout only when it constructs a new task object. Reusing one object
        would abandon the previous scene without closing it.
        """

        environment: Any = None
        try:
            environment = self.task_factory(self.task)
            args = self.task_args_factory(
                robotwin_root=self.robotwin_root,
                task=self.task,
                task_config=self.task_config,
                runtime_output=self.output_root / "runtime",
            )
            if self.action_codec == "motion_action_v3":
                self._verify_control_profile_sources(args)
            environment.setup_demo(
                now_ep_num=self.episode,
                seed=seed,
                is_test=True,
                **args,
            )
            _restore_expert_only_evaluator_state(environment, self.task)
            if bool(environment.check_success()):
                raise RoboTwinLiveError("official predicate is already true at reset")
        except Exception:
            try:
                if environment is not None:
                    environment.close_env(clear_cache=False)
            except Exception:
                pass
            raise
        return environment

    @property
    def is_terminal(self) -> bool:
        if self.environment is None:
            return False
        return bool(self.status()["terminal"])

    def _task_recoverability_signal(self) -> dict[str, str] | None:
        if self.environment_signal_policy != "task_recoverability_v1":
            return None
        environment = self._require_environment()
        actor_name, functional_point = TASK_RECOVERABILITY_ACTORS[self.task]
        actor = getattr(environment, actor_name, None)
        getter = getattr(actor, "get_functional_point", None)
        if not callable(getter):
            raise RoboTwinLiveError(
                f"task recoverability actor is unavailable: {actor_name}"
            )
        position = [float(value) for value in getter(functional_point)]
        if len(position) < 3 or not all(math.isfinite(value) for value in position[:3]):
            code = "task_object_state_invalid"
        else:
            outside = False
            for index, axis in enumerate(("x", "y", "z")):
                interval = self.workspace_bounds_m.get(axis)
                if (
                    not isinstance(interval, list) or len(interval) != 2
                    or not float(interval[0]) <= position[index] <= float(interval[1])
                ):
                    outside = True
                    break
            if not outside:
                return None
            code = "task_object_out_of_recoverable_workspace"
        return {
            "code": code,
            "severity": "terminal",
            "source": f"robotwin.{self.task}.recoverability_v1",
            "policy_visibility": "control_only",
        }

    def status(self) -> dict[str, Any]:
        environment = self._require_environment()
        signals: list[dict[str, str]] = []
        recoverability = self._task_recoverability_signal()
        if recoverability is not None:
            signals.append(recoverability)
        step_limit = getattr(environment, "step_lim", None)
        exhausted = (
            isinstance(step_limit, int)
            and int(environment.take_action_cnt) >= step_limit
        )
        budget_exhausted = (
            self.max_sim_steps is not None
            and int(environment.take_action_cnt) >= self.max_sim_steps
        )
        reason = (
            self.abort_reason
            or ("official_success" if bool(getattr(environment, "eval_success", False)) else None)
            or (recoverability["code"] if recoverability is not None else None)
            or ("robotwin_step_limit_exhausted" if exhausted else None)
            or ("max_sim_steps_per_episode_exhausted" if budget_exhausted else None)
        )
        terminal = reason is not None
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "terminal" if terminal else "running",
            "terminal": terminal,
            "reason": reason,
            "source": "robotwin_session",
            "policy_visibility": "control_only",
            "signals": signals,
        })

    @property
    def semantic_sim_steps(self) -> int:
        environment = self._require_environment()
        return int(environment.take_action_cnt)

    @property
    def sim_step_budget_exhausted(self) -> bool:
        return (
            self.max_sim_steps is not None
            and self.environment is not None
            and int(self.environment.take_action_cnt) >= self.max_sim_steps
        )

    def abort(self, reason: str) -> None:
        """Trusted fail-closed termination for a policy or budget failure."""

        self._require_environment()
        if self.is_terminal:
            raise RoboTwinLiveError("episode is already terminal")
        if not isinstance(reason, str) or not reason.strip():
            raise RoboTwinLiveError("abort reason must be non-empty")
        self.abort_reason = reason.strip()

    def persist_official_verdict(self) -> dict[str, Any]:
        """Persist the sole official boolean result after terminal transition."""

        environment = self._require_environment()
        status = self.status()
        if not status["terminal"]:
            raise RoboTwinLiveError("official verdict is unavailable before episode termination")
        self._capture_execution_boundary("terminal")
        success = bool(getattr(environment, "eval_success", False))
        if success:
            self._freeze_official_terminal(environment)
        payload = {
            "schema": VERDICT_SCHEMA,
            "task": self.task,
            "episode": self.episode,
            "actual_seed": self.actual_seed,
            "official_success": success,
            "source": "RoboTwin.eval_success",
            "official_terminal": copy.deepcopy(self.official_terminal),
            "eval_success_seen": bool(getattr(environment, "eval_success", False)),
            "semantic_action_count": int(environment.take_action_cnt),
            "trusted_abort_reason": (
                self.abort_reason
                or (status["reason"] if status["reason"] != "official_success" else None)
            ),
            "environment_status": status,
            "persisted_unix_s": time.time(),
        }
        path = self.output_root / "official-verdict.json"
        if path.exists():
            raise RoboTwinLiveError("refusing to overwrite official verdict")
        _atomic_json(path, payload)
        return {
            **payload,
            "artifact": {
                "path": path.relative_to(self.output_root).as_posix(),
                "sha256": _file_hash(path),
            },
        }

    def close(self) -> None:
        self._close_execution_capture()
        try:
            if self.environment is not None:
                self.environment.close_env(clear_cache=False)
        finally:
            self.environment = None
            self._restore_working_directory()

    def _restore_working_directory(self) -> None:
        previous = self.previous_working_directory
        self.previous_working_directory = None
        if previous is not None:
            os.chdir(previous)

    def _motion_backend_adapter(self, **kwargs: Any) -> Any:
        """Construct the RoboTwin motion adapter from this module's binding.

        Every mixin that plans or executes motion builds its adapter here, so
        ``RoboTwinMotionBackendAdapter`` stays one replaceable name.
        """
        return RoboTwinMotionBackendAdapter(**kwargs)


__all__ = [
    "DEFAULT_REVISION",
    "DEFAULT_WORKSPACE_BOUNDS_M",
    "OBSERVATION_SCHEMA",
    "RECEIPT_SCHEMA",
    "SPARSE_RECEIPT_SCHEMA",
    "RoboTwinLiveError",
    "RoboTwinLiveSession",
    "VERDICT_SCHEMA",
]
