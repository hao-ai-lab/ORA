"""Privilege-clean direct LIBERO/LIBERO-Pro session.

RPent informed the reset and OSC conventions during implementation but is not a
runtime dependency. The Harness keeps planning, task source, object ground
truth and success predicates outside this worker and exposes only metric RGB-D,
calibration, proprioception and the shared absolute-pose action contract.
"""

from __future__ import annotations

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


from ora.contracts import (
    MOTION_ACTION_SCHEMA,
    MOTION_ACTION_V3_SCHEMA,
    POSE_ORIENTATION_TOLERANCE_RAD,
    POSE_POSITION_TOLERANCE_M,
    compile_absolute_pose,
)
from ora.environment.controllers.curobo_motion import (
    CuroboPlannerClient, _curobo_failure_reason, _curobo_valid_query,
)
from ora.environment.backends.libero_pro.motion_settle import (
    GRIPPER_STALL_REASON, gripper_outcome, gripper_stalled,
    merge_settled_receipt, precontact_pose, precontact_refusal,
    public_gripper_command, settle_refusal,
)
from ora.environment.controllers.hybrid_motion import (
    LIBERO_ENDPOINT_ROUTE_GLOBAL,
    LIBERO_ENDPOINT_ROUTE_LOCAL,
    aggregate_motion_contact,
    bounded_joint_position_command,
    libero_endpoint_contract,
    libero_global_endpoint_tolerances,
    libero_local_endpoint_tolerances,
    motion_stop_reason,
    precision_force_stop_enabled,
    validate_hybrid_motion_profile,
)
from ora.environment.controllers.force_feedback import (
    precision_force_stop_reason, public_force_direction, sensor_reaction_feedback,
)
from ora.environment.controllers.command_drift import CommandDriftTracker
from ora.contracts.action import (
    SERVO_MAX_ROTATION_RAD,
    SERVO_MAX_TRANSLATION_M,
    validate_runtime_action,
)
from ora.contracts.acceptance import motion_acceptance_check, gripper_acceptance_check
from ora.contracts.workspace import (
    is_monotonic_workspace_recovery,
    normalize_workspace_bounds,
    workspace_state,
)
from ora.environment.base import (
    ENVIRONMENT_STATUS_SCHEMA,
    validate_environment_status,
)
from ora.environment.controller_replay import ControllerReplayRecorder
from ora.environment.execution_capture import resolve_execution_capture
from .execution_capture import LiberoExecutionCaptureMixin, LiberoWorkerCaptureMixin
from ora.environment.controllers.libero_osc import (
    GRIPPER_TOLERANCE_NORMALIZED,
    CONTROL_HEALTH_NUMERIC_TOLERANCE,
    GROSS_ORIENTATION_GROWTH_RAD,
    GROSS_SAFETY_ENVELOPE_OVERRUN_M,
    GROSS_SINGLE_TICK_TRANSLATION_M,
    OSC_ACTION_CONTRACT_SCHEMA,
    TRACKING_CONSECUTIVE_TICKS,
    TRACKING_ORIENTATION_GROWTH_TELEMETRY_RAD,
    TRACKING_POSITION_GROWTH_TELEMETRY_M,
    TRACKING_TRAVEL_OVERSHOOT_TELEMETRY_M,
    inverse_controller_scale as _inverse_controller_scale,
    libero_osc_action_contract,
)

# Every name importable from this module before the split stays importable:
# the imports above and the re-exports below. The mixins import what they use.
# The lifecycle methods that write episode evidence (start, restart_round,
# _latch_environment_outcome, persist_official_verdict) stay here and read
# ``_atomic_json`` from this module, where tests replace it.
from .common import (
    OBSERVATION_SCHEMA,
    RECEIPT_SCHEMA,
    HYBRID_RECEIPT_SCHEMA,
    LIBERO_SPARSE_RECEIPT_SCHEMA,
    VERDICT_SCHEMA,
    RESET_SCHEMA,
    CONTROL_HEALTH_SCHEMA,
    TRACKING_EVIDENCE_SCHEMA,
    LIBERO_POLICY_CAMERA_NAMES,
    LIBERO_EMBODIMENTS,
    _sparse_planning_failure,
    _public_gripper_command,
    _within_local_servo_envelope,
    LiberoProError,
    bddl_task_language,
    resolve_episode_task_language,
    libero_environment_init_params,
    libero_public_environment_context,
    _jsonable,
    _atomic_json,
    _file_hash,
    _git,
    _metric_depth,
    _quaternion_error_rotvec,
    _pose_tracking_sample,
    _normalized_gripper,
)
from .worker_environment import _CameraMetadataEnvironment, _camera_metadata_env_factory
from .pinned_imports import (  # noqa: F401  re-exported for construction, preview, probe and tests
    _benchmark_data_binding, _path_is_within, _prepare_pinned_libero_imports,
)
from .construction import build_environment as _build_environment

from .source_binding import LiberoSourceBindingMixin
from .observation import LiberoObservationMixin
from .motion_evidence import LiberoMotionEvidenceMixin
from .local_motion import LiberoLocalMotionMixin
from .local_motion_report import LiberoLocalMotionReportMixin
from .planned_motion import LiberoPlannedMotionMixin
from .motion_probe import LiberoMotionProbeMixin
from .probe_feedback import LiberoProbeFeedbackMixin
from .reach_check import LiberoReachCheckMixin
from .sparse_motion import LiberoSparseMotionMixin


@dataclass
class LiberoProSession(
    LiberoSourceBindingMixin,
    LiberoObservationMixin,
    LiberoMotionEvidenceMixin,
    LiberoLocalMotionMixin,
    LiberoLocalMotionReportMixin,
    LiberoPlannedMotionMixin,
    LiberoReachCheckMixin,
    LiberoProbeFeedbackMixin,
    LiberoMotionProbeMixin,
    LiberoSparseMotionMixin,
    LiberoExecutionCaptureMixin,
):
    """One trusted, isolated LIBERO-Pro episode writer."""

    libero_pro_root: Path
    data_root: Path
    output_root: Path
    task: str
    seed: int
    suite: str
    task_id: int
    expected_task_language: str
    required_libero_pro_revision: str
    required_data_revision: str
    workspace_bounds_m: Mapping[str, Any]
    max_sim_steps: int
    robustness_case: str | None = None
    environment_driver: str = "direct_libero_v1"
    workspace_profile: Mapping[str, Any] | None = None
    qualified_controller_binding: Mapping[str, Any] | None = None
    servo_max_steps: int = 80
    execution_capture: Mapping[str, Any] | None = None
    _execution_capture_error: str | None = field(init=False, default=None)
    action_codec: str = "runtime_action_v3"
    motion_planner_config: Mapping[str, Any] | None = None
    motion_profile_config: Mapping[str, Any] | None = None
    public_robot_geometry: Mapping[str, Any] | None = None
    robot_geometry_verification: dict[str, Any] | None = field(init=False, default=None)
    environment_factory: Callable[..., tuple[Any, dict[str, Any]]] = _build_environment
    environment: Any = field(init=False, default=None, repr=False)
    metadata: dict[str, Any] = field(init=False, default_factory=dict)
    last_raw_observation: dict[str, Any] | None = field(init=False, default=None)
    observation_sequence: int = field(init=False, default=0)
    action_sequence: int = field(init=False, default=0)
    round_index: int = field(init=False, default=0)
    sim_steps: int = field(init=False, default=0)
    terminated: bool = field(init=False, default=False)
    truncated: bool = field(init=False, default=False)
    first_termination: dict[str, Any] | None = field(init=False, default=None)
    abort_reason: str | None = field(init=False, default=None)
    control_fault_reason: str | None = field(init=False, default=None)
    last_receipt: dict[str, Any] | None = field(init=False, default=None)
    controller_action_contract: dict[str, Any] = field(
        init=False, default_factory=dict
    )
    robustness_episode: Any = field(init=False, default=None, repr=False)
    motion_planner: CuroboPlannerClient | None = field(
        init=False, default=None, repr=False
    )
    home_joint_state: list[float] | None = field(init=False, default=None)
    home_tcp_pose: list[float] | None = field(init=False, default=None)
    last_backtrack_tcp_pose: list[float] | None = field(init=False, default=None)
    command_drift_tracker: Any = field(init=False, default=None, repr=False)
    last_backtrack_joint_state: list[float] | None = field(
        init=False, default=None
    )
    motion_profile: dict[str, Any] | None = field(init=False, default=None)
    replay_recorder: ControllerReplayRecorder = field(init=False, repr=False)
    world_mutation_sequence: int = field(init=False, default=0)
    world_mutation_actions: list[str] = field(init=False, default_factory=list)
    motion_probe_sequence: int = field(init=False, default=0)
    curobo_probe_sequence: int = field(init=False, default=0)
    plan_references: dict[str, dict[str, Any]] = field(
        init=False, default_factory=dict, repr=False,
    )

    def __post_init__(self) -> None:
        self.execution_capture = resolve_execution_capture(self.execution_capture)
        if self.environment_driver != "direct_libero_v1":
            raise LiberoProError("unsupported LIBERO environment driver")
        self.libero_pro_root = self.libero_pro_root.expanduser().resolve()
        self.data_root = self.data_root.expanduser().resolve()
        self.output_root = self.output_root.expanduser().resolve()
        from ora.environment.benchmarks.libero_pro import (
            LiberoProBenchmarkError,
            validate_runtime_extension,
        )

        try:
            validate_runtime_extension({
                "suite": self.suite,
                "task_id": self.task_id,
                "robustness_case": self.robustness_case,
                "required_libero_pro_revision": self.required_libero_pro_revision,
            })
        except LiberoProBenchmarkError as exc:
            raise LiberoProError(str(exc)) from exc
        if self.robustness_case is not None:
            from .robustness import ROBUSTNESS_CASES, ROBUSTNESS_CODE_REVISION

            if self.robustness_case not in ROBUSTNESS_CASES:
                raise LiberoProError(
                    f"unsupported LIBERO robustness case: {self.robustness_case!r}"
                )
            if self.required_libero_pro_revision != ROBUSTNESS_CODE_REVISION:
                raise LiberoProError(
                    "robustness case and LIBERO-Pro runtime revision are not paired"
                )
        if self.max_sim_steps <= 0 or self.servo_max_steps <= 0:
            raise LiberoProError("step budgets must be positive")
        self.expected_task_language = self.expected_task_language.strip()
        if not self.expected_task_language:
            raise LiberoProError("expected task language must be non-empty")
        self.workspace_bounds_m = normalize_workspace_bounds(self.workspace_bounds_m)
        if self.action_codec == "motion_action_v3":
            # Sparse motion v3 reuses the qualified CuRobo global
            # route and OSC local servo hybrid bindings.
            if not isinstance(self.motion_planner_config, Mapping):
                raise LiberoProError(
                    f"{self.action_codec} requires a CuRobo planner binding"
                )
            if not isinstance(self.motion_profile_config, Mapping):
                raise LiberoProError(
                    f"{self.action_codec} requires a motion profile binding"
                )
            profile = self.motion_profile_config.get("profile")
            path = self.motion_profile_config.get("path")
            digest = self.motion_profile_config.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str):
                raise LiberoProError("motion profile provenance is malformed")
            self.motion_profile = validate_hybrid_motion_profile(profile)
        elif self.motion_planner_config is not None or self.motion_profile_config is not None:
            raise LiberoProError(
                "hybrid motion bindings are only valid for sparse motion codecs"
            )
        if self.workspace_profile is not None:
            if not isinstance(self.workspace_profile, Mapping):
                raise LiberoProError("workspace profile binding must be an object")
            if self.workspace_profile.get("bounds_m") != self.workspace_bounds_m:
                raise LiberoProError("workspace profile and action bounds differ")
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.source_manifest = self._verify_source()
        self.replay_recorder = ControllerReplayRecorder(
            output_root=self.output_root,
            backend="libero_pro",
            environment_identity={
                "task": self.task,
                "suite": self.suite,
                "task_id": self.task_id,
                "seed": self.seed,
                "robustness_case": self.robustness_case,
                "action_codec": self.action_codec,
                "source_manifest": self.source_manifest,
            },
            camera_panels=["agentview", "wrist"],
        )

    def start(self) -> dict[str, Any]:
        if self.environment is not None:
            raise LiberoProError("LIBERO-Pro episode already started")
        environment, metadata = self.environment_factory(
            libero_pro_root=self.libero_pro_root,
            data_root=self.data_root,
            runtime_config_root=self.output_root / "runtime-config/libero",
            suite=self.suite, task_id=self.task_id, seed=self.seed,
            max_sim_steps=self.max_sim_steps,
            robustness_case=self.robustness_case,
            **({"environment_driver": self.environment_driver}
               if self.environment_factory is _build_environment else {}),
        )
        self.environment = environment
        try:
            contract = metadata.get("controller_action_contract")
            if not isinstance(contract, Mapping):
                raise LiberoProError("environment omits its OSC action contract")
            validated_contract = libero_osc_action_contract({
                "control_delta": contract.get("control_delta"),
                "input_min": contract.get("input_min"),
                "input_max": contract.get("input_max"),
                "output_min": contract.get("output_min"),
                "output_max": contract.get("output_max"),
            })
            if validated_contract != dict(contract):
                raise LiberoProError("environment OSC action contract identity drifted")
            self.controller_action_contract = validated_contract
            self.metadata = dict(metadata)
            self._bind_dependency_files()
            self._validate_qualified_controller_binding()
            _obs, _info = environment.reset()
            self.last_raw_observation = self._raw_observation()
            self._verify_loaded_robot_geometry()
            if self.action_codec == "motion_action_v3":
                vector_environment = getattr(environment, "env", None)
                setter = getattr(vector_environment, "set_env_attr", None)
                if not callable(setter):
                    raise LiberoProError("motion profile transport is unavailable")
                setter("harness_motion_profile", self.motion_profile)
                config = dict(self.motion_planner_config or {})
                self.motion_planner = CuroboPlannerClient(
                    python_executable=Path(str(config["python_executable"])),
                    curobo_root=Path(str(config["curobo_root"])),
                    required_revision=str(config["required_revision"]),
                    output_root=self.output_root / "curobo",
                    device=str(config.get("device", "cuda:0")),
                    request_timeout_s=float(config.get("request_timeout_s", 180.0)),
                )
                planner_binding = self.motion_planner.start()
                motion_state = self._motion_state()
                self.home_joint_state = list(motion_state["joint_positions"])
                self.last_backtrack_joint_state = list(self.home_joint_state)
                self.home_tcp_pose = list(motion_state["grip_site_pose_world_xyzw"])
                self.last_backtrack_tcp_pose = list(self.home_tcp_pose)
                self.metadata["public_environment_context"] = {
                    **dict(self.metadata["public_environment_context"]),
                    "controller": "CUROBO_JOINT_POSITION+OSC_POSE",
                }
                self.metadata["motion_planner_binding"] = planner_binding
                self.metadata["motion_profile_binding"] = {
                    "path": str(self.motion_profile_config["path"]),
                    "sha256": str(self.motion_profile_config["sha256"]),
                    "profile_id": str(self.motion_profile["profile_id"]),
                }
            robustness_state = None
            if self.robustness_case is not None:
                from .robustness import RobustnessEpisode

                self.robustness_episode = RobustnessEpisode(self.robustness_case)
                robustness_state = _jsonable(
                    self.robustness_episode.start(environment)
                )
                _atomic_json(
                    self.output_root / "robustness/application.json",
                    robustness_state,
                )
            actual_language, bddl_language = resolve_episode_task_language(
                metadata, getattr(environment, "task_descriptions", None), self.expected_task_language,
            )
            self.metadata = {
                **self.metadata, "task_language": actual_language,
                "registry_task_language": str(metadata["task_language"]).strip(),
                "bddl_task_language": bddl_language,
                "robustness_state": robustness_state,
            }
        except Exception:
            self.close()
            raise
        self._capture_command(
            "init", output_root=str(self.output_root), config=self.execution_capture,
            identity={"run_id": self.output_root.parent.name, "task": self.task,
                      "source_manifest": self.source_manifest},
        )
        self._capture_reset()
        reset = {
            "schema": RESET_SCHEMA,
            "execution_capture": self.execution_capture,
            "execution_capture_error": self._execution_capture_error,
            "task": self.task,
            "suite": self.suite,
            "task_id": self.task_id,
            "seed": self.seed,
            "reset_state_id": self.metadata.get("reset_id"),
            "task_language": self.expected_task_language,
            "official_predicate_at_reset": None,
            "source_manifest": self.source_manifest,
            "workspace_bounds_m": _jsonable(self.workspace_bounds_m),
            "workspace_profile": (
                None if self.workspace_profile is None
                else _jsonable(self.workspace_profile)
            ),
            "controller_action_contract": self.controller_action_contract,
            "motion_planner_binding": self.metadata.get("motion_planner_binding"),
            "motion_profile_binding": self.metadata.get("motion_profile_binding"),
            "qualified_controller_binding": (
                None if self.qualified_controller_binding is None
                else _jsonable(self.qualified_controller_binding)
            ),
            "public_environment_context": dict(
                self.metadata["public_environment_context"]
            ),
            "robustness": self.metadata.get("robustness_state"),
        }
        _atomic_json(self.output_root / "reset.json", reset)
        return reset

    def _reset_with_seed(self, seed: int) -> None:
        """Rebind one fixed-task LIBERO reset without replacing the Session."""

        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise LiberoProError("LIBERO round seed must be a non-negative integer")
        environment = self._require_environment()
        if seed == self.seed:
            environment.reset()
            return
        if self.robustness_case is not None:
            raise LiberoProError(
                "per-round seed rebinding is not qualified for robustness cases"
            )
        trial_count = self.metadata.get("trial_count")
        current_reset_id = self.metadata.get("reset_id")
        if (
            isinstance(trial_count, bool)
            or not isinstance(trial_count, int)
            or trial_count <= 0
            or isinstance(current_reset_id, bool)
            or not isinstance(current_reset_id, int)
        ):
            raise LiberoProError("LIBERO reset-state metadata is unavailable")
        import numpy as np

        first_reset_id = current_reset_id - (self.seed % trial_count)
        reset_state_id = first_reset_id + (seed % trial_count)
        environment.seed = seed
        if hasattr(environment, "cfg"):
            environment.cfg.seed = seed
            environment.cfg.specific_reset_id = reset_state_id
        environment.specific_reset_id = reset_state_id
        if hasattr(environment, "_generator"):
            environment._generator = np.random.default_rng(seed=seed)
        environment.start_idx = 0
        reset_state_ids = np.asarray([reset_state_id], dtype=np.int64)
        if hasattr(environment, "reset_state_ids"):
            environment.reset_state_ids = reset_state_ids.copy()
        environment.reset(reset_state_ids=reset_state_ids)
        self.seed = seed
        self.metadata["reset_id"] = reset_state_id

    def restart_round(
        self, round_index: int, *, seed: int | None = None,
    ) -> dict[str, Any]:
        """Reset the same isolated simulator for another independent attempt.

        Artifact sequences remain session-global so no observation or receipt can
        collide across rounds. Physical/lifecycle state and the simulator's
        per-episode step budget are reset.
        """

        if round_index != self.round_index + 1:
            raise LiberoProError("LIBERO round reset must advance by exactly one")
        self._require_environment()
        if (self.output_root / "official-verdict.json").exists():
            raise LiberoProError(
                "previous round verdict must be archived before environment reset"
            )
        selected_seed = self.seed if seed is None else seed
        self._reset_with_seed(selected_seed)
        self.last_raw_observation = self._raw_observation()
        self.round_index = round_index
        self._verify_loaded_robot_geometry()
        self.replay_recorder.begin_round(round_index)
        self._capture_reset()
        self.sim_steps = 0
        self.terminated = False
        self.truncated = False
        self.first_termination = None
        self.abort_reason = None
        # A prior round's abort persists control/stop.request, and motion
        # executors poll that file on every control tick. A stale file
        # silently converts every later-round global motion into
        # operator_stop_requested at step 0 while the OSC local path keeps
        # working, teaching the policy a false "global is broken" lesson.
        (self.output_root / "control/stop.request").unlink(missing_ok=True)
        # Motion state must not cross the round boundary either: a stale
        # backtrack target would let recover_pose(previous) drive toward the
        # previous round's end configuration, and a stale plan reference
        # could replay a trajectory planned against the previous episode.
        if self.motion_planner is not None:
            self._remember_backtrack_state()
        self.plan_references.clear()
        self.command_drift_tracker = None
        self.control_fault_reason = None
        self.last_receipt = None
        reset = {
            "schema": "ora.libero-pro-round-reset.v1",
            "task": self.task,
            "suite": self.suite,
            "task_id": self.task_id,
            "seed": self.seed,
            "reset_state_id": self.metadata.get("reset_id"),
            "round_index": round_index,
            "task_language": self.expected_task_language,
            "official_predicate_at_reset": None,
            "source_manifest": self.source_manifest,
            "workspace_bounds_m": _jsonable(self.workspace_bounds_m),
            "workspace_profile": (
                None if self.workspace_profile is None
                else _jsonable(self.workspace_profile)
            ),
            "controller_action_contract": self.controller_action_contract,
            "qualified_controller_binding": (
                None if self.qualified_controller_binding is None
                else _jsonable(self.qualified_controller_binding)
            ),
            "public_environment_context": dict(
                self.metadata["public_environment_context"]
            ),
        }
        _atomic_json(
            self.output_root / "round-resets" / f"round-{round_index:04d}.json",
            reset,
        )
        return reset

    def _require_environment(self) -> Any:
        if self.environment is None:
            raise LiberoProError("LIBERO-Pro episode is not started")
        return self.environment

    def _latch_environment_outcome(
        self, *, terminated: Any, truncated: Any, steps: int,
        action: Mapping[str, Any], controller: str,
    ) -> None:
        """Keep official termination before fallible evidence/observation work."""

        import numpy as np

        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise LiberoProError("executed step count is malformed")
        term = bool(np.asarray(terminated).any())
        trunc = bool(np.asarray(truncated).any())
        self.sim_steps += steps
        self.terminated = self.terminated or term
        self.truncated = self.truncated or trunc
        if (term or trunc) and self.first_termination is None:
            self.first_termination = {
                "schema": "ora.libero-first-termination.v1",
                "source": "LIBERO environment termination",
                "round_index": self.round_index,
                "action_id": action["action_id"],
                "action_sequence": self.action_sequence,
                "controller": controller,
                "sim_step": self.sim_steps,
                "terminated": term, "truncated": trunc,
            }
            path = (
                self.output_root / "rounds" / f"round-{self.round_index:04d}"
                / "first-termination.json"
            )
            _atomic_json(path, self.first_termination)

    @property
    def is_terminal(self) -> bool:
        return bool(self.status()["terminal"])

    def status(self) -> dict[str, Any]:
        reason = (
            self.abort_reason
            or ("official_success" if self.terminated else None)
            or self.control_fault_reason
            or ("libero_truncated" if self.truncated else None)
            or ("max_sim_steps_per_episode_exhausted"
                if self.sim_steps >= self.max_sim_steps else None)
        )
        terminal = reason is not None
        return validate_environment_status({
            "schema": ENVIRONMENT_STATUS_SCHEMA,
            "phase": "terminal" if terminal else "running",
            "terminal": terminal,
            "reason": reason,
            "source": "libero_pro_session",
            "policy_visibility": "control_only",
            "signals": (
                [] if self.control_fault_reason is None else [{
                    "code": self.control_fault_reason,
                    "severity": "terminal",
                    "source": "libero_pro.control_health.v2",
                    "policy_visibility": "control_only",
                }]
            ),
        })

    def abort(self, reason: str) -> None:
        if not reason:
            raise LiberoProError("abort reason must be non-empty")
        self.abort_reason = reason
        if self.action_codec == "motion_action_v3":
            stop_path = self.output_root / "control/stop.request"
            stop_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.write_text(reason.strip() + "\n", encoding="utf-8")

    def persist_official_verdict(self) -> dict[str, Any]:
        if self.is_terminal:
            self._capture_command("terminal")
        verdict = {
            "schema": VERDICT_SCHEMA,
            "official_success": bool(self.terminated),
            "suite": self.suite,
            "task_id": self.task_id,
            "seed": self.seed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "abort_reason": self.abort_reason,
            "first_termination": self.first_termination,
            "control_fault_reason": self.control_fault_reason,
            "sim_steps": self.sim_steps,
            "robustness_case": self.robustness_case,
            "robustness_events": (
                0 if self.robustness_episode is None
                else self.robustness_episode.event_sequence
            ),
            "source": "LIBERO environment termination",
            "created_unix_s": time.time(),
        }
        _atomic_json(self.output_root / "official-verdict.json", verdict)
        return verdict

    def close(self) -> None:
        self._capture_command("close")
        planner = self.motion_planner
        self.motion_planner = None
        if planner is not None:
            planner.close()
        environment = self.environment
        self.environment = None
        if environment is None:
            return
        close = getattr(environment, "close", None)
        if callable(close):
            close()
