"""Episode binding and round restart for one RoboTwin session.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

import math
from typing import Any

from ora.environment.controllers.motion_transaction import (
    validate_motion_backend_adapter,
)
from ora.environment.controllers.profile_capabilities import (
    loaded_control_profile_version,
)

from .common import RoboTwinLiveError, _atomic_json, _jsonable, _wxyz_to_xyzw


class RoboTwinRoundMixin:
    """Episode binding and round restart for one RoboTwin session."""

    def _bind_episode_state(self, environment: Any) -> None:
        """Bind one freshly reset environment and clear per-episode state.

        Every field cleared here is per-episode physical, planning or
        lifecycle state that must never cross a round boundary. Session-global
        artifact sequences are deliberately untouched: ``observation_sequence``,
        ``action_sequence``, ``motion_probe_sequence`` and both operator frame
        counters name evidence paths that are created with ``exist_ok=False``,
        and the controller replay recorder additionally requires an append-only
        contiguous action sequence. Rewinding one would collide with the
        previous round's evidence instead of isolating the new round.
        """

        self.environment = environment
        self.official_terminal = None
        self.terminal_observation = None
        self.terminal_public_poses = None
        self.terminal_gripper_targets = None
        self.physics_step_sequence = 0
        self.world_mutation_sequence = 0
        self.world_mutation_actions = []
        # A stale plan reference would let the policy consume a trajectory
        # planned against the previous round's world.
        self.plan_references = {}
        # The Pinocchio contexts cache the previous environment's articulation
        # entity and FK model; they must be rebuilt against the new scene.
        self.pinocchio_models = {}
        # RoboTwin reopens both grippers during its own reset, so a carried
        # directive would misreport the new round's policy-visible state.
        self.gripper_command_targets = {"left": "unknown", "right": "unknown"}
        # Backtrack targets are refreshed from the measured post-reset state
        # rather than cleared: recover_pose still needs a valid target, but it
        # must be this round's, not the previous round's end configuration.
        self.last_safe_pose = {
            arm: _wxyz_to_xyzw(self._public_pose_wxyz(environment, arm))
            for arm in ("left", "right")
        }
        self.home_joints = {}
        self.last_safe_joints = {}
        if self._uses_validated_planner_profile():
            self.home_joints = {
                arm: [
                    float(value) for value in getattr(
                        environment.robot, f"{arm}_homestate"
                    )
                ]
                for arm in ("left", "right")
            }
            if any(
                len(values) != len(self._arm_profile(arm)["planner_joint_names"])
                or not all(math.isfinite(value) for value in values)
                for arm, values in self.home_joints.items()
            ):
                raise RoboTwinLiveError("source-bound recovery home joints are invalid")
            self.last_safe_joints = {
                arm: self._measured_arm_joints(environment, arm)
                for arm in ("left", "right")
            }
            for arm in ("left", "right"):
                self._pinocchio_context(environment, arm)
        if self._uses_motion_transaction_profile():
            capabilities = validate_motion_backend_adapter(
                self._motion_backend_adapter(
                    session=self, environment=environment,
                )
            )
            profile = self.control_profile["profile"]  # type: ignore[index]
            if (
                capabilities.backend_id != "robotwin"
                or capabilities.embodiment_id != profile["embodiment_id"]
                or tuple(profile["arms"]) != capabilities.arms
                or capabilities.controlled_point
                != "parallel_jaw_grip_center_tcp"
                or not any(
                    set(group) == set(capabilities.arms)
                    for group in capabilities.synchronized_groups
                )
            ):
                raise RoboTwinLiveError(
                    "motion transaction adapter and ALOHA profile are incompatible"
                )

    def restart_round(
        self, round_index: int, *, seed: int | None = None,
    ) -> dict[str, Any]:
        """Reset the same isolated simulator for another independent attempt.

        Move paths use the trusted ``abort_reason`` channel, which is round-local.

        Artifact sequences stay session-global so no observation, receipt,
        motion probe, plan or operator frame can collide across rounds.
        """

        if isinstance(round_index, bool) or not isinstance(round_index, int):
            raise RoboTwinLiveError("RoboTwin round index must be an integer")
        if round_index != self.round_index + 1:
            raise RoboTwinLiveError("RoboTwin round reset must advance by exactly one")
        self._require_environment()
        if (self.output_root / "official-verdict.json").exists():
            raise RoboTwinLiveError(
                "previous round verdict must be archived before environment reset"
            )
        selected_seed = self.actual_seed if seed is None else seed
        if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
            raise RoboTwinLiveError("RoboTwin round seed must be an integer")
        if self._execution_step_tap is not None:
            self._execution_step_tap.close()
            self._execution_step_tap = None
        previous = self.environment
        self.environment = None
        self.pinocchio_models = {}
        try:
            previous.close_env(clear_cache=False)
        except Exception as exc:
            raise RoboTwinLiveError(
                f"previous round environment failed to close: {exc}"
            ) from exc
        environment = self._construct_environment(selected_seed)
        self.actual_seed = selected_seed
        self._bind_episode_state(environment)
        self.round_index = round_index
        self._verify_loaded_robot_geometry(environment)
        self.replay_recorder.begin_round(round_index)
        self._start_execution_capture()
        # The trusted stop channel and the last public receipt are per-episode:
        # a carried abort would make the new round terminal at step 0, and a
        # carried receipt would show the next observation a result the freshly
        # reset world no longer holds.
        self.abort_reason = None
        self.last_receipt = None
        self._capture_operator_reset_frames(label="round-reset")
        reset = {
            "schema": "ora.robotwin-round-reset.v1",
            "task": self.task,
            "task_config": self.task_config,
            "episode": self.episode,
            "actual_seed": self.actual_seed,
            "seed": self.actual_seed,
            "round_index": round_index,
            "official_predicate_at_reset": False,
            "source_manifest": self.source_manifest,
            "workspace_bounds_m": _jsonable(self.workspace_bounds_m),
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
            "public_environment_context": None,
            "environment_signal_policy": self.environment_signal_policy,
            "operator_world_video_enabled": self.operator_world_video,
            "operator_execution_video_enabled": self.operator_execution_video,
            "execution_capture": self.execution_capture,
            "execution_capture_error": self._execution_capture_error,
            "artifact_sequences_preserved": {
                "observation_sequence": self.observation_sequence,
                "action_sequence": self.action_sequence,
                "motion_probe_sequence": self.motion_probe_sequence,
                "operator_frame_sequence": self.operator_frame_sequence,
                "operator_execution_frame_sequence": (
                    self.operator_execution_frame_sequence
                ),
            },
        }
        _atomic_json(
            self.output_root / "round-resets" / f"round-{round_index:04d}.json",
            reset,
        )
        return reset
