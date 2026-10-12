"""Motion-transaction dense execution and its control and planner evidence.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ora.environment.backends.robotwin.planning import planner_public_summary
from ora.environment.controllers.inter_arm_clearance import (
    InterArmClearancePolicy,
    minimum_inter_arm_distance as minimum_inter_arm_distance_value,
)
from ora.environment.controllers.motion_transaction import (
    MotionExecutionFault,
    MotionTransactionError,
    execute_measured_convergence,
    require_motion_capabilities,
)
from ora.environment.controllers.profile_capabilities import (
    CONTACT_RECOVERY_SINCE,
    profile_has,
)

from .common import (
    RoboTwinControllerTrackingError,
    RoboTwinLiveError,
    _SparseMotionState,
    _jsonable,
)


class RoboTwinMotionTransactionMixin:
    """Motion-transaction dense execution and its control and planner evidence."""

    def _begin_sparse_decision(self, environment: Any) -> None:
        if bool(getattr(environment, "eval_success", False)):
            raise RoboTwinLiveError("episode is already terminal")
        step_limit = getattr(environment, "step_lim", None)
        if isinstance(step_limit, int) and int(environment.take_action_cnt) >= step_limit:
            raise RoboTwinLiveError("RoboTwin step limit is exhausted")
        if (
            self.max_sim_steps is not None
            and int(environment.take_action_cnt) >= self.max_sim_steps
        ):
            raise RoboTwinLiveError("Harness simulator-step budget is exhausted")
        video = getattr(environment, "eval_video_ffmpeg", None)
        now_obs = getattr(environment, "now_obs", None)
        if (
            getattr(environment, "eval_video_path", None) is not None
            and video is not None
            and isinstance(now_obs, Mapping)
        ):
            video.stdin.write(now_obs["observation"]["head_camera"]["rgb"].tobytes())
        environment.take_action_cnt = int(environment.take_action_cnt) + 1
        updater = getattr(environment, "_update_render", None)
        if callable(updater):
            updater()

    def _record_ik_candidate_evidence(self, result: Any, *, arm: str,
                                      identity: Mapping[str, Any]) -> None:
        """Persist all returned-seed filter evidence even when no plan executes."""
        if not isinstance(result, dict) or not isinstance(result.get("ik"), Mapping):
            return
        raw = result["ik"]
        if not isinstance(raw.get("candidate_validation"), list):
            return
        payload = {
            "schema": "ora.robotwin-ik-candidate-validation.v1",
            "arm": arm, "identity": dict(identity),
            "robotwin_revision": self.source_manifest["robotwin_revision"],
            "control_profile_sha256": self.control_profile["sha256"],
            "candidate_validation": raw["candidate_validation"],
            "planned_source_indices": raw.get("planned_source_indices"),
            "constraint_rejection_counts": raw.get("constraint_rejection_counts"),
        }
        data = json.dumps(payload, sort_keys=True, allow_nan=False).encode() + b"\n"
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("planning-diagnostics") / f"{digest}.json"
        path = self.output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise RoboTwinLiveError("IK diagnostic artifact content mismatch")
        else:
            path.write_bytes(data)
        result["ik_validation_artifact"] = {
            "path": relative.as_posix(), "sha256": digest,
            "schema": payload["schema"], "bytes": len(data),
        }

    @staticmethod
    def _planner_record(result: Any, *, called: bool) -> dict[str, Any]:
        if not called:
            return {"called": False, "status": "not_called", "path_steps": 0}
        if isinstance(result, Mapping) and result.get("schema") in {
            "robotwin.curobo-plan-result.v1",
            "robotwin.curobo-plan-result.v2",
        }:
            return planner_public_summary(result)
        status = result.get("status") if isinstance(result, Mapping) else None
        positions = result.get("position") if isinstance(result, Mapping) else None
        shape = getattr(positions, "shape", None)
        return {
            "called": True,
            "status": str(status) if status is not None else "invalid_result",
            "path_steps": int(shape[0]) if shape is not None and len(shape) else 0,
        }

    @staticmethod
    def _replay_control_payload(
        control_sequence: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], int]:
        """Flatten one upstream dense command into compact numeric arrays."""

        arrays: dict[str, Any] = {}
        structure: dict[str, Any] = {}
        physics_steps = 0
        for channel in (
            "left_arm", "left_gripper", "right_arm", "right_gripper",
        ):
            value = control_sequence.get(channel)
            if value is None:
                structure[channel] = None
                continue
            if not isinstance(value, Mapping):
                raise RoboTwinLiveError(
                    f"RoboTwin dense-control channel {channel} is malformed"
                )
            description: dict[str, Any] = {}
            for field, item in value.items():
                if field in {"position", "velocity", "result", "target"}:
                    key = f"{channel}__{field}"
                    arrays[key] = item
                    description[field] = {"array": key}
                    if field != "target":
                        shape = getattr(item, "shape", None)
                        if shape is not None and len(shape):
                            physics_steps = max(physics_steps, int(shape[0]))
                        elif isinstance(item, (list, tuple)):
                            physics_steps = max(physics_steps, len(item))
                else:
                    description[field] = _jsonable(item)
            steps = value.get("num_step")
            if isinstance(steps, int):
                physics_steps = max(physics_steps, steps)
            structure[channel] = description
        return arrays, structure, physics_steps

    def _record_control_sequence(
        self, state: _SparseMotionState,
        control_sequence: Mapping[str, Any], *, replayable: bool,
    ) -> None:
        """Record one executed controller sequence for replay."""

        arrays, structure, physics_steps = self._replay_control_payload(
            control_sequence
        )
        arrays.update(state.controller_tracking_arrays)
        self.replay_recorder.record(
            action_sequence=self.action_sequence,
            action=state.action,
            controller=(
                "robotwin_local_cartesian_dls"
                if state.local_cartesian_requested
                else "robotwin_global_precontact_then_local"
                if state.global_precontact_then_local_requested
                else "robotwin_global_then_local_settle"
                if state.global_then_local_settle_requested
                else "robotwin_dense_joint_targets"
            ),
            arrays=arrays,
            physics_steps=physics_steps,
            replayable=replayable,
            metadata={
                "control_sequence": structure,
                "replay_route": (
                    "motion_transaction_local_cartesian_v1"
                    if self._uses_motion_transaction_profile()
                    and state.local_cartesian_requested
                    else "motion_transaction_global_precontact_then_local_v1"
                    if self._uses_motion_intent_profile()
                    and state.global_precontact_then_local_requested
                    else "motion_transaction_global_then_local_settle_v1"
                    if self._uses_motion_intent_profile()
                    and state.global_then_local_settle_requested
                    else "motion_transaction_global_joint_v1"
                    if self._uses_motion_transaction_profile()
                    else "harness_local_cartesian_dls_v1"
                    if state.local_cartesian_requested
                    else "harness_waypoint_convergence_v1"
                    if self._uses_validated_planner_profile()
                    else "take_dense_action(save_freq=1)"
                ),
                "controller_tracking": state.controller_tracking,
                "plan_artifacts": state.plan_artifacts,
            },
        )
        state.replay_recorded = True

    def _execute_motion_transaction_dense_action(
        self, *, environment: Any, control_sequence: Mapping[str, Any],
        action: Mapping[str, Any],
        endpoint_public_tcp_targets: Mapping[str, Sequence[float]] | None = None,
        endpoint_position_tolerance_m: float | None = None,
        endpoint_orientation_tolerance_rad: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Execute v5 global paths through the shared measured transaction."""

        import numpy as np

        adapter = self._motion_backend_adapter(session=self, environment=environment)
        profile = self.control_profile["profile"]["motion_transaction"][  # type: ignore[index]
            "global_convergence"
        ]

        def path_length(value: Any) -> int:
            if not isinstance(value, Mapping):
                return 0
            positions = value.get("position")
            shape = getattr(positions, "shape", None)
            if shape is not None and len(shape):
                return int(shape[0])
            steps = value.get("num_step")
            return int(steps) if isinstance(steps, int) else 0

        stage_count = max(
            path_length(control_sequence.get(channel))
            for channel in (
                "left_arm", "left_gripper", "right_arm", "right_gripper",
            )
        )
        if stage_count <= 0:
            raise RoboTwinLiveError("motion transaction received an empty path")
        paths = {
            arm: {
                "position": control_sequence[f"{arm}_arm"]["position"],
                "velocity": control_sequence[f"{arm}_arm"]["velocity"],
            }
            for arm in ("left", "right")
            if isinstance(control_sequence.get(f"{arm}_arm"), Mapping)
        }
        if paths:
            require_motion_capabilities(
                adapter.capabilities,
                requested_arms=tuple(paths),
                require_global=True,
                require_synchronized=len(paths) > 1,
            )
        issued_gripper: dict[str, list[float]] = {"left": [], "right": []}

        def replay_from_arrays(value: Mapping[str, Any]) -> dict[str, Any]:
            replay: dict[str, Any] = {}
            for arm in ("left", "right"):
                position = np.asarray(
                    value.get(f"issued_{arm}_arm__position", []),
                    dtype=np.float64,
                )
                velocity = np.asarray(
                    value.get(f"issued_{arm}_arm__velocity", []),
                    dtype=np.float64,
                )
                replay[f"{arm}_arm"] = (
                    {
                        "status": "IssuedMotionTransaction",
                        "position": position,
                        "velocity": velocity,
                    }
                    if len(position) else None
                )
                if issued_gripper[arm]:
                    gripper_values = np.asarray(
                        issued_gripper[arm], dtype=np.float64,
                    )
                    replay[f"{arm}_gripper"] = {
                        "num_step": int(gripper_values.shape[0]),
                        "per_step": 0.0,
                        "result": gripper_values,
                        "target": float(gripper_values[-1]),
                    }
                else:
                    replay[f"{arm}_gripper"] = None
            return replay

        collision = self.control_profile["profile"][  # type: ignore[index]
            "dual_arm_collision_check"
        ]
        initial_distance, _ = minimum_inter_arm_distance_value(
            self._pinocchio_chain_points(environment, "left", adapter.read_joint_positions("left")),
            self._pinocchio_chain_points(environment, "right", adapter.read_joint_positions("right")),
        )
        clearance_policy = InterArmClearancePolicy.from_profile(collision, initial_distance)
        current_distance = initial_distance

        def guard_measured_pair(
            measured: Mapping[str, Any],
        ) -> Mapping[str, Any] | None:
            nonlocal current_distance
            current_distance, pair = minimum_inter_arm_distance_value(
                self._pinocchio_chain_points(environment, "left", measured["left"]),
                self._pinocchio_chain_points(environment, "right", measured["right"]),
            )
            decision = clearance_policy.assess(current_distance)
            if decision["allowed"]:
                return None
            return {
                "code": "inter_arm_collision",
                "link_pair": pair,
                **{key: value for key, value in decision.items() if key != "allowed"},
            }

        def stage_index(count: int, stage: int) -> int:
            if count <= 1 or stage_count <= 1:
                return 0
            return min(
                count - 1,
                int(round(stage * (count - 1) / (stage_count - 1))),
            )

        def before_step(stage: int, _wait_index: int) -> None:
            # Gripper-only transactions still advance one source stage at a time.
            for arm in ("left", "right"):
                gripper_path = control_sequence.get(f"{arm}_gripper")
                if not isinstance(gripper_path, Mapping):
                    continue
                result = np.asarray(gripper_path["result"], dtype=np.float64)
                index = stage_index(int(result.shape[0]), stage)
                value = float(result[index])
                environment.robot.set_gripper(
                    value, arm, float(gripper_path["per_step"]),
                )
                issued_gripper[arm].append(value)

        def after_step(_stage: int, _wait_index: int) -> None:
            if not self.operator_execution_video:
                return
            updater = getattr(environment, "_update_render", None)
            if callable(updater):
                updater()
            self._capture_operator_execution_frame(
                action=action,
                phase="motion_transaction_global",
                control_step_index=sum(len(values) for values in issued_gripper.values()),
            )

        try:
            arrival = (
                self.control_profile["profile"]["arrival_check"]  # type: ignore[index]
                if endpoint_public_tcp_targets is not None
                else None
            )
            endpoint_position_tolerance = (
                float(arrival["position_tolerance_m"])
                if endpoint_position_tolerance_m is None
                else float(endpoint_position_tolerance_m)
            ) if endpoint_public_tcp_targets is not None else None
            endpoint_orientation_tolerance = (
                float(arrival["orientation_tolerance_rad"])
                if endpoint_orientation_tolerance_rad is None
                else float(endpoint_orientation_tolerance_rad)
            ) if endpoint_public_tcp_targets is not None else None
            report, arrays = execute_measured_convergence(
                adapter=adapter,
                paths=paths,
                all_arms=("left", "right"),
                profile=profile,
                workspace_bounds_m=self.workspace_bounds_m,
                mode="global",
                endpoint_public_tcp_targets=endpoint_public_tcp_targets,
                endpoint_position_tolerance_m=endpoint_position_tolerance,
                endpoint_orientation_tolerance_rad=(
                    endpoint_orientation_tolerance
                ),
                stage_count=stage_count,
                before_step=before_step,
                after_step=after_step,
                terminal_check=lambda _stage, _wait: self._after_physics_step(
                    environment, action=action, phase="motion_transaction_global",
                ),
                measured_state_guard=guard_measured_pair,
                **({"blocked_policy": self.control_profile["profile"]["blocked_motion_recovery"]}
                   if profile_has(self.control_profile, since_version=CONTACT_RECOVERY_SINCE) else {}),
            )
        except MotionExecutionFault as exc:
            raise RoboTwinControllerTrackingError(
                str(exc), summary=exc.report, arrays=exc.arrays,
                control_sequence=replay_from_arrays(exc.arrays),
            ) from exc
        except MotionTransactionError as exc:
            raise RoboTwinLiveError(f"motion transaction failed closed: {exc}") from exc

        if profile_has(self.control_profile, since_version=CONTACT_RECOVERY_SINCE):
            decision = clearance_policy.assess(current_distance, endpoint=True)
            report["inter_arm_clearance"] = {key: value for key, value in decision.items() if key != "allowed"}
            if report["completed"] and not report.get("termination") and not decision["allowed"]:
                report["completed"] = False
                report["abort"] = {"code": "inter_arm_escape_not_improved", **report["inter_arm_clearance"]}
        return report, arrays, replay_from_arrays(arrays)
