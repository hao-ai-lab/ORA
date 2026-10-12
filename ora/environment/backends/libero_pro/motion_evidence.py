"""Receipts, force feedback, planner and joint-tracking evidence persistence.

A mixin of ``session.LiberoProSession``; it holds methods only, no dataclass
fields. Shared constants and helpers come from ``common``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from ora.environment.controllers.curobo_motion import _curobo_valid_query
from ora.environment.controllers.hybrid_motion import (
    precision_force_stop_enabled,
)
from ora.environment.controllers.force_feedback import public_force_direction
from ora.environment.controllers.command_drift import CommandDriftTracker
from ora.contracts.acceptance import (
    motion_acceptance_check,
    gripper_acceptance_check,
)
from ora.environment.controllers.libero_osc import (
    GRIPPER_TOLERANCE_NORMALIZED,
)

from .common import LiberoProError, _atomic_json, _file_hash


class LiberoMotionEvidenceMixin:
    """Receipts, force feedback, planner and joint-tracking evidence persistence."""

    def _persist_new_robustness_event(self) -> dict[str, Any] | None:
        if self.robustness_episode is None:
            return None
        event = self.robustness_episode.poll_event(self._require_environment())
        if event is None:
            return None
        sequence = int(event["sequence"])
        path = (
            self.output_root / "robustness/events"
            / f"event-{sequence:04d}.json"
        )
        if path.exists():
            raise LiberoProError(f"refusing to overwrite robustness event: {path}")
        _atomic_json(path, dict(event))
        return {
            "path": path.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(path),
            "event": dict(event),
        }

    def _force_telemetry(self, state: Mapping[str, Any]) -> tuple[float | None, bool]:
        """Return raw force magnitude and the selected profile threshold flag."""

        magnitude = state.get("external_force_magnitude_n")
        if magnitude is None:
            wrench = state.get("external_wrench_clipped")
            if isinstance(wrench, (list, tuple)) and len(wrench) >= 3:
                try:
                    candidate = [float(value) for value in wrench[:3]]
                except (TypeError, ValueError):
                    candidate = []
                if candidate and all(math.isfinite(value) for value in candidate):
                    magnitude = math.sqrt(sum(value * value for value in candidate))
        if magnitude is None:
            return None, False
        magnitude = float(magnitude)
        if not math.isfinite(magnitude):
            return None, False
        threshold = float(
            self.motion_profile["sensing"]["force_telemetry_threshold_n"]
        )
        return magnitude, magnitude > threshold

    def _force_direction_feedback(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Bind the current force calibration outside the policy projection."""
        if not precision_force_stop_enabled(self.motion_profile):
            return {}
        feedback = {
            **public_force_direction(state),
            "contact_force_threshold_n": float(self.motion_profile["sensing"]["contact_force_threshold_n"]),
        }
        source = state.get("force_source_binding")
        if isinstance(source, Mapping):
            path = self.output_root / "force-feedback" / f"action-{self.action_sequence:04d}.json"
            if path.exists():
                raise LiberoProError(f"refusing to overwrite force evidence: {path}")
            _atomic_json(path, {"schema": "ora.force-source-evidence.v1",
                                "action_sequence": self.action_sequence,
                                "policy_visibility": "control_only", "source": dict(source)})
            feedback["force_evidence"] = {
                "path": path.relative_to(self.output_root).as_posix(), "sha256": _file_hash(path),
            }
        return feedback

    def _persist_planner_evidence(
        self, planned: Mapping[str, Any], *, probe: bool = False,
    ) -> dict[str, Any]:
        """Keep the full numeric plan canonical but out of policy Context."""

        path = (
            self.output_root / "curobo/plans"
            / f"action-{self.action_sequence:04d}.json"
        )
        if probe:
            sequence = getattr(self, "curobo_probe_sequence", 0)
            path = self.output_root / "curobo/probes" / f"request-{sequence:06d}.json"
        if path.exists():
            raise LiberoProError(f"refusing to overwrite CuRobo plan: {path}")
        _atomic_json(path, planned)
        if probe:
            self.curobo_probe_sequence = sequence + 1
        trajectory = planned.get("trajectory")
        trajectory_summary = None
        if isinstance(trajectory, Mapping):
            positions = trajectory.get("positions")
            trajectory_summary = {
                "joint_names": trajectory.get("joint_names"),
                "waypoint_count": (
                    len(positions) if isinstance(positions, list) else None
                ),
                "dt_s": trajectory.get("dt_s"),
            }
        from .planning_diagnostics import public_summary

        return {
            "schema": "ora.curobo-plan-summary.v1",
            "status": planned.get("status"),
            "reachable": bool(planned.get("reachable")),
            "planner_status": planned.get("planner_status"),
            "valid_query": _curobo_valid_query(planned),
            "world_model": planned.get("world_model"),
            **({"planning_diagnostics": public_summary(planned)}
               if isinstance(planned.get("planning_diagnostics"), Mapping) else {}),
            "trajectory": trajectory_summary,
            "canonical_artifact": {
                "path": path.relative_to(self.output_root).as_posix(),
                "sha256": _file_hash(path),
            },
        }

    def _persist_joint_execution_evidence(
        self, execution: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist per-tick tracking and return sensor-equivalent aggregates."""

        path = (
            self.output_root / "tracking"
            / f"joint-action-{self.action_sequence:04d}.json"
        )
        if path.exists():
            raise LiberoProError(
                f"refusing to overwrite joint tracking evidence: {path}"
            )
        _atomic_json(path, execution)
        public_fields = {
            "controller", "steps_used", "joint_positions",
            "tracking_error_l2_rad", "tracking_error_max_abs_rad",
            "waypoints_total", "waypoints_reached", "terminated", "truncated",
            "stop_reason", "execution_phase",
            "terminal_settle",
        }
        summary = {
            key: value for key, value in execution.items() if key in public_fields
        }
        summary["canonical_artifact"] = {
            "path": path.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(path),
        }
        return summary

    def _persist_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        feedback = receipt.get("feedback")
        if (isinstance(feedback, Mapping) and isinstance(feedback.get("endpoint_error"), Mapping)
                and isinstance(feedback.get("endpoint_tolerance"), Mapping)
                and feedback.get("endpoint_comparison") in {"<", "<="}):
            tolerance = feedback["endpoint_tolerance"]
            receipt["acceptance_checks"] = [motion_acceptance_check(stage="final_endpoint",
                errors={"right": feedback["endpoint_error"]},
                position_tolerance_m=tolerance["position_m"],
                orientation_tolerance_rad=tolerance["orientation_rad"],
                comparison=feedback["endpoint_comparison"])]
        if receipt.get("action") == "set_gripper":
            target = receipt.get("arguments", {}).get("gripper")
            # Only the two canonical jaw commands earn a label. Any other
            # normalized target is a number without a name, and naming it
            # would invent a threshold the executor does not apply.
            if target == 1.0:
                self._gripper_command_target = "open"
            elif target == 0.0:
                self._gripper_command_target = "close"
            measured = receipt.get("reached_gripper")
            if type(target) in (int, float) and type(measured) in (int, float):
                receipt.setdefault("acceptance_checks", []).append(gripper_acceptance_check(
                    errors={"right": abs(measured - target)},
                    tolerance_normalized=GRIPPER_TOLERANCE_NORMALIZED, comparison="<"))
                receipt["gripper_targets_effective"] = {"right": target}
        transform = getattr(self, "_sparse_receipt_transform", None)
        if transform is not None:
            acceptance = receipt.get("acceptance_checks")
            gripper_targets = receipt.get("gripper_targets_effective")
            receipt = transform(receipt)
            if acceptance is not None:
                receipt["acceptance_checks"] = acceptance
            if gripper_targets is not None:
                receipt["gripper_targets_effective"] = gripper_targets
        if precision_force_stop_enabled(self.motion_profile) and receipt.get("action") in {
            "move_pose", "move_free_space", "servo_delta", "recover_motion",
        }:
            if self.command_drift_tracker is None:
                self.command_drift_tracker = CommandDriftTracker()
            segment = receipt.get("segment", receipt)
            before = receipt.get("before", {}).get("right", {}).get("pose_xyz_xyzw")
            after = receipt.get("after", {}).get("right", {}).get("pose_xyz_xyzw")
            commanded = segment.get("commanded_pose_xyz_xyzw")
            drift = self.command_drift_tracker.record(
                action_id=str(receipt["action_id"]),
                arms={"right": {
                    "start_position_m": before[:3] if isinstance(before, list) else None,
                    "final_position_m": after[:3] if isinstance(after, list) else None,
                    "commanded_position_m": commanded[:3] if isinstance(commanded, list) else None,
                }},
                steps_used=int(receipt.get("steps_used", 0)),
            )
            if drift:
                receipt["command_drift"] = drift
        if receipt.get("state_changed"):
            # A6-parity freshness authority: every executed world write is
            # logged by public action name so probed plans can survive
            # gripper-only intervals and expire on anything else.
            self.world_mutation_sequence += 1
            self.world_mutation_actions.append(str(receipt.get("action")))
        receipt_path = self.output_root / "receipts" / f"action-{self.action_sequence:04d}.json"
        if receipt_path.exists():
            raise LiberoProError(f"refusing to overwrite receipt: {receipt_path}")
        _atomic_json(receipt_path, receipt)
        self.last_receipt = {
            "path": receipt_path.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(receipt_path), "receipt": receipt,
        }
        self.action_sequence += 1
        return receipt
