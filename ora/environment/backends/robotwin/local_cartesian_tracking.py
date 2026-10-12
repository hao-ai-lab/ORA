"""Per-call state of one bounded local Cartesian servo action.

``local_cartesian.RoboTwinLocalCartesianMixin._execute_local_cartesian_action``
opens one ``LocalCartesianRun`` per action: it checks the request against the
source-bound delta envelope, samples the initial joints, TCP poses and
inter-arm clearance, then accumulates the issued and measured samples, guard
decisions and solver diagnostics that the tracking report, the evidence arrays
and the replayable control sequence are built from.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ora.environment.controllers.inter_arm_clearance import (
    InterArmClearancePolicy,
    minimum_inter_arm_distance as minimum_inter_arm_distance_value,
)
from ora.environment.controllers.profile_capabilities import (
    CONTACT_RECOVERY_SINCE,
    profile_has,
)

from .common import (
    RoboTwinControllerTrackingError,
    RoboTwinLiveError,
    _pose_error,
    _wxyz_to_xyzw,
)


def _per_arm_samples() -> dict[str, list[list[float]]]:
    return {"left": [], "right": []}


def _per_arm_maximum() -> dict[str, float]:
    return {"left": 0.0, "right": 0.0}


@dataclass
class LocalCartesianRun:
    """Accumulated evidence of one local Cartesian action, both arms sampled."""

    session: Any
    environment: Any
    target_pose_xyzw: Mapping[str, list[float]]
    action: Mapping[str, Any]
    profile: Mapping[str, Any]
    local: Mapping[str, Any]
    gross: Mapping[str, Any]
    arms: tuple[str, ...]
    scene_step: Callable[[], Any]
    robot: Any
    local_motion_adapter: Any
    initial_joints: dict[str, Any]
    initial_pose: dict[str, Any]
    clearance_policy: InterArmClearancePolicy
    measured_joints: dict[str, list[list[float]]] = field(default_factory=_per_arm_samples)
    measured_tcp: dict[str, list[list[float]]] = field(default_factory=_per_arm_samples)
    issued_position: dict[str, list[list[float]]] = field(default_factory=_per_arm_samples)
    issued_velocity: dict[str, list[list[float]]] = field(default_factory=_per_arm_samples)
    maximum_joint_error: dict[str, float] = field(default_factory=_per_arm_maximum)
    maximum_tcp_step: dict[str, float] = field(default_factory=_per_arm_maximum)
    maximum_position_drift: dict[str, float] = field(default_factory=_per_arm_maximum)
    previous_tcp: dict[str, list[float]] = field(default_factory=dict)
    iteration_wait_steps: list[int] = field(default_factory=list)
    solver_iterations: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    stall_count: int = 0
    best_score: float = math.inf
    abort: dict[str, Any] | None = None
    minimum_inter_arm_distance: float = math.inf
    payload_validation: dict[str, Any] | None = None
    execution_started: bool = False
    controller_write_attempts: int = 0
    physics_step_attempts: int = 0

    def arrays(self) -> dict[str, Any]:
        import numpy as np

        return {
            "issued_left_arm__position": np.asarray(
                self.issued_position["left"], dtype=np.float32,
            ),
            "issued_left_arm__velocity": np.asarray(
                self.issued_velocity["left"], dtype=np.float32,
            ),
            "issued_right_arm__position": np.asarray(
                self.issued_position["right"], dtype=np.float32,
            ),
            "issued_right_arm__velocity": np.asarray(
                self.issued_velocity["right"], dtype=np.float32,
            ),
            "measured_left_arm__position": np.asarray(
                self.measured_joints["left"], dtype=np.float32,
            ),
            "measured_right_arm__position": np.asarray(
                self.measured_joints["right"], dtype=np.float32,
            ),
            "measured_left_tcp__pose_xyzw": np.asarray(
                self.measured_tcp["left"], dtype=np.float32,
            ),
            "measured_right_tcp__pose_xyzw": np.asarray(
                self.measured_tcp["right"], dtype=np.float32,
            ),
        }

    def report(self, *, completed: bool) -> dict[str, Any]:
        session = self.session
        environment = self.environment
        arms = self.arms
        current_pose = {
            arm: (
                self.measured_tcp[arm][-1]
                if self.measured_tcp[arm] else self.initial_pose[arm]
            )
            for arm in ("left", "right")
        }
        return {
            "schema": "ora.robotwin-local-cartesian-tracking.v1",
            "backend": (
                "sapien_pinocchio_analytic_tcp_jacobian_weighted_svd_dls"
                if self.local_motion_adapter is not None
                else "pinocchio_numeric_jacobian_dls"
            ),
            "sample_semantics": "after_every_actual_physics_step",
            "target_arms": list(arms),
            "iterations": self.iterations,
            "sampled_steps": len(self.measured_tcp["left"]),
            "controller_write_attempts": self.controller_write_attempts,
            "physics_step_attempts": self.physics_step_attempts,
            "execution_started": self.execution_started,
            "completed": completed,
            "abort": self.abort,
            **({"termination": {"reason": "environment_terminal"}}
               if bool(getattr(environment, "eval_success", False)) else {}),
            "iteration_wait_steps": self.iteration_wait_steps,
            "solver_iterations": self.solver_iterations,
            "minimum_inter_arm_center_distance_m": (
                round(self.minimum_inter_arm_distance, 8)
                if math.isfinite(self.minimum_inter_arm_distance) else None
            ),
            "payload_validation": self.payload_validation,
            **({"inter_arm_clearance": {
                key: value for key, value in self.clearance_policy.assess(
                    minimum_inter_arm_distance_value(
                        session._pinocchio_chain_points(environment, "left", session._measured_arm_joints(environment, "left")),
                        session._pinocchio_chain_points(environment, "right", session._measured_arm_joints(environment, "right")),
                    )[0],
                ).items() if key != "allowed"
            }} if profile_has(session.control_profile, since_version=CONTACT_RECOVERY_SINCE) else {}),
            "arms": {
                arm: {
                    "targeted": arm in arms,
                    "target_error": (
                        {
                            key: round(value, 8)
                            for key, value in _pose_error(
                                self.target_pose_xyzw[arm], current_pose[arm],
                            ).items()
                        }
                        if arm in arms else None
                    ),
                    "maximum_joint_tracking_error_rad": round(
                        self.maximum_joint_error[arm], 8,
                    ),
                    "maximum_tcp_step_m": round(self.maximum_tcp_step[arm], 8),
                    "maximum_position_drift_from_action_start_m": round(
                        self.maximum_position_drift[arm], 8,
                    ),
                }
                for arm in ("left", "right")
            },
        }

    def replay(self) -> dict[str, Any]:
        import numpy as np

        value = {
            f"{arm}_arm": (
                {
                    "status": "IssuedLocalCartesian",
                    "position": np.asarray(
                        self.issued_position[arm], dtype=np.float64,
                    ),
                    "velocity": np.asarray(
                        self.issued_velocity[arm], dtype=np.float64,
                    ),
                }
                if self.issued_position[arm] else None
            )
            for arm in ("left", "right")
        }
        value.update({"left_gripper": None, "right_gripper": None})
        return value

    def result(
        self, *, completed: bool,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """The executor's return value: tracking report, arrays, control sequence."""

        return self.report(completed=completed), self.arrays(), self.replay()

    def merge_transaction(
        self,
        transaction_report: Mapping[str, Any],
        transaction_arrays: Mapping[str, Any],
    ) -> None:
        import numpy as np

        self.execution_started = bool(
            self.execution_started or transaction_report.get("execution_started")
        )
        self.controller_write_attempts += int(
            transaction_report.get("controller_write_attempts", 0)
        )
        self.physics_step_attempts += int(
            transaction_report.get("physics_step_attempts", 0)
        )
        for arm in ("left", "right"):
            self.issued_position[arm].extend(
                np.asarray(
                    transaction_arrays.get(
                        f"issued_{arm}_arm__position", []
                    ),
                ).tolist()
            )
            self.issued_velocity[arm].extend(
                np.asarray(
                    transaction_arrays.get(
                        f"issued_{arm}_arm__velocity", []
                    ),
                ).tolist()
            )
            self.measured_joints[arm].extend(
                np.asarray(
                    transaction_arrays.get(
                        f"measured_{arm}_arm__position", []
                    ),
                ).tolist()
            )
            new_tcp = np.asarray(
                transaction_arrays.get(
                    f"measured_{arm}_tcp__pose_xyzw", []
                ),
                dtype=np.float64,
            )
            if new_tcp.size:
                new_tcp = new_tcp.reshape((-1, 7))
                self.measured_tcp[arm].extend(new_tcp.tolist())
                self.maximum_position_drift[arm] = max(
                    self.maximum_position_drift[arm],
                    max(
                        float(np.linalg.norm(sample[:3] - self.initial_pose[arm][:3]))
                        for sample in new_tcp
                    ),
                )
                self.previous_tcp[arm] = new_tcp[-1].tolist()
            arm_report = transaction_report.get("arms", {}).get(arm, {})
            self.maximum_joint_error[arm] = max(
                self.maximum_joint_error[arm],
                float(arm_report.get(
                    "maximum_joint_tracking_error_rad", 0.0,
                )),
            )
            self.maximum_tcp_step[arm] = max(
                self.maximum_tcp_step[arm],
                float(arm_report.get("maximum_tcp_step_m", 0.0)),
            )
        self.iteration_wait_steps.extend(
            int(value)
            for value in transaction_report.get("waypoint_wait_steps", [])
        )

    def fail_gross(self, code: str, arm: str, value: float) -> None:
        self.abort = {
            "code": code,
            "arm": arm,
            "iteration": self.iterations,
            "value": round(float(value), 8) if math.isfinite(value) else "non_finite",
        }
        raise RoboTwinControllerTrackingError(
            f"local Cartesian tracking guard {code} on {arm}",
            summary=self.report(completed=False), arrays=self.arrays(),
        )

    def check_pair_clearance(self, next_joints: Mapping[str, list[float]], *,
                             planning: bool = False, endpoint: bool = False) -> bool:
        session = self.session
        minimum, pair = minimum_inter_arm_distance_value(
            session._pinocchio_chain_points(self.environment, "left", next_joints["left"]),
            session._pinocchio_chain_points(self.environment, "right", next_joints["right"]),
        )
        self.minimum_inter_arm_distance = min(self.minimum_inter_arm_distance, minimum)
        decision = self.clearance_policy.assess(minimum, planning=planning, endpoint=endpoint)
        if not decision["allowed"]:
            self.abort = {
                "code": "inter_arm_collision",
                "iteration": self.iterations,
                "link_pair": pair,
                **{key: value for key, value in decision.items() if key != "allowed"},
            }
            return False
        return True


def open_local_cartesian_run(
    session: Any,
    *,
    environment: Any,
    target_pose_xyzw: Mapping[str, list[float]],
    action: Mapping[str, Any],
) -> LocalCartesianRun:
    """Admit the target against the delta envelope and sample the start state."""

    profile = session.control_profile["profile"]  # type: ignore[index]
    local = profile["local_cartesian"]
    gross = profile["controller_tracking"]
    collision = profile["dual_arm_collision_check"]
    arms = tuple(arm for arm in ("left", "right") if arm in target_pose_xyzw)
    if not arms:
        raise RoboTwinLiveError("local Cartesian execution has no target arm")
    scene = getattr(environment, "scene", None)
    step = getattr(scene, "step", None)
    robot = getattr(environment, "robot", None)
    if not callable(step) or robot is None:
        raise RoboTwinLiveError(
            "local Cartesian execution requires scene.step and robot"
        )
    local_motion_adapter = (
        session._motion_backend_adapter(session=session, environment=environment)
        if session._uses_motion_transaction_profile() else None
    )

    initial_joints = {
        arm: session._measured_arm_joints(environment, arm)
        for arm in ("left", "right")
    }
    initial_pose = {
        arm: _wxyz_to_xyzw(session._public_pose_wxyz(environment, arm))
        for arm in ("left", "right")
    }
    initial_clearance, _ = minimum_inter_arm_distance_value(
        session._pinocchio_chain_points(environment, "left", initial_joints["left"]),
        session._pinocchio_chain_points(environment, "right", initial_joints["right"]),
    )
    clearance_policy = InterArmClearancePolicy.from_profile(collision, initial_clearance)
    for arm in arms:
        error = _pose_error(target_pose_xyzw[arm], initial_pose[arm])
        # The admission tolerance must match the motion-transaction
        # contract validator bit-for-bit (1e-12): a wider gate here
        # admits edge targets the validator then rejects as fatal
        # (the LIBERO 15-degree boundary incident, e661c30).
        if (
            error["translation_m"]
            > float(local["maximum_translation_m"]) + 1e-12
            or error["orientation_rad"]
            > float(local["maximum_rotation_rad"]) + 1e-12
        ):
            raise RoboTwinLiveError(
                f"local Cartesian target for {arm} exceeds the source-bound "
                "delta envelope"
            )

    return LocalCartesianRun(
        session=session,
        environment=environment,
        target_pose_xyzw=target_pose_xyzw,
        action=action,
        profile=profile,
        local=local,
        gross=gross,
        arms=arms,
        scene_step=step,
        robot=robot,
        local_motion_adapter=local_motion_adapter,
        initial_joints=initial_joints,
        initial_pose=initial_pose,
        clearance_policy=clearance_policy,
        previous_tcp={arm: list(initial_pose[arm]) for arm in ("left", "right")},
    )
