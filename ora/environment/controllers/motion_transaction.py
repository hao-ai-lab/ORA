"""Backend-neutral motion transaction contracts and measured execution.

This module deliberately contains no simulator imports.  Environment backends
provide a small adapter for measured joints, public TCP poses, controller
targets and physics stepping; the shared executor owns convergence, gross
safety checks, status classification and compact numeric evidence.

Planner candidate generation remains adapter-owned.  The live RoboTwin
adapter follows the public algorithmic pattern used by CaP-X (MIT, commit
53e9966), but plans and admits a bounded top-K set rather than trusting a
single nearest IK solution.  No CaP-X runtime, privileged simulator wrapper or
task helper is imported here.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ora.contracts.action import MOTION_INTERACTIONS
from ora.environment.controllers.plan_validation import (
    MotionPlanRejected,
)


MOTION_TRANSACTION_SCHEMA = "ora.motion-transaction.v1"
MOTION_BACKEND_ADAPTER_SCHEMA = "ora.motion-backend-adapter.v1"
MOTION_PLAN_ARTIFACT_SCHEMA = "ora.motion-plan-artifact.v1"
MOTION_EXECUTION_SCHEMA = "ora.motion-execution.v1"
MOTION_INTENT_SCHEMA = "ora.motion-intent.v1"
MOTION_STRATEGY_DECISION_SCHEMA = "ora.motion-strategy-decision.v1"
MOTION_STRATEGIES = frozenset({
    "global", "local", "global_then_local_settle",
    "global_precontact_then_local",
})
MOTION_STATUSES = frozenset({
    "planner_failed", "plan_rejected", "not_reached",
    "execution_fault", "reached",
})


class MotionTransactionError(RuntimeError):
    """The shared motion contract or backend adapter failed closed."""


class MotionExecutionFault(MotionTransactionError):
    """Execution crossed a gross guard after preserving partial evidence."""

    def __init__(
        self, message: str, *, report: Mapping[str, Any], arrays: Mapping[str, Any],
    ) -> None:
        super().__init__(message)
        self.report = dict(report)
        self.arrays = dict(arrays)


@dataclass(frozen=True)
class MotionStrategyDecision:
    """Backend-neutral route selected from physical displacement and intent."""

    strategy: str
    interactions: Mapping[str, str]
    physical_delta: Mapping[str, Mapping[str, float]]
    local_eligible: bool
    endpoint_settle_enabled: bool
    reason: str
    precontact_public_tcp_poses: Mapping[str, Sequence[float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        value = {
            "schema": MOTION_STRATEGY_DECISION_SCHEMA,
            "strategy": self.strategy,
            "interactions": dict(self.interactions),
            "physical_delta": {
                arm: dict(value) for arm, value in self.physical_delta.items()
            },
            "local_eligible": self.local_eligible,
            "endpoint_settle_enabled": self.endpoint_settle_enabled,
            "reason": self.reason,
        }
        if self.precontact_public_tcp_poses is not None:
            value["precontact_public_tcp_poses"] = {
                arm: [float(item) for item in pose]
                for arm, pose in self.precontact_public_tcp_poses.items()
            }
        return value


def _normalize_quaternion(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all() or norm <= 1e-12:
        raise MotionTransactionError("motion strategy quaternion is invalid")
    return quaternion / norm


def _slerp_xyzw(
    start: Sequence[float], target: Sequence[float], fraction: float,
) -> list[float]:
    first = _normalize_quaternion(start)
    second = _normalize_quaternion(target)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        value = first + float(fraction) * (second - first)
        value /= np.linalg.norm(value)
        return value.tolist()
    angle = math.acos(dot)
    sine = math.sin(angle)
    value = (
        math.sin((1.0 - float(fraction)) * angle) / sine * first
        + math.sin(float(fraction) * angle) / sine * second
    )
    value /= np.linalg.norm(value)
    return value.tolist()


def _precontact_pose(
    start: Sequence[float], target: Sequence[float], *,
    local_translation_m: float, local_rotation_rad: float,
) -> list[float]:
    """Leave at most the bounded final translation/rotation for local servo."""

    start_value = np.asarray(start, dtype=np.float64)
    target_value = np.asarray(target, dtype=np.float64)
    if start_value.shape != (7,) or target_value.shape != (7,):
        raise MotionTransactionError("pre-contact pose inputs are invalid")
    translation = target_value[:3] - start_value[:3]
    distance = float(np.linalg.norm(translation))
    position_fraction = (
        max(0.0, distance - float(local_translation_m)) / distance
        if distance > 1e-12 else 0.0
    )
    start_q = _normalize_quaternion(start_value[3:])
    target_q = _normalize_quaternion(target_value[3:])
    dot = min(1.0, abs(float(np.dot(start_q, target_q))))
    rotation = 2.0 * math.acos(dot)
    rotation_fraction = (
        max(0.0, rotation - float(local_rotation_rad)) / rotation
        if rotation > 1e-12 else 0.0
    )
    position = start_value[:3] + position_fraction * translation
    orientation = _slerp_xyzw(start_q, target_q, rotation_fraction)
    return [*[float(item) for item in position], *orientation]


def select_motion_strategy(
    *,
    action: str,
    start_public_tcp_poses: Mapping[str, Sequence[float]],
    target_public_tcp_poses: Mapping[str, Sequence[float]],
    interactions: Mapping[str, str],
    capabilities: MotionBackendCapabilities,
    profile: Mapping[str, Any],
    plan_reference: bool = False,
) -> MotionStrategyDecision:
    """Select global/local execution without consulting request syntax.

    ``start_public_tcp_poses`` and ``target_public_tcp_poses`` are canonical
    public TCP poses.  The caller may have compiled either absolute or delta
    syntax; that provenance is intentionally unavailable to the router.
    """

    schema = profile.get("schema")
    if schema == "ora.motion-strategy-router.v1":
        expected_profile = {
            "schema", "maximum_local_translation_m", "maximum_local_rotation_rad",
            "contact_approach_requires_local", "free_space_strategy",
            "precision_large_delta_strategy", "endpoint_settle",
        }
        if set(profile) != expected_profile:
            raise MotionTransactionError("motion strategy router profile drifted")
        return _select_motion_strategy_v1(
            action=action,
            start_public_tcp_poses=start_public_tcp_poses,
            target_public_tcp_poses=target_public_tcp_poses,
            interactions=interactions,
            capabilities=capabilities,
            profile=profile,
            plan_reference=plan_reference,
        )
    expected_profile = {
        "schema", "direct_local_translation_m", "direct_local_rotation_rad",
        "comparison_epsilon_m", "comparison_epsilon_rad",
        "precontact_arrival_reserve_m", "precontact_arrival_reserve_rad",
        "free_space_strategy", "precision_strategy",
        "contact_approach_strategy", "pure_rotation_local",
        "endpoint_settle",
    }
    if set(profile) != expected_profile or schema != (
        "ora.motion-strategy-router.v2"
    ):
        raise MotionTransactionError("motion strategy router profile drifted")
    if profile.get("free_space_strategy") != "global":
        raise MotionTransactionError("free-space routing contract drifted")
    if profile.get("precision_strategy") != "global_then_local_settle":
        raise MotionTransactionError("precision routing contract drifted")
    if profile.get("contact_approach_strategy") != (
        "bounded_local_or_global_precontact_then_local"
    ):
        raise MotionTransactionError("contact-approach routing contract drifted")
    rotation_policy = profile.get("pure_rotation_local")
    if not isinstance(rotation_policy, Mapping) or set(rotation_policy) != {
        "enabled", "maximum_translation_m", "maximum_rotation_rad",
    } or rotation_policy.get("enabled") is not True:
        raise MotionTransactionError("pure-rotation routing profile drifted")
    endpoint = profile.get("endpoint_settle")
    if not isinstance(endpoint, Mapping) or set(endpoint) != {
        "enabled", "maximum_translation_m", "maximum_rotation_rad",
        "only_after_converged_global_execution", "never_on_contact_approach",
    }:
        raise MotionTransactionError("endpoint-settle routing profile drifted")
    if (
        endpoint.get("enabled") is not True
        or endpoint.get("only_after_converged_global_execution") is not True
        or endpoint.get("never_on_contact_approach") is not True
    ):
        raise MotionTransactionError("endpoint-settle routing semantics drifted")
    maximum_translation = float(profile["direct_local_translation_m"])
    maximum_rotation = float(profile["direct_local_rotation_rad"])
    comparison_epsilon_m = float(profile["comparison_epsilon_m"])
    comparison_epsilon_rad = float(profile["comparison_epsilon_rad"])
    precontact_arrival_reserve_m = float(
        profile["precontact_arrival_reserve_m"]
    )
    precontact_arrival_reserve_rad = float(
        profile["precontact_arrival_reserve_rad"]
    )
    if not (0.0 < maximum_translation <= 0.1 and 0.0 < maximum_rotation <= math.pi):
        raise MotionTransactionError("motion strategy local bounds are invalid")
    if not (
        0.0 <= comparison_epsilon_m <= 1e-4
        and 0.0 <= comparison_epsilon_rad <= 1e-4
        and 0.0 < precontact_arrival_reserve_m < maximum_translation
        and 0.0 < precontact_arrival_reserve_rad < maximum_rotation
    ):
        raise MotionTransactionError("motion strategy numerical reserves are invalid")

    arms = tuple(target_public_tcp_poses)
    require_motion_capabilities(
        capabilities,
        requested_arms=arms,
        require_global=action == "recover_motion" or plan_reference,
        require_synchronized=len(arms) > 1,
    )
    if set(start_public_tcp_poses) != set(arms) or set(interactions) != set(arms):
        raise MotionTransactionError("motion intent arm maps differ")
    if any(value not in MOTION_INTERACTIONS for value in interactions.values()):
        raise MotionTransactionError("motion intent interaction is invalid")

    deltas = {
        arm: _pose_error(target_public_tcp_poses[arm], start_public_tcp_poses[arm])
        for arm in arms
    }
    local_eligible = all(
        item["translation_m"] <= maximum_translation + comparison_epsilon_m
        and item["orientation_rad"] <= maximum_rotation + comparison_epsilon_rad
        for item in deltas.values()
    )
    unique_interactions = set(interactions.values())
    if len(unique_interactions) > 1:
        raise MotionTransactionError(
            "one synchronized motion cannot mix interaction semantics"
        )
    precontact: dict[str, list[float]] | None = None
    if plan_reference:
        strategy, reason = "global", "immutable_plan_reference"
    elif action == "recover_motion":
        strategy, reason = "global", "joint_configuration_recovery"
    elif unique_interactions == {"free_space"}:
        strategy, reason = "global", "free_space_requires_collision_aware_global_plan"
    elif unique_interactions == {"contact_approach"} and local_eligible:
        require_motion_capabilities(
            capabilities, requested_arms=arms, require_local=True,
            require_synchronized=len(arms) > 1,
        )
        strategy, reason = "local", "bounded_contact_approach"
    elif unique_interactions == {"contact_approach"}:
        require_motion_capabilities(
            capabilities, requested_arms=arms, require_global=True,
            require_local=True, require_synchronized=len(arms) > 1,
        )
        strategy, reason = (
            "global_precontact_then_local", "staged_contact_approach"
        )
        precontact = {
            arm: _precontact_pose(
                start_public_tcp_poses[arm], target_public_tcp_poses[arm],
                local_translation_m=(
                    maximum_translation - precontact_arrival_reserve_m
                ),
                local_rotation_rad=(
                    maximum_rotation - precontact_arrival_reserve_rad
                ),
            )
            for arm in arms
        }
    elif unique_interactions == {"precision"} and all(
        item["translation_m"]
        <= (
            float(rotation_policy["maximum_translation_m"])
            + comparison_epsilon_m
        )
        and 0.0 < item["orientation_rad"]
        <= (
            float(rotation_policy["maximum_rotation_rad"])
            + 1e-12
        )
        for item in deltas.values()
    ):
        require_motion_capabilities(
            capabilities, requested_arms=arms, require_local=True,
            require_synchronized=len(arms) > 1,
        )
        strategy, reason = "local", "bounded_in_place_rotation"
    else:
        require_motion_capabilities(
            capabilities, requested_arms=arms, require_global=True,
            require_synchronized=len(arms) > 1,
        )
        strategy = "global_then_local_settle"
        reason = "precision_requires_global_plan"
    if strategy not in MOTION_STRATEGIES:
        raise MotionTransactionError("motion strategy is invalid")
    settle_enabled = strategy == "global_then_local_settle" and capabilities.local_cartesian
    return MotionStrategyDecision(
        strategy=strategy,
        interactions=dict(interactions),
        physical_delta=deltas,
        local_eligible=local_eligible,
        endpoint_settle_enabled=settle_enabled,
        reason=reason,
        precontact_public_tcp_poses=precontact,
    )


def _select_motion_strategy_v1(
    *, action: str,
    start_public_tcp_poses: Mapping[str, Sequence[float]],
    target_public_tcp_poses: Mapping[str, Sequence[float]],
    interactions: Mapping[str, str], capabilities: MotionBackendCapabilities,
    profile: Mapping[str, Any], plan_reference: bool,
) -> MotionStrategyDecision:
    """Frozen v1 routing retained for historical v10 RunPlans and traces."""

    if profile.get("free_space_strategy") != "global_then_local_settle":
        raise MotionTransactionError("free-space routing contract drifted")
    if profile.get("precision_large_delta_strategy") != "global_then_local_settle":
        raise MotionTransactionError("precision routing contract drifted")
    if profile.get("contact_approach_requires_local") is not True:
        raise MotionTransactionError("contact-approach routing contract drifted")
    endpoint = profile.get("endpoint_settle")
    if not isinstance(endpoint, Mapping) or set(endpoint) != {
        "enabled", "maximum_translation_m", "maximum_rotation_rad",
        "only_after_converged_global_execution", "never_on_contact_approach",
    }:
        raise MotionTransactionError("endpoint-settle routing profile drifted")
    maximum_translation = float(profile["maximum_local_translation_m"])
    maximum_rotation = float(profile["maximum_local_rotation_rad"])
    arms = tuple(target_public_tcp_poses)
    require_motion_capabilities(
        capabilities, requested_arms=arms,
        require_global=action == "recover_motion" or plan_reference,
        require_synchronized=len(arms) > 1,
    )
    if set(start_public_tcp_poses) != set(arms) or set(interactions) != set(arms):
        raise MotionTransactionError("motion intent arm maps differ")
    if any(value not in MOTION_INTERACTIONS for value in interactions.values()):
        raise MotionTransactionError("motion intent interaction is invalid")
    deltas = {
        arm: _pose_error(target_public_tcp_poses[arm], start_public_tcp_poses[arm])
        for arm in arms
    }
    local_eligible = all(
        item["translation_m"] <= maximum_translation + 1e-12
        and item["orientation_rad"] <= maximum_rotation + 1e-12
        for item in deltas.values()
    )
    if (
        "contact_approach" in interactions.values()
        and "free_space" in interactions.values()
    ):
        raise MotionTransactionError(
            "one synchronized motion cannot mix free_space and contact_approach"
        )
    if plan_reference:
        strategy, reason = "global", "immutable_plan_reference"
    elif action == "recover_motion":
        strategy, reason = "global", "joint_configuration_recovery"
    elif "contact_approach" in interactions.values() and not local_eligible:
        raise MotionTransactionError(
            "contact_approach exceeds the qualified local Cartesian envelope"
        )
    elif all(
        value in {"precision", "contact_approach"}
        for value in interactions.values()
    ) and local_eligible:
        require_motion_capabilities(
            capabilities, requested_arms=arms, require_local=True,
            require_synchronized=len(arms) > 1,
        )
        strategy, reason = "local", "bounded_precision_or_contact_motion"
    else:
        require_motion_capabilities(
            capabilities, requested_arms=arms, require_global=True,
            require_synchronized=len(arms) > 1,
        )
        strategy, reason = "global_then_local_settle", (
            "free_space_requires_collision_aware_global_plan"
            if "free_space" in interactions.values()
            else "precision_motion_exceeds_local_envelope"
        )
    return MotionStrategyDecision(
        strategy=strategy, interactions=dict(interactions), physical_delta=deltas,
        local_eligible=local_eligible,
        endpoint_settle_enabled=(
            strategy == "global_then_local_settle" and capabilities.local_cartesian
        ),
        reason=reason,
    )


@dataclass(frozen=True)
class MotionBackendCapabilities:
    """Source-bound abilities exposed by one simulator-specific adapter."""

    backend_id: str
    embodiment_id: str
    arms: tuple[str, ...]
    controlled_point: str
    planner_frames: Mapping[str, str]
    synchronized_groups: tuple[tuple[str, ...], ...]
    global_joint_planning: bool
    local_cartesian: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": MOTION_BACKEND_ADAPTER_SCHEMA,
            "backend_id": self.backend_id,
            "embodiment_id": self.embodiment_id,
            "arms": list(self.arms),
            "controlled_point": self.controlled_point,
            "planner_frames": dict(self.planner_frames),
            "synchronized_groups": [list(group) for group in self.synchronized_groups],
            "global_joint_planning": self.global_joint_planning,
            "local_cartesian": self.local_cartesian,
        }


def validate_motion_backend_capabilities(value: Any) -> MotionBackendCapabilities:
    """Validate the closed backend capability descriptor."""

    if not isinstance(value, Mapping):
        raise MotionTransactionError("motion backend capabilities must be an object")
    expected = {
        "schema", "backend_id", "embodiment_id", "arms", "controlled_point",
        "planner_frames", "synchronized_groups", "global_joint_planning",
        "local_cartesian",
    }
    if set(value) != expected or value.get("schema") != MOTION_BACKEND_ADAPTER_SCHEMA:
        raise MotionTransactionError("motion backend capability fields drifted")
    for field in ("backend_id", "embodiment_id", "controlled_point"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise MotionTransactionError(f"motion backend {field} is invalid")
    raw_arms = value.get("arms")
    if (
        not isinstance(raw_arms, list) or not raw_arms
        or len(set(raw_arms)) != len(raw_arms)
        or any(not isinstance(arm, str) or not arm for arm in raw_arms)
    ):
        raise MotionTransactionError("motion backend arms are invalid")
    frames = value.get("planner_frames")
    if (
        not isinstance(frames, Mapping) or set(frames) != set(raw_arms)
        or any(not isinstance(item, str) or not item for item in frames.values())
    ):
        raise MotionTransactionError("motion backend planner frames are invalid")
    raw_groups = value.get("synchronized_groups")
    if not isinstance(raw_groups, list):
        raise MotionTransactionError("motion backend synchronized groups are invalid")
    groups: list[tuple[str, ...]] = []
    for group in raw_groups:
        if (
            not isinstance(group, list) or len(group) < 2
            or len(set(group)) != len(group) or set(group) - set(raw_arms)
        ):
            raise MotionTransactionError("motion backend synchronized group is invalid")
        groups.append(tuple(group))
    for field in ("global_joint_planning", "local_cartesian"):
        if not isinstance(value.get(field), bool):
            raise MotionTransactionError(f"motion backend {field} must be Boolean")
    return MotionBackendCapabilities(
        backend_id=str(value["backend_id"]),
        embodiment_id=str(value["embodiment_id"]),
        arms=tuple(raw_arms),
        controlled_point=str(value["controlled_point"]),
        planner_frames={str(key): str(item) for key, item in frames.items()},
        synchronized_groups=tuple(groups),
        global_joint_planning=bool(value["global_joint_planning"]),
        local_cartesian=bool(value["local_cartesian"]),
    )


@runtime_checkable
class MotionBackendAdapter(Protocol):
    """Planner, kinematics and plant boundary for one motion backend."""

    @property
    def capabilities(self) -> MotionBackendCapabilities: ...

    def capture_action_start(self) -> Mapping[str, Any]: ...

    def compile_public_target(
        self, arm: str, public_tcp_pose_xyzw: Sequence[float],
    ) -> Any: ...

    def plan_global(
        self, arm: str, planner_target: Any, action_start: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def plan_batch_global(
        self, arm: str, planner_targets: Sequence[Any],
        action_start: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def fk_control_pose(self, arm: str, joints: Sequence[float]) -> Sequence[float]: ...

    def fk_public_tcp_pose(self, arm: str, joints: Sequence[float]) -> Sequence[float]: ...

    def joint_limits(self, arm: str) -> Sequence[Sequence[float]]: ...

    def workspace_bounds_m(self) -> Mapping[str, Sequence[float]]: ...

    def validate_collision(
        self, plans: Mapping[str, Any], action_start: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def read_joint_positions(self, arm: str) -> Sequence[float]: ...

    def read_public_tcp_pose(self, arm: str) -> Sequence[float]: ...

    def write_joint_target(
        self, arm: str, position: Sequence[float], velocity: Sequence[float],
    ) -> None: ...

    def step_physics(self) -> None: ...


@runtime_checkable
class JointLocalMotionBackendAdapter(MotionBackendAdapter, Protocol):
    """Optional local boundary for backends that compile Cartesian error to joints."""

    def compute_local_joint_target(
        self, arm: str, current_joints: Sequence[float],
        current_public_tcp_pose_xyzw: Sequence[float],
        target_public_tcp_pose_xyzw: Sequence[float], profile: Mapping[str, Any],
    ) -> Mapping[str, Sequence[float]]: ...


def validate_motion_backend_adapter(adapter: Any) -> MotionBackendCapabilities:
    """Fail at startup when a selected adapter omits a transaction boundary."""

    capabilities = getattr(adapter, "capabilities", None)
    if not isinstance(capabilities, MotionBackendCapabilities):
        raise MotionTransactionError("motion backend adapter lacks capabilities")
    required_methods = (
        "capture_action_start", "compile_public_target", "plan_global",
        "plan_batch_global",
        "fk_control_pose", "fk_public_tcp_pose", "joint_limits",
        "workspace_bounds_m", "validate_collision", "read_joint_positions",
        "read_public_tcp_pose", "write_joint_target", "step_physics",
    )
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    if missing:
        raise MotionTransactionError(
            f"motion backend adapter methods are unavailable: {missing}"
        )
    local_joint = callable(getattr(adapter, "compute_local_joint_target", None))
    if capabilities.local_cartesian and not local_joint:
        raise MotionTransactionError(
            "motion backend must expose exactly one local Cartesian command boundary"
        )
    if not capabilities.local_cartesian and local_joint:
        raise MotionTransactionError(
            "motion backend exposes an undeclared local Cartesian command boundary"
        )
    return validate_motion_backend_capabilities(capabilities.as_dict())


def require_motion_capabilities(
    capabilities: MotionBackendCapabilities,
    *, requested_arms: Sequence[str], require_global: bool = False,
    require_local: bool = False, require_synchronized: bool = False,
) -> None:
    """Fail before execution when the selected backend cannot honor a request."""

    arms = tuple(requested_arms)
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(capabilities.arms):
        raise MotionTransactionError("requested arms do not match backend topology")
    if require_global and not capabilities.global_joint_planning:
        raise MotionTransactionError("backend does not support global joint planning")
    if require_local and not capabilities.local_cartesian:
        raise MotionTransactionError("backend does not support local Cartesian motion")
    if require_synchronized and len(arms) > 1:
        requested = frozenset(arms)
        if not any(requested <= frozenset(group) for group in capabilities.synchronized_groups):
            raise MotionTransactionError("backend cannot synchronize the requested arms")


def select_nearest_joint_solution(
    solutions: Any, reference: Sequence[float], *, continuous: Sequence[bool] | None = None,
) -> tuple[list[float], int, list[float]]:
    """Select the finite IK solution nearest to measured joints.

    Revolute distances are wrapped into ``[-pi, pi]`` by default.  Callers may
    mark non-continuous coordinates false when direct subtraction is required.
    """

    candidates = np.asarray(solutions, dtype=np.float64)
    current = np.asarray(reference, dtype=np.float64)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    if (
        candidates.ndim != 2 or current.ndim != 1
        or candidates.shape[1] != current.shape[0]
        or not len(candidates) or not np.isfinite(current).all()
    ):
        raise MotionTransactionError("IK solution set or reference joints are invalid")
    finite = np.isfinite(candidates).all(axis=1)
    if not np.any(finite):
        raise MotionTransactionError("IK solution set contains no finite candidate")
    mask = np.ones(current.shape[0], dtype=bool)
    if continuous is not None:
        if len(continuous) != current.shape[0]:
            raise MotionTransactionError("IK continuous-joint mask is invalid")
        mask = np.asarray(continuous, dtype=bool)
    delta = candidates - current[None, :]
    delta[:, mask] = (delta[:, mask] + math.pi) % (2.0 * math.pi) - math.pi
    distances = np.linalg.norm(delta, axis=1)
    distances[~finite] = math.inf
    index = int(np.argmin(distances))
    return candidates[index].tolist(), index, distances.tolist()


def _finite_vector(value: Any, *, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise MotionTransactionError(f"{label} must be one finite vector")
    return result


def _validated_pose(value: Any, *, label: str) -> np.ndarray:
    result = _finite_vector(value, label=label)
    if result.shape != (7,):
        raise MotionTransactionError(f"{label} must be one 7D pose")
    if float(np.linalg.norm(result[3:])) <= 1e-12:
        raise MotionTransactionError(f"{label} quaternion is degenerate")
    return result


def _pose_error(expected: Sequence[float], actual: Sequence[float]) -> dict[str, float]:
    target = _validated_pose(expected, label="target public TCP pose")
    measured = _validated_pose(actual, label="measured public TCP pose")
    translation = float(np.linalg.norm(target[:3] - measured[:3]))
    target_norm = float(np.linalg.norm(target[3:]))
    measured_norm = float(np.linalg.norm(measured[3:]))
    left = target[3:] / target_norm
    right = measured[3:] / measured_norm
    dot = min(1.0, max(0.0, abs(float(np.dot(left, right)))))
    return {"translation_m": translation, "orientation_rad": 2.0 * math.acos(dot)}


def select_post_global_phase(
    *, staged_contact: bool, tracking_completed: bool,
    stage_endpoint_errors: Mapping[str, Mapping[str, float]],
    final_endpoint_errors: Mapping[str, Mapping[str, float]],
    arrival_position_tolerance_m: float,
    arrival_orientation_tolerance_rad: float,
    local_translation_envelope_m: float,
    local_rotation_envelope_rad: float,
    stage_arrival_position_tolerance_m: float | None = None,
    stage_arrival_orientation_tolerance_rad: float | None = None,
    final_arrival_position_tolerance_m: float | None = None,
    final_arrival_orientation_tolerance_rad: float | None = None,
    acceptance_evidence: list[dict[str, Any]] | None = None,
) -> str:
    """Choose whether a completed global stage needs bounded local control.

    Precision plans target the final pose globally and may locally repair the
    residual band between the arrival threshold and the settle envelope.
    Staged contact plans instead must first reach their explicit pre-contact
    target. This helper keeps those two gates distinct so local settle cannot
    become logically unreachable.
    """

    if not isinstance(staged_contact, bool) or not isinstance(
        tracking_completed, bool
    ):
        raise MotionTransactionError("post-global phase flags are invalid")
    if (
        not isinstance(stage_endpoint_errors, Mapping)
        or not isinstance(final_endpoint_errors, Mapping)
        or not stage_endpoint_errors
        or set(stage_endpoint_errors) != set(final_endpoint_errors)
    ):
        raise MotionTransactionError("post-global endpoint error maps differ")
    stage_position_tolerance = (
        arrival_position_tolerance_m
        if stage_arrival_position_tolerance_m is None
        else stage_arrival_position_tolerance_m
    )
    stage_orientation_tolerance = (
        arrival_orientation_tolerance_rad
        if stage_arrival_orientation_tolerance_rad is None
        else stage_arrival_orientation_tolerance_rad
    )
    final_position_tolerance = (
        arrival_position_tolerance_m
        if final_arrival_position_tolerance_m is None
        else final_arrival_position_tolerance_m
    )
    final_orientation_tolerance = (
        arrival_orientation_tolerance_rad
        if final_arrival_orientation_tolerance_rad is None
        else final_arrival_orientation_tolerance_rad
    )
    limits = (
        arrival_position_tolerance_m, arrival_orientation_tolerance_rad,
        local_translation_envelope_m, local_rotation_envelope_rad,
        stage_position_tolerance, stage_orientation_tolerance,
        final_position_tolerance, final_orientation_tolerance,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) <= 0.0
        for value in limits
    ):
        raise MotionTransactionError("post-global phase limits are invalid")

    def within(
        errors: Mapping[str, Mapping[str, float]],
        position_limit: float, orientation_limit: float,
        stage: str,
    ) -> bool:
        if acceptance_evidence is not None:
            from ora.contracts.acceptance import motion_acceptance_check
            acceptance_evidence.append(motion_acceptance_check(stage=stage, errors=errors,
                position_tolerance_m=position_limit, orientation_tolerance_rad=orientation_limit))
        for arm, error in errors.items():
            if not isinstance(arm, str) or not isinstance(error, Mapping):
                raise MotionTransactionError("post-global endpoint error is invalid")
            if set(error) != {"translation_m", "orientation_rad"}:
                raise MotionTransactionError("post-global endpoint fields drifted")
            values = (error["translation_m"], error["orientation_rad"])
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value)) or float(value) < 0.0
                for value in values
            ):
                raise MotionTransactionError("post-global endpoint value is invalid")
            if (
                float(values[0]) > float(position_limit)
                or float(values[1]) > float(orientation_limit)
            ):
                return False
        return True

    if not tracking_completed:
        return "global_tracking_not_converged"
    if staged_contact and not within(
        stage_endpoint_errors,
        stage_position_tolerance,
        stage_orientation_tolerance,
        "precontact_global",
    ):
        return "precontact_global_not_reached"
    if within(
        final_endpoint_errors,
        final_position_tolerance,
        final_orientation_tolerance,
        "post_global_final",
    ):
        return "not_needed"
    if not within(
        final_endpoint_errors,
        local_translation_envelope_m,
        local_rotation_envelope_rad,
        "local_entry_envelope",
    ):
        return (
            "precontact_local_envelope_exceeded"
            if staged_contact else "outside_settle_envelope"
        )
    return "run_local"


def classify_motion_status(
    *, planner_failed: bool = False, plan_rejected: bool = False,
    execution_started: bool = False, execution_fault: str | None = None,
    convergence_completed: bool = False,
    target_poses: Mapping[str, Sequence[float]] | None = None,
    measured_poses: Mapping[str, Sequence[float]] | None = None,
    position_tolerance_m: float, orientation_tolerance_rad: float,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Classify one motion without treating Planner success as arrival."""

    if not (
        math.isfinite(float(position_tolerance_m))
        and 0.0 < float(position_tolerance_m) <= 0.2
        and math.isfinite(float(orientation_tolerance_rad))
        and 0.0 < float(orientation_tolerance_rad) <= math.pi
    ):
        raise MotionTransactionError("endpoint tolerances are invalid")
    errors: dict[str, dict[str, float]] = {}
    if target_poses is not None:
        if measured_poses is None or set(target_poses) - set(measured_poses):
            raise MotionTransactionError("endpoint classification lacks measured poses")
        errors = {
            arm: _pose_error(target, measured_poses[arm])
            for arm, target in target_poses.items()
        }
    if execution_fault is not None:
        return "execution_fault", errors
    if plan_rejected:
        return "plan_rejected", errors
    if planner_failed and not execution_started:
        return "planner_failed", errors
    reached = bool(errors) and all(
        item["translation_m"] <= float(position_tolerance_m)
        and item["orientation_rad"] <= float(orientation_tolerance_rad)
        for item in errors.values()
    )
    return ("reached" if convergence_completed and reached else "not_reached"), errors


def _stage_index(count: int, stage_count: int, stage: int) -> int:
    if count <= 1 or stage_count <= 1:
        return 0
    return min(count - 1, int(round(stage * (count - 1) / (stage_count - 1))))


def _strict_real(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MotionTransactionError(f"{label} is invalid; expected one finite number")
    return float(value)


def _strict_positive_int(value: Any, *, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MotionTransactionError(f"{label} is invalid; expected one integer")
    if not 1 <= value <= maximum:
        raise MotionTransactionError(f"{label} is outside its supported range")
    return value


def _validated_workspace_bounds(
    value: Any, *, margin_m: float,
) -> dict[str, tuple[float, float]]:
    if not isinstance(value, Mapping) or set(value) != {"x", "y", "z"}:
        raise MotionTransactionError("workspace bounds must contain exactly x, y and z")
    result: dict[str, tuple[float, float]] = {}
    for axis in ("x", "y", "z"):
        interval = value[axis]
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise MotionTransactionError(f"workspace bound {axis} is invalid")
        lower = _strict_real(interval[0], label=f"workspace {axis} lower bound")
        upper = _strict_real(interval[1], label=f"workspace {axis} upper bound")
        if lower >= upper or lower - margin_m >= upper + margin_m:
            raise MotionTransactionError(f"workspace bound {axis} is not increasing")
        result[axis] = (lower, upper)
    return result


def execute_measured_convergence(
    *, adapter: MotionBackendAdapter,
    paths: Mapping[str, Mapping[str, Any]] | None = None,
    all_arms: Sequence[str],
    profile: Mapping[str, Any],
    workspace_bounds_m: Mapping[str, Sequence[float]],
    mode: str,
    endpoint_public_tcp_targets: Mapping[str, Sequence[float]] | None = None,
    endpoint_position_tolerance_m: float | None = None,
    endpoint_orientation_tolerance_rad: float | None = None,
    stage_count: int | None = None,
    before_step: Callable[[int, int], None] | None = None,
    after_step: Callable[[int, int], None] | None = None,
    terminal_check: Callable[[int, int], bool] | None = None,
    measured_state_guard: Callable[
        [Mapping[str, Sequence[float]]], Mapping[str, Any] | None
    ] | None = None,
    blocked_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute joint paths by measured convergence, including a strict endpoint.

    Intermediate joint waypoints use the profile's joint tolerance.  When a
    joint-path caller supplies the three endpoint arguments, the final
    waypoint keeps its final joint target active until the measured public TCP
    also satisfies the requested Cartesian tolerance.  This avoids turning a
    deliberately loose interpolation tolerance into a false reached result.
    ``terminal_check`` is polled immediately after each physics step. A true
    environment terminal ends the transaction normally, preserving measured
    evidence without claiming arrival or dispatching any remaining waypoint.

    The steps live beside this module: ``convergence_request.py`` admits the
    request, and ``measured_convergence.MeasuredConvergence`` owns the stage
    loop, its guards and the report.
    """

    # The steps import this module's contract primitives, so they load here.
    from .convergence_request import (
        admit_convergence_request, joint_blocked_monitor, read_initial_tcp,
    )
    from .measured_convergence import MeasuredConvergence

    request = admit_convergence_request(
        adapter=adapter, paths=paths,
        all_arms=all_arms, profile=profile, workspace_bounds_m=workspace_bounds_m,
        mode=mode, endpoint_public_tcp_targets=endpoint_public_tcp_targets,
        endpoint_position_tolerance_m=endpoint_position_tolerance_m,
        endpoint_orientation_tolerance_rad=endpoint_orientation_tolerance_rad,
        stage_count=stage_count,
    )
    previous_tcp = read_initial_tcp(adapter, request)
    blocked_monitor = joint_blocked_monitor(blocked_policy, request)
    return MeasuredConvergence(
        adapter, request, previous_tcp=previous_tcp,
        blocked_monitor=blocked_monitor, before_step=before_step,
        after_step=after_step, terminal_check=terminal_check,
        measured_state_guard=measured_state_guard,
    ).run()


def write_motion_plan_artifact(
    root: Path, *, transaction_id: str, backend_id: str, embodiment_id: str,
    binding: Mapping[str, Any], validation: Mapping[str, Any],
    arrays: Mapping[str, Any],
) -> dict[str, Any]:
    """Write one compact backend-neutral plan/execution artifact."""

    if not transaction_id or not backend_id or not embodiment_id:
        raise MotionTransactionError("motion artifact identity is incomplete")
    target = root / "motion-transactions" / transaction_id
    target.mkdir(parents=True, exist_ok=False)
    normalized: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.kind not in {"b", "i", "u", "f"} or not np.isfinite(array).all():
            raise MotionTransactionError(f"motion artifact array {key} is invalid")
        normalized[str(key)] = array
    arrays_path = target / "arrays.npz"
    temporary = target / "arrays.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **normalized)
    temporary.replace(arrays_path)
    body = {
        "schema": MOTION_PLAN_ARTIFACT_SCHEMA,
        "transaction_id": transaction_id,
        "backend_id": backend_id,
        "embodiment_id": embodiment_id,
        "binding": json.loads(json.dumps(binding, allow_nan=False, sort_keys=True)),
        "validation": json.loads(json.dumps(validation, allow_nan=False, sort_keys=True)),
        "arrays": {
            "path": "arrays.npz",
            "sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            "keys": sorted(normalized),
        },
    }
    artifact_sha256 = hashlib.sha256(json.dumps(
        body, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    manifest = {**body, "artifact_sha256": artifact_sha256}
    manifest_path = target / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema": MOTION_PLAN_ARTIFACT_SCHEMA,
        "path": manifest_path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "artifact_sha256": artifact_sha256,
    }


__all__ = [
    "MOTION_BACKEND_ADAPTER_SCHEMA", "MOTION_EXECUTION_SCHEMA",
    "MOTION_PLAN_ARTIFACT_SCHEMA", "MOTION_STATUSES",
    "MOTION_TRANSACTION_SCHEMA", "MotionBackendAdapter",
    "JointLocalMotionBackendAdapter",
    "MotionBackendCapabilities", "MotionExecutionFault", "MotionTransactionError",
    "classify_motion_status", "execute_measured_convergence",
    "require_motion_capabilities", "select_nearest_joint_solution",
    "select_post_global_phase",
    "validate_motion_backend_adapter", "validate_motion_backend_capabilities",
    "write_motion_plan_artifact",
]
