"""Receipt-schema-keyed public action-effect projection for Native sessions.

The shared Native session layer must not grow per-environment branches: a
backend that introduces a new receipt contract registers a projector here
instead of editing ``session.py`` (EXP08A cross-embodiment parity boundary).
Each projector receives the verified receipt record plus a small context and
returns the policy-visible minimal action effect. Receipt schemas without a
registered projector fall back to the shared action-name-based projection
owned by the session.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ora.contracts.acceptance import (
    motion_acceptance_check, validate_motion_acceptance_checks,
)
from ..evidence.controller_feedback import public_gripper_detail
from ora.contracts.wait import public_wait_detail
from ..context import causal_provider_value
from ora.contracts.planning_diagnostics import public_planning_diagnostics

from ..evidence.motion_feedback import (
    _public_command_drift, public_contact_motion_cause, public_inter_arm_clearance,
    stall_evidenced_code,
)
from ..evidence.plan_feedback import public_plan_validation
from .contact_detail import (
    CAPABILITY as ARM_TRANSLATION_CAPABILITY, arm_translations, cause_evidence,
    contact_detail as _libero_contact_detail, with_arm_translations,
    without_skipped_retract_diagnostics,
)

def _planning_detail(record: Mapping[str, Any]) -> dict[str, Any]:
    # LIBERO sparse receipts wrap the original hybrid receipt in ``segment``.
    # Read that specific evidence owner, never recursively search raw traces.
    source = record.get("segment")
    source = source if isinstance(source, Mapping) else record
    planner = source.get("planner")
    raw = planner.get("planning_diagnostics") if isinstance(planner, Mapping) else source.get("planning_diagnostics")
    value = public_planning_diagnostics(raw)
    return {"planning_diagnostics": value} if value else {}


def _public_planner_summary(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return None
    result = {k: copy.deepcopy(value[k]) for k in (
        "called", "status", "stage", "valid_query", "attempts", "trajopt_attempts",
        "used_graph", "goalset_index", "path_steps", "interpolation_dt_s",
        "errors", "timings_s", "error", "topk_selection", "constraint_rejection_counts",
    ) if k in value}
    diagnostic = public_planning_diagnostics(value.get("planning_diagnostics"))
    if diagnostic:
        result["planning_diagnostics"] = diagnostic
    return result


def _public_unified_arms(value: Any) -> dict[str, Any] | None:
    """Keep arm effects while isolating the canonical IK candidate evidence."""
    if not isinstance(value, Mapping):
        return None
    result = {}
    for arm in ("left", "right"):
        effect = value.get(arm)
        if not isinstance(effect, Mapping):
            continue
        result[arm] = {key: copy.deepcopy(effect[key]) for key in (
            "targeted", "interaction", "pose_error", "gripper_error", "gripper_outcome",
            "endpoint_contract", "plan_validation", "plan_ref", "segments",
            "route_constraints", "recovery_progress", "motion_progress", "inactive_arm_drift",
        ) if key in effect}
        result[arm].update(_planning_detail(effect))
        if "planner" in effect:
            result[arm]["planner"] = _public_planner_summary(effect["planner"])
    return result


@dataclass(frozen=True)
class ReceiptProjectionContext:
    """Session facts a projector may consult without touching the session."""

    last_action_id: str | None
    last_action_public_name: str | None
    surface_supports: Callable[[str], bool] | None = None

    def supports(self, capability: str) -> bool:
        if self.surface_supports is None:
            return False
        return bool(self.surface_supports(capability))


ReceiptEffectProjector = Callable[
    [Mapping[str, Any], ReceiptProjectionContext], dict[str, Any]
]

_RECEIPT_EFFECT_PROJECTORS: dict[str, ReceiptEffectProjector] = {}


def register_receipt_effect_projector(
    schemas: tuple[str, ...], projector: ReceiptEffectProjector,
) -> None:
    for schema in schemas:
        if schema in _RECEIPT_EFFECT_PROJECTORS:
            raise ValueError(
                f"receipt effect projector already registered: {schema}"
            )
        _RECEIPT_EFFECT_PROJECTORS[schema] = projector


def resolve_receipt_effect_projector(
    schema: Any,
) -> ReceiptEffectProjector | None:
    """The registered projector, wrapped by the capability-gated delivery rules."""
    from .effect_delivery import delivered_projector
    return delivered_projector(_RECEIPT_EFFECT_PROJECTORS.get(schema) if isinstance(schema, str) else None)


def public_controller_tracking(
    value: Any, *, contact_feedback: bool = False,
) -> dict[str, Any] | None:
    """Keep convergence evidence while hiding controller implementation details."""

    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {
        key: copy.deepcopy(value[key])
        for key in ("completed",)
        if key in value
    }
    abort = value.get("abort")
    if isinstance(abort, Mapping):
        raw_code = str(abort.get("code") or "execution_fault")
        code = public_motion_failure_code("execution", raw_code)
        result["abort"] = {
            "code": code,
            **({"arm": str(abort["arm"])} if abort.get("arm") else {}),
        }
        if contact_feedback:
            for key in ("blocked_at_stage", "frozen_steps", "progress", "stall_shape"):
                if key in abort:
                    result["abort"][key] = copy.deepcopy(abort[key])
                    result[key] = copy.deepcopy(abort[key])
    arms = value.get("arms")
    if isinstance(arms, Mapping):
        result["arms"] = {
            str(arm): {
                key: copy.deepcopy(record[key])
                for key in (
                    "targeted", "target_error",
                    "maximum_joint_tracking_error_rad",
                    "maximum_tcp_step_m",
                    "maximum_position_drift_from_action_start_m",
                )
                if key in record
            }
            for arm, record in arms.items()
            if isinstance(record, Mapping)
        }
    return result


def public_motion_failure_code(stage: Any, code: Any) -> str | None:
    if code is None:
        return None
    stage_name = str(stage or "")
    raw = str(code)
    if raw == "blocked_recovery_direction_unavailable":
        return raw
    if stage_name == "planner":
        # A failed solver request and an implementation failure require
        # different next decisions. Keep the exception text private, but
        # never tell the policy to change its target to repair our software.
        if raw in {"planner_error", "planner_internal_error", "planner_timeout"}:
            return "planner_error"
        if "INVALID_START_STATE_" in raw.upper() or raw == "invalid_start_state":
            return "invalid_start_state"
        return "planner_failed"
    if stage_name == "endpoint_verification":
        return "endpoint_residual_exceeded"
    if stage_name == "execution":
        if raw in {
            "local_servo_stalled", "controller_stalled",
        }:
            return "controller_stalled"
        if raw in {
            "local_servo_max_iterations", "waypoint_tracking_timeout",
            "terminal_endpoint_tracking_timeout", "terminal_tracking_timeout",
            "execution_step_budget_exhausted", "controller_timeout",
            "native_action_budget_exhausted",
        }:
            return "controller_timeout"
        if raw in {
            "inter_arm_collision", "payload_collision",
            "invalid_measured_state", "joint_target_rejected",
            "blocked", "contact_stop", "gripper_not_reached", "gripper_stalled_before_target",
            "episode_terminal_during_execution",
            # A recovery stopped mid-route because no self-clear order exists
            # from the measured joints, or a step would meet the other arm: the
            # same meaning as its pre-motion refusal, not an infrastructure fault.
            "recovery_self_collision_path", "recovery_inter_arm_collision_path",
        }:
            return raw
        if raw in {
            "sustained_joint_tracking_error",
            "waypoint_tracking_error",
        }:
            return "controller_tracking_error"
        if raw.startswith(("gross_", "workspace_")):
            return "safety_guard"
        return "execution_fault"
    return raw


PUBLIC_MOTION_EFFECT_SCHEMA = "ora.public-motion-effect.v1"
#: ``acceptance_evidence_status`` with and without executor-bound checks.
ACCEPTANCE_REPORTED = "reported"
ACCEPTANCE_UNAVAILABLE = "unavailable_in_this_receipt"
PUBLIC_MOTION_EFFECT_CAPABILITY = "public_motion_effect_v1"
CONTACT_MOTION_FEEDBACK_CAPABILITY = "contact_motion_feedback_v1"
PER_ARM_TRANSLATION_CAPABILITY = ARM_TRANSLATION_CAPABILITY


def public_acceptance_checks(record: Mapping[str, Any], context: ReceiptProjectionContext) -> dict[str, Any]:
    """Deliver only checks bound by the executor, never today's global defaults."""
    if not context.supports("current_public_robot_facts_v1"):
        return {}
    checks = validate_motion_acceptance_checks(record.get("acceptance_checks", []))
    tracking = record.get("controller_tracking")
    endpoint = tracking.get("terminal_endpoint_check") if isinstance(tracking, Mapping) else None
    if (isinstance(endpoint, Mapping) and endpoint.get("enabled") is True
            and isinstance(endpoint.get("final_errors"), Mapping) and endpoint["final_errors"]):
        checks.append(motion_acceptance_check(stage="controller_terminal_endpoint",
            errors=endpoint["final_errors"], position_tolerance_m=endpoint["position_tolerance_m"],
            orientation_tolerance_rad=endpoint["orientation_tolerance_rad"]))
    return {"acceptance_checks": checks,
            "acceptance_evidence_status": ACCEPTANCE_REPORTED if checks else ACCEPTANCE_UNAVAILABLE}


def _public_terminal_settle(record: Mapping[str, Any]) -> dict[str, Any]:
    """Explain the actual joint/TCP conjunction without changing its verdict."""
    segment = record.get("segment", record)
    execution = segment.get("execution") if isinstance(segment, Mapping) else None
    if not isinstance(execution, Mapping):
        return {}
    settle = execution.get("terminal_settle")
    if not isinstance(settle, Mapping):
        return {}
    def finite(value: Any) -> float | None:
        return float(value) if type(value) in (int, float) and math.isfinite(value) else None
    error = finite(execution.get("tracking_error_max_abs_rad"))
    limit = finite(settle.get("joint_tolerance_rad"))
    def endpoint(value: Any) -> dict[str, float | None] | None:
        return {key: finite(value.get(key)) for key in ("position_m", "orientation_rad")} if isinstance(value, Mapping) else None
    detail = {
        "phase": "terminal_settle", "completion_requires": "joint_and_tcp_bounds",
        "joint_error_max_abs_rad": error, "joint_tolerance_rad": limit,
        "joint_within_tolerance": error <= limit if error is not None and limit is not None else None,
        "steps_used": settle.get("ticks"), "step_budget": settle.get("max_ticks"),
        "budget_exhausted": settle.get("exhausted"),
        "tcp_endpoint_error": endpoint(settle.get("endpoint_error")),
        "tcp_endpoint_tolerance": endpoint(settle.get("endpoint_tolerance")),
        "claim": "TCP arrival alone does not imply joint convergence or task success",
    }
    return {"terminal_settle": detail}


# Executor names are truthful evidence, not policy-selectable options: the
# unified envelope must name what actually executed (roadmap: removing
# aliases must not hide a distinct executor).
_ROBOTWIN_STRATEGY_EXECUTORS = {
    "global": ("robotwin_vendored_capx_planner",),
    "local": ("robotwin_weighted_dls_local",),
    "global_then_local_settle": (
        "robotwin_vendored_capx_planner", "robotwin_weighted_dls_local",
    ),
    "global_precontact_then_local": (
        "robotwin_vendored_capx_planner", "robotwin_weighted_dls_local",
    ),
}


def public_motion_effect_envelope(
    *,
    action: str,
    status: str,
    executor: tuple[str, ...] | None,
    motion_strategy: Any = None,
    failure_stage: Any = None,
    failure_code: str | None = None,
    failed_arms: list[str] | None = None,
    round_terminated: Any = None,
    detail: Mapping[str, Any] | None = None,
    physical_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One cross-embodiment policy-visible action-end envelope.

    Top-level fields carry exactly what the policy needs for its next
    decision: the closed-enum outcome, who actually executed, how the action
    ended, and whether the round terminated. ``physical_cause`` is always
    ``unknown`` unless separate qualified evidence proves otherwise — the
    stop reason describes how the action ended, never its physical cause.
    ``failed_arms`` names which arm(s) the folded per-arm evidence blames,
    and appears only when there is a name to give.
    Backend-specific evidence stays under ``detail`` as an explicit
    divergence rather than a silent schema fork. The envelope carries no
    ``schema`` field: version identity lives in the ToolSurface capability
    and the audit layer, and the causal projection forbids schema fields in
    policy-visible content.
    """

    return {
        "action": action,
        "status": status,
        "executor": list(executor) if executor else None,
        "motion_strategy": copy.deepcopy(motion_strategy),
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        # Only present when the folded per-arm evidence actually names a
        # culprit; an always-null field would be one more constant to ignore.
        **({"failed_arms": list(failed_arms)} if failed_arms else {}),
        "round_terminated": round_terminated,
        "physical_cause": (
            public_contact_motion_cause(physical_evidence)
            if physical_evidence is not None else "unknown"
        ),
        "detail": copy.deepcopy(dict(detail)) if detail is not None else {},
    }


_PLANNER_SUCCESS_STATUS = "Success"


def _robotwin_failed_arms(record: Mapping[str, Any]) -> list[str]:
    """Fold the per-arm planner status into the arm names that failed.

    A multi-arm RoboTwin request is all-or-nothing: if either arm's plan
    fails, neither executes. The per-arm status is already recorded and
    already projected, but the top-level failure code says only that
    planning failed, so the policy cannot tell "retarget the left arm" from
    "never use both arms" -- and the cheaper lesson to learn is the wrong
    one. Fold the existing evidence; invent no new per-arm data.

    Single-arm requests are left alone: there ``target_arms`` already names
    the only candidate, and a field that merely repeats it is noise.
    """

    arms = record.get("arms")
    if not isinstance(arms, Mapping):
        return []
    targeted = [
        arm for arm in ("left", "right")
        if isinstance(arms.get(arm), Mapping) and arms[arm].get("targeted")
    ]
    if len(targeted) < 2:
        return []
    failed = []
    for arm in targeted:
        planner = arms[arm].get("planner")
        if not isinstance(planner, Mapping) or not planner.get("called"):
            continue
        status = planner.get("status")
        if isinstance(status, str) and status != _PLANNER_SUCCESS_STATUS:
            failed.append(arm)
    return failed


ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS = (
    "ora.robotwin-sparse-motion-receipt.v2",
    "ora.robotwin-sparse-motion-receipt.v3",
    "ora.robotwin-sparse-motion-receipt.v4",
    "ora.robotwin-sparse-motion-receipt.v5",
    "ora.robotwin-sparse-motion-receipt.v6",
    "ora.robotwin-sparse-motion-receipt.v7",
)

_VERIFIED_ENDPOINT_SCHEMAS = ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS[1:]
_CONTROLLER_TRACKING_SCHEMAS = ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS[2:]
_STRATEGY_SCHEMA = ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS[-1]


def _project_robotwin_sparse_motion(
    record: Mapping[str, Any], context: ReceiptProjectionContext,
) -> dict[str, Any]:
    raw_action = record.get("action")
    if isinstance(raw_action, Mapping):
        action_name = raw_action.get("action")
    else:
        action_name = raw_action
    arms = record.get("arms")
    synchronization = record.get("synchronization")
    if not isinstance(arms, Mapping) or not isinstance(synchronization, Mapping):
        return {"action": str(action_name), "status": "world_state_unknown"}
    receipt_action_id = record.get("action_id")
    public_action_name = (
        context.last_action_public_name
        if isinstance(receipt_action_id, str)
        and receipt_action_id == context.last_action_id
        else None
    )
    if not isinstance(public_action_name, str) or not public_action_name:
        public_action_name = (
            "recover_pose" if action_name == "recover_motion"
            and record.get("schema") == _STRATEGY_SCHEMA
            else str(action_name)
        )
    effects: dict[str, Any] = {}
    for arm in ("left", "right"):
        effect = arms.get(arm)
        if not isinstance(effect, Mapping):
            return {"action": str(action_name), "status": "world_state_unknown"}
        effects[arm] = {
            key: copy.deepcopy(effect.get(key))
            for key in (
                "targeted", "planner", "pose_error", "inactive_arm_drift",
                "command_preservation", "hold_controller",
            )
        }
        effects[arm]["planner"] = _public_planner_summary(effect.get("planner"))
        effects[arm]["plan_validation"] = public_plan_validation(
            effect.get("plan_validation")
        )
        # The canonical receipt retains the immutable artifact reference,
        # but the policy needs only to know whether replayable numeric
        # evidence was recorded.  Paths and hashes are Control provenance,
        # not action-selection state.
        effects[arm]["plan_artifact_recorded"] = isinstance(
            effect.get("plan_artifact"), Mapping,
        )
    verified_endpoint = record.get("schema") in _VERIFIED_ENDPOINT_SCHEMAS
    body = {
        "action": public_action_name,
        "status": (
            str(record.get("execution_status")) if verified_endpoint
            and isinstance(record.get("execution_status"), str)
            else "execution_fault" if record.get("execution_fault")
            else "executed" if record.get("state_changed") is True
            else "not_executed"
        ),
        "target_arms": copy.deepcopy(record.get("target_arms", [])),
        "synchronization": copy.deepcopy(synchronization),
        "arms": effects,
        **({
            "controller_tracking": public_controller_tracking(
                record.get("controller_tracking"),
                contact_feedback=context.supports(CONTACT_MOTION_FEEDBACK_CAPABILITY),
            ),
        } if record.get("schema") in _CONTROLLER_TRACKING_SCHEMAS else {}),
        **({
            "motion_strategy": copy.deepcopy(record.get("motion_strategy")),
            "failure_stage": record.get("failure_stage"),
            "failure_code": public_motion_failure_code(
                record.get("failure_stage"), record.get("failure_code"),
            ),
        } if record.get("schema") == _STRATEGY_SCHEMA else {}),
        "terminal_after_action": record.get("terminal_after_action"),
    }
    if context.supports(PUBLIC_MOTION_EFFECT_CAPABILITY):
        strategy = body.get("motion_strategy")
        executor = (
            _ROBOTWIN_STRATEGY_EXECUTORS.get(str(strategy))
            if isinstance(strategy, str) else None
        )
        detail = {
            key: body[key]
            for key in ("target_arms", "synchronization", "arms",
                        "controller_tracking")
            if key in body
        }
        detail.update(public_acceptance_checks(record, context))
        detail.update(public_gripper_detail(record, context), **public_wait_detail(record))
        physical_evidence = None
        if context.supports(CONTACT_MOTION_FEEDBACK_CAPABILITY):
            tracking = record.get("controller_tracking")
            abort = tracking.get("abort") if isinstance(tracking, Mapping) else None
            if isinstance(abort, Mapping) and abort.get("code") == "blocked":
                physical_evidence = {"blocked_evidence": abort.get("cause_evidence")}
            drift = _public_command_drift(record)
            if drift:
                detail["command_drift"] = drift
            clearance = public_inter_arm_clearance(record.get("inter_arm_clearance"))
            if clearance:
                detail["inter_arm_clearance"] = clearance
            recovery = record.get("blocked_recovery")
            if isinstance(recovery, Mapping):
                detail["blocked_recovery"] = {key: copy.deepcopy(recovery[key]) for key in (
                    "retreat_distance_m", "retreated_arms", "held_selected_arms",
                    "direction_claim", "retreat_completed", "global_planning_after_retreat",
                ) if key in recovery}
        return causal_provider_value(public_motion_effect_envelope(
            action=str(body["action"]),
            status=str(body["status"]),
            executor=executor,
            motion_strategy=body.get("motion_strategy"),
            failure_stage=body.get("failure_stage"),
            failure_code=body.get("failure_code"),
            failed_arms=_robotwin_failed_arms(record),
            round_terminated=body.get("terminal_after_action"),
            detail=detail,
            physical_evidence=physical_evidence,
        ))
    return causal_provider_value(body)


register_receipt_effect_projector(
    ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS, _project_robotwin_sparse_motion,
)


LIBERO_SPARSE_MOTION_RECEIPT_SCHEMA = (
    "ora.libero-sparse-motion-receipt.v1"
)
ROBODOJO_SPARSE_MOTION_RECEIPT_SCHEMA = (
    "ora.robodojo-sparse-motion-receipt.v1"
)


def _project_unified_sparse_motion(
    record: Mapping[str, Any], context: ReceiptProjectionContext,
) -> dict[str, Any]:
    """Project a born-unified sparse receipt (LIBERO, RoboDojo) into the envelope.

    These schemas already carry the cross-embodiment fields (EXP08A wiring
    v1), so the projector emits the envelope directly. The embedded
    backend evidence — LIBERO's legacy ``segment``, RoboDojo's per-segment
    IK/journal detail — stays Control-only and never enters the
    policy-visible projection.
    """

    receipt_action_id = record.get("action_id")
    public_action_name = (
        context.last_action_public_name
        if isinstance(receipt_action_id, str)
        and receipt_action_id == context.last_action_id
        else None
    )
    if not isinstance(public_action_name, str) or not public_action_name:
        public_action_name = str(record.get("action"))
    contact_feedback = context.supports(CONTACT_MOTION_FEEDBACK_CAPABILITY)
    per_arm = contact_feedback and context.supports(PER_ARM_TRANSLATION_CAPABILITY)
    contact_detail = (
        _libero_contact_detail(record, translations=not per_arm)
        if contact_feedback else {}
    )
    arms = _public_unified_arms(record.get("arms"))
    if context.supports(PER_ARM_TRANSLATION_CAPABILITY):
        arms = without_skipped_retract_diagnostics(arms, record)  # E11G7-001
    physical_evidence = contact_detail
    if per_arm:
        translations = arm_translations(record)
        arms = with_arm_translations(arms, translations)
        physical_evidence = cause_evidence(contact_detail, record, translations)
    return causal_provider_value(public_motion_effect_envelope(
        action=public_action_name,
        status=str(record.get("execution_status")),
        executor=(
            tuple(record["executor"])
            if isinstance(record.get("executor"), list) else None
        ),
        motion_strategy=record.get("motion_strategy"),
        failure_stage=record.get("failure_stage"),
        # A budget stop stays a timeout only while the per-arm progress shows a moving arm.
        failure_code=stall_evidenced_code(record, public_motion_failure_code(
            record.get("failure_stage"), record.get("failure_code"))),
        round_terminated=record.get("terminal_after_action"),
        detail={
            "target_arms": copy.deepcopy(record.get("target_arms", [])),
            "synchronization": copy.deepcopy(record.get("synchronization")),
            "arms": arms,
            **_planning_detail(record),
            **public_acceptance_checks(record, context),
            **public_gripper_detail(record, context), **public_wait_detail(record),
            **_public_terminal_settle(record),
            **({"contact_motion": contact_detail} if contact_detail else {}),
            **({"command_drift": _public_command_drift(record)}
               if context.supports(CONTACT_MOTION_FEEDBACK_CAPABILITY)
               and _public_command_drift(record) else {}),
        },
        physical_evidence=physical_evidence or None,
    ))


register_receipt_effect_projector(
    (LIBERO_SPARSE_MOTION_RECEIPT_SCHEMA, ROBODOJO_SPARSE_MOTION_RECEIPT_SCHEMA),
    _project_unified_sparse_motion,
)


def _project_libero_hybrid_motion(
    record: Mapping[str, Any], context: ReceiptProjectionContext,
) -> dict[str, Any]:
    """Recover and planner refusals use the same public action-end contract."""
    feedback = record.get("feedback")
    feedback = feedback if isinstance(feedback, Mapping) else {}
    motion_status = record.get("motion_status")
    if motion_status == "refused":
        status, stage = "plan_rejected", "planner"
        code = public_motion_failure_code(stage, feedback.get("planning_failure_reason"))
    elif motion_status == "reached":
        status, stage, code = "reached", None, None
    elif motion_status == "stopped":
        status, stage = "not_reached", "execution"
        code = public_motion_failure_code(stage, feedback.get("stop_reason"))
    else:
        status, stage, code = "not_reached", "endpoint_verification", "endpoint_residual_exceeded"
    action = record.get("action")
    if record.get("action_id") == context.last_action_id and context.last_action_public_name:
        action = context.last_action_public_name
    elif action == "recover_motion":
        action = "recover_pose"
    contact_detail = (_libero_contact_detail(record)
                      if context.supports(CONTACT_MOTION_FEEDBACK_CAPABILITY) else {})
    return causal_provider_value(public_motion_effect_envelope(
        action=str(action), status=status,
        executor=("libero_curobo_joint_position",) if record.get("planner") else None,
        failure_stage=stage, failure_code=code,
        round_terminated=record.get("terminal_after_action"),
        detail={
            **_planning_detail(record),
            **({"endpoint_contract": copy.deepcopy(record["endpoint_contract"])}
               if isinstance(record.get("endpoint_contract"), Mapping) else {}),
            **public_acceptance_checks(record, context),
            **public_gripper_detail(record, context),
            **_public_terminal_settle(record),
            **({"gripper_outcome": copy.deepcopy(record["gripper_outcome"])}
               if isinstance(record.get("gripper_outcome"), Mapping) else {}),
            **({"contact_motion": contact_detail} if contact_detail else {}),
            **({"command_drift": _public_command_drift(record)}
               if context.supports(CONTACT_MOTION_FEEDBACK_CAPABILITY)
               and _public_command_drift(record) else {}),
        },
        physical_evidence=contact_detail or None,
    ))


register_receipt_effect_projector(
    ("ora.libero-pro-hybrid-motion-receipt.v2",),
    _project_libero_hybrid_motion,
)


MANIPULATION_SKILL_SEQUENCE_RECEIPT_SCHEMA = (
    "ora.manipulation-skill-sequence-receipt.v1"
)


def _project_manipulation_skill_sequence(
    record: Mapping[str, Any], context: ReceiptProjectionContext,
) -> dict[str, Any]:
    """A closed skill's effect is its stages; without them it is a black box.

    The tool's own help promises the stage statuses, reasons and measurements
    on the next turn. Before this projector existed the session fell back to
    ``{action: unknown, status: world_state_unknown}``, so a policy that had
    just run a three-stage grasp learned nothing from it and had to re-ground
    to find out what its own action did.

    The driven receipts nested under each stage stay behind: they are
    controller evidence, audited by the evaluator under their own contracts,
    and their schema fields are forbidden in policy-visible content. What
    crosses is what a next decision turns on -- which stages ran, how each
    ended, what each measured, the absorbed verdict, and which stages a
    resume would still have to run.
    """

    stages_raw = record.get("stage_receipts")
    stages_raw = stages_raw if isinstance(stages_raw, list) else []
    stages: list[dict[str, Any]] = []
    verification: dict[str, Any] | None = None
    failure_stage: str | None = None
    failure_code: str | None = None
    for item in stages_raw:
        if not isinstance(item, Mapping):
            continue
        stage: dict[str, Any] = {
            "stage": item.get("stage"),
            "status": item.get("status"),
            "reason": item.get("reason"),
        }
        measurements = item.get("measurements")
        if isinstance(measurements, list) and measurements:
            stage["measurements"] = [
                dict(entry) for entry in measurements if isinstance(entry, Mapping)
            ]
        stages.append(stage)
        if isinstance(item.get("verification"), Mapping):
            verification = dict(item["verification"])
        if failure_stage is None and item.get("status") not in {
            "executed", "skipped", "not_requested", "terminated",
        }:
            failure_stage = (
                str(item["stage"]) if isinstance(item.get("stage"), str) else None
            )
            failure_code = (
                str(item["reason"]) if isinstance(item.get("reason"), str) else None
            )

    committed = record.get("committed_sequence")
    committed = list(committed) if isinstance(committed, list) else []
    ran = {item.get("stage") for item in stages}
    # A resume decision turns on what has not happened yet, so name it rather
    # than leaving the policy to subtract two lists.
    not_reached = [name for name in committed if name not in ran]

    detail: dict[str, Any] = {
        "skill": record.get("skill"),
        "candidate_id": record.get("candidate_id"),
        "committed_sequence": committed,
        "stages": stages,
        "stages_not_reached": not_reached,
        "stop_reason": record.get("stop_reason"),
        "resumed_from": record.get("resumed_from"),
        "resumable": record.get("status") == "stopped",
    }
    if record.get("status") == "terminated":
        detail["lifecycle"] = "environment_terminal"
    if verification is not None:
        detail["verification"] = verification

    action = record.get("skill")
    if record.get("action_id") == context.last_action_id and (
        context.last_action_public_name
    ):
        action = context.last_action_public_name
    return causal_provider_value(public_motion_effect_envelope(
        action=str(action),
        status=str(record.get("status") or "unknown"),
        executor=None,
        failure_stage=failure_stage,
        failure_code=failure_code,
        round_terminated=record.get("terminal_after_action"),
        detail=detail,
    ))


register_receipt_effect_projector(
    (MANIPULATION_SKILL_SEQUENCE_RECEIPT_SCHEMA,),
    _project_manipulation_skill_sequence,
)




__all__ = [
    "MANIPULATION_SKILL_SEQUENCE_RECEIPT_SCHEMA",
    "PUBLIC_MOTION_EFFECT_CAPABILITY",
    "CONTACT_MOTION_FEEDBACK_CAPABILITY",
    "PER_ARM_TRANSLATION_CAPABILITY",
    "PUBLIC_MOTION_EFFECT_SCHEMA",
    "ReceiptProjectionContext",
    "ReceiptEffectProjector",
    "ROBOTWIN_SPARSE_MOTION_RECEIPT_SCHEMAS",
    "public_motion_effect_envelope",
    "public_contact_motion_cause",
    "public_controller_tracking",
    "public_motion_failure_code",
    "public_plan_validation",
    "register_receipt_effect_projector",
    "resolve_receipt_effect_projector",
]
