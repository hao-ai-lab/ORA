"""Per-evidence-kind checks that ``evaluation.evaluate`` runs in order.

Each check reads one kind of run evidence and returns the reasons it found,
plus any verification record the evaluation report carries. The checks never
write to the run directory except ``_derived_evidence_reasons`` when the caller
persists the derived metrics and context audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ora.agent.composition import ToolSurfaceError, resolve_tool_surface
from ora.agent.evidence.policy_state import verify_policy_state
from ora.agent.evidence.trajectory import verify_agent_trace
from ora.agent.memory.artifacts import LOSSLESS_CONTEXT_RETENTION
from ora.contracts.action import ActionValidationError, validate_runtime_action
from ora.orchestrator.evidence_io import EvaluationError, _load

from .context_audit import ContextAuditError, verify_context_audit, write_context_audit
from .evaluation_lossless import _lossless_context_evidence
from .metrics import RuntimeMetricsError, verify_runtime_metrics, write_runtime_metrics
from .policy_finish import admitted_abort_round
from .spec import RunSpecError, validate_run_plan


@dataclass(frozen=True)
class AgentTraceEvidence:
    """What the agent-event trace contributes to one evaluation."""

    reasons: list[str]
    check: dict[str, Any] | None = None
    artifact_provenance: dict[str, Any] | None = None


def _expected_context_manifest_tools(plan: Mapping[str, Any]) -> list[Any]:
    """Return the exact ordered ToolSurface recorded by the resolved RunPlan.

    Current plans snapshot derived context tools such as ``artifact.read`` in
    ``policy_composition.tool_surface.definitions``.  Re-resolving only the
    frozen base surface would silently omit those derived layers during the
    downstream manifest audit.  The legacy reconstruction remains solely for
    older plans that predate the effective ToolSurface snapshot.
    """

    composition = plan.get("policy_composition", {})
    planned_definitions = composition.get("tool_surface", {}).get("definitions")
    if isinstance(planned_definitions, list):
        expected = [
            item.get("name") if isinstance(item, Mapping) else None
            for item in planned_definitions
        ]
    else:
        configured_tools = set(plan.get("policy", {}).get(
            "tool_allowlist", ["robot.submit_action"]
        ))
        surface_config = plan.get("policy", {}).get("tool_surface", {})
        surface = resolve_tool_surface(
            surface_config.get("profile", "robot_runtime_v1"),
            component_overrides=surface_config.get("component_overrides", {}),
        )
        expected = [
            item["name"] for item in surface.definitions
            if item["name"] in configured_tools
        ]
    return expected


def _formal_plan_reasons(
    run_root: Path, plan: Mapping[str, Any], identity: str,
) -> list[str]:
    """A launched formal RunPlan must keep its identity and preflight binding."""

    reasons: list[str] = []
    formal_plan = isinstance(plan.get("plan_sha256"), str) and bool(
        re.fullmatch(r"[0-9a-f]{64}", plan["plan_sha256"])
    )
    if formal_plan and (run_root / "launch.json").is_file():
        try:
            validate_run_plan(
                plan, run_root=run_root,
                admission="backfill" if identity == "backfill" else "live",
            )
        except RunSpecError:
            reasons.append("run_plan_identity_invalid")
        try:
            preflight = _load(run_root / "preflight.json")
        except EvaluationError:
            reasons.append("preflight_binding_missing")
        else:
            if (
                preflight.get("schema")
                != "ora.preflight.v1"
                or preflight.get("passed") is not True
                or preflight.get("plan_id") != plan.get("plan_id")
                or preflight.get("plan_sha256") != plan.get("plan_sha256")
            ):
                reasons.append("preflight_binding_invalid")
    return reasons


def _derived_evidence_reasons(run_root: Path, persist: bool) -> list[str]:
    """Derive (when persisting) and verify runtime metrics and context audit."""

    reasons: list[str] = []
    try:
        if persist:
            write_runtime_metrics(run_root)
        runtime_metrics_check = verify_runtime_metrics(run_root)
    except RuntimeMetricsError as exc:
        runtime_metrics_check = {
            "status": "invalid", "errors": [str(exc)], "metrics": {},
        }
    if runtime_metrics_check["status"] != "valid":
        reasons.extend(runtime_metrics_check["errors"])
    try:
        if persist:
            write_context_audit(run_root)
        context_audit_check = verify_context_audit(run_root)
    except ContextAuditError as exc:
        context_audit_check = {"status": "invalid", "errors": [str(exc)]}
    if context_audit_check["status"] != "valid":
        reasons.extend(context_audit_check["errors"])
    return reasons


def _agent_trace_evidence(
    run_root: Path, plan: Mapping[str, Any],
) -> AgentTraceEvidence:
    """Verify the agent trace and the context retention it records."""

    if not (run_root / "agent-events.jsonl").is_file():
        return AgentTraceEvidence(reasons=[])
    reasons: list[str] = []
    agent_trace_check = verify_agent_trace(run_root)
    if agent_trace_check["status"] != "valid":
        reasons.extend(
            f"agent_trace:{error}" for error in agent_trace_check["errors"]
        )
    retention = plan.get("policy", {}).get("context_retention")
    if retention != LOSSLESS_CONTEXT_RETENTION:
        return AgentTraceEvidence(reasons=reasons, check=agent_trace_check)
    lossless = _lossless_context_evidence(run_root)
    reasons.extend(lossless.reasons)
    return AgentTraceEvidence(
        reasons=reasons, check=agent_trace_check,
        artifact_provenance=lossless.artifact_provenance,
    )


def _context_manifest_reasons(
    run_root: Path, plan: Mapping[str, Any], context_manifest_path: Path,
) -> list[str]:
    """The persisted context manifest must match the plan it describes."""

    reasons: list[str] = []
    if "policy_context" not in plan:
        return reasons
    if not context_manifest_path.is_file():
        reasons.append("missing_context_manifest")
        return reasons
    try:
        context_manifest = _load(context_manifest_path)
        if context_manifest.get("plan_sha256") != plan.get("plan_sha256"):
            reasons.append("context_manifest_plan_hash_mismatch")
        try:
            expected_tools = _expected_context_manifest_tools(plan)
        except ToolSurfaceError:
            reasons.append("context_manifest_tool_surface_profile_invalid")
            expected_tools = []
        observed_tools = [
            item.get("name")
            for item in context_manifest.get("tool_surface", {}).get(
                "advertised_in_order", []
            )
        ]
        if observed_tools != expected_tools:
            reasons.append("context_manifest_tool_surface_mismatch")
        environment = context_manifest.get("environment", {})
        if (
            environment.get("session_contract")
            != "ora.environment-session.v1"
        ):
            reasons.append("context_manifest_environment_contract_mismatch")
        if environment.get("backend") != plan.get("runtime", {}).get("kind"):
            reasons.append("context_manifest_environment_backend_mismatch")
    except EvaluationError:
        reasons.append("invalid_context_manifest")
    return reasons


def _policy_state_reasons(
    run_root: Path, plan: Mapping[str, Any], result: Mapping[str, Any],
) -> list[str]:
    """Policy state, context bundles and the agent transcript must agree."""

    reasons: list[str] = []
    state_verification = verify_policy_state(run_root)
    if state_verification["status"] != "valid":
        reasons.extend(state_verification["errors"])
        return reasons
    state_manifest = state_verification["manifest"]
    if state_manifest.get("run_id") != result.get("run_id"):
        reasons.append("state_manifest_run_id_mismatch")
    if state_manifest.get("plan_sha256") != plan.get("plan_sha256"):
        reasons.append("state_manifest_plan_hash_mismatch")
    if state_manifest.get("finalized") is not True:
        reasons.append("state_manifest_not_finalized")
    calls = state_manifest.get("context_calls", [])
    transcript_path = run_root / "agent-transcript.jsonl"
    transcript: list[dict[str, Any]] = []
    if transcript_path.is_file():
        try:
            transcript = [
                json.loads(line)
                for line in transcript_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            reasons.append("invalid_agent_transcript")
    if len(transcript) != len(calls):
        reasons.append("context_transcript_count_mismatch")
    for index, (call, record) in enumerate(zip(transcript, calls)):
        linked = call.get("input", {}).get("context_bundle", {})
        if (
            call.get("call") != index
            or linked.get("path") != record.get("path")
            or linked.get("sha256") != record.get("sha256")
            or linked.get("bundle_sha256") != record.get("bundle_sha256")
        ):
            reasons.append(f"context_transcript_link_mismatch:{index}")
    return reasons


def _learning_contract_reasons(
    plan: Mapping[str, Any], learning_eligibility: Mapping[str, Any],
) -> list[str]:
    """Promote receipt-contract failures that also invalidate the run."""

    reasons: list[str] = []
    if plan.get("runtime", {}).get("kind") == "libero_pro" and any(
        reason.startswith("control_health_fault")
        or reason == "missing_or_invalid_controller_action_contract"
        or reason == "missing_or_invalid_controller_source_qualification"
        or reason.startswith("missing_or_invalid_control_health")
        or reason.startswith("missing_or_invalid_hybrid_motion_evidence")
        or reason.startswith("missing_or_invalid_pose_tracking")
        for reason in learning_eligibility["reasons"]
    ):
        reasons.append("libero_control_contract_invalid_or_faulted")
    return reasons


def _submitted_action_reasons(events: list[Mapping[str, Any]]) -> list[str]:
    """Every submitted action must satisfy the shared Runtime action contract."""

    reasons: list[str] = []
    for event in events:
        if event.get("type") == "action_submitted":
            action = event.get("action", {})
            try:
                validate_runtime_action(action)
            except ActionValidationError:
                reasons.append("forbidden_policy_action")
    return reasons


def _official_verdict_reasons(
    plan: Mapping[str, Any], result: Mapping[str, Any],
) -> list[str]:
    """A completed RoboTwin or RoboDojo run must carry a boolean verdict."""

    completed = result.get("state") == "completed"
    runtime_kind = plan.get("runtime", {}).get("kind")
    plan_session = plan.get("session")
    multi_round = (
        isinstance(plan_session, Mapping)
        and plan_session.get("cadence") == "persistent_agent_k_rounds_v1"
    )
    if not (completed and runtime_kind in {"robotwin", "robodojo"}):
        return []
    # A RoboTwin or RoboDojo episode must carry the simulator's own
    # boolean verdict. A persistent multi-round session holds one verdict
    # per round, and verify_multi_round_session requires the aggregate
    # field to stay null, so demanding a top-level boolean made every
    # multi-round run unsatisfiable against that same file's other
    # contract: three-for-three successes were rejected with the same
    # reason as a worker that never started. Check the verdicts where a
    # multi-round run actually records them, and keep the single-episode
    # rule as it was. LIBERO never hit this only because the guard names
    # the runtimes with an official boundary, which is an exemption
    # rather than multi-round support.
    if multi_round:
        round_records = result.get("rounds")
        # A round an admitted infrastructure abort ended carries no verdict
        # by design (policy_finish); every other round needs its Boolean.
        if not isinstance(round_records, list) or not round_records or not all(
            isinstance(record, Mapping)
            and (isinstance(record.get("official_success"), bool) or admitted_abort_round(record))
            for record in round_records
        ):
            return ["missing_trusted_boolean_verdict"]
    elif not isinstance(result.get("official_success"), bool):
        return ["missing_trusted_boolean_verdict"]
    return []


def _evaluation_status(result: Mapping[str, Any], reasons: list[str]) -> str:
    """An incomplete run stays incomplete only for its expected reasons."""

    if result.get("state") == "incomplete":
        allowed_incomplete_reasons = {"trusted_fault", "causal_count_mismatch"}
        return (
            "incomplete"
            if set(reasons).issubset(allowed_incomplete_reasons)
            else "invalid"
        )
    if reasons:
        return "invalid"
    return "valid"
