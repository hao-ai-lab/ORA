"""Deterministic read-only evaluation, sealing and export inventory.

``evaluate`` reads one finished run directory and returns, and unless
``persist=False`` also writes, ``evaluation.json``. It runs a fixed sequence
of checks, one per kind of evidence, implemented in sibling modules:

- ``evaluation_checks``: RunPlan admission, derived metrics and context
  audit, the agent trace, the context manifest, policy state,
  submitted actions and the official verdict;
- ``evaluation_lossless``: lossless-context projections and artifacts;
- ``evaluation_learning``: the steps of ``assess_learning_eligibility``;
- ``evaluation_receipts``: flattening of nested physical motion receipts.

``seal`` evaluates a run, records its file inventory and makes those files
read-only; ``export`` archives a sealed run. Some names imported below are no
longer used here; they stay importable from this module for its callers.
"""

from __future__ import annotations

from ora.orchestrator.robodojo_evaluation import (
    ROBODOJO_VERDICT_SCHEMA, ROBODOJO_ROUND_ISOLATION_SCHEMA,
    ROBODOJO_ROUND_ISOLATION_CHECKS, ROBODOJO_TERMINAL_BASIS, ROBODOJO_POLICY_BUDGET_BASIS,
    _robodojo_verdict_untrusted, _robodojo_round_verdict_errors, _robodojo_worker_receipts, _robodojo_evidence_errors,
)

from ora.orchestrator.evidence_io import (
    EvaluationError, _hash, _load, _receipt_execution_fault,
)

import hashlib
import io
import time
import json
import os
import re
import stat
import subprocess
import tarfile
from ora.orchestrator.context_coverage import coverage_violations
from ora.orchestrator.artifact_provenance import (
    context_artifact_entry_errors, verify_artifact_provenance,
)

from pathlib import Path
from typing import Any, Mapping

from ora.agent.evidence import request_archive
from ora.agent.evidence.policy_state import verify_policy_state
from ora.agent.composition import ToolSurfaceError, resolve_tool_surface
from ora.agent.evidence.trajectory import verify_agent_trace
from ora.orchestrator.policy_finish import (
    admitted_abort_round, policy_finish_reasons, policy_finish_report,
)
from ora.agent.memory.artifacts import (
    CONTEXT_ARTIFACT_INDEX_SCHEMA,
    LOSSLESS_CONTEXT_RETENTION,
)
from ora.contracts.action import ActionValidationError, validate_runtime_action
from .context_audit import ContextAuditError, verify_context_audit, write_context_audit
from .metrics import RuntimeMetricsError, verify_runtime_metrics, write_runtime_metrics
from .wait_accounting import session_wait_accounting
from .paths import PROJECT_ROOT
from .spec import (
    RunSpecError, json_sha256, validate_run_plan, verify_launch_admission,
)
from .evaluation_checks import (
    AgentTraceEvidence, _agent_trace_evidence, _context_manifest_reasons,
    _derived_evidence_reasons, _evaluation_status, _expected_context_manifest_tools,
    _formal_plan_reasons, _learning_contract_reasons,
    _official_verdict_reasons, _policy_state_reasons, _submitted_action_reasons,
)
from .evaluation_learning import (
    _backend_execution_fault_reasons, _eligibility_record, _libero_control_reasons,
)
from .evaluation_receipts import _physical_motion_receipts, _referenced_json_artifact  # noqa: F401


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def assess_learning_eligibility(
    plan: Mapping[str, Any], result: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    *, run_root: Path | None = None,
) -> dict[str, Any]:
    """Separate behavioral evidence validity from cross-episode learnability.

    The steps live in ``evaluation_learning``: RoboDojo evidence, backend
    execution faults, and for LIBERO-Pro the qualified OSC controller
    evidence of every physical receipt.
    """

    reasons: list[str] = []
    if result.get("state") != "completed":
        reasons.append("episode_not_completed")
    if any(event.get("type") == "trusted_fault" for event in events):
        reasons.append("trusted_runtime_fault")
    runtime = plan.get("runtime", {})
    if runtime.get("kind") == "robodojo" and run_root is not None:
        reasons.extend(_robodojo_evidence_errors(run_root, plan, result, events))
    if run_root is not None:
        reasons.extend(_backend_execution_fault_reasons(run_root))
    if runtime.get("kind") == "libero_pro":
        reasons.extend(_libero_control_reasons(plan, runtime, events, run_root))
    return _eligibility_record(reasons)


def verify_multi_round_session(
    run_root: Path, plan: Mapping[str, Any], result: Mapping[str, Any],
    events: list[Mapping[str, Any]],
) -> list[str]:
    session = plan.get("session")
    if not isinstance(session, Mapping):
        return []
    errors: list[str] = []
    rounds = session.get("rounds")
    result_rounds = result.get("rounds")
    if (
        session.get("cadence") != "persistent_agent_k_rounds_v1"
        or not isinstance(rounds, int)
        or rounds < 1
        or result.get("schema") != "ora.session-result.v1"
        or result.get("official_success") is not None
        or result.get("termination") not in {
            "session_round_limit", "operator_interrupt",
        }
        or not isinstance(result_rounds, list)
        or len(result_rounds) != rounds
        or result.get("round_count") != rounds
    ):
        errors.append("multi_round_result_contract_invalid")
        return errors
    reset_events = [event for event in events if event.get("type") == "reset"]
    finished_events = [
        event for event in events if event.get("type") == "round_finished"
    ]
    if len(reset_events) != rounds or len(finished_events) != rounds:
        errors.append("multi_round_event_count_invalid")
    round_seeds = session.get("round_seeds")
    explicit_seed_schedule = isinstance(round_seeds, list)
    expected_seeds = (
        list(round_seeds)
        if explicit_seed_schedule
        else [plan.get("seed")] * rounds
    )
    if len(expected_seeds) != rounds:
        errors.append("multi_round_seed_schedule_invalid")
        expected_seeds = [None] * rounds
    for index, event in enumerate(reset_events[:rounds]):
        reset = event.get("record")
        if (
            not isinstance(reset, Mapping)
            or reset.get("seed") != expected_seeds[index]
        ):
            errors.append(f"multi_round_reset_seed_invalid:{index}")
    total_decisions = 0
    total_refused = 0
    total_model_calls = 0
    success_count = 0
    for index, record in enumerate(result_rounds):
        if not isinstance(record, Mapping) or record.get("round_index") != index:
            errors.append(f"multi_round_identity_invalid:{index}")
            continue
        if (
            (explicit_seed_schedule or "seed" in record)
            and record.get("seed") != expected_seeds[index]
        ):
            errors.append(f"multi_round_result_seed_invalid:{index}")
        # An abort the Harness admitted on recorded infrastructure ends only
        # its own round, as an invalid sample (policy_finish); any other
        # incomplete round still breaks a completed session.
        if (
            result.get("state") == "completed" and record.get("state") != "completed"
            and not admitted_abort_round(record)
        ):
            errors.append(f"multi_round_state_invalid:{index}")
        if (
            plan.get("runtime", {}).get("kind") == "libero_pro"
            and (
                (
                    record.get("termination_authority") == "infrastructure"
                    and record.get("official_success") is not None
                )
                or (
                    record.get("termination_authority") != "infrastructure"
                    and not isinstance(record.get("official_success"), bool)
                )
            )
        ):
            errors.append(f"multi_round_libero_verdict_invalid:{index}")
        path = run_root / "rounds" / f"round-{index:04d}" / "result.json"
        try:
            persisted = _load(path)
        except EvaluationError:
            errors.append(f"multi_round_result_missing:{index}")
            continue
        if persisted != dict(record):
            errors.append(f"multi_round_result_drift:{index}")
        interaction = (
            run_root / "native" / "rounds" / f"round-{index:04d}"
            / "episode-interaction.json"
        )
        if not interaction.is_file():
            errors.append(f"multi_round_interaction_missing:{index}")
        decisions = record.get("decisions")
        calls = record.get("model_calls")
        if not isinstance(decisions, int) or decisions < 0:
            errors.append(f"multi_round_decisions_invalid:{index}")
        else:
            total_decisions += decisions
        # Absent on runs recorded before refusals stopped spending decisions.
        refused = record.get("refused_proposals", 0)
        if type(refused) is not int or refused < 0:
            errors.append(f"multi_round_refused_proposals_invalid:{index}")
        else:
            total_refused += refused
        if not isinstance(calls, int) or calls < 0:
            errors.append(f"multi_round_model_calls_invalid:{index}")
        else:
            total_model_calls += calls
        # Not type-checked here on purpose: this validator is runtime-neutral
        # and Synthetic M0 has no official verdict authority at all, so its
        # rounds carry null by design. Requiring a per-round boolean belongs
        # with the runtime that has an official boundary, which is where the
        # RoboTwin trusted-verdict check applies it.
        success_count += record.get("official_success") is True
    if (
        total_decisions != result.get("decisions")
        or total_refused != result.get("refused_proposals", 0)
        or total_model_calls != result.get("model_calls")
        or success_count != result.get("round_success_count")
    ):
        errors.append("multi_round_aggregate_mismatch")
    errors.extend(_wait_accounting_errors(result))
    return errors


def _wait_accounting_errors(result: Mapping[str, Any]) -> list[str]:
    """The session wait totals must be the sum of the round blocks.

    Runs recorded before the accounting existed carry it nowhere; a block on
    only one side, or totals that differ from the rounds', is a mismatch. The
    accounting is measurement only and never touches a verdict.
    """
    rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
    present = "wait_accounting" in result or any(
        isinstance(record, Mapping) and "wait_accounting" in record for record in rounds
    )
    if not present:
        return []
    records = [record for record in rounds if isinstance(record, Mapping)]
    expected = session_wait_accounting(records) if len(records) == len(rounds) else None
    if expected is None or result.get("wait_accounting") != expected:
        return ["multi_round_wait_accounting_mismatch"]
    return []


def evaluate(
    run_root: Path, *, identity: str = "live", persist: bool = True,
) -> dict[str, Any]:
    """Evaluate evidence; ``persist=False`` never changes the source run.

    Read-only evaluation verifies existing derived metrics/audits rather than
    replacing them, so historical derivation differences remain explicit.

    The checks run in a fixed order: RunPlan admission, derived metrics and
    context audit, the multi-round session, the agent trace and its context
    retention, the context manifest, policy
    state, the event stream with learning eligibility, and the official
    verdict. Each failed check adds a reason; the report lists them sorted.
    """
    if identity not in {"live", "backfill"}:
        raise EvaluationError("unknown evaluation identity mode")
    run_root = run_root.resolve()
    plan = _load(run_root / "plan.json")
    result = _load(run_root / "result.json")
    events_path = run_root / "events.jsonl"
    events = _load_events(events_path)
    event_types = [event.get("type") for event in events]
    reasons: list[str] = []
    reasons.extend(_formal_plan_reasons(run_root, plan, identity))
    reasons.extend(_derived_evidence_reasons(run_root, persist))
    if result.get("plan_sha256") != plan.get("plan_sha256"):
        reasons.append("plan_result_hash_mismatch")
    reasons.extend(verify_multi_round_session(run_root, plan, result, events))
    agent_trace = _agent_trace_evidence(run_root, plan)
    reasons.extend(agent_trace.reasons)
    context_manifest_path = run_root / "context-manifest.json"
    reasons.extend(_context_manifest_reasons(run_root, plan, context_manifest_path))
    reasons.extend(_policy_state_reasons(run_root, plan, result))
    if not event_types or event_types[0] != "run_started":
        reasons.append("missing_run_start")
    if "trusted_fault" in event_types:
        reasons.append("trusted_fault")
    observations = event_types.count("observation")
    actions = event_types.count("action_submitted")
    motions = event_types.count("motion_receipt")
    learning_eligibility = assess_learning_eligibility(
        plan, result, events, run_root=run_root,
    )
    reasons.extend(_learning_contract_reasons(plan, learning_eligibility))
    if not (observations >= actions == motions):
        reasons.append("causal_count_mismatch")
    reasons.extend(_submitted_action_reasons(events))
    if plan.get("runtime", {}).get("kind") == "robodojo":
        reasons.extend(_robodojo_evidence_errors(run_root, plan, result, events))
    reasons.extend(_official_verdict_reasons(plan, result))
    reasons.extend(policy_finish_reasons(result, run_root=run_root, events=events))
    evaluation = _evaluation_report(
        run_root, plan, result,
        status=_evaluation_status(result, reasons), reasons=reasons,
        counts=(observations, actions, motions),
        learning_eligibility=learning_eligibility,
        events_path=events_path, context_manifest_path=context_manifest_path,
        agent_trace=agent_trace,
    )
    if persist:
        _atomic(run_root / "evaluation.json", evaluation)
    return evaluation


def _load_events(events_path: Path) -> list[Any]:
    events = []
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid events trace: {exc}") from exc
    return events


def _evaluation_report(
    run_root: Path, plan: Mapping[str, Any], result: Mapping[str, Any], *,
    status: str, reasons: list[str], counts: tuple[int, int, int],
    learning_eligibility: dict[str, Any], events_path: Path,
    context_manifest_path: Path, agent_trace: AgentTraceEvidence,
) -> dict[str, Any]:
    """Assemble the evaluation record, binding the hashes of its evidence."""

    observations, actions, motions = counts
    agent_trace_check = agent_trace.check
    artifact_provenance_check = agent_trace.artifact_provenance
    evaluation = {
        "schema": "ora.evaluation.v1",
        "run_id": result.get("run_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "status": status,
        "reasons": sorted(set(reasons)),
        "official_success": result.get("official_success"),
        "evidence_tier": plan.get("evidence_tier"),
        "counts": {
            "observations": observations,
            "actions": actions,
            "motions": motions,
        },
        "learning_eligibility": learning_eligibility,
        "trace_sha256": _hash(events_path),
        "result_sha256": _hash(run_root / "result.json"),
        "context_manifest_sha256": (
            _hash(context_manifest_path) if context_manifest_path.is_file() else None
        ),
        "policy_state_manifest_sha256": (
            _hash(run_root / "policy-state/manifest.json")
            if (run_root / "policy-state/manifest.json").is_file() else None
        ),
        "agent_session_manifest_sha256": (
            _hash(run_root / "native/agent-session.json")
            if (run_root / "native/agent-session.json").is_file() else None
        ),
        "agent_trace_sha256": (
            _hash(run_root / "agent-events.jsonl")
            if (run_root / "agent-events.jsonl").is_file() else None
        ),
        "agent_trace_final_event_sha256": (
            agent_trace_check.get("final_event_sha256")
            if isinstance(agent_trace_check, Mapping) else None
        ),
        "runtime_metrics_sha256": (
            _hash(run_root / "runtime-metrics.json")
            if (run_root / "runtime-metrics.json").is_file() else None
        ),
        "context_audit_sha256": (
            _hash(run_root / "context-audit.json")
            if (run_root / "context-audit.json").is_file() else None
        ),
        "coding_audit_sha256": (
            _hash(run_root / "native/coding-audit.jsonl")
            if (run_root / "native/coding-audit.jsonl").is_file() else None
        ),
    }
    if plan.get("runtime", {}).get("kind") == "robodojo":
        # The declared and resolved physics device of the run (CPU-physics lane).
        from ora.orchestrator.robodojo_physics_attestation import physics_record
        evaluation["robodojo_physics"] = physics_record(plan, run_root / "native")
    finishes = policy_finish_report(result)
    if finishes is not None:
        # Only runs on a v2 finish contract carry it.
        evaluation["policy_finish"] = finishes
    if artifact_provenance_check is not None:
        evaluation["context_artifact_provenance"] = artifact_provenance_check
    return evaluation


def inventory(run_root: Path) -> list[dict[str, Any]]:
    excluded = {"seal.json"}
    records: list[dict[str, Any]] = []
    paths: list[Path] = []
    reserved_aliases = {
        "sandbox/workspace/scratch/AGENTS.md": "../AGENTS.md",
        "sandbox/workspace/scratch/CLAUDE.md": "../CLAUDE.md",
        "sandbox/workspace/scratch/contract": "../contract",
        "sandbox/workspace/scratch/inputs": "../inputs",
    }
    for directory, names, files in os.walk(run_root, followlinks=False):
        base = Path(directory)
        for name in [*names, *files]:
            candidate = base / name
            metadata = os.lstat(candidate)
            if stat.S_ISLNK(metadata.st_mode):
                relative = candidate.relative_to(run_root).as_posix()
                if reserved_aliases.get(relative) == os.readlink(candidate):
                    continue
                raise EvaluationError(
                    f"symbolic links cannot enter sealed evidence: {candidate.relative_to(run_root)}"
                )
        for name in files:
            candidate = base / name
            if not stat.S_ISLNK(os.lstat(candidate).st_mode):
                paths.append(candidate)
    for path in sorted(paths):
        relative = path.relative_to(run_root).as_posix()
        if relative in excluded or relative.startswith("exports/") or relative.endswith(".tmp"):
            continue
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode):
            raise EvaluationError(f"non-regular artifact cannot be sealed: {relative}")
        if metadata.st_nlink != 1:
            raise EvaluationError(f"hard-linked artifact cannot be sealed: {relative}")
        lowered = relative.lower()
        if any(fragment in lowered for fragment in ("credential", "api-key", "auth.json")):
            raise EvaluationError(f"credential-like artifact cannot be sealed: {relative}")
        records.append({"path": relative, "sha256": _hash(path), "bytes": metadata.st_size})
    return records


def seal(run_root: Path, *, mode: str = "operator") -> dict[str, Any]:
    """Evaluate and seal a run."""
    if mode not in {"operator", "landing", "backfill"}:
        raise EvaluationError("unknown seal mode")
    run_root = run_root.resolve()
    if (run_root / "seal.json").exists():
        raise EvaluationError("run is already sealed")
    plan = _load(run_root / "plan.json")
    admission_mode = "backfill" if mode == "backfill" else "live"
    admission_verification = "absent"
    formal = (run_root / "launch.json").is_file() and isinstance(
        plan.get("plan_sha256"), str,
    ) and re.fullmatch(
        r"[0-9a-f]{64}", plan["plan_sha256"],
    )
    if formal:
        try:
            validate_run_plan(
                plan, run_root=run_root, admission=admission_mode,
            )
            admission_verification = verify_launch_admission(
                plan, run_root=run_root, admission=admission_mode,
            )
        except RunSpecError as exc:
            raise EvaluationError(str(exc)) from exc
    head = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    dirty = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=normal"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    sealed_source_commit = head.stdout.strip() if not head.returncode else None
    sealed_source_dirty = (
        bool(dirty.stdout.strip()) if not dirty.returncode else None
    )
    backfill_evaluator_commit = None
    backfill_evaluator_dirty = None
    if mode == "backfill":
        # A backfill seals evidence produced under the run's own recorded
        # source, possibly long ago. The seal pins that recorded identity;
        # the repository state of the machine running the backfill is a
        # property of the evaluator, not of the evidence, and is recorded
        # separately instead of overwriting the evidence identity.
        source = plan.get("source")
        if not isinstance(source, Mapping) or not isinstance(
            source.get("commit"), str,
        ):
            raise EvaluationError(
                "backfill requires the run's recorded source identity"
            )
        backfill_evaluator_commit = sealed_source_commit
        backfill_evaluator_dirty = sealed_source_dirty
        sealed_source_commit = source["commit"]
        sealed_source_dirty = source.get("dirty")
    if formal and mode != "backfill":
        if (
            head.returncode or dirty.returncode
            or sealed_source_commit != plan["source"]["commit"]
            or sealed_source_dirty
        ):
            raise EvaluationError(
                "seal source differs from the admitted clean RunPlan"
            )
    if mode == "backfill":
        # Keep the pre-backfill evaluation as evidence of the original
        # verdict instead of silently overwriting it.
        previous = run_root / "evaluation.json"
        superseded = run_root / "evaluation-superseded-by-backfill.json"
        if previous.is_file() and not superseded.exists():
            superseded.write_bytes(previous.read_bytes())
    evaluation = evaluate(
        run_root, identity="backfill" if mode == "backfill" else "live",
    )
    if evaluation.get("status") not in {"valid", "incomplete"}:
        raise EvaluationError("invalid runs cannot be sealed as evidence")
    records = inventory(run_root)
    body = {
        "schema": "ora.seal.v2",
        "run_id": evaluation["run_id"],
        "mode": mode,
        "evaluation_status": evaluation["status"],
        "sealed_source_commit": sealed_source_commit,
        "sealed_source_dirty": sealed_source_dirty,
        "launch_admission_verification": admission_verification,
        **(
            {
                "backfill_evaluator_commit": backfill_evaluator_commit,
                "backfill_evaluator_dirty": backfill_evaluator_dirty,
            }
            if mode == "backfill" else {}
        ),
        "inventory": records,
    }
    if isinstance(evaluation.get("robodojo_physics"), Mapping):
        body["physics_device"] = evaluation["robodojo_physics"]["physics_device"]
    body["inventory_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic(run_root / "seal.json", body)
    for record in records:
        path = run_root / record["path"]
        metadata = os.lstat(path)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != record["bytes"]
            or _hash(path) != record["sha256"]
        ):
            (run_root / "seal.json").unlink(missing_ok=True)
            raise EvaluationError(f"inventory changed before sealing: {record['path']}")
        path.chmod(0o444)
    return body


def export(run_root: Path, destination: Path) -> Path:
    run_root = run_root.resolve()
    seal_record = _load(run_root / "seal.json")
    archived: set[str] = set()
    for record in seal_record["inventory"]:
        path = run_root / record["path"]
        try:
            metadata = os.lstat(path)
        except OSError:
            metadata = None
        if metadata is None and request_archive.inventory_record_present(run_root, record):
            archived.add(record["path"])  # stored losslessly in the run's request archive
            continue
        if (
            metadata is None
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size != record["bytes"]
            or _hash(path) != record["sha256"]
        ):
            raise EvaluationError(f"sealed inventory drift: {record['path']}")
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise EvaluationError(f"refusing to overwrite export: {destination}")
    if destination.is_relative_to(run_root):
        raise EvaluationError("export destination must remain outside the sealed run")
    with tarfile.open(destination, "w:gz") as archive:
        for record in seal_record["inventory"]:
            if record["path"] in archived:
                data = request_archive.read_bytes(run_root, record["path"])
                info = tarfile.TarInfo(f"{run_root.name}/{record['path']}")
                info.size, info.mode, info.mtime = len(data), 0o444, int(time.time())
                archive.addfile(info, io.BytesIO(data))
                continue
            archive.add(
                run_root / record["path"],
                arcname=f"{run_root.name}/{record['path']}",
                recursive=False,
            )
        archive.add(
            run_root / "seal.json",
            arcname=f"{run_root.name}/seal.json",
            recursive=False,
        )
    return destination
