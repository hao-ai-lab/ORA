"""Behavioral-ownership audit over one completed run's canonical evidence.

The Harness requires auditing behavioral ownership in addition to nominal tool
calls: how many decisions the Language Model made, how many explicit
robot-semantic Actions it selected, how much work was delegated to the
planners and the controllers, and where every
re-observation boundary sat. Until now those counts were derived by hand
from sealed traces; this module rebuilds them
deterministically from the run's append-only evidence.

The report describes selection and delegation. It deliberately makes no
competence claim: a one-call macro does not count as direct Language-Model
action competence merely because the Agent selected it, and official success
remains the Environment's verdict alone.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from ora.agent.tools import tool_names


OWNERSHIP_AUDIT_SCHEMA = "ora.behavioral-ownership-audit.v1"


class OwnershipAuditError(RuntimeError):
    """The run evidence needed for the ownership audit is unavailable."""


def _load_object(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise OwnershipAuditError(f"required evidence is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OwnershipAuditError(f"cannot read evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OwnershipAuditError(f"evidence must be a JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise OwnershipAuditError(
                f"cannot read event line in {path}: {exc}"
            ) from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def _find_surface_snapshot(value: Any) -> Mapping[str, Any] | None:
    """Locate the resolved ToolSurface snapshot inside plan.json."""

    if isinstance(value, Mapping):
        if (
            isinstance(value.get("profile_id"), str)
            and isinstance(value.get("mutation_tools"), list)
        ):
            return value
        for child in value.values():
            found = _find_surface_snapshot(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_surface_snapshot(child)
            if found is not None:
                return found
    return None


def _receipt_iter(events: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for event in events:
        if event.get("type") != "motion_receipt":
            continue
        record = event.get("record")
        if isinstance(record, Mapping):
            yield record


def _planner_evidence(record: Mapping[str, Any]) -> tuple[bool, int]:
    """Detect hidden planner/controller work inside one receipt."""

    planner_used = False
    sampled_steps = 0
    arms = record.get("arms")
    if isinstance(arms, Mapping):
        for effect in arms.values():
            if not isinstance(effect, Mapping):
                continue
            if isinstance(effect.get("plan_validation"), Mapping):
                planner_used = True
            planner = effect.get("planner")
            if isinstance(planner, Mapping) and planner.get("called"):
                planner_used = True
    tracking = record.get("controller_tracking")
    if isinstance(tracking, Mapping):
        steps = tracking.get("sampled_steps")
        if isinstance(steps, int) and not isinstance(steps, bool):
            sampled_steps = max(steps, 0)
    return planner_used, sampled_steps


def build_ownership_audit(run_root: Path) -> dict[str, Any]:
    """Derive one deterministic behavioral-ownership report from a run."""

    run_root = run_root.resolve()
    plan = _load_object(run_root / "plan.json", required=True)
    events = _load_jsonl(run_root / "events.jsonl")
    agent_events = _load_jsonl(run_root / "agent-events.jsonl")
    session = _load_object(run_root / "native/agent-session.json")

    surface = _find_surface_snapshot(plan)
    mutation_tools = set(
        surface.get("mutation_tools", []) if surface is not None else []
    )
    lifecycle_tools = set(
        surface.get("lifecycle_tools", []) if surface is not None else []
    )

    calls = session.get("calls")
    model_calls = len(calls) if isinstance(calls, list) else None
    proposed = sum(
        1 for event in agent_events if event.get("type") == "action_proposed"
    )

    tool_results: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    for event in events:
        name = event.get("tool")
        if not isinstance(name, str):
            continue
        if event.get("type") == "tool_result":
            tool_results[name] += 1
        elif event.get("type") == "tool_error":
            tool_errors[name] += 1

    def classify(counter: Counter[str]) -> dict[str, int]:
        return {
            "mutation": sum(
                count for name, count in counter.items()
                if name in mutation_tools
            ),
            "lifecycle": sum(
                count for name, count in counter.items()
                if name in lifecycle_tools
            ),
            "read_only": sum(
                count for name, count in counter.items()
                if name not in mutation_tools and name not in lifecycle_tools
            ),
        }

    actions_by_name: Counter[str] = Counter()
    for event in events:
        if event.get("type") != "action_submitted":
            continue
        action = event.get("action")
        if not isinstance(action, Mapping):
            continue
        name = str(action.get("action"))
        actions_by_name[name] += 1

    execution_status: Counter[str] = Counter()
    planner_backed_receipts = 0
    controller_sampled_steps = 0
    motion_receipts = 0
    for record in _receipt_iter(events):
        motion_receipts += 1
        status = record.get("execution_status")
        if isinstance(status, str) and status:
            execution_status[status] += 1
        planner_used, sampled = _planner_evidence(record)
        if planner_used:
            planner_backed_receipts += 1
        controller_sampled_steps += sampled

    policy_observations = sum(
        1 for event in events if event.get("type") == "observation"
    )

    explicit_actions = sum(actions_by_name.values())
    return {
        "schema": OWNERSHIP_AUDIT_SCHEMA,
        "surface_profile_id": (
            surface.get("profile_id") if surface is not None else None
        ),
        "model_calls": model_calls,
        "agent_action_proposals": proposed,
        "tool_calls": {
            "results_by_tool": dict(sorted(tool_results.items())),
            "errors_by_tool": dict(sorted(tool_errors.items())),
            "result_classes": classify(tool_results),
            "classification_source": (
                "plan_tool_surface_snapshot" if surface is not None
                else "unclassified_missing_surface_snapshot"
            ),
        },
        "explicit_robot_actions": {
            "total": explicit_actions,
            "by_action": dict(sorted(actions_by_name.items())),
        },
        "hidden_planner_controller_work": {
            "planner_backed_receipts": planner_backed_receipts,
            "controller_sampled_steps": controller_sampled_steps,
            "motion_probe_calls": tool_results.get("motion.probe", 0),
            # Counted across both spellings: a surface advertising the
            # canonical name would otherwise audit as zero grasp.plan calls.
            "grasp_plan_calls": sum(
                count for tool, count in tool_results.items()
                if tool_names.canonical(str(tool)) == "grasp_plan"
            ),
        },
        "reobservation_boundaries": {
            "policy_visible_observations": policy_observations,
        },
        "execution_status": dict(sorted(execution_status.items())),
        "motion_receipts": motion_receipts,
        "claim_boundary": (
            "counts describe action selection and delegation from append-only "
            "run evidence; a one-call macro is not by itself direct "
            "Language-Model action competence, and official success is only the "
            "Environment's verdict"
        ),
    }


def write_ownership_audit(run_root: Path) -> dict[str, Any]:
    """Build and atomically persist ``ownership-audit.json``."""

    value = build_ownership_audit(run_root)
    target = run_root.resolve() / "ownership-audit.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return value


__all__ = [
    "OWNERSHIP_AUDIT_SCHEMA",
    "OwnershipAuditError",
    "build_ownership_audit",
    "write_ownership_audit",
]
