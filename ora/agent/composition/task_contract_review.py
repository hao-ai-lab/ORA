"""Offline review records and deterministic public contract rendering.

This checks review completeness and integrity, never proves natural-language
equivalence to a predicate. Private source and witness records stay off-policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

REVIEW_SCHEMA = "ora.task-contract-review.v1"
CHECKS = (
    "predicate_and_helpers", "reset_and_randomization", "assets_and_frames",
    "success_timing", "public_observability", "success_witness",
    "counterexamples", "no_expert_strategy", "privacy",
)


def render_public_contract(record: Mapping[str, Any]) -> str:
    clauses = record.get("public_clauses", [])
    if not isinstance(clauses, list) or not clauses or any(
        not isinstance(item, str) or not item.strip() or "\n" in item for item in clauses
    ):
        raise ValueError("public_clauses must contain nonempty single-line strings")
    return "Public completion contract for this task:\n\n" + "".join(
        f"- {item}\n" for item in clauses
    )


def review_errors(record: Mapping[str, Any]) -> list[str]:
    errors = []
    if record.get("schema") != REVIEW_SCHEMA:
        errors.append("review_schema_mismatch")
    if record.get("status") != "reviewed":
        errors.append("semantic_review_incomplete")
    for field in ("backend", "task", "reviewer", "reviewed_at", "scope", "limitations"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"missing_{field}")
    checks = record.get("checks", {})
    for check in CHECKS:
        item = checks.get(check) if isinstance(checks, Mapping) else None
        if not isinstance(item, Mapping) or item.get("status") != "passed" or not item.get("basis"):
            errors.append(f"unchecked_{check}")
    for group in ("source_dependencies", "evidence"):
        items = record.get(group)
        if not isinstance(items, list) or not items:
            errors.append(f"missing_{group}")
            continue
        for item in items:
            if not isinstance(item, Mapping) or not item.get("role") or not item.get("path") or not re.fullmatch(
                r"[0-9a-f]{64}", str(item.get("sha256", ""))
            ):
                errors.append(f"invalid_{group}_reference")
    try:
        text = render_public_contract(record)
    except ValueError as exc:
        errors.append(str(exc))
        return errors
    if re.search(r"\d|\{\{|\}\}|\b(?:quaternion|qpos|check_success|eval_success|epsilon|threshold|tolerance)\b", text, re.I):
        errors.append("public_text_contains_private_detail_or_placeholder")
    for claim in record.get("forbidden_public_claims", []):
        if isinstance(claim, str) and claim.lower() in text.lower():
            errors.append(f"contradictory_public_claim:{claim}")
    conditions = record.get("conditions")
    covered = set()
    if not isinstance(conditions, list) or not conditions:
        errors.append("missing_condition_mapping")
    else:
        identifiers = set()
        for item in conditions:
            if not isinstance(item, Mapping):
                errors.append("invalid_condition_mapping")
                continue
            identifier = item.get("id")
            if not isinstance(identifier, str) or not identifier or identifier in identifiers:
                errors.append("missing_or_duplicate_condition_id")
            identifiers.add(str(identifier))
            if not item.get("source_basis") or not item.get("interpretation"):
                errors.append("condition_lacks_source_or_interpretation")
            if item.get("disposition") not in {"included", "task_intent_only", "withheld_private", "unobservable"}:
                errors.append("invalid_condition_disposition")
            indices = item.get("public_clause_indices", [])
            if not isinstance(indices, list):
                errors.append("invalid_clause_indices")
                continue
            for index in indices:
                if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(record["public_clauses"]):
                    errors.append("invalid_clause_index")
                else:
                    covered.add(index)
            if not indices and not item.get("omission_reason"):
                errors.append("condition_omission_not_explained")
    if covered != set(range(len(record["public_clauses"]))):
        errors.append("public_clause_without_source_mapping")
    if not record.get("counterexamples"):
        errors.append("missing_counterexamples")
    return errors


def load_review_record(root: Path, reference: Any) -> tuple[dict[str, Any], str]:
    """Load a registry's semantic review by its path; return it and the sha256 of its bytes.

    The registry names the review by path only; ``policy.lock`` carries its
    integrity. The task's public layer text keeps its reviewed hash.
    """
    from .locked_assets import locked_asset

    asset = locked_asset(Path(root), reference, label="semantic review", error=ValueError)
    value = json.loads(asset.data)
    if not isinstance(value, dict):
        raise ValueError("semantic review must be an object")
    return value, asset.sha256


def load_review(root: Path, reference: Any) -> dict[str, Any]:
    return load_review_record(root, reference)[0]


def audit_registry(root: Path, backend: str, *, source_root: Path | None = None,
                   evidence_root: Path | None = None) -> dict[str, Any]:
    from .task_contract import REGISTRIES, REGISTRY_SCHEMA
    relative = REGISTRIES.get(backend)
    if relative is None:
        return {"backend": backend, "status": "no_registry", "tasks": [], "counts": {}}
    registry = json.loads((root / relative).read_text())
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise ValueError("registry schema mismatch")
    tasks = registry["tasks"]
    roster = set(tasks)
    historical = registry.get("contract_source")
    if historical:
        roster.update(p.stem for p in (root / historical).glob("*.json"))
    rows = []
    for task in sorted(roster):
        entry = tasks.get(task)
        row: dict[str, Any] = {"task": task, "errors": []}
        rows.append(row)
        if entry is None:
            row["status"] = "not_registered"
            continue
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            row.update(status="invalid", errors=["missing_or_unsafe_contract_path"])
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            row["errors"].append("contract_hash_mismatch")
        reference = entry.get("semantic_review")
        if reference is None:
            row["status"] = "invalid" if row["errors"] else "needs_semantic_review"
            continue
        try:
            record = load_review(root, reference)
            row["errors"].extend(review_errors(record))
            if record.get("task") != task or record.get("backend") != backend:
                row["errors"].append("review_identity_mismatch")
            if render_public_contract(record) != path.read_text():
                row["errors"].append("public_projection_differs_from_review")
            if record.get("verified_revisions") != entry.get("verified_revisions"):
                row["errors"].append("review_revision_scope_mismatch")
            for group, base in (("source_dependencies", source_root), ("evidence", evidence_root)):
                if base is None:
                    continue
                for item in record.get(group, []):
                    relative_path = Path(item["path"])
                    target = base / relative_path
                    # Asset trees legitimately symlink a shared read-only
                    # cache. Bind bytes, allow those links, reject traversal.
                    if relative_path.is_absolute() or ".." in relative_path.parts or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                        row["errors"].append(f"{group}_hash_mismatch:{item['role']}")
            row["status"] = "invalid" if row["errors"] else "review_record_valid"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status="invalid", errors=row["errors"] + [str(exc)])
    return {
        "schema": "ora.task-contract-audit.v1", "backend": backend,
        "registry": relative, "tasks": rows,
        "counts": dict(Counter(row["status"] for row in rows)),
        "source_artifacts_checked": source_root is not None,
        "witness_artifacts_checked": evidence_root is not None,
        "claim": "review_completeness_and_integrity_not_automated_semantic_proof",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "template", "render"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--backend", default="robotwin")
    parser.add_argument("--task", default="")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "template":
        record = {"schema": REVIEW_SCHEMA, "backend": args.backend, "task": args.task,
                  "status": "needs_review", "reviewer": "", "reviewed_at": "", "scope": "",
                  "verified_revisions": [], "source_dependencies": [], "evidence": [],
                  "conditions": [], "public_clauses": [], "forbidden_public_claims": [],
                  "checks": {key: {"status": "pending", "basis": ""} for key in CHECKS},
                  "counterexamples": [], "limitations": ""}
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    if args.command == "render":
        if args.record is None:
            parser.error("render requires --record")
        record = json.loads(args.record.read_text())
        errors = review_errors(record)
        if errors:
            print(json.dumps({"errors": errors}, ensure_ascii=False))
            return 1
        print(render_public_contract(record), end="")
        return 0
    report = audit_registry(args.root.resolve(), args.backend,
                            source_root=args.source_root, evidence_root=args.evidence_root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report.get("status") == "no_registry":
        return 1 if args.require_reviewed else 0
    return int(any(row["status"] == "invalid" or (
        args.require_reviewed and row["status"] != "review_record_valid"
    ) for row in report["tasks"]))
