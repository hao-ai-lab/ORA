"""Independent historical evaluation reports; never rewrite or reseal source runs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from .evaluation import evaluate
from .evidence_io import EvaluationError, _hash
from .paths import PROJECT_ROOT


def _snapshot(root: Path) -> dict[str, Any]:
    """Bind source bytes, including original seal/evaluation and referenced media."""
    return {
        path.relative_to(root).as_posix(): (
            {"symlink": str(path.readlink())} if path.is_symlink() else
            {"bytes": path.stat().st_size, "sha256": _hash(path)}
        )
        for path in sorted(root.rglob("*")) if path.is_file() or path.is_symlink()
    }


def _checker_identity() -> dict[str, Any]:
    package = Path(__file__).resolve().parents[1]
    source_files = {
        path.relative_to(package.parent).as_posix(): _hash(path)
        for path in sorted(package.rglob("*.py"))
    }
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), *args], capture_output=True, text=True,
            check=True,
        )
        return result.stdout.strip()
    return {
        "commit": git("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": sys.version,
        "python_sources_sha256": hashlib.sha256(json.dumps(
            source_files, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "python_sources": source_files,
        "mode": "backfill_read_only_preserve_existing_derivations",
    }


def _read_optional(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _revalidate_one(
    root: Path, output: Path, checker_source_sha256: str,
) -> dict[str, Any]:
    original = _read_optional(root / "evaluation.json")
    result = _read_optional(root / "result.json")
    report: dict[str, Any] = {
        "schema": "ora.historical-revalidation-run.v1",
        "run_root": str(root), "run_id": root.name,
        "original_evaluation": original, "original_seal": _read_optional(root / "seal.json"),
        "checker_source_sha256": checker_source_sha256,
    }
    if not isinstance(result, dict) or result.get("state") in {None, "running", "starting", "pending"}:
        report.update(status="deferred_unfinished_prefix", new_evaluation=None)
    else:
        before = _snapshot(root)
        try:
            new = evaluate(root, identity="backfill", persist=False)
        except (EvaluationError, OSError, ValueError) as exc:
            report.update(status="revalidation_error", error=str(exc), new_evaluation=None)
        else:
            report.update(status="reviewed", new_evaluation=new)
            old_reasons = set((original or {}).get("reasons", []))
            new_reasons = set(new.get("reasons", []))
            report["difference"] = {
                "original_status": (original or {}).get("status"),
                "new_status": new.get("status"),
                "removed_reasons": sorted(old_reasons - new_reasons),
                "added_reasons": sorted(new_reasons - old_reasons),
                "changed_fields": sorted(key for key in set(original or {}) | set(new)
                                         if (original or {}).get(key) != new.get(key)),
            }
        after = _snapshot(root)
        report["original_files"] = before
        report["source_unchanged"] = before == after
        if before != after:
            report["status"] = "source_changed_during_revalidation"
            report["changed_source_files"] = sorted(
                key for key in set(before) | set(after) if before.get(key) != after.get(key)
            )
    name = root.name + "-" + hashlib.sha256(str(root).encode()).hexdigest()[:12] + ".json"
    (output / name).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    row = {"run_root": str(root), "status": report["status"], "report": name,
           "difference": report.get("difference"),
           "source_unchanged": report.get("source_unchanged")}
    return row


def revalidate_runs(
    run_roots: Sequence[Path], output_dir: Path, *, workers: int = 2,
) -> dict[str, Any]:
    """Write a separate report set for an exact list of completed source runs.

    Active prefixes are reported as deferred. Reports are neither evaluations
    attached to a run nor authority to update its seal or learning eligibility.
    """
    roots = list(dict.fromkeys(root.resolve() for root in run_roots))
    output = output_dir.resolve()
    if not roots or workers < 1:
        raise EvaluationError("revalidation requires run directories and positive workers")
    if any(not root.is_dir() for root in roots):
        raise EvaluationError("revalidation source directory does not exist")
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise EvaluationError("revalidation output must be separate from source runs")
    if any(ancestor.name == "runs" or (ancestor / "plan.json").is_file()
           or (ancestor / "result.json").is_file() for ancestor in (output, *output.parents)):
        raise EvaluationError("revalidation output cannot be a formal run directory")
    if output.exists() and any(output.iterdir()):
        raise EvaluationError("revalidation output directory must be empty")
    checker = _checker_identity()
    output.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=min(workers, len(roots))) as pool:
        reports = list(pool.map(
            _revalidate_one, roots, repeat(output), repeat(checker["python_sources_sha256"]),
        ))
    manifest = {
        "schema": "ora.historical-revalidation.v1",
        "checker": checker, "workers": min(workers, len(roots)), "runs": reports,
        "claim_boundary": "Independent review only; original evaluations, seals and identities remain authoritative historical records.",
    }
    (output / "revalidation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
