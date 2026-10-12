"""Discover and classify run evidence without maintaining a second registry."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ora.orchestrator.control_audit import audit_libero_run
from ora.orchestrator.experiment_identity import load_catalog, project_identity


@lru_cache(maxsize=1)
def _identity_catalog() -> dict[str, Any]:
    """Tracked identity metadata is fixed for this Viewer source process."""
    return load_catalog()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _stable_run_id(value: str) -> bool:
    return bool(value) and all(character.isalnum() or character in "._-" for character in value)


def discover_run_roots(output_roots: Iterable[Path]) -> Iterator[Path]:
    """Yield unique run roots below outputs roots or experiment workspaces.

    Recursive discovery intentionally includes legacy ``worktree-snapshots``
    collections.  A run is still recognized only by a stable directory name
    and an exact ``plan.json`` below a ``runs`` directory.
    """
    found: set[Path] = set()
    for configured in output_roots:
        target = configured.expanduser().resolve()
        candidates: list[Path] = []
        if (target / "plan.json").is_file():
            candidates.append(target)
        if (target / "runs").is_dir():
            candidates.extend((target / "runs").iterdir())
        if target.is_dir():
            candidates.extend(
                plan.parent for plan in target.glob("**/runs/*/plan.json")
            )
        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                resolved in found
                or not resolved.is_relative_to(target)
                or not resolved.is_dir()
                or resolved.parent.name != "runs"
                or not (resolved / "plan.json").is_file()
                or not _stable_run_id(resolved.name)
            ):
                continue
            found.add(resolved)
    yield from sorted(found, key=lambda path: path.name, reverse=True)


def resolve_discovered_run(run_id: str, output_roots: Iterable[Path]) -> Path:
    if not _stable_run_id(run_id):
        raise RuntimeError("run-id must be a stable id")
    matches = [root for root in discover_run_roots(output_roots) if root.name == run_id]
    if len(matches) != 1:
        qualifier = "not found" if not matches else "ambiguous across configured output roots"
        raise RuntimeError(f"run-id {run_id} is {qualifier}")
    return matches[0]


def _state(status: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    value = status.get("state", result.get("state", "staged"))
    return str(value) if value is not None else "staged"


def _official_success(
    status: Mapping[str, Any], result: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> bool | None:
    for source in (result, status, evaluation):
        value = source.get("official_success")
        if isinstance(value, bool):
            return value
    return None


def _started_unix_s(root: Path, launch: Mapping[str, Any], result: Mapping[str, Any]) -> float | None:
    for source, field in (
        (result, "started_unix_s"),
        (launch, "created_unix_s"),
        (result, "finished_unix_s"),
    ):
        value = source.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    timestamp = root.name.rpartition("--")[2]
    try:
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        ).timestamp()
    except ValueError:
        return None


@dataclass(frozen=True)
class RunRegistration:
    root: Path
    run_id: str
    experiment: str
    config: str
    milestone: str
    task: str
    seed: int | str | None
    backend: str
    suite: str | None
    task_id: int | None
    task_key: str | None
    policy_kind: str
    model: str
    evidence_tier: str
    state: str
    evaluation: str
    sealed: bool
    official_success: bool | None
    control_validity: str
    control_issues: tuple[str, ...]
    session_cadence: str | None
    round_count: int | None
    round_success_count: int | None
    category: str
    synthetic: bool
    source_commit: str
    video_cached: bool | None
    started_unix_s: float | None

    experiment_identity: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value["root"] = str(self.root)
        return value


def register_run(root: Path, *, inspect_video: bool = True) -> RunRegistration:
    root = root.resolve()
    plan = _load(root / "plan.json")
    status = _load(root / "status.json")
    result = _load(root / "result.json")
    launch = _load(root / "launch.json")
    evaluation = _load(root / "evaluation.json")
    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    policy = plan.get("policy") if isinstance(plan.get("policy"), Mapping) else {}
    source = plan.get("source") if isinstance(plan.get("source"), Mapping) else {}
    task = str(plan.get("task") or "unknown")
    backend = str(runtime.get("kind") or "unknown")
    suite: str | None = None
    task_id: int | None = None
    if backend == "libero_pro":
        raw_suite = runtime.get("suite")
        raw_task_id = runtime.get("task_id")
        if (
            isinstance(raw_suite, str)
            and re.fullmatch(r"[a-z0-9_]+", raw_suite)
            and isinstance(raw_task_id, int)
            and not isinstance(raw_task_id, bool)
            and raw_task_id >= 0
        ):
            suite, task_id = raw_suite, raw_task_id
        else:
            legacy = re.fullmatch(r"([a-z0-9_]+)-t([0-9]+)", task)
            if legacy is not None:
                suite, task_id = legacy.group(1), int(legacy.group(2))
    task_key = (
        f"libero_pro:{suite}:t{task_id:02d}"
        if suite is not None and task_id is not None else None
    )
    synthetic = backend == "synthetic" or task.casefold().startswith("synthetic")
    evidence_tier = str(plan.get("evidence_tier") or "unknown")
    state = _state(status, result)
    evaluation_status = str(evaluation.get("status") or "missing")
    sealed = (root / "seal.json").is_file()
    session = result.get("session") if isinstance(result.get("session"), Mapping) else {}
    rounds = result.get("rounds") if isinstance(result.get("rounds"), list) else []
    round_count = result.get("round_count")
    if not isinstance(round_count, int) or isinstance(round_count, bool):
        round_count = len(rounds) if rounds else None
    round_success_count = result.get("round_success_count")
    if not isinstance(round_success_count, int) or isinstance(round_success_count, bool):
        round_success_count = (
            sum(
                isinstance(item, Mapping) and item.get("official_success") is True
                for item in rounds
            ) if rounds else None
        )
    if (
        evidence_tier == "behavioral"
        and not synthetic
        and state == "completed"
        and evaluation_status == "valid"
        and sealed
    ):
        category = "formal"
    elif evidence_tier == "behavioral" and not synthetic:
        category = "attempt"
    elif evidence_tier == "infrastructure" and not synthetic:
        category = "qualification"
    else:
        category = "diagnostic"
    cached = False if inspect_video else None
    if inspect_video and category == "formal":
        from viewer.trajectory import trajectory_video_status

        try:
            cached = bool(trajectory_video_status(root)["cached"])
        except (OSError, RuntimeError, ValueError):
            cached = False
    control_audit = audit_libero_run(root)
    return RunRegistration(
        root=root,
        run_id=root.name,
        experiment_identity=project_identity(plan, _identity_catalog()),
        experiment=str(plan.get("experiment") or root.parent.parent.name),
        config=str(plan.get("config") or "—"),
        milestone=str(plan.get("milestone") or "—"),
        task=task,
        seed=plan.get("seed"),
        backend=backend,
        suite=suite,
        task_id=task_id,
        task_key=task_key,
        policy_kind=str(policy.get("kind") or "unknown"),
        model=str(policy.get("model") or "—"),
        evidence_tier=evidence_tier,
        state=state,
        evaluation=evaluation_status,
        sealed=sealed,
        official_success=_official_success(status, result, evaluation),
        control_validity=str(control_audit["control_validity"]),
        control_issues=tuple(
            str(item) for item in control_audit.get("issues", [])
        ),
        session_cadence=(str(session.get("cadence")) if session.get("cadence") else None),
        round_count=round_count,
        round_success_count=round_success_count,
        category=category,
        synthetic=synthetic,
        source_commit=str(source.get("commit") or "unknown"),
        video_cached=cached,
        started_unix_s=_started_unix_s(root, launch, result),
    )


# Registration re-reads and re-hashes run evidence, which is far too slow to
# repeat for every page load over sealed, immutable runs.  The fingerprint
# tracks every evidence file registration reads plus the derived-video
# directory listing, so compiling a video or (exceptionally) touching a run
# invalidates exactly that entry.  Worker/source-code identity changes are
# covered by process restart.
_REGISTRATION_FINGERPRINT_NAMES = (
    "plan.json", "status.json", "result.json", "launch.json",
    "evaluation.json", "seal.json", "events.jsonl",
)
_REGISTRATION_CACHE: dict[tuple[Path, bool], tuple[tuple[Any, ...], RunRegistration]] = {}


def _registration_fingerprint(root: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    for name in _REGISTRATION_FINGERPRINT_NAMES:
        try:
            stat = (root / name).stat()
            parts.append((name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            parts.append((name, None, None))
    workspace = root.parent.parent if root.parent.name == "runs" else root.parent
    derived = workspace / "viewer-derived" / root.name
    try:
        with os.scandir(derived) as entries:
            listing = sorted(
                (entry.name, entry.stat(follow_symlinks=False).st_mtime_ns)
                for entry in entries
            )
    except OSError:
        listing = []
    parts.append(tuple(listing))
    return tuple(parts)


def registry(roots: Iterable[Path], *, inspect_video: bool = True) -> list[RunRegistration]:
    """Inspect video availability only on surfaces that display that status.

    Result matrices need round evidence, not a full projection of every model
    trace. An omitted video inspection is represented by None, never by a
    fabricated cache miss, and cannot poison a later complete registration.
    """
    runs: list[RunRegistration] = []
    for root in roots:
        root = root.resolve()
        fingerprint = _registration_fingerprint(root)
        key = (root, inspect_video)
        cached = _REGISTRATION_CACHE.get(key)
        if cached is not None and cached[0] == fingerprint:
            runs.append(cached[1])
            continue
        run = register_run(root, inspect_video=inspect_video)
        _REGISTRATION_CACHE[key] = (fingerprint, run)
        runs.append(run)
    return sorted(
        runs,
        key=lambda run: (
            run.started_unix_s is not None,
            run.started_unix_s or 0,
            run.run_id,
        ),
        reverse=True,
    )


def visible_runs(
    registrations: Iterable[RunRegistration], *, view: str, backend: str = "",
    experiment: str = "", task_key: str = "", seed: str = "",
    state: str = "", outcome: str = "",
) -> list[RunRegistration]:
    allowed = {
        "formal": {"formal"},
        "attempts": {"formal", "attempt"},
        "qualifications": {"qualification"},
        "diagnostics": {"diagnostic"},
        "all": {"formal", "attempt", "qualification", "diagnostic"},
    }
    categories = allowed.get(view, allowed["formal"])
    return [
        run for run in registrations
        if run.category in categories
        and (not backend or run.backend == backend)
        and (not experiment or run.experiment == experiment)
        and (not task_key or run.task_key == task_key)
        and (not seed or str(run.seed) == seed)
        and (not state or run.state == state)
        and (
            not outcome
            or (outcome == "success" and run.official_success is True)
            or (outcome == "failure" and run.official_success is False)
            or (outcome == "unscored" and run.official_success is None)
        )
    ]
