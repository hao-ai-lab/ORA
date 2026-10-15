"""Read-only RoboDojo native controller journals as immutable tick and segment records.

A sealed RoboDojo run keeps, per policy action, ``receipts/action-NNNN.json``
and two controller files: ``controller/action-NNNN.json`` (one sample per
native tick: ``native_action``, ``segment``, the recorded ``control`` dict and
the tick-end ``measured`` state) and ``controller/action-NNNN.jsonl`` (one line
per physics step with the applied targets and the measured state after that
step; ten per tick). This module joins them without Isaac and without writing
anything. A control replay writes the same two files under its own root; pass
that root as ``journal_root`` to read the replayed motion against the original
receipts.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ora.contracts.robodojo_arx import ARX_X5_CONTRACT
from ora.environment.benchmarks.robodojo import NATIVE_ACTION_LIMITS

#: ``session.CONTROL_TRACE_SCHEMA``; repeated so this host-side reader imports no worker module.
TRACE_SCHEMA = "ora.robodojo-controller-trace.v1"
ARMS = ("left", "right")
PHYSICS_STEPS_PER_TICK = int(ARX_X5_CONTRACT["physics_steps_per_native_action"])
#: Segment names the v1 executor writes (``sparse_motion``, ``staged_contact``,
#: ``recovery``, ``wait``, ``motion_stall``). Every ``recover*`` name is a
#: recovery joint segment. ``plan_ref`` is a probe-planned global move (none in
#: the recorded batch, so its global population is move/precontact/recover/retract).
GLOBAL_SEGMENTS = frozenset({"move", "plan_ref", "precontact", "retract"})
LOCAL_SEGMENTS = frozenset({"local"})


class JournalError(RuntimeError):
    """A native root is incomplete or its controller files disagree."""


@dataclass(frozen=True)
class Tick:
    """One native action: its recorded control and ten physics-step lines."""

    action_sequence: int
    round_index: int
    index: int
    segment: str
    control: Mapping[str, Any]
    measured: Mapping[str, Any]
    steps: tuple[Mapping[str, Any], ...]
    predecessor: Mapping[str, Any] | None


@dataclass(frozen=True)
class Action:
    sequence: int
    round_index: int
    receipt: Mapping[str, Any]
    ticks: tuple[Tick, ...]


@dataclass(frozen=True)
class RunJournal:
    run_id: str
    native_root: Path
    journal_root: Path
    task: str | None
    native_action_limit: int | None
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class SegmentRun:
    """Consecutive ticks of one action and one segment name commanding one arm.

    ``target`` is the arm's joint target on the first tick; ``constant_target``
    says whether every tick kept it (v1 endpoint interpolation does; a planned
    trajectory or a stall hold does not). ``entry`` is the receipt's segment
    record for this run, matched in order by name, or ``None``.
    """

    action: Action
    arm: str
    segment: str
    target: tuple[float, ...]
    constant_target: bool
    ticks: tuple[Tick, ...]
    entry: Mapping[str, Any] | None


def native_root_of(path: Path) -> Path:
    """Accept a run directory (with ``native/``) or a native root."""
    path = Path(path).resolve()
    native = path / "native"
    if (native / "receipts").is_dir():
        return native
    if (path / "receipts").is_dir():
        return path
    raise JournalError(f"{path} is neither a run directory nor a RoboDojo native root")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise JournalError(f"cannot read {path}: {exc}") from exc


def journal_lines(path: Path) -> list[dict[str, Any]]:
    """Physics-step lines; a truncated or half-written line is a ``JournalError``."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise JournalError(f"cannot read {path}: {exc}") from exc


def _sequences(journal_root: Path, upto: int | None, *, strict: bool) -> list[int]:
    """Action sequences from 0; a replay may stop early, an original may not have a gap."""
    found = sorted(int(path.stem.split("-")[1]) for path in (journal_root / "controller").glob("action-*.json"))
    contiguous = [value for index, value in enumerate(found) if value == index]
    if strict and len(contiguous) != len(found):
        raise JournalError(f"{journal_root}/controller traces are not contiguous from action-0000")
    return contiguous if upto is None else contiguous[:upto]


def _check_samples(sequence: int, samples: Sequence[Mapping[str, Any]], lines: Sequence[Mapping[str, Any]],
                   last_native_action: int) -> None:
    if len(lines) != PHYSICS_STEPS_PER_TICK * len(samples):
        raise JournalError(
            f"controller/action-{sequence:04d}: {len(lines)} journal lines for {len(samples)} ticks; "
            f"expected {PHYSICS_STEPS_PER_TICK} physics lines per tick")
    numbers = [int(sample.get("native_action", -1)) for sample in samples]
    if numbers != list(range(last_native_action + 1, last_native_action + 1 + len(numbers))):
        raise JournalError(f"controller/action-{sequence:04d}: native actions {numbers[:3]}... do not continue "
                           f"the round's count {last_native_action}")


def _ticks(sequence: int, round_index: int, trace: Mapping[str, Any], lines: Sequence[Mapping[str, Any]],
           previous: Mapping[str, Any] | None) -> tuple[Tick, ...]:
    ticks = []
    for index, sample in enumerate(trace.get("samples", [])):
        steps = tuple(lines[PHYSICS_STEPS_PER_TICK * index: PHYSICS_STEPS_PER_TICK * (index + 1)])
        contiguous = previous is not None and previous.get("physics_clock") == steps[0].get("physics_clock", 0) - 1
        ticks.append(Tick(sequence, round_index, index, str(sample["segment"]), sample.get("control", {}),
                          sample.get("measured", {}), steps, previous if contiguous else None))
        previous = steps[-1]
    return tuple(ticks)


def _task(native_root: Path) -> tuple[str | None, int | None]:
    reset = native_root / "reset.json"
    task = _load_json(reset).get("task") if reset.is_file() else None
    return task, NATIVE_ACTION_LIMITS.get(task) if isinstance(task, str) else None


def _action(native: Path, root: Path, sequence: int) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    receipt = _load_json(native / "receipts" / f"action-{sequence:04d}.json")
    trace = _load_json(root / "controller" / f"action-{sequence:04d}.json")
    if trace.get("schema") != TRACE_SCHEMA:
        raise JournalError(f"controller/action-{sequence:04d}.json is not {TRACE_SCHEMA}")
    if trace.get("action_id") != receipt.get("action_id"):
        raise JournalError(f"controller/action-{sequence:04d}.json action_id differs from its receipt")
    return receipt, trace, journal_lines(root / "controller" / f"action-{sequence:04d}.jsonl")


def load_run_journal(run: Path, *, journal_root: Path | None = None, upto: int | None = None) -> RunJournal:
    """Join receipts with controller traces; ``upto`` keeps the first N actions.

    With ``journal_root`` the traces come from that root (a replay) and stop
    at the first action it does not contain; receipts always come from the
    original native root. Trace and receipt must name the same action, and
    native actions must count up without a gap inside each round.
    """
    native = native_root_of(run)
    root = Path(journal_root).resolve() if journal_root is not None else native
    actions, previous, previous_round, counted = [], None, None, 0
    for sequence in _sequences(root, upto, strict=journal_root is None):
        receipt, trace, lines = _action(native, root, sequence)
        round_index = int(receipt.get("round_index", 0))
        if round_index != previous_round:
            previous, previous_round, counted = None, round_index, 0
        _check_samples(sequence, trace.get("samples", []), lines, counted)
        ticks = _ticks(sequence, round_index, trace, lines, previous)
        previous = ticks[-1].steps[-1] if ticks else previous
        counted += len(ticks)
        actions.append(Action(sequence, round_index, receipt, ticks))
    task, limit = _task(native)
    run_id = native.parent.name if native.name == "native" else native.name
    return RunJournal(run_id, native, root, task, limit, tuple(actions))


def commanded_arms(tick: Tick) -> tuple[str, ...]:
    return tuple(arm for arm in ARMS if f"{arm}_arm_joint_state" in tick.control)


def arm_target(tick: Tick, arm: str) -> tuple[float, ...] | None:
    value = tick.control.get(f"{arm}_arm_joint_state")
    return None if value is None else tuple(float(item) for item in value["position"])


def _path(tick: Tick, arm: str, pick: Any) -> list[list[float]]:
    head = [tick.predecessor] if tick.predecessor is not None else []
    return [[float(v) for v in pick(line, arm)] for line in [*head, *tick.steps]]


def joint_path(tick: Tick, arm: str) -> list[list[float]]:
    """Joint vectors: the predecessor physics step (when known) then the tick's ten steps."""
    return _path(tick, arm, lambda line, a: line["measured"][a]["arm_joint_qpos"])


def position_path(tick: Tick, arm: str) -> list[list[float]]:
    return _path(tick, arm, lambda line, a: line["measured"][a]["pose_xyz_xyzw"][:3])


def applied_steps(tick: Tick, arm: str) -> list[list[float]] | None:
    """Applied arm-joint targets of the tick's own ten physics steps, never the predecessor.

    The step from the previous tick's last target is excluded: v1's ``interpolate`` holds the
    endpoint at the end of a tick and restarts from the measured joints, so that step is the
    restart, not a commanded motion. ``None`` when a step did not record the arm's target.
    """
    key = f"{arm}_arm_joint_state"
    if any(key not in line.get("applied_position_targets", {}) for line in tick.steps):
        return None
    return [[float(v) for v in line["applied_position_targets"][key]] for line in tick.steps]


def joint_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Largest single-joint change; the distance the stall rule uses."""
    return max(abs(float(a) - float(b)) for a, b in zip(first, second, strict=True))


def segment_class(name: str) -> str | None:
    if name in GLOBAL_SEGMENTS or name.startswith("recover"):
        return "global"
    if name in LOCAL_SEGMENTS:
        return "local"
    return None


def _split(ticks: Iterable[Tick], key: Any) -> list[tuple[Any, tuple[Tick, ...]]]:
    groups: list[tuple[Any, list[Tick]]] = []
    for item in ticks:
        value = key(item)
        if groups and groups[-1][0] == value:
            groups[-1][1].append(item)
        else:
            groups.append((value, [item]))
    return [(value, tuple(items)) for value, items in groups]


def _arm_runs(item: Action, arm: str) -> list[SegmentRun]:
    commanded = [entry for entry in item.ticks if arm_target(entry, arm) is not None]
    entries = [entry for entry in item.receipt.get("arms", {}).get(arm, {}).get("segments", [])
               if int(entry.get("native_actions", 0) or 0) > 0]
    runs, cursor = [], 0
    for segment, ticks in _split(commanded, lambda t: t.segment):
        entry = entries[cursor] if cursor < len(entries) and entries[cursor].get("segment") == segment else None
        cursor += entry is not None
        targets = {arm_target(t, arm) for t in ticks}
        runs.append(SegmentRun(item, arm, segment, arm_target(ticks[0], arm), len(targets) == 1, ticks, entry))
    return runs


def segment_runs(journal: RunJournal) -> tuple[SegmentRun, ...]:
    """Per-arm segments: consecutive ticks of one action with one segment name for that arm."""
    return tuple(run for item in journal.actions for arm in ARMS for run in _arm_runs(item, arm))


def command_segments(action: Action) -> list[tuple[str, int]]:
    """Arm-independent segments (a synchronized pair counts once) and their tick counts."""
    return [(segment, len(ticks)) for segment, ticks in _split(action.ticks, lambda t: t.segment)]


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    """Count, mean and the pinned percentiles: the value at index floor(q * n) of the sorted list."""
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}

    def rank(q: float) -> float:
        return ordered[min(len(ordered) - 1, math.floor(q * len(ordered)))]

    return {"count": len(ordered), "mean": sum(ordered) / len(ordered), "p50": rank(0.5), "p90": rank(0.9),
            "max": ordered[-1]}


__all__ = [
    "ARMS", "Action", "GLOBAL_SEGMENTS", "JournalError", "LOCAL_SEGMENTS", "PHYSICS_STEPS_PER_TICK", "RunJournal",
    "SegmentRun", "TRACE_SCHEMA", "Tick", "applied_steps", "arm_target", "command_segments", "commanded_arms",
    "joint_distance", "joint_path", "journal_lines", "load_run_journal", "native_root_of", "position_path",
    "segment_class", "segment_runs", "summarize_values",
]
