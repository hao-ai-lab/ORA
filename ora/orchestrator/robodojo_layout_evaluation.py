"""Per-round layout identity of a persistent RoboDojo session, evaluation side.

An unscheduled RunPlan (no ``runtime.layout_schedule``; every g003-g007 cell)
replays layout 0 in every round: one identity for all rounds, ``reset.v1`` /
``round-reset.v1`` / ``official-verdict.v2`` records, an in-place reset whose
isolation proof includes ``layout_identity_unchanged``, and a PhysX clock that
never goes back across rounds. Those joins are unchanged.

A scheduled RunPlan starts every round as a new episode on its planned layout
inside a relaunched simulation. Each round then has its own identity
(layout id, file, bytes, ``env_seed`` equal to the layout id and the physics
clock epoch, which a relaunch advances by one), the records are ``reset.v2`` /
``round-reset.v2`` / ``official-verdict.v3``, the isolation proof replaces the
layout identity check with the scheduled-layout, relaunch, plant and scene
checks, and the clock is monotone only within one epoch.

This module is host-side and imports no backend; the worker's names are
repeated here and ``tests/test_robodojo_layout_schedule.py`` holds them equal.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

UNSCHEDULED_START_RESET_SCHEMA = "ora.robodojo-reset.v1"
UNSCHEDULED_ROUND_RESET_SCHEMA = "ora.robodojo-round-reset.v1"
UNSCHEDULED_VERDICT_SCHEMA = "ora.robodojo-official-verdict.v2"
SCHEDULED_START_RESET_SCHEMA = "ora.robodojo-reset.v2"
SCHEDULED_ROUND_RESET_SCHEMA = "ora.robodojo-round-reset.v2"
SCHEDULED_VERDICT_SCHEMA = "ora.robodojo-official-verdict.v3"
RELAUNCH_SCHEMA = "ora.robodojo-simulation-relaunch.v1"
LAYOUT_VERIFICATION_SCHEMA = "ora.robodojo-layout-verification.v1"
#: The facts round 0's layout verification and every scheduled round measure.
LOADED_LAYOUT_CHECKS = ("layout_matches_round_schedule", "scene_objects_match_layout")
SCENE_RESET_INITIAL = "initial_launch"
SCENE_RESET_RELAUNCH = "simulation_relaunch"
SCHEDULED_ISOLATION_CHECKS = (
    "arm_home_restored",
    "carried_channels_cleared",
    "episode_counters_zeroed",
    "failure_latch_cleared",
    "layout_matches_round_schedule",
    "official_lifecycle_rebound",
    "official_terminal_cleared",
    "plant_embodiment_unchanged",
    "scene_objects_match_layout",
    "simulation_relaunched",
    "task_instruction_unchanged",
)
#: The fields a scheduled round's result carries under ``scene``.
ROUND_SCENE_FIELDS = (
    "layout_id", "layout_file", "layout_sha256", "env_seed", "physics_clock_epoch", "scene_reset_mode",
)


def scheduled_rounds(plan: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The plan's per-round layouts, or ``None`` for the unscheduled treatment."""

    schedule = (plan.get("runtime") or {}).get("layout_schedule")
    if not isinstance(schedule, Mapping):
        return None
    rounds = schedule.get("rounds")
    return [dict(entry) for entry in rounds] if isinstance(rounds, list) else []


def round_identities(plan: Mapping[str, Any], rounds: int) -> list[dict[str, Any]]:
    """The identity every record of each round must repeat."""

    runtime = plan.get("runtime") or {}
    base = {"task": plan.get("task"), "eval_seed": runtime.get("eval_seed", plan.get("seed"))}
    schedule = scheduled_rounds(plan)
    if schedule is None:
        return [{**base, "layout_id": runtime.get("layout_id")} for _ in range(rounds)]
    return [
        {**base, "layout_id": entry.get("layout_id"), "layout_file": entry.get("layout_file"),
         "layout_sha256": entry.get("layout_sha256"), "env_seed": entry.get("layout_id"),
         "physics_clock_epoch": index}
        for index, entry in enumerate(schedule)
    ] + [dict(base, layout_id=None)] * max(0, rounds - len(schedule))


def record_schemas(plan: Mapping[str, Any]) -> tuple[str, str, str]:
    """(start reset, round reset, official verdict) schemas this plan's worker writes."""

    if scheduled_rounds(plan) is None:
        return UNSCHEDULED_START_RESET_SCHEMA, UNSCHEDULED_ROUND_RESET_SCHEMA, UNSCHEDULED_VERDICT_SCHEMA
    return SCHEDULED_START_RESET_SCHEMA, SCHEDULED_ROUND_RESET_SCHEMA, SCHEDULED_VERDICT_SCHEMA


def isolation_check_names(plan: Mapping[str, Any], unscheduled: Sequence[str]) -> tuple[str, ...]:
    return tuple(unscheduled) if scheduled_rounds(plan) is None else SCHEDULED_ISOLATION_CHECKS


def isolation_layout_errors(
    plan: Mapping[str, Any], measured: Mapping[str, Any], index: int,
) -> bool:
    """Whether the isolation proof measured a layout other than the round's own."""

    runtime = plan.get("runtime") or {}
    schedule = scheduled_rounds(plan)
    if schedule is None:
        return (measured.get("layout_sha256") != runtime.get("layout_sha256")
                or measured.get("layout_sha256_at_start") != runtime.get("layout_sha256"))
    entry = schedule[index] if index < len(schedule) else {}
    return (measured.get("layout_id") != entry.get("layout_id")
            or measured.get("layout_sha256") != entry.get("layout_sha256")
            or measured.get("layout_sha256_planned") != entry.get("layout_sha256")
            or measured.get("env_seed") != entry.get("layout_id")
            or measured.get("physics_clock_epoch") != index)


def _round0_verified(record: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    verification = record.get("layout_verification")
    if not isinstance(verification, Mapping) or verification.get("schema") != LAYOUT_VERIFICATION_SCHEMA:
        return False
    checks = verification.get("checks") or {}
    measured = verification.get("measured") or {}
    return (verification.get("passed") is True and all(checks.get(name) is True for name in LOADED_LAYOUT_CHECKS)
            and measured.get("layout_id") == entry.get("layout_id")
            and measured.get("layout_sha256") == measured.get("layout_sha256_planned") == entry.get("layout_sha256")
            and measured.get("env_seed") == entry.get("layout_id"))


def _offset(value: Any) -> bool:
    return type(value) is float and math.isfinite(value) and value >= 0


def _relaunch_proven(
    record: Mapping[str, Any], previous: Mapping[str, Any], previous_record: Mapping[str, Any] | None,
) -> bool:
    """The worker's relaunch proof, derived again from its measured stage, clock and time fields.

    The new epoch's camera-annotator offset is the previous epoch's offset
    plus the closed context's simulated time at close; the reset record and
    the relaunch must name that same offset.
    """

    relaunch = record.get("relaunch")
    if (not isinstance(relaunch, Mapping) or relaunch.get("schema") != RELAUNCH_SCHEMA
            or not isinstance(previous_record, Mapping)):
        return False
    before, after = relaunch.get("stage_before") or {}, relaunch.get("stage_after_close") or {}
    times = [relaunch.get(name) for name in (
        "simulation_time_before_close_s", "previous_simulation_time_offset_s", "simulation_time_offset_s")]
    return (relaunch.get("layout_id") == record.get("layout_id")
            and relaunch.get("previous_layout_id") == previous.get("layout_id")
            and before.get("env_prim_present") is True and after.get("env_prim_present") is False
            and relaunch.get("sim_closed") is True and relaunch.get("stage_replaced") is True
            and type(relaunch.get("physics_clock_after_reset")) is int
            and relaunch["physics_clock_after_reset"] == record.get("physics_steps_before_policy")
            and all(_offset(value) for value in times)
            and times[1] == previous_record.get("simulation_time_offset_s")
            and times[2] == record.get("simulation_time_offset_s")
            and abs(times[2] - (times[1] + times[0])) <= 1e-9)


def scheduled_reset_invalid(
    plan: Mapping[str, Any], record: Mapping[str, Any], index: int,
    previous_record: Mapping[str, Any] | None = None,
) -> bool:
    """A scheduled reset record that does not prove how its scene was made.

    Round 0 must carry a passing verification that the loaded scene is its
    planned layout; a later round must carry a relaunch whose measured stage and
    clock fields show the previous planned layout's scene closed and the new
    simulation's clock at the reset the record reports.
    """

    schedule = scheduled_rounds(plan)
    if schedule is None:
        return False
    if index >= len(schedule):
        return True
    if index == 0:
        return (record.get("scene_reset_mode") != SCENE_RESET_INITIAL or not _round0_verified(record, schedule[0])
                or record.get("simulation_time_offset_s") != 0.0)
    return (record.get("scene_reset_mode") != SCENE_RESET_RELAUNCH
            or not _relaunch_proven(record, schedule[index - 1], previous_record))


def render_time_errors(records: Sequence[Mapping[str, Any]], resets: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every camera synchronization used its own epoch's offset, and matched on it.

    A scheduled session's synchronization records name the annotator offset
    they compared against. Epochs are entered in order, so each record's offset
    must be the current or a later reset record's offset, and a matched record's
    annotator times must equal that offset plus the record's context clock.
    """

    offsets = [record.get("simulation_time_offset_s") for record in resets]
    if not records or not all(_offset(value) for value in offsets):
        return ["robodojo_render_time_offset_invalid"]
    position = 0
    for record in records:
        offset = record.get("simulation_time_offset_s") if isinstance(record, Mapping) else None
        while position < len(offsets) and offsets[position] != offset:
            position += 1
        if position == len(offsets):
            return ["robodojo_render_time_offset_invalid"]
        samples = record.get("samples") or []
        if record.get("matched") is True and (not samples or any(
                not isinstance(camera.get("simulation_time_s"), (int, float))
                or abs(camera["simulation_time_s"] - offset - float(record.get("simulation_time_s", math.nan))) > 1e-6
                for camera in samples[-1].get("cameras", []))):
            return ["robodojo_render_time_offset_invalid"]
    return []


def render_time_file_errors(plan: Mapping[str, Any], native: Path, resets: Sequence[Mapping[str, Any]]) -> list[str]:
    """The scheduled join over ``native/render-synchronization.jsonl``; nothing unscheduled."""

    if scheduled_rounds(plan) is None:
        return []
    path = native / "render-synchronization.jsonl"
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError("render synchronization record is missing or unsafe")
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, UnicodeError):
        return ["robodojo_render_time_offset_invalid"]
    return render_time_errors(records, resets)


def same_clock_epoch(previous: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    """Only one simulation context's clock is monotone; a relaunch starts a new one."""

    return previous.get("physics_clock_epoch", 0) == current.get("physics_clock_epoch", 0)


def clock_continuity_applies(
    plan: Mapping[str, Any], previous: Mapping[str, Any], current: Mapping[str, Any],
) -> bool:
    """Whether a later round's reset clock must follow the previous round's.

    Always for an unscheduled plan (one simulation context for the session,
    epoch fields never consulted); within one epoch for a scheduled plan.
    """

    return scheduled_rounds(plan) is None or same_clock_epoch(previous, current)


def source_schedule_errors(plan: Mapping[str, Any], source: Mapping[str, Any]) -> list[str]:
    """The worker hashed exactly the planned files at start, or nothing unscheduled."""

    schedule = scheduled_rounds(plan)
    hashed = source.get("layout_schedule")
    if schedule is None:
        return [] if hashed is None else ["robodojo_layout_schedule_binding_invalid"]
    if (not isinstance(hashed, list) or len(hashed) != len(schedule) or any(
            not isinstance(item, Mapping) or item.get("layout_id") != entry.get("layout_id")
            or item.get("sha256") != entry.get("layout_sha256")
            or not isinstance(item.get("path"), str)
            or not Path(item["path"]).as_posix().endswith("/" + str(entry.get("layout_file")))
            for item, entry in zip(hashed, schedule))):
        return ["robodojo_layout_schedule_binding_invalid"]
    return []


def round_scene_errors(
    plan: Mapping[str, Any], result: Mapping[str, Any], resets: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Each scheduled round result names the scene its reset record measured."""

    if scheduled_rounds(plan) is None:
        return []
    records = result.get("rounds") if isinstance(result.get("rounds"), list) else []
    errors = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or index >= len(resets):
            continue
        expected = {field: resets[index].get(field) for field in ROUND_SCENE_FIELDS}
        if record.get("scene") != expected:
            errors.append(f"robodojo_round_scene_invalid:{index}")
    return errors


__all__ = [
    "ROUND_SCENE_FIELDS", "SCHEDULED_ISOLATION_CHECKS", "SCHEDULED_ROUND_RESET_SCHEMA",
    "SCHEDULED_START_RESET_SCHEMA", "SCHEDULED_VERDICT_SCHEMA", "isolation_check_names",
    "clock_continuity_applies", "isolation_layout_errors", "record_schemas", "render_time_errors",
    "render_time_file_errors", "round_identities",
    "round_scene_errors", "same_clock_epoch", "scheduled_reset_invalid", "scheduled_rounds",
    "source_schedule_errors",
]
