"""Host-safe per-round layout schedule of a persistent RoboDojo session.

A persistent RoboDojo session either replays its round-0 layout in every round
(the legacy treatment: a RunPlan without ``runtime.layout_schedule``) or starts
each round as a new episode on the next layout of a reviewed schedule. The
schedule names official saved layouts of one evaluation seed by their upstream
ordinal (``SeedManager.init_eval`` sorts ``<task>_<n>.json`` numerically) and
pins each file's bytes. Upstream's per-environment seed equals that ordinal,
so a round's ``env_seed`` is its ``layout_id``; the RunPlan ``seed`` stays the
evaluation seed.

The RunSpec, the isolated worker and the evaluation read this module. It
imports no simulator and reads no file: which layouts are admissible at all is
the reviewed catalog's decision (``robodojo_layout_catalog``), and whether the
bytes on disk match is the worker's and preflight's.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping

from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError

LAYOUT_SCHEDULE_SCHEMA = "ora.robodojo-layout-schedule.v1"
#: A round's upstream per-environment seed is its layout ordinal.
ENV_SEED_POLICY = "layout_id"
LAYOUT_CATALOG_PATH = "ora/environment/benchmarks/robodojo-layout-pools-v1.json"
#: A builder selection that keeps the round-0 layout in every round: the
#: deliberate historical treatment of the g003-g007 cells, which carries no
#: ``layout_schedule`` field.
REPEAT_ROUND0_LAYOUT = "repeat_round0_layout"
#: How a round's scene came to be, recorded on every scheduled reset record.
SCENE_RESET_INITIAL = "initial_launch"
SCENE_RESET_RELAUNCH = "simulation_relaunch"
SCHEDULE_FIELDS = frozenset({"schema", "eval_seed", "env_seed_policy", "rounds", "selection"})
ROUND_FIELDS = frozenset({"layout_id", "layout_file", "layout_sha256"})
SELECTION_FIELDS = frozenset({"catalog", "catalog_sha256"})
MAXIMUM_SCHEDULED_ROUNDS = 32
#: Tasks whose scene setup is not relaunch-safe: fold_clothes prepares CUDA
#: garments by wrapping scene-manager and garment methods once per setup, so a
#: relaunch would wrap them again. They keep the unscheduled treatment.
RELAUNCH_UNSAFE_TASKS = frozenset({"fold_clothes"})
#: Layout sections whose instances upstream registers as scene objects on
#: CUDA, mapped to the scene-manager collection that holds them. Fluid is
#: disabled upstream on CUDA and never registered.
LAYOUT_OBJECT_COLLECTIONS = {
    "Rigid": "rigid", "Dynamic": "rigid", "Articulation": "articulation",
    "Garment": "garment", "Geometry": "geometry",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RoboDojoBenchmarkError(message)


def _ordinal(value: Any) -> bool:
    return type(value) is int and value >= 0


def layout_file_for(task: str, eval_seed: int, layout_id: int) -> str:
    """The pinned Assets-relative file of one official layout ordinal.

    Every reviewed task numbers its files contiguously from 0, so ordinal and
    suffix coincide; the worker and preflight re-resolve the ordinal by
    upstream's numeric sort and refuse a file that is not this one.
    """

    return f"Eval_Layout/RoboDojo/arx_x5/{eval_seed}/{task}_{layout_id}.json"


def validate_layout_schedule(
    schedule: Any, *, task: str, eval_seed: Any, layout_id: Any, layout_file: Any,
    layout_sha256: Any, rounds: int | None = None,
) -> dict[str, Any]:
    """Refuse a schedule that is not one distinct official layout per round.

    The first entry must be the RunPlan's own round-0 layout fields, every
    entry must name its task's canonical file of the same evaluation seed, and
    no two rounds may share a layout or its bytes: a repeated layout is the
    legacy treatment, which carries no schedule at all.
    """

    _require(task not in RELAUNCH_UNSAFE_TASKS,
             f"{task} scene setup is not relaunch-safe and cannot follow a layout schedule")
    _require(isinstance(schedule, Mapping) and set(schedule) == SCHEDULE_FIELDS,
             "runtime.layout_schedule fields must be " + ", ".join(sorted(SCHEDULE_FIELDS)))
    _require(schedule["schema"] == LAYOUT_SCHEDULE_SCHEMA,
             f"runtime.layout_schedule.schema must be {LAYOUT_SCHEDULE_SCHEMA}")
    _require(type(schedule["eval_seed"]) is int and schedule["eval_seed"] == eval_seed,
             "runtime.layout_schedule.eval_seed must equal the evaluation seed")
    _require(schedule["env_seed_policy"] == ENV_SEED_POLICY,
             f"runtime.layout_schedule.env_seed_policy must be {ENV_SEED_POLICY!r}")
    entries = schedule["rounds"]
    _require(isinstance(entries, list) and 2 <= len(entries) <= MAXIMUM_SCHEDULED_ROUNDS,
             f"runtime.layout_schedule.rounds must list 2..{MAXIMUM_SCHEDULED_ROUNDS} layouts")
    _require(rounds is None or len(entries) == rounds,
             "runtime.layout_schedule must schedule exactly one layout per session round")
    for entry in entries:
        _require(isinstance(entry, Mapping) and set(entry) == ROUND_FIELDS
                 and _ordinal(entry["layout_id"])
                 and entry["layout_file"] == layout_file_for(task, eval_seed, entry["layout_id"])
                 and isinstance(entry["layout_sha256"], str)
                 and _SHA256.fullmatch(entry["layout_sha256"]) is not None,
                 "every scheduled round needs its task's official layout id, file and sha256")
    _require(len({entry["layout_id"] for entry in entries}) == len(entries)
             and len({entry["layout_sha256"] for entry in entries}) == len(entries),
             "a layout schedule may not repeat a layout; repeating round 0 is the unscheduled treatment")
    _require(dict(entries[0]) == {"layout_id": layout_id, "layout_file": layout_file,
                                  "layout_sha256": layout_sha256},
             "runtime.layout_schedule.rounds[0] must equal the runtime layout_id/layout_file/layout_sha256")
    selection = schedule["selection"]
    _require(isinstance(selection, Mapping) and set(selection) == SELECTION_FIELDS
             and selection["catalog"] == LAYOUT_CATALOG_PATH
             and isinstance(selection["catalog_sha256"], str)
             and _SHA256.fullmatch(selection["catalog_sha256"]) is not None,
             f"runtime.layout_schedule.selection must bind {LAYOUT_CATALOG_PATH} by sha256")
    return deepcopy(dict(schedule))


def configured_layout_schedule(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """The worker's validated schedule, or ``None`` for the legacy treatment."""

    schedule = config.get("layout_schedule")
    if schedule is None:
        return None
    return validate_layout_schedule(
        schedule, task=str(config.get("task", config.get("task_variant"))),
        eval_seed=config.get("eval_seed", 0), layout_id=config.get("layout_id"),
        layout_file=config.get("layout_file"), layout_sha256=config.get("layout_sha256"),
    )


def scheduled_layout(schedule: Mapping[str, Any], round_index: int) -> dict[str, Any]:
    """One round's planned layout, refusing a round the schedule does not cover."""

    entries = schedule["rounds"]
    _require(type(round_index) is int and 0 <= round_index < len(entries),
             f"round {round_index!r} is outside the layout schedule")
    return dict(entries[round_index])


def layout_cell_tag(schedule: Mapping[str, Any] | None, layout_id: int = 0) -> str:
    """The cell-id layout segment: ``l<layout_id>`` unscheduled (``l0`` by default), ``l21.37.51`` scheduled."""

    if schedule is None:
        _require(_ordinal(layout_id), "an unscheduled layout tag needs one layout ordinal")
        return f"l{layout_id}"
    return "l" + ".".join(str(entry["layout_id"]) for entry in schedule["rounds"])


def layout_object_counts(layout: Mapping[str, Any]) -> dict[str, int]:
    """Scene-object counts a saved layout must produce, per scene collection."""

    counts = {name: 0 for name in sorted(set(LAYOUT_OBJECT_COLLECTIONS.values()))}
    for section, collection in LAYOUT_OBJECT_COLLECTIONS.items():
        categories = layout.get(section) or {}
        _require(isinstance(categories, Mapping), f"layout section {section} is malformed")
        for instances in categories.values():
            _require(isinstance(instances, list), f"layout section {section} is malformed")
            counts[collection] += len(instances)
    return counts


__all__ = [
    "ENV_SEED_POLICY", "LAYOUT_CATALOG_PATH", "LAYOUT_OBJECT_COLLECTIONS",
    "LAYOUT_SCHEDULE_SCHEMA", "RELAUNCH_UNSAFE_TASKS", "REPEAT_ROUND0_LAYOUT", "SCENE_RESET_INITIAL",
    "SCENE_RESET_RELAUNCH", "configured_layout_schedule", "layout_cell_tag",
    "layout_file_for", "layout_object_counts", "scheduled_layout",
    "validate_layout_schedule",
]
