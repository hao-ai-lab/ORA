"""Worker-side per-round layout schedule of a persistent RoboDojo session.

A scheduled round is a new episode on a different official layout: the worker
closes the simulation and stage of the previous round inside the same Kit
process and launches the next layout (``EvaluationLifecycle.
relaunch_for_layout``). Nothing of the previous scene survives, so the
round-isolation proof cannot compare scene identities with round 0 as the
in-place reset does. It proves instead that the loaded scene is exactly the
scheduled layout file, that the simulation was relaunched, that the robot and
cameras are the same plant and that every layout object is a scene object.

Every helper here reads measurements; the legacy unscheduled session never
calls any of them and keeps its records byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ora.environment.benchmarks.robodojo_layouts import (
    SCENE_RESET_INITIAL, SCENE_RESET_RELAUNCH, layout_object_counts,
)

START_RESET_V2_SCHEMA = "ora.robodojo-reset.v2"
ROUND_RESET_V2_SCHEMA = "ora.robodojo-round-reset.v2"
VERDICT_V3_SCHEMA = "ora.robodojo-official-verdict.v3"
RELAUNCH_SCHEMA = "ora.robodojo-simulation-relaunch.v1"
LAYOUT_VERIFICATION_SCHEMA = "ora.robodojo-layout-verification.v1"
#: The restore facts a scheduled round must measure as True. The in-place
#: reset's ``layout_identity_unchanged`` is replaced by the four layout,
#: relaunch, plant and scene facts; the lifecycle, channel and home checks stay.
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


class LayoutScheduleError(RuntimeError):
    """A scheduled layout file is missing or differs from the RunPlan."""


def scheduled_layout_files(
    asset_root: Path, task: str, schedule: Mapping[str, Any] | None,
) -> list[dict[str, Any]] | None:
    """Hash every scheduled layout at worker start, or refuse the worker.

    Each ordinal is re-resolved through upstream's numeric sort and must be
    the planned file with the planned bytes.
    """

    if schedule is None:
        return None
    from ora.environment.benchmarks.robodojo import selected_layout

    assets = Path(asset_root).resolve()
    files = []
    for entry in schedule["rounds"]:
        path = selected_layout(assets, task=task, eval_seed=int(schedule["eval_seed"]),
                               layout_id=int(entry["layout_id"]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.relative_to(assets).as_posix() != entry["layout_file"] or digest != entry["layout_sha256"]:
            raise LayoutScheduleError(
                f"scheduled layout {entry['layout_id']} differs from the RunPlan file or bytes")
        files.append({"layout_id": int(entry["layout_id"]), "path": str(path), "sha256": digest})
    return files


def current_layout(session: Any) -> dict[str, Any]:
    """The layout fields of the round the session is in."""

    entry = session.layout_schedule["rounds"][session.round_index]
    return {"layout_id": entry["layout_id"], "layout_file": entry["layout_file"],
            "layout_sha256": entry["layout_sha256"], "env_seed": entry["layout_id"],
            "physics_clock_epoch": session.physics_clock_epoch}


def simulation_time_offset(session: Any) -> float:
    """Simulated time of every earlier epoch, which the camera annotators still count."""

    return float(getattr(session.environment, "harness_simulation_time_offset_s", None) or 0.0)


def start_reset_fields(session: Any) -> dict[str, Any]:
    """What a scheduled session's round-0 reset record adds; nothing unscheduled."""

    if session.layout_schedule is None:
        return {}
    return {"schema": START_RESET_V2_SCHEMA, **current_layout(session),
            "scene_reset_mode": SCENE_RESET_INITIAL,
            "layout_verification": session.round0_layout_verification,
            "simulation_time_offset_s": simulation_time_offset(session)}


def verdict_fields(session: Any) -> dict[str, Any]:
    """What a scheduled round's official verdict adds; nothing unscheduled."""

    if session.layout_schedule is None:
        return {}
    return {"schema": VERDICT_V3_SCHEMA, **current_layout(session)}


def plant_facts(session: Any) -> dict[str, Any]:
    """The compiled robot/camera binding and loaded finger geometry, read live."""

    geometry = None
    if session.config.get("public_robot_geometry") is not None:
        from .robot_asset_geometry import loaded_geometry
        geometry = loaded_geometry(session)
    return {"runtime_embodiment": session._runtime_embodiment(), "loaded_geometry": geometry}


def _manifest_file(session: Any, layout_id: int) -> dict[str, Any] | None:
    for item in session.source.get("layout_schedule") or ():
        if item.get("layout_id") == layout_id:
            return item
    return None


def _registered_counts(registration: Mapping[str, Any]) -> dict[str, int]:
    counts = {"articulation": 0, "garment": 0, "geometry": 0, "rigid": 0}
    for kind in registration["instances"].values():
        counts["rigid" if kind == "dynamic" else kind] += 1
    return counts


def loaded_layout_checks(session: Any, entry: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    """Is the live scene exactly this scheduled layout, object for object?

    The planned file still hashes to its planned bytes (as at worker start),
    the saved layout the scene manager loaded equals the parsed file, the
    upstream env seed is the layout id, and the scene manager's object counts
    and the registered instance kinds equal the layout file's.
    """

    layout_id = int(entry["layout_id"])
    manifest = _manifest_file(session, layout_id)
    try:
        data = Path(str((manifest or {}).get("path"))).read_bytes()
        current_hash = hashlib.sha256(data).hexdigest()
        planned_layout = json.loads(data)
    except (OSError, TypeError, ValueError):
        current_hash, planned_layout = "", None
    loaded = session.environment.loaded_layout_state()
    expected_counts = layout_object_counts(planned_layout) if isinstance(planned_layout, Mapping) else None
    registered_counts = _registered_counts(session._scene_registration())
    checks = {
        "layout_matches_round_schedule": (
            manifest is not None and current_hash == entry["layout_sha256"] == manifest.get("sha256")
            and loaded["layout"] == planned_layout and loaded["env_seed"] == layout_id),
        "scene_objects_match_layout": (
            expected_counts is not None and loaded["object_counts"] == expected_counts
            and registered_counts == expected_counts),
    }
    measured = {
        "layout_id": layout_id, "layout_file": entry["layout_file"],
        "layout_sha256": current_hash, "layout_sha256_planned": entry["layout_sha256"],
        "env_seed": loaded["env_seed"], "layout_object_counts": expected_counts,
        "loaded_object_counts": loaded["object_counts"], "registered_object_counts": registered_counts,
    }
    return checks, measured


def round0_layout_verification(session: Any) -> dict[str, Any]:
    """Round 0 must also be its scheduled layout; the worker refuses round 0 otherwise."""

    checks, measured = loaded_layout_checks(session, session.layout_schedule["rounds"][0])
    return {"schema": LAYOUT_VERIFICATION_SCHEMA, "passed": all(checks.values()),
            "checks": checks, "measured": measured}


def relaunch_proven(relaunch: Any, *, layout_id: int, previous_layout_id: int, reset_clock: int,
                    previous_offset: float, offset: float) -> bool:
    """Derive the relaunch from its measured stage, clock and time fields, not from its flags.

    The new epoch's annotator offset must be the previous offset plus the
    closed context's own simulated time at close.
    """

    if not isinstance(relaunch, Mapping) or relaunch.get("schema") != RELAUNCH_SCHEMA:
        return False
    before, after = relaunch.get("stage_before") or {}, relaunch.get("stage_after_close") or {}
    times = [relaunch.get(name) for name in (
        "simulation_time_before_close_s", "previous_simulation_time_offset_s", "simulation_time_offset_s")]
    return (relaunch.get("layout_id") == layout_id and relaunch.get("previous_layout_id") == previous_layout_id
            and before.get("env_prim_present") is True and after.get("env_prim_present") is False
            and relaunch.get("sim_closed") is True and relaunch.get("stage_replaced") is True
            and type(relaunch.get("physics_clock_after_reset")) is int
            and relaunch["physics_clock_after_reset"] == reset_clock
            and all(type(value) is float and math.isfinite(value) and value >= 0 for value in times)
            and times[1] == previous_offset and times[2] == offset
            and abs(times[2] - (times[1] + times[0])) <= 1e-9)


def scheduled_layout_checks(
    session: Any, entry: Mapping[str, Any], relaunch: Mapping[str, Any], *,
    previous_epoch: int, previous_layout_id: int, previous_offset: float,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Measure the scheduled-round facts that replace the layout identity check."""

    checks, measured = loaded_layout_checks(session, entry)
    checks.update({
        "simulation_relaunched": (
            relaunch_proven(relaunch, layout_id=int(entry["layout_id"]), previous_layout_id=previous_layout_id,
                            reset_clock=session.reset_physics_steps, previous_offset=previous_offset,
                            offset=simulation_time_offset(session))
            and session.physics_clock_epoch == previous_epoch + 1),
        "plant_embodiment_unchanged": plant_facts(session) == session.session_plant_facts,
        "task_instruction_unchanged": session._task_instruction() == session.session_task_instruction,
    })
    measured["physics_clock_epoch"] = session.physics_clock_epoch
    measured["simulation_time_offset_s"] = simulation_time_offset(session)
    return checks, measured


def round_reset_v2(session: Any, base: Mapping[str, Any], relaunch: Mapping[str, Any]) -> dict[str, Any]:
    """The scheduled round-reset record: the in-place record's fields plus the layout."""

    return {**base, "schema": ROUND_RESET_V2_SCHEMA, **current_layout(session),
            "scene_reset_mode": SCENE_RESET_RELAUNCH, "relaunch": dict(relaunch),
            "simulation_time_offset_s": simulation_time_offset(session)}


__all__ = [
    "LAYOUT_VERIFICATION_SCHEMA", "LayoutScheduleError", "RELAUNCH_SCHEMA", "ROUND_RESET_V2_SCHEMA",
    "SCHEDULED_ISOLATION_CHECKS", "START_RESET_V2_SCHEMA", "VERDICT_V3_SCHEMA", "current_layout",
    "loaded_layout_checks", "plant_facts", "relaunch_proven", "round0_layout_verification",
    "simulation_time_offset",
    "round_reset_v2", "scheduled_layout_checks", "scheduled_layout_files", "start_reset_fields",
    "verdict_fields",
]
