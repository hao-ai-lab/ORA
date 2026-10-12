"""Pinned RoboDojo motion-quality metrics (U1 v2 layer L0), host-safe and read-only.

The checker reads a sealed run's native controller journals (or a control
replay's journals against the original receipts) and reports every metric as
a numerator over a denominator, or a distribution, under one versioned
definition set (``robodojo_motion_measures``). It never imports Isaac, never
writes inside a run and never reaches a policy.

Saturation and out-of-step describe v1's endpoint interpolation only and are
grouped under ``v1_baseline_diagnostics``. The design's metrics for both
versions are commanded-speed violation, tracking error (for v1 the recorded
endpoint stands in for the waypoint), sag, held-then-lost, arrival and cost.
Peak ratio and impulse saturation need a planned trajectory and are
``not_applicable`` to v1.

    python -m ora.orchestrator.robodojo_motion_metrics RUN [RUN ...] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .robodojo_motion_journal import JournalError, load_run_journal, native_root_of, summarize_values
from .robodojo_motion_measures import (
    COHORTS, GRIPPER_TOLERANCE, HELD_OBJECT_DISPLACEMENT_M, JAW_CONTACT_CENTRE_LINK6_X_M, METRICS_VERSION,
    NATIVE_ACTION_S, ORIENTATION_TOLERANCE_RAD, OUT_OF_STEP_MINIMUM_EXCURSION_RAD, OUT_OF_STEP_SPREADS,
    PEAK_RATIO_BOUND, PHYSICS_DT_S, POSITION_TOLERANCE_M, SAG_MARGIN_M, SAG_SEGMENTS,
    SATURATED_JOINT_SPEED_RAD_S, SATURATED_TICK_DISPLACEMENT_RAD, STILL_JOINT_STEP_RAD,
    STILL_WINDOW_NATIVE_ACTIONS, TRACKING_TOLERANCE_RAD, V_CMD_CANDIDATE_RAD_S, link6_frame_offset,
    measure_journal, merge_partials, object_frame_check,
)

REPORT_SCHEMA = "ora.robodojo-motion-metrics-report.v1"
PLANNED_WAYPOINTS_SCHEMA = "ora.robodojo-planned-waypoints.v1"

DEFINITIONS: dict[str, Any] = {
    "version": METRICS_VERSION,
    "arm_tick": "one native action of one arm whose <arm>_arm_joint_state is in the recorded control",
    "speed_sample": "change between consecutive journaled physics steps over 0.004 s; a tick's first step "
                    "uses the previous tick's last step when the physics clock is contiguous, never across "
                    "a round start",
    "cohorts": "closed_command: tick-end recorded gripper directive close; holding_mode: after a set_gripper "
               "close ending stalled_before_target, until the arm's next set_gripper, the round end or the "
               "jaw reaching its floor",
    "segment": "consecutive ticks of one action with one segment name commanding one arm; global: move, "
               "plan_ref, precontact, retract, recover*; local: local",
    "saturation_physics_step": f"v1 only: arm-ticks with a joint speed sample >= {SATURATED_JOINT_SPEED_RAD_S} rad/s",
    "saturation_tick_displacement": f"v1 only: moving arm-ticks (joint change >= {STILL_JOINT_STEP_RAD} rad) "
                                    f"whose change is >= {SATURATED_TICK_DISPLACEMENT_RAD:.2f} rad",
    "out_of_step": f"v1 only: constant-target segments with at least two joints whose target is > "
                   f"{OUT_OF_STEP_MINIMUM_EXCURSION_RAD} rad from the start; normalized progress spread above "
                   "the threshold at any physics step",
    "commanded_speed_violation": "arm-ticks with an applied-target step between two of the tick's own "
                                 "physics steps, over 0.004 s, above v_cmd; the step from the previous tick "
                                 "(v1's interpolation restart) is excluded",
    "tracking_error": f"per arm-tick, largest joint |tick-end measured - waypoint|, above {TRACKING_TOLERANCE_RAD} "
                      "rad; v1 uses the recorded endpoint target as the waypoint",
    "peak_ratio": f"planned segments only: measured peak joint speed over the planned peak, above {PEAK_RATIO_BOUND}",
    "impulse_saturation": f"planned ticks whose predicted in-tick command is below v_cmd and whose measured "
                          f"speed reaches {SATURATED_JOINT_SPEED_RAD_S} rad/s",
    "sag": f"{sorted(SAG_SEGMENTS)} segments whose link6 z falls below min(z start, receipt goal z) - "
           f"{SAG_MARGIN_M} m; holding when the segment starts in the holding_mode cohort",
    "held_then_lost": f"closes ending above the jaw floor (closed_command; stalled ones also holding_mode), then "
                      f"the jaw at floor + {GRIPPER_TOLERANCE} at a tick end before the arm's next jaw command, or "
                      f"(replay only) the object nearest the jaw centre ({JAW_CONTACT_CENTRE_LINK6_X_M} m on link6 +x) "
                      f"drifting more than {HELD_OBJECT_DISPLACEMENT_M} m in link6",
    "arrival": f"pose commands whose targeted arms end within {POSITION_TOLERANCE_M} m and "
               f"{ORIENTATION_TOLERANCE_RAD} rad of the final commanded pose; the denominator is every pose command",
    "native_actions": "per command; per arm-independent segment (global and local separately); per round "
                      "against the task's official limit",
    "timeout_stillness": f"pose commands ending native_action_budget_exhausted whose targeted arms each moved no "
                         f"joint >= {STILL_JOINT_STEP_RAD} rad in each of the last {STILL_WINDOW_NATIVE_ACTIONS} ticks",
    "percentile": "value at index floor(q * n) of the ascending sort",
    "physics_dt_s": PHYSICS_DT_S, "native_action_s": NATIVE_ACTION_S,
}


def _ratio(counts: Mapping[str, int], name: str) -> dict[str, Any]:
    numerator = int(counts.get(f"{name}.numerator", 0))
    denominator = int(counts.get(f"{name}.denominator", 0))
    return {"numerator": numerator, "denominator": denominator,
            "share": round(numerator / denominator, 4) if denominator else None}


def _prefixed(values: Mapping[str, Sequence[float]], prefix: str) -> dict[str, Any]:
    return {key[len(prefix):]: summarize_values(items) for key, items in sorted(values.items())
            if key.startswith(prefix)}


def _by_cohort(counts: Mapping[str, int], name: str) -> dict[str, Any]:
    return {**_ratio(counts, f"{name}.all"), **{cohort: _ratio(counts, f"{name}.{cohort}") for cohort in COHORTS}}


def _not_applicable(reason: str) -> dict[str, Any]:
    return {"status": "not_applicable", "reason": reason}


def _v1_baseline(counts: Mapping[str, int]) -> dict[str, Any]:
    return {
        "label": "v1 baseline diagnostics: endpoint interpolation only, never v2 acceptance",
        "saturation_physics_step": _by_cohort(counts, "saturation_physics_step"),
        "saturation_tick_displacement": _ratio(counts, "saturation_tick_displacement"),
        "out_of_step": {**{f"spread_above_{value}": _ratio(counts, f"out_of_step.{value}")
                           for value in OUT_OF_STEP_SPREADS},
                        "changing_target_segments": int(counts.get("out_of_step.changing_target", 0)),
                        "applies_to": "v1 endpoint interpolation only; not applicable to planned trajectories"},
    }


def _planned(counts: Mapping[str, int], values: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    if not counts.get("inputs.planned_waypoints"):
        reason = "v1 has no planned trajectory"
        return {"peak_ratio": _not_applicable(reason),
                "impulse_saturation": _not_applicable(reason + " and no predicted in-tick command")}
    impulse = (_ratio(counts, "impulse_saturation") if counts.get("impulse_saturation.denominator")
               else _not_applicable("no planned tick carried a predicted in-tick command below v_cmd"))
    return {"peak_ratio": {**_ratio(counts, "peak_ratio"), "bound": PEAK_RATIO_BOUND,
                           "ratio": summarize_values(values.get("peak_ratio.ratio", []))},
            "impulse_saturation": impulse}


def _speed(counts: Mapping[str, int], values: Mapping[str, Sequence[float]], v_cmd: float) -> dict[str, Any]:
    return {
        "commanded_speed_violation": {
            **_by_cohort(counts, "commanded_speed"), "v_cmd_rad_s": v_cmd,
            "v_cmd_status": "candidate" if v_cmd == V_CMD_CANDIDATE_RAD_S else "operator_supplied",
            "commanded_peak_rad_s": summarize_values(values.get("commanded_speed.all.peak", [])),
            "unrecorded_arm_ticks": int(counts.get("commanded_speed.unrecorded", 0))},
        "link6_peak_speed_m_s": {cohort: summarize_values(values.get(f"link6_peak_speed.{cohort}", []))
                                 for cohort in ("all", *COHORTS)},
    }


def _held(counts: Mapping[str, int], records: Mapping[str, Sequence[Any]]) -> dict[str, Any]:
    objects = bool(counts.get("inputs.object_poses"))
    selected = list(records.get("held_then_lost.objects", []))
    frame = object_frame_check(selected) if objects else None
    # Object drift in a frame other than link6's is not evidence: keep the jaw proxy only.
    trusted = objects and frame is not None and frame["status"] != "inconsistent"

    def cohort(name: str) -> dict[str, Any]:
        prefix = f"held_then_lost.{name}"
        jaw = int(counts.get(f"{prefix}.jaw_floor", 0))
        ratio = _ratio(counts, prefix) if trusted or not objects else {
            **_ratio(counts, prefix), "numerator": jaw,
            "share": round(jaw / counts[f"{prefix}.denominator"], 4) if counts.get(f"{prefix}.denominator") else None}
        return {**ratio, "jaw_floor": jaw,
                "object_displacement": int(counts.get(f"{prefix}.object_displacement", 0)) if trusted else None,
                "object_measured": int(counts.get(f"{prefix}.object_measured", 0)) if trusted else None}

    return {**{name: cohort(name) for name in COHORTS}, "objects": selected if objects else None,
            "object_frame_check": frame,
            "object_evidence": None if not objects else "used" if trusted
            else "withheld: the object positions are not in link6's frame"}


def summarize(partial: Mapping[str, Any], *, v_cmd: float = V_CMD_CANDIDATE_RAD_S) -> dict[str, Any]:
    """Render merged counts, values and records as the fixed-definition metric object."""
    counts, values = partial.get("counts", {}), partial.get("values", {})
    waypoints = "planned_waypoints" if counts.get("inputs.planned_waypoints") else "recorded_control_target"
    return {
        "v1_baseline_diagnostics": _v1_baseline(counts),
        **_speed(counts, values, v_cmd),
        "tracking_error": {**_ratio(counts, "tracking_error"), "waypoint_source": waypoints,
                           "tolerance_rad": TRACKING_TOLERANCE_RAD,
                           "error_rad": summarize_values(values.get("tracking_error.error_rad", []))},
        **_planned(counts, values),
        "sag": {**{state: _ratio(counts, f"sag.{state}") for state in ("free_space", "holding")},
                "segments": sorted(SAG_SEGMENTS), "unmatched": int(counts.get("sag.unmatched", 0))},
        "held_then_lost": _held(counts, partial.get("records", {})),
        "arrival": {**_ratio(counts, "arrival"), "executed": _ratio(counts, "arrival.executed"),
                    "receipt_status": {key.split(".", 2)[2]: int(value) for key, value in sorted(counts.items())
                                       if key.startswith("arrival.status.")}},
        "native_actions": {
            "per_command": _prefixed(values, "native_actions.per_command."),
            "per_segment": {"global": summarize_values(values.get("native_actions.per_segment.global", [])),
                            "local": summarize_values(values.get("native_actions.per_segment.local", [])),
                            "by_segment": _prefixed(values, "native_actions.by_segment.")},
            "per_round": {**_ratio(counts, "native_actions.per_round"),
                          "rounds": int(counts.get("native_actions.per_round.rounds", 0))}},
        "timeout_stillness": _ratio(counts, "timeout_stillness"),
    }


def _run_entry(run: Path, *, journal_root: Path | None, upto: int | None, planned: Any, object_poses: Any,
               v_cmd: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """One run's metrics and partial, or a recorded refusal when its journals are malformed."""
    try:
        journal = load_run_journal(run, journal_root=journal_root, upto=upto)
    except JournalError as exc:
        return {"run_id": Path(run).name, "error": str(exc), "metrics": None}, None
    partial = measure_journal(journal, planned=planned, object_poses=object_poses, v_cmd=v_cmd)
    entry = {"run_id": journal.run_id, "native_root": str(journal.native_root),
             "journal_root": str(journal.journal_root), "task": journal.task,
             "native_action_limit": journal.native_action_limit, "actions": len(journal.actions),
             "native_actions": sum(len(item.ticks) for item in journal.actions), "error": None,
             "metrics": summarize(partial, v_cmd=v_cmd)}
    return entry, partial


def build_report(runs: Sequence[Path], *, journal_root: Path | None = None, upto: int | None = None,
                 planned: Mapping[tuple[int, str], Any] | None = None,
                 object_poses: Mapping[tuple[int, int], Any] | None = None,
                 v_cmd: float = V_CMD_CANDIDATE_RAD_S) -> dict[str, Any]:
    """Per-run metrics and the pooled metrics of every readable run under one definition version."""
    results = [_run_entry(run, journal_root=journal_root, upto=upto, planned=planned, object_poses=object_poses,
                          v_cmd=v_cmd) for run in runs]
    entries = [entry for entry, _ in results]
    partials = [partial for _, partial in results if partial is not None]
    return {"schema": REPORT_SCHEMA, "metric_definition_version": METRICS_VERSION, "definitions": DEFINITIONS,
            "parameters": {"v_cmd_rad_s": v_cmd}, "runs": entries,
            "errors": [entry for entry in entries if entry["error"]],
            "pooled": {"runs": len(partials), "metrics": summarize(merge_partials(partials), v_cmd=v_cmd)}}


def load_planned_waypoints(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    """``ora.robodojo-planned-waypoints.v1``: per action and arm, one joint waypoint per native
    action and optionally ``predicted_in_tick_speed_rad_s`` per waypoint."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != PLANNED_WAYPOINTS_SCHEMA:
        raise ValueError(f"planned waypoints must be {PLANNED_WAYPOINTS_SCHEMA}")
    result = {}
    for row in value.get("actions", []):
        positions = [[float(item) for item in waypoint] for waypoint in row["positions"]]
        if not positions:
            raise ValueError("a planned waypoint sequence is empty")
        predicted = row.get("predicted_in_tick_speed_rad_s")
        result[(int(row["action_sequence"]), str(row["arm"]))] = {
            "positions": positions,
            "predicted_in_tick_speed_rad_s": None if predicted is None else [float(v) for v in predicted]}
    return result


def load_object_poses(path: Path) -> dict[tuple[int, int], dict[str, tuple[float, ...]]]:
    """A control replay's private per-tick object positions (qualification only); error rows are skipped."""
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line) if line.strip() else {}
        except ValueError:
            continue
        objects = row.get("objects")
        if isinstance(objects, Mapping):
            result[(int(row["action_sequence"]), int(row["tick_index"]))] = {
                str(name): tuple(float(value) for value in item["position_m"]) for name, item in objects.items()}
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ora.orchestrator.robodojo_motion_metrics",
        description=f"Fixed-definition RoboDojo motion metrics ({METRICS_VERSION}); read-only, no simulator.")
    parser.add_argument("runs", nargs="+", type=Path, help="run directories or RoboDojo native roots")
    parser.add_argument("--journal-root", type=Path, help="controller traces of a control replay (one run)")
    parser.add_argument("--upto", type=int, help="measure only the first N actions")
    parser.add_argument("--planned-waypoints", type=Path, help=f"{PLANNED_WAYPOINTS_SCHEMA} file (one run)")
    parser.add_argument("--object-poses", type=Path, help="a control replay's private/object-poses.jsonl (one run)")
    parser.add_argument("--v-cmd", type=float, default=V_CMD_CANDIDATE_RAD_S,
                        help=f"commanded-speed bound in rad/s (default: the {V_CMD_CANDIDATE_RAD_S} candidate)")
    parser.add_argument("--output", type=Path, help="report path outside every source run; default stdout")
    return parser


def _run_root(run: Path) -> Path:
    native = native_root_of(run)
    return native.parent if native.name == "native" else native


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if len(args.runs) > 1 and any(value is not None for value in (args.journal_root, args.planned_waypoints,
                                                                    args.object_poses)):
        parser.error("--journal-root, --planned-waypoints and --object-poses apply to exactly one run")
    if (args.upto is not None and args.upto < 1) or args.v_cmd <= 0:
        parser.error("--upto must be at least 1 and --v-cmd positive")
    output = args.output.resolve() if args.output is not None else None
    if output is not None and any(output.is_relative_to(_run_root(run)) for run in args.runs):
        parser.error("--output must be outside every source run")
    report = build_report(
        args.runs, journal_root=args.journal_root, upto=args.upto, v_cmd=args.v_cmd,
        planned=load_planned_waypoints(args.planned_waypoints) if args.planned_waypoints else None,
        object_poses=load_object_poses(args.object_poses) if args.object_poses else None)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(text)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(json.dumps({"report": str(output), "runs": len(report["runs"]), "errors": len(report["errors"])}))
    return 1 if report["errors"] else 0


__all__ = [
    "DEFINITIONS", "METRICS_VERSION", "PLANNED_WAYPOINTS_SCHEMA", "REPORT_SCHEMA", "build_report",
    "link6_frame_offset", "load_object_poses", "load_planned_waypoints", "main", "summarize",
]


if __name__ == "__main__":
    sys.exit(main())
