"""Control replay of a sealed RoboDojo run, inside the isolated RoboDojo interpreter.

Qualification only (layer L0, replay contract ``control``). No policy,
Provider or ToolSurface reaches this module, and it never writes inside the
source run. ``ora.orchestrator.robodojo_motion_replay`` builds the
worker configuration from the run's own RunPlan and starts this module with the
run's RoboDojo interpreter; see that module for the operator command.

Contract:

* Inputs, read-only from the sealed native root: ``source-manifest.json``,
  ``reset.json``, ``receipts/action-NNNN.json`` (round index and status) and
  ``controller/action-NNNN.json`` (per native tick: ``segment``, the recorded
  ``control`` dict and the tick-end ``measured`` state). The plant must report
  the run's source revisions and ``actual_source_fingerprint`` (loaded files,
  installed packages, resolved configuration; the device binding is not part
  of it), or nothing is replayed.
* Round 0 starts from ``session.start()``; every later round from
  ``session.restart_round`` (the session-home joint write and upstream reset
  the run used). Before each tick the session's gripper directives are
  restored from the tick's recorded ``gripper_feedback`` (the published
  ``gripper_command_target`` derives from them, not from physics); then the
  recorded ``control`` dict goes through the unchanged ``native_tick``
  (``session._conformance_tick``). SDK IK, planner and executor never run, so
  the replay reproduces physics, not decisions.
* The executor can rewrite a retained SDK target outside any tick (a
  ``motion_stall`` hold with no native action left writes ``prev_control``
  directly). Before each tick, every target-arm dimension the tick does not
  command (``<arm>_arm_joint_state``, ``<arm>_ee_joint_state``) is compared
  with the recorded first physics line's resolved ``command`` -- which is that
  dimension's exact value at the tick start -- and seeded from it when they
  differ; each seed is listed. Support-arm keys are never seeded.
* Every physics step is journaled in the session's ``record_physics`` fields.
  Applied targets that differ from the resolved command are counted, not
  raised. The original session's second check, that held dimensions kept
  their pre-action targets, is omitted here; the host compares every applied
  target with the original journal line instead.
* Actions with no native tick (refusals) are recorded and skipped. A
  ``plan_ref`` move replays its recorded controls; the reference is not
  re-resolved, so the probe that minted it need not run. Read-only probes
  (``probe_motion``, ``reach_check``, ``align_plan``, grasp IK annotations)
  and observations advance no physics (the probe clock guard) and are not
  replayed. A round that turns terminal before its recorded end stops there.
* Outputs under ``--output``: ``environment/`` (the session root, with the
  replayed ``controller/action-NNNN.json{,l}`` in the original layout),
  ``private/object-poses.jsonl`` (private rigid-object positions after every
  tick, env-relative as the evaluator reads them and assumed equal to the
  episode-local link6 frame until the host's frame check confirms it;
  qualification evidence, never policy-visible) and ``replay-worker.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

WORKER_REPORT_SCHEMA = "ora.robodojo-control-replay-worker.v1"
OBJECT_POSES_SCHEMA = "ora.robodojo-private-object-poses.v1"
OBJECT_POSITION_FRAME = "env_relative (assumed = episode_local), unverified"
REPLAY_CONTRACT = "control"
ARMS = ("left", "right")
#: Kit sometimes never returns from close; evidence is already on disk (seen while reproducing a reset).
CLOSE_TIMEOUT_S = 60.0
#: The session's own applied-target equality bound (``session._run_native_control``).
APPLIED_TARGET_ATOL = 1e-6
DIVERGENCE_FIELDS = ("joint_rad", "finger_m", "gripper")
SEEDED_KEYS = tuple(f"{arm}_{part}_joint_state" for arm in ARMS for part in ("arm", "ee"))


class ReplayError(RuntimeError):
    """The recorded run cannot be replayed on this plant."""


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m ora.environment.backends.robodojo.control_replay",
                                     description="Replay a sealed RoboDojo run's native controls (qualification only).")
    parser.add_argument("--config", type=Path, required=True, help="worker configuration JSON from the launcher")
    parser.add_argument("--native", type=Path, required=True, help="the sealed run's native root (read-only)")
    parser.add_argument("--output", type=Path, required=True, help="replay pass directory outside the run")
    parser.add_argument("--upto", type=int, default=None, help="replay only the first N recorded actions")
    arguments = parser.parse_args(argv)
    if arguments.upto is not None and arguments.upto < 1:
        parser.error("--upto must be at least 1")
    return arguments


def _load(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tick_commands(journal: Path, ticks: int) -> tuple[dict[str, Any], ...]:
    """The resolved ``command`` of each tick's first physics line (10 lines per tick)."""
    try:
        with journal.open(encoding="utf-8") as stream:
            commands = tuple(json.loads(line)["command"] for index, line in enumerate(stream) if index % 10 == 0)
    except (OSError, ValueError, KeyError) as exc:
        raise ReplayError(f"{journal.name} cannot be read: {exc}") from exc
    if len(commands) != ticks:
        raise ReplayError(f"{journal.name} has {len(commands)} tick starts for {ticks} recorded ticks")
    return commands


def recorded_actions(native_root: Path, *, upto: int | None = None) -> tuple[dict[str, Any], ...]:
    """Every recorded action in sequence: round index, native-tick samples and tick-start commands."""
    from .session import CONTROL_TRACE_SCHEMA

    native = Path(native_root)
    sequences = sorted(int(path.stem.split("-")[1]) for path in (native / "controller").glob("action-*.json"))
    if sequences != list(range(len(sequences))):
        raise ReplayError("controller traces are not contiguous from action-0000")
    result = []
    for sequence in sequences[:upto] if upto is not None else sequences:
        receipt = _load(native / "receipts" / f"action-{sequence:04d}.json")
        trace = _load(native / "controller" / f"action-{sequence:04d}.json")
        if trace.get("schema") != CONTROL_TRACE_SCHEMA:
            raise ReplayError(f"controller/action-{sequence:04d}.json is not {CONTROL_TRACE_SCHEMA}")
        if trace.get("action_id") != receipt.get("action_id"):
            raise ReplayError(f"controller/action-{sequence:04d}.json action_id differs from its receipt")
        result.append({"sequence": sequence, "round_index": int(receipt.get("round_index", 0)),
                       "action_id": receipt.get("action_id"), "action": receipt.get("action"),
                       "execution_status": receipt.get("execution_status"),
                       "samples": tuple(trace.get("samples", [])),
                       "tick_commands": _tick_commands(native / "controller" / f"action-{sequence:04d}.jsonl",
                                                       len(trace.get("samples", [])))})
    return tuple(result)


def rounds_of(actions: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, tuple[Mapping[str, Any], ...]], ...]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for item in actions:
        groups.setdefault(int(item["round_index"]), []).append(item)
    return tuple((index, tuple(members)) for index, members in sorted(groups.items()))


def _largest(pairs: Any) -> float:
    return max((abs(float(a) - float(b)) for a, b in pairs), default=0.0)


def _same_jaw_source(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    sources = [(state.get("gripper_feedback") or {}).get("measurement_source") for state in (first, second)]
    return sources[0] == sources[1]


def state_divergence(recorded: Mapping[str, Any], replayed: Mapping[str, Any]) -> dict[str, float]:
    """Largest arm-joint, finger-joint and normalized-jaw difference over both arms.

    The normalized jaw reading is compared only where both states derive it
    from the same measurement source: a run recorded before the two-finger
    reading published the primary finger alone, and the finger
    joints themselves are compared in ``finger_m`` either way.
    """
    return {
        "joint_rad": _largest((a, b) for arm in ARMS for a, b in zip(
            recorded[arm]["arm_joint_qpos"], replayed[arm]["arm_joint_qpos"], strict=True)),
        "finger_m": _largest((a, b) for arm in ARMS for a, b in zip(
            recorded[arm].get("gripper_finger_qpos", []), replayed[arm].get("gripper_finger_qpos", []), strict=True)),
        "gripper": _largest((recorded[arm]["gripper"], replayed[arm]["gripper"]) for arm in ARMS
                            if _same_jaw_source(recorded[arm], replayed[arm])),
    }


def restore_gripper_directives(session: Any, sample: Mapping[str, Any]) -> None:
    """Set the session's directives to the ones the tick recorded (reporting state, not physics)."""
    restored = {arm: float(state["gripper_feedback"]["requested_opening_normalized"])
                for arm, state in sample["measured"].items()
                if isinstance(state.get("gripper_feedback"), Mapping)
                and state["gripper_feedback"].get("requested_opening_normalized") is not None}
    session.gripper_directives = {**dict(session.gripper_directives), **restored}


def _plain(value: Any) -> Any:
    from .common import json_default

    return json.loads(json.dumps(value, default=json_default))


def seed_held_targets(session: Any, sample: Mapping[str, Any], command: Mapping[str, Any]) -> list[str]:
    """Seed each uncommanded target-arm dimension that differs from its recorded tick-start value."""
    retained = session.environment.robot_manager.control_manager.prev_control[0]
    seeded = [key for key in SEEDED_KEYS
              if key in command and key not in sample["control"] and _plain(retained.get(key)) != command[key]]
    for key in seeded:
        retained[key] = deepcopy(command[key])
    return seeded


class ReplayJournal:
    """Replay traces in the original ``controller/`` layout under the session output root."""

    def __init__(self, session_root: Path) -> None:
        self.root = Path(session_root) / "controller"
        self.root.mkdir(parents=True, exist_ok=True)

    def open(self, sequence: int) -> Path:
        path = self.root / f"action-{sequence:04d}.jsonl"
        if path.exists():
            raise ReplayError(f"refusing to overwrite a replay journal: {path}")
        path.touch()
        return path

    def append(self, path: Path, line: Mapping[str, Any]) -> None:
        from .common import json_default

        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(line, default=json_default, allow_nan=False, sort_keys=True) + "\n")

    def close(self, action: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], path: Path) -> None:
        from .common import file_hash, write_json
        from .session import CONTROL_TRACE_SCHEMA

        write_json(self.root / f"action-{int(action['sequence']):04d}.json", {
            "schema": CONTROL_TRACE_SCHEMA, "action_id": action["action_id"], "samples": list(samples),
            "physics_journal": {"path": f"controller/{path.name}", "sha256": file_hash(path)},
        })


def _rigid_instances(layout: Any) -> list[str]:
    return [str(name) for name, kind in sorted(layout.instance_type_by_env[0].items()) if kind == "rigid"]


def object_pose_reader(session: Any) -> Callable[[], dict[str, Any]]:
    """Private rigid-object poses through the evaluator's own CPU adapter (qualification only)."""

    def read() -> dict[str, Any]:
        from .bridge import EvaluatorLayoutInput

        layout = session.environment.scene_manager.layout_manager
        adapter = EvaluatorLayoutInput(layout)
        objects = {}
        for name in _rigid_instances(layout):
            position, orientation = adapter.get_instance_pose(0, inst_name=name, relative=True)
            objects[name] = {"position_m": [float(v) for v in position],
                             "orientation": [float(v) for v in orientation]}
        return objects

    return read


def object_frame_probe(session: Any) -> dict[str, Any]:
    """The layout manager's env origin (absolute minus relative pose of one rigid instance)."""
    try:
        from .bridge import EvaluatorLayoutInput

        layout = session.environment.scene_manager.layout_manager
        adapter = EvaluatorLayoutInput(layout)
        name = _rigid_instances(layout)[0]
        relative, _ = adapter.get_instance_pose(0, inst_name=name, relative=True)
        absolute, _ = adapter.get_instance_pose(0, inst_name=name, relative=False)
        return {"status": "read", "instance": name,
                "layout_env_origin_m": [float(a) - float(r) for a, r in zip(absolute, relative)]}
    except Exception as exc:  # noqa: BLE001 - a probe failure is evidence, not a replay failure
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


@dataclass
class ObjectLog:
    """Append one private object-pose row after every replayed tick; errors are recorded, never raised."""

    path: Path
    reader: Callable[[], Mapping[str, Any]]
    rows: int = 0
    errors: int = 0
    first_error: str | None = None

    def record(self, *, sequence: int, index: int, round_index: int, native_action: int) -> None:
        row: dict[str, Any] = {"action_sequence": sequence, "tick_index": index, "round_index": round_index,
                               "native_action": native_action}
        try:
            row["objects"] = dict(self.reader())
        except Exception as exc:  # noqa: BLE001 - the replay itself must not depend on this read
            row["error"] = f"{type(exc).__name__}: {exc}"
            self.errors += 1
            self.first_error = self.first_error or row["error"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        self.rows += 1

    def summary(self) -> dict[str, Any]:
        return {"schema": OBJECT_POSES_SCHEMA, "path": "private/object-poses.jsonl", "rows": self.rows,
                "errors": self.errors, "first_error": self.first_error, "position_frame": OBJECT_POSITION_FRAME,
                "orientation": "as returned by the upstream layout manager, not converted",
                "visibility": "qualification_only_never_policy_visible"}


@dataclass
class _Progress:
    ticks: int = 0
    mismatches: int = 0
    divergence: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in DIVERGENCE_FIELDS})
    first: dict[str, Any] | None = None
    skipped: list[dict[str, Any]] = field(default_factory=list)
    seeds: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, sequence: int, index: int, value: Mapping[str, float]) -> None:
        self.ticks += 1
        self.divergence = {name: max(self.divergence[name], value[name]) for name in DIVERGENCE_FIELDS}
        if any(value[name] > 0.0 for name in DIVERGENCE_FIELDS) and self.first is None:
            self.first = {"action_sequence": sequence, "tick_index": index, **dict(value)}


def _physics_recorder(session: Any, action: Mapping[str, Any], segment: str, journal: ReplayJournal,
                      path: Path, progress: _Progress) -> Callable[[Mapping[str, Any]], None]:
    """The session's ``record_physics`` fields, counting mismatches instead of faulting."""
    import numpy as np

    def record(command: Mapping[str, Any]) -> None:
        applied = session._applied_position_targets()
        journal.append(path, {"action_id": action["action_id"], "segment": segment,
                              "physics_clock": session._clock(),
                              "native_action": int(session.environment.take_action_cnt[0]),
                              "command": command, "applied_position_targets": applied,
                              "measured": session._proprioception()["end_effectors"]})
        progress.mismatches += sum(
            1 for key, target in command.items() if key in applied
            and not np.allclose(applied[key], target["position"], rtol=0, atol=APPLIED_TARGET_ATOL))

    return record


def _replay_tick(session: Any, action: Mapping[str, Any], index: int, journal: ReplayJournal,
                 path: Path, progress: _Progress) -> dict[str, Any]:
    environment, sample = session.environment, action["samples"][index]
    restore_gripper_directives(session, sample)
    seeded = seed_held_targets(session, sample, action["tick_commands"][index])
    if seeded:
        progress.seeds.append({"action_sequence": action["sequence"], "tick_index": index, "keys": seeded})
    environment.controller_trace_callback = _physics_recorder(session, action, str(sample["segment"]),
                                                              journal, path, progress)
    try:
        session._conformance_tick(deepcopy(sample["control"]))
    finally:
        environment.controller_trace_callback = None
    return {"native_action": int(environment.take_action_cnt[0]), "physics_steps": session.sim_steps,
            "segment": sample["segment"], "control": deepcopy(sample["control"]),
            "measured": session._proprioception()["end_effectors"]}


def _replay_action(session: Any, action: Mapping[str, Any], journal: ReplayJournal, progress: _Progress,
                   objects: ObjectLog | None) -> int:
    """Replay one action's ticks; returns how many recorded ticks the terminal left unreplayed."""
    path = journal.open(int(action["sequence"]))
    replayed: list[dict[str, Any]] = []
    if not action["samples"]:
        progress.skipped.append({"action_sequence": action["sequence"],
                                 "execution_status": action.get("execution_status")})
    for index, sample in enumerate(action["samples"]):
        if session.is_terminal:
            journal.close(action, replayed, path)
            return len(action["samples"]) - index
        replayed.append(_replay_tick(session, action, index, journal, path, progress))
        progress.observe(int(action["sequence"]), index, state_divergence(sample["measured"], replayed[-1]["measured"]))
        if objects is not None:
            objects.record(sequence=int(action["sequence"]), index=index, round_index=int(action["round_index"]),
                           native_action=replayed[-1]["native_action"])
    journal.close(action, replayed, path)
    return 0


def _replay_round(session: Any, round_index: int, members: Sequence[Mapping[str, Any]], journal: ReplayJournal,
                  progress: _Progress, objects: ObjectLog | None) -> dict[str, Any]:
    recorded = sum(len(item["samples"]) for item in members)
    before, unreplayed = progress.ticks, 0
    for item in members:
        if unreplayed:
            journal.close(item, [], journal.open(int(item["sequence"])))
            unreplayed += len(item["samples"])
            continue
        unreplayed = _replay_action(session, item, journal, progress, objects)
    return {"round_index": round_index, "ticks_recorded": recorded, "ticks_replayed": progress.ticks - before,
            "ticks_unreplayed": unreplayed, "terminal_before_recorded_end": unreplayed > 0,
            "terminal_after_round": bool(session.is_terminal),
            "official_terminal": deepcopy(getattr(session, "official_terminal", None))}


def replay_control(session: Any, actions: Sequence[Mapping[str, Any]], *, journal: ReplayJournal,
                   objects: ObjectLog | None = None) -> dict[str, Any]:
    """Feed every recorded control dict through ``native_tick``, round by round."""
    progress, rounds, started = _Progress(), [], time.time()
    for round_index, members in rounds_of(actions):
        for index in range(int(getattr(session, "round_index", 0)) + 1, round_index + 1):
            session.restart_round(index, seed=0)
        rounds.append(_replay_round(session, round_index, members, journal, progress, objects))
    return {
        "replay_contract": REPLAY_CONTRACT, "actions_selected": len(actions),
        "ticks_recorded": sum(len(item["samples"]) for item in actions), "ticks_replayed": progress.ticks,
        "max_divergence": progress.divergence if progress.ticks else None,
        "max_joint_divergence_rad": progress.divergence["joint_rad"] if progress.ticks else None,
        "first_divergence": progress.first, "applied_target_mismatches": progress.mismatches,
        "held_target_check": "omitted; the host compares every applied target with the original journal",
        "held_target_seeds": progress.seeds,
        "skipped_zero_tick_actions": progress.skipped,
        "not_replayed": {
            "zero_tick_actions": "refusals advanced no physics; listed above",
            "probes_and_observations": "advance no physics (probe clock guard); never replayed",
            "plan_ref": "replayed through its recorded controls; the reference is not re-resolved",
        },
        "rounds": rounds, "seconds": round(time.time() - started, 3),
    }


def _require_same_sources(session: Any, reset: Mapping[str, Any], native_root: Path) -> None:
    """Refuse a plant whose revisions, layout or actual source fingerprint differ from the run's."""
    recorded = _load(Path(native_root) / "source-manifest.json")
    current = getattr(session, "source", {}) or {}
    if recorded.get("source_revisions") != current.get("source_revisions"):
        raise ReplayError("worker source revisions differ from the run: "
                          f"{current.get('source_revisions')} != {recorded.get('source_revisions')}")
    layouts = (recorded.get("layout_file", {}).get("sha256"), current.get("layout_file", {}).get("sha256"))
    if all(layouts) and layouts[0] != layouts[1]:
        raise ReplayError("worker layout file differs from the run")
    fingerprint = _load(Path(native_root) / "reset.json").get("actual_source_fingerprint")
    if fingerprint is None or fingerprint != reset.get("actual_source_fingerprint"):
        raise ReplayError("worker actual_source_fingerprint (loaded files, installed packages, resolved "
                          "configuration) differs from the run's")


def _bounded_close(session: Any, timeout_s: float = CLOSE_TIMEOUT_S) -> bool:
    thread = threading.Thread(target=session.close, daemon=True)
    thread.start()
    thread.join(timeout_s)
    return not thread.is_alive()


def _write_report(output: Path, report: Mapping[str, Any]) -> None:
    (output / "replay-worker.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                                               encoding="utf-8")


def _session(config: Mapping[str, Any], output_root: Path) -> Any:
    from .session import RobodojoSession

    return RobodojoSession(config=config, output_root=output_root)


def _replay_session(session: Any, config: Mapping[str, Any], native_root: Path, output: Path, upto: int | None,
                    read_objects: Callable[[], Mapping[str, Any]] | None) -> dict[str, Any]:
    actions = recorded_actions(native_root, upto=upto)
    started = time.time()
    reset = session.start()
    stages = {"start_s": round(time.time() - started, 3)}
    _require_same_sources(session, reset, native_root)
    probe = object_frame_probe(session)
    objects = ObjectLog(output / "private" / "object-poses.jsonl", read_objects or object_pose_reader(session))
    replay = replay_control(session, actions, journal=ReplayJournal(Path(config["output_root"])), objects=objects)
    return {"stages": stages, "replay": replay, "object_poses": {**objects.summary(), "frame_probe": probe},
            "completed": replay["ticks_replayed"] == replay["ticks_recorded"]}


def run_worker(config: Mapping[str, Any], native_root: Path, output: Path, *, upto: int | None = None,
               session_factory: Callable[..., Any] | None = None,
               read_objects: Callable[[], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Start the plant, check its sources, replay, record and close; never raises a replay failure."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"schema": WORKER_REPORT_SCHEMA, "replay_contract": REPLAY_CONTRACT,
                              "native_root": str(native_root), "upto": upto, "error": None, "completed": False,
                              "replay": None, "object_poses": None, "stages": {}}
    session = (session_factory or _session)(config=config, output_root=Path(config["output_root"]))
    try:
        report.update(_replay_session(session, config, native_root, output, upto, read_objects))
    except Exception as exc:  # noqa: BLE001 - recorded for the host report, then the plant is closed
        report.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        _write_report(output, report)
        report["closed_within_timeout"] = _bounded_close(session)
        _write_report(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    config = _load(arguments.config)
    report = run_worker(config, arguments.native, arguments.output, upto=arguments.upto)
    return 0 if report["error"] is None else 1


__all__ = [
    "OBJECT_POSES_SCHEMA", "OBJECT_POSITION_FRAME", "REPLAY_CONTRACT", "ReplayError", "ReplayJournal",
    "WORKER_REPORT_SCHEMA", "main", "object_frame_probe", "object_pose_reader", "parse_arguments",
    "recorded_actions", "replay_control", "restore_gripper_directives", "rounds_of", "run_worker",
    "seed_held_targets", "state_divergence",
]


if __name__ == "__main__":  # pragma: no cover - runs in the RoboDojo interpreter
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Kit keeps non-daemon threads after close; the official worker exits hard too.
    os._exit(code)
