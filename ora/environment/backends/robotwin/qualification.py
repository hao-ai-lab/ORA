"""Trusted RoboTwin environment expert/oracle qualification runner.

This module is intentionally outside the policy boundary. It imports the
pinned RoboTwin task implementation, executes ``play_once()``, and records the
official expert-feasibility predicate plus a post-success physics hold. Its
artifacts are qualification evidence, never policy observations or model
demonstrations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
from .asset_manifest import (  # noqa: F401  re-exported for session.py and tests
    ARTICULATED_MODEL_NAMES, TASK_MODEL_ATTRIBUTES, _canonical_json, _sha256_bytes, _sha256_file, _jsonable, _directory_manifest, _rigid_model_manifest, _articulated_variant, _selected_model_ids, selected_environment_assets,
)


SCHEMA = "ora.robotwin-oracle-qualification.v1"
PROCESS_ROTATION_EXIT = 75
DEFAULT_REVISION = "c3ddfa8b97d5519efa828b075999bd0006778e5e"
DEFAULT_TASKS = (
    "blocks_ranking_size",
    "click_alarmclock",
    "open_laptop",
    "place_a2b_right",
    "place_mouse_pad",
    "open_microwave",
    "adjust_bottle",
    "click_bell",
    "move_stapler_pad",
    "place_container_plate",
    "place_empty_cup",
    "place_object_stand",
    "press_stapler",
    "stack_blocks_two",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_dual_shoes",
    "place_fan",
    "place_object_basket",
    "place_object_scale",
    "place_phone_stand",
    "place_shoe",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)
def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=_jsonable,
        ).encode("utf-8")
        + b"\n"
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _git(root: Path, *args: str) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or process.stdout.strip())
    return process.stdout.strip()


def _pose_json(entity: Any) -> dict[str, Any] | None:
    try:
        pose = entity.get_pose()
        return {"p": _jsonable(pose.p), "q": _jsonable(pose.q)}
    except Exception:
        return None


def verifier_telemetry(task_env: Any) -> dict[str, Any]:
    """Capture trusted terminal state for predicate audit and hashing."""

    scene = task_env.scene
    actors: list[dict[str, Any]] = []
    for index, actor in enumerate(scene.get_all_actors()):
        actors.append(
            {
                "index": index,
                "name": actor.get_name(),
                "pose": _pose_json(actor),
            }
        )
    articulations: list[dict[str, Any]] = []
    get_articulations = getattr(scene, "get_all_articulations", None)
    if callable(get_articulations):
        for index, articulation in enumerate(get_articulations()):
            item: dict[str, Any] = {
                "index": index,
                "name": articulation.get_name(),
                "pose": _pose_json(articulation),
            }
            for key, method_name in (("qpos", "get_qpos"), ("qlimits", "get_qlimits")):
                method = getattr(articulation, method_name, None)
                if callable(method):
                    try:
                        item[key] = _jsonable(method())
                    except Exception as exc:
                        item[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
            articulations.append(item)
    return {
        "actors": actors,
        "articulations": articulations,
        "left_gripper_open": bool(task_env.is_left_gripper_open()),
        "right_gripper_open": bool(task_env.is_right_gripper_open()),
    }


@dataclass(frozen=True)
class StabilityResult:
    passed: bool
    checks: int
    first_false_step: int | None


@dataclass
class ProcessAttemptBudget:
    limit: int
    used: int = 0

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    def consume(self) -> None:
        if self.exhausted:
            raise RuntimeError("process attempt budget is already exhausted")
        self.used += 1


def hold_official_predicate(
    task_env: Any,
    *,
    physics_steps: int,
    predicate: Callable[[], bool] | None = None,
) -> StabilityResult:
    """Advance exactly one physics step between official-predicate checks."""

    check = predicate or task_env.check_success
    for step in range(1, physics_steps + 1):
        task_env.scene.step()
        if not bool(check()):
            return StabilityResult(False, step, step)
    return StabilityResult(True, physics_steps, None)


def _save_rgbd(observation: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    import imageio.v2 as imageio
    import numpy as np

    artifacts: list[dict[str, Any]] = []
    cameras = observation.get("observation", {})
    for camera_name in sorted(cameras):
        camera = cameras[camera_name]
        if not isinstance(camera, Mapping):
            continue
        for modality in ("rgb", "depth"):
            if modality not in camera:
                continue
            array = np.asarray(camera[modality])
            npy_path = output / f"{camera_name}-{modality}.npy"
            np.save(npy_path, array, allow_pickle=False)
            artifact = {
                "camera": camera_name,
                "modality": modality,
                "authoritative_path": npy_path.name,
                "sha256": _sha256_file(npy_path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            if modality == "rgb":
                rgb = array
                if rgb.dtype.kind == "f":
                    rgb = np.clip(rgb * (255 if float(np.nanmax(rgb)) <= 1.0 else 1), 0, 255)
                png_path = output / f"{camera_name}-rgb.png"
                imageio.imwrite(png_path, rgb.astype(np.uint8))
                artifact["preview_path"] = png_path.name
                artifact["preview_sha256"] = _sha256_file(png_path)
            artifacts.append(artifact)
    return artifacts


def _load_task_args(
    *,
    robotwin_root: Path,
    task: str,
    task_config: str,
    runtime_output: Path,
    evaluation_mode: bool = True,
) -> dict[str, Any]:
    import yaml

    config_path = robotwin_root / "task_config" / f"{task_config}.yml"
    args = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    args["task_name"] = task
    args["task_config"] = task_config
    args["save_path"] = str(runtime_output)
    args["need_plan"] = True
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = evaluation_mode
    args["is_test"] = evaluation_mode
    args["eval_video_log"] = False
    args["render_freq"] = 0
    args.setdefault("data_type", {})["rgb"] = True
    args["data_type"]["depth"] = True

    from envs._GLOBAL_CONFIGS import CONFIGS_PATH

    embodiment_map = yaml.safe_load(
        (Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text(encoding="utf-8")
    )

    def embodiment_file(name: str) -> str:
        path = embodiment_map[name]["file_path"]
        if not path:
            raise ValueError(f"embodiment {name!r} has no file_path")
        return path

    def embodiment_config(path: str) -> dict[str, Any]:
        return yaml.safe_load((Path(path) / "config.yml").read_text(encoding="utf-8"))

    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    elif len(embodiment) == 3:
        args["left_robot_file"] = embodiment_file(embodiment[0])
        args["right_robot_file"] = embodiment_file(embodiment[1])
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("embodiment must have either one or three entries")
    args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
    return args


def _task_instance(task: str) -> Any:
    module = importlib.import_module(f"envs.{task}")
    return getattr(module, task)()


def _existing_records(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(ledger.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"ledger line {line_number} is not an object")
        records.append(value)
    return records


def _resume_requested_seed(
    records: list[dict[str, Any]], *, start_requested_seed: int = 0
) -> int:
    if not records:
        return start_requested_seed
    last = records[-1]
    if last.get("classification") in {"infra_failure", "invalid_artifact"}:
        return int(last["requested_seed"])
    return max(int(record["requested_seed"]) for record in records) + 1


def _attempt_summary(
    *,
    task: str,
    requested_seed: int,
    accepted_index: int,
    started_at: float,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "task": task,
        "task_config": source["task_config"],
        "requested_seed": requested_seed,
        "accepted_index_if_successful": accepted_index,
        "started_at_unix": started_at,
        "source": dict(source),
        "reset_official_predicate": None,
        "expert": {
            "plan_success": None,
            "post_play_official_predicate": None,
            "eval_success_seen": None,
        },
        "stability": {
            "physics_hz": 250,
            "required_steps": None,
            "checks": 0,
            "first_false_step": None,
            "passed": False,
        },
        "artifacts": [],
        "environment_assets": None,
        "verifier_telemetry": None,
        "classification": "running",
        "accepted": False,
        "error": None,
    }


def _finalize_attempt(record: dict[str, Any], path: Path, ledger: Path) -> None:
    record["finished_at_unix"] = time.time()
    record["duration_seconds"] = record["finished_at_unix"] - record["started_at_unix"]
    record["record_sha256"] = _sha256_bytes(_canonical_json(record))
    _atomic_json(path, record)
    _append_jsonl(ledger, record)


def _raised_in_task_code(exc: BaseException, robotwin_root: Path) -> bool:
    """Whether the innermost frame of ``exc`` is RoboTwin task code (envs/)."""
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return False
    envs = (Path(robotwin_root).resolve() / "envs").as_posix() + "/"
    return Path(frames[-1].filename).resolve().as_posix().startswith(envs)


def _run_task(
    *,
    robotwin_root: Path,
    task: str,
    task_config: str,
    output_root: Path,
    accepted_target: int,
    max_requested_seeds: int,
    start_requested_seed: int,
    stability_steps: int,
    source: Mapping[str, Any],
    process_budget: ProcessAttemptBudget,
) -> dict[str, Any]:
    task_root = output_root / task
    attempts_root = task_root / "attempts"
    ledger = task_root / "seed-ledger.jsonl"
    attempts_root.mkdir(parents=True, exist_ok=True)
    existing = _existing_records(ledger)
    accepted = sum(record.get("accepted") is True for record in existing)
    requested_seed = _resume_requested_seed(
        existing,
        start_requested_seed=start_requested_seed,
    )
    status = {
        "schema": SCHEMA,
        "task": task,
        "accepted_target": accepted_target,
        "accepted_count": accepted,
        "requested_count": len(existing),
        "next_requested_seed": requested_seed,
        "state": "running",
        "updated_at_unix": time.time(),
    }
    _atomic_json(task_root / "status.json", status)

    while accepted < accepted_target and requested_seed < max_requested_seeds:
        if process_budget.exhausted:
            status["state"] = "rotation_required"
            status["updated_at_unix"] = time.time()
            _atomic_json(task_root / "status.json", status)
            return status
        attempt_number = sum(
            int(record["requested_seed"]) == requested_seed for record in existing
        )
        attempt_root = attempts_root / f"seed-{requested_seed:06d}-try-{attempt_number:02d}"
        while attempt_root.exists():
            attempt_number += 1
            attempt_root = attempts_root / f"seed-{requested_seed:06d}-try-{attempt_number:02d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        record = _attempt_summary(
            task=task,
            requested_seed=requested_seed,
            accepted_index=accepted,
            started_at=time.time(),
            source=source,
        )
        record["stability"]["required_steps"] = stability_steps
        task_env = None
        phase = "setup"
        try:
            task_env = _task_instance(task)
            args = _load_task_args(
                robotwin_root=robotwin_root,
                task=task,
                task_config=task_config,
                runtime_output=attempt_root / "runtime",
                evaluation_mode=source.get("evaluation_mode", True),
            )
            task_env.setup_demo(now_ep_num=accepted, seed=requested_seed, **args)
            # Three predicates read state upstream assigns only inside
            # play_once, and this runner polls the predicate at reset before
            # the expert runs. Restoring it here uses the same helper the
            # live session does, so the two paths cannot drift apart.
            from ora.environment.backends.robotwin.session import (
                _restore_expert_only_evaluator_state,
            )

            _restore_expert_only_evaluator_state(task_env, task)
            record["environment_assets"] = selected_environment_assets(
                task_env,
                robotwin_root=robotwin_root,
                task=task,
            )
            reset_predicate = bool(task_env.check_success())
            record["reset_official_predicate"] = reset_predicate
            if reset_predicate:
                record["classification"] = "reset_predicate_true"
            else:
                phase = "oracle"
                task_env.play_once()
                plan_success = bool(task_env.plan_success)
                post_predicate = bool(task_env.check_success())
                record["expert"] = {
                    "plan_success": plan_success,
                    "post_play_official_predicate": post_predicate,
                    "eval_success_seen": bool(getattr(task_env, "eval_success", False)),
                }
                if not plan_success or not post_predicate:
                    record["classification"] = "oracle_failure"
                    # Preserve private failure evidence too. Capture failures
                    # must never turn an official rejection into acceptance,
                    # or conceal the original oracle outcome.
                    try:
                        telemetry_path = attempt_root / "verifier-telemetry.json"
                        _atomic_json(telemetry_path, verifier_telemetry(task_env))
                        record["verifier_telemetry"] = {
                            "path": telemetry_path.name,
                            "sha256": _sha256_file(telemetry_path),
                        }
                        record["artifacts"] = _save_rgbd(task_env.get_obs(), attempt_root)
                    except Exception as exc:
                        record["diagnostic_capture_error"] = f"{type(exc).__name__}: {exc}"
                else:
                    phase = "stability"
                    stability = hold_official_predicate(
                        task_env,
                        physics_steps=stability_steps,
                    )
                    record["stability"].update(
                        {
                            "checks": stability.checks,
                            "first_false_step": stability.first_false_step,
                            "passed": stability.passed,
                        }
                    )
                    # Upstream accepts a seed on ``plan_success and
                    # check_success()`` at the end of the expert
                    # (script/collect_data.py), and Base_Task latches
                    # ``eval_success`` the first physics step the predicate
                    # holds, so a later loss of the state cannot change any
                    # official verdict. Holding the predicate is therefore
                    # recorded as evidence about the task, never a reason to
                    # reject a seed the benchmark itself would collect.
                    phase = "artifact_capture"
                    telemetry = verifier_telemetry(task_env)
                    telemetry_path = attempt_root / "verifier-telemetry.json"
                    _atomic_json(telemetry_path, telemetry)
                    record["verifier_telemetry"] = {
                        "path": telemetry_path.name,
                        "sha256": _sha256_file(telemetry_path),
                    }
                    observation = task_env.get_obs()
                    record["artifacts"] = _save_rgbd(observation, attempt_root)
                    modalities = {
                        (item["camera"], item["modality"])
                        for item in record["artifacts"]
                    }
                    cameras_with_rgb = {camera for camera, modality in modalities if modality == "rgb"}
                    cameras_with_depth = {camera for camera, modality in modalities if modality == "depth"}
                    if not cameras_with_rgb or cameras_with_rgb != cameras_with_depth:
                        record["classification"] = "invalid_artifact"
                        record["error"] = "terminal observation does not contain paired RGB-D cameras"
                    else:
                        record["classification"] = "accepted"
                        record["accepted"] = True
                        accepted += 1
        except Exception as exc:
            if type(exc).__name__ == "UnStableError":
                record["classification"] = "scene_unstable"
            elif phase == "oracle" and _raised_in_task_code(exc, robotwin_root):
                # The expert's own task code raised (e.g. envs/utils/action.py
                # asserting a None grasp pose). Upstream collect_data.py counts
                # any play_once exception as a failed seed and moves on, so the
                # seed is rejected and the scan continues; the traceback stays.
                record["classification"] = "oracle_exception"
            elif phase == "oracle":
                # Raised outside the task code (CUDA, CuRobo, SAPIEN, the
                # Harness): not evidence the seed is infeasible. Stop instead
                # of hiding an infrastructure fault by sampling on.
                record["classification"] = "infra_failure"
            elif phase == "stability":
                record["classification"] = "infra_failure"
            elif phase == "artifact_capture":
                record["classification"] = "invalid_artifact"
            else:
                record["classification"] = "infra_failure"
            record["error"] = f"{type(exc).__name__}: {exc}"
            (attempt_root / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            if task_env is not None:
                try:
                    task_env.close_env(clear_cache=False)
                except Exception as exc:
                    record.setdefault("cleanup_errors", []).append(f"{type(exc).__name__}: {exc}")
            _finalize_attempt(record, attempt_root / "attempt.json", ledger)

        process_budget.consume()
        existing.append(record)
        blocked = record["classification"] in {"infra_failure", "invalid_artifact"}
        if not blocked:
            requested_seed += 1
        status.update(
            {
                "accepted_count": accepted,
                "requested_count": status["requested_count"] + 1,
                "next_requested_seed": requested_seed,
                "last_classification": record["classification"],
                "updated_at_unix": time.time(),
            }
        )
        if blocked:
            status["state"] = (
                "blocked_infrastructure"
                if record["classification"] == "infra_failure"
                else "blocked_invalid_artifact"
            )
        _atomic_json(task_root / "status.json", status)
        if blocked:
            return status

    status["state"] = "passed" if accepted == accepted_target else "failed"
    status["updated_at_unix"] = time.time()
    status["ledger_sha256"] = _sha256_file(ledger)
    _atomic_json(task_root / "status.json", status)
    return status


def _source_manifest(
    *,
    robotwin_root: Path,
    task_config: str,
    tasks: Iterable[str],
    required_revision: str,
) -> dict[str, Any]:
    actual_revision = _git(robotwin_root, "rev-parse", "HEAD")
    if actual_revision != required_revision:
        raise RuntimeError(
            f"RoboTwin revision mismatch: expected {required_revision}, got {actual_revision}"
        )
    paths = [f"envs/{task}.py" for task in tasks]
    tracked_diff = _git(robotwin_root, "diff", "--name-only", required_revision, "--", *paths)
    if tracked_diff:
        raise RuntimeError(f"selected task sources have tracked modifications: {tracked_diff}")
    task_sources = {
        task: {
            "path": f"envs/{task}.py",
            "sha256": _sha256_file(robotwin_root / "envs" / f"{task}.py"),
        }
        for task in tasks
    }
    config_path = robotwin_root / "task_config" / f"{task_config}.yml"
    return {
        "robotwin_revision": actual_revision,
        "task_config": task_config,
        "task_config_path": str(config_path.relative_to(robotwin_root)),
        "task_config_sha256": _sha256_file(config_path),
        "qualification_override": {"data_type.rgb": True, "data_type.depth": True},
        "task_sources": task_sources,
    }


def run(args: argparse.Namespace) -> int:
    robotwin_root = args.robotwin_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not (robotwin_root / "envs").is_dir():
        raise SystemExit(f"invalid RoboTwin root: {robotwin_root}")
    unknown = sorted(set(args.tasks) - set(DEFAULT_TASKS))
    if unknown:
        raise SystemExit(
            f"tasks are outside the registered qualification set: {unknown}"
        )
    if str(robotwin_root) not in sys.path:
        sys.path.insert(0, str(robotwin_root))
    curobo = robotwin_root / "envs" / "curobo" / "src"
    if str(curobo) not in sys.path:
        sys.path.insert(0, str(curobo))

    source = _source_manifest(
        robotwin_root=robotwin_root,
        task_config=args.task_config,
        tasks=args.tasks,
        required_revision=args.required_revision,
    )
    source["evaluation_mode"] = args.scene_mode == "evaluation"
    source["background_split"] = "unseen" if source["evaluation_mode"] else "seen"
    source["qualification_override"].update({
        "eval_mode": source["evaluation_mode"],
        "is_test": source["evaluation_mode"],
        "eval_video_log": False,
    })
    run_manifest = {
        "schema": SCHEMA,
        "created_at_unix": time.time(),
        "host": platform.node(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "runner_sha256": _sha256_file(Path(__file__).resolve()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tasks": args.tasks,
        "accepted_target_per_task": args.accepted_per_task,
        "max_requested_seeds_per_task": args.max_requested_seeds,
        "start_requested_seed": args.start_requested_seed,
        "terminal_stability_physics_hz": 250,
        "terminal_stability_physics_steps": args.stability_steps,
        "source": source,
        "policy_boundary": "no policy or model is invoked by this trusted qualification runner",
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run-manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable_keys = (
            "schema",
            "host",
            "python",
            "python_version",
            "runner_sha256",
            "cuda_visible_devices",
            "tasks",
            "accepted_target_per_task",
            "max_requested_seeds_per_task",
            "start_requested_seed",
            "terminal_stability_physics_hz",
            "terminal_stability_physics_steps",
            "source",
            "policy_boundary",
        )
        mismatched = [
            key for key in stable_keys if existing_manifest.get(key) != run_manifest.get(key)
        ]
        if mismatched:
            raise RuntimeError(f"existing run manifest mismatch: {mismatched}")
    else:
        _atomic_json(manifest_path, run_manifest)
    process_budget = ProcessAttemptBudget(args.process_attempt_limit)
    statuses = []
    for task in args.tasks:
        task_source = {
            **source,
            "task_source": source["task_sources"][task],
        }
        task_status = _run_task(
            robotwin_root=robotwin_root,
            task=task,
            task_config=args.task_config,
            output_root=output_root,
            accepted_target=args.accepted_per_task,
            max_requested_seeds=args.max_requested_seeds,
            start_requested_seed=args.start_requested_seed,
            stability_steps=args.stability_steps,
            source=task_source,
            process_budget=process_budget,
        )
        statuses.append(task_status)
        if str(task_status["state"]).startswith("blocked_") or task_status[
            "state"
        ] == "rotation_required":
            break
    summary = {
        "schema": SCHEMA,
        "tasks": statuses,
        "passed": len(statuses) == len(args.tasks)
        and all(status["state"] == "passed" for status in statuses),
        "accepted_total": sum(status["accepted_count"] for status in statuses),
        "accepted_required": args.accepted_per_task * len(args.tasks),
        "finished_at_unix": time.time(),
    }
    summary["summary_sha256"] = _sha256_bytes(_canonical_json(summary))
    _atomic_json(output_root / "summary.json", summary)
    if any(status["state"] == "rotation_required" for status in statuses):
        return PROCESS_ROTATION_EXIT
    return 0 if summary["passed"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="robotwin-qualify", description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=DEFAULT_TASKS, required=True)
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument("--scene-mode", choices=("evaluation", "collection"), default="evaluation",
                        help="evaluation uses the official unseen background split; collection is explicit legacy compatibility")
    parser.add_argument("--required-revision", default=DEFAULT_REVISION)
    parser.add_argument("--accepted-per-task", type=int, default=5)
    parser.add_argument("--max-requested-seeds", type=int, default=50)
    parser.add_argument("--start-requested-seed", type=int, default=0)
    parser.add_argument("--stability-steps", type=int, default=250)
    parser.add_argument("--process-attempt-limit", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.accepted_per_task <= 0:
        raise SystemExit("--accepted-per-task must be positive")
    if args.max_requested_seeds < args.accepted_per_task:
        raise SystemExit("--max-requested-seeds must be >= --accepted-per-task")
    if args.start_requested_seed < 0:
        raise SystemExit("--start-requested-seed must be non-negative")
    if args.max_requested_seeds <= args.start_requested_seed:
        raise SystemExit("--max-requested-seeds must be greater than --start-requested-seed")
    if args.stability_steps <= 0:
        raise SystemExit("--stability-steps must be positive")
    if args.process_attempt_limit <= 0:
        raise SystemExit("--process-attempt-limit must be positive")
    result = run(args)
    if result == PROCESS_ROTATION_EXIT:
        arguments = list(sys.argv[1:] if argv is None else argv)
        sys.stdout.flush()
        sys.stderr.flush()
        # Re-enter as a module, as the parent launched it: the file imports
        # sibling modules relatively, which fails when run as a bare script.
        module = __spec__.name if __spec__ else "ora.environment.backends.robotwin.qualification"
        os.execv(sys.executable, [sys.executable, "-m", module, *arguments])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
