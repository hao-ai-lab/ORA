"""Standalone exact-source worker for historical LIBERO semantic re-enactment.

This file is executed directly while ``PYTHONPATH`` points at an extracted
historical Harness commit.  Keep imports of ``ora`` inside ``run`` so
the historical implementation, rather than the current checkout, owns action
validation and controller behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any, Mapping


JOB_SCHEMA = "ora.viewer-historical-libero-reenactment-job.v1"
RESULT_SCHEMA = "ora.viewer-historical-libero-reenactment.v1"
FRAME_SCHEMA = "ora.viewer-historical-libero-reenactment-frame.v1"
CAPTURE_SEMANTICS = (
    "derived_exact_harness_source_semantic_action_reenactment_action_keyframes"
)


class HistoricalReplayError(RuntimeError):
    """The historical semantic action stream could not be re-enacted safely."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise HistoricalReplayError(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def _uniform_capture_steps(step_count: int, frame_limit: int) -> list[int]:
    if step_count <= 0:
        return []
    if frame_limit <= 0:
        raise HistoricalReplayError("frame limit must be positive")
    if frame_limit == 1:
        return [step_count - 1]
    if step_count <= frame_limit:
        return list(range(step_count))
    denominator = frame_limit - 1
    return sorted({
        (index * (step_count - 1) + denominator // 2) // denominator
        for index in range(frame_limit)
    })


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _contact_sheet(images: Mapping[str, Any]) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    canvas = Image.new("RGB", (1280, 720), "#101413")
    layout = {
        "agentview": (0, 0, 640, 720),
        "wrist": (640, 0, 1280, 720),
    }
    draw = ImageDraw.Draw(canvas)
    for name, bounds in layout.items():
        left, top, right, bottom = bounds
        draw.rectangle((left, top, right - 1, bottom - 1), outline="#50605a")
        value = images.get(name)
        if value is not None:
            array = np.asarray(value)
            if array.ndim == 3 and array.shape[2] >= 3:
                source = Image.fromarray(
                    np.clip(array[..., :3], 0, 255).astype(np.uint8), "RGB",
                )
                rendered = ImageOps.contain(
                    source, (right - left, bottom - top), Image.Resampling.LANCZOS,
                )
                canvas.paste(
                    rendered,
                    (
                        left + (right - left - rendered.width) // 2,
                        top + (bottom - top - rendered.height) // 2,
                    ),
                )
        draw.rectangle((left, top, left + 190, top + 32), fill="#101413")
        draw.text(
            (left + 8, top + 6), name.upper(), font=_font(18, bold=True),
            fill="#f2f0e8",
        )
    return canvas


def _libero_images(session: Any) -> dict[str, Any]:
    import numpy as np

    raw = session._raw_observation()
    return {
        "agentview": np.array(raw["agentview_image"], copy=True)[::-1],
        "wrist": np.array(raw["robot0_eye_in_hand_image"], copy=True)[::-1],
    }


def _receipt_endpoint(receipt: Mapping[str, Any]) -> Mapping[str, Any]:
    after = receipt.get("after")
    if isinstance(after, Mapping):
        return after
    phases = receipt.get("phase_receipts")
    if isinstance(phases, list):
        for phase in reversed(phases):
            nested = phase.get("receipt") if isinstance(phase, Mapping) else None
            nested_after = nested.get("after") if isinstance(nested, Mapping) else None
            if isinstance(nested_after, Mapping):
                return nested_after
    return {}


def _receipt_action_name(receipt: Mapping[str, Any]) -> Any:
    action = receipt.get("action")
    return action.get("action") if isinstance(action, Mapping) else action


def _receipt_steps_used(receipt: Mapping[str, Any]) -> int | None:
    direct = receipt.get("steps_used")
    if isinstance(direct, int) and not isinstance(direct, bool):
        return direct
    phases = receipt.get("phase_receipts")
    if not isinstance(phases, list):
        return None
    values: list[int] = []
    for phase in phases:
        nested = phase.get("receipt") if isinstance(phase, Mapping) else None
        value = nested.get("steps_used") if isinstance(nested, Mapping) else None
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        values.append(value)
    return sum(values) if values else None


def _target_state_phases(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Use the exact archived action compiler across its released layouts."""

    try:
        from ora.contracts.action import target_state_phases
    except ModuleNotFoundError as error:
        if error.name not in {
            "ora.contracts",
            "ora.contracts.action",
        }:
            raise
        legacy_action = importlib.import_module(".".join(("environment", "action")))
        target_state_phases = legacy_action.target_state_phases
    return target_state_phases(payload)


class _Writer:
    def __init__(self, root: Path, sampling: Mapping[str, Any]) -> None:
        self.root = root
        self.frames = root / "frames"
        self.frames.mkdir(parents=True, exist_ok=True)
        self.timeline = root / "timeline.jsonl"
        self.sampling = dict(sampling)
        self.sequence = 0

    def append(
        self, images: Mapping[str, Any], *, entry: Mapping[str, Any] | None,
        phase: str, control_step: int | None, round_index: int,
    ) -> None:
        path = self.frames / f"frame-{self.sequence:06d}.jpg"
        _contact_sheet(images).save(path, format="JPEG", quality=88, subsampling=1)
        record = {
            "schema": FRAME_SCHEMA,
            "frame_index": self.sequence,
            "path": path.relative_to(self.root).as_posix(),
            "action_sequence": (
                entry.get("action_sequence") if entry is not None else None
            ),
            "round_action_sequence": (
                entry.get("round_action_sequence") if entry is not None else None
            ),
            "round_action_count": (
                entry.get("round_action_count") if entry is not None else None
            ),
            "round_index": round_index,
            "round_count": (
                entry.get("round_count") if entry is not None
                else None
            ),
            "action_id": (
                entry.get("action", {}).get("action_id")
                if entry is not None and isinstance(entry.get("action"), Mapping)
                else None
            ),
            "action": (
                entry.get("action", {}).get("action", "mutation")
                if entry is not None and isinstance(entry.get("action"), Mapping)
                else "reset"
            ),
            "phase": phase,
            "control_step_index": control_step,
            "sampling": self.sampling,
            "cameras": ["agentview", "wrist"],
            "recorded_execution": (
                {
                    "status": "executed",
                    "steps_used": _receipt_steps_used(
                        entry.get("recorded_receipt", {}),
                    ),
                }
                if entry is not None else None
            ),
            "capture_semantics": CAPTURE_SEMANTICS,
        }
        with self.timeline.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.sequence += 1


def _quaternion_error_rad(left: list[float], right: list[float]) -> float:
    import numpy as np

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm <= 0:
        return math.inf
    dot = min(1.0, max(-1.0, abs(float(np.dot(a, b)) / norm)))
    return 2.0 * math.acos(dot)


def _validate_reenacted_receipt(
    recorded: Mapping[str, Any], actual: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    recorded_after = _receipt_endpoint(recorded).get("right", {})
    actual_after = _receipt_endpoint(actual).get("right", {})
    recorded_pose = recorded_after.get("pose_xyz_xyzw")
    actual_pose = actual_after.get("pose_xyz_xyzw")
    if not (
        isinstance(recorded_pose, list) and len(recorded_pose) == 7
        and isinstance(actual_pose, list) and len(actual_pose) == 7
    ):
        raise HistoricalReplayError("receipt omits comparable right-arm pose")
    translation_m = float(np.linalg.norm(
        np.asarray(recorded_pose[:3]) - np.asarray(actual_pose[:3])
    ))
    orientation_rad = _quaternion_error_rad(recorded_pose[3:], actual_pose[3:])
    recorded_gripper = recorded_after.get("gripper")
    actual_gripper = actual_after.get("gripper")
    gripper = (
        abs(float(recorded_gripper) - float(actual_gripper))
        if isinstance(recorded_gripper, (int, float))
        and isinstance(actual_gripper, (int, float))
        else math.inf
    )
    valid = (
        recorded.get("action_id") == actual.get("action_id")
        and _receipt_action_name(recorded) == _receipt_action_name(actual)
        and translation_m <= float(thresholds["translation_m"])
        and orientation_rad <= float(thresholds["orientation_rad"])
        and gripper <= float(thresholds["gripper_normalized"])
    )
    return {
        "valid": valid,
        "translation_m": translation_m,
        "orientation_rad": orientation_rad,
        "gripper_normalized": gripper,
        "recorded_steps_used": _receipt_steps_used(recorded),
        "reenacted_steps_used": _receipt_steps_used(actual),
    }


def _archive_round_verdict(session: Any, output_root: Path, round_index: int) -> None:
    verdict = output_root / "runtime/official-verdict.json"
    if not verdict.exists():
        persist = getattr(session, "persist_official_verdict", None)
        if callable(persist):
            persist()
    if verdict.exists():
        target = output_root / f"runtime/rounds/round-{round_index:04d}/official-verdict.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(verdict, target)


def _validate_semantic_actions(actions: list[Any]) -> None:
    """Reject unsupported control semantics before constructing a simulator.

    New Servo requires the issued controller prefix and its non-advancing hold
    events. Re-solving its relative request through historical Session code
    cannot reproduce that evidence; ordinary endpoint comparison is insufficient.
    The older ``servo_delta`` motion action keeps its existing archived path.
    """

    for entry in actions:
        payload = entry.get("action") if isinstance(entry, Mapping) else None
        if not isinstance(payload, Mapping):
            raise HistoricalReplayError("historical replay action is malformed")
        if (payload.get("schema") == "ora-servo-action.v1"
                or payload.get("action") == "servo"):
            raise HistoricalReplayError(
                "new Servo requires supported recorded-controller replay; "
                "historical semantic re-enactment is unavailable"
            )


def run(job_path: Path) -> dict[str, Any]:
    job = _load(job_path)
    if job.get("schema") != JOB_SCHEMA:
        raise HistoricalReplayError("invalid historical LIBERO replay job")
    output_root = Path(str(job["output_root"])).resolve()
    run_root = Path(str(job["run_root"])).resolve()
    if output_root == run_root or output_root.is_relative_to(run_root):
        raise HistoricalReplayError("derived replay output must remain outside the run")
    plan = job.get("plan")
    runtime_paths = job.get("runtime_paths")
    actions = job.get("actions")
    sampling = job.get("sampling")
    thresholds = job.get("validation_thresholds")
    if not all(isinstance(value, Mapping) for value in (plan, runtime_paths, sampling, thresholds)):
        raise HistoricalReplayError("historical replay job metadata is malformed")
    if not isinstance(actions, list):
        raise HistoricalReplayError("historical replay actions are malformed")
    if sampling.get("mode") != "action_keyframes_v1":
        raise HistoricalReplayError("historical replay requires action-keyframe sampling")
    limit = sampling.get("max_frames_per_action")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 32:
        raise HistoricalReplayError("historical replay frame limit is invalid")
    _validate_semantic_actions(actions)

    # PYTHONPATH is bound by the host to an extracted exact historical commit.
    try:
        from ora.environment.backends.libero_pro.session import LiberoProSession
    except ModuleNotFoundError as error:
        # Released pre-namespace Harness commits owned the same backend below
        # ``environment.*``.  PYTHONPATH is already bound to the exact archived
        # source, so this is source-layout compatibility rather than a current
        # implementation fallback.
        if error.name not in {
            "ora.environment",
            "ora.environment.backends",
            "ora.environment.backends.libero_pro",
        }:
            raise
        legacy_session = importlib.import_module(".".join((
            "environment", "backends", "libero_pro", "session",
        )))
        LiberoProSession = legacy_session.LiberoProSession

    runtime = plan.get("runtime")
    budgets = plan.get("budgets")
    if not isinstance(runtime, Mapping) or not isinstance(budgets, Mapping):
        raise HistoricalReplayError("run plan omits Runtime or budgets")
    candidates = {
        "rlinf_root": (Path(str(runtime_paths["rlinf_root"]))
                       if runtime.get("environment_driver", "rlinf_v1") == "rlinf_v1" else None),
        "environment_driver": runtime.get("environment_driver", "rlinf_v1"),
        "libero_pro_root": Path(str(runtime_paths["libero_pro_root"])),
        "data_root": Path(str(runtime_paths["data_root"])),
        "output_root": output_root.with_name(output_root.name + ".tmp") / "runtime",
        "task": str(plan["task"]),
        "seed": int(plan["seed"]),
        "suite": str(runtime["suite"]),
        "task_id": int(runtime["task_id"]),
        "expected_task_language": str(runtime["expected_task_language"]),
        "required_rlinf_revision": runtime.get("required_rlinf_revision"),
        "required_libero_pro_revision": str(runtime["required_libero_pro_revision"]),
        "required_data_revision": str(runtime["required_data_revision"]),
        "workspace_bounds_m": runtime["workspace_bounds_m"],
        "workspace_profile": runtime.get("workspace_profile"),
        "max_sim_steps": int(budgets["max_sim_steps"]),
        "servo_max_steps": int(runtime.get("servo_max_steps", 80)),
        "qualified_controller_binding": job.get("qualified_controller_binding"),
    }
    accepted = inspect.signature(LiberoProSession).parameters
    session = LiberoProSession(**{
        name: value for name, value in candidates.items() if name in accepted
    })
    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    writer = _Writer(temporary, sampling)
    validations: list[dict[str, Any]] = []
    current_round = 0
    try:
        session.start()
        writer.append(
            _libero_images(session), entry=None, phase="reset",
            control_step=None, round_index=0,
        )
        for entry in actions:
            if not isinstance(entry, Mapping):
                raise HistoricalReplayError("historical replay action is malformed")
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                if round_index != current_round + 1:
                    raise HistoricalReplayError("round sequence is not contiguous")
                _archive_round_verdict(session, temporary, current_round)
                restart = getattr(session, "restart_round", None)
                if not callable(restart):
                    raise HistoricalReplayError("historical Session cannot restart rounds")
                restart(round_index)
                current_round = round_index
                writer.append(
                    _libero_images(session), entry=None, phase="round_reset",
                    control_step=None, round_index=round_index,
                )
            environment = session._require_environment()
            original_step = environment.step
            snapshots: list[dict[str, Any]] = []

            def captured_step(command: Any) -> Any:
                result = original_step(command)
                snapshots.append(_libero_images(session))
                return result

            environment.step = captured_step
            try:
                payload = entry["action"]
                if payload.get("schema") == "ora-action.v4":
                    phase_receipts = []
                    for phase in _target_state_phases(payload):
                        phase_receipt = session.execute(phase)
                        phase_receipts.append({
                            "phase": (
                                "pose" if phase.get("action") == "move_pose"
                                else "gripper"
                            ),
                            "status": "reached",
                            "receipt": phase_receipt,
                        })
                    actual_receipt = {
                        "action_id": payload.get("action_id"),
                        "action": payload,
                        "phase_receipts": phase_receipts,
                    }
                else:
                    actual_receipt = session.execute(payload)
            finally:
                environment.step = original_step
            validation = _validate_reenacted_receipt(
                entry["recorded_receipt"], actual_receipt, thresholds,
            )
            validation["action_sequence"] = entry.get("action_sequence")
            validation["action_id"] = entry.get("action", {}).get("action_id")
            validations.append(validation)
            if not validation["valid"]:
                raise HistoricalReplayError(
                    "semantic re-enactment diverged at action "
                    f"{validation['action_id']}: {validation}"
                )
            if not snapshots:
                snapshots.append(_libero_images(session))
            selected = _uniform_capture_steps(len(snapshots), limit)
            for ordinal, control_step in enumerate(selected):
                writer.append(
                    snapshots[control_step], entry=entry,
                    phase=("control_final" if ordinal == len(selected) - 1 else "control"),
                    control_step=control_step, round_index=round_index,
                )
        result = {
            "schema": RESULT_SCHEMA,
            "identity": job.get("identity"),
            "backend": "libero_pro",
            "source_run_id": run_root.name,
            "historical_harness_commit": job.get("historical_harness_commit"),
            "source_events": job.get("source_events"),
            "frame_count": writer.sequence,
            "sampling": dict(sampling),
            "camera_panels": ["agentview", "wrist"],
            "round_count": int(job.get("round_count", 1)),
            "viewer_resource": job.get("viewer_resource"),
            "timeline": {
                "path": "timeline.jsonl",
                "sha256": _hash(writer.timeline),
            },
            "capture_semantics": CAPTURE_SEMANTICS,
            "validation": {
                "status": "valid",
                "thresholds": dict(thresholds),
                "actions": validations,
            },
            "evidence_boundary": (
                "viewer_derived_exact_historical_action_and_controller_code_"
                "regenerated_physics_not_original_visual_or_verdict_evidence"
            ),
        }
        _atomic_json(temporary / "manifest.json", result)
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
        return result
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-enact historical LIBERO semantic actions for Viewer media."
    )
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.job.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
