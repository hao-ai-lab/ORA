"""Isolated simulator worker for Viewer-derived controller replay videos."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


JOB_SCHEMA = "ora.viewer-controller-replay-job.v2"
RESULT_SCHEMA = "ora.viewer-controller-replay.v2"
FRAME_SCHEMA = "ora.viewer-controller-replay-frame.v2"
ROBOTWIN_DENSE_TARGET_CONTROLLERS = frozenset({
    "robotwin_dense_joint_targets",
    "robotwin_local_cartesian_dls",
    "robotwin_global_then_local_settle",
    "robotwin_global_precontact_then_local",
})


class ReplayWorkerError(RuntimeError):
    """A derived simulator replay could not be compiled safely."""


def _uniform_capture_steps(step_count: int, frame_limit: int) -> frozenset[int]:
    """Choose a bounded, endpoint-preserving review sample for one action."""

    if step_count <= 0:
        return frozenset()
    if frame_limit <= 0:
        raise ReplayWorkerError("Viewer replay frame limit must be positive")
    if frame_limit == 1:
        return frozenset({step_count - 1})
    if step_count <= frame_limit:
        return frozenset(range(step_count))
    denominator = frame_limit - 1
    return frozenset(
        (index * (step_count - 1) + denominator // 2) // denominator
        for index in range(frame_limit)
    )


def _capture_steps(step_count: int, sampling: Mapping[str, Any]) -> frozenset[int]:
    """Resolve one closed replay sampling profile to global action-step indices."""

    mode = sampling.get("mode")
    if mode == "action_keyframes_v1":
        frame_limit = sampling.get("max_frames_per_action")
        hold_steps = sampling.get("hold_steps_per_sample")
        if (
            isinstance(frame_limit, bool)
            or not isinstance(frame_limit, int)
            or not 1 <= frame_limit <= 32
        ):
            raise ReplayWorkerError(
                "Viewer replay max_frames_per_action must be in [1, 32]"
            )
        if (
            isinstance(hold_steps, bool)
            or not isinstance(hold_steps, int)
            or not 1 <= hold_steps <= 64
        ):
            raise ReplayWorkerError(
                "Viewer replay hold_steps_per_sample must be in [1, 64]"
            )
        return _uniform_capture_steps(step_count, frame_limit)
    if mode == "physics_stride_v1":
        stride = sampling.get("capture_stride")
        if (
            isinstance(stride, bool)
            or not isinstance(stride, int)
            or not 1 <= stride <= 64
        ):
            raise ReplayWorkerError("Viewer replay capture stride must be in [1, 64]")
        return frozenset(range(0, step_count, stride))
    raise ReplayWorkerError(f"unsupported Viewer replay sampling mode: {mode}")


def _needs_final_capture(step_count: int, last_captured_step: int | None) -> bool:
    """Keep the exact end state when periodic capture did not include it."""

    return step_count > 0 and last_captured_step != step_count - 1


def _robotwin_replay_route(controller: Any) -> str:
    """Resolve one closed recorded-controller label to its replay executor."""

    if isinstance(controller, str) and controller in ROBOTWIN_DENSE_TARGET_CONTROLLERS:
        return "dense_targets"
    if controller == "robotwin_ee_action":
        return "ee_action"
    raise ReplayWorkerError(
        f"unsupported RoboTwin replay controller: {controller}"
    )


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ReplayWorkerError(f"expected JSON object: {path}")
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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _required_environment_path(runtime: Mapping[str, Any], field: str) -> Path:
    name = runtime.get(field)
    if not isinstance(name, str) or not name:
        raise ReplayWorkerError(f"Runtime omits {field}")
    value = os.environ.get(name)
    if not value:
        raise ReplayWorkerError(f"required replay environment variable is unset: {name}")
    return Path(value).expanduser().resolve()


def _artifact_arrays(
    manifest_path: Path, entry: Mapping[str, Any],
) -> dict[str, Any]:
    import numpy as np

    artifact = entry.get("artifact")
    if not isinstance(artifact, Mapping) or not isinstance(artifact.get("path"), str):
        return {}
    run_native_root = manifest_path.parent.parent.resolve()
    target = (run_native_root / str(artifact["path"])).resolve()
    if not target.is_relative_to(run_native_root) or not target.is_file():
        raise ReplayWorkerError("controller replay artifact escaped the run")
    if _hash(target) != artifact.get("sha256"):
        raise ReplayWorkerError("controller replay artifact hash drifted")
    with np.load(target, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        return ImageFont.load_default()


def _contact_sheet(images: Mapping[str, Any], cameras: list[str]) -> Any:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    if cameras == [
        "observer_camera", "head_camera", "front_camera",
        "left_camera", "right_camera",
    ]:
        canvas = Image.new("RGB", (1280, 960), "#101413")
        layout = {
            "observer_camera": (0, 0, 427, 480),
            "head_camera": (427, 0, 853, 480),
            "front_camera": (853, 0, 1280, 480),
            "left_camera": (0, 480, 640, 960),
            "right_camera": (640, 480, 1280, 960),
        }
    else:
        canvas = Image.new("RGB", (1280, 720), "#101413")
        width = 1280 // max(1, len(cameras))
        layout = {
            name: (index * width, 0, 1280 if index == len(cameras) - 1 else (index + 1) * width, 720)
            for index, name in enumerate(cameras)
        }
    draw = ImageDraw.Draw(canvas)
    for name in cameras:
        left, top, right, bottom = layout[name]
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
                x = left + (right - left - rendered.width) // 2
                y = top + (bottom - top - rendered.height) // 2
                canvas.paste(rendered, (x, y))
        draw.rectangle((left, top, min(right, left + 230), top + 30), fill="#101413")
        draw.text((left + 8, top + 6), name.upper(), font=_font(18, bold=True), fill="#f2f0e8")
    return canvas


class _FrameWriter:
    def __init__(
        self, root: Path, *, cameras: list[str], sampling: Mapping[str, Any],
    ) -> None:
        self.root = root
        self.frames_root = root / "frames"
        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.timeline = root / "timeline.jsonl"
        self.cameras = cameras
        self.sampling = dict(sampling)
        self.sequence = 0

    def append(
        self, images: Mapping[str, Any], *, entry: Mapping[str, Any] | None,
        phase: str, control_step: int | None,
    ) -> None:
        path = self.frames_root / f"frame-{self.sequence:06d}.jpg"
        _contact_sheet(images, self.cameras).save(
            path, format="JPEG", quality=88, subsampling=1,
        )
        record = {
            "schema": FRAME_SCHEMA,
            "frame_index": self.sequence,
            "path": path.relative_to(self.root).as_posix(),
            "action_sequence": (
                entry.get("action_sequence") if entry is not None else None
            ),
            "action_id": entry.get("action_id") if entry is not None else None,
            "action": entry.get("action") if entry is not None else "reset",
            "phase": phase,
            "control_step_index": control_step,
            "sampling": self.sampling,
            "cameras": self.cameras,
            "recorded_execution": (
                dict(entry["execution"])
                if entry is not None and isinstance(entry.get("execution"), Mapping)
                else None
            ),
            "capture_semantics": (
                "derived_controller_replay_action_keyframes"
                if self.sampling.get("mode") == "action_keyframes_v1"
                else "derived_controller_replay_after_simulator_step"
                if self.sampling.get("capture_stride") == 1
                else "derived_controller_replay_temporally_subsampled"
            ),
        }
        with self.timeline.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.sequence += 1


def _robotwin_images(environment: Any) -> dict[str, Any]:
    manager = getattr(environment, "cameras", None)
    update = getattr(manager, "update_picture", None)
    get_rgb = getattr(manager, "get_rgb", None)
    get_observer = getattr(manager, "get_observer_rgb", None)
    if not all(callable(item) for item in (update, get_rgb, get_observer)):
        raise ReplayWorkerError("RoboTwin replay cameras are unavailable")
    update()
    raw = get_rgb()
    if not isinstance(raw, Mapping):
        raise ReplayWorkerError("RoboTwin replay RGB result is malformed")
    result = {
        str(name): value.get("rgb")
        for name, value in raw.items()
        if isinstance(value, Mapping) and value.get("rgb") is not None
    }
    result["observer_camera"] = get_observer()
    return result


def _robotwin_control_sequence(
    entry: Mapping[str, Any], arrays: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = entry.get("metadata")
    structure = (
        metadata.get("control_sequence") if isinstance(metadata, Mapping) else None
    )
    if not isinstance(structure, Mapping):
        raise ReplayWorkerError("RoboTwin replay control structure is unavailable")
    result: dict[str, Any] = {}
    for channel, value in structure.items():
        if value is None:
            result[str(channel)] = None
            continue
        if not isinstance(value, Mapping):
            raise ReplayWorkerError("RoboTwin replay channel is malformed")
        compiled: dict[str, Any] = {}
        for field, item in value.items():
            if isinstance(item, Mapping) and isinstance(item.get("array"), str):
                key = str(item["array"])
                if key not in arrays:
                    raise ReplayWorkerError(f"RoboTwin replay array is missing: {key}")
                compiled[str(field)] = arrays[key]
            else:
                compiled[str(field)] = item
        result[str(channel)] = compiled
    return result


def _recorded_control_step_count(
    entry: Mapping[str, Any], arrays: Mapping[str, Any],
) -> int:
    """Recover the exact issued-step count across current and legacy traces."""

    metadata = entry.get("metadata")
    tracking = (
        metadata.get("controller_tracking")
        if isinstance(metadata, Mapping) else None
    )
    if isinstance(tracking, Mapping):
        value = tracking.get("physics_step_attempts")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    structure = (
        metadata.get("control_sequence")
        if isinstance(metadata, Mapping) else None
    )
    if not isinstance(structure, Mapping):
        return 0
    counts: list[int] = []
    for channel in structure.values():
        if not isinstance(channel, Mapping):
            continue
        num_step = channel.get("num_step")
        if isinstance(num_step, int) and not isinstance(num_step, bool):
            counts.append(max(0, num_step))
        for field in ("position", "result", "command"):
            binding = channel.get(field)
            key = binding.get("array") if isinstance(binding, Mapping) else None
            value = arrays.get(key) if isinstance(key, str) else None
            shape = getattr(value, "shape", ())
            if shape and isinstance(shape[0], int):
                counts.append(int(shape[0]))
    return max(counts, default=0)


def _robotwin_replay_needs_planner(manifest: Mapping[str, Any]) -> bool:
    """Return whether any replayable entry must reconstruct a legacy EE action."""

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ReplayWorkerError("RoboTwin replay entries are malformed")
    return any(
        isinstance(entry, Mapping)
        and bool(entry.get("replayable"))
        and _robotwin_replay_route(entry.get("controller")) == "ee_action"
        for entry in entries
    )


def _disable_unused_robotwin_planners(robotwin_root: Path) -> tuple[Any, Any] | None:
    """Avoid constructing CuRobo when replay consumes recorded dense targets.

    RoboTwin constructs two planners during ``setup_demo`` even though dense
    replay never calls either planner.  The Viewer worker is isolated and
    source-bound, so replacing only that constructor hook is safe and removes
    the dominant cold-start cost without modifying the pinned source tree.
    """

    root_text = str(robotwin_root)
    curobo_text = str(robotwin_root / "envs/curobo/src")
    for path in (root_text, curobo_text):
        if path not in sys.path:
            sys.path.insert(0, path)
    previous_working_directory = Path.cwd().resolve()
    try:
        os.chdir(robotwin_root)
        module = importlib.import_module("envs.robot.robot")
    finally:
        os.chdir(previous_working_directory)
    robot_class = getattr(module, "Robot", None)
    original = getattr(robot_class, "set_planner", None)
    if robot_class is None or not callable(original):
        raise ReplayWorkerError("RoboTwin replay planner hook is unavailable")

    class SetupOnlyGripperPlanner:
        """Minimal deterministic helper required by RoboTwin reset."""

        @staticmethod
        def plan_grippers(now_val: float, target_val: float) -> dict[str, Any]:
            num_step = 200
            return {
                "num_step": num_step,
                "per_step": (target_val - now_val) / num_step,
                "result": module.np.linspace(now_val, target_val, num_step),
            }

    def recorded_target_only(self: Any, scene: Any = None) -> None:
        del scene
        self.communication_flag = False
        self.left_planner = SetupOnlyGripperPlanner()
        self.right_planner = SetupOnlyGripperPlanner()

    robot_class.set_planner = recorded_target_only
    return robot_class, original


def _apply_robotwin_control_target(
    environment: Any, control_sequence: Mapping[str, Any], control_index: int,
) -> None:
    """Apply one recorded dense target without invoking a planner."""

    def length(value: Any) -> int:
        shape = getattr(value, "shape", ())
        return int(shape[0]) if shape and isinstance(shape[0], int) else 0

    robot = environment.robot
    for arm in ("left", "right"):
        arm_channel = control_sequence.get(f"{arm}_arm")
        if isinstance(arm_channel, Mapping):
            position = arm_channel.get("position")
            velocity = arm_channel.get("velocity")
            if (
                length(position) > control_index
                and length(velocity) > control_index
            ):
                robot.set_arm_joints(
                    position[control_index], velocity[control_index], arm,
                )
        gripper = control_sequence.get(f"{arm}_gripper")
        if isinstance(gripper, Mapping):
            num_step = gripper.get("num_step")
            result = gripper.get("result")
            if (
                isinstance(num_step, int)
                and control_index < num_step
                and length(result) > control_index
            ):
                robot.set_gripper(
                    result[control_index], arm, gripper.get("per_step"),
                )


def _replay_robotwin(
    *, plan: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, project_root: Path, output_root: Path,
    sampling: Mapping[str, Any],
) -> tuple[int, list[str]]:
    from ora.environment.backends.robotwin.session import RoboTwinLiveSession
    from ora.environment.controllers.embodiment_control import (
        load_embodiment_control_profile,
    )

    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ReplayWorkerError("run omits Runtime configuration")
    binding = runtime.get("control_profile")
    control_profile = None
    if isinstance(binding, Mapping):
        relative = binding.get("path")
        digest = binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ReplayWorkerError("RoboTwin replay control profile is malformed")
        control_profile = load_embodiment_control_profile(
            project_root / relative, digest,
        )
    session = RoboTwinLiveSession(
        robotwin_root=_required_environment_path(runtime, "root_env"),
        output_root=output_root / "runtime",
        task=str(plan["task"]),
        actual_seed=int(plan["seed"]),
        episode=0,
        task_config=str(runtime["task_config"]),
        required_revision=str(runtime["required_revision"]),
        workspace_bounds_m=runtime["workspace_bounds_m"],
        max_sim_steps=int(plan["budgets"]["max_sim_steps"]),
        operator_world_video=False,
        operator_execution_video=False,
        execution_capture={"enabled": False},
        environment_signal_policy=str(
            runtime.get("environment_signal_policy", "lifecycle_only")
        ),
        action_codec=str(plan.get("policy", {}).get("action_codec", "runtime_action_v3")),
        control_profile=control_profile,
    )
    cameras = list(manifest["camera_panels"])
    writer = _FrameWriter(
        output_root, cameras=cameras, sampling=sampling,
    )
    planner_patch = None
    if not _robotwin_replay_needs_planner(manifest):
        planner_patch = _disable_unused_robotwin_planners(session.robotwin_root)
    try:
        session.start()
        environment = session._require_environment()
        writer.append(_robotwin_images(environment), entry=None, phase="reset", control_step=None)
        current_round = 0
        for entry in manifest["entries"]:
            if not isinstance(entry, Mapping):
                raise ReplayWorkerError("RoboTwin replay entry is malformed")
            # Replay must cross the same round boundaries the run did.
            # Without this the derived media shows every later round acting
            # on the previous round's end state, so the video and the
            # trajectory it claims to depict diverge.
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                session.restart_round(round_index)
                current_round = round_index
                environment = session._require_environment()
                writer.append(
                    _robotwin_images(environment), entry=None,
                    phase="round_reset", control_step=None,
                )
            arrays = _artifact_arrays(manifest_path, entry)
            if not entry.get("replayable"):
                writer.append(
                    _robotwin_images(environment), entry=entry,
                    phase="not_replayed", control_step=None,
                )
                continue
            controller = entry.get("controller")
            if bool(getattr(environment, "eval_success", False)):
                raise ReplayWorkerError("RoboTwin replay would write after a terminal boundary")
            step = 0
            last_captured_global_step: int | None = None
            execution = entry.get("execution")
            settle = (
                int(execution.get("settle_steps", 0))
                if isinstance(execution, Mapping) else 0
            )
            expected_control_steps = _recorded_control_step_count(entry, arrays)
            selected_steps = _capture_steps(
                expected_control_steps + settle, sampling,
            )
            original_update = getattr(environment, "_update_render")

            # Sampling controls camera capture only. Every recorded target and
            # settle step must run: skipping targets changes contact/grasp physics.
            def recorded_update() -> None:
                nonlocal step, last_captured_global_step
                if step in selected_steps:
                    original_update()
                    writer.append(
                        _robotwin_images(environment), entry=entry,
                        phase="control", control_step=step,
                    )
                    last_captured_global_step = step
                step += 1

            setattr(environment, "_update_render", recorded_update)
            try:
                replay_route = _robotwin_replay_route(controller)
                if replay_route == "dense_targets":
                    original_picture = getattr(environment, "_take_picture", None)
                    original_render_freq = getattr(environment, "render_freq", None)
                    if callable(original_picture):
                        setattr(environment, "_take_picture", lambda: None)
                    if original_render_freq is not None:
                        setattr(environment, "render_freq", 0)
                    try:
                        environment.take_dense_action(
                            _robotwin_control_sequence(entry, arrays), save_freq=1,
                        )
                    finally:
                        if callable(original_picture):
                            setattr(environment, "_take_picture", original_picture)
                        if original_render_freq is not None:
                            setattr(environment, "render_freq", original_render_freq)
                else:
                    environment.take_action(arrays["ee_action"], action_type="ee")
            finally:
                setattr(environment, "_update_render", original_update)
            scene_step = getattr(getattr(environment, "scene", None), "step", None)
            for index in range(settle):
                if not callable(scene_step):
                    break
                scene_step()
                global_step = step + index
                if global_step in selected_steps:
                    original_update()
                    writer.append(
                        _robotwin_images(environment), entry=entry,
                        phase="settle", control_step=index,
                    )
                    last_captured_global_step = global_step
            actual_total_steps = step + settle
            if _needs_final_capture(actual_total_steps, last_captured_global_step):
                original_update()
                writer.append(
                    _robotwin_images(environment), entry=entry,
                    phase="settle_final" if settle else "control_final",
                    control_step=(settle - 1) if settle else (step - 1),
                )
    finally:
        try:
            session.close()
        finally:
            if planner_patch is not None:
                robot_class, original = planner_patch
                robot_class.set_planner = original
    return writer.sequence, cameras


def _libero_images(session: Any) -> dict[str, Any]:
    import numpy as np

    raw = session._raw_observation()
    return {
        "agentview": np.asarray(raw["agentview_image"])[::-1],
        "wrist": np.asarray(raw["robot0_eye_in_hand_image"])[::-1],
    }


def _refresh_libero_replay_observation(session: Any) -> None:
    """Transfer the actual step result; never label the reset camera cache fresh."""
    observation = session._worker_attr("harness_replay_observation")
    if not isinstance(observation, Mapping):
        raise ReplayWorkerError("LIBERO replay returned no step-bound observation")
    session._require_environment().current_raw_obs = [dict(observation)]


def _libero_qualified_controller_binding(manifest_path: Path) -> Mapping[str, Any]:
    """Load the exact launch-time qualification sealed by the source run."""

    reset = _load(manifest_path.parent.parent / "reset.json")
    if reset.get("schema") != "ora.libero-pro-reset.v1":
        raise ReplayWorkerError("run omits its sealed LIBERO reset evidence")
    binding = reset.get("qualified_controller_binding")
    if not isinstance(binding, Mapping):
        raise ReplayWorkerError(
            "run reset evidence omits its qualified LIBERO controller binding"
        )
    return binding


def _replay_libero(
    *, plan: Mapping[str, Any], manifest: Mapping[str, Any],
    manifest_path: Path, output_root: Path,
) -> tuple[int, list[str]]:
    from ora.environment.backends.libero_pro.session import LiberoProSession

    runtime = plan.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ReplayWorkerError("run omits Runtime configuration")
    qualified_controller_binding = _libero_qualified_controller_binding(manifest_path)
    session = LiberoProSession(
        environment_driver=str(runtime.get("environment_driver", "direct_libero_v1")),
        libero_pro_root=_required_environment_path(runtime, "libero_pro_root_env"),
        data_root=_required_environment_path(runtime, "data_root_env"),
        output_root=output_root / "runtime",
        task=str(plan["task"]),
        seed=int(plan["seed"]),
        suite=str(runtime["suite"]),
        task_id=int(runtime["task_id"]),
        robustness_case=(
            str(runtime["robustness_case"])
            if runtime.get("robustness_case") is not None else None
        ),
        expected_task_language=str(runtime["expected_task_language"]),
        required_libero_pro_revision=str(runtime["required_libero_pro_revision"]),
        required_data_revision=str(runtime["required_data_revision"]),
        workspace_bounds_m=runtime["workspace_bounds_m"],
        workspace_profile=runtime.get("workspace_profile"),
        # Qualification is a launch-time result, so it is sealed in reset evidence
        # rather than the immutable preflight RunPlan.  Replay must revalidate that
        # exact binding instead of silently bypassing the normal Session gate.
        qualified_controller_binding=qualified_controller_binding,
        max_sim_steps=int(plan["budgets"]["max_sim_steps"]),
        servo_max_steps=int(runtime.get("servo_max_steps", 80)),
        action_codec="runtime_action_v3",
        execution_capture={"enabled": False},
    )
    cameras = list(manifest["camera_panels"])
    writer = _FrameWriter(
        output_root,
        cameras=cameras,
        sampling={"mode": "physics_stride_v1", "capture_stride": 1},
    )
    try:
        session.start()
        writer.append(_libero_images(session), entry=None, phase="reset", control_step=None)
        current_round = 0
        environment = session._require_environment()
        vector_environment = getattr(environment, "env", None)
        setter = getattr(vector_environment, "set_env_attr", None)
        if not callable(setter):
            raise ReplayWorkerError("LIBERO replay command transport is unavailable")
        for entry in manifest["entries"]:
            if not isinstance(entry, Mapping):
                raise ReplayWorkerError("LIBERO replay entry is malformed")
            round_index = int(entry.get("round_index", 0))
            if round_index != current_round:
                session.restart_round(round_index)
                current_round = round_index
                writer.append(
                    _libero_images(session), entry=None,
                    phase="round_reset", control_step=None,
                )
            arrays = _artifact_arrays(manifest_path, entry)
            if not entry.get("replayable"):
                writer.append(
                    _libero_images(session), entry=entry,
                    phase="not_replayed", control_step=None,
                )
                continue
            controller = entry.get("controller")
            if controller == "libero_joint_position":
                setter("harness_replay_controller_mode", "JOINT_POSITION")
                commands = arrays.get("commands", [])
                for index, command in enumerate(commands):
                    setter("harness_replay_command", command.tolist())
                    _refresh_libero_replay_observation(session)
                    writer.append(
                        _libero_images(session), entry=entry,
                        phase="control", control_step=index,
                    )
                setter("harness_replay_controller_mode", "OSC_POSE")
                refresh = arrays.get("refresh_osc_command", [])
                for offset, command in enumerate(refresh):
                    setter("harness_replay_command", command.tolist())
                    _refresh_libero_replay_observation(session)
                    writer.append(
                        _libero_images(session), entry=entry,
                        phase="refresh", control_step=len(commands) + offset,
                    )
            elif controller == "libero_osc_pose":
                setter("harness_replay_controller_mode", "OSC_POSE")
                for index, command in enumerate(arrays.get("commands", [])):
                    setter("harness_replay_command", command.tolist())
                    _refresh_libero_replay_observation(session)
                    writer.append(
                        _libero_images(session), entry=entry,
                        phase="control", control_step=index,
                    )
            else:
                raise ReplayWorkerError(
                    f"unsupported LIBERO replay controller: {controller}"
                )
    finally:
        session.close()
    return writer.sequence, cameras


def run(job_path: Path) -> dict[str, Any]:
    from ora.environment.controller_replay import (
        load_controller_replay_manifest,
    )

    job = _load(job_path)
    if job.get("schema") != JOB_SCHEMA:
        raise ReplayWorkerError("invalid Viewer replay job")
    run_root = Path(str(job["run_root"])).resolve()
    output_root = Path(str(job["output_root"])).resolve()
    project_root = Path(str(job["project_root"])).resolve()
    sampling = job.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ReplayWorkerError("Viewer replay job omits its sampling profile")
    # Validate the closed profile before allocating the simulator.
    _capture_steps(1, sampling)
    if output_root == run_root or output_root.is_relative_to(run_root):
        raise ReplayWorkerError("Viewer replay output must remain outside the run")
    plan = _load(run_root / "plan.json")
    manifest_path = run_root / "native/controller-replay/manifest.json"
    manifest = load_controller_replay_manifest(manifest_path)
    effective_sampling = (
        dict(sampling)
        if manifest["backend"] == "robotwin"
        else {"mode": "physics_stride_v1", "capture_stride": 1}
    )
    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        if manifest["backend"] == "robotwin":
            frame_count, cameras = _replay_robotwin(
                plan=plan, manifest=manifest, manifest_path=manifest_path,
                project_root=project_root, output_root=temporary,
                sampling=effective_sampling,
            )
        elif manifest["backend"] == "libero_pro":
            frame_count, cameras = _replay_libero(
                plan=plan, manifest=manifest, manifest_path=manifest_path,
                output_root=temporary,
            )
        else:
            raise ReplayWorkerError("unsupported controller replay backend")
        result = {
            "schema": RESULT_SCHEMA,
            "backend": manifest["backend"],
            "source_run_id": run_root.name,
            "source_manifest": {
                "path": "native/controller-replay/manifest.json",
                "sha256": _hash(manifest_path),
            },
            "frame_count": frame_count,
            "sampling": effective_sampling,
            "camera_panels": cameras,
            "timeline": {
                "path": "timeline.jsonl",
                "sha256": _hash(temporary / "timeline.jsonl"),
            },
            "capture_semantics": (
                "derived_same_source_seed_controller_replay_action_keyframes"
                if effective_sampling.get("mode") == "action_keyframes_v1"
                else "derived_same_source_seed_controller_replay"
                if effective_sampling.get("capture_stride") == 1
                else "derived_same_source_seed_controller_replay_temporally_subsampled"
            ),
            "evidence_boundary": (
                "viewer_derived_not_original_physics_or_official_verdict_evidence"
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compile one Viewer controller replay.")
    parser.add_argument("--job", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args.job.resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
