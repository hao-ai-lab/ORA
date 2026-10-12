"""Operator-only world and execution video capture.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

from .common import (
    OPERATOR_CAMERA_ORDER,
    OPERATOR_EXECUTION_FRAME_SCHEMA,
    OPERATOR_EXECUTION_VIDEO_SCHEMA,
    RoboTwinLiveError,
    _atomic_json,
    _file_hash,
)


class RoboTwinOperatorCaptureMixin:
    """Operator-only world and execution video capture."""

    def _execute_legacy_action_with_operator_capture(
        self, *, environment: Any, ee_action: Any, action: Mapping[str, Any],
    ) -> None:
        """Record every render update made by the legacy dense control loop."""

        self._capture_operator_execution_frame(
            action=action, phase="action_start", control_step_index=None,
        )
        if not self.operator_execution_video:
            environment.take_action(ee_action, action_type="ee")
            return
        original_update = getattr(environment, "_update_render")
        control_step = 0

        def recorded_update() -> None:
            nonlocal control_step
            if self.operator_capture_failure is not None:
                return
            try:
                original_update()
            except Exception as exc:
                self._record_operator_capture_failure(exc)
                return
            self._capture_operator_execution_frame(
                action=action,
                phase="control",
                control_step_index=control_step,
            )
            control_step += 1

        setattr(environment, "_update_render", recorded_update)
        try:
            environment.take_action(ee_action, action_type="ee")
        finally:
            setattr(environment, "_update_render", original_update)

    @staticmethod
    def _operator_contact_sheet(images: Mapping[str, Any]) -> Any:
        """Compose the five synchronized RoboTwin RGB views without metadata."""

        import numpy as np
        from PIL import Image, ImageDraw, ImageFont, ImageOps

        canvas = Image.new("RGB", (1280, 960), "#101413")
        layout = {
            "observer_camera": (0, 0, 427, 480),
            "head_camera": (427, 0, 853, 480),
            "front_camera": (853, 0, 1280, 480),
            "left_camera": (0, 480, 640, 960),
            "right_camera": (640, 480, 1280, 960),
        }
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 18,
            )
        except OSError:
            font = ImageFont.load_default()
        draw = ImageDraw.Draw(canvas)
        for name in OPERATOR_CAMERA_ORDER:
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
            draw.rectangle((left, top, min(right, left + 218), top + 30), fill="#101413")
            draw.text((left + 8, top + 6), name.upper(), font=font, fill="#f2f0e8")
        return canvas

    def _capture_operator_execution_frame(
        self, *, action: Mapping[str, Any] | None, phase: str,
        control_step_index: int | None,
    ) -> None:
        """Persist one synchronized same-session multi-camera execution sample.

        Capture is auxiliary operator evidence.  A media failure is recorded
        and disables later samples, but it never changes controller execution
        or the environment-owned verdict.
        """

        if not self.operator_execution_video or self.operator_capture_failure is not None:
            return
        try:
            environment = self._require_environment()
            manager = getattr(environment, "cameras", None)
            update_picture = getattr(manager, "update_picture", None)
            get_rgb = getattr(manager, "get_rgb", None)
            get_observer = getattr(manager, "get_observer_rgb", None)
            if not all(callable(item) for item in (update_picture, get_rgb, get_observer)):
                raise RoboTwinLiveError("synchronized RoboTwin RGB cameras are unavailable")
            update_picture()
            raw = get_rgb()
            if not isinstance(raw, Mapping):
                raise RoboTwinLiveError("RoboTwin RGB capture returned an invalid mapping")
            images = {
                str(name): value.get("rgb")
                for name, value in raw.items()
                if isinstance(value, Mapping) and value.get("rgb") is not None
            }
            images["observer_camera"] = get_observer()
            missing = [name for name in OPERATOR_CAMERA_ORDER if name not in images]
            if missing:
                raise RoboTwinLiveError(
                    "RoboTwin operator capture lacks cameras: " + ", ".join(missing)
                )
            root = self.output_root / "operator-execution-frames"
            root.mkdir(parents=True, exist_ok=True)
            index = self.operator_execution_frame_sequence
            frame = root / f"frame-{index:06d}.jpg"
            self._operator_contact_sheet(images).save(
                frame, format="JPEG", quality=88, subsampling=1,
            )
            record = {
                "schema": OPERATOR_EXECUTION_FRAME_SCHEMA,
                "frame_index": index,
                "path": frame.relative_to(self.output_root).as_posix(),
                "sha256": _file_hash(frame),
                "captured_unix_s": time.time(),
                "action_sequence": self.action_sequence if action is not None else None,
                "action_id": action.get("action_id") if action is not None else None,
                "action": action.get("action") if action is not None else "reset",
                "phase": phase,
                "control_step_index": control_step_index,
                "cameras": list(OPERATOR_CAMERA_ORDER),
                "visibility": "control_only",
                "capture_semantics": (
                    "same_environment_session_after_actual_physics_control_step"
                    if phase in {"control", "settle"}
                    else "same_environment_session_boundary"
                ),
            }
            timeline = root / "timeline.jsonl"
            with timeline.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            self.operator_execution_frame_sequence += 1
        except Exception as exc:  # media must not alter robot execution
            self._record_operator_capture_failure(exc)

    def _record_operator_capture_failure(self, exc: Exception) -> None:
        self.operator_capture_failure = f"{type(exc).__name__}: {exc}"
        try:
            _atomic_json(
                self.output_root / "operator-execution-capture-failure.json",
                {
                    "schema": "ora.auxiliary-media-failure.v1",
                    "component": "robotwin_operator_execution_capture",
                    "error": self.operator_capture_failure,
                    "frame_count": self.operator_execution_frame_sequence,
                },
            )
        except Exception:
            # A second media-evidence write failure is still auxiliary.
            pass

    def _capture_operator_frame(self, *, label: str) -> None:
        """Capture trusted observer-camera evidence outside policy observations."""

        if not self.operator_world_video:
            return
        environment = self._require_environment()
        cameras = getattr(environment, "cameras", None)
        getter = getattr(cameras, "get_observer_rgb", None)
        if not callable(getter):
            raise RoboTwinLiveError("operator observer camera is unavailable")
        import numpy as np
        from PIL import Image

        rgb = np.asarray(getter())
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RoboTwinLiveError("operator observer camera returned an invalid RGB frame")
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        root = self.output_root / "operator-world-frames"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"frame-{self.operator_frame_sequence:04d}-{label}.png"
        Image.fromarray(rgb).save(path)
        self.operator_frame_sequence += 1

    def finalize_operator_video(self) -> dict[str, Any] | None:
        """Encode trusted observer keyframes without blocking controller steps."""
        if self._execution_recorder is not None:
            self._execution_recorder.close()

        if not (self.operator_world_video or self.operator_execution_video):
            return None
        if self.operator_video_record is not None:
            return dict(self.operator_video_record)
        execution_root = self.output_root / "operator-execution-frames"
        frames = (
            sorted(execution_root.glob("frame-*.jpg"))
            if self.operator_execution_video else []
        )
        continuous = bool(frames)
        if not frames:
            frames = sorted((self.output_root / "operator-world-frames").glob("*.png"))
        if not frames:
            raise RoboTwinLiveError("operator world video has no frames")
        import imageio.v2 as imageio

        video = self.output_root / "operator-world.mp4"
        writer = imageio.get_writer(video, fps=self.operator_video_fps, codec="libx264")
        try:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
        finally:
            writer.close()
        if not video.is_file() or video.stat().st_size == 0:
            raise RoboTwinLiveError("operator world video encoder produced no artifact")
        timeline = execution_root / "timeline.jsonl"
        manifest = {
            "schema": (
                OPERATOR_EXECUTION_VIDEO_SCHEMA
                if continuous else "ora.operator-world-video.v1"
            ),
            "path": video.relative_to(self.output_root).as_posix(),
            "sha256": _file_hash(video),
            "frame_count": len(frames),
            "fps": self.operator_video_fps,
            "policy_visible": False,
            "source": (
                "RoboTwin synchronized five-camera actual control execution"
                if continuous else "RoboTwin observer_camera trusted keyframes"
            ),
            "camera_panels": (
                list(OPERATOR_CAMERA_ORDER) if continuous else ["observer_camera"]
            ),
            "capture_semantics": (
                "same_environment_session_actual_physics_control_frames"
                if continuous else "action_boundary_keyframes"
            ),
            "timeline": (
                {
                    "path": timeline.relative_to(self.output_root).as_posix(),
                    "sha256": _file_hash(timeline),
                }
                if continuous and timeline.is_file() else None
            ),
            "capture_failure": self.operator_capture_failure,
        }
        _atomic_json(self.output_root / "operator-world-video.json", manifest)
        self.operator_video_record = manifest
        return dict(manifest)
