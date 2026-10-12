"""Original RoboDojo pixels sampled after existing physics steps, Control-only."""
from __future__ import annotations

from typing import Any, Mapping

from ora.environment.execution_capture import ExecutionCapture, resolve_execution_capture
from ora.environment.benchmarks.robodojo import PHYSICS_DT_S


class RobodojoExecutionCaptureMixin:
    def _init_execution_capture(self) -> None:
        self.execution_capture = resolve_execution_capture(self.config.get("execution_capture"))
        self._execution_recorder: Any = None
        self._execution_action: dict[str, Any] = {}
        self._execution_capture_error: str | None = None

    def _capture_failure(self, error: Exception) -> None:
        from .common import write_json

        self._execution_capture_error = f"{type(error).__name__}: {error}"
        if self._execution_recorder is not None:
            self._execution_recorder.error = self._execution_capture_error
        try:
            write_json(self.output_root / "execution-capture-failure.json", {
                "schema": "ora.auxiliary-media-failure.v1",
                "component": "original_execution_capture", "error": self._execution_capture_error,
            })
        except Exception:
            pass

    def _start_execution_capture(self) -> None:
        if not self.execution_capture["enabled"]:
            return
        try:
            from .session import CAMERAS

            if self._execution_recorder is None:
                self._execution_recorder = ExecutionCapture(
                    self.output_root, backend="robodojo", cameras=list(CAMERAS),
                    identity={"run_id": self.output_root.parent.name, "task": self.task,
                              "source_manifest_sha256": self.source_hash}, config=self.execution_capture,
                )
            self._execution_action = {}
            self.environment.execution_capture_callback = self._after_capture_physics_step
            self._capture_execution_boundary("reset")
        except Exception as exc:
            self._capture_failure(exc)

    def _after_capture_physics_step(self) -> None:
        self._capture_execution_boundary("control", force=False)

    def _capture_execution_boundary(self, phase: str, *, force: bool = True) -> None:
        recorder = self._execution_recorder
        if recorder is None:
            return
        try:
            from .session import CAMERAS

            step = self._clock() - self.reset_physics_steps
            def pixels() -> dict[str, Any]:
                before = self._clock()
                # Existing get_obs synchronizes camera products and verifies
                # their simulation time and optical pose without physics.
                raw = self.environment.get_obs()
                if self._clock() != before:
                    raise RuntimeError("execution capture readback advanced physics")
                return {name: raw["vision"][source]["color"] for name, source in CAMERAS.items()}

            recorder.capture(
                pixels, round_index=self.round_index, sim_step=step,
                sim_time_s=step * PHYSICS_DT_S, step_unit="physx_step_since_round_reset",
                action_id=self._execution_action.get("action_id"),
                action=self._execution_action.get("action"),
                action_sequence=self._execution_action.get("action_sequence"), phase=phase, force=force,
            )
        except Exception as exc:
            self._capture_failure(exc)

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._execution_action = {"action_id": payload.get("action_id"),
                                  "action": payload.get("action"),
                                  "action_sequence": self.action_sequence}
        self._capture_execution_boundary("action_start")
        try:
            return self._execute_without_capture(payload)
        finally:
            self._capture_execution_boundary("action_end")
            self._execution_action = {}

    def finalize_operator_video(self) -> None:
        if self.environment is not None:
            self.environment.execution_capture_callback = None
        if self._execution_recorder is not None:
            try:
                self._capture_execution_boundary("close")
                self._execution_recorder.close()
            except Exception as exc:
                self._capture_failure(exc)
        return None
