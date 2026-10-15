"""LIBERO original-pixel capture and acknowledged worker lifecycle hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.environment.execution_capture import ExecutionCapture


class LiberoWorkerCaptureMixin:
    """Capture observations from the worker's actual simulator step result."""

    def _init_execution_capture(self) -> None:
        self._execution_recorder: Any = None
        self._capture_active = False
        self._capture_round = 0
        self._capture_origin_s = 0.0
        self._capture_action: dict[str, Any] = {}
        self._capture_last_images: dict[str, Any] = {}

    def step(self, command: Any) -> Any:
        result = self._delegate.step(command)
        if self._execution_recorder is not None:
            try:
                self._capture_last_images = self._capture_images(result[0])
                if self._capture_active:
                    self._capture_frame("control", force=False)
            except Exception as exc:
                self._execution_recorder.error = f"{type(exc).__name__}: {exc}"
        return result

    @staticmethod
    def _capture_images(raw: Mapping[str, Any]) -> dict[str, Any]:
        return {"agentview": raw["agentview_image"][::-1],
                "wrist": raw["robot0_eye_in_hand_image"][::-1]}

    def _capture_frame(self, phase: str, *, force: bool = True) -> None:
        if self._execution_recorder is None or not self._capture_active:
            return
        try:
            sim = self._delegate.sim
            elapsed = max(0.0, float(sim.data.time) - self._capture_origin_s)
            self._execution_recorder.capture(
                lambda: self._capture_last_images,
                round_index=self._capture_round,
                action_id=self._capture_action.get("action_id"),
                action_sequence=self._capture_action.get("action_sequence"),
                action=self._capture_action.get("action"),
                sim_step=round(elapsed / float(sim.model.opt.timestep)),
                sim_time_s=elapsed, step_unit="mujoco_step_since_round_reset",
                phase=phase, force=force,
            )
        except Exception as exc:
            self._execution_recorder.error = f"{type(exc).__name__}: {exc}"

    @property
    def harness_execution_capture(self) -> dict[str, Any]:
        recorder = self._execution_recorder
        return {
            "initialized": recorder is not None,
            "finalized": recorder is not None and recorder._closed,
            "error": getattr(self, "_capture_command_error", None) or (recorder.error if recorder else None),
        }

    @harness_execution_capture.setter
    def harness_execution_capture(self, value: Mapping[str, Any]) -> None:
        try:
            self._execution_capture_command(value)
        except Exception as exc:
            # A media failure must not kill the simulator.
            self._capture_command_error = f"{type(exc).__name__}: {exc}"
            if self._execution_recorder is not None:
                self._execution_recorder.error = self._capture_command_error

    def _execution_capture_command(self, value: Mapping[str, Any]) -> None:
        from .session import LiberoProError

        # Trusted, Control-only vector-worker channel; not a policy tool.
        op = value.get("op")
        if op == "init":
            if self._execution_recorder is not None:
                raise LiberoProError("execution recorder is already initialized")
            self._execution_recorder = ExecutionCapture(
                Path(value["output_root"]), backend="libero_pro",
                identity=value["identity"], cameras=["agentview", "wrist"],
                config=value["config"],
            )
        elif op == "reset":
            self._capture_origin_s = float(self._delegate.sim.data.time)
            self._capture_round = int(value["round_index"])
            self._capture_action = {}
            self._capture_last_images = self._capture_images(value["raw_images"])
            self._capture_active = True
            self._capture_frame("reset")
        elif op == "action_start":
            self._capture_action = dict(value["context"])
            self._capture_frame("action_start")
        elif op in {"action_end", "terminal"}:
            self._capture_frame(str(op))
            if op == "action_end":
                self._capture_action = {}
        elif op == "close":
            if self._execution_recorder is not None:
                self._capture_frame("close")
                self._execution_recorder.close()
            self._capture_active = False
        else:
            raise LiberoProError("unknown execution capture operation")


class LiberoExecutionCaptureMixin:
    """Bind action context and wait for auxiliary-media finalization."""

    def _capture_command(self, op: str, **values: Any) -> None:
        from .session import LiberoProError, _atomic_json

        if not self.execution_capture["enabled"] or self.environment is None:
            return
        try:
            vector = getattr(self.environment, "env", None)
            setter = getattr(vector, "set_env_attr", None)
            if not callable(setter):
                raise LiberoProError("execution capture worker channel is unavailable")
            setter("harness_execution_capture", {"op": op, **values})
            # The attribute setter only sends a command. A following getattr is the
            # FIFO acknowledgement, essential before worker close and sealing.
            status = self._worker_attr("harness_execution_capture")
            if not isinstance(status, Mapping) or not status.get("initialized"):
                raise LiberoProError("execution recorder initialization failed")
            if status.get("error"):
                raise LiberoProError(str(status["error"]))
            if op == "close" and status.get("finalized") is not True:
                raise LiberoProError("execution recorder did not finalize")
        except Exception as exc:
            self._execution_capture_error = f"{type(exc).__name__}: {exc}"
            try:
                _atomic_json(self.output_root / "execution-capture-failure.json", {
                    "schema": "ora.auxiliary-media-failure.v1",
                    "component": "original_execution_capture", "error": self._execution_capture_error,
                })
            except Exception:
                pass

    def _capture_reset(self) -> None:
        from .session import _atomic_json

        if self.execution_capture["enabled"]:
            try:
                raw = self._raw_observation()
                self._capture_command("reset", round_index=self.round_index, raw_images={
                    name: raw[name] for name in ("agentview_image", "robot0_eye_in_hand_image")
                    if name in raw
                })
            except Exception as exc:
                self._execution_capture_error = f"{type(exc).__name__}: {exc}"
                try:
                    _atomic_json(self.output_root / "execution-capture-failure.json", {
                        "schema": "ora.auxiliary-media-failure.v1",
                        "component": "original_execution_capture", "error": self._execution_capture_error,
                    })
                except Exception:
                    pass

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._capture_command("action_start", context={
            "action_id": payload.get("action_id"), "action": payload.get("action"),
            "action_sequence": self.action_sequence,
        })
        try:
            from .wait import execute_wait, is_wait
            if is_wait(payload):  # the receipt-owning session files are descriptor-pinned
                return execute_wait(self, payload)
            return self._execute_without_capture(payload)
        finally:
            self._capture_command("action_end")

    def finalize_operator_video(self) -> None:
        self._capture_command("close")
        return None
