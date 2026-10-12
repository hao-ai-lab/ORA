"""RoboTwin original-execution media hooks, separate from motion and verdicts."""

from __future__ import annotations

from typing import Any, Mapping

from ora.environment.execution_capture import ExecutionCapture, PhysicsStepTap


class RoboTwinExecutionCaptureMixin:
    """Observe the session's existing steps without advancing its world."""

    def _capture_operator_reset_frames(self, *, label: str) -> None:
        """Synchronize the settled reset scene before its first camera sample."""

        if not (self.operator_world_video or self.operator_execution_video):
            return
        # RoboTwin settles newly loaded actors without updating the renderer.
        # take_picture alone still sees pre-settle render transforms, whereas
        # the later policy get_obs refreshes them. Synchronize camera and scene
        # transforms directly: _update_render also randomizes lights, so an
        # extra call there would change the task's RNG and illumination state.
        try:
            environment = self._require_environment()
            environment.cameras.update_wrist_camera(
                environment.robot.left_camera.get_pose(),
                environment.robot.right_camera.get_pose(),
            )
            environment.scene.update_render()
        except Exception as exc:
            self._record_operator_capture_failure(exc)
            return
        self._capture_operator_frame(label=label)
        if self.operator_execution_video:
            self._capture_operator_execution_frame(
                action=None, phase="reset", control_step_index=None,
            )

    def _start_execution_capture(self) -> None:
        if not self.execution_capture["enabled"]:
            return
        try:
            if self._execution_recorder is None:
                self._execution_recorder = ExecutionCapture(
                    self.output_root, backend="robotwin",
                    identity={"run_id": self.output_root.parent.name, "task": self.task,
                              "source_manifest": self.source_manifest},
                    cameras=["head_camera", "front_camera"], config=self.execution_capture,
                )
            self._execution_physics_step = 0
            self._execution_action = None
            scene = self._require_environment().scene
            self._execution_step_tap = PhysicsStepTap(scene, self._after_capture_physics_step)
            self._capture_execution_boundary("reset")
        except Exception as exc:
            self._execution_capture_error = f"{type(exc).__name__}: {exc}"
            if self._execution_recorder is not None:
                self._execution_recorder.error = self._execution_capture_error

    def _after_capture_physics_step(self) -> None:
        self._execution_physics_step += 1
        self._capture_execution_boundary("control", force=False)

    def _capture_execution_boundary(self, phase: str, *, force: bool = True) -> None:
        recorder = self._execution_recorder
        if recorder is None:
            return
        try:
            environment = self._require_environment()
            scene = environment.scene
            sim_time = self._execution_physics_step * float(scene.get_timestep())

            def pixels() -> dict[str, Any]:
                # Do not call Base_Task._update_render: randomized lighting there
                # consumes RNG. Synchronize actual poses without changing lights.
                manager = environment.cameras
                manager.update_wrist_camera(environment.robot.left_camera.get_pose(),
                                            environment.robot.right_camera.get_pose())
                scene.update_render()
                manager.update_picture()
                raw = manager.get_rgb()
                return {name: raw[name]["rgb"] for name in recorder.cameras}

            context = self._execution_action or {}
            recorder.capture(
                pixels, round_index=self.round_index,
                action_id=context.get("action_id"), action=context.get("action"),
                action_sequence=context.get("action_sequence"),
                sim_step=self._execution_physics_step, sim_time_s=sim_time,
                step_unit="sapien_step_since_round_reset", phase=phase, force=force,
            )
        except Exception as exc:
            self._execution_capture_error = f"{type(exc).__name__}: {exc}"
            recorder.error = self._execution_capture_error

    def execute(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._execution_action = {"action_id": payload.get("action_id"),
                                  "action": payload.get("action"),
                                  "action_sequence": self.action_sequence}
        self._capture_execution_boundary("action_start")
        try:
            from .wait import execute_wait, is_wait
            if is_wait(payload):  # session.py is descriptor-pinned; wait.py drives its helpers
                return execute_wait(self, payload)
            return self._execute_without_capture(payload)
        finally:
            self._capture_execution_boundary("action_end")
            self._execution_action = None

    def _close_execution_capture(self) -> None:
        if self._execution_step_tap is not None:
            self._execution_step_tap.close()
            self._execution_step_tap = None
        if self._execution_recorder is not None:
            self._capture_execution_boundary("close")
            self._execution_recorder.close()
