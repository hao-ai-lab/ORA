"""Shared admission and submission for the motion tools.

Mutation availability, arm and target parsing, pose compilation,
workspace validation and ``_atomic_submit``, which persists the one
action proposal the Supervisor executes.

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ora.contracts.workspace import (
    is_monotonic_workspace_recovery, workspace_state,
)

from ora.agent.tools import tool_errors
from ora.agent.tools.runtime.common import _matrix_quat, _quat_matrix
from ora.agent.tools.robot import ToolError, _finite_vector


class MotionSubmissionRuntimeMixin:
    """Shared admission and submission for the motion tools."""

    def _load_previews(self) -> dict[str, Any]:
        return json.loads(self.preview_path.read_text(encoding="utf-8")) if self.preview_path.is_file() else {}

    def _ensure_mutation_available(self, tool: str) -> None:
        from ora.agent.tools.skill_driver import ensure_paused_mutation_available
        ensure_paused_mutation_available(self, tool)
        task_check_path = self.control_root / "policy-task-check.json"
        if not task_check_path.is_file():
            return
        task_check = json.loads(task_check_path.read_text(encoding="utf-8"))
        if task_check.get("terminal") is True:
            raise ToolError(
                f"{tool} is unavailable after the trusted environment terminal boundary",
                code="episode_terminal", retryable=False,
            )

    def _atomic_arm(self, observation: Mapping[str, Any]) -> str:
        end_effectors = observation.get("proprioception", {}).get("end_effectors", {})
        available = [
            str(name) for name, state in end_effectors.items()
            if isinstance(state, Mapping)
        ] if isinstance(end_effectors, Mapping) else []
        if self.controlled_end_effector is not None:
            if self.controlled_end_effector not in available:
                raise ToolError(
                    "environment-bound end effector is absent from the current observation",
                    code="embodiment_binding", retryable=False,
                    constraints={"available_effectors": sorted(available)},
                )
            return self.controlled_end_effector
        if len(available) != 1:
            raise ToolError(
                "atomic single-effector surface requires exactly one public end effector",
                code="embodiment_binding", retryable=False,
                constraints={"available_effectors": sorted(available)},
            )
        return available[0]

    def _atomic_pose_arguments(
        self, arguments: Mapping[str, Any], *, current_pose: list[float], arm: str,
    ) -> dict[str, Any]:
        allowed = {
            "xyz_m", "delta_mm", "delta_frame", "approach_world", "jaw_world",
        }
        if set(arguments) - allowed:
            raise ToolError("pose request contains unknown fields")
        xyz = arguments.get("xyz_m")
        delta = arguments.get("delta_mm")
        tool_errors.require_at_most_one_of(arguments, ("xyz_m", "delta_mm"), message="provide xyz_m or delta_mm, not both")
        target_position = list(current_pose[:3])
        if xyz is not None:
            target_position = _finite_vector(xyz, 3, "xyz_m")
        elif delta is not None:
            import numpy as np

            displacement = (
                np.asarray(_finite_vector(delta, 3, "delta_mm"), dtype=float)
                / 1000.0
            )
            frame = arguments.get("delta_frame", "world")
            if frame not in {"world", "grip_site"}:
                raise ToolError("delta_frame must be world or grip_site")
            if frame == "grip_site":
                displacement = _quat_matrix(current_pose[3:]) @ displacement
            target_position = (
                np.asarray(current_pose[:3], dtype=float) + displacement
            ).tolist()
        orientation = list(current_pose[3:])
        approach = arguments.get("approach_world")
        jaw = arguments.get("jaw_world")
        if approach is not None or jaw is not None:
            import numpy as np

            rotation = _quat_matrix(orientation)
            z_axis = (
                rotation[:, 2]
                if approach is None
                else np.asarray(_finite_vector(approach, 3, "approach_world"))
            )
            x_axis = (
                rotation[:, 0]
                if jaw is None
                else np.asarray(_finite_vector(jaw, 3, "jaw_world"))
            )
            z_axis = z_axis / np.linalg.norm(z_axis)
            x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
            if np.linalg.norm(x_axis) < 1e-6:
                raise ToolError("jaw_world must not be parallel to approach_world")
            x_axis = x_axis / np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            orientation = _matrix_quat(np.column_stack([x_axis, y_axis, z_axis]))
        if xyz is None and delta is None and approach is None and jaw is None:
            raise ToolError("pose request requires a position, delta, or orientation")
        result = {
            "arm": arm,
            "position_m": target_position,
            "orientation_xyzw": orientation,
        }
        self._validate_target_workspace({
            "pose": {
                "position_m": target_position,
                "orientation_xyzw": orientation,
            }
        }, current_position_m=current_pose[:3])
        return result

    def _atomic_compiled_target(
        self, arguments: Mapping[str, Any], *, current_pose: list[float], arm: str,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        return (
            self._atomic_pose_arguments(
                arguments, current_pose=current_pose, arm=arm,
            ),
            None,
        )

    def _atomic_submit(
        self, *, control: Mapping[str, Any], action_name: str,
        arguments: Mapping[str, Any], compiled_pose: Mapping[str, Any] | None = None,
        schema: str = "ora-action.v3",
        public_action_name: str | None = None,
    ) -> dict[str, Any]:
        current_control, _root, _observation, _pose, _gripper = self._current_grip_state()
        if current_control["observation_id"] != control["observation_id"]:
            raise ToolError(
                "observation changed before action persistence",
                code="stale_observation",
            )
        action = {
            "schema": schema,
            "observation_id": control["observation_id"],
            "action_id": f"action-{control['generation']}",
            "action": action_name,
            "arguments": dict(arguments),
            "bound_observation_sha256": control["observation_sha256"],
        }
        from ora.contracts.action import validate_runtime_action

        try:
            action = validate_runtime_action(action)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        self._persist_action(
            self.control_root / "submitted-action.json", dict(control), action,
        )
        self._submitted_mutations.append(action_name)
        submitted_name = public_action_name or action_name
        if public_action_name is None and action_name == "set_gripper" and "gripper" in arguments:
            submitted_name = (
                "close_gripper" if float(arguments["gripper"]) == 0.0
                else "open_gripper"
            )
        result = {
            "status": "submitted",
            "action": submitted_name,
            "_runtime": {
                "directive": "action_submitted",
                "action_id": action["action_id"],
            },
        }
        if compiled_pose is not None:
            result["compiled_pose"] = dict(compiled_pose)
        return result

    def _surface_targets(
        self, arguments: Mapping[str, Any], *,
        extra: frozenset[str] = frozenset(), scalar_key: str | None = None,
    ) -> dict[str, Any]:
        """Sparse dual-arm targets, or the flat single-arm form (M1 ruling).

        A surface declaring ``single_arm_flat_targets_v1`` exposes no
        ``targets`` wrapper: the arguments themselves are the single right
        arm's request, and the Harness compiles them to the shared sparse
        codec internally. A never-fillable ``left`` slot on a single-arm
        embodiment would be a fake shared schema, not parity.
        """

        if self.surface.supports("single_arm_flat_targets_v1"):
            if "targets" in arguments:
                raise ToolError(
                    "this single-arm surface takes flat arguments without "
                    "a targets wrapper",
                    code="invalid_motion_arguments",
                )
            core = {
                key: value for key, value in arguments.items()
                if key not in extra
            }
            if scalar_key is not None:
                if set(core) != {scalar_key}:
                    raise ToolError(
                        f"arguments must contain exactly {scalar_key}",
                        code="invalid_motion_arguments",
                    )
                return {"right": core[scalar_key]}
            if not core:
                raise ToolError(
                    "flat single-arm arguments are required",
                    code="invalid_motion_arguments",
                )
            return {"right": core}
        return self._sparse_targets(arguments, extra=extra)

    def _surface_arm_states(
        self, observation: Mapping[str, Any],
    ) -> dict[str, list[float]]:
        """Measured arm states for the surface's declared topology."""

        if not self.surface.supports("single_arm_flat_targets_v1"):
            return self._dual_arm_states(observation)
        effectors = observation.get("proprioception", {}).get(
            "end_effectors", {},
        )
        state = (
            effectors.get("right") if isinstance(effectors, Mapping) else None
        )
        pose = state.get("pose_xyz_xyzw") if isinstance(state, Mapping) else None
        if not isinstance(pose, list) or len(pose) != 7:
            raise ToolError(
                "current right end-effector pose is unavailable",
                code="embodiment_binding", retryable=False,
            )
        return {"right": [float(item) for item in pose]}

    @staticmethod
    def _sparse_targets(
        arguments: Mapping[str, Any], *, extra: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        if set(arguments) != {"targets", *extra}:
            raise ToolError(
                "sparse motion arguments contain unexpected fields",
                code="invalid_motion_arguments",
            )
        targets = arguments.get("targets")
        if (
            not isinstance(targets, Mapping)
            or not targets
            or set(targets) - {"left", "right"}
        ):
            raise ToolError(
                "targets must specify left, right, or both",
                code="invalid_motion_arguments",
            )
        return dict(targets)

    @staticmethod
    def _public_end_effector_compilation(
        compiled: Mapping[str, Any], *, delta_frame: str | None = None,
    ) -> dict[str, Any]:
        """Project a compiler receipt without Panda/grip-site vocabulary."""

        result = {
            "controlled_point": "end_effector",
            "current_pose_xyz_xyzw": list(compiled["current_pose_xyz_xyzw"]),
            "target_pose_xyz_xyzw": list(compiled["target_pose_xyz_xyzw"]),
            "translation_world_m": list(compiled["translation_world_m"]),
            "translation_distance_m": float(compiled["translation_distance_m"]),
            "rotation_angle_rad": float(compiled["rotation_angle_rad"]),
        }
        if delta_frame is not None:
            result["delta_frame"] = delta_frame
        return result

    def _public_sparse_pose_compilation(
        self, compiled: Mapping[str, Any], *, delta_frame: str | None = None,
    ) -> dict[str, Any]:
        result = self._public_end_effector_compilation(
            compiled, delta_frame=delta_frame,
        )
        if self.surface.supports("contact_graspnet_robotwin_aloha_grip_center_compiler_v3"):
            result["controlled_point_semantics"] = (
                "parallel_jaw_grip_center_tcp"
            )
        return result

    def _validate_target_workspace(
        self, target_state: Mapping[str, Any], *,
        current_position_m: Sequence[float] | None = None,
        field: str | None = "xyz_m",
    ) -> None:
        """Refuse a target outside the action workspace, naming the argument
        ``field`` it came from (``xyz_m`` on the atomic tools that take it)."""
        pose = target_state.get("pose")
        if not isinstance(pose, Mapping):
            return
        position = pose.get("position_m")
        if not isinstance(position, list) or len(position) != 3:
            return
        state = workspace_state(position, self.workspace_bounds_m)
        recovery = (
            current_position_m is not None
            and is_monotonic_workspace_recovery(
                current_position_m, position, self.workspace_bounds_m,
            )
        )
        if not state["inside"] and not recovery:
            raise ToolError(
                "target pose is outside the configured action workspace",
                code="workspace_bounds", field=field, received=position,
                constraints={
                    "workspace_bounds_m": self.workspace_bounds_m,
                    "violations": state["violations"],
                    "monotonic_recovery_allowed": current_position_m is not None,
                },
            )
