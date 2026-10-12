"""Target-state proposal validation; mutation stays at the host submission port."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Mapping
from ora.contracts.workspace import is_monotonic_workspace_recovery, workspace_state
from .robot import ToolError, _finite_vector

def _submit_target_state(
    self, arguments: Mapping[str, Any], target: Path,
) -> dict[str, Any]:
    from ora.contracts.action import validate_runtime_action

    if set(arguments) - {"arm", "pose", "gripper"}:
        raise ToolError("target state fields are arm, optional pose and gripper")
    arm = arguments.get("arm")
    if arm not in {"left", "right"}:
        raise ToolError(
            "arm must be left or right", field="arm", received=arm,
            constraints={"enum": ["left", "right"]},
        )
    normalized: dict[str, Any] = {"arm": arm}
    control, _root, observation = self._current()
    pose = arguments.get("pose")
    if pose is not None:
        if not isinstance(pose, Mapping) or set(pose) != {
            "position_m", "orientation_xyzw",
        }:
            raise ToolError(
                "pose must contain position_m and orientation_xyzw", field="pose"
            )
        position = _finite_vector(pose["position_m"], 3, "position_m")
        orientation = _finite_vector(
            pose["orientation_xyzw"], 4, "orientation_xyzw"
        )
        current_position = self._observed_arm_position(observation, str(arm))
        target_workspace = workspace_state(position, self.workspace_bounds_m)
        recovery = (
            current_position is not None
            and is_monotonic_workspace_recovery(
                current_position, position, self.workspace_bounds_m,
            )
        )
        if not target_workspace["inside"] and not recovery:
            for index, axis in enumerate(("x", "y", "z")):
                details = target_workspace["violations"].get(axis)
                if details is not None:
                    interval = details["allowed"]
                    raise ToolError(
                        f"position {axis} is outside workspace bounds",
                        code="out_of_bounds",
                        field=f"pose.position_m[{index}]",
                        received=position[index],
                        constraints={"minimum": interval[0], "maximum": interval[1]},
                    )
        normalized["pose"] = {
            "position_m": position, "orientation_xyzw": orientation,
        }
    if "gripper" in arguments and arguments.get("gripper") is not None:
        gripper = arguments["gripper"]
        if (
            isinstance(gripper, bool) or not isinstance(gripper, (int, float))
            or not math.isfinite(gripper) or not 0 <= gripper <= 1
        ):
            raise ToolError(
                "gripper must be in [0,1]", field="gripper", received=gripper,
                constraints={"minimum": 0, "maximum": 1},
            )
        normalized["gripper"] = float(gripper)
    action = {
        "schema": "ora-action.v4",
        "observation_id": control["observation_id"],
        "action_id": f"action-{control['generation']}",
        "action": "target_state",
        "arguments": normalized,
        "bound_observation_sha256": control["observation_sha256"],
    }
    try:
        action = validate_runtime_action(action)
    except ValueError as exc:
        raise ToolError(str(exc), field="arguments") from exc
    return self._persist_action(target, control, action)
