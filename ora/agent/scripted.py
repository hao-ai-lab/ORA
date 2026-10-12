"""Deterministic scripted policy: replays a fixed list of tool calls.

Used by the Synthetic quickstart and by tests before any live model is involved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ora.agent.tools.robot import RobotToolRuntime


class ScriptedPlannerError(RuntimeError):
    """A scripted action list is invalid for the current observation."""


@dataclass
class ScriptedPlanner:
    """Deterministic policy used before every live model milestone."""

    tools: RobotToolRuntime
    actions: Sequence[Mapping[str, Any]]
    state_probe: bool = False
    index: int = 0

    def run_observation(self, **kwargs: Any) -> dict[str, Any]:
        if self.index >= len(self.actions):
            return {
                "submitted": False,
                "termination": "policy_no_action",
                "attempts": 1,
            }
        probe: dict[str, Any] | None = None
        if self.state_probe:
            context = kwargs.get("context", {})
            state_ref = (
                context.get("policy_state") if isinstance(context, Mapping) else None
            )
            if not isinstance(state_ref, Mapping) or not isinstance(
                state_ref.get("step"), int,
            ):
                raise ScriptedPlannerError("scripted state probe lacks a policy-state step")
            dumped = self.tools.invoke(
                "observation.dump_state", {"step": state_ref["step"]},
            )
            state = dumped.get("state")
            if not isinstance(state, Mapping) or state.get("step") != state_ref["step"]:
                raise ScriptedPlannerError("scripted state probe returned the wrong step")
            if self.index == 0:
                if state.get("previous_public_action") is not None:
                    raise ScriptedPlannerError(
                        "initial policy-state unexpectedly contains an action"
                    )
            elif not isinstance(state.get("previous_public_action"), Mapping):
                raise ScriptedPlannerError(
                    "policy-state chain lost the previous public action"
                )
            elif not isinstance(state.get("previous_execution_receipt"), Mapping):
                raise ScriptedPlannerError(
                    "policy-state chain lost the previous execution receipt"
                )
            probe = {
                "step": state["step"],
                "state_sha256": state.get("state_sha256"),
                "previous_action_present": (
                    state.get("previous_public_action") is not None
                ),
                "previous_receipt_present": (
                    state.get("previous_execution_receipt") is not None
                ),
            }
        arguments = dict(self.actions[self.index])
        self.index += 1
        tool_name = str(arguments.pop("tool", "robot.submit_action"))
        offset = arguments.pop("current_pose_offset_m", None)
        if offset is not None:
            context = kwargs.get("context", {})
            arm = arguments.get("arm")
            effectors = (context.get("proprioception") or {}).get(
                "end_effectors", {},
            )
            current = (effectors.get(arm) or {}).get("pose_xyz_xyzw")
            if (
                not isinstance(offset, list)
                or len(offset) != 3
                or not isinstance(current, list)
                or len(current) != 7
            ):
                raise ScriptedPlannerError(
                    "relative scripted action lacks a current EEF pose"
                )
            arguments["position_m"] = [
                float(current[index]) + float(offset[index]) for index in range(3)
            ]
            arguments.setdefault(
                "orientation_xyzw", [float(value) for value in current[3:]],
            )
        if tool_name == "robot.submit_action":
            if arguments.get("action") not in {"move_pose", "set_gripper"}:
                raise ScriptedPlannerError(
                    "scripted action must explicitly select move_pose or set_gripper"
                )
        elif tool_name not in self.tools.surface.mutation_tools:
            raise ScriptedPlannerError(
                "scripted action tool must select a surface mutation tool"
            )
        result = self.tools.invoke(tool_name, arguments)
        return {
            "submitted": True,
            "tool_result": result,
            "state_probe": probe,
            "attempts": 1,
        }
