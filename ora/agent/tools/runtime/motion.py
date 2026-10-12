"""Recovery and gripper motion tools.

``recover_pose``, ``set_gripper``, ``open_gripper`` and
``close_gripper``, each over the shared submission path.

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.contracts import MOTION_ACTION_V3_SCHEMA

from ora.agent.tools import observed_clearance_runtime, retention
from ora.agent.tools.robot import ToolError


class MotionRuntimeMixin:
    """Recovery and gripper motion tools."""

    def _atomic_gripper(self, arguments: Mapping[str, Any], *, target: float) -> dict[str, Any]:
        self._ensure_mutation_available(
            "close_gripper" if target == 0.0 else "open_gripper"
        )
        if arguments:
            raise ToolError("gripper action accepts no arguments")
        control, _root, observation, _pose, _gripper = self._current_grip_state()
        result = self._atomic_submit(
            control=control,
            action_name="set_gripper",
            arguments={"arm": self._atomic_arm(observation), "gripper": target},
            schema="ora-action.v3",
        )
        return result

    def _atomic_close_gripper(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.surface.supports("atomic_gripper_v1"):
            return self._gripper_atomic_v3(arguments, directive="close")
        return self._atomic_gripper(arguments, target=0.0)

    def _atomic_open_gripper(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if self.surface.supports("atomic_gripper_v1"):
            return self._gripper_atomic_v3(arguments, directive="open")
        return self._atomic_gripper(arguments, target=1.0)

    def _gripper_atomic_v3(
        self, arguments: Mapping[str, Any], *, directive: str,
    ) -> dict[str, Any]:
        """Atomic open/close over the shared sparse v3 codec (H3 ruling).

        The atomic verbs return to the EXP01B lineage: one small decision the
        policy (and a later small student) can learn directly, instead of a
        parameterised set_gripper form. Dual-arm surfaces select one arm per
        call; single-arm surfaces take no arguments.
        """

        name = "close_gripper" if directive == "close" else "open_gripper"
        self._ensure_mutation_available(name)
        intent_target_ref = None
        if directive == "close" and self.surface.supports(retention.ADVISORY_CAPABILITY):
            intent_target_ref = arguments.get("target_ref")
            if intent_target_ref is not None and (
                not isinstance(intent_target_ref, str) or not intent_target_ref.strip()
            ):
                raise ToolError("target_ref must be a non-empty intent annotation or null",
                                code="invalid_arguments", field="target_ref")
            arguments = {key: value for key, value in arguments.items() if key != "target_ref"}
        if self.surface.supports("dual_arm_sparse_targets_v1"):
            if set(arguments) != {"arm"} or arguments.get("arm") not in {
                "left", "right",
            }:
                raise ToolError(
                    f"{name} requires exactly arm=left or arm=right",
                    code="invalid_arguments", field="arm",
                )
            arm = str(arguments["arm"])
        else:
            if arguments:
                raise ToolError(
                    f"{name} accepts no arguments on a single-arm surface",
                    code="invalid_arguments",
                )
            arm = "right"
        control, _root, observation, _pose, _gripper = self._current_grip_state()
        self._surface_arm_states(observation)
        result = self._atomic_submit(
            control=control,
            action_name="set_gripper",
            arguments={"targets": {arm: directive}},
            schema=MOTION_ACTION_V3_SCHEMA,
            public_action_name=name,
        )
        if directive == "close" and self.surface.supports(retention.ADVISORY_CAPABILITY):
            result["intent"] = {"target_ref": intent_target_ref, "role": "agent_annotation_only"}
        return result

    def _motion_set_gripper(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        return self._motion_set_gripper_v2(arguments)

    def _motion_set_gripper_v2(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        from ora.agent.tools import gripper_intent

        self._ensure_mutation_available("set_gripper")
        arguments, intent_target_ref, annotated = gripper_intent.extract(
            arguments, surface=self.surface,
        )
        targets = self._surface_targets(arguments, scalar_key="action")
        if any(value not in {"open", "close"} for value in targets.values()):
            raise ToolError(
                "gripper targets must be open or close",
                code="invalid_motion_arguments",
            )
        control, _root, observation, _pose, _gripper = self._current_grip_state()
        self._surface_arm_states(observation)
        result = self._atomic_submit(
            control=control,
            action_name="set_gripper",
            arguments={"targets": targets},
            schema=MOTION_ACTION_V3_SCHEMA,
            public_action_name="set_gripper",
        )
        observed_clearance_runtime.note_gripper(self, targets, target_ref=intent_target_ref, observation=observation)
        return gripper_intent.annotate(
            result, targets=targets, reference=intent_target_ref,
            published=annotated,
        )

    def _motion_recover_pose_v2(
        self, arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Expose truthful previous/home pose recovery over recover_motion."""

        self._ensure_mutation_available("recover_pose")
        targets = self._surface_targets(arguments, scalar_key="strategy")
        if any(value not in {"previous", "home"} for value in targets.values()):
            raise ToolError(
                "recover_pose targets must be previous or home",
                code="invalid_motion_arguments",
            )
        control, _root, observation, _pose, _gripper = self._current_grip_state()
        self._surface_arm_states(observation)
        canonical_targets = {
            arm: "backtrack_last_safe" if strategy == "previous" else "home"
            for arm, strategy in targets.items()
        }
        return self._atomic_submit(
            control=control,
            action_name="recover_motion",
            arguments={"targets": canonical_targets},
            compiled_pose={
                "semantic_primitive": "recover_pose",
                "canonical_action": "recover_motion",
                "targets": {
                    arm: {
                        "strategy": strategy,
                        "canonical_strategy": canonical_targets[arm],
                    }
                    for arm, strategy in targets.items()
                },
            },
            schema=MOTION_ACTION_V3_SCHEMA,
            public_action_name="recover_pose",
        )
