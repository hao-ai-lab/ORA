"""Placement, alignment-plan and closed-skill tools.

Mostly protocol edges: ``place_plan``/``place_verify`` delegate to
``placement_runtime``, ``align_plan`` to ``alignment``, the closed skills to
``skill_driver``. The frozen-surface ``align_gripper``/``align_gripper.propose``
and ``payload.bind``/``payload.unbind`` run only at the commits that composed
them (``core.ManipulationToolRuntime._refuse_frozen_tool``).

A mixin of ``core.ManipulationToolRuntime``; it holds
methods only, no dataclass fields.
"""

from __future__ import annotations

from typing import Any, Mapping

from ora.agent.tools import observed_clearance_runtime
from ora.agent.tools.robot import ToolError


class SkillRuntimeMixin:
    """Placement, alignment-plan and closed-skill tools."""

    def _place_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        from ora.agent.tools.placement_runtime import invoke_placement
        return invoke_placement(self, "place_plan", arguments)

    def _closed_skill(self, skill: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Protocol edge only; the driver lives in tools/skill_driver."""

        from ora.agent.tools import skill_driver
        return (skill_driver.resume_skill(self, arguments) if skill == "place_resume" else
                observed_clearance_runtime.after_skill(self, skill, skill_driver.begin_skill(self, skill, arguments)))

    def drive_skill_stage(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from ora.agent.tools import skill_driver
        return skill_driver.drive_skill_stage(self, *args, **kwargs)

    def build_skill_receipt(self, **kwargs: Any) -> dict[str, Any]:
        from ora.agent.tools import skill_driver
        return skill_driver.build_skill_receipt(self, **kwargs)

    def _align_plan(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Protocol edge only; the work lives in tools/alignment."""

        from ora.agent.tools import alignment

        control, root, observation, _pose, _gripper = self._current_grip_state()
        convention = (self.surface.view_config or {}).get("controlled_frame_convention")
        if not convention:
            raise ToolError(
                "align_plan requires the surface to declare its controlled-frame "
                "convention", code="tool_unavailable", retryable=False,
            )
        return alignment.plan(
            arguments,
            bridge=self.align_plan_callback,
            centroid_of=lambda ref: self._current_target_geometry(
                control=control, root=root, observation=observation, target_ref=ref,
            )["centroid_world_m"],
            frame_convention=convention,
        )
