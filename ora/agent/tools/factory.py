"""Registered construction of canonical policy-visible tool runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ora.agent.composition import resolve_tool_surface
from .runtime import ManipulationToolRuntime
from .robot import RobotToolRuntime

from .runtime_protocol import ToolRuntime


def build_tool_runtime(
    *,
    control_root: Path,
    manual_path: Path,
    event_path: Path,
    workspace_bounds_m: Mapping[str, list[float]],
    allowed_tools: Sequence[str] | None,
    state_root: Path | None,
    profile_id: str,
    component_overrides: Mapping[str, Any] | None,
    prerequisite_binding: Mapping[str, Any] | None = None,
    controlled_end_effector: str | None = None,
    motion_probe_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    align_plan_callback: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> ToolRuntime:
    surface = resolve_tool_surface(profile_id, component_overrides=component_overrides)
    runtime_type = {
        "robot_runtime": RobotToolRuntime,
        "native_atomic_causal": ManipulationToolRuntime,
    }.get(surface.runtime_id)
    if runtime_type is None:
        raise ValueError(f"unregistered tool runtime: {surface.runtime_id}")
    runtime_arguments: dict[str, Any] = dict(
        control_root=control_root,
        manual_path=manual_path,
        event_path=event_path,
        workspace_bounds_m=workspace_bounds_m,
        allowed_tools=allowed_tools,
        state_root=state_root,
        tool_surface_profile=profile_id,
        tool_surface_overrides=component_overrides,
        controlled_end_effector=controlled_end_effector,
        motion_probe_callback=motion_probe_callback,
        align_plan_callback=align_plan_callback,
    )
    if runtime_type is ManipulationToolRuntime:
        runtime_arguments["prerequisite_binding"] = prerequisite_binding
    return runtime_type(**runtime_arguments)
