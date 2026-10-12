"""Current and historical public-camera projection for dual-arm backends."""
from __future__ import annotations
from typing import Any, Mapping
from .robot import ToolError
from .observe_arguments import solved_history
from .runtime.core import _png_base64

def _dual_arm_observe(
    self, arguments: Mapping[str, Any], *, allowed: set[str], backend: str,
) -> dict[str, Any]:
    if set(arguments) - {"views", "history_point_ids"}:
        raise ToolError("observe accepts views and history_point_ids")
    requested = arguments.get("views") or list(
        self.view_config["default_observe_views"]
    )
    if (
        not isinstance(requested, list) or not requested
        or len(requested) != len(set(requested))
        or not set(requested).issubset(allowed)
    ):
        raise ToolError(
            f"views must be a non-empty unique {backend} camera list",
            constraints={"enum": sorted(allowed)},
        )
    control, root, observation, _pose, _gripper = self._current_grip_state()
    states = self._dual_arm_states(observation)
    effectors = observation["proprioception"]["end_effectors"]
    history_points = solved_history(
        self._load_points(), arguments.get("history_point_ids") or [],
    )
    encoded: list[str] = []
    dimensions: dict[str, dict[str, int]] = {}
    arm_points = [
        ("L", states["left"][:3], (255, 0, 255)),
        ("R", states["right"][:3], (255, 180, 0)),
    ]
    for view in requested:
        rendered = self._annotated_rgb(
            root=root, observation=observation, camera=view,
            labelled_world_points=[*arm_points, *history_points],
            identity_kind="observe",
        )
        encoded.append(_png_base64(rendered))
        dimensions[view] = self._image_dimensions(rendered)
    result = {
        "success": True,
        "observation_id": control["observation_id"],
        "views": requested,
        "end_effectors": {
            arm: {
                "pose_xyz_xyzw": states[arm],
                "gripper_command_target": self._dual_arm_gripper_target(
                    effectors[arm],
                ),
                **({"gripper_opening_m": effectors[arm].get("gripper_opening_m")}
                   if self.surface.supports("measured_jaw_separation_v1") else {}),
            }
            for arm in ("left", "right")
        },
        "view_dimensions": dimensions,
        "images_base64": encoded,
    }
    return self._attach_observe_public_facts(result, root, observation)
