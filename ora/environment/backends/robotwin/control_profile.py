"""Control-profile source checks and the capability predicates derived from it.

A mixin of ``session.RoboTwinLiveSession``, moved verbatim from
``session.py`` on 2026-10-01; it holds methods only, no dataclass fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.environment.controllers.profile_capabilities import (
    CONTROLLER_TRACKING_SINCE,
    GRIP_CENTER_SINCE,
    MOTION_INTENT_SINCE,
    MOTION_TRANSACTION_SINCE,
    RGBD_GRASP_STAGING_SINCE,
    VALIDATED_PLANNER_SINCE,
    profile_has,
)

from .common import RoboTwinLiveError, _file_hash


class RoboTwinControlProfileMixin:
    """Control-profile source checks and the capability predicates derived from it."""

    def _verify_control_profile_sources(self, args: Mapping[str, Any]) -> None:
        profile = self.control_profile["profile"]  # type: ignore[index]
        source = profile["source_binding"]
        if source.get("robotwin_revision") != self.required_revision:
            raise RoboTwinLiveError(
                "control profile RoboTwin revision differs from the selected runtime"
            )
        selected = {
            "base_task_sha256": _file_hash(self.robotwin_root / "envs/_base_task.py"),
            "robot_adapter_sha256": _file_hash(self.robotwin_root / "envs/robot/robot.py"),
        }
        if "planner_adapter_sha256" in source:
            selected["planner_adapter_sha256"] = _file_hash(
                self.robotwin_root / "envs/robot/planner.py"
            )
        if "planner_attribution_sha256" in source:
            selected["planner_attribution_sha256"] = _file_hash(
                self.robotwin_root / "CAPX_MIT_ATTRIBUTION.md"
            )
        robot_root = Path(str(args["left_robot_file"])).resolve()
        config_path = robot_root / "config.yml"
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        urdf_path = robot_root / str(config["urdf_path"])
        if self.public_robot_geometry is not None:
            if Path(str(args["right_robot_file"])).resolve() != robot_root:
                raise RoboTwinLiveError("public ALOHA geometry requires the same verified dual-arm asset")
            self._robot_geometry_urdf = urdf_path.resolve()
        selected.update({
            "aloha_config_sha256": _file_hash(config_path),
            "aloha_urdf_sha256": _file_hash(urdf_path.resolve()),
        })
        for field, digest in selected.items():
            if source.get(field) != digest:
                raise RoboTwinLiveError(
                    f"control profile source binding drifted: {field}"
                )

    # Capability checks derive from one registered profile-version source
    # (profile_capabilities). Adding a profile version never touches these.
    def _uses_grip_center_profile(self) -> bool:
        return profile_has(
            self.control_profile, since_version=GRIP_CENTER_SINCE,
        )

    def _uses_controller_tracking_profile(self) -> bool:
        return profile_has(
            self.control_profile, since_version=CONTROLLER_TRACKING_SINCE,
        )

    def _uses_validated_planner_profile(self) -> bool:
        return profile_has(
            self.control_profile, since_version=VALIDATED_PLANNER_SINCE,
        )

    def _uses_motion_transaction_profile(self) -> bool:
        return profile_has(
            self.control_profile, since_version=MOTION_TRANSACTION_SINCE,
        )

    def _uses_motion_intent_profile(self) -> bool:
        return profile_has(
            self.control_profile, since_version=MOTION_INTENT_SINCE,
        )

    def _uses_rgbd_grasp_staging_profile(self) -> bool:
        return profile_has(
            self.control_profile, since_version=RGBD_GRASP_STAGING_SINCE,
        )

    def _arm_profile(self, arm: str) -> Mapping[str, Any]:
        if not isinstance(self.control_profile, Mapping):
            raise RoboTwinLiveError("control profile is unavailable")
        profile = self.control_profile.get("profile")
        arms = profile.get("arms") if isinstance(profile, Mapping) else None
        record = arms.get(arm) if isinstance(arms, Mapping) else None
        if not isinstance(record, Mapping):
            raise RoboTwinLiveError(f"control profile lacks arm {arm}")
        return record
