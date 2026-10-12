"""Source-owned hard action envelopes for registered environments."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from ora.contracts.workspace import normalize_workspace_bounds


LIBERO_PRO_PANDA_TABLETOP_V1 = "libero_pro_panda_tabletop_v1"
LIBERO_PRO_PANDA_HYBRID_MOTION_V1 = "libero_pro_panda_hybrid_motion_v1"
ROBODOJO_ARX_X5_TABLETOP_V1 = "robodojo_arx_x5_tabletop_v1"

_PROFILES: dict[str, dict[str, Any]] = {
    LIBERO_PRO_PANDA_TABLETOP_V1: {
        "schema": "ora.action-workspace-profile.v1",
        "profile_id": LIBERO_PRO_PANDA_TABLETOP_V1,
        "environment_id": "libero_pro",
        "embodiment_id": "franka_panda",
        "frame": "LIBERO world",
        "semantics": "hard_action_envelope_not_reachability",
        "bounds_m": {
            "x": [-0.60, 0.60],
            "y": [-0.50, 0.50],
            "z": [-0.10, 1.30],
        },
        "provenance": {
            "kind": "pinned_backend_scene_extent_and_regression_fixture",
            "backend": "LIBERO-Pro",
            "controller": "OSC_POSE",
            "qualification": "required_before_live_resume",
        },
    },
    LIBERO_PRO_PANDA_HYBRID_MOTION_V1: {
        "schema": "ora.action-workspace-profile.v1",
        "profile_id": LIBERO_PRO_PANDA_HYBRID_MOTION_V1,
        "environment_id": "libero_pro",
        "embodiment_id": "franka_panda",
        "frame": "LIBERO world",
        "semantics": "hard_action_envelope_not_reachability",
        "bounds_m": {
            "x": [-0.60, 0.60],
            "y": [-0.50, 0.50],
            "z": [-0.10, 1.30],
        },
        "provenance": {
            "kind": "pinned_backend_scene_extent_and_regression_fixture",
            "backend": "LIBERO-Pro",
            "controller": "CUROBO_JOINT_POSITION+OSC_POSE",
            "qualification": "required_before_live_resume",
        },
    },
    # RoboDojo episode-local world frame: the 1.4 x 1.1 m table top lies at
    # z = 0.765 (scene Table scale [1.4, 1.1, 0.05] at [0.0, -0.05, 0.74]),
    # both ARX X5 bases sit on its -y edge at (+/-0.3, -0.45, 0.765) and the
    # link6 reset poses are (+/-0.3, -0.352, 0.922). The table spans
    # x in [-0.7, 0.7] and y in [-0.6, 0.5]; the envelope covers it with a
    # small margin and stops below the fixed head camera (z = 1.308). It is a
    # rejection bound for link6 targets, not reach.
    ROBODOJO_ARX_X5_TABLETOP_V1: {
        "schema": "ora.action-workspace-profile.v1",
        "profile_id": ROBODOJO_ARX_X5_TABLETOP_V1,
        "environment_id": "robodojo",
        "embodiment_id": "arx_x5",
        "frame": "robodojo_episode_local",
        "semantics": "hard_action_envelope_not_reachability",
        "bounds_m": {
            "x": [-0.75, 0.75],
            "y": [-0.65, 0.55],
            "z": [0.72, 1.40],
        },
        "provenance": {
            "kind": "pinned_backend_scene_extent_and_reset_pose",
            "backend": "RoboDojo",
            "backend_revision": "ee67a1468510da7624a089164402359f2afc72c8",
            "scene_sources": [
                "env_cfg/scene/default.yml:Table scale [1.4, 1.1, 0.05] at "
                "[0.0, -0.05, 0.74]",
                "env_cfg/robot/dual_x5.yml:default_root_pos "
                "[-0.3, -0.45, 0.765] and [0.3, -0.45, 0.765]",
            ],
            "controlled_point": "link6",
            "controller": "SDK_IK_JOINT_INTERPOLATION+LOCAL_CARTESIAN_STEP",
            "qualification": "required_before_live_resume",
        },
    },
}


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def resolve_workspace_profile(profile_id: str) -> dict[str, Any]:
    try:
        body = dict(_PROFILES[profile_id])
    except KeyError as exc:
        raise ValueError(f"unknown action workspace profile: {profile_id}") from exc
    body["bounds_m"] = normalize_workspace_bounds(body["bounds_m"])
    return {**body, "profile_sha256": _digest(body)}


__all__ = [
    "LIBERO_PRO_PANDA_HYBRID_MOTION_V1", "LIBERO_PRO_PANDA_TABLETOP_V1",
    "ROBODOJO_ARX_X5_TABLETOP_V1", "resolve_workspace_profile",
]
