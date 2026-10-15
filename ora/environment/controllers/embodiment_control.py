"""Hash-bound execution semantics kept separate from embodiment facts.

Validation model:

- Every profile version's identity comes from the single registry in
  :mod:`ora.environment.controllers.profile_capabilities`.
- **Historical versions (v1-v9)** are frozen files. Their byte integrity is
  already enforced by :func:`load_embodiment_control_profile`'s mandatory
  sha256 pin, so validation only confirms schema/identity registration. The
  former per-version equality pyramid (every numeric pinned in code, ~25
  schema-set memberships per new version) duplicated the sha pin and made
  each parameter change a whole-file surgery; it is retired.
- **The current version (v10)** is validated structurally and semantically in
  code: required sections, arm topology, numeric sanity. Its exact values
  live in the JSON only; content freezing remains the loader's sha pin,
  which every consumer must supply.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ora.environment.controllers.profile_capabilities import (
    PROFILE_VERSION_IDS,
)

EMBODIMENT_CONTROL_PROFILE_SCHEMA = "embodiment-control-profile.v1"
EMBODIMENT_CONTROL_PROFILE_V2_SCHEMA = "embodiment-control-profile.v2"
EMBODIMENT_CONTROL_PROFILE_V3_SCHEMA = "embodiment-control-profile.v3"
EMBODIMENT_CONTROL_PROFILE_V4_SCHEMA = "embodiment-control-profile.v4"
EMBODIMENT_CONTROL_PROFILE_V5_SCHEMA = "embodiment-control-profile.v5"
EMBODIMENT_CONTROL_PROFILE_V6_SCHEMA = "embodiment-control-profile.v6"
EMBODIMENT_CONTROL_PROFILE_V7_SCHEMA = "embodiment-control-profile.v7"
EMBODIMENT_CONTROL_PROFILE_V8_SCHEMA = "embodiment-control-profile.v8"
EMBODIMENT_CONTROL_PROFILE_V9_SCHEMA = "embodiment-control-profile.v9"
EMBODIMENT_CONTROL_PROFILE_V10_SCHEMA = "embodiment-control-profile.v10"

#: The one profile version whose content is validated semantically in code.
CURRENT_PROFILE_VERSION = 10

_SCHEMA_PREFIX = "embodiment-control-profile.v"

_CURRENT_TOP_LEVEL_KEYS = {
    "arms", "arrival_check", "control_groups", "controller_tracking",
    "dual_arm_collision_check", "embodiment_id", "goalset_planning",
    "gripper_actuator_mapping", "hold_semantics", "ik_topk_planning",
    "local_cartesian", "motion_transaction", "plan_references",
    "plan_validation", "profile_id", "schema", "source_binding",
    "strategy_router", "waypoint_execution", "blocked_motion_recovery",
}
_CURRENT_ARM_KEYS = {
    "controlled_point", "controlled_point_semantics", "measured_joint_route",
    "pinocchio_chain_links", "pinocchio_link_name",
    "pinocchio_to_sapien_link_post_rotation_matrix", "planner_input_point",
    "planner_joint_names", "planner_joint_route", "planner_route",
    "public_id", "public_pose_route", "public_tcp_post_rotation_matrix",
    "public_tcp_translation_local_m", "public_to_planner_translation_local_m",
    "recovery_home_route",
}
_CURRENT_SOURCE_BINDING_KEYS = {
    "aloha_config_sha256", "aloha_urdf_sha256", "base_task_sha256",
    "planner_adapter_sha256", "planner_attribution_sha256",
    "robot_adapter_sha256", "robotwin_revision",
}
_CURRENT_MOTION_TRANSACTION_KEYS = {
    "adapter_schema", "capx_reference", "global_convergence",
    "local_convergence", "metadata_extensions", "plan_artifact_schema",
    "qualification_sweep", "schema", "statuses",
}
_CONVERGENCE_KEYS = {
    "joint_convergence_tolerance_rad",
    "maximum_physics_steps_per_waypoint",
    "maximum_total_physics_steps",
    "maximum_tcp_step_m",
    "workspace_margin_m",
}


class EmbodimentControlProfileError(ValueError):
    """A control profile is malformed or no longer source bound."""


def _schema_version(schema: Any) -> int | None:
    if not isinstance(schema, str) or not schema.startswith(_SCHEMA_PREFIX):
        return None
    suffix = schema[len(_SCHEMA_PREFIX):]
    return int(suffix) if suffix.isdigit() else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EmbodimentControlProfileError(message)


def _positive_number(record: Mapping[str, Any], key: str, *, label: str) -> None:
    value = record.get(key)
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and float(value) > 0.0,
        f"{label} {key} must be a positive number",
    )


def _validate_current_profile(value: Mapping[str, Any]) -> None:
    _require(
        set(value) == _CURRENT_TOP_LEVEL_KEYS,
        "current control profile top-level contract drifted",
    )
    _require(
        value.get("embodiment_id") == "aloha_agilex",
        "control profile embodiment identity drifted",
    )
    arms = value.get("arms")
    _require(
        isinstance(arms, Mapping) and set(arms) == {"left", "right"},
        "control profile requires left and right arms",
    )
    for arm, record in arms.items():
        _require(
            isinstance(record, Mapping) and set(record) == _CURRENT_ARM_KEYS,
            f"control profile arm contract drifted: {arm}",
        )
        prefix = "fl" if arm == "left" else "fr"
        joints = record.get("planner_joint_names")
        _require(
            isinstance(joints, list) and joints
            and all(isinstance(name, str) and name.startswith(prefix) for name in joints),
            f"control profile planner joints drifted: {arm}",
        )
        _require(
            record.get("controlled_point_semantics")
            == "parallel_jaw_grip_center_tcp",
            f"control profile controlled point drifted: {arm}",
        )
    source = value.get("source_binding")
    _require(
        isinstance(source, Mapping)
        and set(source) == _CURRENT_SOURCE_BINDING_KEYS,
        "control profile source binding contract drifted",
    )
    for field, digest in source.items():
        if field == "robotwin_revision":
            _require(
                isinstance(digest, str) and len(digest) == 40
                and all(c in "0123456789abcdef" for c in digest),
                "control profile RoboTwin revision must be a commit hash",
            )
            continue
        _require(
            isinstance(digest, str) and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            f"control profile source digest must be sha256: {field}",
        )
    transaction = value.get("motion_transaction")
    _require(
        isinstance(transaction, Mapping)
        and set(transaction) == _CURRENT_MOTION_TRANSACTION_KEYS,
        "motion transaction contract drifted",
    )
    for mode in ("global_convergence", "local_convergence"):
        convergence = transaction.get(mode)
        _require(
            isinstance(convergence, Mapping)
            and set(convergence) == _CONVERGENCE_KEYS,
            f"{mode} contract drifted",
        )
        for key in _CONVERGENCE_KEYS:
            _positive_number(convergence, key, label=mode)
    arrival = value.get("arrival_check")
    _require(
        isinstance(arrival, Mapping)
        and {"position_tolerance_m", "orientation_tolerance_rad"} <= set(arrival),
        "arrival check contract drifted",
    )
    _positive_number(arrival, "position_tolerance_m", label="arrival_check")
    _positive_number(arrival, "orientation_tolerance_rad", label="arrival_check")
    collision = value.get("dual_arm_collision_check")
    _require(isinstance(collision, Mapping), "inter-arm collision contract is missing")
    for key in ("link_radius_m", "minimum_surface_clearance_m",
                "execution_minimum_surface_clearance_m", "escape_maximum_inward_deviation_m"):
        item = collision.get(key)
        _require(isinstance(item, (int, float)) and not isinstance(item, bool)
                 and math.isfinite(item) and item >= 0,
                 f"inter-arm collision {key} must be finite and nonnegative")
    _require(collision["link_radius_m"] > 0
             and collision["minimum_surface_clearance_m"] > collision["execution_minimum_surface_clearance_m"]
             and collision["escape_maximum_inward_deviation_m"] > 0,
             "planning clearance must exceed execution clearance with bounded escape")
    blocked = value.get("blocked_motion_recovery")
    _require(isinstance(blocked, Mapping) and set(blocked) == {
        "consecutive_frozen_physics_steps", "maximum_frozen_joint_step_rad", "retreat_distance_m",
    }, "blocked motion recovery contract drifted")
    for key in blocked:
        _positive_number(blocked, key, label="blocked_motion_recovery")
        _require(math.isfinite(blocked[key]), f"blocked_motion_recovery {key} must be finite")
    count = blocked["consecutive_frozen_physics_steps"]
    _require(isinstance(count, int) and not isinstance(count, bool), "blocked window must be physics steps")
    _require(count <= transaction["global_convergence"]["maximum_physics_steps_per_waypoint"],
             "blocked window must fit the global waypoint wait")
    _require(blocked["retreat_distance_m"] <= value["local_cartesian"]["maximum_translation_m"],
             "blocked recovery must fit the existing local envelope")


def validate_embodiment_control_profile(value: Any) -> dict[str, Any]:
    """Validate one parsed control profile.

    Identity is checked for every version against the single registered
    version/identity table. Only the current version receives semantic
    content validation here; historical versions are frozen byte-for-byte by
    the loader's sha256 pin.
    """

    if not isinstance(value, Mapping):
        raise EmbodimentControlProfileError("control profile must be an object")
    version = _schema_version(value.get("schema"))
    if version is None or version not in PROFILE_VERSION_IDS:
        raise EmbodimentControlProfileError("control profile schema drifted")
    expected_identity = PROFILE_VERSION_IDS[version]
    if value.get("profile_id") != expected_identity:
        raise EmbodimentControlProfileError(
            f"control profile v{version} identity drifted"
        )
    for field in ("profile_id", "embodiment_id"):
        entry = value.get(field)
        if not isinstance(entry, str) or not entry:
            raise EmbodimentControlProfileError(f"control profile needs {field}")
    if version == CURRENT_PROFILE_VERSION:
        _validate_current_profile(value)
    return dict(value)


def load_embodiment_control_profile(
    path: Path, expected_sha256: str,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise EmbodimentControlProfileError("control profile file is unavailable")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise EmbodimentControlProfileError("control profile hash drifted")
    return {
        "path": str(resolved),
        "sha256": digest,
        "profile": validate_embodiment_control_profile(
            json.loads(resolved.read_text(encoding="utf-8"))
        ),
    }


__all__ = [
    "CURRENT_PROFILE_VERSION",
    "EMBODIMENT_CONTROL_PROFILE_SCHEMA", "EMBODIMENT_CONTROL_PROFILE_V2_SCHEMA",
    "EMBODIMENT_CONTROL_PROFILE_V3_SCHEMA", "EMBODIMENT_CONTROL_PROFILE_V4_SCHEMA",
    "EMBODIMENT_CONTROL_PROFILE_V5_SCHEMA", "EMBODIMENT_CONTROL_PROFILE_V6_SCHEMA",
    "EMBODIMENT_CONTROL_PROFILE_V7_SCHEMA",
    "EMBODIMENT_CONTROL_PROFILE_V8_SCHEMA",
    "EMBODIMENT_CONTROL_PROFILE_V9_SCHEMA", "EMBODIMENT_CONTROL_PROFILE_V10_SCHEMA",
    "EmbodimentControlProfileError",
    "load_embodiment_control_profile", "validate_embodiment_control_profile",
]
