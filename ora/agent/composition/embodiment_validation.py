"""The section steps of :func:`embodiment.validate_embodiment_descriptor`.

Each function validates one descriptor section and is called in the order the
validator reports its errors, so the first failing check is the same field
with the same message. A step returns the identifiers that later sections
cross-reference: manipulator and end-effector ids from the topology, and
controlled-point ids from the control section. The field helpers, vocabularies
and the projected-fact inventory stay in ``embodiment.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .embodiment import (
    DESCRIPTOR_SCHEMA, DESCRIPTOR_SCHEMA_V2, EmbodimentDescriptorError,
    _COLLISION_PROJECTION_TEXT, _DYNAMIC_FACT_TEXT, _PLACEHOLDER, _PRIVATE_TOKENS,
    _ROOT_KEYS, _SHA256, _SOURCE_KINDS, _WORKSPACE_PROJECTION_TEXT, _exact_keys,
    _identifier, _nonnegative_number, _object, _objects, _policy_identifier,
    _positive_integer, _projected_fact_inventory, _string, _strings,
)
from .embodiment_control_card import (
    _CONTROLLED_POINT_TEXT, _EVIDENCE_LIMIT_TEXT, _STATE_FIELD_TEXT,
    _STATE_UNIT_TEXT,
)


def _validate_header(root: Mapping[str, Any]) -> bool:
    """Check the root keys, schema and naming; return whether it is schema v2."""

    _exact_keys(root, required=_ROOT_KEYS, optional=set(), path="descriptor")
    if root["schema"] not in {DESCRIPTOR_SCHEMA, DESCRIPTOR_SCHEMA_V2}:
        raise EmbodimentDescriptorError("descriptor.schema is invalid")
    geometry_v2 = root["schema"] == DESCRIPTOR_SCHEMA_V2
    _identifier(root["descriptor_id"], "descriptor.descriptor_id")
    _identifier(root["embodiment_id"], "descriptor.embodiment_id")
    _string(root["display_name"], "descriptor.display_name")
    _string(root["status"], "descriptor.status", choices={"draft", "reviewed"})
    return geometry_v2


def _validate_identity(value: Any) -> None:
    identity = _object(value, "descriptor.identity")
    _exact_keys(
        identity,
        required={"manufacturer", "model", "variant", "base_mobility"},
        optional=set(), path="descriptor.identity",
    )
    for name in ("manufacturer", "model", "variant"):
        _string(identity[name], f"descriptor.identity.{name}")
    _string(
        identity["base_mobility"], "descriptor.identity.base_mobility",
        choices={"fixed", "mobile", "hybrid", "virtual"},
    )


def _validate_topology(value: Any) -> tuple[set[str], set[str]]:
    """Return the manipulator ids and the end-effector ids, in that order."""

    topology = _object(value, "descriptor.topology")
    _exact_keys(
        topology, required={"manipulators", "end_effectors"}, optional=set(),
        path="descriptor.topology",
    )
    manipulator_ids = _validate_manipulators(topology["manipulators"])
    effector_ids = _validate_end_effectors(
        topology["end_effectors"], manipulator_ids=manipulator_ids,
    )
    return manipulator_ids, effector_ids


def _validate_manipulators(value: Any) -> set[str]:
    manipulators = _objects(value, "descriptor.topology.manipulators")
    manipulator_ids: set[str] = set()
    for index, item in enumerate(manipulators):
        path = f"descriptor.topology.manipulators[{index}]"
        _exact_keys(
            item, required={"id", "kind", "side", "joint_dof"}, optional=set(),
            path=path,
        )
        item_id = _identifier(item["id"], f"{path}.id")
        if item_id in manipulator_ids:
            raise EmbodimentDescriptorError(f"{path}.id is duplicated")
        manipulator_ids.add(item_id)
        _string(
            item["kind"], f"{path}.kind",
            choices={"serial_arm", "parallel_arm", "mobile_manipulator"},
        )
        _string(
            item["side"], f"{path}.side",
            choices={"left", "right", "center", "none"},
        )
        _positive_integer(item["joint_dof"], f"{path}.joint_dof")
    return manipulator_ids


def _validate_end_effectors(value: Any, *, manipulator_ids: set[str]) -> set[str]:
    effectors = _objects(
        value, "descriptor.topology.end_effectors",
        allow_empty=True,
    )
    effector_ids: set[str] = set()
    for index, item in enumerate(effectors):
        path = f"descriptor.topology.end_effectors[{index}]"
        _exact_keys(
            item,
            required={"id", "kind", "manipulator_id", "command_dof"},
            optional=set(), path=path,
        )
        item_id = _identifier(item["id"], f"{path}.id")
        if item_id in effector_ids:
            raise EmbodimentDescriptorError(f"{path}.id is duplicated")
        effector_ids.add(item_id)
        _string(
            item["kind"], f"{path}.kind",
            choices={"parallel_jaw_gripper", "suction_gripper", "tool"},
        )
        parent = _identifier(item["manipulator_id"], f"{path}.manipulator_id")
        if parent not in manipulator_ids and parent != _PLACEHOLDER:
            raise EmbodimentDescriptorError(f"{path}.manipulator_id is unknown")
        _positive_integer(item["command_dof"], f"{path}.command_dof")
    return effector_ids


def _validate_control(
    value: Any, *, geometry_v2: bool, manipulator_ids: set[str],
) -> set[str]:
    """Check controlled points, then the action conventions; return point ids."""

    control = _object(value, "descriptor.control")
    _exact_keys(
        control,
        required={
            "controlled_points", "action_abstraction", "position_frame",
            "position_unit", "orientation", "controller_family",
            "internal_controller_policy",
        },
        optional=set(), path="descriptor.control",
    )
    points = _objects(control["controlled_points"], "descriptor.control.controlled_points")
    point_ids: set[str] = set()
    for index, item in enumerate(points):
        _validate_controlled_point(
            item, f"descriptor.control.controlled_points[{index}]",
            geometry_v2=geometry_v2, manipulator_ids=manipulator_ids,
            point_ids=point_ids,
        )
    _validate_control_conventions(control)
    return point_ids


def _validate_controlled_point(
    item: Mapping[str, Any], path: str, *, geometry_v2: bool,
    manipulator_ids: set[str], point_ids: set[str],
) -> None:
    """Check one controlled point and add its id to ``point_ids``."""

    _exact_keys(
        item,
        required={"id", "manipulator_id", "source_name", "description"},
        optional={"offset_from", "offset_m"} | ({"local_axes"} if geometry_v2 else set()), path=path,
    )
    point_id = _policy_identifier(item["id"], f"{path}.id")
    if point_id in point_ids:
        raise EmbodimentDescriptorError(f"{path}.id is duplicated")
    point_ids.add(point_id)
    if geometry_v2:
        from .embodiment_geometry import validate_local_axes
        validate_local_axes(item.get("local_axes"))
    parent = _identifier(item["manipulator_id"], f"{path}.manipulator_id")
    if parent not in manipulator_ids and parent != _PLACEHOLDER:
        raise EmbodimentDescriptorError(f"{path}.manipulator_id is unknown")
    source_name = _string(item["source_name"], f"{path}.source_name")
    if any(token in source_name.lower() for token in _PRIVATE_TOKENS):
        raise EmbodimentDescriptorError(f"{path}.source_name names private evidence")
    _string(
        item["description"], f"{path}.description",
        choices=set(_CONTROLLED_POINT_TEXT),
    )
    offset_present = "offset_m" in item or "offset_from" in item
    if offset_present and not {"offset_m", "offset_from"}.issubset(item):
        raise EmbodimentDescriptorError(f"{path} must bind offset_from and offset_m together")
    if "offset_m" in item:
        offset = item["offset_m"]
        if (
            not isinstance(offset, list) or len(offset) != 3
            or any(
                not isinstance(number, (int, float)) or isinstance(number, bool)
                for number in offset
            )
        ):
            raise EmbodimentDescriptorError(f"{path}.offset_m must be XYZ numbers")
        _string(
            item["offset_from"], f"{path}.offset_from",
            choices={"end_effector_root", "manipulator_flange"},
        )


def _validate_control_conventions(control: Mapping[str, Any]) -> None:
    """Check the action abstraction, frame, unit, orientation and controller."""

    _string(
        control["action_abstraction"], "descriptor.control.action_abstraction",
        choices={"cartesian_target", "joint_target", "base_velocity", "hybrid"},
    )
    _string(
        control["position_frame"], "descriptor.control.position_frame",
        choices={"world", "robot_base", "controlled_point_local"},
    )
    _string(
        control["position_unit"], "descriptor.control.position_unit",
        choices={"metre"},
    )
    orientation = _object(control["orientation"], "descriptor.control.orientation")
    _exact_keys(
        orientation, required={"representation", "order", "angle_unit"},
        optional=set(), path="descriptor.control.orientation",
    )
    _string(
        orientation["representation"], "descriptor.control.orientation.representation",
        choices={"quaternion", "rotation_matrix", "axis_angle", "euler", "none"},
    )
    _string(
        orientation["order"], "descriptor.control.orientation.order",
        choices={"xyzw", "wxyz", "xyz", "none"},
    )
    _string(
        orientation["angle_unit"], "descriptor.control.orientation.angle_unit",
        choices={"radian", "none"},
    )
    _string(
        control["controller_family"], "descriptor.control.controller_family",
        choices={
            "operational_space_pose", "joint_position", "joint_velocity",
            "base_velocity", "hybrid",
        },
    )
    _string(
        control["internal_controller_policy"],
        "descriptor.control.internal_controller_policy",
        choices={"tool_surface_owned", "policy_visible"},
    )


def _validate_geometry(
    value: Any, *, geometry_v2: bool, effector_ids: set[str], point_ids: set[str],
) -> None:
    """Check the card projection, grippers, then workspace and collision text."""

    geometry = _object(value, "descriptor.geometry")
    _exact_keys(
        geometry, required={"grippers", "workspace_limits", "collision_model"},
        optional={"card_projection"}, path="descriptor.geometry",
    )
    if "card_projection" in geometry:
        _string(geometry["card_projection"], "descriptor.geometry.card_projection",
                choices=({"full_finger_reference_v1", "full_finger_reference_v2", "full_finger_reference_v3",
                          "grouped_symmetric_sides_v1"} if geometry_v2 else {"control_only_nine_node_v1"}))
    grippers = _objects(
        geometry["grippers"], "descriptor.geometry.grippers", allow_empty=True,
    )
    for index, item in enumerate(grippers):
        _validate_gripper(
            item, f"descriptor.geometry.grippers[{index}]",
            geometry_v2=geometry_v2, effector_ids=effector_ids, point_ids=point_ids,
        )
    for name in ("workspace_limits", "collision_model"):
        _validate_geometry_projection(geometry[name], name)


def _validate_gripper(
    item: Mapping[str, Any], path: str, *, geometry_v2: bool,
    effector_ids: set[str], point_ids: set[str],
) -> None:
    _exact_keys(
        item,
        required={
            "end_effector_id", "opening_mode", "maximum_aperture_m",
            "aperture_semantics",
        },
        optional={"contact_geometry", "reference_parameters"} if geometry_v2 else set(), path=path,
    )
    effector = _identifier(item["end_effector_id"], f"{path}.end_effector_id")
    if effector not in effector_ids and effector != _PLACEHOLDER:
        raise EmbodimentDescriptorError(f"{path}.end_effector_id is unknown")
    _string(
        item["opening_mode"], f"{path}.opening_mode",
        choices={"symmetric", "asymmetric", "binary", "other"},
    )
    if not (geometry_v2 and item["maximum_aperture_m"] is None):
        _nonnegative_number(item["maximum_aperture_m"], f"{path}.maximum_aperture_m")
    if geometry_v2:
        from .embodiment_geometry import validate_contact_geometry
        validate_contact_geometry(item.get("contact_geometry"), controlled_points=point_ids)
        if "reference_parameters" in item:
            from .embodiment_geometry import validate_reference_parameters
            validate_reference_parameters(item["reference_parameters"])
    _string(
        item["aperture_semantics"], f"{path}.aperture_semantics",
        choices={"jaw_separation"},
    )


def _validate_geometry_projection(value: Any, name: str) -> None:
    """Check ``workspace_limits`` or ``collision_model`` against its vocabulary."""

    section = _object(value, f"descriptor.geometry.{name}")
    _exact_keys(
        section, required={"source", "projection"}, optional=set(),
        path=f"descriptor.geometry.{name}",
    )
    _string(
        section["source"], f"descriptor.geometry.{name}.source",
        choices={"descriptor", "runtime", "observation", "tool_surface"},
    )
    vocabulary = (
        set(_WORKSPACE_PROJECTION_TEXT)
        if name == "workspace_limits" else set(_COLLISION_PROJECTION_TEXT)
    )
    _string(
        section["projection"], f"descriptor.geometry.{name}.projection",
        choices=vocabulary,
    )


def _validate_sensing(
    value: Any, *, manipulator_ids: set[str], effector_ids: set[str],
) -> None:
    sensing = _object(value, "descriptor.sensing")
    _exact_keys(
        sensing, required={"cameras", "state_fields"}, optional=set(),
        path="descriptor.sensing",
    )
    _validate_cameras(
        sensing["cameras"], manipulator_ids=manipulator_ids, effector_ids=effector_ids,
    )
    _validate_state_fields(sensing["state_fields"])


def _validate_cameras(
    value: Any, *, manipulator_ids: set[str], effector_ids: set[str],
) -> None:
    cameras = _objects(value, "descriptor.sensing.cameras", allow_empty=True)
    camera_names: set[str] = set()
    for index, item in enumerate(cameras):
        path = f"descriptor.sensing.cameras[{index}]"
        _exact_keys(
            item,
            required={
                "public_name", "source_name", "mounting", "attached_to",
                "modalities", "calibration",
            },
            optional=set(), path=path,
        )
        public_name = _policy_identifier(item["public_name"], f"{path}.public_name")
        if public_name in camera_names:
            raise EmbodimentDescriptorError(f"{path}.public_name is duplicated")
        camera_names.add(public_name)
        _string(item["source_name"], f"{path}.source_name")
        _string(
            item["mounting"], f"{path}.mounting",
            choices={"fixed_world", "base", "manipulator", "end_effector", "other"},
        )
        attachment = _identifier(item["attached_to"], f"{path}.attached_to")
        if attachment not in {"world", *manipulator_ids, *effector_ids}:
            raise EmbodimentDescriptorError(f"{path}.attached_to is unknown")
        modalities = _strings(item["modalities"], f"{path}.modalities")
        unknown_modalities = set(modalities) - {"rgb", "metric_depth"}
        if unknown_modalities:
            raise EmbodimentDescriptorError(
                f"{path}.modalities has unknown values {sorted(unknown_modalities)}"
            )
        if any(
            token in modality.lower()
            for modality in modalities for token in _PRIVATE_TOKENS
        ):
            raise EmbodimentDescriptorError(f"{path}.modalities includes private evidence")
        _string(
            item["calibration"], f"{path}.calibration",
            choices={"per_observation", "per_session", "static", "unavailable"},
        )


def _validate_state_fields(value: Any) -> None:
    state_fields = _objects(
        value, "descriptor.sensing.state_fields", allow_empty=True,
    )
    state_names: set[str] = set()
    for index, item in enumerate(state_fields):
        path = f"descriptor.sensing.state_fields[{index}]"
        _exact_keys(
            item, required={"name", "meaning", "unit", "source"}, optional={"joint_names"},
            path=path,
        )
        if "joint_names" in item:
            names = _strings(item["joint_names"], f"{path}.joint_names")
            if item["meaning"] != "rotary_joint_positions" or item["unit"] != "radian":
                raise EmbodimentDescriptorError("joint_names requires rotary positions in radians")
            for joint_name in names:
                _policy_identifier(joint_name, f"{path}.joint_names")
        name = _policy_identifier(item["name"], f"{path}.name")
        if name in state_names:
            raise EmbodimentDescriptorError(f"{path}.name is duplicated")
        state_names.add(name)
        _string(
            item["meaning"], f"{path}.meaning", choices=set(_STATE_FIELD_TEXT),
        )
        _string(
            item["unit"], f"{path}.unit", choices=set(_STATE_UNIT_TEXT),
        )
        _string(
            item["source"], f"{path}.source",
            choices={"observation", "tool_result", "runtime"},
        )


def _validate_dynamic_facts(value: Any) -> None:
    dynamic_facts = _objects(value, "descriptor.dynamic_facts", allow_empty=True)
    dynamic_names: set[str] = set()
    for index, item in enumerate(dynamic_facts):
        path = f"descriptor.dynamic_facts[{index}]"
        _exact_keys(item, required={"name", "owner", "reason"}, optional=set(), path=path)
        name = _policy_identifier(item["name"], f"{path}.name")
        if name in dynamic_names:
            raise EmbodimentDescriptorError(f"{path}.name is duplicated")
        dynamic_names.add(name)
        _string(
            item["owner"], f"{path}.owner",
            choices={"runtime", "observation", "tool_surface"},
        )
        _string(
            item["reason"], f"{path}.reason", choices=set(_DYNAMIC_FACT_TEXT),
        )


def _validate_evidence_limits(value: Any) -> None:
    evidence_limits = _strings(value, "descriptor.evidence_limits")
    unknown_limits = set(evidence_limits) - set(_EVIDENCE_LIMIT_TEXT)
    if unknown_limits:
        raise EmbodimentDescriptorError(
            f"descriptor.evidence_limits has unknown values {sorted(unknown_limits)}"
        )


def _validate_provenance(root: Mapping[str, Any], *, geometry_v2: bool) -> None:
    """Check every source, then that the sources cover every projected fact."""

    provenance = _object(root["provenance"], "descriptor.provenance")
    _exact_keys(
        provenance, required={"runtime_truth", "sources"},
        optional=set(), path="descriptor.provenance",
    )
    _string(
        provenance["runtime_truth"], "descriptor.provenance.runtime_truth",
        choices={"compiled_model", "hardware_interface", "synthetic_contract"},
    )
    sources = _objects(provenance["sources"], "descriptor.provenance.sources")
    projected_facts = _projected_fact_inventory(root)
    supported_facts: set[str] = set()
    source_ids: set[str] = set()
    for index, item in enumerate(sources):
        supports = _validate_source(
            item, f"descriptor.provenance.sources[{index}]",
            geometry_v2=geometry_v2, projected_facts=projected_facts,
            source_ids=source_ids,
        )
        supported_facts.update(supports)
    missing_supports = set(projected_facts) - supported_facts
    if missing_supports:
        raise EmbodimentDescriptorError(
            "descriptor.provenance does not cover projected facts "
            f"{sorted(missing_supports)}"
        )


def _validate_source(
    item: Mapping[str, Any], path: str, *, geometry_v2: bool,
    projected_facts: Mapping[str, Any], source_ids: set[str],
) -> list[str]:
    """Check one source, add its id to ``source_ids``; return its supported facts.

    A repository source (``local_path``) is named by path only: the RunPlan's
    source commit identifies its bytes (and ``policy.lock`` a policy asset's),
    so a hand-written ``revision`` or ``sha256`` beside it is refused, never
    verified. An external source carries its ``revision`` and ``sha256``.
    """

    if "local_path" in item:
        pinned = sorted({"revision", "sha256"} & set(item))
        if pinned:
            raise EmbodimentDescriptorError(
                f"{path} names a repository source by path only; {pinned} are not accepted "
                "(the RunPlan source commit identifies repository bytes)"
            )
        _exact_keys(
            item, required={"source_id", "kind", "uri", "local_path", "supports"},
            optional=set(), path=path,
        )
    else:
        _exact_keys(
            item, required={"source_id", "kind", "uri", "revision", "sha256", "supports"},
            optional=set(), path=path,
        )
    source_id = _identifier(item["source_id"], f"{path}.source_id")
    if source_id in source_ids:
        raise EmbodimentDescriptorError(f"{path}.source_id is duplicated")
    source_ids.add(source_id)
    _string(item["kind"], f"{path}.kind", choices=_SOURCE_KINDS | (
        {"robot_urdf", "robot_usd", "collision_mesh", "robot_asset_verification"} if geometry_v2 else set()
    ))
    _string(item["uri"], f"{path}.uri")
    if "local_path" in item:
        local_path = _string(item["local_path"], f"{path}.local_path")
        candidate = Path(local_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise EmbodimentDescriptorError(
                f"{path}.local_path must be project-relative"
            )
    else:
        _string(item["revision"], f"{path}.revision")
        digest = _string(item["sha256"], f"{path}.sha256")
        if digest != _PLACEHOLDER and not _SHA256.fullmatch(digest):
            raise EmbodimentDescriptorError(f"{path}.sha256 must be lowercase SHA-256")
    supports = _strings(item["supports"], f"{path}.supports")
    unknown_supports = set(supports) - set(projected_facts)
    if unknown_supports:
        raise EmbodimentDescriptorError(
            f"{path}.supports names unknown projected facts "
            f"{sorted(unknown_supports)}"
        )
    return supports
