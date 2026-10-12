"""Validate and compile policy-safe embodiment descriptors into short cards."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .embodiment_control_card import (
    _CONTROLLED_POINT_TEXT, _EVIDENCE_LIMIT_TEXT, _STATE_FIELD_TEXT,
    _STATE_UNIT_TEXT, render_control_only_card,
)


DESCRIPTOR_SCHEMA = "ora.embodiment-descriptor.v1"
DESCRIPTOR_SCHEMA_V2 = "ora.embodiment-descriptor.v2"
COMPILED_CARD_SCHEMA = "ora.compiled-embodiment-card.v1"
ACTIVATION_RECORD_SCHEMA = "ora.embodiment-activation-record.v1"
RUNTIME_VERIFICATION_SCHEMA = "ora.embodiment-runtime-verification.v1"
REVIEW_TOKEN_LIMIT = 700
_PLACEHOLDER = "__REQUIRED__"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")
_PRIVATE_TOKENS = {
    "actor_id", "bddl", "contact", "demonstration", "expert",
    "ground_truth", "object_pose", "predicate", "segmentation", "task_source",
}

_ROOT_KEYS = {
    "schema", "descriptor_id", "embodiment_id", "display_name", "status",
    "identity", "topology", "control", "geometry", "sensing",
    "dynamic_facts", "evidence_limits", "provenance",
}
_WORKSPACE_PROJECTION_TEXT = {
    "tool_surface_enforced_bounds": (
        "Use the current ToolSurface-enforced bounds; do not assume a universal reach box."
    ),
    "runtime_enforced_bounds": (
        "Use the current Runtime-enforced bounds; do not assume a universal reach box."
    ),
}
_COLLISION_PROJECTION_TEXT = {
    "bounded_motion_execution_evidence": (
        "Use bounded motions and current execution evidence; source geometry is not a collision guarantee."
    ),
}
_DYNAMIC_FACT_TEXT = {
    "camera_calibration": "camera calibration",
    "robot_base_pose": "robot base pose",
    "workspace_bounds": "workspace bounds",
}
_SOURCE_KINDS = {
    "robot_mjcf", "gripper_mjcf", "controller_config",
    "environment_robot_binding", "environment_runtime_adapter",
    "policy_runtime_adapter", "tool_surface_contract",
}


class EmbodimentDescriptorError(RuntimeError):
    """One descriptor or compiled projection violates the public contract."""


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], path: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing:
        raise EmbodimentDescriptorError(f"{path} lacks required fields {sorted(missing)}")
    if unknown:
        raise EmbodimentDescriptorError(f"{path} has unknown fields {sorted(unknown)}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EmbodimentDescriptorError(f"{path} must be an object")
    return value


def _objects(value: Any, path: str, *, allow_empty: bool = False) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise EmbodimentDescriptorError(f"{path} must be {qualifier}")
    if not all(isinstance(item, Mapping) for item in value):
        raise EmbodimentDescriptorError(f"{path} must contain only objects")
    return value


def _strings(value: Any, path: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "an array" if allow_empty else "a non-empty array"
        raise EmbodimentDescriptorError(f"{path} must be {qualifier}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise EmbodimentDescriptorError(f"{path} must contain non-empty strings")
    if len(value) != len(set(value)):
        raise EmbodimentDescriptorError(f"{path} must not contain duplicates")
    return value


def _string(value: Any, path: str, *, choices: set[str] | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmbodimentDescriptorError(f"{path} must be a non-empty string")
    if choices is not None and value != _PLACEHOLDER and value not in choices:
        raise EmbodimentDescriptorError(f"{path} must be one of {sorted(choices)}")
    return value


def _identifier(value: Any, path: str) -> str:
    selected = _string(value, path)
    if selected != _PLACEHOLDER and not _IDENTIFIER.fullmatch(selected):
        raise EmbodimentDescriptorError(f"{path} must be a snake-case identifier")
    return selected


def _policy_identifier(value: Any, path: str) -> str:
    selected = _identifier(value, path)
    lowered = selected.lower()
    if any(token in lowered for token in _PRIVATE_TOKENS):
        raise EmbodimentDescriptorError(f"{path} names policy-private evidence")
    return selected


def _positive_integer(value: Any, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise EmbodimentDescriptorError(f"{path} must be a positive integer")
    return value


def _nonnegative_number(value: Any, path: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or not float(value) >= 0.0
    ):
        raise EmbodimentDescriptorError(f"{path} must be a non-negative number")
    return float(value)


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return value == _PLACEHOLDER


def _projected_fact_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete, typed fact inventory that can affect rendered text."""

    facts: dict[str, Any] = {
        "identity.base_mobility": value["identity"]["base_mobility"],
    }
    for item in value["topology"]["manipulators"]:
        facts[f"topology.manipulator.{item['id']}"] = dict(item)
    for item in value["topology"]["end_effectors"]:
        facts[f"topology.end_effector.{item['id']}"] = dict(item)
    for item in value["control"]["controlled_points"]:
        facts[f"embodiment_control.controlled_point.{item['id']}"] = dict(item)
    for name in (
        "action_abstraction", "position_frame", "position_unit",
        "orientation", "internal_controller_policy",
    ):
        facts[f"embodiment_control.{name}"] = copy.deepcopy(value["control"][name])
    for item in value["geometry"]["grippers"]:
        facts[f"geometry.gripper.{item['end_effector_id']}"] = dict(item)
    if value["geometry"].get("card_projection") == "control_only_nine_node_v1":
        facts["geometry.card_projection"] = value["geometry"]["card_projection"]
        facts["identity.display_name"] = value["display_name"]
    facts["geometry.workspace_limits"] = dict(value["geometry"]["workspace_limits"])
    facts["geometry.collision_model"] = dict(value["geometry"]["collision_model"])
    for item in value["sensing"]["cameras"]:
        facts[f"sensing.camera.{item['public_name']}"] = dict(item)
    for item in value["sensing"]["state_fields"]:
        facts[f"sensing.state.{item['name']}"] = dict(item)
    for item in value["dynamic_facts"]:
        facts[f"dynamic.{item['name']}"] = dict(item)
    return facts


def validate_embodiment_descriptor(value: Any) -> dict[str, Any]:
    """Return a deep canonical copy after strict, privilege-clean validation.

    The section steps live in ``embodiment_validation.py`` and run in this
    order; topology and control return the ids later sections cross-check.
    """

    from .embodiment_validation import (
        _validate_control, _validate_dynamic_facts, _validate_evidence_limits,
        _validate_geometry, _validate_header, _validate_identity,
        _validate_provenance, _validate_sensing, _validate_topology,
    )

    root = _object(value, "descriptor")
    geometry_v2 = _validate_header(root)
    _validate_identity(root["identity"])
    manipulator_ids, effector_ids = _validate_topology(root["topology"])
    point_ids = _validate_control(
        root["control"], geometry_v2=geometry_v2, manipulator_ids=manipulator_ids,
    )
    _validate_geometry(
        root["geometry"], geometry_v2=geometry_v2, effector_ids=effector_ids,
        point_ids=point_ids,
    )
    _validate_sensing(
        root["sensing"], manipulator_ids=manipulator_ids, effector_ids=effector_ids,
    )
    _validate_dynamic_facts(root["dynamic_facts"])
    _validate_evidence_limits(root["evidence_limits"])
    _validate_provenance(root, geometry_v2=geometry_v2)
    return copy.deepcopy(dict(root))


def load_embodiment_descriptor(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EmbodimentDescriptorError(f"cannot read descriptor {path}: {exc}") from exc
    return validate_embodiment_descriptor(value)


def audit_embodiment_sources(
    value: Mapping[str, Any], *, project_root: Path,
) -> dict[str, Any]:
    """Check that every provenance source is present where the descriptor says.

    A repository source must be a file inside ``project_root``; its bytes are
    identified by the source commit the RunPlan records, not by a pin here, so
    an edit to it never invalidates the descriptor. An external source keeps
    its content address, recorded but not readable from this tree.
    """

    descriptor = validate_embodiment_descriptor(value)
    root = project_root.resolve()
    checks = []
    for source in descriptor["provenance"]["sources"]:
        local_path = source.get("local_path")
        if local_path is None:
            checks.append({
                "source_id": source["source_id"],
                "kind": source["kind"],
                "status": "content_addressed_external",
                "expected_sha256": source["sha256"],
                "observed_sha256": None,
            })
            continue
        path = (root / local_path).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise EmbodimentDescriptorError(
                f"provenance source is unavailable: {local_path}"
            )
        checks.append({
            "source_id": source["source_id"],
            "kind": source["kind"],
            "status": "repository_source",
            "local_path": local_path,
        })
    return {
        "schema": "ora.embodiment-source-audit.v1",
        "descriptor_sha256": canonical_sha256(descriptor),
        "projected_facts_sha256": canonical_sha256(
            _projected_fact_inventory(descriptor)
        ),
        "fact_count": len(_projected_fact_inventory(descriptor)),
        "checks": checks,
        "passed": all(item["status"] != "failed" for item in checks),
    }


def validate_activation_record(
    record: Any, *, descriptor: Mapping[str, Any], card: str,
) -> dict[str, Any]:
    """Validate a separately bound review/runtime attestation.

    The descriptor cannot certify itself.  A launchable profile must bind this
    immutable record and the Runtime evidence artifact named by its hash.
    """

    value = _object(record, "activation_record")
    _exact_keys(
        value,
        required={
            "schema", "descriptor_sha256", "card_sha256", "review",
            "runtime_verification",
        },
        optional=set(), path="activation_record",
    )
    if value["schema"] != ACTIVATION_RECORD_SCHEMA:
        raise EmbodimentDescriptorError("activation_record.schema is invalid")
    expected_descriptor = canonical_sha256(descriptor)
    expected_card = hashlib.sha256(card.encode("utf-8")).hexdigest()
    if value["descriptor_sha256"] != expected_descriptor:
        raise EmbodimentDescriptorError("activation record descriptor hash mismatch")
    if value["card_sha256"] != expected_card:
        raise EmbodimentDescriptorError("activation record card hash mismatch")
    review = _object(value["review"], "activation_record.review")
    _exact_keys(
        review, required={"decision", "card_sha256", "record_sha256"},
        optional=set(), path="activation_record.review",
    )
    _string(review["decision"], "activation_record.review.decision", choices={"accepted"})
    for name in ("card_sha256", "record_sha256"):
        digest = _string(review[name], f"activation_record.review.{name}")
        if not _SHA256.fullmatch(digest):
            raise EmbodimentDescriptorError(
                f"activation_record.review.{name} must be lowercase SHA-256"
            )
    if review["card_sha256"] != expected_card:
        raise EmbodimentDescriptorError("activation review card hash mismatch")
    runtime = _object(
        value["runtime_verification"], "activation_record.runtime_verification",
    )
    _exact_keys(
        runtime,
        required={
            "schema", "descriptor_sha256", "projected_facts_sha256",
            "evidence_sha256",
        },
        optional=set(), path="activation_record.runtime_verification",
    )
    if runtime["schema"] != RUNTIME_VERIFICATION_SCHEMA:
        raise EmbodimentDescriptorError("runtime verification schema is invalid")
    if runtime["descriptor_sha256"] != expected_descriptor:
        raise EmbodimentDescriptorError("runtime verification descriptor hash mismatch")
    expected_facts = canonical_sha256(_projected_fact_inventory(descriptor))
    if runtime["projected_facts_sha256"] != expected_facts:
        raise EmbodimentDescriptorError("runtime verification fact hash mismatch")
    evidence_digest = _string(
        runtime["evidence_sha256"],
        "activation_record.runtime_verification.evidence_sha256",
    )
    if not _SHA256.fullmatch(evidence_digest):
        raise EmbodimentDescriptorError(
            "runtime verification evidence_sha256 must be lowercase SHA-256"
        )
    return copy.deepcopy(dict(value))


def descriptor_activation_ready(
    value: Mapping[str, Any], *, activation_record: Mapping[str, Any] | None = None,
) -> bool:
    descriptor = validate_embodiment_descriptor(value)
    if _contains_placeholder(descriptor) or activation_record is None:
        return False
    card = render_embodiment_card(descriptor)
    validate_activation_record(activation_record, descriptor=descriptor, card=card)
    return True


def _human_join(items: Sequence[str]) -> str:
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _format_length_m(value: float) -> str:
    millimetres = value * 1000.0
    if abs(millimetres - round(millimetres)) < 1e-9:
        return f"{int(round(millimetres))} mm"
    return f"{millimetres:.1f} mm"



def render_embodiment_card(value: Mapping[str, Any]) -> str:
    """Render only the normalized policy projection; omit source provenance."""

    descriptor = validate_embodiment_descriptor(value)
    if _contains_placeholder(descriptor):
        raise EmbodimentDescriptorError("descriptor contains unresolved template placeholders")
    if descriptor["geometry"].get("card_projection") == "control_only_nine_node_v1":
        return render_control_only_card(descriptor)
    if descriptor["schema"] == DESCRIPTOR_SCHEMA_V2:
        from .embodiment_geometry import render_geometry_card
        # Full XYZ and nominal/encoder distinctions must not be dropped to fit
        # the compact predecessor's review target. Explicit versioned cards
        # opt into this still-bounded budget; old compiled bytes stay fixed.
        limit = (1200 if descriptor["geometry"].get("card_projection")
                 in {"full_finger_reference_v1", "full_finger_reference_v2", "full_finger_reference_v3"}
                 else REVIEW_TOKEN_LIMIT)
        return render_geometry_card(descriptor, token_limit=limit)
    topology = descriptor["topology"]
    control = descriptor["control"]
    geometry = descriptor["geometry"]
    sensing = descriptor["sensing"]

    manipulators = topology["manipulators"]
    arms = [
        f"a {item['joint_dof']}-DoF {item['kind'].replace('_', ' ')}"
        for item in manipulators
    ]
    base = descriptor["identity"]["base_mobility"].replace("_", " ")
    lines = [
        "# Embodiment Card v1",
        "",
        "## Body and command frame",
        "",
        (
            f"This deployment uses {_human_join(arms)} on a {base} base."
        ),
    ]
    for point in control["controlled_points"]:
        sentence = (
            f"Policy motion controls `{point['id']}`: "
            f"{_CONTROLLED_POINT_TEXT[point['description']]}"
        )
        if "offset_m" in point:
            x, y, z = point["offset_m"]
            sentence += (
                ". Its model offset from the declared "
                f"{point['offset_from'].replace('_', ' ')} is "
                f"`[{x:g}, {y:g}, {z:g}] m` in that local frame"
            )
            sentence += "; policy commands already target this point, so do not add the offset again"
        lines.append(sentence + ".")
    orientation = control["orientation"]
    lines.extend([
        (
            f"Position values use {control['position_unit']} in the "
            f"{control['position_frame']} frame. Measured controlled-point poses use "
            f"{orientation['representation'].replace('_', ' ')} "
            f"order `[{','.join(orientation['order'])}]`; advertised tools define "
            "their own action-orientation arguments."
        ),
        (
            "The resolved ToolSurface owns exact action arguments, enforced limits "
            "and result thresholds."
        ),
        "",
        "## Cameras and policy-visible state",
        "",
    ])
    if sensing["cameras"]:
        for camera in sensing["cameras"]:
            mounting = {
                "fixed_world": "fixed in the scene",
                "base": "fixed to the robot base",
                "manipulator": "moving with a manipulator",
                "end_effector": "moving with an end effector",
                "other": "mounted on the declared robot body",
            }[camera["mounting"]]
            modalities = _human_join([item.replace("_", " ") for item in camera["modalities"]])
            calibration = {
                "per_observation": "calibration supplied with each observation",
                "per_session": "calibration supplied for the current session",
                "static": "the declared static calibration",
                "unavailable": "no geometric calibration",
            }[camera["calibration"]]
            modalities = modalities.replace("rgb", "RGB")
            lines.append(
                f"- `{camera['public_name']}` is {mounting} and provides {modalities}; "
                f"use {calibration}."
            )
    else:
        lines.append("No policy-visible camera is declared by this descriptor.")
    for field in sensing["state_fields"]:
        lines.append(
            f"- `{field['name']}` reports {_STATE_FIELD_TEXT[field['meaning']]} in "
            f"{_STATE_UNIT_TEXT[field['unit']]} "
            f"from the current {field['source'].replace('_', ' ')}."
        )

        if "joint_names" in field:
            lines.append("  Joint order: `" + ", ".join(field["joint_names"]) + "`.")

    lines.extend([
        "",
        "## Motion and geometry",
        "",
        f"The policy action abstraction is {control['action_abstraction'].replace('_', ' ')}.",
    ])
    if control["internal_controller_policy"] == "tool_surface_owned":
        lines.append(
            "Internal controller increments and gains are not the policy action contract."
        )
    lines.append(_WORKSPACE_PROJECTION_TEXT[geometry["workspace_limits"]["projection"]])
    lines.append(_COLLISION_PROJECTION_TEXT[geometry["collision_model"]["projection"]])
    if descriptor["dynamic_facts"]:
        names = _human_join([
            f"{_DYNAMIC_FACT_TEXT[item['reason']]} (`{item['name']}`)"
            for item in descriptor["dynamic_facts"]
        ])
        lines.append(
            f"Treat {names} as dynamic facts from the current Runtime, observation or ToolSurface."
        )

    if geometry["grippers"]:
        lines.extend(["", "## End effectors", ""])
        for gripper in geometry["grippers"]:
            aperture = _format_length_m(float(gripper["maximum_aperture_m"]))
            lines.append(
                f"- `{gripper['end_effector_id']}` has {gripper['opening_mode']} opening "
                f"with nominal maximum aperture `{aperture}`; "
                "aperture means full jaw separation."
            )

    lines.extend(["", "## Evidence limits", ""])
    for name in descriptor["evidence_limits"]:
        lines.append(f"- {_EVIDENCE_LIMIT_TEXT[name]}")
    rendered = "\n".join(lines).replace("  ", " ").rstrip() + "\n"
    estimate = (len(rendered.encode("utf-8")) + 3) // 4
    if estimate > REVIEW_TOKEN_LIMIT:
        raise EmbodimentDescriptorError(
            f"compiled card estimate {estimate} exceeds {REVIEW_TOKEN_LIMIT} tokens"
        )
    return rendered


def compile_embodiment_card(
    value: Mapping[str, Any], *, source_file_sha256: str | None = None,
    activation_record: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    descriptor = validate_embodiment_descriptor(value)
    card = render_embodiment_card(descriptor)
    encoded = card.encode("utf-8")
    manifest = {
        "schema": COMPILED_CARD_SCHEMA,
        "descriptor_schema": descriptor["schema"],
        "descriptor_id": descriptor["descriptor_id"],
        "embodiment_id": descriptor["embodiment_id"],
        "descriptor_sha256": canonical_sha256(descriptor),
        "source_file_sha256": source_file_sha256,
        "card_sha256": hashlib.sha256(encoded).hexdigest(),
        "card_bytes": len(encoded),
        "harness_token_estimate": (len(encoded) + 3) // 4,
        "review_token_limit": REVIEW_TOKEN_LIMIT,
        "descriptor_status": descriptor["status"],
        "projected_facts_sha256": canonical_sha256(
            _projected_fact_inventory(descriptor)
        ),
        "activation_record_schema": ACTIVATION_RECORD_SCHEMA,
        "activation_record_bound": activation_record is not None,
        "activation_ready": descriptor_activation_ready(
            descriptor, activation_record=activation_record,
        ),
        "provenance_model_visible": False,
        "free_form_descriptor_text_model_visible": False,
    }
    return card, manifest


def descriptor_template(
    *, descriptor_id: str, embodiment_id: str, display_name: str,
) -> dict[str, Any]:
    """Return one structurally valid draft with explicit unresolved fields."""

    return {
        "schema": DESCRIPTOR_SCHEMA,
        "descriptor_id": descriptor_id,
        "embodiment_id": embodiment_id,
        "display_name": display_name,
        "status": "draft",
        "identity": {
            "manufacturer": _PLACEHOLDER,
            "model": _PLACEHOLDER,
            "variant": _PLACEHOLDER,
            "base_mobility": "fixed",
        },
        "topology": {
            "manipulators": [{
                "id": "primary_arm", "kind": "serial_arm", "side": "none",
                "joint_dof": 1,
            }],
            "end_effectors": [{
                "id": "primary_effector", "kind": _PLACEHOLDER,
                "manipulator_id": "primary_arm", "command_dof": 1,
            }],
        },
        "control": {
            "controlled_points": [{
                "id": "primary_controlled_point", "manipulator_id": "primary_arm",
                "source_name": _PLACEHOLDER, "description": "tool_center",
            }],
            "action_abstraction": "cartesian_target",
            "position_frame": "world", "position_unit": "metre",
            "orientation": {
                "representation": "quaternion", "order": "xyzw",
                "angle_unit": "radian",
            },
            "controller_family": _PLACEHOLDER,
            "internal_controller_policy": "tool_surface_owned",
        },
        "geometry": {
            "grippers": [],
            "workspace_limits": {
                "source": "runtime",
                "projection": "runtime_enforced_bounds",
            },
            "collision_model": {
                "source": "runtime",
                "projection": "bounded_motion_execution_evidence",
            },
        },
        "sensing": {"cameras": [], "state_fields": []},
        "dynamic_facts": [],
        "evidence_limits": [
            "motion_completion_not_contact",
            "geometry_not_collision_guarantee",
        ],
        "provenance": {
            "runtime_truth": "compiled_model",
            "sources": [{
                "source_id": "primary_source", "kind": _PLACEHOLDER,
                "uri": _PLACEHOLDER,
                "revision": _PLACEHOLDER, "sha256": _PLACEHOLDER,
                "supports": [
                    "identity.base_mobility",
                    "topology.manipulator.primary_arm",
                    "topology.end_effector.primary_effector",
                    "embodiment_control.controlled_point.primary_controlled_point",
                    "embodiment_control.action_abstraction",
                    "embodiment_control.position_frame",
                    "embodiment_control.position_unit",
                    "embodiment_control.orientation",
                    "embodiment_control.internal_controller_policy",
                    "geometry.workspace_limits", "geometry.collision_model",
                ],
            }],
        },
    }


def _atomic_text(path: Path, text: str, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text and not replace:
        raise EmbodimentDescriptorError(
            f"refusing to replace drifted output without --replace: {path}"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def compiled_card_files(
    descriptor_bytes: bytes, *, project_root: Path,
) -> tuple[str, dict[str, Any], str]:
    """The card, manifest and manifest text ``write_compiled_card`` writes for these bytes."""

    try:
        value = json.loads(descriptor_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EmbodimentDescriptorError(f"cannot read descriptor: {exc}") from exc
    descriptor = validate_embodiment_descriptor(value)
    source_audit = audit_embodiment_sources(descriptor, project_root=project_root)
    card, manifest = compile_embodiment_card(
        descriptor, source_file_sha256=hashlib.sha256(descriptor_bytes).hexdigest(),
    )
    manifest["source_audit"] = source_audit
    manifest_text = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n"
    return card, manifest, manifest_text


def write_compiled_card(
    descriptor_path: Path, output_dir: Path, *, replace: bool = False,
    project_root: Path | None = None,
) -> dict[str, Any]:
    try:
        descriptor_bytes = descriptor_path.read_bytes()
    except OSError as exc:
        raise EmbodimentDescriptorError(f"cannot read descriptor {descriptor_path}: {exc}") from exc
    card, manifest, manifest_text = compiled_card_files(
        descriptor_bytes, project_root=Path.cwd() if project_root is None else project_root,
    )
    _atomic_text(output_dir / "manual.md", card, replace=replace)
    _atomic_text(output_dir / "manifest.json", manifest_text, replace=replace)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compile-embodiment-card",
        description=(
            "Create, validate and deterministically compile policy-safe robot "
            "embodiment descriptors."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="write a new draft descriptor")
    template.add_argument("output", type=Path)
    template.add_argument("--descriptor-id", required=True)
    template.add_argument("--embodiment-id", required=True)
    template.add_argument("--display-name", required=True)
    template.add_argument("--replace", action="store_true")
    validate = commands.add_parser("validate", help="validate one descriptor")
    validate.add_argument("descriptor", type=Path)
    preview = commands.add_parser("preview", help="print the exact model-visible card")
    preview.add_argument("descriptor", type=Path)
    compile_command = commands.add_parser(
        "compile", help="write manual.md and manifest.json to an output directory",
    )
    compile_command.add_argument("descriptor", type=Path)
    compile_command.add_argument("--output-dir", type=Path, required=True)
    compile_command.add_argument("--replace", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "template":
            value = descriptor_template(
                descriptor_id=args.descriptor_id,
                embodiment_id=args.embodiment_id,
                display_name=args.display_name,
            )
            validate_embodiment_descriptor(value)
            _atomic_text(
                args.output,
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                replace=args.replace,
            )
            print(args.output)
            return 0
        descriptor = load_embodiment_descriptor(args.descriptor)
        if args.command == "validate":
            source_audit = audit_embodiment_sources(
                descriptor, project_root=Path.cwd(),
            )
            print(json.dumps({
                "schema": DESCRIPTOR_SCHEMA,
                "descriptor_id": descriptor["descriptor_id"],
                "descriptor_sha256": canonical_sha256(descriptor),
                "activation_ready": descriptor_activation_ready(descriptor),
                "source_audit": source_audit,
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.command == "preview":
            print(render_embodiment_card(descriptor), end="")
            return 0
        manifest = write_compiled_card(
            args.descriptor, args.output_dir, replace=args.replace,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except EmbodimentDescriptorError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())
