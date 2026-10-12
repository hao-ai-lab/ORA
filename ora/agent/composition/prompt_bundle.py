"""Static authoring and delivery audit for modular robot prompt bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .embodiment import (
    EmbodimentDescriptorError,
    audit_embodiment_sources,
    load_embodiment_descriptor,
)
from .environment_context import (
    EnvironmentContextError,
    resolve_environment_prompt_profile,
)
from .tool_surface import (
    TOOL_HELP_CONTRACT_CAPABILITY,
    ToolSurfaceError,
    resolve_tool_surface,
)


PROMPT_BUNDLE_SCHEMA = "ora.prompt-bundle-authoring.v1"
PROMPT_BUNDLE_AUDIT_SCHEMA = "ora.prompt-bundle-audit.v1"
PROMPT_BUNDLE_EMBODIMENT_FORMAT = "prompt_bundle_embodiment_v1"
PROMPT_BUNDLE_LAYERS = (
    "planner",
    "core",
    "environment",
    "embodiment",
    "episode_task",
    "tool_manual",
    "tool_surface",
    "dynamic_runtime",
)
EMBODIMENT_SECTION_HEADINGS = (
    "Identity and topology",
    "Controlled points",
    "Frames and units",
    "End-effector axes",
    "Motion capabilities",
    "Gripper semantics",
    "Sensors and cameras",
    "Preservation and evidence limits",
    "Dynamic facts and provenance",
)

_PLACEHOLDER = "__REQUIRED__"
_RUNTIME_KEYS = {"policy_kind", "context_schema", "provider_scaffold"}
_LAYER_KEYS = set(PROMPT_BUNDLE_LAYERS)
_BINDING_KEYS = {"environment_prompt_profile"}
_PROFILE_BINDING_KEYS = {"profile_id", "public_environment_context"}
_OWNERSHIP_KEYS = {"environment_terms", "embodiment_terms"}
_PATH_LIST_KEYS = {"sources", "delivery"}
_NATIVE_DELIVERY = {
    "planner": "system",
    "core": "project_instructions",
    "environment": "project_instructions",
    "embodiment": "project_instructions",
    "episode_task": "first_user_message",
    "tool_surface": "provider_tools_field",
    "dynamic_runtime": "current_user_tail",
}
_MANAGED_DELIVERY = {
    "planner": "managed_project_document",
    "core": "managed_project_document",
    "environment": "managed_project_document",
    "embodiment": "managed_project_document",
    "episode_task": "managed_episode_input",
    "tool_surface": "managed_mcp_session",
    "dynamic_runtime": "managed_outer_turn",
}


class PromptBundleError(RuntimeError):
    """A prompt bundle is malformed, incomplete, or semantically inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptBundleError(f"{label} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise PromptBundleError(
            f"{label} keys differ: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PromptBundleError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _strings(value: Any, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise PromptBundleError(f"{label} must be a {'possibly empty ' if allow_empty else ''}list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise PromptBundleError(f"{label} must contain non-empty strings")
    values = [item.strip() for item in value]
    if len(values) != len(set(values)):
        raise PromptBundleError(f"{label} contains duplicates")
    return values


def _path(project_root: Path, value: Any, label: str) -> tuple[str, Path]:
    relative = Path(_string(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise PromptBundleError(f"{label} must remain project-relative")
    resolved = (project_root / relative).resolve()
    if not resolved.is_file() or not resolved.is_relative_to(project_root):
        raise PromptBundleError(f"{label} is unavailable: {relative.as_posix()}")
    return relative.as_posix(), resolved


def _source_record(project_root: Path, value: Any, label: str) -> dict[str, Any]:
    relative, resolved = _path(project_root, value, label)
    text = resolved.read_text(encoding="utf-8")
    if _PLACEHOLDER in text:
        raise PromptBundleError(f"{label} contains unresolved {_PLACEHOLDER}")
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "estimated_tokens": max(1, (resolved.stat().st_size + 3) // 4),
        "sha256": _file_sha256(resolved),
        "text": text,
    }


def _source_records(
    project_root: Path, values: Any, label: str,
) -> list[dict[str, Any]]:
    return [
        _source_record(project_root, value, f"{label}[{index}]")
        for index, value in enumerate(_strings(values, label))
    ]


def _term_present(text: str, term: str) -> bool:
    normalized = " ".join(text.split())
    pattern = rf"(?<![\w]){re.escape(' '.join(term.split()))}(?![\w])"
    return re.search(pattern, normalized, flags=re.IGNORECASE) is not None


def _present_terms(text: str, terms: Sequence[str]) -> list[str]:
    return sorted(term for term in terms if _term_present(text, term))


def _check(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _long_sentences(text: str) -> set[str]:
    sentences: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip().lstrip("-* ")
        if not stripped or stripped.startswith("#"):
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            normalized = " ".join(sentence.lower().split())
            if len(normalized.split()) >= 12:
                sentences.add(normalized)
    return sentences


def _validate_layer_shapes(layers: Mapping[str, Any]) -> None:
    _exact_keys(layers, _LAYER_KEYS, "layers")
    for name in ("planner", "core"):
        layer = _object(layers[name], f"layers.{name}")
        _exact_keys(layer, _PATH_LIST_KEYS, f"layers.{name}")
        _strings(layer["sources"], f"layers.{name}.sources")
        _string(layer["delivery"], f"layers.{name}.delivery")
    environment = _object(layers["environment"], "layers.environment")
    _exact_keys(environment, {"source", "delivery"}, "layers.environment")
    _string(environment["source"], "layers.environment.source")
    _string(environment["delivery"], "layers.environment.delivery")
    embodiment = _object(layers["embodiment"], "layers.embodiment")
    _exact_keys(
        embodiment,
        {"source", "format", "descriptor", "control_profile", "delivery"},
        "layers.embodiment",
    )
    _string(embodiment["source"], "layers.embodiment.source")
    if _string(embodiment["format"], "layers.embodiment.format") != (
        PROMPT_BUNDLE_EMBODIMENT_FORMAT
    ):
        raise PromptBundleError(
            f"layers.embodiment.format must be {PROMPT_BUNDLE_EMBODIMENT_FORMAT}"
        )
    control_profile = _optional_string(
        embodiment["control_profile"], "layers.embodiment.control_profile"
    )
    descriptor = _optional_string(
        embodiment["descriptor"], "layers.embodiment.descriptor"
    )
    if descriptor is None and control_profile is None:
        raise PromptBundleError(
            "layers.embodiment requires a descriptor or control_profile binding"
        )
    _string(embodiment["delivery"], "layers.embodiment.delivery")
    episode = _object(layers["episode_task"], "layers.episode_task")
    _exact_keys(episode, {"source", "placeholder", "delivery"}, "layers.episode_task")
    _string(episode["source"], "layers.episode_task.source")
    if episode["placeholder"] != "{{TASK}}":
        raise PromptBundleError("layers.episode_task.placeholder must be {{TASK}}")
    _string(episode["delivery"], "layers.episode_task.delivery")
    manual = _object(layers["tool_manual"], "layers.tool_manual")
    _exact_keys(manual, {"sources", "delivery", "reader_tool"}, "layers.tool_manual")
    _strings(manual["sources"], "layers.tool_manual.sources")
    _string(manual["delivery"], "layers.tool_manual.delivery")
    _optional_string(manual["reader_tool"], "layers.tool_manual.reader_tool")
    tools = _object(layers["tool_surface"], "layers.tool_surface")
    _exact_keys(tools, {"profile", "delivery"}, "layers.tool_surface")
    _string(tools["profile"], "layers.tool_surface.profile")
    _string(tools["delivery"], "layers.tool_surface.delivery")
    dynamic = _object(layers["dynamic_runtime"], "layers.dynamic_runtime")
    _exact_keys(dynamic, {"owner", "delivery"}, "layers.dynamic_runtime")
    if _string(dynamic["owner"], "layers.dynamic_runtime.owner") != "ora":
        raise PromptBundleError("layers.dynamic_runtime.owner must be ora")
    _string(dynamic["delivery"], "layers.dynamic_runtime.delivery")


def validate_prompt_bundle(value: Any) -> dict[str, Any]:
    """Validate the closed authoring manifest shape without reading its sources."""

    bundle = dict(_object(value, "prompt bundle"))
    _exact_keys(
        bundle,
        {"schema", "bundle_id", "status", "runtime", "layers", "bindings", "ownership"},
        "prompt bundle",
    )
    if bundle.get("schema") != PROMPT_BUNDLE_SCHEMA:
        raise PromptBundleError("invalid prompt bundle schema")
    _string(bundle["bundle_id"], "bundle_id")
    if bundle.get("status") not in {"draft", "reviewed"}:
        raise PromptBundleError("status must be draft or reviewed")
    runtime = _object(bundle["runtime"], "runtime")
    _exact_keys(runtime, _RUNTIME_KEYS, "runtime")
    policy_kind = _string(runtime["policy_kind"], "runtime.policy_kind")
    context_schema = _string(runtime["context_schema"], "runtime.context_schema")
    scaffold = _string(runtime["provider_scaffold"], "runtime.provider_scaffold")
    if policy_kind == "native_api":
        if context_schema != "ora.policy-context.v3" or scaffold != "none":
            raise PromptBundleError("Native bundles require typed v3 context and no provider scaffold")
    else:
        raise PromptBundleError("runtime.policy_kind is unsupported")
    _validate_layer_shapes(_object(bundle["layers"], "layers"))
    bindings = _object(bundle["bindings"], "bindings")
    _exact_keys(bindings, _BINDING_KEYS, "bindings")
    environment_binding = _object(
        bindings["environment_prompt_profile"],
        "bindings.environment_prompt_profile",
    )
    _exact_keys(
        environment_binding,
        _PROFILE_BINDING_KEYS,
        "bindings.environment_prompt_profile",
    )
    _string(environment_binding["profile_id"], "bindings.environment_prompt_profile.profile_id")
    _object(
        environment_binding["public_environment_context"],
        "bindings.environment_prompt_profile.public_environment_context",
    )
    ownership = _object(bundle["ownership"], "ownership")
    _exact_keys(ownership, _OWNERSHIP_KEYS, "ownership")
    _strings(ownership["environment_terms"], "ownership.environment_terms")
    _strings(ownership["embodiment_terms"], "ownership.embodiment_terms")
    if _PLACEHOLDER in json.dumps(bundle, ensure_ascii=False):
        raise PromptBundleError(f"prompt bundle contains unresolved {_PLACEHOLDER}")
    return bundle


def load_prompt_bundle(path: Path) -> dict[str, Any]:
    """Load one project-relative prompt-bundle authoring manifest."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PromptBundleError(f"cannot load prompt bundle: {exc}") from exc
    return validate_prompt_bundle(value)


def _embodiment_sections(text: str) -> tuple[list[str], list[str]]:
    headings = [
        line[3:].strip() for line in text.splitlines() if line.startswith("## ")
    ]
    missing = [heading for heading in EMBODIMENT_SECTION_HEADINGS if heading not in headings]
    return headings, missing


def _delivery_inventory(
    bundle: Mapping[str, Any], sources: Mapping[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    runtime = _object(bundle["runtime"], "runtime")
    native = runtime["policy_kind"] == "native_api"
    layers = _object(bundle["layers"], "layers")
    inventory: list[dict[str, Any]] = []
    for layer_name in ("planner", "core", "environment", "embodiment", "episode_task"):
        layer = _object(layers[layer_name], f"layers.{layer_name}")
        role = {
            "planner": "system",
            "core": "user",
            "environment": "user",
            "embodiment": "user",
            "episode_task": "user",
        }[layer_name] if native else None
        for source in sources[layer_name]:
            inventory.append({
                "layer": layer_name,
                "source_path": source["path"],
                "source_sha256": source["sha256"],
                "delivery": layer["delivery"],
                "provider_role": role,
                "provider_visible": True if native else None,
                "provider_visibility_status": "exact" if native else "managed_product_owned",
                "delivery_timing": "conversation_start",
            })
    manual = _object(layers["tool_manual"], "layers.tool_manual")
    for source in sources["tool_manual"]:
        inventory.append({
            "layer": "tool_manual",
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "delivery": manual["delivery"],
            "provider_role": None,
            "provider_visible": False if manual["delivery"] == "manifest_only" else None,
            "provider_visibility_status": (
                "not_delivered" if manual["delivery"] == "manifest_only"
                else "on_demand_or_managed_product_owned"
            ),
            "delivery_timing": "never" if manual["delivery"] == "manifest_only" else "on_demand",
        })
    inventory.append({
        "layer": "tool_surface",
        "source_path": None,
        "source_sha256": None,
        "delivery": layers["tool_surface"]["delivery"],
        "provider_role": "tools" if native else None,
        "provider_visible": True if native else None,
        "provider_visibility_status": "exact" if native else "managed_product_owned",
        "delivery_timing": "every_model_call" if native else "session_configuration",
    })
    inventory.append({
        "layer": "dynamic_runtime",
        "source_path": None,
        "source_sha256": None,
        "delivery": layers["dynamic_runtime"]["delivery"],
        "provider_role": "user" if native else None,
        "provider_visible": True if native else None,
        "provider_visibility_status": "exact" if native else "managed_product_owned",
        "delivery_timing": "every_outer_turn",
    })
    return inventory


def audit_prompt_bundle(
    value: Any, *, project_root: Path, include_text: bool = False,
) -> dict[str, Any]:
    """Audit layer ownership, exact bindings, help, and provider delivery."""

    bundle = validate_prompt_bundle(value)
    root = project_root.resolve()
    layers = _object(bundle["layers"], "layers")
    sources = {
        "planner": _source_records(root, layers["planner"]["sources"], "layers.planner.sources"),
        "core": _source_records(root, layers["core"]["sources"], "layers.core.sources"),
        "environment": [_source_record(root, layers["environment"]["source"], "layers.environment.source")],
        "embodiment": [_source_record(root, layers["embodiment"]["source"], "layers.embodiment.source")],
        "episode_task": [_source_record(root, layers["episode_task"]["source"], "layers.episode_task.source")],
        "tool_manual": _source_records(root, layers["tool_manual"]["sources"], "layers.tool_manual.sources"),
    }
    policy_kind = bundle["runtime"]["policy_kind"]
    expected_delivery = _NATIVE_DELIVERY if policy_kind == "native_api" else _MANAGED_DELIVERY
    delivery_actual = {
        name: layers[name]["delivery"] for name in expected_delivery
    }
    checks = [_check(
        "layer_delivery_contract",
        delivery_actual == expected_delivery,
        {"expected": expected_delivery, "actual": delivery_actual},
    )]

    episode_text = sources["episode_task"][0]["text"]
    placeholder = layers["episode_task"]["placeholder"]
    checks.append(_check(
        "episode_task_placeholder_exactly_once",
        episode_text.count(placeholder) == 1,
        {"placeholder": placeholder, "count": episode_text.count(placeholder)},
    ))

    embodiment_text = sources["embodiment"][0]["text"]
    headings, missing_headings = _embodiment_sections(embodiment_text)
    checks.append(_check(
        "embodiment_nine_node_format",
        not missing_headings and [
            heading for heading in headings if heading in EMBODIMENT_SECTION_HEADINGS
        ] == list(EMBODIMENT_SECTION_HEADINGS),
        {"required": list(EMBODIMENT_SECTION_HEADINGS), "observed": headings, "missing": missing_headings},
    ))

    binding = bundle["bindings"]["environment_prompt_profile"]
    try:
        resolved_environment = resolve_environment_prompt_profile(
            binding["public_environment_context"],
            project_root=root,
            profile_id=binding["profile_id"],
        )
        binding_error = None
    except EnvironmentContextError as exc:
        resolved_environment = {}
        binding_error = str(exc)
    expected_sources = {
        "environment_policy": sources["environment"][0]["path"],
        "embodiment_policy": sources["embodiment"][0]["path"],
    }
    checks.append(_check(
        "environment_profile_exact_sources",
        binding_error is None and all(
            resolved_environment.get(key) == path for key, path in expected_sources.items()
        ),
        {"error": binding_error, "expected": expected_sources, "resolved": resolved_environment},
    ))

    public_context = binding["public_environment_context"]
    public_cameras = public_context.get("public_cameras", [])
    missing_cameras = [
        camera for camera in public_cameras
        if isinstance(camera, str) and not _term_present(embodiment_text, camera)
    ]
    checks.append(_check(
        "embodiment_camera_inventory",
        isinstance(public_cameras, list) and bool(public_cameras) and not missing_cameras,
        {"advertised": public_cameras, "missing_from_embodiment": missing_cameras},
    ))

    surface_profile = layers["tool_surface"]["profile"]
    try:
        surface = resolve_tool_surface(surface_profile)
        tool_error = None
        tool_names = [item["name"] for item in surface.definitions]
        help_capability = TOOL_HELP_CONTRACT_CAPABILITY in surface.capabilities
    except ToolSurfaceError as exc:
        surface = None
        tool_error = str(exc)
        tool_names = []
        help_capability = False
    checks.append(_check(
        "tool_surface_help_contract",
        tool_error is None and help_capability,
        {"profile": surface_profile, "tools": tool_names, "error": tool_error, "capability": help_capability},
    ))

    manual = layers["tool_manual"]
    reader_tool = manual["reader_tool"]
    manual_delivery = manual["delivery"]
    valid_manual_delivery = (
        manual_delivery == "manifest_only" and reader_tool is None
    ) or (
        manual_delivery == "on_demand" and reader_tool in tool_names
    )
    checks.append(_check(
        "tool_manual_delivery_truth",
        valid_manual_delivery,
        {"delivery": manual_delivery, "reader_tool": reader_tool, "tool_names": tool_names},
    ))

    ownership = bundle["ownership"]
    environment_terms = list(ownership["environment_terms"])
    embodiment_terms = list(ownership["embodiment_terms"])
    forbidden_by_layer = {
        "planner": environment_terms + embodiment_terms + tool_names,
        "core": environment_terms + embodiment_terms + tool_names,
        "environment": embodiment_terms + tool_names,
        "embodiment": environment_terms + tool_names,
        "episode_task": environment_terms + embodiment_terms + tool_names,
    }
    for layer_name, forbidden in forbidden_by_layer.items():
        text = "\n".join(item["text"] for item in sources[layer_name])
        if layer_name == "episode_task":
            text = text.replace(placeholder, "")
        present = _present_terms(text, forbidden)
        checks.append(_check(
            f"{layer_name}_single_owner_vocabulary",
            not present,
            {"forbidden_terms_present": present},
        ))
    checks.extend([
        _check(
            "environment_identity_present",
            bool(_present_terms(sources["environment"][0]["text"], environment_terms)),
            {"accepted_terms": environment_terms},
        ),
        _check(
            "embodiment_identity_present",
            bool(_present_terms(embodiment_text, embodiment_terms)),
            {"accepted_terms": embodiment_terms},
        ),
    ])

    sentence_owners: dict[str, set[str]] = {}
    for layer_name in (
        "planner", "core", "environment", "embodiment", "episode_task",
    ):
        for item in sources[layer_name]:
            for sentence in _long_sentences(item["text"].replace(placeholder, "")):
                sentence_owners.setdefault(sentence, set()).add(layer_name)
    repeated_sentences = [
        {"sentence": sentence, "layers": sorted(owners)}
        for sentence, owners in sorted(sentence_owners.items())
        if len(owners) > 1
    ]
    checks.append(_check(
        "cross_layer_exact_sentence_uniqueness",
        not repeated_sentences,
        repeated_sentences,
    ))

    descriptor_path = layers["embodiment"]["descriptor"]
    descriptor_record: dict[str, Any] | None = None
    if descriptor_path is not None:
        descriptor_source = _source_record(root, descriptor_path, "layers.embodiment.descriptor")
        try:
            descriptor = load_embodiment_descriptor(root / descriptor_source["path"])
            descriptor_audit = audit_embodiment_sources(descriptor, project_root=root)
            if descriptor.get("schema") == "ora.embodiment-descriptor.v2":
                from .embodiment import render_embodiment_card
                if render_embodiment_card(descriptor) != embodiment_text:
                    raise EmbodimentDescriptorError("v2 embodiment text is not the compiled descriptor projection")
                if (surface is None or surface.view_config is None
                        or surface.view_config.get("public_geometry_descriptor", {}).get("sha256") != descriptor_source["sha256"]):
                    raise EmbodimentDescriptorError("v2 embodiment and ToolSurface geometry bindings differ")
            descriptor_error = None
        except EmbodimentDescriptorError as exc:
            descriptor = {}
            descriptor_audit = {}
            descriptor_error = str(exc)
        descriptor_record = {
            key: descriptor_source[key]
            for key in ("path", "bytes", "estimated_tokens", "sha256")
        }
        descriptor_record.update({
            "descriptor_id": descriptor.get("descriptor_id"),
            "embodiment_id": descriptor.get("embodiment_id"),
            "source_audit": descriptor_audit,
            "error": descriptor_error,
        })
        checks.append(_check(
            "embodiment_descriptor_binding",
            descriptor_error is None
            and descriptor.get("embodiment_id") == public_context.get("embodiment_id")
            and descriptor_audit.get("passed") is True,
            descriptor_record,
        ))

    control_profile_path = layers["embodiment"]["control_profile"]
    control_profile_record: dict[str, Any] | None = None
    if control_profile_path is not None:
        control_source = _source_record(
            root, control_profile_path, "layers.embodiment.control_profile"
        )
        try:
            control_value = json.loads(control_source["text"])
        except json.JSONDecodeError as exc:
            raise PromptBundleError(f"invalid control profile JSON: {exc}") from exc
        if not isinstance(control_value, Mapping):
            raise PromptBundleError("control profile must be an object")
        control_profile_record = {
            key: control_source[key]
            for key in ("path", "bytes", "estimated_tokens", "sha256")
        }
        control_profile_record.update({
            "schema": control_value.get("schema"),
            "profile_id": control_value.get("profile_id"),
            "embodiment_id": control_value.get("embodiment_id"),
        })
        compatible_embodiment = control_value.get("embodiment_id")
        checks.append(_check(
            "embodiment_control_profile_binding",
            compatible_embodiment in {None, public_context.get("embodiment_id")},
            control_profile_record,
        ))

    inventory = _delivery_inventory(bundle, sources)
    result = {
        "schema": PROMPT_BUNDLE_AUDIT_SCHEMA,
        "bundle_id": bundle["bundle_id"],
        "bundle_sha256": _sha256(bundle),
        "status": bundle["status"],
        "runtime": dict(bundle["runtime"]),
        "layer_order": list(PROMPT_BUNDLE_LAYERS),
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "source_inventory": {
            layer: [
                {
                    key: item[key]
                    for key in ("path", "bytes", "estimated_tokens", "sha256")
                }
                for item in records
            ]
            for layer, records in sources.items()
        },
        "review_budget_inventory": {
            layer: {
                "estimated_tokens": sum(
                    item["estimated_tokens"] for item in sources[layer]
                ),
                "review_target_tokens": target,
                "within_review_target": sum(
                    item["estimated_tokens"] for item in sources[layer]
                ) <= target,
            }
            for layer, target in {
                "planner": 500,
                "core": 600,
                "environment": 400,
                "embodiment": 700,
                "episode_task": 350,
            }.items()
        },
        "delivery_inventory": inventory,
        "environment_profile": resolved_environment,
        "tool_surface": {
            "profile": surface_profile,
            "tools": tool_names,
            "composition_sha256": getattr(surface, "composition_sha256", None),
        },
        "embodiment_descriptor": descriptor_record,
        "embodiment_control_profile": control_profile_record,
    }
    if include_text:
        result["authored_text"] = {
            layer: [item["text"] for item in records]
            for layer, records in sources.items()
        }
    return result


def prompt_bundle_template(bundle_id: str) -> dict[str, Any]:
    """Return a closed draft manifest with explicit unresolved sentinels."""

    return {
        "schema": PROMPT_BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "status": "draft",
        "runtime": {
            "policy_kind": "native_api",
            "context_schema": "ora.policy-context.v3",
            "provider_scaffold": "none",
        },
        "layers": {
            "planner": {"sources": [_PLACEHOLDER], "delivery": "system"},
            "core": {"sources": [_PLACEHOLDER], "delivery": "project_instructions"},
            "environment": {"source": _PLACEHOLDER, "delivery": "project_instructions"},
            "embodiment": {
                "source": _PLACEHOLDER,
                "format": PROMPT_BUNDLE_EMBODIMENT_FORMAT,
                "descriptor": None,
                "control_profile": _PLACEHOLDER,
                "delivery": "project_instructions",
            },
            "episode_task": {
                "source": _PLACEHOLDER,
                "placeholder": "{{TASK}}",
                "delivery": "first_user_message",
            },
            "tool_manual": {
                "sources": [_PLACEHOLDER],
                "delivery": "manifest_only",
                "reader_tool": None,
            },
            "tool_surface": {"profile": _PLACEHOLDER, "delivery": "provider_tools_field"},
            "dynamic_runtime": {"owner": "ora", "delivery": "current_user_tail"},
        },
        "bindings": {
            "environment_prompt_profile": {
                "profile_id": _PLACEHOLDER,
                "public_environment_context": {
                    "schema": "ora.public-environment-context.v1",
                    "environment_id": _PLACEHOLDER,
                    "embodiment_id": _PLACEHOLDER,
                    "robot_models": [_PLACEHOLDER],
                    "controller": _PLACEHOLDER,
                    "public_cameras": [_PLACEHOLDER],
                },
            },
        },
        "ownership": {
            "environment_terms": [_PLACEHOLDER],
            "embodiment_terms": [_PLACEHOLDER],
        },
    }


def _atomic_text(path: Path, text: str, *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text and not replace:
        raise PromptBundleError(f"refusing to replace drifted output without --replace: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audit-prompt-bundle",
        description="Create or statically audit one modular robot prompt bundle.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template", help="write a closed draft bundle manifest")
    template.add_argument("output", type=Path)
    template.add_argument("--bundle-id", required=True)
    template.add_argument("--replace", action="store_true")
    audit = commands.add_parser("audit", help="audit one complete bundle without a simulator")
    audit.add_argument("manifest", type=Path)
    audit.add_argument("--full", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "template":
            text = yaml.safe_dump(
                prompt_bundle_template(args.bundle_id),
                sort_keys=False,
                allow_unicode=True,
            )
            _atomic_text(args.output, text, replace=args.replace)
            print(args.output)
            return 0
        bundle = load_prompt_bundle(args.manifest)
        result = audit_prompt_bundle(
            bundle, project_root=Path.cwd(), include_text=args.full,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1
    except PromptBundleError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EMBODIMENT_SECTION_HEADINGS",
    "PROMPT_BUNDLE_AUDIT_SCHEMA",
    "PROMPT_BUNDLE_EMBODIMENT_FORMAT",
    "PROMPT_BUNDLE_LAYERS",
    "PROMPT_BUNDLE_SCHEMA",
    "PromptBundleError",
    "audit_prompt_bundle",
    "load_prompt_bundle",
    "prompt_bundle_template",
    "validate_prompt_bundle",
]
