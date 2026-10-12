"""Versioned, hash-bound composition of policy-visible tool surfaces."""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence


from .locked_assets import locked_asset

PROFILE_SCHEMA = "ora.tool-surface-profile.v1"
RESOLVED_SCHEMA = "ora.resolved-tool-surface.v1"
VIEW_CONFIG_SCHEMA = "ora.policy-view-config.v1"
TOOL_HELP_CONTRACT_CAPABILITY = "policy_tool_help_v1"
TOOL_HELP_DESCRIPTION_MARKERS = (
    "Use when:",
    "Effect:",
    "Returns:",
    "Does not:",
    "On error:",
)
from .profile_catalog import PROFILE_CATALOG, PROFILE_ROOT  # noqa: F401

# These source-frozen profiles predate the model-visible help contract.  New
# catalog entries must opt in to TOOL_HELP_CONTRACT_CAPABILITY instead of
# silently inheriting this grandfather list.
LEGACY_TOOL_HELP_EXEMPT_PROFILES = frozenset({
    "robot_runtime_v1",
    "robot_runtime_v2",
    "native_atomic_causal_v1",
})


from .schema_validation import (  # noqa: E402 - re-exported public validator API
    ANNOTATION_KEYWORDS, SchemaViolation, ToolSurfaceError, VALIDATED_KEYWORDS,
    lint_input_schema, validate_schema_instance,
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_schema(node: Any, *, source: str) -> None:
    """Validate the structural JSON-Schema subset used by Harness tools.

    Runtime dependencies intentionally stay unchanged. This checker catches
    malformed schema composition while provider/backend tests exercise the
    concrete accepted shapes.
    """

    if not isinstance(node, Mapping):
        raise ToolSurfaceError(f"{source} must be a JSON Schema object")
    known_types = {"array", "boolean", "integer", "null", "number", "object", "string"}
    declared = node.get("type")
    if declared is not None:
        values = declared if isinstance(declared, list) else [declared]
        if not values or any(value not in known_types for value in values):
            raise ToolSurfaceError(f"{source} has an invalid type")
    properties = node.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            raise ToolSurfaceError(f"{source}.properties must be an object")
        for name, child in properties.items():
            _validate_schema(child, source=f"{source}.properties.{name}")
    required = node.get("required")
    if required is not None:
        if (
            not isinstance(required, list)
            or len(required) != len(set(required))
            or not all(isinstance(name, str) and name for name in required)
        ):
            raise ToolSurfaceError(f"{source}.required must contain unique names")
        if isinstance(properties, Mapping) and not set(required).issubset(properties):
            raise ToolSurfaceError(f"{source}.required references unknown properties")
    items = node.get("items")
    if items is not None:
        _validate_schema(items, source=f"{source}.items")
    for keyword in ("oneOf", "anyOf", "allOf"):
        branches = node.get(keyword)
        if branches is not None:
            if not isinstance(branches, list) or not branches:
                raise ToolSurfaceError(f"{source}.{keyword} must be non-empty")
            for index, branch in enumerate(branches):
                _validate_schema(branch, source=f"{source}.{keyword}[{index}]")


from .canonical_names import (
    CANONICAL_TOOL_NAMES_CAPABILITY, canonical_tool_name, canonicalise_component,
    canonicalise_prose,
)
from .tool_surface_resolution import (  # noqa: E402 - resolution steps
    apply_canonical_tool_names, bind_capability_contracts, exclude_and_order_tools,
    load_ordered_components, load_tool_definitions, resolve_effect_semantics,
    resolve_tool_semantics,
)


def compose_tool_definitions(
    base_definitions: Sequence[Mapping[str, Any]],
    description_layers: Sequence[Mapping[str, Any]],
    *,
    schema_replacements: Sequence[Mapping[str, Any]] = (),
    excluded: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Resolve exact descriptions and complete schema replacements in order.

    ``excluded`` names tools this surface has stopped declaring. A layer that
    still carries text or a schema for one is skipped, the way a result schema
    keyed on an excluded tool already is: a component may be shared by a
    generation's arms, and an arm that excludes a tool cannot be made to edit a
    component it shares in order to say so. A layer naming a tool that is
    neither declared nor excluded still raises -- that is a typo, not a
    retirement.
    """

    definitions = [copy.deepcopy(dict(item)) for item in base_definitions]
    names = [item.get("name") for item in definitions]
    if (
        not all(isinstance(name, str) and name for name in names)
        or len(names) != len(set(names))
    ):
        raise ToolSurfaceError("base tool names must be unique and non-empty")
    by_name = {str(item["name"]): item for item in definitions}
    for name, item in by_name.items():
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            raise ToolSurfaceError(f"tool {name!r} has an empty description")
        _validate_schema(item.get("inputSchema"), source=f"{name}.inputSchema")

    for index, layer in enumerate(description_layers):
        unknown_operations = set(layer) - {"replace", "prepend", "append"}
        if unknown_operations:
            raise ToolSurfaceError(
                f"description layer {index} has unknown operations: {sorted(unknown_operations)}"
            )
        for operation in ("replace", "prepend", "append"):
            values = layer.get(operation, {})
            if not isinstance(values, Mapping):
                raise ToolSurfaceError(f"description layer {index}.{operation} must be an object")
            for raw_name, raw_text in values.items():
                name = str(raw_name)
                text = str(raw_text).strip()
                if name in excluded:
                    continue
                if name not in by_name:
                    raise ToolSurfaceError(f"description layer references unknown tool {name!r}")
                if not text:
                    raise ToolSurfaceError(f"description layer gives {name!r} empty text")
                current = str(by_name[name]["description"]).strip()
                if operation == "replace":
                    by_name[name]["description"] = text
                elif operation == "prepend":
                    by_name[name]["description"] = f"{text} {current}"
                else:
                    by_name[name]["description"] = f"{current} {text}"

    for index, layer in enumerate(schema_replacements):
        if not isinstance(layer, Mapping):
            raise ToolSurfaceError(f"schema layer {index} must be an object")
        for raw_name, raw_schema in layer.items():
            name = str(raw_name)
            if name in excluded:
                continue
            if name not in by_name:
                raise ToolSurfaceError(f"schema layer references unknown tool {name!r}")
            _validate_schema(raw_schema, source=f"schema layer {index}.{name}")
            by_name[name]["inputSchema"] = copy.deepcopy(dict(raw_schema))
    return definitions


def validate_tool_help_contract(
    definitions: Sequence[Mapping[str, Any]],
    result_schemas: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate the model-visible ``--help`` contract for a ToolSurface.

    Historical surfaces predate this contract and remain source-frozen.  A
    successor opts in through ``policy_tool_help_v1`` and must then document
    selection, effects, public results, non-guarantees and recovery for every
    tool.  Input properties are documented recursively, and every tool owns a
    Runtime-validated public result schema even when a provider protocol does
    not have a native output-schema field.
    """

    def check_input_schema(node: Mapping[str, Any], *, path: str) -> None:
        declared = node.get("type")
        declared_types = declared if isinstance(declared, list) else [declared]
        if "object" in declared_types:
            if node.get("additionalProperties") is not False:
                raise ToolSurfaceError(
                    f"{path} must set additionalProperties=false under "
                    f"{TOOL_HELP_CONTRACT_CAPABILITY}"
                )
            properties = node.get("properties", {})
            if not isinstance(properties, Mapping):
                raise ToolSurfaceError(f"{path}.properties must be an object")
            for name, child in properties.items():
                if not isinstance(child, Mapping):
                    raise ToolSurfaceError(f"{path}.{name} must be a schema object")
                description = child.get("description")
                if not isinstance(description, str) or not description.strip():
                    raise ToolSurfaceError(
                        f"{path}.{name} lacks a parameter description"
                    )
                check_input_schema(child, path=f"{path}.{name}")
        items = node.get("items")
        if isinstance(items, Mapping):
            check_input_schema(items, path=f"{path}[]")
        for keyword in ("oneOf", "anyOf", "allOf"):
            branches = node.get(keyword, [])
            if isinstance(branches, list):
                for index, branch in enumerate(branches):
                    if isinstance(branch, Mapping):
                        check_input_schema(
                            branch, path=f"{path}.{keyword}[{index}]",
                        )

    names: list[str] = []
    for definition in definitions:
        name = definition.get("name")
        if not isinstance(name, str) or not name:
            raise ToolSurfaceError("tool-help contract requires named tools")
        names.append(name)
        description = definition.get("description")
        if not isinstance(description, str):
            raise ToolSurfaceError(f"tool {name!r} lacks a description")
        missing = [
            marker for marker in TOOL_HELP_DESCRIPTION_MARKERS
            if marker not in description
        ]
        if missing:
            raise ToolSurfaceError(
                f"tool {name!r} help lacks sections: {', '.join(missing)}"
            )
        schema = definition.get("inputSchema")
        if not isinstance(schema, Mapping):
            raise ToolSurfaceError(f"tool {name!r} lacks an input schema")
        check_input_schema(schema, path=f"{name}.inputSchema")

    missing_results = set(names) - set(result_schemas)
    extra_results = set(result_schemas) - set(names)
    if missing_results or extra_results:
        raise ToolSurfaceError(
            "tool-help result schemas must cover exactly the advertised tools; "
            f"missing={sorted(missing_results)}, extra={sorted(extra_results)}"
        )


@dataclass(frozen=True)
class ResolvedToolSurface:
    profile_id: str
    revision: int
    runtime_id: str
    definitions: tuple[dict[str, Any], ...]
    result_schemas: Mapping[str, Mapping[str, Any]]
    mutation_tools: tuple[str, ...]
    lifecycle_tools: tuple[str, ...]
    observation_tools: tuple[str, ...]
    call_effect_profile: str | None
    view_config: Mapping[str, Any] | None
    capabilities: tuple[str, ...]
    provenance: Mapping[str, Any]
    components: tuple[dict[str, str], ...]
    manifest_sha256: str
    composition_sha256: str
    #: Tool name -> {tool, version, arm_profile, compiled_sha256} for tools the
    #: manifest binds to a composed library entry; empty on every other surface.
    composed_tools: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def supports(self, capability: str) -> bool:
        """Return one manifest-bound semantic capability.

        Runtime code must branch on these capabilities rather than duplicating
        profile-name allowlists.  A new profile therefore inherits only the
        behavior it explicitly declares in its hash-bound manifest.
        """

        return capability in self.capabilities

    def select(self, allowed: Sequence[str] | None = None) -> list[dict[str, Any]]:
        selected = {item["name"] for item in self.definitions} if allowed is None else set(allowed)
        known = {item["name"] for item in self.definitions}
        if not selected.issubset(known):
            raise ToolSurfaceError(
                f"tool allowlist contains unknown names: {sorted(selected - known)}"
            )
        return [copy.deepcopy(item) for item in self.definitions if item["name"] in selected]

    def snapshot(self) -> dict[str, Any]:
        snapshot = {
            "schema": RESOLVED_SCHEMA,
            "profile_id": self.profile_id,
            "revision": self.revision,
            "runtime_id": self.runtime_id,
            "definitions": [copy.deepcopy(item) for item in self.definitions],
            "result_schemas": copy.deepcopy(dict(self.result_schemas)),
            "mutation_tools": list(self.mutation_tools),
            "lifecycle_tools": list(self.lifecycle_tools),
            "capabilities": list(self.capabilities),
            "provenance": copy.deepcopy(dict(self.provenance)),
            "components": [dict(item) for item in self.components],
            "manifest_sha256": self.manifest_sha256,
            "composition_sha256": self.composition_sha256,
        }
        if self.call_effect_profile is not None:
            snapshot["observation_tools"] = list(self.observation_tools)
            snapshot["call_effect_profile"] = self.call_effect_profile
        if self.view_config is not None:
            snapshot["view_config"] = copy.deepcopy(dict(self.view_config))
        if self.composed_tools:
            snapshot["composed_tools"] = copy.deepcopy(dict(self.composed_tools))
        return snapshot


COMPOSED_TOOLS_CAPABILITY = "composed_tools_v1"
#: A component override ``composed:<tool>: "<tool>/<version>"`` rebinds a composed tool.
COMPOSED_OVERRIDE_PREFIX = "composed:"

from .composed_binding import bind_composed_tools  # noqa: E402 - re-exported; it reads this module lazily


def _resolve_view_config(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Validate one shared policy-view contract for tools and session delivery."""

    raw = manifest.get("view_config")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ToolSurfaceError("view_config must be an object")
    required = {
        "schema", "rgb_resolution", "derived_resolution",
        "automatic_actual_views", "default_observe_views",
        "coordinate_convention",
    }
    optional = {"auxiliary_contact_sheet_cameras", "current_visual_delivery", "current_observation_delivery", "controlled_frame_convention", "public_geometry_descriptor", "gripper_command_delivery", "controlled_point_projection", "jaw_contact_center_projection"}
    if not required.issubset(set(raw)) or set(raw) - required - optional:
        raise ToolSurfaceError(
            "view_config must contain exactly " + ", ".join(sorted(required))
            + " (optionally " + ", ".join(sorted(optional)) + ")"
        )
    if raw.get("schema") != VIEW_CONFIG_SCHEMA:
        raise ToolSurfaceError("view_config schema is unsupported")

    if "current_visual_delivery" in raw and raw["current_visual_delivery"] != "full_rgb_v1":
        raise ToolSurfaceError("unsupported current visual delivery")
    if "gripper_command_delivery" in raw and (
        raw["gripper_command_delivery"] != "commanded_beside_measured_v1"
        or raw.get("current_observation_delivery") != "all_public_v1"
    ):
        raise ToolSurfaceError("unsupported gripper command delivery")
    if "controlled_point_projection" in raw and (
        raw["controlled_point_projection"] != "per_camera_pixel_v1"
        or raw.get("current_observation_delivery") != "all_public_v1"
    ):
        raise ToolSurfaceError("unsupported controlled point projection")
    if "jaw_contact_center_projection" in raw and (
        raw["jaw_contact_center_projection"] != "descriptor_contact_center_v1"
        or "controlled_point_projection" not in raw
        or not isinstance(raw.get("public_geometry_descriptor"), str)
    ):
        raise ToolSurfaceError("unsupported jaw contact-centre projection")
    rgb_resolution = raw.get("rgb_resolution")
    if rgb_resolution != "native":
        if not (
            isinstance(rgb_resolution, list)
            and len(rgb_resolution) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value > 0
                for value in rgb_resolution
            )
        ):
            raise ToolSurfaceError("view_config.rgb_resolution must be native or [width, height]")
    derived_resolution = raw.get("derived_resolution")
    if not (
        isinstance(derived_resolution, list)
        and len(derived_resolution) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in derived_resolution
        )
    ):
        raise ToolSurfaceError("view_config.derived_resolution must be [width, height]")
    for field in ("automatic_actual_views", "default_observe_views"):
        values = raw.get(field)
        if not (
            isinstance(values, list)
            and values
            and len(values) == len(set(values))
            and all(isinstance(value, str) and value for value in values)
        ):
            raise ToolSurfaceError(f"view_config.{field} must contain unique view names")
    contact_sheet = "auxiliary_contact_sheet"
    capabilities = manifest.get("capabilities", [])
    if contact_sheet in raw["automatic_actual_views"] and (
        not isinstance(capabilities, list)
        or "automatic_current_auxiliary_contact_sheet_v1" not in capabilities
    ):
        raise ToolSurfaceError(
            "auxiliary_contact_sheet requires its automatic projection capability"
        )
    if contact_sheet in raw["default_observe_views"]:
        raise ToolSurfaceError(
            "auxiliary_contact_sheet is automatic derived evidence, not an observe camera"
        )
    sheet_cameras = raw.get("auxiliary_contact_sheet_cameras")
    if sheet_cameras is not None:
        if contact_sheet not in raw["automatic_actual_views"]:
            raise ToolSurfaceError(
                "auxiliary_contact_sheet_cameras requires the sheet in "
                "automatic_actual_views"
            )
        # The admitted counts are exactly the layouts the projection can
        # materialize, so a new sheet layout admits itself here. Imported
        # lazily: agent.native imports this module through the context layer.
        from ora.agent.native.observation_projection import (
            AUXILIARY_CONTACT_SHEET_LAYOUTS,
        )

        admitted_counts = sorted(AUXILIARY_CONTACT_SHEET_LAYOUTS)
        if not (
            isinstance(sheet_cameras, list)
            and len(sheet_cameras) in admitted_counts
            and len(sheet_cameras) == len(set(sheet_cameras))
            and all(
                isinstance(value, str) and value and value != contact_sheet
                for value in sheet_cameras
            )
        ):
            raise ToolSurfaceError(
                "auxiliary_contact_sheet_cameras must list three or four "
                "unique source camera names (admitted counts: "
                + ", ".join(str(count) for count in admitted_counts) + ")"
            )
    if raw.get("coordinate_convention") != "opencv_top_left_xy":
        raise ToolSurfaceError("view_config coordinate convention is unsupported")
    delivery = raw.get("current_observation_delivery")
    if delivery is not None:
        from ora.agent.context.public_facts import PUBLIC_FACTS_DELIVERY
        from .embodiment_geometry import PUBLIC_GEOMETRY_DESCRIPTORS
        if delivery != PUBLIC_FACTS_DELIVERY or raw.get("controlled_frame_convention") not in PUBLIC_GEOMETRY_DESCRIPTORS:
            raise ToolSurfaceError("full public observation requires a supported controlled-frame convention")
        if raw.get("rgb_resolution") != "native":
            raise ToolSurfaceError("full public observation requires native RGB coordinates")
        if "current_public_robot_facts_v1" not in capabilities:
            raise ToolSurfaceError("full public observation requires its delivery capability")
        from .embodiment_geometry import bind_public_geometry_descriptor, load_public_geometry_binding
        from .embodiment import EmbodimentDescriptorError
        try:
            # The manifest names the descriptor by path; the resolved view
            # carries the {path, sha256} record of the bytes read here.
            geometry_record = bind_public_geometry_descriptor(raw.get("public_geometry_descriptor"), project_root=Path(__file__).resolve().parents[3])
            geometry_binding = load_public_geometry_binding(geometry_record, project_root=Path(__file__).resolve().parents[3])
        except EmbodimentDescriptorError as exc:
            raise ToolSurfaceError(str(exc)) from exc
        convention = raw["controlled_frame_convention"]
        from .embodiment_geometry import PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS
        expected_paths = {
            "ora/policy/embodiments/" + path
            for path in PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS[convention]
        }
        if geometry_record["path"] not in expected_paths:
            raise ToolSurfaceError("geometry descriptor does not match the controlled-frame convention")
        return {**copy.deepcopy(dict(raw)), "public_geometry_descriptor": geometry_record}
    elif "controlled_frame_convention" in raw or "public_geometry_descriptor" in raw:
        raise ToolSurfaceError("controlled-frame convention requires full public observation delivery")
    return copy.deepcopy(dict(raw))


def _project_root() -> Path:
    """The source tree whose ``policy.lock`` and retired registry bind the catalog.

    A rebound PROFILE_ROOT at <root>/ora/policy/tool_surfaces answers
    from that root; a test that widens the loading boundary to a filesystem
    anchor keeps this checkout.
    """

    profile_root = PROFILE_ROOT.resolve()
    if profile_root.parts[-3:] == ("ora", "policy", "tool_surfaces"):
        return profile_root.parents[2]
    return Path(__file__).resolve().parents[3]


def _component(
    *, profile_root: Path, allowed_root: Path, slot: str,
    spec: Any, override: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load one component by its path, relative to the manifest's directory.

    A component reference is a path string; ``policy.lock`` carries its
    integrity and this loader records the sha256 of the bytes it read.
    """

    selected = override if override is not None else spec
    ref = selected if isinstance(selected, str) else None
    path = (profile_root / ref).resolve() if ref else None
    # Registered profiles may reuse components from a sibling profile. The
    # catalog root remains the hard loading boundary.
    if path is not None and (not path.is_relative_to(allowed_root.resolve()) or not path.is_file()):
        raise ToolSurfaceError(f"component {slot!r} is unavailable: {ref}")
    asset = locked_asset(
        _project_root(), selected, base=profile_root, label=f"component {slot!r}", error=ToolSurfaceError,
    )
    value = json.loads(asset.data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ToolSurfaceError(f"component {slot!r} must contain an object")
    return value, {"slot": slot, "ref": ref, "sha256": asset.sha256}


def _load_manifest(profile_id: str) -> tuple[Path, Path, dict[str, Any], int]:
    """Find a registered manifest, its component loading boundary and revision.

    Reads ``PROFILE_CATALOG`` and ``PROFILE_ROOT`` from this module at call
    time, so a test that rebinds either still steers resolution.
    """

    manifest_path = PROFILE_CATALOG.get(profile_id)
    if manifest_path is None:
        raise ToolSurfaceError(f"unknown tool surface profile: {profile_id}")
    allowed_component_root = (
        PROFILE_ROOT
        if manifest_path.resolve().is_relative_to(PROFILE_ROOT.resolve())
        else manifest_path.parent
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PROFILE_SCHEMA or manifest.get("profile_id") != profile_id:
        raise ToolSurfaceError(f"invalid tool surface manifest: {manifest_path}")
    revision = manifest.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ToolSurfaceError("tool surface revision must be positive")
    return manifest_path, allowed_component_root, manifest, revision


def _collect_result_schemas(
    result_layers: Sequence[Mapping[str, Any]],
    definitions: Sequence[Mapping[str, Any]],
    excluded: Collection[str],
) -> dict[str, Mapping[str, Any]]:
    """Merge the ordered result layers for every declared tool."""

    results: dict[str, Mapping[str, Any]] = {}
    for layer in result_layers:
        for name, schema in layer.items():
            if name not in {item["name"] for item in definitions}:
                if name in set(excluded):
                    continue
                raise ToolSurfaceError(f"result schema references unknown tool {name!r}")
            _validate_schema(schema, source=f"result schema {name}")
            results[name] = copy.deepcopy(dict(schema))
    return results


def _resolve_runtime_identity(
    manifest: Mapping[str, Any], profile_id: str,
) -> tuple[str, list[str]]:
    """Validate ``runtime_id`` and ``capabilities`` and the help-contract opt-in."""

    runtime_id = manifest.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ToolSurfaceError("tool surface runtime_id must be non-empty")
    capabilities = manifest.get("capabilities", [])
    if (
        not isinstance(capabilities, list)
        or not all(isinstance(name, str) and name for name in capabilities)
        or len(capabilities) != len(set(capabilities))
    ):
        raise ToolSurfaceError("tool surface capabilities must contain unique names")
    if (
        profile_id not in LEGACY_TOOL_HELP_EXEMPT_PROFILES
        and TOOL_HELP_CONTRACT_CAPABILITY not in capabilities
    ):
        raise ToolSurfaceError(
            f"new tool surface {profile_id!r} must declare "
            f"{TOOL_HELP_CONTRACT_CAPABILITY}"
        )
    return runtime_id, capabilities


def _resolved_surface(
    *, profile_id: str, revision: int, runtime_id: str, manifest: Mapping[str, Any],
    manifest_path: Path, definitions: list[dict[str, Any]],
    results: dict[str, Mapping[str, Any]], mutation_tools: tuple[str, ...],
    lifecycle_tools: tuple[str, ...], observation_tools: tuple[str, ...],
    call_effect_profile: str | None, view_config: dict[str, Any] | None,
    capabilities: list[str], descriptors: list[dict[str, str]],
    composed_tools: Mapping[str, Mapping[str, str]] | None = None,
) -> ResolvedToolSurface:
    """Hash the composition and freeze it into a ResolvedToolSurface."""

    manifest_hash = file_sha256(manifest_path)
    composition = {
        "profile_id": profile_id,
        "revision": revision,
        "runtime_id": runtime_id,
        "definitions": definitions,
        "result_schemas": results,
        "mutation_tools": list(mutation_tools),
        "lifecycle_tools": list(lifecycle_tools),
        "capabilities": capabilities,
        "provenance": manifest.get("provenance", {}),
        "components": descriptors,
        "manifest_sha256": manifest_hash,
    }
    if call_effect_profile is not None:
        composition["observation_tools"] = list(observation_tools)
        composition["call_effect_profile"] = call_effect_profile
    if view_config is not None:
        composition["view_config"] = copy.deepcopy(view_config)
    if composed_tools:
        composition["composed_tools"] = copy.deepcopy(dict(composed_tools))
    return ResolvedToolSurface(
        profile_id=profile_id,
        revision=revision,
        runtime_id=runtime_id,
        definitions=tuple(definitions),
        result_schemas=results,
        mutation_tools=mutation_tools,
        lifecycle_tools=lifecycle_tools,
        observation_tools=observation_tools,
        call_effect_profile=call_effect_profile,
        view_config=view_config,
        capabilities=tuple(capabilities),
        provenance=copy.deepcopy(dict(manifest.get("provenance", {}))),
        components=tuple(descriptors),
        manifest_sha256=manifest_hash,
        composition_sha256=canonical_sha256(composition),
        composed_tools=copy.deepcopy(dict(composed_tools or {})),
    )


def resolve_tool_surface(
    profile_id: str = "robot_runtime_v2",
    *,
    component_overrides: Mapping[str, Any] | None = None,
) -> ResolvedToolSurface:
    """Resolve a registered surface without permitting arbitrary code loading.

    The steps run in the order their errors are reported. Manifest-reading
    steps live in :mod:`.tool_surface_resolution`; component files load only
    through ``_component`` bound to this manifest's root and boundary.
    """

    manifest_path, allowed_component_root, manifest, revision = _load_manifest(profile_id)
    component_overrides = dict(component_overrides or {})
    composed_overrides = {
        key[len(COMPOSED_OVERRIDE_PREFIX):]: component_overrides.pop(key)
        for key in [key for key in component_overrides if key.startswith(COMPOSED_OVERRIDE_PREFIX)]
    }
    load = functools.partial(
        _component, profile_root=manifest_path.parent, allowed_root=allowed_component_root,
    )
    base, definition_descriptors = load_tool_definitions(manifest, load)
    canonical_names = apply_canonical_tool_names(manifest, base)
    base, excluded = exclude_and_order_tools(manifest, base)
    description_layers, schema_layers, result_layers, component_descriptors = (
        load_ordered_components(
            manifest, component_overrides, load, canonical_names=canonical_names,
        )
    )
    definitions = compose_tool_definitions(
        base, description_layers, schema_replacements=schema_layers,
        excluded=set(excluded),
    )
    if canonical_names:
        # After composition, so a layer that appends a near-miss clause is
        # rewritten with the rest of the text it was merged into.
        definitions = [canonicalise_prose(item) for item in definitions]
    results = _collect_result_schemas(result_layers, definitions, excluded)
    mutation_tools, lifecycle_tools = resolve_tool_semantics(manifest, definitions)
    view_config = _resolve_view_config(manifest)
    observation_tools, call_effect_profile = resolve_effect_semantics(
        manifest, definitions, mutation_tools, lifecycle_tools,
    )
    runtime_id, capabilities = _resolve_runtime_identity(manifest, profile_id)
    composed_tools, composed_descriptors = bind_composed_tools(
        manifest, definitions, results, capabilities, composed_overrides)
    mutation_tools += tuple(name for name, item in composed_tools.items() if item.get("scope") == "added")
    bound_descriptors = bind_capability_contracts(
        definitions, results, capabilities, view_config, profile_id=profile_id,
    )
    if TOOL_HELP_CONTRACT_CAPABILITY in capabilities:
        validate_tool_help_contract(definitions, results)
    return _resolved_surface(
        profile_id=profile_id, revision=revision, runtime_id=runtime_id,
        manifest=manifest, manifest_path=manifest_path, definitions=definitions,
        results=results, mutation_tools=mutation_tools,
        lifecycle_tools=lifecycle_tools, observation_tools=observation_tools,
        call_effect_profile=call_effect_profile, view_config=view_config,
        capabilities=capabilities,
        descriptors=[
            *definition_descriptors, *component_descriptors, *composed_descriptors, *bound_descriptors,
        ],
        composed_tools=composed_tools,
    )
