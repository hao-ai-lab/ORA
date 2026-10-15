"""The manifest-reading steps of :func:`tool_surface.resolve_tool_surface`.

Each function is one step of resolution, called in the order the resolver
reports its errors. None of them reads a module global of ``tool_surface``:
component files are loaded through the ``load`` callable the resolver binds
to the manifest's own root and loading boundary, and every step that needs
another ``tool_surface`` helper (composition, result schemas, view config,
identity, hashing) stays in that module.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping

from .canonical_names import (
    CANONICAL_TOOL_NAMES_CAPABILITY, canonical_tool_name, canonicalise_component,
)
from .schema_validation import ToolSurfaceError, lint_input_schema


# ``tool_surface._component`` with ``profile_root`` and ``allowed_root`` bound:
# ``load(slot=..., spec=..., override=...)`` returns ``(value, descriptor)``.
ComponentLoader = Callable[..., tuple[dict[str, Any], dict[str, str]]]


def _base_definitions(
    manifest: Mapping[str, Any], load: ComponentLoader,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    """Start from the manifest's base catalog, or the Runtime's own tools."""

    base_component = manifest.get("base_definitions")
    base_descriptor: dict[str, str] | None = None
    if base_component is None:
        from ora.agent.tools.robot import TOOL_DEFINITIONS

        base = [copy.deepcopy(item) for item in TOOL_DEFINITIONS]
    else:
        base_value, base_descriptor = load(
            slot="base_definitions",
            spec=base_component,
            override=None,
        )
        raw_tools = base_value.get("tools")
        if not isinstance(raw_tools, list):
            raise ToolSurfaceError("base_definitions must contain a tools list")
        base = [copy.deepcopy(dict(item)) for item in raw_tools if isinstance(item, Mapping)]
        if len(base) != len(raw_tools):
            raise ToolSurfaceError("base_definitions tools must be objects")
    return base, base_descriptor


def _additional_definitions(
    manifest: Mapping[str, Any], load: ComponentLoader, base: list[dict[str, Any]],
) -> dict[str, str] | None:
    """Append the optional ``additional_definitions`` catalog to ``base``."""

    additional_descriptor: dict[str, str] | None = None
    additional_component = manifest.get("additional_definitions")
    if additional_component is not None:
        additional_value, additional_descriptor = load(
            slot="additional_definitions",
            spec=additional_component,
            override=None,
        )
        additional_tools = additional_value.get("tools")
        if not isinstance(additional_tools, list) or not all(
            isinstance(item, Mapping) for item in additional_tools
        ):
            raise ToolSurfaceError(
                "additional_definitions must contain a tools list of objects"
            )
        base.extend(copy.deepcopy(dict(item)) for item in additional_tools)
    return additional_descriptor


def _supplementary_definitions(
    manifest: Mapping[str, Any], load: ComponentLoader, base: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Append each ``supplementary_definitions`` catalog to ``base``.

    A live surface adds a tool as its own layer instead of rewriting the
    additional_definitions file it shares with frozen surfaces. Each entry is
    one component path; a name already defined is an error, never an override
    (overrides are description, schema and result layers).
    """

    supplementary_descriptors: list[dict[str, str]] = []
    supplementary = manifest.get("supplementary_definitions", [])
    if not isinstance(supplementary, list):
        raise ToolSurfaceError("supplementary_definitions must be a list of components")
    for index, spec in enumerate(supplementary):
        value, descriptor = load(
            slot=f"supplementary_definitions[{index}]",
            spec=spec,
            override=None,
        )
        tools = value.get("tools")
        if not isinstance(tools, list) or not tools or not all(
            isinstance(item, Mapping) for item in tools
        ):
            raise ToolSurfaceError(
                "supplementary_definitions must contain a non-empty tools list of objects"
            )
        defined = {item.get("name") for item in base}
        if any(item.get("name") in defined for item in tools):
            raise ToolSurfaceError("supplementary_definitions redefine an existing tool")
        base.extend(copy.deepcopy(dict(item)) for item in tools)
        supplementary_descriptors.append(descriptor)
    return supplementary_descriptors


def load_tool_definitions(
    manifest: Mapping[str, Any], load: ComponentLoader,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Return the declared tool catalog and the descriptors of its components.

    Base, additional and supplementary catalogs load in that order, which is
    also the order of the returned descriptors.
    """

    base, base_descriptor = _base_definitions(manifest, load)
    additional_descriptor = _additional_definitions(manifest, load, base)
    supplementary_descriptors = _supplementary_definitions(manifest, load, base)
    descriptors: list[dict[str, str]] = (
        [base_descriptor] if base_descriptor is not None else []
    )
    if additional_descriptor is not None:
        descriptors.append(additional_descriptor)
    descriptors.extend(supplementary_descriptors)
    return base, descriptors


def apply_canonical_tool_names(
    manifest: Mapping[str, Any], base: list[dict[str, Any]],
) -> bool:
    """Rename ``base`` in place when the surface opts in; return whether it did.

    One spelling per tool, owned by agent/tools/tool_names.py. A surface opts
    in; frozen surfaces keep the dotted names their sealed runs were shown.
    The map is not restated in the manifest, because a second copy of it is a
    copy that can disagree with the first.
    """

    canonical_names = CANONICAL_TOOL_NAMES_CAPABILITY in manifest.get("capabilities", [])
    if canonical_names:
        for item in base:
            item["name"] = canonical_tool_name(str(item.get("name")))
    return canonical_names


def exclude_and_order_tools(
    manifest: Mapping[str, Any], base: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Any]:
    """Drop ``exclude_tools`` and check ``base`` against ``tool_order``.

    Returns the ordered catalog and the manifest's exclusion list as given.
    """

    excluded = manifest.get("exclude_tools", [])
    if not isinstance(excluded, list) or not all(isinstance(name, str) for name in excluded):
        raise ToolSurfaceError("exclude_tools must contain tool names")
    known_base = {item.get("name") for item in base}
    if not set(excluded).issubset(known_base):
        raise ToolSurfaceError("exclude_tools references unknown tools")
    base = [item for item in base if item.get("name") not in set(excluded)]
    expected_order = manifest.get("tool_order")
    if "canonical_tool_order_v1" in manifest.get("capabilities", []):
        names = [item["name"] for item in base]
        if (
            not isinstance(expected_order, list)
            or not all(isinstance(name, str) for name in expected_order)
            or len(set(names)) != len(names)
            or len(expected_order) != len(names)
            or set(expected_order) != set(names)
        ):
            raise ToolSurfaceError("canonical tool order must name every tool exactly once")
        by_name = {item["name"]: item for item in base}
        base = [by_name[name] for name in expected_order]
    if expected_order != [item["name"] for item in base]:
        raise ToolSurfaceError("tool surface manifest order differs from its runtime base")
    return base, excluded


def load_ordered_components(
    manifest: Mapping[str, Any],
    component_overrides: Mapping[str, Any] | None,
    load: ComponentLoader,
    *,
    canonical_names: bool,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, str]],
]:
    """Load source artifacts and every ordered description/schema/result layer.

    Returns the description layers, schema layers and result layers in their
    manifest orders, then the descriptors of the source artifacts followed by
    those of the ordered components.
    """

    components_raw = manifest.get("components", {})
    description_order = manifest.get("description_order", [])
    schema_order = manifest.get("schema_order", [])
    result_order = manifest.get("result_order", [])
    if not isinstance(components_raw, Mapping):
        raise ToolSurfaceError("tool surface components must be an object")
    overrides = dict(component_overrides or {})
    if not set(overrides).issubset(components_raw):
        raise ToolSurfaceError(
            f"unknown component overrides: {sorted(set(overrides) - set(components_raw))}"
        )
    resolved: dict[str, dict[str, Any]] = {}
    descriptors: list[dict[str, str]] = []
    source_artifacts = manifest.get("source_artifacts", {})
    if not isinstance(source_artifacts, Mapping):
        raise ToolSurfaceError("source_artifacts must be an object")
    for slot, spec in source_artifacts.items():
        _value, descriptor = load(
            slot=f"source:{slot}",
            spec=spec,
            override=None,
        )
        descriptors.append(descriptor)
    for slot in [*description_order, *schema_order, *result_order]:
        if slot in resolved:
            raise ToolSurfaceError(f"component slot appears more than once: {slot}")
        spec = components_raw.get(slot)
        if spec is None:
            raise ToolSurfaceError(f"ordered component is missing: {slot}")
        value, descriptor = load(
            slot=str(slot),
            spec=spec,
            override=overrides.get(str(slot)),
        )
        resolved[str(slot)] = (
            canonicalise_component(value, nested=slot in set(description_order))
            if canonical_names else value
        )
        descriptors.append(descriptor)
    if set(resolved) != set(components_raw):
        raise ToolSurfaceError("component orders do not cover the manifest components")
    return (
        [resolved[str(slot)] for slot in description_order],
        [resolved[str(slot)] for slot in schema_order],
        [resolved[str(slot)] for slot in result_order],
        descriptors,
    )


def resolve_tool_semantics(
    manifest: Mapping[str, Any], definitions: list[dict[str, Any]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the disjoint mutation and lifecycle tool sets."""

    raw_mutations = manifest.get("mutation_tools", [])
    raw_lifecycle = manifest.get("lifecycle_tools", [])
    for label, values in (
        ("mutation_tools", raw_mutations), ("lifecycle_tools", raw_lifecycle),
    ):
        if (
            not isinstance(values, list)
            or not all(isinstance(name, str) and name for name in values)
            or len(values) != len(set(values))
        ):
            raise ToolSurfaceError(f"{label} must contain unique tool names")
    mutation_tools = tuple(raw_mutations)
    lifecycle_tools = tuple(raw_lifecycle)
    known = {item["name"] for item in definitions}
    if not set(mutation_tools + lifecycle_tools).issubset(known):
        raise ToolSurfaceError("tool semantics reference unknown tools")
    if set(mutation_tools) & set(lifecycle_tools):
        raise ToolSurfaceError("mutation and lifecycle tool semantics must be disjoint")
    return mutation_tools, lifecycle_tools


def resolve_effect_semantics(
    manifest: Mapping[str, Any],
    definitions: list[dict[str, Any]],
    mutation_tools: tuple[str, ...],
    lifecycle_tools: tuple[str, ...],
) -> tuple[tuple[str, ...], str | None]:
    """Return the optional observation tools and call-effect profile."""

    known = {item["name"] for item in definitions}
    has_effect_semantics = (
        "observation_tools" in manifest or "call_effect_profile" in manifest
    )
    observation_tools: tuple[str, ...] = ()
    call_effect_profile: str | None = None
    if has_effect_semantics:
        raw_observations = manifest.get("observation_tools")
        raw_effect_profile = manifest.get("call_effect_profile")
        if (
            not isinstance(raw_observations, list)
            or not raw_observations
            or not all(isinstance(name, str) and name for name in raw_observations)
            or len(raw_observations) != len(set(raw_observations))
        ):
            raise ToolSurfaceError("observation_tools must contain unique tool names")
        if not isinstance(raw_effect_profile, str) or not raw_effect_profile:
            raise ToolSurfaceError("call_effect_profile must be a stable id")
        observation_tools = tuple(raw_observations)
        call_effect_profile = raw_effect_profile
        if not set(observation_tools).issubset(known):
            raise ToolSurfaceError("observation_tools references unknown tools")
        if set(observation_tools) & set(mutation_tools + lifecycle_tools):
            raise ToolSurfaceError(
                "observation tools must be disjoint from mutation/lifecycle tools"
            )
    return observation_tools, call_effect_profile


def bind_capability_contracts(
    definitions: list[dict[str, Any]],
    results: dict[str, Mapping[str, Any]],
    capabilities: list[str],
    view_config: Mapping[str, Any] | None,
    *,
    profile_id: str,
) -> list[dict[str, str]]:
    """Run every declared capability binder, in the fixed binding order.

    Binders rewrite ``definitions`` and ``results`` in place; the returned
    descriptors are theirs, in binding order. Input schemas are linted before
    the null-as-omitted binder, which must read the final schemas.
    """

    descriptors: list[dict[str, str]] = []
    from ..tools.motion_arguments import MODE_HELP_CAPABILITY, annotate_mode_companions
    if MODE_HELP_CAPABILITY in capabilities:
        annotate_mode_companions(definitions)
    from .controller_feedback import CAPABILITY as CONTROLLER_FEEDBACK_CAPABILITY, bind_controller_feedback
    if CONTROLLER_FEEDBACK_CAPABILITY in capabilities:
        if view_config.get("controlled_frame_convention") != "arx_x5_link6":
            raise ToolSurfaceError("ARX controller feedback requires its own controlled frame")
        descriptors.extend(bind_controller_feedback(definitions))
    from .grasp_rejection_schema import CAPABILITY as GRASP_REJECTION_CAPABILITY, bind_grasp_rejection_diagnostics
    if GRASP_REJECTION_CAPABILITY in capabilities:
        descriptors.append(bind_grasp_rejection_diagnostics(results))
    from .grasp_ik_reachability_schema import (
        CAPABILITY as GRASP_IK_REACHABILITY_CAPABILITY, bind_grasp_ik_reachability,
    )
    from . import grasp_ik_reachability_libero_schema as grasp_ik_libero
    if grasp_ik_libero.declared(capabilities):  # LIBERO: its own reason set, released binder unchanged
        descriptors.extend(grasp_ik_libero.bind_libero_grasp_ik_reachability(results, capabilities))
    elif GRASP_IK_REACHABILITY_CAPABILITY in capabilities:
        descriptors.append(bind_grasp_ik_reachability(results, capabilities))
    from .grasp_jaw_symmetry_schema import CAPABILITY as JAW_SYMMETRY_CAPABILITY, bind_grasp_jaw_symmetry
    if JAW_SYMMETRY_CAPABILITY in capabilities:
        descriptors.append(bind_grasp_jaw_symmetry(results, capabilities, view_config))
    from .move_pose_reach_schema import JAW_CAPABILITY as MOVE_JAW, REACH_CAPABILITY as MOVE_REACH, bind_move_pose_reach
    if MOVE_JAW in capabilities or MOVE_REACH in capabilities:
        descriptors.append(bind_move_pose_reach(results, capabilities, view_config))
    from .pose_primitive_reach_schema import CAPABILITY as PRIMITIVE_REACH, bind_pose_primitive_reach
    if PRIMITIVE_REACH in capabilities:
        descriptors.append(bind_pose_primitive_reach(results, capabilities))
    from . import reach_reference_schema as reach_v2
    if reach_v2.MOVE_POSE_CAPABILITY in capabilities:
        descriptors.append(reach_v2.bind_move_pose_reach_reference(results, capabilities))
    if reach_v2.PRIMITIVE_CAPABILITY in capabilities:
        descriptors.append(reach_v2.bind_pose_primitive_reach_reference(results, capabilities))
    from .reach_orientation_schema import CAPABILITY as NEAREST_ORIENTATION, bind_reach_nearest_orientation
    if NEAREST_ORIENTATION in capabilities:
        descriptors.append(bind_reach_nearest_orientation(results, capabilities, view_config))
    if reach_v2.NEAREST_CAPABILITY in capabilities:
        descriptors.append(reach_v2.bind_reach_nearest_orientation_reference(results, capabilities, view_config))
    # The inter-arm screen comes after the grasp IK binder, whose reason enum the screen widens.
    if reach_v2.INTER_ARM_SCREEN_CAPABILITY in capabilities:
        descriptors.append(reach_v2.bind_inter_arm_screen(definitions, results, capabilities))
    from . import observed_clearance_v2_schema as clearance_v2
    if clearance_v2.CAPABILITY in capabilities:
        descriptors.append(clearance_v2.bind_observed_clearance_v2(results, capabilities, view_config))
    from .finish_outcomes_schema import (
        ABORT_CAPABILITY as ABORT_ROUND, CAPABILITY as FINISH_OUTCOMES, bind_abort_round, bind_finish_outcomes,
    )
    if FINISH_OUTCOMES in capabilities:
        descriptors.append(bind_finish_outcomes(definitions, results, capabilities))
    if ABORT_ROUND in capabilities:
        descriptors.append(bind_abort_round(definitions, results, capabilities))
    from .world_axes_schema import CAPABILITY as WORLD_AXES, bind_world_axes_orientation
    if WORLD_AXES in capabilities:
        descriptors.append(bind_world_axes_orientation(definitions, results, capabilities, view_config))
    from .compute_grammar_schema import CAPABILITY as COMPUTE_GRAMMAR, bind_compute_grammar
    if COMPUTE_GRAMMAR in capabilities:
        descriptors.append(bind_compute_grammar(definitions, capabilities))
    # After every binder that writes prose, so generated rules are restated in degrees too.
    from .policy_units_schema import CAPABILITY as POLICY_DEGREES, bind_policy_angles_degrees
    if POLICY_DEGREES in capabilities:
        descriptors.extend(bind_policy_angles_degrees(definitions, results, capabilities))
    for item in definitions:
        lint_input_schema(item.get("inputSchema"), source=f"{profile_id}.{item.get('name')}")
    # Last: it reads the final input schemas every binder above may have rewritten.
    from .null_optional_arguments import CAPABILITY as NULL_OMITS, bind_null_optional_arguments
    if NULL_OMITS in capabilities:
        descriptors.append(bind_null_optional_arguments(definitions))
    return descriptors
