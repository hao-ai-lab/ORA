"""``model_visible_sha256``: one digest of what a RunPlan lets the model see.

The digest is built from five items of a resolved RunPlan. It stays fixed when
an asset edit only changes a pin (a ``*sha256`` value), a provenance note, a
manifest revision or a file path, and it moves when a delivered tool
definition, a prompt layer's text, an embodiment card, the projected public
facts or a policy/runtime/budget setting changes. Later refactors and
cross-commit comparisons (for example the development line against a cluster
tag) compare this one recorded field, ``policy_identity.model_visible_sha256``;
``scripts/compare-model-visible`` recomputes the same items for commits that
predate it.

This module imports nothing from the orchestrator, so the comparison script
can load it from its own checkout while resolving another commit's tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

#: Keys of ``policy_composition.tool_surface`` that are the advertised surface:
#: the delivered definitions and result schemas, the tool roles and effect
#: profile, the derived layers and the runtime that executes them.
ADVERTISED_KEYS = (
    "definitions", "result_schemas", "mutation_tools", "lifecycle_tools",
    "observation_tools", "call_effect_profile", "derived_layers", "runtime_id",
)
#: Runtime keys that choose where a run executes -- host, GPU slot, renderer
#: and execution-resource lease -- never what the policy reads. Lanes of one
#: cohort differ only here, and so do two hosts running the same cell.
PLACEMENT_KEYS = ("resource_binding", "required_host", "cuda_visible_devices", "renderer_gpu")
#: ``policy_composition`` keys covered by another item, or identity only.
_COMPOSITION_EXCLUDED = {"schema", "prompt_layers", "tool_surface", "composition_sha256"}
_EMBODIMENT_ROLES = ("compiled_embodiment",)


def canonical_sha256(value: Any) -> str:
    """The same canonical JSON hash as ``spec_values.json_sha256``."""

    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def mask_pins(value: Any) -> Any:
    """Drop every hash and every pin-only path, keeping all other settings.

    In each mapping this removes:

    - every key ending in ``sha256`` (content pins, composition and profile
      hashes, external identities written as hashes);
    - ``path``, ``ref`` and ``bytes`` when the mapping also has ``sha256``
      (a ``{path, sha256}`` file binding or ``{ref, sha256}`` component);
    - ``X_path`` when the mapping has ``X_sha256`` (``profile_path``);
    - ``X`` when the mapping has ``X_sha256`` (a RoboDojo contract ``record``).

    A path that is not beside its own hash is a setting and is kept.
    """

    if isinstance(value, Mapping):
        masked = {}
        for key, item in value.items():
            key = str(key)
            if key.endswith("sha256"):
                continue
            if key in {"path", "ref", "bytes"} and "sha256" in value:
                continue
            if key.endswith("_path") and f"{key[:-5]}_sha256" in value:
                continue
            if f"{key}_sha256" in value:
                continue
            masked[key] = mask_pins(item)
        return masked
    if isinstance(value, list):
        return [mask_pins(item) for item in value]
    return copy.deepcopy(value)


def _card_facts(root: Path, relative: str) -> dict[str, Any]:
    """Card and projected-fact hashes of one descriptor, by the tree's own renderer."""

    try:
        from ora.agent.composition.embodiment import (
            compile_embodiment_card, load_embodiment_descriptor,
        )

        _card, manifest = compile_embodiment_card(load_embodiment_descriptor(Path(root) / relative))
        return {
            "card_sha256": manifest["card_sha256"],
            "projected_facts_sha256": manifest["projected_facts_sha256"],
        }
    except Exception as exc:  # recorded, never raised: the item stays comparable
        return {"unavailable": type(exc).__name__}


def _embodiment(plan: Mapping[str, Any], root: Path) -> dict[str, Any]:
    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    descriptors: dict[str, str] = {}
    geometry = runtime.get("public_geometry_descriptor")
    if isinstance(geometry, Mapping) and isinstance(geometry.get("path"), str):
        descriptors["public_geometry_descriptor"] = geometry["path"]
    profile = runtime.get("environment_prompt_profile")
    if isinstance(profile, Mapping) and isinstance(profile.get("profile_path"), str):
        try:
            value = json.loads((Path(root) / profile["profile_path"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            value = {}
        for role in _EMBODIMENT_ROLES:
            binding = value.get(role) if isinstance(value, Mapping) else None
            descriptor = binding.get("descriptor") if isinstance(binding, Mapping) else None
            # A profile names its descriptor by path; trees before stage 2 unit
            # L2 wrote {path, sha256}. Both are read so commits on either side compare.
            path = descriptor.get("path") if isinstance(descriptor, Mapping) else descriptor
            if isinstance(path, str):
                descriptors[role] = path
    return {role: _card_facts(root, path) for role, path in sorted(descriptors.items())}


def model_visible_items(plan: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """The five model-visible items of a resolved RunPlan.

    ``root`` is the source tree the plan was resolved from; item 3 reads the
    bound descriptors from it.

    1. ``advertised``: the advertised composition. From
       ``policy_composition.tool_surface`` the ``ADVERTISED_KEYS`` (tool
       definitions as delivered, result schemas, tool roles, effect profile,
       derived layers, runtime id), plus the remaining ``policy_composition``
       entries (action codec, initial observation) with pins masked. The manifest's
       components, provenance, revision, profile id and hashes are left out.
    2. ``prompt_layers``: for each ``policy_context`` channel, in delivery
       order, the sha256 of each layer's text, and a composition hash over
       those text hashes. Layer paths and byte counts are left out.
    3. ``embodiment``: for each bound descriptor (the runtime's
       ``public_geometry_descriptor`` and the context profile's compiled or
       source-bound embodiment), the sha256 of the card rendered from it and
       of its projected public facts. Neither includes provenance.
    4. ``observation_projection``: the ToolSurface ``view_config`` with pins
       masked and its ``capabilities`` list.
    5. ``settings``: the ``policy``, ``runtime`` and ``budgets`` blocks with
       ``mask_pins`` applied, and the runtime's ``PLACEMENT_KEYS`` removed.

    Not included: the RunPlan's episode identity (experiment, config,
    milestone, task, task instruction, seed), ``source``, ``qualification``,
    ``session``, ``launch_admission`` and ``experiment_identity``.
    """

    composition = plan.get("policy_composition") or {}
    surface = composition.get("tool_surface") or {}
    advertised = {key: copy.deepcopy(surface[key]) for key in ADVERTISED_KEYS if key in surface}
    advertised["composition"] = mask_pins({
        key: value for key, value in composition.items() if key not in _COMPOSITION_EXCLUDED
    })
    context = plan.get("policy_context") or {}
    channels = {
        channel: [layer.get("sha256") for layer in layers if isinstance(layer, Mapping)]
        for channel, layers in sorted(context.items()) if isinstance(layers, list)
    }
    prompt_layers = {
        "schema": context.get("schema"), "channels": channels,
        "composition_sha256": canonical_sha256({"schema": context.get("schema"), "channels": channels}),
    }
    return {
        "advertised": advertised,
        "prompt_layers": prompt_layers,
        "embodiment": _embodiment(plan, Path(root)),
        "observation_projection": {
            "view_config": mask_pins(surface.get("view_config")),
            "capabilities": copy.deepcopy(surface.get("capabilities")),
        },
        "settings": {
            "policy": mask_pins(plan.get("policy")),
            "runtime": mask_pins({key: value for key, value in (plan.get("runtime") or {}).items()
                                  if key not in PLACEMENT_KEYS}),
            "budgets": mask_pins(plan.get("budgets")),
        },
    }


def model_visible_sha256(plan: Mapping[str, Any], *, root: Path) -> str:
    """The canonical sha256 of ``model_visible_items(plan, root=root)``."""

    return canonical_sha256(model_visible_items(plan, root=root))
