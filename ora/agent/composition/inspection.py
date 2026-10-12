"""Read-only inspection of one exact, resolved policy configuration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "ora.policy-config-inspection.v1"
V3_SCHEMA = "ora.policy-context.v3"


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compiled_policy_paths(
    root: Path, policy_context: Mapping[str, Any],
) -> dict[str, Path]:
    directory = root / "compiled-policy"
    if policy_context.get("schema") == V3_SCHEMA:
        names = {
            "harness_system": "harness-system.md",
            "project_instructions": "project-instructions.md",
            "episode_task": "episode-task.md",
            "tool_manual": "tool-manual.md",
        }
    else:
        names = {
            "codex_project": "AGENTS.md",
            "episode_prefix": "episode-prefix.md",
            "tool_manual": "robot-manual.md",
        }
    return {channel: directory / filename for channel, filename in names.items()}


def inspect_policy_configuration(
    plan: Mapping[str, Any], *, context_manifest: Mapping[str, Any],
    compiled_policy: Mapping[str, Path], run_id: str | None = None,
) -> dict[str, Any]:
    """Build one operator view without inventing unavailable provider context."""

    policy = plan.get("policy") if isinstance(plan.get("policy"), Mapping) else {}
    runtime = plan.get("runtime") if isinstance(plan.get("runtime"), Mapping) else {}
    policy_context = (
        plan.get("policy_context")
        if isinstance(plan.get("policy_context"), Mapping) else {}
    )
    composition = (
        plan.get("policy_composition")
        if isinstance(plan.get("policy_composition"), Mapping) else {}
    )
    planned_surface = (
        composition.get("tool_surface")
        if isinstance(composition.get("tool_surface"), Mapping) else {}
    )
    manifest_surface = (
        context_manifest.get("tool_surface")
        if isinstance(context_manifest.get("tool_surface"), Mapping) else {}
    )

    channels: list[dict[str, Any]] = []
    for order, (channel, path) in enumerate(compiled_policy.items()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        channels.append({
            "order": order,
            "channel": channel,
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "text": text,
        })

    sources: list[dict[str, Any]] = []
    source_order = 0
    for channel, layers in policy_context.items():
        if channel in {"schema", "composition_sha256"} or not isinstance(layers, list):
            continue
        for channel_index, layer in enumerate(layers):
            if not isinstance(layer, Mapping):
                continue
            sources.append({
                "order": source_order,
                "channel": channel,
                "channel_index": channel_index,
                "path": layer.get("path"),
                "bytes": layer.get("bytes"),
                "sha256": layer.get("sha256"),
            })
            source_order += 1

    slots: list[dict[str, Any]] = []
    raw_order = context_manifest.get("context_order")
    if isinstance(raw_order, list):
        for fallback_order, item in enumerate(raw_order):
            if not isinstance(item, Mapping):
                continue
            kind = str(item.get("kind") or f"slot_{fallback_order}")
            exact_text = item.get("exact_text")
            dynamic = "history" in kind or item.get("example") is not None
            definitions = item.get("definitions")
            slots.append({
                "order": item.get("order", fallback_order),
                "kind": kind,
                "role": item.get("role"),
                "owner": item.get("owner"),
                "delivery": item.get("delivery"),
                "status": (
                    "exact" if isinstance(exact_text, str)
                    else "runtime_dynamic" if dynamic
                    else "separate_tool_field" if isinstance(definitions, list)
                    else "structural"
                ),
                "sha256": item.get("exact_sha256"),
                "text": exact_text if isinstance(exact_text, str) else None,
                "example": item.get("example") if dynamic else None,
                "source_layers": item.get("source_layers"),
                "note": item.get("placement_note"),
            })
    if not slots:
        for channel in channels:
            slots.append({
                "order": channel["order"],
                "kind": channel["channel"],
                "role": "unknown",
                "delivery": "compiled policy file",
                "status": "exact_compiled_text",
                "sha256": channel["sha256"],
                "text": channel["text"],
            })

    for slot in slots:
        kind = str(slot.get("kind") or "")
        related_channels: set[str] = set()
        for channel in policy_context:
            if channel in kind:
                related_channels.add(channel)
        if kind == "stable_system_policy_prefix":
            related_channels.update(("codex_project", "episode_prefix"))
        elif kind == "canonical_tool_schemas":
            related_channels.add("tool_manual")
        slot["sources"] = [
            source for source in sources
            if source["channel"] in related_channels
        ]

    definitions = manifest_surface.get("advertised_in_order")
    if not isinstance(definitions, list):
        definitions = planned_surface.get("definitions")
    definitions = definitions if isinstance(definitions, list) else []
    mutation_tools = manifest_surface.get("state_changing_tools")
    if not isinstance(mutation_tools, list):
        mutation_tools = planned_surface.get("mutation_tools", [])
    lifecycle_tools = manifest_surface.get("lifecycle_tools")
    if not isinstance(lifecycle_tools, list):
        lifecycle_tools = planned_surface.get("lifecycle_tools", [])
    tools = []
    for definition in definitions:
        if not isinstance(definition, Mapping):
            continue
        name = str(definition.get("name") or "unknown")
        tools.append({
            "name": name,
            "state_effect": definition.get("state_effect") or (
                "state_changing_terminal_for_observation"
                if name in mutation_tools
                else "policy_lifecycle" if name in lifecycle_tools
                else "read_only"
            ),
            "description": definition.get("description"),
            "input_schema": definition.get("input_schema"),
            "result_schema": (
                planned_surface.get("result_schemas", {}).get(name)
                if isinstance(planned_surface.get("result_schemas"), Mapping)
                else None
            ),
        })

    return {
        "schema": SCHEMA,
        "run_id": run_id,
        "identity": {
            "experiment": plan.get("experiment"),
            "config": plan.get("config"),
            "milestone": plan.get("milestone"),
            "task": plan.get("task"),
            "task_instruction": plan.get("task_instruction"),
            "seed": plan.get("seed"),
            "plan_id": plan.get("plan_id"),
            "plan_sha256": plan.get("plan_sha256"),
            "source_commit": (
                plan.get("source", {}).get("commit")
                if isinstance(plan.get("source"), Mapping) else None
            ),
        },
        "runtime": {
            "backend": runtime.get("kind"),
            "transport": runtime.get("transport"),
            "environment_prompt_profile": runtime.get("environment_prompt_profile"),
        },
        "planner": context_manifest.get("planner", {
            key: policy.get(key) for key in (
                "kind", "provider", "provider_profile", "model",
                "reasoning_effort", "context_composition", "context_profile",
                "context_retention",
                "initial_observation",
            )
        }),
        "composition": {
            "schema": composition.get("schema"),
            "composition_sha256": composition.get("composition_sha256"),
            "prompt_composition_sha256": policy_context.get("composition_sha256"),
            "tool_surface_profile": (
                manifest_surface.get("profile") or planned_surface.get("profile_id")
            ),
            "tool_surface_sha256": (
                manifest_surface.get("composition_sha256")
                or planned_surface.get("composition_sha256")
            ),
            "action_codec": composition.get("action_codec"),
            "initial_observation": composition.get("initial_observation"),
        },
        "prompt": {
            "slots": sorted(slots, key=lambda item: int(item["order"])),
            "sources": sources,
            "compiled_channels": channels,
            "dynamic_call_evidence": context_manifest.get("dynamic_call_evidence"),
        },
        "tool_surface": {
            "profile": manifest_surface.get("profile") or planned_surface.get("profile_id"),
            "definitions": tools,
            "observation_tools": manifest_surface.get(
                "observation_tools", planned_surface.get("observation_tools", [])
            ),
            "mutation_tools": mutation_tools,
            "lifecycle_tools": lifecycle_tools,
            "call_effect_profile": manifest_surface.get(
                "call_effect_profile", planned_surface.get("call_effect_profile")
            ),
            "view_config": manifest_surface.get(
                "view_config", planned_surface.get("view_config")
            ),
        },
        "evidence": {
            "context_manifest_schema": context_manifest.get("schema"),
            "context_manifest_run_id": context_manifest.get("run_id"),
            "projection": "read_only_exact_run_configuration",
        },
    }


def inspect_run_configuration(root: Path) -> dict[str, Any]:
    root = root.resolve()
    plan = _load(root / "plan.json")
    if not plan:
        raise ValueError("run has no readable plan.json")
    context_manifest = _load(root / "context-manifest.json")
    paths = compiled_policy_paths(root, plan.get("policy_context", {}))
    inspection = inspect_policy_configuration(
        plan, context_manifest=context_manifest, compiled_policy=paths,
        run_id=root.name,
    )
    result = _load(root / "result.json")
    launch = _load(root / "launch.json")
    inspection["identity"]["started_unix_s"] = result.get(
        "started_unix_s", launch.get("created_unix_s")
    )
    return inspection


__all__ = [
    "SCHEMA", "compiled_policy_paths", "inspect_policy_configuration",
    "inspect_run_configuration",
]
