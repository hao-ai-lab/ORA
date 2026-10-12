"""Materialize hash-bound Markdown layers selected by a RunPlan."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from ora.agent.context.slots import ContextBundle, ContextSlot, ContextSource


LEGACY_CHANNELS = ("codex_project", "episode_prefix", "tool_manual")
POLICY_CONTEXT_V3_SCHEMA = "ora.policy-context.v3"
V2_CHANNELS = (
    "harness_system", "project_instructions", "episode_task", "tool_manual",
)
V3_CHANNELS = V2_CHANNELS


class PolicyCompositionError(RuntimeError):
    """A planned policy layer is missing or changed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def materialize_policy(
    composition: Mapping[str, Any], *, project_root: Path, destination: Path
) -> dict[str, Path]:
    """Compile ordered source layers without allowing post-plan drift."""
    project_root = project_root.resolve()
    destination = destination.resolve()
    typed_schema = composition.get("schema") == POLICY_CONTEXT_V3_SCHEMA
    channels = V2_CHANNELS if typed_schema else LEGACY_CHANNELS
    outputs = (
        {
            "harness_system": destination / "harness-system.md",
            "project_instructions": destination / "project-instructions.md",
            "episode_task": destination / "episode-task.md",
            "tool_manual": destination / "tool-manual.md",
        }
        if typed_schema else {
            "codex_project": destination / "AGENTS.md",
            "episode_prefix": destination / "episode-prefix.md",
            "tool_manual": destination / "robot-manual.md",
        }
    )
    for channel in channels:
        layers = composition.get(channel)
        if not isinstance(layers, list) or not layers:
            raise PolicyCompositionError(f"policy_context.{channel} must be non-empty")
        rendered: list[str] = []
        for layer in layers:
            path = (project_root / layer["path"]).resolve()
            if (
                not path.is_file()
                or not path.is_relative_to(project_root)
                or _sha256(path) != layer["sha256"]
            ):
                raise PolicyCompositionError(f"policy layer changed after planning: {layer['path']}")
            text = path.read_text(encoding="utf-8").strip()
            rendered.append(
                text
                if typed_schema
                else f"<!-- policy-layer: {layer['path']} -->\n{text}"
            )
        _atomic_text(outputs[channel], "\n\n".join(rendered) + "\n")
    if typed_schema:
        # Workspace projection still consumes these historical aliases. Native
        # request assembly uses the role-aware paths above.
        outputs["codex_project"] = outputs["project_instructions"]
        outputs["episode_prefix"] = outputs["episode_task"]
    source_channels = V2_CHANNELS if typed_schema else LEGACY_CHANNELS
    roles = (
        {
            "harness_system": "system",
            "project_instructions": "user",
            "episode_task": "user",
            "tool_manual": "tool_manual",
        }
        if typed_schema else {
            "codex_project": "project_document",
            "episode_prefix": "user",
            "tool_manual": "tool_manual",
        }
    )
    sources = tuple(
        ContextSource(
            kind=channel,
            role=roles[channel],
            order=index,
            status="compiled_exact",
            content=outputs[channel].read_text(encoding="utf-8"),
        )
        for index, channel in enumerate(source_channels)
    )
    bundle = ContextBundle(
        slots=tuple(
            ContextSlot(source=source, message_content=source.content)
            for source in sources if source.kind != "tool_manual"
        ),
        manifest_only_sources=tuple(
            source for source in sources if source.kind == "tool_manual"
        ),
    )
    _atomic_text(
        destination / "context-sources.json",
        json.dumps({
            "schema": "ora.compiled-context-sources.v1",
            "policy_context_schema": composition.get("schema", "legacy-v1"),
            "bundle_sha256": bundle.sha256(),
            "sources": bundle.sources(),
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(
        destination / "composition.json",
        json.dumps(dict(composition), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return outputs
