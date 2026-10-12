"""Deterministic protected-prefix composition for the Native runtime."""

from __future__ import annotations

import hashlib
from typing import Any

from ora.agent.context import ContextBundle, ContextSlot, ContextSource
from ora.agent.contracts import PolicySessionError


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def assemble_native_v3_static_context(
    *,
    harness_system: str,
    project_instructions: str,
    episode_task_template: str,
    task_instruction: str,
    task_id: str,
) -> dict[str, Any]:
    """Build the minimal protected v3 prefix without runtime/audit payloads."""

    task = task_instruction.strip()
    if not task:
        raise PolicySessionError("native task instruction must be non-empty")
    if episode_task_template.count("{{TASK}}") != 1:
        raise PolicySessionError(
            "native v3 episode task template must bind {{TASK}} once"
        )
    task_sha256 = text_sha256(task)
    system_text = harness_system.strip()
    project_text = project_instructions.strip()
    episode_text = episode_task_template.strip().replace("{{TASK}}", task)
    slots = [
        ContextSlot(ContextSource(
            kind="harness_system",
            role="system",
            order=0,
            status="protected",
            content=system_text,
        ), system_text),
        ContextSlot(ContextSource(
            kind="project_instructions",
            role="user",
            order=1,
            status="protected",
            content=project_text,
        ), project_text),
        ContextSlot(ContextSource(
            kind="episode_task",
            role="user",
            order=2,
            status="protected",
            content=episode_text,
        ), episode_text),
    ]
    bundle = ContextBundle(slots=tuple(slots))
    return {
        "task_instruction": task,
        "task_id": task_id,
        "task_sha256": task_sha256,
        "system_text": system_text,
        "project_text": project_text,
        "episode_text": episode_text,
        "messages": bundle.messages(),
        "sources": bundle.sources(),
        "context_bundle_sha256": bundle.sha256(),
    }
