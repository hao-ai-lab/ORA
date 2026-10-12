"""Recover facts about an episode from what its run already persisted.

Both helpers read only files the run wrote -- the observation's main-camera RGB
artifact and the provider completions in the agent events -- and
neither holds coordinator state. They moved out of ``coordinator.py`` to keep
it within its size ratchet; ``coordinator`` re-exports them for its callers and
the existing tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator.episode_bindings import SupervisorError


def _main_observation_image(
    observation: Mapping[str, Any], observation_root: Path,
) -> Path:
    main_camera = observation.get("main_camera", "head_camera")
    cameras = observation.get("cameras")
    camera = cameras.get(main_camera) if isinstance(cameras, Mapping) else None
    files = camera.get("files") if isinstance(camera, Mapping) else None
    reference = (
        files.get("rgb_lossless_png") if isinstance(files, Mapping) else None
    )
    relative = reference.get("path") if isinstance(reference, Mapping) else None
    if not isinstance(main_camera, str) or not isinstance(relative, str):
        raise SupervisorError("observation main-camera RGB reference is unavailable")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise SupervisorError("observation main-camera RGB reference is unsafe")
    root = observation_root.resolve()
    image = (root / relative_path).resolve()
    if not image.is_relative_to(root) or not image.is_file() or image.is_symlink():
        raise SupervisorError("observation main-camera RGB artifact is unavailable")
    return image


def _reconciled_provider_model_calls(
    run_root: Path, plan: Mapping[str, Any], fallback: int,
) -> int:
    """Use persisted provider completions even when a policy turn fails."""

    # A Native API turn can fail after one or more provider completions but
    # before PolicySession.act returns its aggregate count.  Each accepted
    # completion is already an exact, append-only agent event, so recover the
    # count from that evidence instead of reporting zero.
    if plan.get("policy", {}).get("kind") == "native_api":
        events_path = run_root / "agent-events.jsonl"
        try:
            accepted_ids: set[str] = set()
            for line in events_path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                if (
                    isinstance(event, Mapping)
                    and event.get("type") == "provider_response"
                    and isinstance(event.get("payload"), Mapping)
                    and event["payload"].get("accepted") is True
                    and isinstance(event.get("model_call_id"), str)
                ):
                    accepted_ids.add(event["model_call_id"])
            return max(fallback, len(accepted_ids))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return fallback
    return fallback
