"""Lazy façade for typed context sources and provider projections."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ContextBundle": (".slots", "ContextBundle"),
    "ContextSlot": (".slots", "ContextSlot"),
    "ContextSource": (".slots", "ContextSource"),
    "NativeContextPipeline": (".native", "NativeContextPipeline"),
    "ContextCompactionRejected": (".projection", "ContextCompactionRejected"),
    "causal_provider_value": (".projection", "causal_provider_value"),
    "episode_index_keyframe": (".projection", "episode_index_keyframe"),
    "RolloutCheckpointCompactor": (
        ".rollout_compactor", "RolloutCheckpointCompactor",
    ),
    "ContextError": (".codex", "ContextError"),
    "ContextPolicy": (".codex", "ContextPolicy"),
    "ContextProjector": (".codex", "ContextProjector"),
    "CurrentObservationContextPolicy": (".codex", "CurrentObservationContextPolicy"),
    "build_context_policy": (".codex", "build_context_policy"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
