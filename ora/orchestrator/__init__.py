"""Lazy public façade for trusted episode orchestration."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "EpisodeCoordinator": (".coordinator", "EpisodeCoordinator"),
    "SupervisorError": (".coordinator", "SupervisorError"),
    "run_episode": (".coordinator", "run_episode"),
    "HarnessFault": (".faults", "HarnessFault"),
    "HarnessFaultError": (".faults", "HarnessFaultError"),
    "classify_harness_fault": (".faults", "classify_harness_fault"),
    "EpisodeInteractionState": (".interaction", "EpisodeInteractionState"),
    "InteractionStateError": (".interaction", "InteractionStateError"),
    "HarnessLiveEvent": (".live", "HarnessLiveEvent"),
    "LiveInteractionPort": (".live", "LiveInteractionPort"),
    "NullLiveInteractionPort": (".live", "NullLiveInteractionPort"),
    "TerminationAuthority": (".termination", "TerminationAuthority"),
    "TerminationDecision": (".termination", "TerminationDecision"),
    "policy_outcome_decision": (".termination", "policy_outcome_decision"),
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
