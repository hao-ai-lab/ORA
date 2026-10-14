"""Shared typed façade for concrete PolicySession adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..contracts import (
    PolicyOutcome,
    PolicyRuntimeEvent,
    PolicySessionCapabilities,
    PolicyTurnInput,
    policy_outcome_from_mapping,
)


@dataclass
class DelegatingPolicySession:
    """Translate the typed outer contract to one runtime-owned delegate."""

    delegate: Any
    capabilities: PolicySessionCapabilities

    def start(self) -> None:
        self.delegate.start()

    def act(self, turn: PolicyTurnInput) -> PolicyOutcome:
        value = self.delegate.act(**turn.runtime_kwargs())
        if not isinstance(value, Mapping):
            raise TypeError("PolicySession delegate returned an invalid outcome")
        return policy_outcome_from_mapping(value)

    def accept_runtime_event(self, event: PolicyRuntimeEvent) -> None:
        payload = event.mutable_payload()
        method = {
            "fresh_observation": "record_fresh_observation",
            "environment_receipt": "record_environment_receipt",
            "official_verdict": "record_official_verdict",
            "run_termination": "record_run_termination",
            "operator_message": "record_operator_message",
        }[event.kind]
        callback = getattr(self.delegate, method, None)
        if callable(callback):
            callback(
                payload.get("value")
                if event.kind == "official_verdict"
                else payload
            )
        elif event.kind == "operator_message":
            raise TypeError("PolicySession delegate does not support operator messages")

    def interrupt(self, reason: str = "operator_interrupt") -> None:
        callback = getattr(self.delegate, "interrupt", None)
        if callable(callback):
            callback(reason)

    def close(self) -> None:
        callback = getattr(self.delegate, "close", None)
        if callable(callback):
            callback()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)
