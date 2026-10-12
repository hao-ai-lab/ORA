"""Typed PolicySession wrapper for deterministic scripted qualification."""

from __future__ import annotations

from typing import Any, Mapping

from ..contracts import (
    PolicyOutcome,
    PolicySessionCapabilities,
    PolicyTurnInput,
    policy_outcome_from_mapping,
)
from .base import DelegatingPolicySession


class ScriptedPolicySession(DelegatingPolicySession):
    def __init__(self, delegate: Any) -> None:
        super().__init__(
            delegate=delegate,
            capabilities=PolicySessionCapabilities(
                runtime_kind="scripted",
                context_owner="none",
                tool_transport="in_process",
                request_visibility="none",
                supports_images=False,
                supports_interrupt=False,
            ),
        )

    def start(self) -> None:
        return None

    def act(self, turn: PolicyTurnInput) -> PolicyOutcome:
        value = self.delegate.run_observation(**turn.runtime_kwargs())
        if not isinstance(value, Mapping):
            raise TypeError("scripted planner returned an invalid outcome")
        return policy_outcome_from_mapping(value)
