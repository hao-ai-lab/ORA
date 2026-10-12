"""Typed PolicySession capability declaration for the Native runtime."""

from __future__ import annotations

from typing import Any

from ora.agent.contracts import PolicySessionCapabilities
from ora.agent.sessions.base import DelegatingPolicySession


class NativePolicySession(DelegatingPolicySession):
    def __init__(self, delegate: Any, *, supports_images: bool) -> None:
        super().__init__(
            delegate=delegate,
            capabilities=PolicySessionCapabilities(
                runtime_kind="native",
                context_owner="harness",
                tool_transport="direct",
                request_visibility="wire_exact",
                supports_images=supports_images,
                supports_interrupt=True,
            ),
        )
