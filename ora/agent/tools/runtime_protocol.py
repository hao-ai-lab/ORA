"""Shared tool-runtime contract used by direct and MCP transports."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class ToolRuntime(Protocol):
    """One validation, permission, execution and normalization choke point."""

    def definitions(self) -> Sequence[Mapping[str, Any]]: ...
    def classify_call_effect(
        self, name: str, arguments: Mapping[str, Any],
    ) -> str: ...
    def invoke(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...
