"""Canonical environment contract with a policy-safe runtime descriptor."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .base import EnvironmentSession as LegacyEnvironmentSession


@runtime_checkable
class EnvironmentSession(LegacyEnvironmentSession, Protocol):
    def describe(self) -> Mapping[str, Any]:
        """Return only the registered, policy-safe runtime descriptor."""
