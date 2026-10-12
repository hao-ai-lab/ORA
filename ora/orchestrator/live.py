"""Read-only live-event seam; interactive commands remain a future feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class HarnessLiveEvent:
    sequence: int
    kind: str
    payload: Mapping[str, Any]


class LiveInteractionPort(Protocol):
    def publish(self, event: HarnessLiveEvent) -> None: ...
    def poll(self) -> Sequence[Mapping[str, Any]]: ...


class NullLiveInteractionPort:
    """Default behavior-preserving port used until an operator console exists."""

    def publish(self, event: HarnessLiveEvent) -> None:
        del event

    def poll(self) -> tuple[Mapping[str, Any], ...]:
        return ()
