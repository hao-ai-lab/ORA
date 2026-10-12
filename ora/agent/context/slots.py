"""Ordered, hash-bound context slots independent of provider wire roles."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContextSource:
    kind: str
    role: str
    order: int
    status: str
    content: str

    def __post_init__(self) -> None:
        if not self.kind or not self.role or self.order < 0 or not self.status:
            raise ValueError("context source identity is invalid")

    def record(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "role": self.role,
            "order": self.order,
            "status": self.status,
            "bytes": len(self.content.encode("utf-8")),
            "sha256": _sha256(self.content),
        }


@dataclass(frozen=True)
class ContextSlot:
    source: ContextSource
    message_content: str
    protected: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def message(self) -> dict[str, Any]:
        value = {
            "role": self.source.role,
            "content": self.message_content,
        }
        if self.protected:
            value["_protected_context"] = True
        value["_context_source"] = self.source.kind
        value.update(copy.deepcopy(dict(self.metadata)))
        return value


@dataclass(frozen=True)
class ContextBundle:
    slots: tuple[ContextSlot, ...]
    manifest_only_sources: tuple[ContextSource, ...] = ()

    def __post_init__(self) -> None:
        sources = [slot.source for slot in self.slots] + list(self.manifest_only_sources)
        kinds = [source.kind for source in sources]
        orders = [source.order for source in sources]
        if len(kinds) != len(set(kinds)) or len(orders) != len(set(orders)):
            raise ValueError("context source kinds and orders must be unique")

    def messages(self) -> list[dict[str, Any]]:
        return [slot.message() for slot in sorted(
            self.slots, key=lambda item: item.source.order,
        )]

    def sources(self) -> list[dict[str, Any]]:
        values = [slot.source for slot in self.slots] + list(self.manifest_only_sources)
        return [source.record() for source in sorted(values, key=lambda item: item.order)]

    def sha256(self) -> str:
        payload = {
            "sources": self.sources(),
            "messages": self.messages(),
        }
        return hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
