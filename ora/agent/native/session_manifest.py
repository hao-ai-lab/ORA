"""Byte-identical incremental text for ``native/agent-session.json``.

The Session manifest embeds every Provider call record, and each record embeds
that call's full Context projection manifest. It is rewritten after every
Provider response, so serializing it from scratch made the post-response work
quadratic in the run (155 MB of ``indent=2`` pure-Python encoding per call at
the end of a 188-call recorded session).

:class:`IncrementalManifestText` produces exactly
``json.dumps(body, ensure_ascii=False, indent=2, sort_keys=True) + "\\n"`` but
serializes each append-only list element (``calls``, ``usage_events``) only
once. Elements are keyed by list position and object identity and pinned
alive, so a replaced element is re-serialized. Records are never mutated after
they are appended; the Session still performs one full serialization for the
final ``closed`` manifest, so the sealed file never depends on this cache.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping


INDENT = 2
SPLICED_LIST_KEYS = ("calls", "usage_events")
# A long run's manifest reaches 155 MB; rewriting it after every call put about
# 14 GB on the shared HDD per run. Snapshots are a crash convenience only.
SNAPSHOT_MIN_INTERVAL_S = 120.0


def dump_manifest_text(body: Mapping[str, Any]) -> str:
    """The reference serialization of the Session manifest."""

    return json.dumps(body, ensure_ascii=False, indent=INDENT, sort_keys=True) + "\n"


class IncrementalManifestText:
    """Serialize top-level append-only lists once per element."""

    def __init__(self) -> None:
        self._token = uuid.uuid4().hex
        self._items: dict[str, list[tuple[Any, str]]] = {}

    def _item_text(self, key: str, index: int, item: Any) -> str:
        cache = self._items.setdefault(key, [])
        if index < len(cache) and cache[index][0] is item:
            return cache[index][1]
        # A top-level list element sits at nesting level 2. Standalone
        # ``indent=2`` output nested there gains 2 * INDENT spaces per line;
        # structural newlines are the only raw newlines JSON emits.
        text = json.dumps(
            item, ensure_ascii=False, indent=INDENT, sort_keys=True,
        ).replace("\n", "\n" + " " * (2 * INDENT))
        del cache[index:]
        cache.append((item, text))
        return text

    def _list_text(self, key: str, values: list[Any]) -> str:
        if not values:
            self._items.pop(key, None)
            return "[]"
        prefix = "\n" + " " * (2 * INDENT)
        items = [self._item_text(key, index, item) for index, item in enumerate(values)]
        return "[" + prefix + ("," + prefix).join(items) + "\n" + " " * INDENT + "]"

    def render(self, body: Mapping[str, Any]) -> str:
        spliced: dict[str, str] = {}
        skeleton = dict(body)
        for key in SPLICED_LIST_KEYS:
            value = body.get(key)
            if type(value) is list:
                marker = f"@@ora-splice-{self._token}-{key}@@"
                skeleton[key] = marker
                spliced[json.dumps(marker)] = self._list_text(key, value)
        text = dump_manifest_text(skeleton)
        for marker, replacement in spliced.items():
            if text.count(marker) != 1:
                return dump_manifest_text(body)
            text = text.replace(marker, replacement)
        return text


class SessionManifestWriter:
    """Publish mid-run snapshots off the loop and the closed manifest exactly.

    Mid-run snapshots go to the durability worker latest-wins and at most once
    per ``min_interval_s``; the event log stays the authority for every call,
    so after a crash the file may lag it by that interval plus the queued
    snapshot. The closed manifest first drains every pending snapshot, then
    uses the reference serializer and is fsynced behind the close barrier.
    """

    def __init__(
        self, target: Path, durability: Any, *,
        min_interval_s: float = SNAPSHOT_MIN_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.target = target
        self.durability = durability
        self.text = IncrementalManifestText()
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.last_snapshot: float | None = None
        self.skipped_snapshots = 0

    def write(self, body: Mapping[str, Any], *, final: bool) -> None:
        if not final:
            now = self.clock()
            if (
                self.last_snapshot is not None
                and now - self.last_snapshot < self.min_interval_s
            ):
                self.skipped_snapshots += 1
                return
            self.last_snapshot = now
            self.durability.replace_text(self.target, self.text.render(body))
            return
        self.durability.barrier()
        temporary = self.target.with_suffix(".json.tmp")
        temporary.write_text(dump_manifest_text(body), encoding="utf-8")
        os.replace(temporary, self.target)
        self.durability.submit(self.target)
