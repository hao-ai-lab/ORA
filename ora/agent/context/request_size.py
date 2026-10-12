"""Exact canonical-JSON sizes that do not re-serialize inline media each time.

Lossless Context evidence records ``len(canonical_bytes(value))`` for whole
histories (``passes[*].pre_bytes``, ``pre_lossless_request_utf8_bytes`` and the
``(bytes + 2) // 3`` token estimates derived from them). Those histories keep
every original ``data:`` image URL, so serializing them made each projection
linear in all media ever observed. A value's canonical length equals the
length of the same value with every large ``data:`` string replaced by ``""``
plus, per replaced string, its own canonical length minus the two quote bytes.
Strings are immutable, so a string's length can be memoized by the string
itself; the numbers stay byte-exact while the per-projection work becomes the
text plus media first seen in that projection.
"""
from __future__ import annotations

import contextvars
import functools
from typing import Any, Callable, Hashable, Mapping, TypeVar

from ora.agent.evidence.trajectory import canonical_bytes

MEMOIZED_STRING_MIN_CHARS = 1024
_SCALARS = (int, float, bool, type(None))
_ACTIVE: contextvars.ContextVar["CanonicalSizer | None"] = contextvars.ContextVar(
    "ora_context_canonical_sizer", default=None,
)
_Method = TypeVar("_Method", bound=Callable[..., Any])


class GenerationalMemo:
    """Keep only entries used by the current or immediately previous projection."""

    def __init__(self) -> None:
        self._current: dict[Hashable, Any] = {}
        self._previous: dict[Hashable, Any] = {}

    def rotate(self) -> None:
        self._previous, self._current = self._current, {}

    def get(self, key: Hashable) -> Any:
        if key in self._current:
            return self._current[key]
        if key in self._previous:
            value = self._current[key] = self._previous.pop(key)
            return value
        return None

    def put(self, key: Hashable, value: Any) -> None:
        self._current[key] = value

    def discard(self, key: Hashable) -> None:
        self._current.pop(key, None)
        self._previous.pop(key, None)


def _legacy_source_ids(value: Any, found: set[str]) -> None:
    """The historical recursive ``_source_event_ids`` walk for rare non-dicts."""

    if isinstance(value, Mapping):
        raw = value.get("_source_event_ids")
        if isinstance(raw, list):
            found.update(item for item in raw if isinstance(item, str))
        for child in value.values():
            _legacy_source_ids(child, found)
    elif isinstance(value, list):
        for child in value:
            _legacy_source_ids(child, found)


class CanonicalSizer:
    """Measure canonical length and source-event identities in one walk."""

    def __init__(self) -> None:
        self._lengths = GenerationalMemo()
        # id(message) -> (message, length, source ids) for the current
        # projection only; the pinned object keeps its id from being reused.
        self._elements: dict[int, tuple[Any, int, frozenset[str]]] = {}

    def rotate(self) -> None:
        self._lengths.rotate()
        self._elements = {}

    def _string_length(self, value: str) -> int:
        cached = self._lengths.get(value)
        if cached is None:
            cached = len(canonical_bytes(value))
            self._lengths.put(value, cached)
        return cached

    def measure(self, value: Any) -> tuple[int, set[str]]:
        """Return ``(len(canonical_bytes(value)), recursive _source_event_ids)``."""

        if type(value) is list and value and all(type(item) is dict for item in value):
            return self._measure_message_list(value)
        return self._measure_value(value)

    def _measure_message_list(self, value: list[dict[str, Any]]) -> tuple[int, set[str]]:
        """Measure a message list element by element.

        ``canonical_bytes`` joins array elements with ``,`` inside ``[]``, so
        the list length is ``2 + (n - 1) + sum(element lengths)`` and the id
        set is the union of the elements' sets. Lossless passes are
        copy-on-write, so within one projection an unchanged message is the
        same, never-edited object and is walked once.
        """

        total = 1 + len(value)
        found: set[str] = set()
        elements = self._elements
        for item in value:
            cached = elements.get(id(item))
            if cached is None or cached[0] is not item:
                length, ids = self._measure_value(item)
                cached = (item, length, frozenset(ids))
                elements[id(item)] = cached
            total += cached[1]
            found.update(cached[2])
        return total, found

    def _measure_value(self, value: Any) -> tuple[int, set[str]]:
        found: set[str] = set()
        extra = 0

        def walk(item: Any, collect: bool) -> Any:
            nonlocal extra
            kind = type(item)
            if kind is str:
                if len(item) >= MEMOIZED_STRING_MIN_CHARS and item.startswith("data:"):
                    extra += self._string_length(item) - 2
                    return ""
                return item
            if kind in _SCALARS:
                return item
            if isinstance(item, dict):
                if collect:
                    raw = item.get("_source_event_ids")
                    if isinstance(raw, list):
                        found.update(child for child in raw if isinstance(child, str))
                return {key: walk(child, collect) for key, child in item.items()}
            if isinstance(item, Mapping):
                # JSON serialization rejects it below exactly as before.
                if collect:
                    _legacy_source_ids(item, found)
                return item
            if isinstance(item, list):
                return [walk(child, collect) for child in item]
            if isinstance(item, tuple):
                # Serialized as an array; the historical id walk never entered it.
                return [walk(child, False) for child in item]
            return item

        scrubbed = walk(value, True)
        return len(canonical_bytes(scrubbed)) + extra, found

    def size(self, value: Any) -> int:
        return self.measure(value)[0]


def active_sizer() -> CanonicalSizer:
    """The projection-scoped sizer, or a fresh one outside a projection."""

    return _ACTIVE.get() or CanonicalSizer()


def json_bytes(value: Any) -> int:
    return active_sizer().size(value)


def estimated_tokens_from_bytes(byte_count: int) -> int:
    # Dense JSON is often less favorable than natural-language chars/4.
    return max(1, (byte_count + 2) // 3)


def with_projection_sizer(method: _Method) -> _Method:
    """Bind ``self.canonical_sizer`` for one projection and prune stale memo entries."""

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        sizer = self.canonical_sizer
        sizer.rotate()
        token = _ACTIVE.set(sizer)
        try:
            return method(self, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    return wrapper  # type: ignore[return-value]


def measured_pass_record(
    name: str, before: tuple[int, set[str]], after: tuple[int, set[str]],
    **fields: Any,
) -> dict[str, Any]:
    before_bytes, before_ids = before
    after_bytes, after_ids = after
    return {
        "name": name,
        "pre_bytes": before_bytes,
        "post_bytes": after_bytes,
        "pre_estimated_tokens": estimated_tokens_from_bytes(before_bytes),
        "post_estimated_tokens": estimated_tokens_from_bytes(after_bytes),
        "retained_event_ids": fields.pop("retained_event_ids", sorted(after_ids)),
        "externalized_event_ids": fields.pop("externalized_event_ids", []),
        "replaced_event_ids": fields.pop(
            "replaced_event_ids", sorted(before_ids - after_ids)
        ),
        "artifact_refs": fields.pop("artifact_refs", []),
        **fields,
    }


def pass_record(name: str, before: Any, after: Any, **fields: Any) -> dict[str, Any]:
    sizer = active_sizer()
    return measured_pass_record(
        name, sizer.measure(before), sizer.measure(after), **fields,
    )
