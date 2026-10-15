"""Content-addressed parse cache for tracked YAML declarations read on hot paths.

One recorded ``run.py status`` pass parsed one host profile 1244 times with
the pure-Python loader (198 s of 287 s under cProfile), because
every ``definition()`` call revalidated each lane and card through the host
profile. Every call still reads the file, but a parse is reused while the
bytes are unchanged: the cache key is the SHA-256 of those bytes (1244 reads
of the 22 KB profile cost about 10 ms). A key on path, mtime, size and inode
served stale values here: the kernel stamps mtime from a coarse clock, so a
same-size rewrite within one tick kept its key (477 of 500 immediate rewrites
read the old value in review, 2026-09-30).

The cache holds a frozen value: mappings as ``MappingProxyType``, sequences
(including the pair tuples of ``!!omap``/``!!pairs``) as tuples, sets as
frozensets. See ``load_frozen`` and ``load`` for who may use which.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml


class RecursiveDocumentError(ValueError):
    """A YAML alias refers to a node that contains it; such a value cannot be frozen."""


def select_safe_loader(module: Any = yaml) -> type:
    """libyaml's ``CSafeLoader`` when PyYAML was built with it, else ``SafeLoader``."""

    return getattr(module, "CSafeLoader", None) or module.SafeLoader


SAFE_LOADER = select_safe_loader()

#: Distinct document contents retained. A status pass reads three
#: declarations; the bound only keeps superseded contents from accumulating.
CACHE_ENTRIES = 128

_LOCK = threading.Lock()
_PARSES: OrderedDict[str, Any] = OrderedDict()


def safe_load_text(text: str) -> Any:
    """Parse one YAML document with the safe loader this interpreter provides."""

    return yaml.load(text, Loader=SAFE_LOADER)


def _freeze(value: Any, active: frozenset[int] = frozenset()) -> Any:
    if not isinstance(value, (dict, list, tuple, set)):
        return value
    if id(value) in active:
        raise RecursiveDocumentError("YAML document contains a recursive alias")
    if isinstance(value, set):
        return frozenset(value)
    inner = active | {id(value)}
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item, inner) for key, item in value.items()})
    return tuple(_freeze(item, inner) for item in value)


def thaw(value: Any) -> Any:
    """A caller-owned copy of a frozen parse: every tuple becomes a list."""

    if isinstance(value, MappingProxyType):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    if isinstance(value, frozenset):
        return set(value)
    return value


def _parsed(data: bytes) -> Any:
    digest = hashlib.sha256(data).hexdigest()
    with _LOCK:
        if digest in _PARSES:
            _PARSES.move_to_end(digest)
            return _PARSES[digest]
    value = _freeze(safe_load_text(data.decode("utf-8")))
    with _LOCK:
        _PARSES[digest] = value
        _PARSES.move_to_end(digest)
        while len(_PARSES) > CACHE_ENTRIES:
            _PARSES.popitem(last=False)
    return value


def load_frozen(path: Path | str) -> Any:
    """The shared frozen parse of ``path``'s current bytes, for read-only internal use.

    Only a caller that reads the value and copies out what it keeps may use
    this. A ``MappingProxyType`` cannot be passed to ``json.dumps``,
    ``copy.deepcopy`` or ``yaml.safe_dump``; call ``load`` (or ``thaw``) for a
    value that leaves the caller. Raises ``OSError`` for an unreadable file,
    ``UnicodeDecodeError`` for non-UTF-8 bytes, ``yaml.YAMLError`` for
    malformed YAML and ``RecursiveDocumentError`` for a recursive alias.
    """

    return _parsed(Path(path).read_bytes())


def load(path: Path | str) -> Any:
    """A caller-owned ``dict``/``list``/``set`` copy of the parse of ``path``.

    ``!!omap``/``!!pairs`` entries come back as two-item lists, not tuples.
    """

    return thaw(load_frozen(path))


def clear() -> None:
    """Drop every cached parse."""

    with _LOCK:
        _PARSES.clear()


__all__ = [
    "RecursiveDocumentError", "SAFE_LOADER", "clear", "load", "load_frozen",
    "safe_load_text", "select_safe_loader", "thaw",
]
