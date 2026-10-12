"""The primitive tunable table: the only parameters a composed tool may set.

Each entry is one internal parameter of a base primitive (a lift height, a
probe variant policy) with its unit, frame, per-embodiment default and, once
qualified, the range a composed tool may choose from. A composed tool names an
entry with ``tunable("<id>")``; it can never invent a parameter, and it can
expose one to the policy only after the entry carries qualification evidence
and a bound. Until then the default is the only admissible value.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "ora.primitive-tunables.v1"
TUNABLES_PATH = Path(__file__).resolve().parents[3] / "policy" / "primitives" / "tunables" / "tunables-v1.json"
_ID = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_PRIMITIVES = ("gripper", "move", "check")


class TunableError(ValueError):
    """A tunable table entry, or a value chosen for one, is invalid."""


@dataclass(frozen=True)
class Tunable:
    id: str
    primitive: str
    type: str  # number | enum
    unit: str | None
    frame: str | None
    description: str
    defaults: Mapping[str, Any]
    range: tuple[float, float] | None
    enum: tuple[str, ...] | None
    qualification: Mapping[str, Any] | None
    source: str

    @property
    def qualified(self) -> bool:
        """Whether a composed tool may expose this parameter to the policy."""

        bounded = self.range is not None if self.type == "number" else bool(self.enum)
        return self.qualification is not None and bounded

    def default_for(self, embodiment: str | None) -> Any:
        if embodiment is not None and embodiment in self.defaults:
            return self.defaults[embodiment]
        return self.defaults["*"]

    def admit(self, value: Any) -> Any:
        """Return ``value`` if this entry's declared bound admits it."""

        if self.type == "enum":
            if value not in (self.enum or ()):
                raise TunableError(f"{self.id} must be one of {list(self.enum or ())}")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise TunableError(f"{self.id} must be a finite number")
        if self.range is not None and not self.range[0] <= value <= self.range[1]:
            raise TunableError(f"{self.id} must lie in [{self.range[0]}, {self.range[1]}]")
        return float(value)


@dataclass(frozen=True)
class TunableTable:
    entries: Mapping[str, Tunable]
    sha256: str
    path: str

    def __getitem__(self, tunable_id: str) -> Tunable:
        return self.entries[tunable_id]

    def __contains__(self, tunable_id: object) -> bool:
        return tunable_id in self.entries


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise TunableError(message)


def _entry(raw: Any, index: int) -> Tunable:
    where = f"entries[{index}]"
    _expect(isinstance(raw, Mapping), f"{where} must be an object")
    expected = {"id", "primitive", "type", "unit", "frame", "description", "defaults", "range", "qualification", "source"}
    if raw.get("type") == "enum":
        expected = expected | {"enum"}
    _expect(set(raw) == expected, f"{where} must have exactly {sorted(expected)}")
    tunable_id = raw["id"]
    _expect(isinstance(tunable_id, str) and bool(_ID.match(tunable_id)), f"{where}.id must be a dotted snake_case name")
    _expect(raw["primitive"] in _PRIMITIVES, f"{tunable_id}: primitive must be one of {list(_PRIMITIVES)}")
    _expect(raw["type"] in ("number", "enum"), f"{tunable_id}: type must be number or enum")
    for key in ("description", "source"):
        _expect(isinstance(raw[key], str) and raw[key].strip() != "", f"{tunable_id}: {key} must be non-empty text")
    defaults = raw["defaults"]
    _expect(isinstance(defaults, Mapping) and "*" in defaults, f"{tunable_id}: defaults needs a '*' entry")
    bound: tuple[float, float] | None = None
    if raw["range"] is not None:
        low, high = raw["range"]
        _expect(raw["type"] == "number" and low < high, f"{tunable_id}: range must be [low, high] on a number")
        bound = (float(low), float(high))
    enum = None
    if raw["type"] == "enum":
        _expect(isinstance(raw["enum"], list) and raw["enum"] and all(isinstance(v, str) for v in raw["enum"]),
                f"{tunable_id}: enum must list strings")
        enum = tuple(raw["enum"])
    qualification = raw["qualification"]
    _expect(qualification is None or (isinstance(qualification, Mapping) and set(qualification) == {"evidence", "date"}),
            f"{tunable_id}: qualification is null or {{evidence, date}}")
    entry = Tunable(
        id=tunable_id, primitive=raw["primitive"], type=raw["type"], unit=raw["unit"], frame=raw["frame"],
        description=raw["description"], defaults=dict(defaults), range=bound, enum=enum,
        qualification=qualification, source=raw["source"],
    )
    for embodiment, value in defaults.items():
        try:
            entry.admit(value)
        except TunableError as exc:
            raise TunableError(f"{tunable_id}: default for {embodiment!r}: {exc}") from None
    return entry


def parse_tunables(document: Any, *, path: str, sha256: str) -> TunableTable:
    _expect(isinstance(document, Mapping) and document.get("schema") == SCHEMA, f"{path}: schema must be {SCHEMA}")
    _expect(set(document) == {"schema", "entries"}, f"{path}: only schema and entries are allowed")
    entries = [_entry(item, index) for index, item in enumerate(document["entries"])]
    ids = [item.id for item in entries]
    _expect(len(set(ids)) == len(ids), f"{path}: tunable ids must be unique")
    return TunableTable({item.id: item for item in entries}, sha256, path)


def load_tunables(path: Path | None = None) -> TunableTable:
    source = Path(path or TUNABLES_PATH)
    data = source.read_bytes()
    return parse_tunables(json.loads(data), path=str(source), sha256="sha256:" + hashlib.sha256(data).hexdigest())
