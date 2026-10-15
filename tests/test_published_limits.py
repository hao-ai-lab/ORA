"""Every enforced number in policy-visible tool help equals the code that enforces it.

Integration 0059 unit N1, phase 1 (2026-10-04). One table,
`ora/policy/tool_surfaces/published_limits_v1/limits.json`, maps each
limit to its owner -- the constant, profile value or source literal the handler
or backend enforces -- and lists every sentence that states it, per surface.
The test composes each surface, finds each sentence in the named tool's
description or input-schema text, parses the stated number and compares it
with the owner after unit conversion.

Surfaces: the frozen v1 RoboTwin default that direct_v3 bound, which keeps its
wording for reproducibility, and the current defaults (direct_v4, integration
0059 unit A0), registered surfaces whose manifests end with the two
published-limits layers. `*-review.json` names each
current surface's base profile, its id and its layer files; the layers are the
review's edits applied to that surface's own text before them.

A sentence that disagrees with its owner on a frozen surface is a strict
expected failure with its reason; the current texts carry the correction and
may not have any. A census then requires every digit in every composed
tool text to belong to a registered statement or a reviewed informational entry,
so a new number cannot enter the help without an owner.
"""

from __future__ import annotations

import fnmatch
import functools
import importlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

from ora.agent.composition import description_edits, tool_surface
from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.tool_surface import resolve_tool_surface

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "ora/policy/tool_surfaces/published_limits_v1"
TABLE = json.loads((HERE / "limits.json").read_text(encoding="utf-8"))
SURFACES: dict[str, dict[str, Any]] = TABLE["surfaces"]
FROZEN_KEYS = tuple(key for key, spec in SURFACES.items() if spec["generation"] == "frozen")
CURRENT_KEYS = tuple(key for key, spec in SURFACES.items() if spec["generation"] == "current")
#: Experiment candidates that bind generated layers without being a default (EXP12B/libero-skills).
#: Their texts are checked exactly as the current ones; only the default-instance rules exclude them.
CANDIDATE_KEYS = tuple(key for key, spec in SURFACES.items() if spec["generation"] == "candidate")
#: Former defaults that keep their generated layers and stay selectable by an explicit
#: tool_surface; checked like candidates.
RETAINED_KEYS = tuple(key for key, spec in SURFACES.items() if spec["generation"] == "retained")
GENERATED_KEYS = CURRENT_KEYS + CANDIDATE_KEYS + RETAINED_KEYS

WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "sixteen": 16, "twenty": 20, "pi": math.pi,
}
#: Every unit a statement or owner may use, as a factor to one base unit.
UNITS = {
    "m": ("length", 1.0), "cm": ("length", 1e-2), "mm": ("length", 1e-3),
    "rad": ("angle", 1.0), "deg": ("angle", math.pi / 180.0),
    "s": ("time", 1.0), "ms": ("time", 1e-3), "Hz": ("frequency", 1.0),
}
#: A digit run that is not part of an identifier (T0, link6, x0, sha256, v3).
DIGITS = re.compile(r"(?<![A-Za-z_\d.])\d+(?:\.\d+)?(?:e-?\d+)?")


# --------------------------------------------------------------- composition

def current_surfaces() -> tuple[str, ...]:
    """The surfaces a fresh or live consumer binds: every default instance and the defaults they replaced.

    The replaced defaults are read from ci/backend-parity.json previous_defaults;
    one that is no longer registered cannot be bound and is not listed.
    """

    activation = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
    found = {instance["tool_surface"] for instance in activation["instances"].values()}
    parity = json.loads((ROOT / "ci/backend-parity.json").read_text(encoding="utf-8"))
    found |= {record["profile_id"] for record in parity["previous_defaults"].values()
              if record["profile_id"] in PROFILE_CATALOG}
    return tuple(sorted(found))


def _review(key: str) -> dict[str, Any]:
    return description_edits.load_review(HERE / SURFACES[key]["review"])


def _presupposed(key: str) -> list[str]:
    """Capabilities a current text is written for (each must be declared; see below)."""

    return [] if SURFACES[key]["generation"] == "frozen" else _review(key)["presupposes_capabilities"]


@functools.cache
def _surface(key: str) -> Any:
    return resolve_tool_surface(SURFACES[key]["profile"])


def _walk(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(node, dict):
        for name, value in node.items():
            if name == "description" and isinstance(value, str):
                yield path + "/description", value
            else:
                yield from _walk(value, f"{path}/{name}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}/{index}")


@functools.cache
def _fields(key: str) -> dict[str, tuple[tuple[str, str], ...]]:
    """Every policy-visible text of every tool: its description, then its schema texts."""

    return {
        item["name"]: (("description", item["description"]),
                       *_walk(item.get("inputSchema", {}), "inputSchema"))
        for item in _surface(key).definitions
    }


def _keys(patterns: list[str]) -> list[str]:
    found = [key for key in SURFACES if any(fnmatch.fnmatchcase(key, p) for p in patterns)]
    assert found, f"no table surface matches {patterns}"
    return found


# --------------------------------------------------------------- owners

class Unverifiable(Exception):
    """The table records a number with no enforcing owner in this repository."""


def _python(reference: str, path: list[Any]) -> Any:
    module, name = reference.split(":")
    value: Any = getattr(importlib.import_module(module), name)
    for step in path:
        if step == "()":
            value = value()
        elif isinstance(value, (dict, list, tuple)):
            value = value[step]
        else:
            value = getattr(value, step)
    return value


def _pointer(document: Any, pointer: str) -> Any:
    for part in pointer.split("/")[1:]:
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def _convert(value: float, unit: str, into: str) -> float:
    if unit == into:
        return float(value)
    (dimension, scale), (target, target_scale) = UNITS[unit], UNITS[into]
    assert dimension == target, f"cannot convert {unit} into {into}"
    return float(value) * scale / target_scale


def _types(node: dict[str, Any]) -> list[Any]:
    declared = node.get("type")
    return declared if isinstance(declared, list) else [declared]


def owner_value(owner: dict[str, Any], key: str) -> float:
    """The enforcing value of one owner, in the owner's own unit."""

    if "unenforced" in owner or "external" in owner:
        raise Unverifiable(owner.get("unenforced") or owner["external"])
    if "python" in owner:
        value = _python(owner["python"], owner.get("path", []))
    elif "json" in owner:
        value = _pointer(json.loads((ROOT / owner["json"]).read_text(encoding="utf-8")), owner["pointer"])
    elif "source" in owner:
        found = re.findall(owner["pattern"], (ROOT / owner["source"]).read_text(encoding="utf-8"))
        assert len(found) == 1, f"{owner['source']}: {owner['pattern']!r} matched {len(found)} times"
        value = float(found[0])
    elif "count_excluding" in owner:
        spec = owner["count_excluding"]
        items = list(_python(spec["items"], []))
        if "after" in spec:
            items = items[items.index(spec["after"]) + 1:]
        value = sum(1 for item in items if item not in set(_python(spec["exclude"], [])))
    elif "schema_keyword" in owner:
        spec = owner["schema_keyword"]
        tool = next(item for item in _surface(key).definitions if item["name"] == spec["tool"])
        value = _pointer(tool["inputSchema"], spec["pointer"])
    elif "schema_array_depth" in owner:
        spec = owner["schema_array_depth"]
        tool = next(item for item in _surface(key).definitions if item["name"] == spec["tool"])
        node, value = _pointer(tool["inputSchema"], spec["pointer"]), 0
        while isinstance(node, dict) and "array" in _types(node):
            value, node = value + 1, node.get("items")
    elif "derive" in owner:
        # Each part in its base unit (a count stays a count); the result is read
        # in the owner's unit.
        values = [owner_value(part, key) * UNITS.get(part["unit"], ("", 1.0))[1]
                  for part in owner["of"]]
        operation = owner["derive"]
        if operation == "difference":
            value = values[0] - sum(values[1:])
        elif operation == "product":
            value = math.prod(values)
        elif operation == "reciprocal":
            value = 1.0 / values[0]
        elif operation == "negative":
            value = -values[0]
        else:
            raise AssertionError(f"unknown derivation {operation!r}")
        value /= UNITS.get(owner["unit"], ("", 1.0))[1]
    else:
        raise AssertionError(f"owner has no recognised kind: {sorted(owner)}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError(f"owner value is not a finite number: {value!r}")
    return float(value)


def stated_value(literal: str) -> float:
    sign, word = (-1.0, literal[1:].lower()) if literal.startswith("-") else (1.0, literal.lower())
    return sign * WORDS[word] if word in WORDS else float(literal.replace(",", ""))


def _split_dimensions(text: str) -> str:
    """Read a card size such as 1024x576 as two numbers: that x is not part of an identifier."""

    return re.sub(r"(?<=\d)x(?=\d)", " ", text)


def _literal_offset(statement: dict[str, Any]) -> int:
    """Where the stated number sits in its sentence: the n-th whole-token occurrence."""

    pattern = rf"(?<![A-Za-z_\d.]){re.escape(statement['literal'])}(?![A-Za-z_\d])"
    found = [match.start() for match in re.finditer(pattern, _split_dimensions(statement["text"]))]
    index = statement.get("literal_index", 0)
    assert index < len(found), f"{statement['literal']!r} is not a number in {statement['text']!r}"
    return found[index]


# --------------------------------------------------------------- cases

def _statement_cases() -> Iterator[Any]:
    for limit in TABLE["limits"]:
        for index, statement in enumerate(limit["statements"]):
            expected = statement.get("expected_failure", {})
            for key in _keys(statement["surfaces"]):
                marks = ([pytest.mark.xfail(reason=expected["reason"], strict=True)]
                         if key in expected.get("surfaces", []) else [])
                yield pytest.param(key, limit, statement, marks=marks,
                                   id=f"{limit['key']}-{index}-{key}")


@pytest.mark.parametrize("key,limit,statement", list(_statement_cases()))
def test_the_stated_number_is_the_enforced_number(key, limit, statement):
    fields = _fields(key).get(statement["tool"])
    assert fields is not None, f"{key} declares no {statement['tool']}"
    texts = [text for _path, text in fields if statement["text"] in text]
    assert texts, f"{key} {statement['tool']} no longer says: {statement['text']!r}"
    assert _literal_offset(statement) >= 0
    capabilities = set(_surface(key).capabilities) | set(_presupposed(key))
    if "requires_capability" in limit:
        assert limit["requires_capability"] in capabilities, f"{key} lacks {limit['requires_capability']}"
    if "absent_capability" in limit:
        assert limit["absent_capability"] not in capabilities, f"{key} declares {limit['absent_capability']}"
    unit = statement.get("unit", limit["unit"])
    stated = stated_value(statement["literal"])
    actual = owner_value(limit["owner"], key)
    if unit in UNITS and limit["unit"] in UNITS:
        actual = _convert(actual, limit["unit"], unit)
    else:
        assert unit == limit["unit"], f"{limit['key']}: {unit} against {limit['unit']}"
    if statement.get("rounded"):
        decimals = len(statement["literal"].partition(".")[2])
        tolerance = 0.5 * 10 ** -decimals + 1e-12
    else:
        tolerance = 1e-9 * max(1.0, abs(actual))
    assert abs(stated - actual) <= tolerance, (
        f"{key} {statement['tool']}: the help says {statement['literal']} {unit}; "
        f"{limit['key']} enforces {actual:g} {unit}"
    )


def _schema_cases() -> Iterator[Any]:
    for limit in TABLE["limits"]:
        for index, check in enumerate(limit.get("schema", [])):
            for key in _keys(check["surfaces"]):
                yield pytest.param(key, limit, check, id=f"{limit['key']}-schema{index}-{key}")


@pytest.mark.parametrize("key,limit,check", list(_schema_cases()))
def test_the_schema_keyword_is_the_enforced_number(key, limit, check):
    tool = next(item for item in _surface(key).definitions if item["name"] == check["tool"])
    declared = _pointer(tool["inputSchema"], check["pointer"])
    actual = owner_value(limit["owner"], key)
    unit = check.get("unit", limit["unit"])
    if unit in UNITS and limit["unit"] in UNITS:
        actual = _convert(actual, limit["unit"], unit)
    assert math.isclose(float(declared) * check.get("sign", 1), actual, rel_tol=1e-9), (
        f"{key} {check['tool']}{check['pointer']} = {declared}; {limit['key']} enforces {actual:g}"
    )


# --------------------------------------------------------------- census

def _covered(key: str, tool: str, text: str) -> list[tuple[int, int]]:
    spans = []
    for limit in TABLE["limits"]:
        for statement in limit["statements"]:
            if statement["tool"] != tool or key not in _keys(statement["surfaces"]):
                continue
            start = text.find(statement["text"])
            while start >= 0:
                offset = start + _literal_offset(statement)
                spans.append((offset, offset + len(statement["literal"])))
                start = text.find(statement["text"], start + 1)
    for entry in TABLE["informational"]:
        if tool not in entry["tools"] or key not in _keys(entry["surfaces"]):
            continue
        start = text.find(entry["text"])
        while start >= 0:
            spans.append((start, start + len(entry["text"])))
            start = text.find(entry["text"], start + 1)
    return spans


@pytest.mark.parametrize("key", list(SURFACES))
def test_every_number_in_the_help_is_registered(key):
    unregistered = []
    for tool, fields in _fields(key).items():
        for path, text in fields:
            spans = _covered(key, tool, text)
            for match in DIGITS.finditer(_split_dimensions(text)):
                if not any(start <= match.start() and match.end() <= end for start, end in spans):
                    context = text[max(0, match.start() - 50):match.end() + 30]
                    unregistered.append(f"{tool} {path}: {match.group(0)} in ...{context}...")
    assert not unregistered, "numbers without a registered owner:\n" + "\n".join(unregistered)


# --------------------------------------------------------------- table rules

def test_the_table_covers_every_current_surface():
    profiles = {spec["profile"]: key for key, spec in SURFACES.items()}
    for profile in current_surfaces():
        assert profile in profiles or profile in TABLE["not_covered"], profile
    assert all(reason.strip() for reason in TABLE["not_covered"].values())
    backends = {"robotwin", "libero_pro", "robodojo"}
    assert {SURFACES[key]["backend"] for key in CURRENT_KEYS} == backends
    assert {SURFACES[key]["backend"] for key in FROZEN_KEYS} <= backends
    frozen = {SURFACES[key]["profile"] for key in FROZEN_KEYS}
    # Each current surface descends, through generated surfaces it succeeded, from a frozen
    # one or from a predecessor that is no longer registered.
    generated = {SURFACES[key]["profile"]: key for key in GENERATED_KEYS}
    roots = set()
    for key in CURRENT_KEYS:
        base = _review(key)["base_profile"]
        while base in generated:
            base = _review(generated[base])["base_profile"]
        roots.add(base)
    assert {root for root in roots if root in PROFILE_CATALOG} == frozen
    for key in CURRENT_KEYS:
        assert _review(key)["successor_profile"] == SURFACES[key]["profile"], key
    # Every default instance is a current surface.
    activation = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
    assert {instance["tool_surface"] for instance in activation["instances"].values()} == {
        SURFACES[key]["profile"] for key in CURRENT_KEYS}


def test_expected_failures_are_only_on_current_surfaces_and_carry_a_reason():
    expected = 0
    for limit in TABLE["limits"]:
        for statement in limit["statements"]:
            failure = statement.get("expected_failure")
            if failure is None:
                continue
            expected += 1
            assert failure["reason"].strip()
            assert set(failure["surfaces"]) <= set(FROZEN_KEYS), statement["text"]
            assert set(failure["surfaces"]) <= set(_keys(statement["surfaces"]))
    assert expected, "phase 1 records the known mismatches as expected failures"


def test_an_owner_without_enforcement_is_never_stated_on_a_successor():
    for limit in TABLE["limits"]:
        owner = limit["owner"]
        if "unenforced" not in owner and "external" not in owner:
            continue
        for statement in limit["statements"]:
            assert not set(_keys(statement["surfaces"])) & set(GENERATED_KEYS), limit["key"]
            assert set(statement["expected_failure"]["surfaces"]) == set(_keys(statement["surfaces"]))


def test_keys_are_unique_and_every_limit_is_stated():
    keys = [limit["key"] for limit in TABLE["limits"]]
    assert len(keys) == len(set(keys))
    for limit in TABLE["limits"]:
        assert limit["statements"], limit["key"]
        assert limit["kind"] in {"enforced", "parameter"}, limit["key"]
        assert limit["enforced_by"].strip(), limit["key"]
    for entry in TABLE["informational"]:
        assert entry["reason"].strip() and entry["text"].strip()


# --------------------------------------------------------------- current texts

@pytest.mark.parametrize("key", GENERATED_KEYS)
def test_the_successor_layers_are_the_review_edits_applied_to_the_base_text(key):
    review = _review(key)
    for name, value in description_edits.generated_layers(review).items():
        assert (HERE / name).read_text(encoding="utf-8") == description_edits.render(value), name


@pytest.mark.parametrize("key", GENERATED_KEYS)
def test_every_edit_ships_a_chinese_reference(key):
    review = _review(key)
    assert "English text is authoritative" in review["public_text_zh_reference"]["note"]
    for edit in review["edits"]:
        assert edit["why"].strip(), edit["after"][:60]
        assert re.search(r"[一-鿿]", edit["zh"]), edit["after"][:60]


@pytest.mark.parametrize("key", GENERATED_KEYS)
def test_every_successor_tool_states_what_it_refuses(key):
    """N2: every tool's help names its ex-ante refusals, as rules and nothing more.

    A tool whose text a binder writes after every layer (``refusals_owned_elsewhere``)
    states them too: its binder writes the sentence.
    """

    owned_elsewhere = _review(key)["refusals_owned_elsewhere"]
    assert all(reason.strip() for reason in owned_elsewhere.values())
    declared = {item["name"] for item in _surface(key).definitions}
    assert set(owned_elsewhere) <= declared, key
    for item in _surface(key).definitions:
        on_error = item["description"].split("On error:", 1)[1]
        assert "Refused:" in on_error, f"{key} {item['name']}"


@pytest.mark.parametrize("key", GENERATED_KEYS)
def test_a_current_surface_declares_what_its_text_presupposes(key):
    assert set(_presupposed(key)) <= set(_surface(key).capabilities), key


@pytest.mark.parametrize("key", GENERATED_KEYS)
def test_the_published_limits_layers_come_last_on_the_current_surfaces(key):
    """Unit A0: the layers replace whole descriptions, so nothing may follow them."""

    review = _review(key)
    path = PROFILE_CATALOG[review["successor_profile"]]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert review["edits_apply_to"] == description_edits.APPLY_TO_SUCCESSOR
    for order, kind in (("description_order", "description"), ("schema_order", "schemas")):
        slot = review["slots"][kind]
        assert manifest[order][-1] == slot, (key, order)
        layer = HERE / review["layers"][kind]
        # Bound by path; policy.lock carries the layer's integrity.
        assert (path.parent / manifest["components"][slot]).resolve() == layer.resolve()


def test_only_the_current_surfaces_bind_the_published_limits_layers():
    current = {SURFACES[key]["profile"] for key in GENERATED_KEYS}
    for profile, manifest in sorted(PROFILE_CATALOG.items()):
        if profile in current:
            continue
        assert "published_limits_v1" not in manifest.read_text(encoding="utf-8"), manifest
