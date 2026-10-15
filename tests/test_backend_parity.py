"""Cross-backend capability parity of the default ToolSurfaces (integration 0059, unit A0).

`tests/data/default-expectations.json` checks each backend's default against its own
snapshot, so dropping a capability from one backend used to be a one-line edit
of that backend's expected list (parity audit 2026-10-04, root cause R2), and
features landed on whichever surface an experiment ran (R3, R4). This guard
compares the backends with each other:

1. every capability and tool a maintained default declares has a row in
   `ci/backend-parity.json`, so a new name forces its author to classify it;
2. a ``generic`` or ``generic_needs_backend_code`` row is declared by every
   default (or through a successor), or an exception names the backend, gives
   its reason and, for missing backend code, names the hook;
3. an exception expires: it fails as stale once the backend declares the name,
   and a ``backend_missing`` exception fails as soon as its hook exists, so the
   author of the hook must also declare the capability;
4. a ``backend_specific`` row is declared only on the backends it applies to,
   and a generic row scoped by ``required_with_tools`` is owed only by a default
   declaring one of its sender tools;
5. a default swap keeps a superset of the default it replaced, minus listed
   retirements, and the swap updates ``previous_defaults`` in the same change;
6. a ``label_only`` row has no reader under ``ora`` (no string
   constant equal to the name in any module), so a new reader forces a
   reclassification.

The maintained defaults are derived from the expected defaults, never copied.
"""
from __future__ import annotations

import ast
import json
import re
from functools import cache
from pathlib import Path

import pytest

from ora.agent.composition.profile_catalog import PROFILE_CATALOG
from ora.agent.composition.tool_surface import resolve_tool_surface

ROOT = Path(__file__).resolve().parents[1]
PARITY = json.loads((ROOT / "ci/backend-parity.json").read_text(encoding="utf-8"))
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
CLASSES = {"generic", "generic_needs_backend_code", "backend_specific"}
KINDS = {"retired", "deferred", "backend_missing", "version_split"}
#: P<four digits>-<letter><two digits>: one id per recorded exception.
EXCEPTION_ID = re.compile(r"^P\d{4}-[A-Z]\d{2}$")


def defaults() -> dict[str, str]:
    """backend -> the ToolSurface its fresh consumers inherit."""

    return {backend: instance["tool_surface"] for backend, instance in ACTIVATION["instances"].items()}


@cache
def declared(profile: str) -> dict[str, frozenset[str]]:
    surface = resolve_tool_surface(profile)
    return {"capabilities": frozenset(surface.capabilities),
            "tools": frozenset(item["name"] for item in surface.definitions)}


def family(name: str) -> set[str]:
    """``name`` and every successor it is superseded by, transitively."""

    found, current = {name}, name
    while current in PARITY["supersedes"]:
        current = PARITY["supersedes"][current]
        found.add(current)
    return found


def predecessors(name: str) -> set[str]:
    return {old for old in PARITY["supersedes"] if name in family(old) and old != name}


def covers(backend: str, kind: str, name: str) -> bool:
    """The backend's default declares the name or one of its successors."""

    return bool(family(name) & declared(defaults()[backend])[kind])


def exceptions_for(kind: str, name: str) -> list[dict]:
    return [entry for entry in PARITY["exceptions"] if entry["kind_of_name"] == kind and entry["name"] == name]


# ------------------------------------------------------------------ hooks

def _module_tree(path: Path) -> ast.Module | None:
    return ast.parse(path.read_text(encoding="utf-8")) if path.is_file() else None


def hook_exists(hook: dict) -> bool:
    """Whether the backend code an exception waits for exists, read statically.

    Three forms, all host-safe (no backend module is imported):
    ``{"path", "defines"}`` a function or method of that name in the file;
    ``{"path", "constant", "contains"}`` a literal tuple/list/set constant holding the value;
    ``{"glob", "mentions"}`` any module under the glob names the identifier.
    """

    if "defines" in hook:
        tree = _module_tree(ROOT / hook["path"])
        return tree is not None and any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == hook["defines"]
            for node in ast.walk(tree))
    if "constant" in hook:
        tree = _module_tree(ROOT / hook["path"])
        for node in ast.walk(tree) if tree is not None else ():
            targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
            if any(isinstance(t, ast.Name) and t.id == hook["constant"] for t in targets):
                try:
                    return hook["contains"] in ast.literal_eval(node.value)
                except ValueError:
                    return False
        return False
    if "mentions" in hook:
        pattern = re.compile(rf"\b{re.escape(hook['mentions'])}\b")
        return any(pattern.search(path.read_text(encoding="utf-8")) for path in ROOT.glob(hook["glob"]))
    raise AssertionError(f"unknown hook form: {sorted(hook)}")


# ------------------------------------------------------------------ table rules

def test_the_maintained_defaults_are_every_backend_of_the_activation_contract():
    assert set(defaults()) == set(ACTIVATION["backends"]) == {"libero_pro", "robotwin", "robodojo"}
    assert PARITY["schema"] == "ora.backend-parity.v1"


@pytest.mark.parametrize("kind", ["capabilities", "tools"])
def test_every_declared_name_is_classified(kind):
    rows = PARITY[kind]
    unclassified = sorted({name for profile in defaults().values() for name in declared(profile)[kind]} - set(rows))
    assert not unclassified, f"classify these {kind} in ci/backend-parity.json: {unclassified}"
    for name, row in rows.items():
        assert row["class"] in CLASSES, name
        assert row["evidence"].strip(), name
        if "required_with_tools" in row:
            assert row["class"] != "backend_specific" and row["required_with_tools"]["tools"], name
            assert row["required_with_tools"]["reason"].strip(), name
        if row["class"] == "backend_specific":
            assert set(row["applies_to"]) <= set(defaults()) and row["applies_to"], name
            assert row["reason"].strip(), name


@pytest.mark.parametrize("kind", ["capabilities", "tools"])
def test_rows_name_only_what_some_default_or_exception_still_needs(kind):
    """A row for a name no default declares is stale unless an exception keeps it open."""

    on_defaults = {name for profile in defaults().values() for name in declared(profile)[kind]}
    excepted = {entry["name"] for entry in PARITY["exceptions"] if entry["kind_of_name"] == kind}
    stale = sorted(name for name in PARITY[kind]
                   if not family(name) & on_defaults and name not in excepted)
    assert not stale, f"rows for names no default declares: {stale}"


@pytest.mark.parametrize("kind", ["capabilities", "tools"])
def test_a_generic_name_is_on_every_backend_or_excepted_with_a_decision(kind):
    missing = []
    for name, row in PARITY[kind].items():
        # A superseded name's requirement is carried by its successor's row.
        if row["class"] == "backend_specific" or name in PARITY["supersedes"]:
            continue
        scope = row.get("required_with_tools")
        for backend in defaults():
            if covers(backend, kind, name):
                continue
            # A capability whose only senders are certain tools is owed only where one is declared.
            if scope and not set(scope["tools"]) & declared(defaults()[backend])["tools"]:
                continue
            if not any(backend in entry["backends"] for entry in exceptions_for(kind, name)):
                missing.append(f"{name} on {backend}")
    assert not missing, (
        "generic names a default lacks with no recorded exception (declare them, or record an "
        f"exception with its decision): {missing}")


def test_every_exception_carries_a_reason_and_approval_date():
    ids = [entry["id"] for entry in PARITY["exceptions"]]
    assert len(ids) == len(set(ids))
    for entry in PARITY["exceptions"]:
        assert EXCEPTION_ID.match(entry["id"]), entry["id"]
        assert entry["kind"] in KINDS, entry["id"]
        assert entry["kind_of_name"] in {"capabilities", "tools"}, entry["id"]
        assert entry["name"] in PARITY[entry["kind_of_name"]], entry["id"]
        assert set(entry["backends"]) <= set(defaults()) and entry["backends"], entry["id"]
        assert entry["reason"].strip() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry["approved"]), entry["id"]
        if entry["kind"] == "backend_missing":
            assert entry["missing_hook"] and entry["follow_up"].strip(), entry["id"]


def test_an_exception_expires_once_the_backend_declares_the_name():
    stale = [f"{entry['id']} {entry['name']} on {backend}"
             for entry in PARITY["exceptions"] for backend in entry["backends"]
             if covers(backend, entry["kind_of_name"], entry["name"])]
    assert not stale, f"these exceptions are stale; remove them: {stale}"


def test_a_backend_missing_exception_expires_once_its_hook_exists():
    """The author of the missing backend code must also declare the capability."""

    ready = [f"{entry['id']} ({entry['name']}, {', '.join(entry['backends'])})"
             for entry in PARITY["exceptions"]
             if entry["kind"] == "backend_missing" and hook_exists(entry["missing_hook"])]
    assert not ready, (
        "the backend code these exceptions wait for now exists: declare the capability on the "
        f"backend's default surface and remove the exception: {ready}")


@pytest.mark.parametrize("kind", ["capabilities", "tools"])
def test_a_backend_specific_name_is_declared_only_where_it_applies(kind):
    misplaced = []
    for backend, profile in defaults().items():
        for name in declared(profile)[kind]:
            row = PARITY[kind].get(name)
            if row is not None and row["class"] == "backend_specific" and backend not in row["applies_to"]:
                misplaced.append(f"{name} on {backend}")
    assert not misplaced, f"backend_specific names declared outside applies_to (misclassified generic?): {misplaced}"


def test_no_default_declares_a_name_and_its_successor():
    both = [f"{old} and {new} on {backend}" for old, new in PARITY["supersedes"].items()
            for backend, profile in defaults().items() for kind in ("capabilities", "tools")
            if {old, new} <= declared(profile)[kind]]
    assert not both, both


# ------------------------------------------------------------------ default swaps

@pytest.mark.parametrize("backend", ["libero_pro", "robotwin", "robodojo"])
def test_a_default_swap_keeps_a_superset_minus_recorded_retirements(backend):
    record = PARITY["previous_defaults"][backend]
    assert record["replaced_by"] == defaults()[backend], (
        f"the {backend} default changed: record the default it replaced and its retirements in "
        "ci/backend-parity.json previous_defaults in the same change")
    retired = {item["name"]: item for item in record["retirements"]}
    for item in retired.values():
        assert item["reason"].strip(), item["name"]
    # The recorded sets are the old default as it resolves in this tree; once
    # its manifest has left the tree the record is the only copy and stands.
    if record["profile_id"] in PROFILE_CATALOG:
        old = declared(record["profile_id"])
        assert sorted(old["capabilities"]) == record["capabilities"]
        assert sorted(old["tools"]) == record["tools"]
    lost = [f"{kind} {name}" for kind in ("capabilities", "tools") for name in record[kind]
            if not covers(backend, kind, name) and name not in retired]
    assert not lost, f"the {backend} default swap dropped names no retirement records: {lost}"
    unused = sorted(name for name in retired if any(
        covers(backend, kind, name) for kind in ("capabilities", "tools")))
    assert not unused, f"retirements of names the new default still declares: {unused}"


# ------------------------------------------------------------------ label-only rows

@cache
def _string_constants() -> frozenset[str]:
    found: set[str] = set()
    for path in (ROOT / "ora").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 80:
                found.add(node.value)
    return frozenset(found)


@pytest.mark.parametrize("kind", ["capabilities", "tools"])
def test_a_label_only_row_has_no_reader(kind):
    read = sorted(name for name, row in PARITY[kind].items()
                  if row.get("label_only") and name in _string_constants())
    assert not read, f"label_only rows that ora now reads; reclassify them: {read}"
