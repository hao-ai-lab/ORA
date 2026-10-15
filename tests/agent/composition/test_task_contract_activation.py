"""Which task-contract registry is live is stated once, and cannot drift.

Both maintained backends ship three registry generations. Only one of each is
bound, and before this the answer lived in two different places: RoboTwin's in
`tests/data/default-expectations.json`, RoboDojo's in a Python constant. A reader had to
know which backend kept its answer where, and nothing compared the two.

These checks keep the activation file the single statement of what is bound,
keep the Python constant equal to it, and keep the retired generations from
being silently reachable again or silently deleted while immutable integration
inventories still cite them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ACTIVATION = json.loads((ROOT / "tests/data/default-expectations.json").read_text(encoding="utf-8"))
BOUND = ACTIVATION["task_contracts"]
FROZEN = ACTIVATION["frozen_task_contracts"]


def _registry(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@pytest.mark.parametrize("backend", sorted(BOUND))
def test_the_bound_registry_exists_and_holds_the_declared_task_count(backend: str) -> None:
    """A count that drifts from the file is the kind of claim this file exists to stop."""

    entry = BOUND[backend]
    path = ROOT / entry["registry"]
    assert path.is_file(), f"{backend} binds a registry that is not in the tree"
    assert len(_registry(entry["registry"])["tasks"]) == entry["task_count"]


def test_the_robodojo_python_constant_equals_the_activation_entry() -> None:
    """Two statements of one fact must agree, or one of them is decoration."""

    from ora.agent.composition.robodojo_task_contract import REGISTRY_PATH

    assert REGISTRY_PATH == BOUND["robodojo"]["registry"]


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_each_retired_registry_is_present_reasoned_and_unbound(name: str) -> None:
    """Retired is a recorded state, not a deletion and not a second live answer."""

    entry = FROZEN[name]
    path = ROOT / entry["path"]
    assert path.is_file(), (
        f"{name} is recorded as retained but is not in the tree; integration "
        "inventories still cite this path")
    assert len(_registry(entry["path"])["tasks"]) == entry["task_count"]
    for field in ("reason", "retained_because", "decision"):
        assert entry.get(field), f"{name} is retired without stating {field}"
    assert (ROOT / entry["decision"]).is_file(), f"{name} cites a missing decision"
    assert entry["path"] not in {b["registry"] for b in BOUND.values()}, (
        f"{name} is both retired and bound")


def test_every_registry_in_the_tree_is_either_bound_or_retired() -> None:
    """A fourth generation appearing unrecorded is the failure mode here."""

    found = {
        p.relative_to(ROOT).as_posix()
        for p in ROOT.glob("ora/policy/environments/*/task-contracts/registry-*.json")
    }
    accounted = {b["registry"] for b in BOUND.values()} | {f["path"] for f in FROZEN.values()}
    unrecorded = sorted(found - accounted)
    assert not unrecorded, (
        f"these contract registries are neither bound nor retired: {unrecorded}. "
        "Bind one, or record why it is kept.")


@pytest.mark.parametrize("backend", sorted(BOUND))
def test_a_bound_registry_resolves_every_contract_text_it_names(backend: str) -> None:
    """A registry that names a missing text withholds delivery without saying so."""

    registry = _registry(BOUND[backend]["registry"])
    missing = []
    for task, entry in registry["tasks"].items():
        # RoboTwin keys the text as `path`; RoboDojo keys it as `layer`, and its
        # registry generation is independent of the text version it binds.
        relative = entry.get("path") or entry.get("layer")
        assert relative, f"{backend}/{task} names no contract text"
        if not (ROOT / relative).is_file():
            missing.append(f"{task} -> {relative}")
    assert not missing, f"{backend} binds unresolvable contract texts: {missing}"
