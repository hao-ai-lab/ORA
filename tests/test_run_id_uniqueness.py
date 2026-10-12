"""A spawn proves a fresh run-id unused without walking any output tree."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.orchestrator import paths

PLAN = {"experiment": "exp00", "output_workspace": "outputs/exp00"}
RUN_ID = "exp00-m0-deadbeef0000--20260930T000000Z"


@pytest.fixture
def layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    """A shared workspace, one legacy linked-worktree root and one extra root."""
    workspace, linked, extra = tmp_path / "home", tmp_path / "linked", tmp_path / "data1/outputs"
    for directory in (workspace / "outputs", linked / "outputs", extra):
        directory.mkdir(parents=True)
    monkeypatch.setattr(paths, "PROJECT_ROOT", workspace)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", workspace / "outputs")
    monkeypatch.setattr(paths, "STATE_ROOT", workspace / ".harness-runtime")
    monkeypatch.setattr(paths, "RUN_INDEX_ROOT", paths.STATE_ROOT / "run-index")
    monkeypatch.setattr(paths, "LEGACY_RUNS_ROOT", paths.STATE_ROOT / "runs")
    monkeypatch.setattr(paths, "linked_worktree_roots", lambda: (workspace, linked))
    monkeypatch.setenv(paths.EXTRA_OUTPUT_ROOTS_ENV, str(extra))
    return SimpleNamespace(workspace=workspace, linked=linked, extra=extra,
                           planned=paths.new_run_root(PLAN, RUN_ID))


def _run(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(PLAN), encoding="utf-8")
    return root


def _forbid_directory_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    def listing(*_args, **_kwargs):
        raise AssertionError("the run-id check must not enumerate a directory")

    for name in ("scandir", "listdir", "walk"):
        monkeypatch.setattr(os, name, listing)
    for name in ("glob", "rglob", "iterdir"):
        monkeypatch.setattr(Path, name, listing)


def test_a_free_run_id_is_admitted_without_listing_any_directory(monkeypatch, layout):
    # Populated trees in every discovery root: the old check walked all of them.
    for outputs in (layout.workspace / "outputs", layout.linked / "outputs", layout.extra):
        for index in range(3):
            run = _run(outputs / "exp00/runs" / f"exp00-m0-deadbeef0000--2026092{index}T000000Z")
            (run / "artifacts/a/b").mkdir(parents=True)
    # A file where a namespace directory would be is a free path, not an error.
    (layout.extra.parent / "file-root").mkdir()
    (layout.extra.parent / "file-root/exp00").write_text("", encoding="utf-8")
    monkeypatch.setenv(paths.EXTRA_OUTPUT_ROOTS_ENV,
                       os.pathsep.join((str(layout.extra), str(layout.extra.parent / "file-root"))))
    _forbid_directory_listing(monkeypatch)
    assert paths.existing_run_location(PLAN, RUN_ID) is None


@pytest.mark.parametrize("where", ["index", "planned", "linked", "extra", "legacy", "dangling"])
def test_a_taken_run_id_is_found_at_its_exact_location(monkeypatch, layout, where):
    locations = {
        "index": paths.RUN_INDEX_ROOT / f"{RUN_ID}.json",
        "planned": layout.planned,
        "linked": layout.linked / "outputs/exp00/runs" / RUN_ID,
        "extra": layout.extra / "exp00/runs" / RUN_ID,
        "legacy": paths.LEGACY_RUNS_ROOT / RUN_ID,
        "dangling": layout.planned,
    }
    expected = locations[where]
    expected.parent.mkdir(parents=True, exist_ok=True)
    if where == "index":
        expected.write_text("{}", encoding="utf-8")
    elif where == "dangling":
        expected.symlink_to(layout.workspace / "missing-target")
    else:
        expected.mkdir()
    _forbid_directory_listing(monkeypatch)
    found = paths.existing_run_location(PLAN, RUN_ID)
    assert found is not None
    assert (found.parent.resolve(), found.name) == (expected.parent.resolve(), expected.name)


def test_the_namespace_is_the_plan_experiment_not_a_resolved_link_name(monkeypatch, layout):
    # `outputs/exp00` links to a directory of another name, as
    # `outputs/graspgenx-coverage -> .../coverage-output` does on a shared host.
    storage = layout.workspace.parent / "volume/renamed-storage"
    storage.mkdir(parents=True)
    (layout.workspace / "outputs/exp00").symlink_to(storage, target_is_directory=True)
    taken = layout.linked / "outputs/exp00/runs" / RUN_ID
    taken.mkdir(parents=True)
    _forbid_directory_listing(monkeypatch)
    assert paths.new_run_root(PLAN, RUN_ID).parent.parent.name == "renamed-storage"
    assert paths.existing_run_location(PLAN, RUN_ID) == taken


def test_a_location_that_cannot_be_checked_refuses_the_run_id(monkeypatch, layout):
    real_lstat = os.lstat

    def denied(path, *args, **kwargs):
        if Path(path) == layout.extra / "exp00/runs" / RUN_ID:
            raise PermissionError(13, "Permission denied", str(path))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", denied)
    with pytest.raises(PermissionError):
        paths.existing_run_location(PLAN, RUN_ID)


def test_discovery_keeps_its_recursive_reach_and_the_index_covers_other_layouts(layout):
    nested = _run(layout.workspace / "outputs/exp00/groups/g1/runs" / RUN_ID)
    # Reading an existing run still walks every root (semantics unchanged) ...
    assert paths.resolve_run_root(RUN_ID) == nested.resolve()
    # ... while the spawn check reads only exact paths; a registered run-id is
    # refused through the index whatever its layout.
    assert paths.existing_run_location(PLAN, RUN_ID) is None
    index = paths.RUN_INDEX_ROOT / f"{RUN_ID}.json"
    index.parent.mkdir(parents=True)
    index.write_text("{}", encoding="utf-8")
    assert paths.existing_run_location(PLAN, RUN_ID) == index


@pytest.mark.parametrize("plan, run_id, message", [
    (PLAN, "../escape", "run-id must be a stable id"),
    ({"output_workspace": "outputs/exp00"}, RUN_ID, "experiment must be a stable id"),
    ({"experiment": "exp00", "output_workspace": "outputs/other"}, RUN_ID, "output_workspace must be"),
])
def test_an_unstable_run_id_or_plan_namespace_is_refused(layout, plan, run_id, message):
    with pytest.raises(RuntimeError, match=message):
        paths.existing_run_location(plan, run_id)


def test_spawn_checks_exact_locations_before_it_claims_a_lease(monkeypatch, tmp_path):
    from ora.orchestrator import cli

    plan = {"plan_id": "exp00-m0-deadbeef0000", "plan_sha256": "a" * 64, "source": {"commit": "c" * 40},
            "experiment": "exp00", "output_workspace": "outputs/exp00"}
    preflight = {"schema": "ora.preflight.v1", "passed": True,
                 "plan_id": plan["plan_id"], "plan_sha256": plan["plan_sha256"]}
    monkeypatch.setattr(cli, "validate_run_plan", lambda *_: None)
    monkeypatch.setattr(cli, "_require_launchable", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **_kwargs: subprocess.CompletedProcess(
        command, 0, stdout=plan["source"]["commit"] if "rev-parse" in command else ""))
    monkeypatch.setattr(cli, "launch_environment", lambda *_args: {})
    monkeypatch.setattr(cli.paths, "new_run_root", lambda _plan, _run_id: tmp_path / "exp00/runs/planned")

    def walk(_run_id):
        raise AssertionError("spawn must not walk the output trees")

    monkeypatch.setattr(cli.paths, "resolve_run_root", walk)
    checked, leases = [], []
    monkeypatch.setattr(cli.paths, "existing_run_location",
                        lambda checked_plan, run_id: checked.append((checked_plan, run_id)) or tmp_path / "taken")
    monkeypatch.setattr(cli, "_create_lease", lambda *args: leases.append(args) or {"claim_token": "t"})
    with pytest.raises(RuntimeError, match="run-id already exists"):
        cli._spawn(plan, preflight, yes=True)
    assert leases == []
    assert checked[0][0] is plan and checked[0][1].startswith(plan["plan_id"] + "--")

    class LeaseReached(Exception):
        pass

    def lease(*_args):
        raise LeaseReached

    monkeypatch.setattr(cli.paths, "existing_run_location", lambda _plan, _run_id: None)
    monkeypatch.setattr(cli, "_create_lease", lease)
    with pytest.raises(LeaseReached):
        cli._spawn(plan, preflight, yes=True)
