"""Control plane workspace contract regressions."""

from __future__ import annotations
import json
import subprocess
from pathlib import Path
import pytest
from ora.orchestrator import paths
from test_control_plane import (
    ROOT,
)


def test_run_index_resolves_exact_experiment_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / ".harness-runtime")
    monkeypatch.setattr(paths, "RUN_INDEX_ROOT", paths.STATE_ROOT / "run-index")
    monkeypatch.setattr(paths, "LEGACY_RUNS_ROOT", paths.STATE_ROOT / "runs")
    plan = {"experiment": "exp00", "output_workspace": "outputs/exp00"}
    run_id = "exp00-loop-deadbeef0000--20260804T000000Z"
    root = paths.new_run_root(plan, run_id)
    root.mkdir(parents=True)
    paths.register_run(run_id, plan, root)
    assert paths.resolve_run_root(run_id) == root.resolve()
    index = json.loads((paths.RUN_INDEX_ROOT / f"{run_id}.json").read_text())
    assert index["schema"] == "ora.run-index.v2"
    assert index["experiment"] == "exp00"
    assert index["root"] == f"outputs/exp00/runs/{run_id}"
    assert index["workspace_root"] == str(tmp_path)
    assert index["source_worktree"] == str(tmp_path)


def test_duplicate_run_id_across_workspaces_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", tmp_path)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(paths, "STATE_ROOT", tmp_path / ".harness-runtime")
    monkeypatch.setattr(paths, "RUN_INDEX_ROOT", paths.STATE_ROOT / "run-index")
    monkeypatch.setattr(paths, "LEGACY_RUNS_ROOT", paths.STATE_ROOT / "runs")
    monkeypatch.setattr(paths, "linked_worktree_roots", lambda: (tmp_path,))
    monkeypatch.delenv(paths.EXTRA_OUTPUT_ROOTS_ENV, raising=False)
    run_id = "shared-run-id"
    for root in (
        paths.OUTPUTS_ROOT / "exp00/runs" / run_id,
        paths.OUTPUTS_ROOT / "other/runs" / run_id,
    ):
        root.mkdir(parents=True)
        (root / "plan.json").write_text("{}")
    with pytest.raises(RuntimeError, match="ambiguous"):
        paths.resolve_run_root(run_id)


def test_linked_worktrees_share_one_runtime_workspace(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    linked = tmp_path / "linked"
    worktree_git = primary / ".git/worktrees/linked"
    worktree_git.mkdir(parents=True)
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {worktree_git}\n")
    (worktree_git / "commondir").write_text("../..\n")

    assert paths.git_common_checkout_root(linked) == primary.resolve()
    assert paths.resolve_workspace_root(linked, environ={}) == primary.resolve()
    override = (tmp_path / "shared-runtime").resolve()
    assert paths.resolve_workspace_root(
        linked, environ={paths.WORKSPACE_ROOT_ENV: str(override)},
    ) == override
    with pytest.raises(RuntimeError, match="absolute"):
        paths.resolve_workspace_root(
            linked, environ={paths.WORKSPACE_ROOT_ENV: "relative"},
        )


def test_output_roots_include_shared_and_legacy_worktree_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    shared = tmp_path / "primary/outputs"
    legacy = tmp_path / "linked/outputs"
    shared.mkdir(parents=True)
    legacy.mkdir(parents=True)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", shared)
    monkeypatch.setattr(
        paths, "linked_worktree_roots",
        lambda: (shared.parent, legacy.parent),
    )
    assert paths.output_roots(environ={}) == (
        shared.resolve(), legacy.resolve(),
    )


def test_only_output_gitkeep_is_tracked_while_generated_jobs_and_caches_are_ignored() -> None:
    tracked = subprocess.run(
        ["git", "check-ignore", "--no-index", "outputs/.gitkeep"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert tracked.returncode == 1
    for path in (
        "outputs/README.md",
        "outputs/exp00/README.md",
        "outputs/exp00/runs/example/status.json",
        "outputs/exp00/specs/baseline.yaml",
        ".harness-runtime/run-index/example.json",
        ".ruff_cache/cache.db",
        ".mypy_cache/state.json",
        ".pytest_cache/v/cache/nodeids",
        ".cache/tool-state.json",
        "tests/__pycache__/test_control_plane.cpython-311-pytest.pyc",
        "build/lib/control/cli.py",
        "ora_runtime.egg-info/PKG-INFO",
        ".venv/bin/python",
        ".coverage",
    ):
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", path], cwd=ROOT,
            capture_output=True, text=True, check=False,
        )
        assert ignored.returncode == 0, path


def test_historical_control_plane_trees_are_absent() -> None:
    for name in (
        "archive", "benchmark", "config", "exp", "harnesses", "robotwin",
        "runtime", "slides", "systems",
    ):
        assert not (ROOT / name).exists(), name


def test_pytest_runtime_workspace_is_not_the_operator_workspace() -> None:
    from ora.orchestrator import paths

    assert paths.WORKSPACE_ROOT != paths.PROJECT_ROOT
    assert paths.OUTPUTS_ROOT == paths.WORKSPACE_ROOT / "outputs"
    assert "ora-pytest-workspace-" in paths.WORKSPACE_ROOT.name


def test_managed_evidence_accepts_a_capacity_migration_symlink(tmp_path: Path) -> None:
    """`outputs/<experiment>` is a symlink where /home ran out of capacity."""

    migrated = tmp_path / "migrated-outputs" / "capacity-migration-probe"
    (migrated / "qualification" / "services").mkdir(parents=True)
    report = migrated / "qualification" / "services" / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    link = paths.OUTPUTS_ROOT / "capacity-migration-probe"
    paths.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    link.symlink_to(migrated, target_is_directory=True)
    try:
        assert paths.managed_evidence_path(report, label="probe") == report.resolve()
        through_link = link / "qualification" / "services" / "report.json"
        assert paths.managed_evidence_path(through_link, label="probe") == report.resolve()
    finally:
        link.unlink()


def test_managed_evidence_refuses_a_path_outside_every_managed_root(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "report.json"
    outside.parent.mkdir(parents=True)
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="managed experiment output root"):
        paths.managed_evidence_path(outside, label="probe")


def test_managed_evidence_refuses_a_symlink_that_escapes_the_managed_roots(tmp_path: Path) -> None:
    """A planted symlink resolves out of every root and is refused there."""

    target = tmp_path / "planted.json"
    target.write_text("{}\n", encoding="utf-8")
    paths.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    escape = paths.OUTPUTS_ROOT / "escape-probe.json"
    escape.symlink_to(target)
    try:
        with pytest.raises(ValueError, match="managed experiment output root"):
            paths.managed_evidence_path(escape, label="probe")
    finally:
        escape.unlink()


def test_managed_evidence_refuses_a_symlinked_artefact_inside_the_root(tmp_path: Path) -> None:
    """A recorded locator must name the real file, not a link beside it."""

    migrated = tmp_path / "migrated-outputs" / "symlink-below-probe"
    migrated.mkdir(parents=True)
    real = migrated / "report.json"
    real.write_text("{}\n", encoding="utf-8")
    beside = migrated / "beside.json"
    beside.symlink_to(real)
    link = paths.OUTPUTS_ROOT / "symlink-below-probe"
    paths.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    link.symlink_to(migrated, target_is_directory=True)
    try:
        assert paths.managed_evidence_path(link / "report.json", label="probe") == real.resolve()
        with pytest.raises(ValueError, match="symlink below its output root"):
            paths.managed_evidence_path(link / "beside.json", label="probe")
    finally:
        link.unlink()


def test_portable_locator_maps_back_through_a_migrated_experiment_link(tmp_path: Path) -> None:
    """The link's own name is the locator; the volume behind it is not."""

    migrated = tmp_path / "migrated-outputs" / "locator-probe"
    (migrated / "qualification").mkdir(parents=True)
    report = migrated / "qualification" / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    root = paths.OUTPUTS_ROOT.parent
    paths.OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    link = paths.OUTPUTS_ROOT / "locator-probe"
    link.symlink_to(migrated, target_is_directory=True)
    try:
        expected = "outputs/locator-probe/qualification/report.json"
        assert paths.portable_output_locator(report, root) == expected
        assert paths.portable_output_locator(link / "qualification/report.json", root) == expected
        with pytest.raises(ValueError, match="portable outputs locator"):
            paths.portable_output_locator(tmp_path / "elsewhere.json", root)
    finally:
        link.unlink()
