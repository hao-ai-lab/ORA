"""Shared output storage remains independent of source checkout placement."""
import json
from pathlib import Path
import pytest
from ora.orchestrator import paths

@pytest.mark.parametrize("storage_link", ["outputs", "experiment", "selector"])
def test_run_index_and_discovery_follow_external_output_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, storage_link: str,
) -> None:
    workspace = tmp_path / "home"
    storage = tmp_path / "data1"
    workspace.mkdir()
    storage.mkdir()
    if storage_link == "outputs":
        (workspace / "outputs").symlink_to(storage, target_is_directory=True)
    elif storage_link == "experiment":
        (workspace / "outputs").mkdir()
        (workspace / "outputs/exp00").symlink_to(storage, target_is_directory=True)
    monkeypatch.setattr(paths, "PROJECT_ROOT", workspace)
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", storage if storage_link == "selector" else workspace / "outputs")
    monkeypatch.setattr(paths, "STATE_ROOT", workspace / ".harness-runtime")
    monkeypatch.setattr(paths, "RUN_INDEX_ROOT", paths.STATE_ROOT / "run-index")
    monkeypatch.setattr(paths, "LEGACY_RUNS_ROOT", paths.STATE_ROOT / "runs")
    monkeypatch.setattr(paths, "output_roots", lambda: (paths.OUTPUTS_ROOT,))
    plan = {"experiment": "exp00", "output_workspace": "outputs/exp00"}
    run_id = "external-storage-run"
    root = paths.new_run_root(plan, run_id)
    root.mkdir(parents=True)
    (root / "plan.json").write_text(json.dumps(plan))
    paths.register_run(run_id, plan, root)
    index_path = paths.RUN_INDEX_ROOT / f"{run_id}.json"
    assert (workspace / "outputs/exp00").resolve() == root.parent.parent.resolve()
    index = json.loads(index_path.read_text())
    assert index["root"] == f"outputs/exp00/runs/{run_id}"
    assert paths._indexed_root(run_id) == root.resolve()
    assert paths.resolve_run_root(run_id) == root.resolve()
    index_path.unlink()
    assert paths.resolve_run_root(run_id) == root.resolve()
    assert list(paths.iter_run_roots()) == [root.resolve()]
    with pytest.raises(RuntimeError, match="does not match"):
        paths.register_run(run_id, plan, tmp_path / "wrong")

def test_output_storage_never_overwrites_a_conflicting_logical_namespace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    workspace, storage = tmp_path / "home", tmp_path / "data1"
    (workspace / "outputs/exp00").mkdir(parents=True)
    witness = workspace / "outputs/exp00/preserve.txt"
    witness.write_text("original evidence")
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", storage)
    plan = {"experiment": "exp00", "output_workspace": "outputs/exp00"}
    root = paths.new_run_root(plan, "new-run")
    root.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="logical output namespace conflicts"):
        paths.register_run("new-run", plan, root)
    assert witness.read_text() == "original evidence"

def test_output_storage_link_does_not_admit_an_escaped_run(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    runs = outputs / "exp00/runs"
    runs.mkdir(parents=True)
    outside = tmp_path / "outside/runs/escaped"
    outside.mkdir(parents=True)
    (outside / "plan.json").write_text("{}")
    (runs / "escaped").symlink_to(outside, target_is_directory=True)
    assert list(paths._output_run_candidates(outputs)) == []

def test_output_storage_selector_is_shared_by_fresh_worktrees(tmp_path: Path) -> None:
    primary, linked, storage = (tmp_path / name for name in ("primary", "linked", "data1"))
    worktree_git = primary / ".git/worktrees/linked"
    worktree_git.mkdir(parents=True)
    linked.mkdir()
    storage.mkdir()
    (linked / ".git").write_text(f"gitdir: {worktree_git}\n")
    (worktree_git / "commondir").write_text("../..\n")
    selector = primary / ".git/ora-output-storage.json"
    assert paths.resolve_outputs_root(primary, project_root=linked, environ={}) == primary / "outputs"
    selector.write_text(json.dumps({"schema": "ora.output-storage.v1", "outputs_root": str(storage)}))
    for source in (primary, linked):
        assert paths.resolve_outputs_root(primary, project_root=source, environ={}) == storage
    override = tmp_path / "explicit-workspace"
    assert paths.resolve_outputs_root(
        override, project_root=linked, environ={paths.WORKSPACE_ROOT_ENV: str(override)},
    ) == override / "outputs"
    storage.rmdir()
    with pytest.raises(RuntimeError, match="invalid output storage selector"):
        paths.resolve_outputs_root(primary, project_root=linked, environ={})

@pytest.mark.parametrize("value", ["relative", "/", None])
def test_invalid_output_storage_selector_never_falls_back_to_home(tmp_path: Path, value) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git/ora-output-storage.json").write_text(json.dumps({
        "schema": "ora.output-storage.v1", "outputs_root": value,
    }))
    with pytest.raises(RuntimeError, match="invalid output storage selector"):
        paths.resolve_outputs_root(tmp_path, project_root=tmp_path, environ={})


def test_an_explicit_outputs_root_wins_over_a_workspace_override_and_the_selector(tmp_path: Path) -> None:
    """Per-node workspaces keep their leases while every node writes evidence to one outputs root."""
    workspace, storage = tmp_path / "ws-node-a", tmp_path / "outputs"
    workspace.mkdir()
    storage.mkdir()
    environ = {paths.WORKSPACE_ROOT_ENV: str(workspace), paths.OUTPUTS_ROOT_ENV: str(storage)}
    assert paths.resolve_outputs_root(workspace, project_root=tmp_path, environ=environ) == storage.resolve()
    environ.pop(paths.OUTPUTS_ROOT_ENV)
    assert paths.resolve_outputs_root(workspace, project_root=tmp_path, environ=environ) == workspace / "outputs"


@pytest.mark.parametrize("value, match", [("relative/outputs", "absolute"), ("/nonexistent/outputs-x", "existing")])
def test_an_outputs_root_must_be_an_existing_absolute_directory(tmp_path: Path, value: str, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        paths.resolve_outputs_root(tmp_path, project_root=tmp_path, environ={paths.OUTPUTS_ROOT_ENV: value})


def test_robotwin_seed_scans_write_under_the_shared_outputs_root(tmp_path: Path) -> None:
    """A seed scan on a per-node workspace lands in the one outputs root, like every other output."""
    import os
    import subprocess
    import sys

    workspace, storage = tmp_path / "ws-node-a", tmp_path / "outputs"
    workspace.mkdir()
    storage.mkdir()
    environ = {**os.environ, paths.WORKSPACE_ROOT_ENV: str(workspace), paths.OUTPUTS_ROOT_ENV: str(storage)}
    printed = subprocess.run(
        [sys.executable, "-c", "from ora.orchestrator import robotwin_evaluation as r; print(r.WORKSPACE)"],
        env=environ, capture_output=True, text=True, check=True, cwd=paths.PROJECT_ROOT,
    ).stdout.strip()
    assert Path(printed) == storage.resolve() / "runtime-v1" / "robotwin-evaluation"


def _per_node_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """A node workspace whose ORA_OUTPUTS_ROOT stores every namespace elsewhere."""
    workspace, storage = tmp_path / "ws-node-a", tmp_path / "outputs"
    (workspace / "outputs").mkdir(parents=True)
    storage.mkdir()
    monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", storage)
    monkeypatch.setattr(paths, "output_roots", lambda: (paths.OUTPUTS_ROOT,))
    return workspace, storage


def test_robodojo_revalidation_reads_records_under_the_shared_outputs_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """EXP11B g008 on 2026-10-11: with OUTPUTS_ROOT set, revalidation refused both the real
    path (no locator) and the workspace link (symlink below its output root)."""
    import hashlib

    from ora.orchestrator import robodojo_qualification as qualification
    from ora.orchestrator import robodojo_revalidation as rv

    workspace, storage = _per_node_workspace(monkeypatch, tmp_path)
    record = storage / "robodojo-compat/qualification/lane/smoke.json"
    record.parent.mkdir(parents=True)
    record.write_text('{"passed": true}')
    link = workspace / "outputs/robodojo-compat"
    assert not link.exists()
    reference = rv.reference(record, workspace)
    assert reference["path"] == "outputs/robodojo-compat/qualification/lane/smoke.json"
    assert link.is_symlink() and link.resolve() == (storage / "robodojo-compat").resolve()
    assert rv.reference(link / "qualification/lane/smoke.json", workspace) == reference
    value, path = rv.read_reference(reference, workspace)
    assert value == {"passed": True} and path == link / "qualification/lane/smoke.json"
    assert paths.managed_evidence_path(path, label="probe") == record.resolve()
    assert qualification._read_record(reference, workspace_root=workspace, label="probe")[0] == {"passed": True}
    digest = hashlib.sha256(record.read_bytes()).hexdigest()
    assert reference["sha256"] == digest


def test_a_workspace_link_onto_other_storage_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Only the link onto the namespace's own storage is admitted, not any link in the workspace."""
    workspace, storage = _per_node_workspace(monkeypatch, tmp_path)
    (storage / "exp00").mkdir()
    (storage / "exp01/qualification").mkdir(parents=True)
    report = storage / "exp01/qualification/report.json"
    report.write_text("{}")
    (workspace / "outputs/exp00").symlink_to(storage / "exp01", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink below its output root"):
        paths.managed_evidence_path(workspace / "outputs/exp00/qualification/report.json", label="probe")
    outside = tmp_path / "elsewhere/report.json"
    outside.parent.mkdir()
    outside.write_text("{}")
    with pytest.raises(ValueError, match="portable outputs locator"):
        paths.portable_output_locator(outside, workspace)
    assert not (workspace / "outputs/elsewhere").exists()


def _card_record(path: Path) -> None:
    from ora.orchestrator.robodojo_card_qualification import CARD_QUALIFICATION_SCHEMA

    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema": CARD_QUALIFICATION_SCHEMA, "passed": True}))


@pytest.mark.parametrize("outputs_root", ["set", "unset"])
def test_robodojo_task_step_accepts_the_card_its_card_step_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outputs_root: str,
) -> None:
    """ORA acceptance 2026-10-13: with ORA_OUTPUTS_ROOT set, `card` wrote under the outputs
    root and `task` refused it ("must live in the shared workspace"). Both now bind the
    same workspace-relative pair that the qualification reader resolves."""
    import hashlib

    from ora.orchestrator import robodojo_qualification as qualification
    from ora.orchestrator import robodojo_spec
    from ora.orchestrator.robodojo_card_qualification import _card_reference

    if outputs_root == "set":
        workspace, storage = _per_node_workspace(monkeypatch, tmp_path)
    else:
        workspace = tmp_path / "ws"
        (workspace / "outputs").mkdir(parents=True)
        monkeypatch.setattr(paths, "WORKSPACE_ROOT", workspace)
        monkeypatch.setattr(paths, "OUTPUTS_ROOT", workspace / "outputs")
        monkeypatch.setattr(paths, "output_roots", lambda: (paths.OUTPUTS_ROOT,))
        storage = workspace / "outputs"
    card = storage / "robodojo-compat/qualification/robodojo-lane-card-x/card-qualification.json"
    _card_record(card)
    _, reference = _card_reference(card)
    assert reference == {
        "path": "outputs/robodojo-compat/qualification/robodojo-lane-card-x/card-qualification.json",
        "sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
    }
    assert qualification._read_record(reference, workspace_root=workspace, label="card")[1].resolve() == card.resolve()
    task = storage / "robodojo-compat/qualification/robodojo-lane-task-x/task-qualification.json"
    task.parent.mkdir(parents=True)
    task.write_text(json.dumps({"passed": True, "card_qualification": reference}))
    layered = robodojo_spec.layered_qualification(task)
    assert layered["card"] == reference
    assert layered["task"]["path"] == "outputs/robodojo-compat/qualification/robodojo-lane-task-x/task-qualification.json"
    assert qualification._read_record(layered["task"], workspace_root=workspace, label="task")[0]["passed"] is True


def test_robodojo_card_outside_workspace_and_outputs_root_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from ora.orchestrator import robodojo_spec
    from ora.orchestrator.robodojo_card_qualification import _card_reference

    _per_node_workspace(monkeypatch, tmp_path)
    stray = tmp_path / "elsewhere/robodojo-compat/card-qualification.json"
    _card_record(stray)
    with pytest.raises(RuntimeError, match="shared workspace or the configured outputs root"):
        _card_reference(stray)
    stray_task = stray.with_name("task-qualification.json")
    stray_task.write_text(json.dumps({"passed": True, "card_qualification": {"path": "x", "sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="shared workspace or the configured outputs root"):
        robodojo_spec.layered_qualification(stray_task)
