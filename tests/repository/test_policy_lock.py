"""The generated policy.lock inventory: coverage, determinism and drift detection."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from ora.orchestrator import policy_lock

ROOT = Path(__file__).resolve().parents[2]
LINE = re.compile(r"^[0-9a-f]{64}  [^\s].*$")

FIXTURE_FILES = {
    "ora/policy/tool_surfaces/a/manifest.json": '{"components": []}\n',
    "ora/policy/embodiments/x/card.md": "card\n",
    "ora/policy/baselines/b.json": "{}\n",
    "ora/environment/benchmarks/pools.json": "{}\n",
    "ora/environment/benchmarks/robodojo.py": "# code, not an asset\n",
    "ora/policy/environments/robodojo/task-contracts/records/t.json": "{}\n",
    "experiments/runtime_v1/run.py": "# code, not an asset\n",
    "experiments/runtime_v1/summary.md": "not an asset\n",
    "experiments/runtime_v1/qualification/other.json": "{}\n",
    "README.md": "not an asset\n",
}
COVERED = sorted(
    path for path in FIXTURE_FILES
    if not path.endswith(".py") and path not in {
        "experiments/runtime_v1/summary.md",
        "experiments/runtime_v1/qualification/other.json",
        "README.md",
    }
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Fixture",
         "-c", "user.email=fixture@example.invalid", *args],
        check=True, capture_output=True,
    )


def _repository(root: Path, files: dict[str, str] = FIXTURE_FILES) -> Path:
    root.mkdir(parents=True)
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = _repository(tmp_path / "repository")
    scratch = root / "ora/policy/scratch-untracked.json"
    scratch.write_text("{}\n", encoding="utf-8")
    return root


def test_lock_lists_exactly_the_tracked_covered_assets_in_sha256sum_format(repository):
    text = policy_lock.render_lock(repository)
    assert text.endswith("\n") and "\r" not in text
    lines = text.splitlines()
    assert all(LINE.match(line) for line in lines)
    paths = [line[66:] for line in lines]
    assert paths == COVERED
    for line in lines:
        digest, path = line[:64], line[66:]
        assert digest == hashlib.sha256((repository / path).read_bytes()).hexdigest()


def test_lock_is_deterministic_and_independent_of_creation_order(repository, tmp_path):
    reversed_files = dict(reversed(list(FIXTURE_FILES.items())))
    other = _repository(tmp_path / "other", reversed_files)
    assert policy_lock.render_lock(repository) == policy_lock.render_lock(repository)
    assert policy_lock.render_lock(repository) == policy_lock.render_lock(other)


def test_check_detects_a_modified_an_added_and_a_deleted_asset(repository, capsys):
    policy_lock.write_lock(repository)
    assert policy_lock.check_lock(repository).clean
    (repository / "ora/policy/embodiments/x/card.md").write_text("edited\n")
    added = repository / "ora/policy/tool_surfaces/a/new-component.json"
    added.write_text("{}\n")
    _git(repository, "add", added.relative_to(repository).as_posix())
    _git(repository, "rm", "-q", "ora/policy/environments/robodojo/task-contracts/records/t.json")

    drift = policy_lock.check_lock(repository)
    assert drift.changed == ("ora/policy/embodiments/x/card.md",)
    assert drift.missing == ("ora/policy/tool_surfaces/a/new-component.json",)
    assert drift.extra == ("ora/policy/environments/robodojo/task-contracts/records/t.json",)
    assert not drift.clean

    assert policy_lock.command_lock(argparse.Namespace(check=True), repository) == 1
    output = capsys.readouterr()
    assert "changed: ora/policy/embodiments/x/card.md" in output.err
    assert "missing from policy.lock: ora/policy/tool_surfaces/a/new-component.json" in output.err
    assert "extra in policy.lock: ora/policy/environments/robodojo/task-contracts/records/t.json" in output.err
    assert "ora lock" in output.err

    assert policy_lock.command_lock(argparse.Namespace(check=False), repository) == 0
    assert policy_lock.command_lock(argparse.Namespace(check=True), repository) == 0
    assert policy_lock.check_lock(repository).clean


def test_a_worktree_deletion_without_git_rm_is_a_deleted_asset(repository):
    policy_lock.write_lock(repository)
    (repository / "ora/policy/baselines/b.json").unlink()
    assert policy_lock.check_lock(repository).extra == ("ora/policy/baselines/b.json",)


@pytest.mark.parametrize("damage", [
    lambda text: text.replace("\n", "\r\n"),
    lambda text: "\n".join(reversed(text.splitlines())) + "\n",
    lambda text: text.replace("  ", " ", 1),
    lambda text: text + text.splitlines()[0] + "\n",
    lambda text: text.rstrip("\n"),
    lambda text: "# comment\n" + text,
])
def test_a_malformed_or_non_canonical_lock_fails_the_check(repository, damage):
    policy_lock.write_lock(repository)
    path = repository / policy_lock.LOCK_PATH
    path.write_bytes(damage(path.read_text(encoding="utf-8")).encode("utf-8"))
    drift = policy_lock.check_lock(repository)
    assert drift.errors and not drift.clean


def test_an_absent_lock_fails_the_check_and_has_no_hash(repository):
    drift = policy_lock.check_lock(repository)
    assert drift.errors == ("policy.lock is absent; run `ora lock`",)
    assert policy_lock.lock_sha256(repository) is None
    policy_lock.write_lock(repository)
    raw = (repository / "policy.lock").read_bytes()
    assert policy_lock.lock_sha256(repository) == hashlib.sha256(raw).hexdigest()


def test_the_lock_refuses_a_path_sha256sum_would_escape(tmp_path):
    files = {**FIXTURE_FILES, "ora/policy/odd\\name.json": "{}\n"}
    root = _repository(tmp_path / "odd", files)
    with pytest.raises(policy_lock.PolicyLockError, match="cannot be written"):
        policy_lock.render_lock(root)


def test_the_committed_lock_is_in_sha256sum_check_format():
    """`sha256sum -c policy.lock` verifies a checkout without the project installed."""

    text = (ROOT / policy_lock.LOCK_PATH).read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines and all(LINE.match(line) for line in lines)
    assert [line[66:] for line in lines] == sorted(line[66:] for line in lines)


# ---- coverage of every path-loaded asset class -------------------------------

def _covered() -> set[str]:
    return set(policy_lock.covered_paths(ROOT))


def _tracked() -> set[str]:
    listed = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, text=True, check=True,
    )
    return {item for item in listed.stdout.split("\0") if item}


def test_every_coverage_entry_names_its_loader_and_matches_tracked_assets():
    covered = _covered()
    for entry in policy_lock.COVERAGE:
        assert entry.loader.strip() and entry.reason.strip(), entry.pathspec
        matched = policy_lock.covered_paths(ROOT, coverage=(entry,))
        assert matched, f"coverage entry matches no tracked asset: {entry.pathspec}"
        assert set(matched) <= covered
    assert not any(path.endswith(".py") for path in covered)


def test_coverage_includes_each_path_loaded_asset_class():
    """Each loader that reads a policy asset by path finds its file in the lock."""

    from ora.agent.composition import baseline
    from ora.agent.composition.embodiment_geometry import PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS
    from ora.agent.composition.profile_catalog import PROFILE_CATALOG
    from ora.agent.composition.robodojo_layout_compatibility import LAYOUT_CATALOG_PATH
    from ora.agent.composition.robodojo_task_contract import REGISTRY_PATH
    from ora.agent.composition.task_contract import REGISTRIES

    covered, tracked = _covered(), _tracked()
    expected = {
        LAYOUT_CATALOG_PATH, REGISTRY_PATH, *REGISTRIES.values(),
        *baseline.BASELINE_CATALOGS.values(),
        *(path.relative_to(ROOT).as_posix() for path in PROFILE_CATALOG.values()),
        *(f"ora/policy/embodiments/{name}"
          for names in PUBLIC_GEOMETRY_DESCRIPTOR_VERSIONS.values() for name in names),
    }
    # RoboDojo registries name contract records and layers.
    for registry in ROOT.glob("ora/policy/environments/robodojo/task-contracts/registry-*.json"):
        for entry in json.loads(registry.read_text())["tasks"].values():
            expected.update(entry[key] for key in ("record", "layer") if key in entry)
    # Capability manifests name prompt bundles.
    for manifest in ROOT.glob("ora/policy/capabilities/*.json"):
        expected.update(
            path for path in (json.loads(manifest.read_text()).get("prompt_bundles") or {}).values()
            if isinstance(path, str)
        )
    present = {path for path in expected if path in tracked}
    assert {path for path in present if path.startswith("ora/policy/environments/robodojo/task-contracts/reviews/")}
    assert sorted(present - covered) == []


def _string_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _string_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)
    elif isinstance(value, str):
        yield value


def test_every_file_a_covered_asset_names_by_path_is_covered_or_declared_unloaded():
    """The lock is closed under the path references its own assets make."""

    covered, tracked = _covered(), _tracked()
    pattern = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.@+-]+)+\.(json|ya?ml|md)$")
    uncovered: dict[str, str] = {}
    for relative in sorted(covered):
        if not relative.endswith((".json", ".yaml", ".yml")):
            continue
        text = (ROOT / relative).read_text(encoding="utf-8")
        value = json.loads(text) if relative.endswith(".json") else yaml.safe_load(text)
        for item in _string_values(value):
            if not pattern.match(item):
                continue
            for candidate in (item, (Path(relative).parent / item).as_posix()):
                if candidate in tracked and candidate not in covered:
                    uncovered[candidate] = relative
    declared = {
        path: source for path, source in uncovered.items()
        if path.startswith(tuple(item.prefix for item in policy_lock.UNLOADED_REFERENCES))
    }
    assert {path: source for path, source in uncovered.items() if path not in declared} == {}
    for item in policy_lock.UNLOADED_REFERENCES:
        assert item.reason.strip()
