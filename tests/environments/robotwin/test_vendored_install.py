"""Contract tests for the deterministic vendored RoboTwin planner installer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess

import pytest
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

from ora.environment.backends.robotwin import vendored_install
from ora.environment.backends.robotwin.vendored_install import (
    VendoredInstallError,
    expected_profile_source_binding,
    install_vendored_planner,
    vendored_file_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
CURRENT_PROFILE = (
    ROOT / "ora/policy/embodiments/aloha-agilex/control-profile-v10.json"
)
REFRESH_SCRIPT = ROOT / "scripts/dev/refresh-vendored-pins"


def _load_refresh_script():
    loader = SourceFileLoader("refresh_vendored_pins", str(REFRESH_SCRIPT))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "up", "GIT_AUTHOR_EMAIL": "up@example.com",
            "GIT_COMMITTER_NAME": "up", "GIT_COMMITTER_EMAIL": "up@example.com",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00 +0000",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00 +0000",
            "HOME": str(root),
            "PATH": "/usr/bin:/bin:/usr/local/bin",
        },
    )
    return result.stdout


def _make_upstream(root: Path) -> str:
    (root / "envs/robot").mkdir(parents=True)
    (root / "envs/robot/planner.py").write_text("upstream planner\n")
    (root / "envs/robot/robot.py").write_text("upstream robot\n")
    _git(root, "init", "--quiet", "--initial-branch=main")
    _git(root, "add", "-A")
    _git(root, "commit", "--quiet", "--no-verify", "-m", "upstream base")
    return _git(root, "rev-parse", "HEAD").strip()


@pytest.fixture()
def synthetic_install(tmp_path, monkeypatch):
    upstream = tmp_path / "robotwin"
    revision = _make_upstream(upstream)
    payload = tmp_path / "vendored"
    payload.mkdir()
    (payload / "planner.py").write_text("vendored planner with P0 fixes\n")
    (payload / "robot.py").write_text("vendored robot hook\n")
    (payload / "CAPX_MIT_ATTRIBUTION.md").write_text("MIT attribution\n")
    monkeypatch.setattr(vendored_install, "UPSTREAM_REVISION", revision)
    monkeypatch.setattr(vendored_install, "VENDORED_ROOT", payload)
    monkeypatch.setattr(
        vendored_install,
        "UPSTREAM_FILE_SHA256",
        {
            "envs/robot/planner.py": hashlib.sha256(
                b"upstream planner\n"
            ).hexdigest(),
            "envs/robot/robot.py": hashlib.sha256(
                b"upstream robot\n"
            ).hexdigest(),
        },
    )
    return upstream, revision


def test_vendored_install_creates_deterministic_commit_and_noop_reverify(
    synthetic_install, tmp_path,
) -> None:
    upstream, base = synthetic_install
    first = install_vendored_planner(upstream)
    assert first["status"] == "installed"
    assert first["upstream_revision"] == base
    head = _git(upstream, "rev-parse", "HEAD").strip()
    assert head == first["revision"] != base
    assert _git(upstream, "rev-parse", "HEAD^").strip() == base
    assert not _git(upstream, "status", "--porcelain").strip()
    for target in vendored_install.VENDORED_FILES:
        assert (upstream / target).is_file()

    clone = tmp_path / "clone"
    _git(upstream, "worktree", "add", "--detach", str(clone), base)
    second = install_vendored_planner(clone)
    assert second["revision"] == first["revision"]

    reverify = install_vendored_planner(
        upstream, expected_revision=first["revision"],
    )
    assert reverify["status"] == "already_installed"


def test_vendored_install_fails_closed_on_unsafe_checkouts(
    synthetic_install,
) -> None:
    upstream, base = synthetic_install
    (upstream / "envs/robot/planner.py").write_text("local drift\n")
    with pytest.raises(VendoredInstallError, match="not clean"):
        install_vendored_planner(upstream)
    _git(upstream, "checkout", "--", ".")

    _git(upstream, "commit", "--quiet", "--no-verify", "--allow-empty", "-m", "x")
    with pytest.raises(VendoredInstallError, match="vanilla upstream"):
        install_vendored_planner(upstream)
    _git(upstream, "reset", "--quiet", "--hard", base)

    record = install_vendored_planner(upstream)
    with pytest.raises(VendoredInstallError, match="vanilla upstream"):
        install_vendored_planner(upstream)
    with pytest.raises(VendoredInstallError, match="not a git checkout"):
        install_vendored_planner(
            upstream.parent / "missing", expected_revision=record["revision"],
        )
    (upstream / "envs/robot/robot.py").write_text("post-install drift\n")
    _git(upstream, "commit", "--quiet", "--no-verify", "-am", "drift")
    _git(upstream, "commit", "--quiet", "--no-verify", "--allow-empty", "--amend", "-m", "drift")
    with pytest.raises(VendoredInstallError, match="drifted|vanilla upstream"):
        install_vendored_planner(
            upstream, expected_revision=_git(upstream, "rev-parse", "HEAD").strip(),
        )


def test_vendored_payloads_are_tracked_and_hashable() -> None:
    digests = vendored_file_sha256()
    assert set(digests) == set(vendored_install.VENDORED_FILES)
    for value in digests.values():
        assert len(value) == 64 and int(value, 16) >= 0
    assert vendored_install.INSTALL_REVISION != vendored_install.UPSTREAM_REVISION


def test_current_profile_source_binding_matches_vendored_chain() -> None:
    expected = expected_profile_source_binding()
    assert set(expected) == {
        "robotwin_revision", "planner_adapter_sha256",
        "robot_adapter_sha256", "planner_attribution_sha256",
    }
    assert expected["robotwin_revision"] == vendored_install.INSTALL_REVISION
    binding = json.loads(CURRENT_PROFILE.read_text(encoding="utf-8"))[
        "source_binding"
    ]
    assert {key: binding[key] for key in expected} == expected


def test_refresh_script_matches_installer_and_rewrites_idempotently(
    synthetic_install, tmp_path,
) -> None:
    # Historical test citations and all old profile pins remain byte-exact.
    script = _load_refresh_script()
    upstream, base = synthetic_install

    git_dir = upstream / ".git"
    original_git_files = {path.relative_to(git_dir).as_posix(): path.read_bytes()
        for path in git_dir.rglob('*') if path.is_file()}
    computed = script.compute_install_revision(
        upstream,
        upstream_revision=base,
        payload_files={
            target: vendored_install.VENDORED_ROOT / payload
            for target, payload in vendored_install.VENDORED_FILES.items()
        },
        commit_identity=vendored_install._COMMIT_IDENTITY,
        commit_message=vendored_install._COMMIT_MESSAGE,
    )
    assert {path.relative_to(git_dir).as_posix(): path.read_bytes()
        for path in git_dir.rglob('*') if path.is_file()} == original_git_files
    assert _git(upstream, "rev-parse", "HEAD").strip() == base
    installed = install_vendored_planner(upstream)
    assert computed == installed["revision"]

    mirror = tmp_path / "mirror"
    # The frozen control profiles v1-v9 are no longer in this tree; their
    # registry lines below still must not change.
    for relative in (
        script.INSTALLER_PATH, script.CURRENT_PROFILE_PATH,
        script.MOTION_TEST_PATH, script.CAPABILITIES_PATH,
    ):
        destination = mirror / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    binding = {
        "robotwin_revision": "ab" * 20,
        "planner_adapter_sha256": "cd" * 32,
        "robot_adapter_sha256": "ef" * 32,
        "planner_attribution_sha256": "12" * 32,
    }
    historical_paths = (script.MOTION_TEST_PATH,)
    historical_bytes = {relative: (mirror / relative).read_bytes() for relative in historical_paths}
    registry_before = (mirror / script.CAPABILITIES_PATH).read_text()
    first = script.apply_pins(mirror, binding)
    assert sorted(first["changed"]) == sorted((
        script.INSTALLER_PATH, script.CURRENT_PROFILE_PATH,
        script.CAPABILITIES_PATH,
    ))
    rewritten = json.loads(
        (mirror / script.CURRENT_PROFILE_PATH).read_text(encoding="utf-8")
    )["source_binding"]
    assert {key: rewritten[key] for key in binding} == binding
    assert first["profile_sha256"] == hashlib.sha256(
        (mirror / script.CURRENT_PROFILE_PATH).read_bytes()
    ).hexdigest()
    for relative, original in historical_bytes.items():
        assert (mirror / relative).read_bytes() == original
    registry_after = (mirror / script.CAPABILITIES_PATH).read_text()
    for version in range(1, 10):
        prefix = f"    {version}: "
        assert next(line for line in registry_before.splitlines() if line.startswith(prefix)) == next(
            line for line in registry_after.splitlines() if line.startswith(prefix))

    second = script.apply_pins(mirror, binding)
    assert second["changed"] == []
    assert second["profile_sha256"] == first["profile_sha256"]
