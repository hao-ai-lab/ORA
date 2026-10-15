"""Install the harness-vendored RoboTwin planner adapter onto a vanilla checkout.

The reviewed planner chain no longer lives as unreviewable commits in an
external RoboTwin repository.  Its two source files are tracked inside
``ora/environment/backends/robotwin/vendored/`` (with harness-side
P0 corrections) and this installer materializes them onto a **vanilla**
upstream checkout as one deterministic local commit:

- the target checkout must be clean and at the pinned upstream revision (or
  already at the expected install revision, in which case installation is a
  verified no-op);
- the pre-install target files must be byte-identical to their upstream
  versions, so the installer never overwrites unknown local work;
- the install commit uses fixed author/committer identity, date and message,
  so the resulting commit hash is a pure function of the vendored content and
  the upstream base — reproducible anywhere and pinnable by RunPlans.

The installed checkout remains runnable by the native expert/oracle paths:
the vendored planner projects one top-level trajectory for vanilla consumers
and routes ``constraint_pose`` queries through the upstream pose-goal flow,
while the Harness reads only the v2 candidate set and validates it itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

UPSTREAM_REVISION = "c3ddfa8b97d5519efa828b075999bd0006778e5e"

#: Deterministic commit produced by installing the current vendored payloads
#: onto the pinned upstream with the fixed identity below. RunPlans pin this
#: value as the required RoboTwin revision.
LEGACY_INSTALL_REVISION = "9f1fdb1280164f15c4c2d421bf68c552b7dfcd4f"
INSTALL_REVISION = "cf4872cc90538fd4bca761c8db912d8f65e49f2b"
VENDORED_ROOT = Path(__file__).resolve().parent / "vendored"

#: target path inside the RoboTwin checkout -> vendored payload filename.
VENDORED_FILES = {
    "envs/robot/planner.py": "planner.py",
    "envs/robot/robot.py": "robot.py",
    "CAPX_MIT_ATTRIBUTION.md": "CAPX_MIT_ATTRIBUTION.md",
}

#: sha256 of each target file in the vanilla upstream tree; absence means the
#: file does not exist upstream and is created by the install.
UPSTREAM_FILE_SHA256 = {
    "envs/robot/planner.py": (
        "f1012345542483f4cfbac64880a266b7ee0d4a64362d5ec6fd6985ed9c34b564"
    ),
    "envs/robot/robot.py": (
        "3dcd80acc8cab489a4c5edb507cc460dab1724be0226b8d5c4c1b218dee605cb"
    ),
}

_COMMIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "Robot Harness Vendored Installer",
    "GIT_AUTHOR_EMAIL": "harness-vendored@robot-harness.local",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00 +0000",
    "GIT_COMMITTER_NAME": "Robot Harness Vendored Installer",
    "GIT_COMMITTER_EMAIL": "harness-vendored@robot-harness.local",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00 +0000",
}
_COMMIT_MESSAGE = (
    "harness: install vendored planner adapter\n\n"
    "Deterministic install of the harness-owned CaP-X-derived planner chain\n"
    "(P0-corrected) onto vanilla upstream. Content is tracked and reviewed in\n"
    "robot_harness/environment/backends/robotwin/vendored/."
)


class VendoredInstallError(RuntimeError):
    """The RoboTwin checkout cannot safely receive the vendored adapter."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vendored_file_sha256() -> dict[str, str]:
    """Return target-path -> sha256 of the tracked vendored payloads."""

    return {
        target: _file_sha256(VENDORED_ROOT / payload)
        for target, payload in VENDORED_FILES.items()
    }


def expected_profile_source_binding() -> dict[str, str]:
    """Vendored-chain fields a current control profile must pin.

    This is the single source for the hash cascade: the deterministic install
    revision plus the sha256 of every vendored payload, keyed exactly as the
    control-profile ``source_binding`` spells them. Profiles additionally pin
    embodiment sources (URDF/config/base task) that are not vendored here.
    """

    digests = vendored_file_sha256()
    return {
        "robotwin_revision": INSTALL_REVISION,
        "planner_adapter_sha256": digests["envs/robot/planner.py"],
        "robot_adapter_sha256": digests["envs/robot/robot.py"],
        "planner_attribution_sha256": digests["CAPX_MIT_ATTRIBUTION.md"],
    }


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    merged_env = None
    if env is not None:
        import os

        merged_env = {**os.environ, **env}
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=merged_env,
    )
    if result.returncode:
        raise VendoredInstallError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout


def _verify_installed_files(root: Path) -> None:
    expected = vendored_file_sha256()
    for target, digest in expected.items():
        path = root / target
        if not path.is_file() or _file_sha256(path) != digest:
            raise VendoredInstallError(
                f"installed file drifted from the vendored payload: {target}"
            )


def install_vendored_planner(
    robotwin_root: Path, *, expected_revision: str | None = None,
) -> dict[str, Any]:
    """Install (or verify) the vendored planner on ``robotwin_root``.

    Returns a JSON-safe record describing the outcome. Raises
    :class:`VendoredInstallError` when the checkout is in any state other
    than "vanilla upstream" or "already installed as expected".
    """

    root = Path(robotwin_root).resolve()
    if not (root / ".git").exists():
        raise VendoredInstallError(f"not a git checkout: {root}")
    dirty = _git(root, "status", "--porcelain").strip()
    if dirty:
        raise VendoredInstallError(
            f"RoboTwin checkout is not clean:\n{dirty}"
        )
    head = _git(root, "rev-parse", "HEAD").strip()

    if expected_revision is not None and head == expected_revision:
        _verify_installed_files(root)
        parent = _git(root, "rev-parse", "HEAD^").strip()
        if parent != UPSTREAM_REVISION:
            raise VendoredInstallError(
                "install commit does not sit directly on the pinned upstream"
            )
        return {
            "status": "already_installed",
            "robotwin_root": str(root),
            "revision": head,
            "upstream_revision": UPSTREAM_REVISION,
            "files": vendored_file_sha256(),
        }

    if head != UPSTREAM_REVISION:
        raise VendoredInstallError(
            "checkout must be at the pinned vanilla upstream revision "
            f"{UPSTREAM_REVISION} (or the expected install revision); "
            f"found {head}"
        )
    for target, digest in UPSTREAM_FILE_SHA256.items():
        path = root / target
        if not path.is_file() or _file_sha256(path) != digest:
            raise VendoredInstallError(
                f"pre-install file is not the upstream original: {target}"
            )

    for target, payload in VENDORED_FILES.items():
        destination = root / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(VENDORED_ROOT / payload, destination)
    _verify_installed_files(root)
    _git(root, "-c", "core.autocrlf=false", "add", "--", *VENDORED_FILES)
    _git(
        root,
        "-c", "commit.gpgsign=false",
        "commit", "--no-verify", "-m", _COMMIT_MESSAGE,
        env=_COMMIT_IDENTITY,
    )
    revision = _git(root, "rev-parse", "HEAD").strip()
    if expected_revision is not None and revision != expected_revision:
        raise VendoredInstallError(
            f"deterministic install produced {revision}, "
            f"expected {expected_revision}"
        )
    return {
        "status": "installed",
        "robotwin_root": str(root),
        "revision": revision,
        "upstream_revision": UPSTREAM_REVISION,
        "files": vendored_file_sha256(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Install the harness-vendored RoboTwin planner adapter as one "
            "deterministic local commit on a vanilla upstream checkout."
        )
    )
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument(
        "--expected-revision",
        help="fail unless the resulting HEAD equals this commit hash",
    )
    args = parser.parse_args(argv)
    try:
        record = install_vendored_planner(
            args.robotwin_root, expected_revision=args.expected_revision,
        )
    except VendoredInstallError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


__all__ = [
    "INSTALL_REVISION",
    "LEGACY_INSTALL_REVISION",
    "UPSTREAM_REVISION",
    "VENDORED_FILES",
    "VendoredInstallError",
    "expected_profile_source_binding",
    "install_vendored_planner",
    "vendored_file_sha256",
]


if __name__ == "__main__":
    raise SystemExit(main())
