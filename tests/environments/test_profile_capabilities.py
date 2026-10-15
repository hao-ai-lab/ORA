"""Contract tests for single-source control-profile capability derivation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ora.environment.controllers.embodiment_control import (
    CURRENT_PROFILE_VERSION,
    load_embodiment_control_profile,
)
from ora.environment.controllers.profile_capabilities import (
    FROZEN_CONTROL_PROFILE_SHA256,
    FIRST_TERMINAL_SINCE,
    GRIP_CENTER_SINCE,
    MOTION_TRANSACTION_SINCE,
    OLDEST_COMPOSED_CONTROL_PROFILE,
    PROFILE_VERSION_IDS,
    RGBD_GRASP_STAGING_SINCE,
    TOPK_BATCH_PLANNING_SINCE,
    control_profile_version,
    loaded_control_profile_version,
    profile_has,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "ora/policy/embodiments/aloha-agilex"
#: Only the profiles this tree composes keep their files; older versions stay
#: registered by identity for the runs that recorded them.
COMPOSED_VERSIONS = sorted(v for v in PROFILE_VERSION_IDS if v >= OLDEST_COMPOSED_CONTROL_PROFILE)


def _loaded(version: int):
    path = PROFILE_DIR / f"control-profile-v{version}.json"
    return load_embodiment_control_profile(
        path, hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _identified(version: int):
    """A loaded record carrying only the registered identity of ``version``."""
    return {"profile": {"profile_id": PROFILE_VERSION_IDS[version]}}


def test_every_registered_profile_resolves_its_own_version() -> None:
    for version in PROFILE_VERSION_IDS:
        assert loaded_control_profile_version(_identified(version)) == version
    for version in COMPOSED_VERSIONS:
        loaded = _loaded(version)
        assert loaded_control_profile_version(loaded) == version
        assert control_profile_version(loaded["profile"]) == version
        raw = json.loads(
            (PROFILE_DIR / f"control-profile-v{version}.json").read_text()
        )
        assert raw["profile_id"] == PROFILE_VERSION_IDS[version]


def test_capability_thresholds_partition_versions() -> None:
    for version in PROFILE_VERSION_IDS:
        loaded = _loaded(version) if version in COMPOSED_VERSIONS else _identified(version)
        assert profile_has(loaded, since_version=GRIP_CENTER_SINCE) == (
            version >= 2
        )
        assert profile_has(loaded, since_version=MOTION_TRANSACTION_SINCE) == (
            version >= 5
        )
        assert profile_has(loaded, since_version=TOPK_BATCH_PLANNING_SINCE) == (
            version >= 7
        )
        assert profile_has(loaded, since_version=RGBD_GRASP_STAGING_SINCE) == (
            version >= 8
        )
        assert profile_has(loaded, since_version=FIRST_TERMINAL_SINCE) == (
            version >= 9
        )


def test_frozen_profile_table_matches_tracked_files() -> None:
    assert set(FROZEN_CONTROL_PROFILE_SHA256) == set(PROFILE_VERSION_IDS)
    assert COMPOSED_VERSIONS
    for version in COMPOSED_VERSIONS:
        frozen = FROZEN_CONTROL_PROFILE_SHA256[version]
        path = PROFILE_DIR / f"control-profile-v{version}.json"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen, version


def test_current_profile_version_is_the_registry_maximum() -> None:
    # A version registered above CURRENT_PROFILE_VERSION would silently skip
    # semantic validation (fail-open); the current pointer must track the
    # registry maximum in the same change that registers a new version.
    assert CURRENT_PROFILE_VERSION == max(PROFILE_VERSION_IDS)


def test_unknown_profiles_have_no_capabilities() -> None:
    assert control_profile_version(None) is None
    assert control_profile_version({"profile_id": "unregistered_v99"}) is None
    assert loaded_control_profile_version({"profile": {}}) is None
    assert not profile_has({"profile": {"profile_id": "x"}}, since_version=1)
    assert not profile_has(None, since_version=1)
