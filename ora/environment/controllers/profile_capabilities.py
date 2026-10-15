"""Single-source capability derivation for embodiment control profiles.

Historically every capability question ("does this profile use the motion
transaction? the top-K planner?") was answered by a hand-copied set of
profile-id strings at each call site. Eight-plus copies of those sets drifted
independently and every new profile version required a whole-repo sweep; two
of the v9 regressions were direct products of one missed copy. This module is
now the only place that knows which profile identity corresponds to which
version, and capabilities are expressed as minimum-version thresholds.

Adding a profile version means adding exactly one entry to
``PROFILE_VERSION_IDS`` (and, when a genuinely new capability is introduced,
one threshold constant) — never touching consumer call sites.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

#: Exact, closed registry of known profile identities by version.
PROFILE_VERSION_IDS: dict[int, str] = {
    1: "robotwin_aloha_sparse_dual_arm_v1",
    2: "robotwin_aloha_grip_center_dual_arm_v2",
    3: "robotwin_aloha_grip_center_tracked_dual_arm_v3",
    4: "robotwin_aloha_validated_dual_arm_v4",
    5: "robotwin_aloha_motion_transaction_dual_arm_v5",
    6: "robotwin_aloha_motion_intent_dual_arm_v6",
    7: "robotwin_aloha_motion_stability_dual_arm_v7",
    8: "robotwin_aloha_rgbd_grasp_staging_dual_arm_v8",
    9: "robotwin_aloha_correctness_dual_arm_v9",
    10: "robotwin_aloha_contact_recovery_dual_arm_v10",
}

_IDS_TO_VERSION = {identity: version for version, identity in PROFILE_VERSION_IDS.items()}

#: Frozen byte identity of every tracked ALOHA control-profile file, keyed by
#: version. Consumers and tests read this single table instead of hand-copying
#: sha256 literals; `scripts/dev/refresh-vendored-pins` rewrites only the current
#: version's entry when the vendored hash cascade moves it.
FROZEN_CONTROL_PROFILE_SHA256: dict[int, str] = {
    1: "9f3d11f4d074430d249ca0c7a60cf87ced5570539916523d4ebc6e3220b7feed",
    2: "7565ce4503db4cf472fa665390eaea89d9d75f9420cce73a5dda7fb3cba56191",
    3: "821d01eef94e3dbdbb0c15e020bf983ceb809ab5e9264f8eb1fa6b8597034be6",
    4: "66efecd0a4d4ea5c04e011dfe966f0f6132964e2ed6a30836fd173eba4588594",
    5: "a1f242be94dc2e0d5d0cb8f80b4c3e09d21bdba71a36edba6c71d7e90454b28d",
    6: "3c6edffed601ec4f3c03d31e18bc2b2e514ee9ebbc8e2e38cc96a9408a0cb36f",
    7: "7579a98854f5ef036320dc9e920f890e1f661dd4a601b56dcb79de2f38fff0ba",
    8: "7e4e6d28f9d4016eb45ef6fbc6b181bc98083dc6a6d36b03677e05b2516c02ef",
    9: "a8c39aef821180ea5c99adb23ff3ff1e056e16b288c44c3f898631f247403112",
    10: "9730bb978d99f62e474f7701a5701123f482662d460fab9061b3c58cd3770d9e",
}

#: Capability thresholds: the first profile version that provides each one.
GRIP_CENTER_SINCE = 2
CONTROLLER_TRACKING_SINCE = 3
VALIDATED_PLANNER_SINCE = 4
MOTION_TRANSACTION_SINCE = 5
MOTION_INTENT_SINCE = 6
TOPK_BATCH_PLANNING_SINCE = 7
LOCAL_SERVO_DIAGNOSTICS_SINCE = 7
RGBD_GRASP_STAGING_SINCE = 8
FIRST_TERMINAL_SINCE = 9
CONTACT_RECOVERY_SINCE = 10
IK_CONSTRAINT_FEEDBACK_SINCE = 10

#: The oldest registered profile version this tree executes and composes:
#: the RunSpec resolver (``orchestrator/spec_runtime.py``) and the RoboTwin
#: session refuse an older one, and a recorded RoboTwin run bound to an older
#: one is evaluated at its own source commit. It is the contact-recovery v10
#: profile, which every maintained RoboTwin composition binds. The older
#: versions' files, ids, hashes and thresholds stay registered, so a recorded
#: run bound to one is still identified.
OLDEST_COMPOSED_CONTROL_PROFILE = 10

# A pattern string, not a compiled one: this module is on the RoboDojo worker's
# startup chain, whose unbound modules may only declare values at import.
_PROFILE_PATH_VERSION = r"control-profile-v(\d+)\.json\Z"
_VERSION_BY_SHA256 = {digest: version for version, digest in FROZEN_CONTROL_PROFILE_SHA256.items()}


def control_profile_version(profile: Any) -> int | None:
    """Return the registered version for ``profile``'s identity, else None.

    ``profile`` is the inner profile mapping (the object carrying
    ``profile_id``). Unknown identities return None so consumers stay
    fail-closed: an unregistered profile has no capabilities.
    """

    if not isinstance(profile, Mapping):
        return None
    return _IDS_TO_VERSION.get(profile.get("profile_id"))


def loaded_control_profile_version(control_profile: Any) -> int | None:
    """Version for a loaded ``{"profile": {...}, "sha256": ...}`` record."""

    if not isinstance(control_profile, Mapping):
        return None
    return control_profile_version(control_profile.get("profile"))


def profile_has(control_profile: Any, *, since_version: int) -> bool:
    """True when the loaded profile's version is at least ``since_version``."""

    version = loaded_control_profile_version(control_profile)
    return version is not None and version >= since_version


def bound_control_profile_version(binding: Any) -> int | None:
    """The version a ``runtime.control_profile`` binding names, without reading its file.

    A loaded record's ``profile`` identity comes first, then the frozen byte
    identity of ``sha256``, then the ``control-profile-v<N>.json`` file name of
    ``path``. None when the binding names no version in any of these ways.
    """

    if not isinstance(binding, Mapping):
        return None
    version = control_profile_version(binding.get("profile"))
    if version is None:
        version = _VERSION_BY_SHA256.get(binding.get("sha256"))
    if version is None and isinstance(binding.get("path"), str):
        match = re.search(_PROFILE_PATH_VERSION, binding["path"])
        version = int(match.group(1)) if match else None
    return version


__all__ = [
    "CONTACT_RECOVERY_SINCE",
    "CONTROLLER_TRACKING_SINCE",
    "FROZEN_CONTROL_PROFILE_SHA256",
    "FIRST_TERMINAL_SINCE",
    "GRIP_CENTER_SINCE",
    "IK_CONSTRAINT_FEEDBACK_SINCE",
    "LOCAL_SERVO_DIAGNOSTICS_SINCE",
    "MOTION_INTENT_SINCE",
    "MOTION_TRANSACTION_SINCE",
    "OLDEST_COMPOSED_CONTROL_PROFILE",
    "PROFILE_VERSION_IDS",
    "RGBD_GRASP_STAGING_SINCE",
    "TOPK_BATCH_PLANNING_SINCE",
    "VALIDATED_PLANNER_SINCE",
    "bound_control_profile_version",
    "control_profile_version",
    "loaded_control_profile_version",
    "profile_has",
]
