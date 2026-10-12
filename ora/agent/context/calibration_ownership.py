"""One owner for the calibration numbers a delivered observation carries.

Two independent modules render the same hash-bound calibration artifact.
``current_visual.camera_geometry`` publishes ``intrinsic_cv``, ``extrinsic_cv``
and the derived ``cam2world_cv``, ``camera_origin_world_m`` and
``camera_axes_world``. ``public_facts`` publishes the same intrinsic and
world-to-camera matrices beside the pixel-space mapping only it owns. When a
payload carries both, the matrices were byte-identical, so the second rendering
bought the policy nothing and cost it context on every observation it retained.

The richer producer keeps the numbers. This module rewrites the co-delivered
facts entry to name that owner. It lives at the delivery site rather than
inside either producer: whether a payload duplicates is a property of what the
site assembles, not of either projection on its own, and ``public_facts`` is
content-addressed by frozen embodiment descriptors that an edit here must not
disturb.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

from .current_visual import full_visual_enabled

CALIBRATION_DELEGATED_STATUS = "reported_in_camera_geometry"
CAMERA_GEOMETRY_OWNER = "camera_geometry"
CALIBRATION_OWNER_KEY = "calibration_owner"

#: Where the delegated numbers live, stated once for the whole payload.
#:
#: A bare status reads as "unavailable" to a policy, so the mapping has to be
#: stated -- ``world_to_camera_cv`` is published by the owner under a different
#: name, and a policy that searched ``camera_geometry`` for the original name
#: would find nothing and reasonably conclude the value was gone.
#:
#: It is stated once rather than per camera because it is the same sentence for
#: every camera but the name, and this payload carries one entry per camera per
#: retained observation. Repeating it cost 222 B against 95 B, which is 26.9 KB
#: instead of 11.5 KB at model call 40 of a sealed RoboTwin run -- replacing one
#: constant repeated 124 times with a different constant repeated 124 times.
#:
#: ``field`` is relative to the delivered payload, never rooted at one runtime's
#: envelope: Native nests the owner under ``current_actual`` and the managed
#: projection nests the same object under ``current_visual``, so only the path
#: they share is true for both. ``<camera>`` is the key of the entry in
#: ``cameras`` that carries the delegated status.
CALIBRATION_OWNER = {
    "field": f"{CAMERA_GEOMETRY_OWNER}.<camera>",
    "intrinsic_cv": "intrinsic_cv",
    "world_to_camera_cv": "extrinsic_cv",
}


def _pointer() -> dict[str, Any]:
    """Replace a duplicated calibration with its status and what only it owns."""
    return {
        "status": CALIBRATION_DELEGATED_STATUS,
        # The one statement about those matrices that camera_geometry does not
        # make: they index native RGB pixels, before any sheet translation.
        # Kept per entry because a wrong answer here silently shifts every
        # pixel the policy projects, and it is 40 B.
        "pixel_basis": "native_rgb_before_sheet_translation",
    }


def delegated_cameras(view_config: Mapping[str, Any] | None, *, snapshot_role: str) -> frozenset[str]:
    """Cameras whose calibration a co-delivered ``camera_geometry`` also owns.

    Derived from ``view_config``, which is always present, never from the
    possibly absent ``camera_geometry`` itself. ``camera_geometry`` publishes
    exactly ``automatic_actual_views`` of the current observation, so a
    historical snapshot and a contact-sheet source camera - neither of which it
    covers - are excluded here and keep their own numbers.
    """
    if not full_visual_enabled(view_config) or snapshot_role != "current":
        return frozenset()
    return frozenset((view_config or {}).get("automatic_actual_views") or ())


def delegate_calibration_to_camera_geometry(
    public_facts: Mapping[str, Any], *, view_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the facts with every duplicated calibration replaced by a pointer.

    Returns an equal copy when this payload carries no ``camera_geometry``, so
    a profile that declares ``current_public_robot_facts_v1`` without
    ``full_current_visual_v1`` keeps the full matrices. Only an entry that
    actually reported numbers is rewritten; ``unavailable`` stays unavailable.
    """
    facts = copy.deepcopy(dict(public_facts))
    cameras = facts.get("cameras")
    if not isinstance(cameras, Mapping):
        return facts
    owned = delegated_cameras(view_config, snapshot_role=facts.get("snapshot_role", "current"))
    if not owned:
        return facts
    delegated = {
        name: (
            {**entry, "calibration": _pointer()}
            if name in owned and isinstance(entry.get("calibration"), Mapping)
            and entry["calibration"].get("status") == "reported"
            else entry
        )
        for name, entry in cameras.items()
    }
    facts["cameras"] = delegated
    # Only claim an owner when an entry actually delegates to it. A payload
    # where every camera kept its numbers must not carry a pointer to a field
    # nothing in it references.
    if any(entry["calibration"].get("status") == CALIBRATION_DELEGATED_STATUS
           for entry in delegated.values() if isinstance(entry.get("calibration"), Mapping)):
        facts[CALIBRATION_OWNER_KEY] = dict(CALIBRATION_OWNER)
    return facts
