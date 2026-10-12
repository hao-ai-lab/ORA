"""Host-safe, provider-independent Native baseline catalog.

This tree composes the default baseline and the maintained ones registered in
``BASELINE_CATALOGS``; any other id is refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .locked_assets import locked_asset


BASELINE_ID = "native_full_explicit_motion_direct_v4"
CATALOG_PATH = "ora/policy/baselines/native-full-explicit-motion-direct-v4.json"
# Selectable from this tree: the default and every maintained baseline.
BASELINE_CATALOGS = {
    BASELINE_ID: CATALOG_PATH,
}
#: Every catalog carries a LIBERO-Pro and a RoboTwin instance; from direct_v4
#: (integration 0059 unit A0) the default also carries RoboDojo.
REQUIRED_INSTANCES = frozenset({"libero_pro", "robotwin"})
OPTIONAL_INSTANCES = frozenset({"robodojo"})
#: Instance runtime fields that name a profile asset by path.
RUNTIME_PROFILE_FIELDS = ("motion_profile", "control_profile")


def load_native_baseline(root: Path, baseline_id: str = BASELINE_ID) -> dict[str, Any]:
    """Return the tracked baseline and its exact catalog identity; no probes.

    An instance names its runtime motion and control profiles by path; the
    returned catalog carries the ``{path, sha256}`` record of each, hashed
    from the bytes read here, which is what the RunSpec runtime receives.
    """
    path = BASELINE_CATALOGS[baseline_id]
    raw = (root / path).read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "ora.native-baseline.v1"
        or value.get("id") != baseline_id
        or not REQUIRED_INSTANCES <= set(value.get("instances", {})) <= REQUIRED_INSTANCES | OPTIONAL_INSTANCES
    ):
        raise ValueError("invalid Native baseline catalog")
    for name, instance in value["instances"].items():
        runtime = instance.get("runtime", {})
        for field in RUNTIME_PROFILE_FIELDS:
            if field in runtime:
                runtime[field] = locked_asset(
                    root, runtime[field], label=f"{baseline_id} {name} runtime.{field}",
                ).record()
    return {**value, "path": path, "sha256": hashlib.sha256(raw).hexdigest()}
