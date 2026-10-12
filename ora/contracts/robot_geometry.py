"""Host-safe comparison of public robot-only geometry, never scene geometry."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping


GEOMETRY_BINDING_SCHEMA = "ora.public-robot-geometry-binding.v1"
GEOMETRY_VERIFICATION_SCHEMA = "ora.loaded-robot-geometry-verification.v1"


def verify_loaded_robot_geometry(binding: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"schema", "descriptor_sha256", "geometry"} or binding["schema"] != GEOMETRY_BINDING_SCHEMA:
        raise ValueError("invalid public robot geometry binding")
    digest = binding["descriptor_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("public robot geometry requires a descriptor hash")
    # A micrometre is numerical transform tolerance, not motion arrival or a
    # grasp/contact tolerance. It does not alter any controller threshold.
    tolerance_m = 1e-6

    def compare(expected: Any, actual: Any, path: str) -> None:
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping) or set(expected) != set(actual):
                raise ValueError(f"loaded robot geometry fields differ at {path}")
            for key in expected:
                compare(expected[key], actual[key], path + "." + key)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                raise ValueError(f"loaded robot geometry dimensions differ at {path}")
            for index, (a, b) in enumerate(zip(expected, actual)):
                compare(a, b, f"{path}[{index}]")
        elif type(expected) in (int, float):
            if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isfinite(expected) or abs(expected-actual) > tolerance_m:
                raise ValueError(f"loaded robot geometry numeric mismatch at {path}")
        elif expected != actual:
            raise ValueError(f"loaded robot geometry convention differs at {path}")

    expected_geometry = binding["geometry"]
    if not isinstance(expected_geometry, Mapping) or not expected_geometry:
        raise ValueError("empty robot geometry binding")
    compare(expected_geometry, observed, "public_robot_geometry")
    canonical = json.dumps(observed, sort_keys=True, separators=(",", ":"))
    return {"schema": GEOMETRY_VERIFICATION_SCHEMA, "descriptor_sha256": digest,
            "observed_geometry_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            "comparison_tolerance_m": tolerance_m, "scope": "loaded_robot_geometry_only",
            "passed": True, "simulation_steps": 0, "controller_writes": 0}
