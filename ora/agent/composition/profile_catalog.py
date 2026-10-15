"""The registry of profile ids, and the root their manifests live under.

Data, not logic: every registered profile is one line here, and a generation
that adds a sibling adds another. Splitting it keeps the resolver
about resolution, and it is re-exported from :mod:`tool_surface` so every
existing import keeps working.
"""

from __future__ import annotations

from pathlib import Path


PROFILE_ROOT = Path(__file__).resolve().parents[2] / "policy" / "tool_surfaces"
PROFILE_CATALOG = {
    # The default generation of each backend (direct_v4). Each is its v1 base
    # manifest plus the later capability layers, with the published-limits
    # layers last.
    "robotwin_toolchain_b2_closed_noservo_v2": PROFILE_ROOT / "toolchain_b2_noservo_v1" / "robotwin_toolchain_b2_closed_noservo_v2-manifest.json",
    # The LIBERO default: direct_v2 plus the overhead footprint localization.
    "libero_placement_direct_overhead_v1": PROFILE_ROOT / "explicit_motion_v3" / "libero_placement_direct_overhead_v1-manifest.json",
    "robodojo_placement_closed_skills_v2": PROFILE_ROOT / "robodojo_closed_skills_v1" / "robodojo_placement_closed_skills_v2-manifest.json",
    # The earlier RoboTwin closed-skill surface: the default's predecessor,
    # which keeps the place_plan/place_execute/place_resume and
    # grasp_verify_retention tools. No baseline selects it.
    "robotwin_toolchain_b2_closed_noservo_v1": PROFILE_ROOT / "toolchain_b2_noservo_v1" / "robotwin-prepared-grasp-v2-aperture-v1-manifest.json",
    # The runtime_v1 qualification ladder (robot_runtime_v1), the RunSpec
    # default when policy.tool_surface is omitted (robot_runtime_v2,
    # orchestrator/spec_policy.py) and the Synthetic Native surface
    # (native_atomic_causal_v1).
    "robot_runtime_v1": PROFILE_ROOT / "robot_runtime_v1" / "manifest.json",
    "robot_runtime_v2": PROFILE_ROOT / "robot_runtime_v2" / "manifest.json",
    "native_atomic_causal_v1": (
        PROFILE_ROOT / "native_atomic_causal_v1" / "manifest.json"
    ),
    "local_atomic_cpu_v1": PROFILE_ROOT / "local_atomic_cpu_v1" / "manifest.json",
}

