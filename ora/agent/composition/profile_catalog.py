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
    # Integration 0059 unit A0: the default generation of each backend
    # (direct_v4). Each is its v1 base manifest plus the 0059 units, with the
    # published-limits layers last.
    "robotwin_toolchain_b2_closed_noservo_v2": PROFILE_ROOT / "toolchain_b2_noservo_v1" / "robotwin_toolchain_b2_closed_noservo_v2-manifest.json",
    # The LIBERO default since integration 0084: direct_v2 plus the overhead footprint localization
    # (EXP12B/placement-centre/g001).
    "libero_placement_direct_overhead_v1": PROFILE_ROOT / "explicit_motion_v3" / "libero_placement_direct_overhead_v1-manifest.json",
    "robodojo_placement_closed_skills_v2": PROFILE_ROOT / "robodojo_closed_skills_v1" / "robodojo_placement_closed_skills_v2-manifest.json",
    # Composed tools M2a: each default closed-skill surface with grasp_execute
    # served by the composed library entry grasp_execute/c1. Registered for
    # comparison, not selected by any baseline (COMPOSED_SUCCESSORS below).
    "robotwin_toolchain_b2_closed_composed_v1": PROFILE_ROOT / "toolchain_b2_noservo_v1" / "robotwin_toolchain_b2_closed_composed_v1-manifest.json",
    "robodojo_placement_closed_composed_v1": PROFILE_ROOT / "robodojo_closed_skills_v1" / "robodojo_placement_closed_composed_v1-manifest.json",
    # The RoboTwin closed arm without the bounded servo (TOOLCHAIN-B2); the
    # runtime_v1 RoboTwin prepared-grasp qualification binds it.
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


#: Composed-tools successors and the default surface each was derived from.
#: A successor carries every capability and layer of its source and only
#: rebinds tools to composed library entries, so a check of which profiles
#: declare a default-generation capability counts these with their sources.
COMPOSED_SUCCESSORS = {
    "robotwin_toolchain_b2_closed_composed_v1": "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo_placement_closed_composed_v1": "robodojo_placement_closed_skills_v2",
}
