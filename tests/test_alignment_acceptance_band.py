"""align_plan accepts what the backend planner would have solved.

The first version read `local_cartesian` for its tolerances. That block is the
local servo controller's *convergence* bound -- how precisely the controller
must settle once it is tracking -- and it demands 0.003 m and 0.03 rad. The
backend's own inverse kinematics accepts 0.005 m and 0.05 rad
(`backends/robotwin/vendored/planner.py:670-671`), which is also what the
profile declares as `arrival_check`.

So every pose whose best solution landed between 3 and 5 mm was reported
`reachable: false` by a tool whose whole purpose is to tell a policy which
approaches are worth spending a world mutation on. The refusal was written into
the threshold, not found by the solver, and nothing in the result said so: the
candidate came back with a `position_error_m` inside the planner's own bound and
a verdict of unreachable.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "ora/policy/embodiments/aloha-agilex/control-profile-v10.json"
PLANNER = ROOT / "ora/environment/backends/robotwin/vendored/planner.py"


def _profile() -> dict:
    """The file is the profile; the session supplies the {"profile": ...} wrapper."""

    return json.loads(PROFILE.read_text(encoding="utf-8"))


def test_the_two_tolerance_blocks_really_do_disagree():
    """The premise. If they ever converge, this module has nothing to guard."""

    profile = _profile()
    servo = profile["local_cartesian"]
    arrival = profile["arrival_check"]
    assert servo["position_tolerance_m"] < arrival["position_tolerance_m"]
    assert servo["orientation_tolerance_rad"] < arrival["orientation_tolerance_rad"]
    # The band that was being refused, named so a reader sees its size.
    assert arrival["position_tolerance_m"] - servo["position_tolerance_m"] == pytest.approx(0.002)


def test_alignment_accepts_on_the_arrival_bound_not_the_servo_bound():
    """Asserted through behaviour, not a source grep.

    A grep for the lookup expression breaks the moment the lookup is rewritten --
    which it was, to name a missing block instead of raising KeyError. Hand the
    planner a profile whose two blocks disagree and read which tolerance arrived.
    """

    from ora.environment.controllers.alignment import plan_alignment
    from ora.contracts.alignment import REQUEST_SCHEMA

    seen: dict[str, float] = {}

    class _Adapter:
        def solve_batch_reachability(self, arm, poses, *, position_tolerance_m,
                                     orientation_tolerance_rad):
            seen["position"] = position_tolerance_m
            seen["orientation"] = orientation_tolerance_rad
            return [{
                "reachable": True, "position_error_m": 0.004,
                "orientation_error_rad": 0.04, "iterations": 3,
                "claim_boundary": "test",
            } for _ in poses]

    profile = {"profile": {
        "arrival_check": {"position_tolerance_m": 0.005, "orientation_tolerance_rad": 0.05},
        "local_cartesian": {"position_tolerance_m": 0.003, "orientation_tolerance_rad": 0.03},
    }}
    plan_alignment(
        {"schema": REQUEST_SCHEMA, "target_ref": "t", "approach_side": "top",
         "arm": "left", "target_poses_xyzw": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]},
        adapter=_Adapter(), control_profile=profile, available_arms=("left", "right"),
    )
    assert seen == {"position": 0.005, "orientation": 0.05}


def test_a_profile_without_arrival_check_names_the_block_it_lacks():
    """A KeyError here reaches the policy as an opaque internal_tool_error."""

    from ora.contracts.alignment import AlignmentRequestDrift, REQUEST_SCHEMA
    from ora.environment.controllers.alignment import plan_alignment

    with pytest.raises(AlignmentRequestDrift, match="arrival_check"):
        plan_alignment(
            {"schema": REQUEST_SCHEMA, "target_ref": "t", "approach_side": "top",
             "arm": "left", "target_poses_xyzw": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]},
            adapter=object(), control_profile={"profile": {"local_cartesian": {}}},
            available_arms=("left",),
        )


def test_the_arrival_bound_matches_what_the_backend_planner_accepts():
    """A pose this tool calls reachable is one the planner's own IK would admit.

    Pinned against the vendored literals rather than restated, so a planner
    upgrade that moves its thresholds fails here instead of silently reopening
    the gap from the other side.
    """

    planner = PLANNER.read_text(encoding="utf-8")
    arrival = _profile()["arrival_check"]
    assert f"position_threshold={arrival['position_tolerance_m']}," in planner
    assert f"rotation_threshold={arrival['orientation_tolerance_rad']}," in planner


# ---------------------------------------------------------------------------
# Pinned from the other side. These arrived from a measurement unit that
# branched before the arrival_check fix landed; its two tests asserting the
# pre-fix behaviour are deliberately not carried over, because the fix inverted
# them. What survives is what the fix did not change: what the backend's
# threshold literally is, and what this solver structurally cannot see.
# ---------------------------------------------------------------------------

from ora.environment.controllers.reachability import solve_reachability

BACKEND_IK_POSITION_THRESHOLD_M = 0.005
BACKEND_IK_ROTATION_THRESHOLD_RAD = 0.05


def test_backend_ik_thresholds_restated_here_match_the_planner_source() -> None:
    """The restated constants are checked against the file they come from."""
    source = (
        ROOT / "ora/environment/backends/robotwin/vendored/planner.py"
    ).read_text()
    assert f"position_threshold={BACKEND_IK_POSITION_THRESHOLD_M}," in source
    assert f"rotation_threshold={BACKEND_IK_ROTATION_THRESHOLD_RAD}," in source
    # The same block is what makes the backend's verdict a 128-seed,
    # collision-checked one rather than a single damped-least-squares descent.
    assert "seed_count = 128" in source
    assert "self_collision_check=True," in source


def test_the_solver_is_given_no_collision_model_of_any_kind() -> None:
    """What the tool structurally cannot see bounds what its verdict can mean.

    `solve_reachability` takes forward kinematics and a limit box. There is no
    parameter through which a scene, a mesh, a sphere set or the other arm could
    reach it, so every collision refusal the backend makes is one this solver
    could not have made. That is the honest half of the claim boundary, and it
    is a property of the signature rather than of any one call.
    """
    import inspect

    parameters = set(inspect.signature(solve_reachability).parameters)
    assert parameters == {
        "target_pose_xyzw", "seed_joints", "fk_public_pose", "joint_limits",
        "position_tolerance_m", "orientation_tolerance_rad",
        "maximum_iterations", "damping", "maximum_joint_step_rad",
    }
    for forbidden in ("world", "collision", "scene", "mesh", "spheres", "adapter"):
        assert not any(forbidden in name for name in parameters), forbidden
