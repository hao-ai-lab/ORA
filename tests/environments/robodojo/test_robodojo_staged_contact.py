"""RoboDojo staged contact approach: global pre-contact move, then the bounded local step.

Opt-in through ``runtime.contact_approach_route``, which the RunSpec derives
from a surface declaring ``robodojo_staged_contact_approach_v1``. Without it a
contact approach beyond the 50 mm/15 degree local envelope is refused exactly as
the frozen Dojo surfaces state.
"""
from __future__ import annotations

import pytest

from ora.environment.backends.robodojo.bridge import RobodojoError
from ora.environment.backends.robodojo.sparse_motion import (
    GLOBAL_EXECUTOR, LOCAL_EXECUTOR, PLAN_VALIDATION, STAGED_CONTACT_STRATEGY,
)
from ora.environment.backends.robodojo.staged_contact import (
    PRECONTACT_LOCAL_ROTATION_RAD, PRECONTACT_LOCAL_TRANSLATION_M, STAGED_CONTACT_CAPABILITY, handoff_refusal,
    precontact_poses,
)
from tests.environments.robodojo.test_robodojo_backend import fake_sdk_candidate_boundary, pose, sparse, sparse_session, v3  # noqa: F401

STAGED = {"contact_approach_route": STAGED_CONTACT_STRATEGY}


@pytest.fixture
def staged(tmp_path):
    instance = sparse_session(tmp_path, **STAGED)
    try:
        yield instance
    finally:
        instance.close()


def test_a_large_contact_approach_is_staged_and_discloses_its_precontact(staged, tmp_path):
    receipt = staged.execute(v3(staged, tmp_path, targets={"left": pose([0, 0, 0.6])},
                                interaction="contact_approach"))["receipt"]
    assert receipt["execution_status"] == "reached"
    assert receipt["motion_strategy"] == STAGED_CONTACT_STRATEGY
    assert receipt["executor"] == [GLOBAL_EXECUTOR, LOCAL_EXECUTOR]
    assert receipt["failure_stage"] is None and receipt["failure_code"] is None
    arm = receipt["arms"]["left"]
    assert [item["segment"] for item in arm["segments"]] == ["precontact", "local"]
    assert arm["segments"][0]["target_pose_xyz_xyzw"][2] == pytest.approx(0.6 + PRECONTACT_LOCAL_TRANSLATION_M)
    assert arm["precontact"] == {"pose_xyz_xyzw": pytest.approx([0, 0, 0.64, 0, 0, 0, 1]),
                                 "attempted": True, "reason": None}
    assert arm["plan_validation"] == PLAN_VALIDATION
    assert arm["pose_error"]["translation_m"] == pytest.approx(0.0, abs=1e-9)
    assert receipt["native_actions"] == 2 and receipt["native_action_budget"] == 20


def test_without_the_route_the_released_refusal_is_unchanged(sparse, tmp_path):
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0, 0, 0.6])},
                                interaction="contact_approach"))["receipt"]
    assert receipt["execution_status"] == "plan_rejected"
    assert receipt["failure_code"] == "route_contact_approach_exceeds_local_bound"
    assert "precontact" not in receipt["arms"]["left"] and receipt["native_actions"] == 0


def test_a_small_contact_approach_stays_local_on_the_staged_route(staged, tmp_path):
    receipt = staged.execute(v3(staged, tmp_path, targets={"left": pose([0, 0, 0.77])},
                                interaction="contact_approach"))["receipt"]
    assert receipt["motion_strategy"] == "local" and receipt["executor"] == [LOCAL_EXECUTOR]
    assert "precontact" not in receipt["arms"]["left"]


def test_a_precontact_ik_failure_writes_nothing_and_reports_the_precontact_stage(staged, tmp_path):
    staged.environment.robot_manager.fail_ik = True
    receipt = staged.execute(v3(staged, tmp_path, targets={"left": pose([0, 0, 0.6])},
                                interaction="contact_approach"))["receipt"]
    assert receipt["execution_status"] == "planner_failed"
    assert receipt["failure_stage"] == "planner" and receipt["failure_code"] == "ik_failed:precontact"
    assert receipt["executor"] is None and receipt["native_actions"] == 0
    assert staged.environment.commands == []
    assert receipt["arms"]["left"]["precontact"]["attempted"] is False
    assert receipt["arms"]["left"]["precontact"]["reason"] == "precontact_segment_stopped"


def test_a_local_refusal_after_the_precontact_move_is_not_reached_not_plan_rejected(staged, tmp_path):
    staged.environment.robot_manager.unreachable = lambda target: target[2] < 0.62
    receipt = staged.execute(v3(staged, tmp_path, targets={"left": pose([0, 0, 0.6])},
                                interaction="contact_approach"))["receipt"]
    # The pre-contact move wrote the plant, so the action cannot claim zero writes.
    assert receipt["execution_status"] == "not_reached"
    assert receipt["failure_stage"] == "planner" and receipt["failure_code"] == "ik_failed:local"
    assert receipt["executor"] == [GLOBAL_EXECUTOR] and receipt["state_changed"] is True
    assert receipt["arms"]["left"]["precontact"]["attempted"] is True
    assert [item["segment"] for item in receipt["arms"]["left"]["segments"]] == ["precontact", "local"]
    assert receipt["arms"]["left"]["segments"][1]["reached"] is False


def test_free_space_and_precision_ignore_the_contact_route(staged, tmp_path):
    for interaction, height in (("free_space", 0.6), ("precision", 0.8)):
        receipt = staged.execute(v3(staged, tmp_path, targets={"left": pose([0, 0, height])},
                                    interaction=interaction))["receipt"]
        assert receipt["motion_strategy"] == "global", interaction
        assert "precontact" not in receipt["arms"]["left"]


def test_precontact_leaves_one_local_step_with_the_endpoint_tolerance_in_reserve():
    before = {"left": {"pose_xyz_xyzw": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]}}
    far = precontact_poses(before, {"left": [0.3, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0]})["left"]
    assert far[0] == pytest.approx(0.3 - PRECONTACT_LOCAL_TRANSLATION_M)
    assert PRECONTACT_LOCAL_TRANSLATION_M == pytest.approx(0.04)
    assert PRECONTACT_LOCAL_ROTATION_RAD == pytest.approx(0.2617993877991494 - 0.10)


def test_an_unknown_route_is_refused_at_session_construction(tmp_path):
    with pytest.raises(RobodojoError, match="contact_approach_route"):
        sparse_session(tmp_path, contact_approach_route="global")


def test_the_hand_off_refuses_in_a_fixed_order():
    at = {"left": {"pose_xyz_xyzw": [0, 0, 0.64, 0, 0, 0, 1]}}
    precontact = {"left": [0, 0, 0.64, 0, 0, 0, 1]}
    target = {"left": [0, 0, 0.6, 0, 0, 0, 1]}
    common = {"precontact": precontact, "targets": target}
    assert handoff_refusal(terminal=False, landed=at, remaining=5, **common) is None
    assert handoff_refusal(terminal=True, landed=at, remaining=5, **common) == \
        "episode_terminated_before_contact_approach"
    short = {"left": {"pose_xyz_xyzw": [0, 0, 0.7, 0, 0, 0, 1]}}
    assert handoff_refusal(terminal=False, landed=short, remaining=5, **common) == "precontact_segment_not_reached"
    assert handoff_refusal(terminal=False, landed=at, remaining=0, **common) == "native_action_budget_exhausted"
    far = {"left": [0, 0, 0.5, 0, 0, 0, 1]}
    assert handoff_refusal(terminal=False, landed=at, remaining=5, precontact=precontact, targets=far) == \
        "residual_outside_local_servo_envelope"


def test_a_surface_without_the_capability_keeps_the_refusal_route(monkeypatch, tmp_path):
    """The default Dojo surface derives the route; a test-local copy without the capability
    resolves without one and refuses one supplied by hand."""
    import tests.support.finish_v2_surfaces as finish_v2_surfaces
    import tests.environments.robodojo.test_robodojo_closed_skills_default as parity
    from ora.orchestrator.spec import RunSpecError, resolve_run_spec

    assert resolve_run_spec(parity.spec_for("stack_blocks"), root=parity.ROOT)["runtime"][
        "contact_approach_route"] == STAGED_CONTACT_STRATEGY
    profile = finish_v2_surfaces.install(monkeypatch, tmp_path, parity.SURFACE,
                                         profile_id="robodojo_without_staged_contact_test",
                                         drop=[STAGED_CONTACT_CAPABILITY])
    monkeypatch.setattr(parity, "SURFACE", profile)
    spec = parity.spec_for("stack_blocks")
    assert "contact_approach_route" not in resolve_run_spec(spec, root=parity.ROOT)["runtime"]
    spec["runtime"]["contact_approach_route"] = STAGED_CONTACT_STRATEGY
    with pytest.raises(RunSpecError, match="contact_approach_route"):
        resolve_run_spec(spec, root=parity.ROOT)
