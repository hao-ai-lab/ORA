"""Every RoboDojo arm-motion admission screens the other arm (E11G12-004a).

The fake plant has toy kinematics (joints 0-2 are the link6 position), so these
tests drive the executor with a stand-in inter-arm screen whose verdict is a
function of the arms' joint lines; the screen itself and its X5 geometry are
covered in ``test_robodojo_inter_arm_clearance``. They pin where each motion
path asks the screen, what it refuses before writing and what the receipt and
the public projection say.
"""
from __future__ import annotations

import pytest

from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext, public_motion_failure_code, resolve_receipt_effect_projector,
)
from ora.contracts.reach_check import ROBODOJO_REASONS
from ora.environment.backends.robodojo import ik, recovery
from tests.environments.robodojo.test_robodojo_backend import fake_sdk_candidate_boundary as _fake_boundary
from tests.environments.robodojo.test_robodojo_backend import pose, sparse_session, v3
from tests.environments.robodojo.test_robodojo_recovery import WIDE, WOUND, _plant_with, _recover

fake_sdk_candidate_boundary = pytest.fixture(autouse=True)(_fake_boundary.__wrapped__)

BLOCKED = {"min_clearance_m": -0.012, "required_clearance_m": 0.005, "start_clearance_m": 0.08,
           "limiting_links": {"left": "link4", "right": "link6"}, "limiting_fraction": 0.5}


class Screen:
    """Stand-in lockstep screen: ``refuse(moving, starts, ends)`` decides each call."""

    def __init__(self, refuse):
        self.refuse, self.calls = refuse, []

    def straight_paths(self, starts, ends):
        moving = sorted(arm for arm in starts if list(starts[arm]) != list(ends[arm]))
        self.calls.append(moving)
        blocked = self.refuse(moving, starts, ends)
        return {"status": "inter_arm_collision" if blocked else "clear", "moving_arms": moving,
                **(BLOCKED if blocked else {**BLOCKED, "min_clearance_m": 0.2})}


@pytest.fixture
def screen(monkeypatch):
    holder = Screen(lambda moving, starts, ends: False)
    monkeypatch.setattr(ik, "inter_arm_screen", lambda session: holder)
    return holder


@pytest.fixture
def sparse(tmp_path):
    session = sparse_session(tmp_path)
    try:
        yield session
    finally:
        session.close()


def _public(receipt, tool="move_pose"):
    projector = resolve_receipt_effect_projector(receipt["schema"])
    return projector(receipt, ReceiptProjectionContext(
        last_action_id=receipt["action_id"], last_action_public_name=tool,
        surface_supports=lambda capability: True))


def test_a_single_arm_move_into_the_held_other_arm_is_refused_before_writing(sparse, tmp_path, screen):
    screen.refuse = lambda moving, starts, ends: ends["left"][0] >= 0.1
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.12, 0, 0.8])}))["receipt"]
    assert receipt["execution_status"] == "planner_failed" and receipt["failure_stage"] == "planner"
    assert receipt["failure_code"] == "inter_arm_collision:move"
    assert receipt["native_actions"] == 0 and sparse.environment.commands == []
    left = receipt["arms"]["left"]
    assert left["planning_diagnostics"]["outcome"] == "inter_arm_collision_path"
    assert "left link4 to right link6" in left["planning_diagnostics"]["note"]
    assert left["segments"][0]["ik_selection"]["inter_arm_clearance"]["min_clearance_m"] == -0.012
    # The policy reads planner_failed with the outcome, the clearance and the advice.
    public = str(_public(receipt))
    assert "planner_failed" in public and "inter_arm_collision_path" in public and "-0.012" in public


def test_a_move_that_keeps_clear_of_the_other_arm_executes(sparse, tmp_path, screen):
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.05, 0, 0.8])}))["receipt"]
    assert receipt["execution_status"] == "reached" and screen.calls == [["left"]]


def test_two_arms_moving_together_are_screened_in_lockstep_before_writing(sparse, tmp_path, screen):
    screen.refuse = lambda moving, starts, ends: moving == ["left", "right"]
    receipt = sparse.execute(v3(sparse, tmp_path, targets={
        "left": pose([0, 0, 0.84]), "right": pose([0, 0, 0.86])}))["receipt"]
    # Each arm's IK is screened only against itself; the pair is screened together.
    assert screen.calls == [["left", "right"]]
    assert receipt["execution_status"] == "planner_failed"
    assert receipt["failure_code"] == "inter_arm_collision:move" and receipt["native_actions"] == 0
    assert sparse.environment.commands == []
    for arm in ("left", "right"):
        diagnostics = receipt["arms"][arm]["planning_diagnostics"]
        assert diagnostics["outcome"] == "inter_arm_collision_path" and diagnostics["phase"] == "joint_space"
        assert "lockstep" in diagnostics["evidence_scope"]
        assert receipt["arms"][arm]["segments"][-1]["inter_arm_clearance"]["moving_arms"] == ["left", "right"]


def test_a_local_step_into_the_other_arm_is_refused(sparse, tmp_path, screen):
    screen.refuse = lambda moving, starts, ends: bool(moving)
    receipt = sparse.execute(v3(sparse, tmp_path, targets={"left": pose([0.02, 0, 0.8])},
                                interaction="precision"))["receipt"]
    assert receipt["motion_strategy"] == "local"
    assert receipt["failure_code"] == "inter_arm_collision:local" and receipt["native_actions"] == 0


def test_a_probe_target_the_other_arm_blocks_is_not_reachable(sparse, tmp_path, screen):
    from tests.environments.robodojo.test_robodojo_backend import probe_request

    screen.refuse = lambda moving, starts, ends: bool(moving)
    _, request = probe_request(sparse, tmp_path, [0.06, 0, 0.8])
    result = sparse.probe_motion(request)["results"][0]
    assert result["reachable"] is False and result["plan_ref"] is None


# --- recover_pose ------------------------------------------------------------

def _changes_yaw_and_fold(starts, ends, arm):
    first, last = starts[arm], ends[arm]
    return abs(first[0] - last[0]) > 1e-9 and any(abs(a - b) > 1e-9 for a, b in zip(first[1:], last[1:]))


def test_a_one_arm_recovery_picks_an_unwind_order_that_keeps_clear_of_the_other_arm(tmp_path, screen):
    screen.refuse = lambda moving, starts, ends: moving == ["right"] and _changes_yaw_and_fold(starts, ends, "right")
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0080-right"], basket=False)
        receipt = _recover(session, tmp_path)
    finally:
        session.close()
    progress = receipt["arms"]["right"]["recovery_progress"]
    assert progress["unwind_order"] == "fold_then_yaw"
    assert {"order": "straight_joint_line", "status": "inter_arm_collision"}.items() <= progress["path_screens"][0].items()


def test_a_two_arm_recovery_blocked_at_its_first_step_is_refused_before_any_write(tmp_path, screen):
    screen.refuse = lambda moving, starts, ends: moving == ["left", "right"]
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0080-right"], basket=False)
        _plant_with(session, "left", WOUND["f3a0-0079-left"], basket=False)
        receipt = session.execute(v3(session, tmp_path, kind="recover_motion",
                                     targets={"left": "home", "right": "home"}))["receipt"]
    finally:
        session.close()
    assert receipt["execution_status"] == "plan_rejected" and receipt["native_actions"] == 0
    assert receipt["failure_stage"] == "plan_validation"
    assert receipt["failure_code"] == recovery.INTER_ARM_REFUSAL == "recovery_inter_arm_collision_path"
    for arm in ("left", "right"):
        progress = receipt["arms"][arm]["recovery_progress"]
        assert progress["outcome"] == "refused_before_motion"
        assert progress["retract"]["reason"] == "inter_arm_collision_path"
        assert progress["inter_arm_clearance"]["moving_arms"] == ["left", "right"]
    assert public_motion_failure_code("plan_validation", receipt["failure_code"]) == recovery.INTER_ARM_REFUSAL


def test_a_recovery_step_blocked_from_the_measured_state_stops_with_its_own_code(tmp_path, screen):
    # Admission (retract, then the first unwind step) is clear; the re-screen
    # from the measured state after the retract finds the other arm.
    screen.refuse = lambda moving, starts, ends: len(screen.calls) > 3
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0080-right"], basket=False)
        _plant_with(session, "left", WOUND["f3a0-0079-left"], basket=False)
        receipt = session.execute(v3(session, tmp_path, kind="recover_motion",
                                     targets={"left": "home", "right": "home"}))["receipt"]
    finally:
        session.close()
    assert receipt["native_actions"] > 0 and receipt["failure_stage"] == "execution"
    assert receipt["failure_code"] == recovery.INTER_ARM_REFUSAL
    for arm in ("left", "right"):
        progress = receipt["arms"][arm]["recovery_progress"]
        assert progress["stopped"]["reason"] == recovery.INTER_ARM_FROM_MEASURED
        assert progress["outcome"] in {"refused_after_retract", "refused_mid_unwind"}
    # Not an infrastructure fault on the public channel.
    assert public_motion_failure_code("execution", recovery.INTER_ARM_REFUSAL) == recovery.INTER_ARM_REFUSAL


def test_the_released_straight_recovery_is_screened_before_writing(tmp_path, screen):
    screen.refuse = lambda moving, starts, ends: bool(moving)
    session = sparse_session(tmp_path, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0080-right"], basket=False)
        receipt = _recover(session, tmp_path)
    finally:
        session.close()
    assert receipt["execution_status"] == "plan_rejected" and receipt["native_actions"] == 0
    assert receipt["failure_code"] == recovery.INTER_ARM_REFUSAL
    assert receipt["arms"]["right"]["recovery_progress"]["outcome"] == "refused_before_motion"


# --- read-only reach and the public code tables ------------------------------

def test_the_reach_check_names_an_other_arm_refusal():
    from ora.environment.backends.robodojo import reach_check

    solved = {"joint_target": None, "ik_selection": {"outcome": "inter_arm_collision_path"}, "candidates": []}
    assert reach_check._unreached(solved, local=False) == "inter_arm_collision_path"
    assert "inter_arm_collision_path" in ROBODOJO_REASONS
    assert "other arm" in reach_check.CLAIM_BOUNDARY


def test_a_planner_stage_other_arm_refusal_reads_planner_failed():
    assert public_motion_failure_code("planner", "inter_arm_collision:move") == "planner_failed"


def test_a_backoff_probe_refused_by_the_other_arm_drives_no_backoff(tmp_path, screen):
    # Level 1 (backoff_then_unwind) after an earlier stall: the straight probe
    # toward home is refused; it made no progress because it never ran, so no
    # joint may be called blocked and nothing is backed off on that evidence.
    screen.refuse = lambda moving, starts, ends: moving == ["right"] and _changes_yaw_and_fold(starts, ends, "right")
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0080-right"], basket=False)
        session._recovery_fallback = {"round": session.round_index, "levels": {"right": 1}}
        receipt = _recover(session, tmp_path)
    finally:
        session.close()
    arm = receipt["arms"]["right"]
    names = [item["segment"] for item in arm["segments"]]
    assert names[0] == "recover_probe" and arm["segments"][0]["refused"] == "inter_arm_collision_path"
    assert "recover_backoff" not in names
    progress = arm["recovery_progress"]
    assert progress["blocked_joints"] == [] and progress["unwind_order"] == "fold_then_yaw"


def test_a_yaw_first_two_arm_recovery_is_admitted_on_the_order_it_will_run(tmp_path, screen):
    # Level 2 prefers yaw_then_fold; its first step (both arms yawing together)
    # meets the other arm, so the refusal comes before any write.
    def yaw_only(moving, starts, ends):
        return moving == ["left", "right"] and all(
            abs(starts[arm][0] - ends[arm][0]) > 1e-9 and list(starts[arm][1:]) == list(ends[arm][1:])
            for arm in moving)

    screen.refuse = yaw_only
    session = sparse_session(tmp_path, recovery_route=recovery.RETRACT_THEN_UNWIND, **WIDE)
    try:
        _plant_with(session, "right", WOUND["f3a0-0080-right"], basket=False)
        _plant_with(session, "left", WOUND["f3a0-0079-left"], basket=False)
        session._recovery_fallback = {"round": session.round_index, "levels": {"left": 2, "right": 2}}
        receipt = session.execute(v3(session, tmp_path, kind="recover_motion",
                                     targets={"left": "home", "right": "home"}))["receipt"]
    finally:
        session.close()
    assert receipt["execution_status"] == "plan_rejected" and receipt["native_actions"] == 0
    assert receipt["failure_code"] == recovery.INTER_ARM_REFUSAL
    for arm in ("left", "right"):
        progress = receipt["arms"][arm]["recovery_progress"]
        assert progress["outcome"] == "refused_before_motion" and progress["unwind_order"] == "yaw_then_fold"
        assert "joints" not in progress["retract"]
