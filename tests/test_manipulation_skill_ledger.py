"""The two closed skills: where a policy is consulted, and where it is not.

The character of the redesigned surface is in these two sequences. A grasp is
one committed run with no stop; a place has exactly one, at the release.
"""

from __future__ import annotations

import pytest

from ora.agent.tools.robot import ToolError
from ora.agent.tools.skill_ledger import (
    GRASP_STAGES, PLACE_STAGES, POLICY_VISIBLE_STOP, SCHEMA,
    abandon, begin, pending_stage, record, resume, stages_for,
    terminal_status, validate_state, yields_to_policy,
)


def _grasp(**overrides):
    kwargs = dict(
        skill="grasp_execute", candidate_id="G1",
        rationale="G1 is the only candidate clear of the neighbouring mug",
        observation_id="obs-1", observation_sha256="a" * 64,
    )
    kwargs.update(overrides)
    return begin(**kwargs)


def _place(**overrides):
    kwargs = dict(
        skill="place_execute", candidate_id="P0",
        rationale="P0 has the widest hull margin of the three patches",
        observation_id="obs-1", observation_sha256="a" * 64,
    )
    kwargs.update(overrides)
    return begin(**kwargs)


def _run(state, stages, status="executed"):
    for index, stage in enumerate(stages):
        state, _transition = record(
            state, stage=stage, status=status, reason="ok", action_id=f"action-{index}",
        )
    return state


def test_a_grasp_runs_approach_close_and_verify_as_one_committed_sequence():
    state = _grasp()

    assert state["schema"] == SCHEMA
    assert state["sequence"] == list(GRASP_STAGES)
    assert pending_stage(state) == "approach"
    assert not yields_to_policy(state)

    state = _run(state, GRASP_STAGES)
    assert terminal_status(state) == "completed"


def test_a_grasp_never_yields_to_the_policy_at_any_stage():
    """The candidate choice is the decision; there is nothing to decide halfway."""

    state = _grasp()
    for stage in GRASP_STAGES:
        assert not yields_to_policy(state), f"{stage} must not consult the policy"
        state, _ = record(state, stage=stage, status="executed", reason="ok")


def test_asking_a_grasp_to_stop_is_refused_rather_than_ignored():
    """Silently dropping the request would leave the policy expecting a view."""

    with pytest.raises(ToolError) as refused:
        _grasp(stop_before_commit=True)
    assert refused.value.field == "stop_before_commit"
    assert "candidate choice" in str(refused.value)


def test_a_place_without_a_requested_stop_has_no_stop_stage():
    state = _place()

    assert POLICY_VISIBLE_STOP not in state["sequence"]
    assert state["sequence"] == ["transport", "descend", "release", "settle", "verify"]
    state = _run(state, state["sequence"])
    assert terminal_status(state) == "completed"


def test_the_place_stop_sits_immediately_before_the_release():
    """Placing's hard moment is the instant of release, so the view lands there."""

    state = _place(stop_before_commit=True)

    assert state["sequence"] == list(PLACE_STAGES)
    index = state["sequence"].index(POLICY_VISIBLE_STOP)
    assert state["sequence"][index - 1] == "descend"
    assert state["sequence"][index + 1] == "release"


def test_the_stop_is_the_only_place_a_closed_skill_waits():
    state = _place(stop_before_commit=True)

    yielded = []
    for stage in PLACE_STAGES:
        if yields_to_policy(state):
            yielded.append(stage)
        status = "stopped" if stage == POLICY_VISIBLE_STOP else "executed"
        state, transition = record(state, stage=stage, status=status, reason="ok")
        if stage == POLICY_VISIBLE_STOP:
            assert transition == "stopped_for_close_view_before_release"
            assert terminal_status(state) == "stopped"
            state, _ = resume(state)

    assert yielded == [POLICY_VISIBLE_STOP]
    assert terminal_status(state) == "completed"


def test_resuming_does_not_reopen_the_candidate_choice():
    """The stop returns a view, not the decision that led to it."""

    state = _place(stop_before_commit=True)
    state = _run(state, ["transport", "descend"])
    state, _ = record(state, stage=POLICY_VISIBLE_STOP, status="stopped", reason="close_view")

    resumed, _ = resume(state)

    assert resumed["candidate_id"] == "P0"
    assert resumed["sequence"] == state["sequence"]
    assert pending_stage(resumed) == "release"


def test_only_a_stopped_skill_can_be_resumed():
    with pytest.raises(ToolError):
        resume(_place())


def test_an_unconfirmed_mutating_stage_fails_the_skill_closed():
    """Descending from an unconfirmed approach would verify something else."""

    state = _grasp()
    state, transition = record(
        state, stage="approach", status="not_reached", reason="planner_rejected_target",
    )

    assert transition == "approach_unconfirmed_skill_aborted"
    assert terminal_status(state) == "aborted"
    assert pending_stage(state) is None


def test_a_read_only_stage_that_did_not_execute_does_not_abort_the_skill():
    """Verification consumes evidence; failing to measure is not failing to act."""

    state = _run(_grasp(), ["approach", "close"])
    state, transition = record(
        state, stage="verify", status="unavailable",
        reason="visible_surface_evidence_is_insufficient",
    )

    assert transition == "sequence_completed"
    assert terminal_status(state) == "completed"


def test_stages_cannot_be_recorded_out_of_order_or_after_the_end():
    state = _grasp()
    with pytest.raises(ToolError) as wrong:
        record(state, stage="verify", status="executed", reason="ok")
    assert "approach is due" in str(wrong.value)

    finished = _run(state, GRASP_STAGES)
    with pytest.raises(ToolError):
        record(finished, stage="verify", status="executed", reason="ok")


def test_a_skill_can_be_abandoned_before_its_remaining_stages_run():
    state = _run(_place(), ["transport"])
    state, transition = abandon(state, "target_moved_under_the_gripper")

    assert transition == "abandoned:target_moved_under_the_gripper"
    assert terminal_status(state) == "aborted"
    with pytest.raises(ToolError):
        abandon(state, "again")


def test_the_rationale_is_committed_with_the_sequence():
    state = _grasp()
    assert state["rationale"].startswith("G1 is the only candidate")

    with pytest.raises(ToolError) as refused:
        _grasp(rationale="   ")
    assert refused.value.field == "rationale"


def test_a_candidate_id_is_required_because_the_plan_chose_it():
    with pytest.raises(ToolError) as refused:
        _grasp(candidate_id="")
    assert refused.value.field == "candidate_id"


def test_an_unknown_skill_is_refused():
    with pytest.raises(ToolError):
        stages_for("grasp_and_place")


def test_a_tampered_ledger_is_refused_rather_than_trusted():
    state = _run(_grasp(), ["approach"])
    state["completed"][0]["stage"] = "verify"
    with pytest.raises(ToolError) as refused:
        validate_state(state)
    assert refused.value.code == "skill_ledger_state"


def test_a_running_skill_has_no_terminal_status():
    with pytest.raises(ToolError):
        terminal_status(_grasp())


@pytest.mark.parametrize("stage", ["approach", "close", "verify"])
def test_terminated_is_final_and_cannot_resume_or_abandon(stage):
    state = _grasp()
    for preceding in GRASP_STAGES[:GRASP_STAGES.index(stage)]:
        state, _ = record(state, stage=preceding, status="executed", reason="executed")
    state, transition = record(state, stage=stage, status="terminated", reason="environment_terminal")
    assert transition == "environment_terminal"
    assert terminal_status(state) == "terminated"
    assert pending_stage(state) is None
    with pytest.raises(ToolError, match="only a stopped skill"):
        resume(state)
    with pytest.raises(ToolError, match="already ended"):
        abandon(state, "retry")
