"""How a RoboDojo round is allowed to have ended, and what may be pooled.

Upstream RoboDojo ends an episode exactly three ways -- the reward manager's
success latch, its failure latch and the task's native action limit -- and has
no notion of a policy conceding or of a Harness allowance. Any other ending is a truncation
this Harness invented, and pooling one into a rate with a round that reached
an upstream limit silently changes what the rate measures. The whole point of
`round_end_basis` / `protocol_comparable` is to make that impossible to do by
accident, so the table that produces them is worth a table-driven test.

It did not have one. It shipped with three defects that a single pass over the
seven declared bases finds, and they were pinned below rather than fixed:
`benchmarks/robodojo.py` is a member of both content-hashed kernel sets, so
editing it forces a full RoboDojo re-measurement. They are scheduled for the
next forced re-measurement; each pin fails the moment it is fixed, which is
what makes it a schedule rather than an excuse. The EXP11A g004 orchestration
unit fixed the escaped third condition (the failure latch, observed ending
`imitate_sorting_sequence` at 12 of 1600 native actions) and replaced its pin
with the positive contracts below.
"""

from __future__ import annotations

import pytest

from ora.environment.benchmarks import robodojo


def _classify(**kwargs):
    base = {"official_terminal": False, "fault_reason": None, "abort_reason": None}
    return robodojo.classify_round_ending(**{**base, **kwargs})


def test_only_an_upstream_ending_is_protocol_comparable() -> None:
    """The one invariant every reported RoboDojo rate depends on."""

    latched = _classify(
        official_terminal=True, official_success=True,
        native_actions=205, native_action_limit=800,
    )
    assert latched["round_end_basis"] == "official_terminal_transition"
    assert latched["protocol_comparable"] is True
    assert latched["upstream_ending_condition"] == "official_success_latch"

    exhausted = _classify(
        official_terminal=True, official_success=False,
        native_actions=800, native_action_limit=800,
    )
    assert exhausted["protocol_comparable"] is True
    assert exhausted["upstream_ending_condition"] == "native_action_limit"

    conceded = _classify(abort_reason="policy_finished_episode")
    assert conceded["round_end_basis"] == "policy_finished_episode"
    assert conceded["protocol_comparable"] is False
    assert conceded["upstream_ending_condition"] is None


@pytest.mark.parametrize("reason", robodojo.HARNESS_BUDGET_REASONS)
def test_every_harness_budget_stop_is_refused_a_denominator(reason: str) -> None:
    ending = _classify(abort_reason=reason)
    assert ending["round_end_basis"] == "harness_budget_exhausted"
    assert ending["protocol_comparable"] is False


def test_an_unrecognized_ending_fails_closed() -> None:
    """An ending the table does not know is not optimistically comparable."""

    ending = _classify(abort_reason="something_nobody_declared")
    assert ending["round_end_basis"] == "harness_boundary_other"
    assert ending["protocol_comparable"] is False


def test_a_round_with_no_behavioural_result_is_null_not_false() -> None:
    """Null and false are different claims and the distinction is load-bearing.

    A round excluded for infrastructure reasons never produced a behaviour
    sample; calling it "not comparable" would imply it produced a truncated
    one.
    """

    faulted = _classify(fault_reason="robodojo_start_failed")
    assert faulted["round_end_basis"] == "infrastructure_fault"
    assert faulted["protocol_comparable"] is None
    running = _classify()
    assert running["round_end_basis"] == "episode_incomplete"
    assert running["protocol_comparable"] is None


def test_every_declared_basis_is_documented_as_a_closed_roster() -> None:
    assert len(set(robodojo.ROUND_END_BASES)) == len(robodojo.ROUND_END_BASES)
    assert set(robodojo.UPSTREAM_ENDING_CONDITIONS) == {
        "official_success_latch", "official_failure_latch", "native_action_limit",
    }


def test_the_failure_latch_is_a_named_upstream_ending() -> None:
    """A false terminal transition before the limit is the failure latch.

    `RewardManager._mark_env_failed` clears the latch on a task's permanent
    failure predicate and `EvalEnv` ends the episode false; the g003
    `imitate_sorting_sequence` round ended that way at 12 of 1600 native
    actions and was recorded as the unrostered `official_episode_end`. The
    predicate's thresholds stay private: only the condition is named.
    """

    ending = _classify(
        official_terminal=True, official_success=False,
        native_actions=12, native_action_limit=1600,
    )
    assert ending["upstream_ending_condition"] == "official_failure_latch"
    assert ending["upstream_ending_condition"] in robodojo.UPSTREAM_ENDING_CONDITIONS
    assert ending["protocol_comparable"] is True
    # Only unreadable counts leave a false ending unattributed.
    unattributed = _classify(official_terminal=True, official_success=False)
    assert unattributed["upstream_ending_condition"] == "official_episode_end"


@pytest.mark.parametrize("terminal, reason", [
    ({"official_success": True, "native_actions": 205}, "official_success"),
    ({"official_success": False, "native_actions": 800}, "official_native_action_limit"),
    ({"official_success": False, "native_actions": 12}, "official_failure_latch"),
    ({"official_success": False, "native_actions": None},
     "official_native_action_limit_or_failure"),
])
def test_the_environment_status_names_the_same_condition(terminal, reason) -> None:
    assert robodojo.official_terminal_reason(terminal, 800) == reason


# ---------------------------------------------------------------------------
# Pinned defects. Each of these asserts the WRONG behaviour on purpose and
# fails the moment it is fixed, so the fix cannot land silently and the pin
# cannot outlive it. Both live in a file both kernel sets hash.
# ---------------------------------------------------------------------------


def test_pinned_a_declared_budget_reason_is_never_emitted() -> None:
    """`simulation_step_cap` matches no `abort_reason` the backend produces.

    `RobodojoSession` reports the simulation cap as
    `max_sim_steps_per_episode_exhausted` (`backends/robodojo/session.py:959`),
    and `HARNESS_BUDGET_REASONS` lists `simulation_step_cap` instead, so a
    round stopped by that cap classifies as `harness_boundary_other` rather
    than `harness_budget_exhausted`. Both are non-comparable, so no rate is
    wrong today -- but the roster reads as coverage that does not exist, and
    the one ending it was written for is the one it misses.
    """

    assert "simulation_step_cap" in robodojo.HARNESS_BUDGET_REASONS
    emitted = _classify(abort_reason="max_sim_steps_per_episode_exhausted")
    assert emitted["round_end_basis"] == "harness_boundary_other", (
        "the simulation cap now classifies as a budget stop; delete this pin"
    )
    assert emitted["protocol_comparable"] is False
