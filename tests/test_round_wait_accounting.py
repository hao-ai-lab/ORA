"""Per-round wait accounting is measured from receipts and changes nothing else.

EXP11A g012 (E11G12-005): with the policy failure exit withheld, 300
``wait_ticks`` calls consumed 5,663 of 13,071 native actions and 280 of them
came after the round's last non-wait proposal, while the round results named
only the budget that ended them. These tests drive the accounting with
synthetic proposal/receipt sequences, the offline event replay with synthetic
``events.jsonl`` rows, and one Synthetic multi-round session end to end.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ora.orchestrator.wait_accounting import (
    ROUND_SCHEMA, SESSION_SCHEMA, RoundWaitAccounting, round_wait_accounting,
    rounds_from_events, session_wait_accounting,
)


def _wait(ticks: int = 20, action_id: str | None = None) -> dict:
    return {"schema": "ora-motion-action.v3", "action": "wait",
            "action_id": action_id or f"wait-{ticks}", "arguments": {"ticks": ticks}}


def _move(action_id: str = "move") -> dict:
    return {"schema": "ora-motion-action.v3", "action": "move_pose", "action_id": action_id,
            "arguments": {"targets": {}}}


def _skill(action_id: str = "skill") -> dict:
    return {"schema": "ora-skill-sequence-action.v1", "action": "execute_skill_sequence",
            "action_id": action_id, "arguments": {"skill": "grasp_execute"}}


def _receipt(native_actions: int | None, *, state_changed: bool = True) -> dict:
    record = {"schema": "ora.motion-receipt.v3", "state_changed": state_changed}
    if native_actions is not None:
        record["native_actions"] = native_actions
    return record


def _skill_receipt(*counts: int) -> dict:
    return {"schema": "ora.manipulation-skill-sequence-receipt.v1",
            "stage_receipts": [{"receipt": _receipt(count)} for count in counts]}


def test_budget_exhaustion_after_consecutive_waits_reports_them_as_terminal():
    """The round ends on its budget; every wait after the last move is terminal."""
    entries = [(_move("m0"), _receipt(18)), (_move("m1"), _receipt(20))]
    # Requested ticks are not what executed: the last wait stops at 5.
    entries += [(_wait(20), _receipt(20)), (_wait(20), _receipt(20)), (_wait(20), _receipt(5))]
    block = round_wait_accounting(entries)
    assert block == {
        "schema": ROUND_SCHEMA, "dispatched_actions": 5, "native_actions": 83,
        "wait_calls": 3, "wait_native_actions": 45,
        "terminal_wait_calls": 3, "terminal_wait_native_actions": 45,
        "last_non_wait_action": {"index": 1, "action_id": "m1", "action": "move_pose"},
    }


def test_waits_interleaved_with_mutations_are_not_terminal():
    entries = [(_wait(), _receipt(20)), (_move("m0"), _receipt(20)), (_wait(), _receipt(20)),
               (_wait(), _receipt(19)), (_skill("s0"), _skill_receipt(20, 15)), (_wait(3), _receipt(3))]
    block = round_wait_accounting(entries)
    assert (block["wait_calls"], block["wait_native_actions"]) == (4, 62)
    assert (block["terminal_wait_calls"], block["terminal_wait_native_actions"]) == (1, 3)
    assert block["last_non_wait_action"] == {"index": 4, "action_id": "s0", "action": "execute_skill_sequence"}
    assert block["native_actions"] == 20 + 20 + 20 + 19 + 35 + 3


def test_a_refused_non_wait_proposal_still_ends_the_terminal_segment():
    """A refusal executed nothing but was an attempt to change the world."""
    entries = [(_wait(), _receipt(20)), (_move("refused"), _receipt(0, state_changed=False)),
               (_wait(), _receipt(20))]
    block = round_wait_accounting(entries)
    assert block["terminal_wait_calls"] == 1 and block["last_non_wait_action"]["action_id"] == "refused"


def test_a_round_of_only_waits_is_all_terminal_and_an_empty_round_is_zero():
    only = round_wait_accounting([(_wait(), _receipt(20)), (_wait(), _receipt(20))])
    assert only["terminal_wait_calls"] == only["wait_calls"] == 2
    assert only["last_non_wait_action"] is None
    empty = round_wait_accounting([])
    assert empty == {"schema": ROUND_SCHEMA, "dispatched_actions": 0, "native_actions": 0, "wait_calls": 0,
                     "wait_native_actions": 0, "terminal_wait_calls": 0, "terminal_wait_native_actions": 0,
                     "last_non_wait_action": None}


def test_an_uncounted_receipt_makes_only_its_sums_unknown():
    entries = [(_move(), _receipt(None)), (_wait(), _receipt(20)), (_wait(), _receipt(None))]
    block = round_wait_accounting(entries)
    assert block["native_actions"] is None
    assert block["wait_calls"] == block["terminal_wait_calls"] == 2
    assert block["wait_native_actions"] is None and block["terminal_wait_native_actions"] is None
    counted_waits = round_wait_accounting([(_move(), _receipt(None)), (_wait(), _receipt(7))])
    assert counted_waits["native_actions"] is None and counted_waits["wait_native_actions"] == 7


def test_the_live_accumulator_matches_the_pure_summary_and_resets_per_round():
    accounting = RoundWaitAccounting()
    entries = [(_move("m"), _receipt(20)), (_wait(), _receipt(12))]
    for action, receipt in entries:
        accounting.record(action, receipt)
    snapshot = accounting.snapshot()
    assert snapshot == round_wait_accounting(entries)
    accounting.record(_wait(), _receipt(1))
    assert snapshot["wait_calls"] == 1, "a snapshot is not a live view"
    accounting.reset()
    assert accounting.snapshot() == round_wait_accounting([])


def test_the_supervisor_ledger_keeps_the_allowance_and_resets_both_per_round():
    from ora.orchestrator.wait_accounting import RoundActionLedger

    plan = {"runtime": {"kind": "robodojo", "native_action_limit": 550}, "budgets": {"max_sim_steps": 5500}}
    ledger = RoundActionLedger(plan)
    ledger.record(_receipt(20), _move("m"))
    ledger.record(_receipt(7), _wait())
    assert ledger.fields() == {"native_actions_limit": 550, "native_actions_remaining": 523}
    assert ledger.wait_accounting()["terminal_wait_native_actions"] == 7
    ledger.reset()
    assert ledger.fields()["native_actions_remaining"] == 550
    assert ledger.wait_accounting() == round_wait_accounting([])


def test_the_run_result_takes_the_session_totals_or_its_one_round_block():
    from ora.orchestrator.round_aggregates import run_round_aggregates

    block = round_wait_accounting([(_move(), _receipt(20)), (_wait(), _receipt(20))])
    rounds = [{"official_success": True, "wait_accounting": block},
              {"official_success": False, "wait_accounting": block}]
    session = run_round_aggregates(rounds, multi_round=True)
    assert session == {"round_count": 2, "round_success_count": 1,
                       "wait_accounting": session_wait_accounting(rounds)}
    assert run_round_aggregates([{"official_success": None}], multi_round=True) == {
        "round_count": 1, "round_success_count": 0}
    assert run_round_aggregates(rounds[:1], multi_round=False) == {"wait_accounting": block}
    assert run_round_aggregates([], multi_round=False) == {}


def _event_rows(rounds: list[list[tuple[dict, dict]]], *, read_only_between: bool = True) -> list[dict]:
    rows: list[dict] = [{"type": "run_started"}]
    for index, entries in enumerate(rounds):
        rows.append({"type": "reset", "record": {"seed": index}})
        for action, receipt in entries:
            if read_only_between:
                # Read-only tool calls and tool-less answers dispatch nothing.
                rows.append({"type": "tool_result", "tool": "observe", "result": {}})
            rows.append({"type": "action_submitted", "action": action})
            rows.append({"type": "observation"})
            rows.append({"type": "motion_receipt", "record": receipt, "consumed_decision": True})
        rows.append({"type": "tool_result", "tool": "check_task", "result": {}})
        rows.append({"type": "round_finished", "round_index": index, "result": {}})
    rows.append({"type": "run_finished", "result": {}})
    return rows


def test_the_event_replay_reproduces_each_round_from_actions_and_receipts():
    rounds = [
        [(_move("a"), _receipt(20)), (_wait(), _receipt(20)), (_wait(), _receipt(4))],
        [(_wait(), _receipt(20)), (_move("b"), _receipt(17))],
        [],
    ]
    replayed = rounds_from_events(_event_rows(rounds))
    assert replayed == [round_wait_accounting(entries) for entries in rounds]
    assert [block["terminal_wait_calls"] for block in replayed] == [2, 0, 0]


def test_the_event_replay_keeps_an_unfinished_tail_out_of_the_finished_rounds():
    rows = _event_rows([[(_move(), _receipt(20))]])[:-1]
    rows += [{"type": "action_submitted", "action": _wait()}, {"type": "motion_receipt", "record": _receipt(20)}]
    assert len(rounds_from_events(rows)) == 1


def test_session_totals_sum_rounds_and_stay_unknown_when_a_round_is():
    first = round_wait_accounting([(_move(), _receipt(20)), (_wait(), _receipt(20))])
    second = round_wait_accounting([(_wait(), _receipt(15)), (_wait(), _receipt(None))])
    total = session_wait_accounting([{"wait_accounting": first}, {"wait_accounting": second}])
    assert total == {"schema": SESSION_SCHEMA, "rounds": 2, "dispatched_actions": 4, "native_actions": None,
                     "wait_calls": 3, "wait_native_actions": None, "terminal_wait_calls": 3,
                     "terminal_wait_native_actions": None}
    assert session_wait_accounting([{"round_index": 0}]) is None
    assert session_wait_accounting([]) is None


# --------------------------------------------------------------------------
# End to end: a Synthetic persistent session records the block per round and
# the session total, and evaluation reconciles them like decisions.
# --------------------------------------------------------------------------


def test_a_multi_round_session_persists_the_accounting_and_evaluation_reconciles_it(tmp_path: Path):
    import test_multi_round_session as session_tests
    from ora.environment import SyntheticSession
    from ora.orchestrator.coordinator import run_episode
    from ora.orchestrator.evaluation import verify_multi_round_session
    from ora.orchestrator.spec import resolve_run_spec

    spec = session_tests._spec()
    spec["session"] = {"cadence": "persistent_agent_k_rounds_v1", "rounds": 2}
    plan = resolve_run_spec(spec, root=session_tests.ROOT)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")

    def environment_factory(**kwargs):
        return SyntheticSession(kwargs["output_root"], kwargs["task"], kwargs["seed"], max_actions=100,
                                main_camera="agentview", camera_names=("agentview", "wrist"))

    result = run_episode(run_root=run_root, plan=plan, run_id="wait-accounting-qualification",
                         provider_override=session_tests._MutationEveryRoundProvider(),
                         environment_session_factory=environment_factory)
    events = [json.loads(line) for line in (run_root / "events.jsonl").read_text().splitlines()]
    # Measurement only: the budget still ends each round exactly as before.
    assert [item["termination"] for item in result["rounds"]] == ["robot_decision_budget"] * 2
    for index, record in enumerate(result["rounds"]):
        block = record["wait_accounting"]
        assert block["schema"] == ROUND_SCHEMA and block["dispatched_actions"] == 1
        assert block["wait_calls"] == block["terminal_wait_calls"] == 0
        assert block["last_non_wait_action"]["index"] == 0
        persisted = json.loads((run_root / "rounds" / f"round-{index:04d}" / "result.json").read_text())
        assert persisted["wait_accounting"] == block
    assert rounds_from_events(events) == [record["wait_accounting"] for record in result["rounds"]]
    assert result["wait_accounting"] == session_wait_accounting(result["rounds"])
    assert result["wait_accounting"]["dispatched_actions"] == 2
    assert "multi_round_wait_accounting_mismatch" not in verify_multi_round_session(run_root, plan, result, events)
    miscounted = copy.deepcopy(result)
    miscounted["rounds"][1]["wait_accounting"]["wait_calls"] = 3
    assert "multi_round_wait_accounting_mismatch" in verify_multi_round_session(
        run_root, plan, miscounted, events)
    # A run recorded before the accounting existed carries it nowhere and reconciles.
    historical = copy.deepcopy(result)
    del historical["wait_accounting"]
    for record in historical["rounds"]:
        del record["wait_accounting"]
    assert "multi_round_wait_accounting_mismatch" not in verify_multi_round_session(
        run_root, plan, historical, events)


@pytest.mark.parametrize("missing", ["session", "round"])
def test_accounting_present_on_only_one_side_is_a_mismatch(tmp_path: Path, missing: str):
    from ora.orchestrator.evaluation import _wait_accounting_errors

    block = round_wait_accounting([(_wait(), _receipt(20))])
    rounds = [{"wait_accounting": block}, {"wait_accounting": copy.deepcopy(block)}]
    result = {"rounds": rounds, "wait_accounting": session_wait_accounting(rounds)}
    assert _wait_accounting_errors(result) == []
    if missing == "session":
        del result["wait_accounting"]
    else:
        del rounds[0]["wait_accounting"]
    assert _wait_accounting_errors(result) == ["multi_round_wait_accounting_mismatch"]
