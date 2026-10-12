"""Evaluation of a scheduled persistent RoboDojo session, and the legacy shape it must not change.

A scheduled round is a new episode on its planned layout in a relaunched
simulation: per-round identity, v2/v3 records, the scheduled isolation proof,
a physics clock that restarts with every epoch, and a round result that names
its scene. The g003-g007 recorded shape (``_multi_round_evidence``) keeps
evaluating exactly as before.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.orchestrator import robodojo_layout_evaluation as layouts
from ora.orchestrator.evaluation import _robodojo_evidence_errors
from ora.orchestrator.robodojo_round_scene import (
    LAYOUT_CHANGE_SENTENCE, layout_change_sentence, require_round_scene, round_scene_field,
)
from ora.orchestrator.episode_bindings import SupervisorError
from test_robodojo_runtime import _multi_round_evidence, _plan, _round_isolation, _sha, _source_record, _write

LAYOUT_IDS = (21, 37, 51)
RESTART_CLOCK = 320


def _entry(ordinal: int) -> dict:
    return {"layout_id": ordinal, "layout_file": f"Eval_Layout/RoboDojo/arx_x5/0/fixture_task_{ordinal}.json",
            "layout_sha256": f"{ordinal:02d}" * 32}


def _schedule() -> dict:
    return {"schema": "ora.robodojo-layout-schedule.v1", "eval_seed": 0, "env_seed_policy": "layout_id",
            "rounds": [_entry(ordinal) for ordinal in LAYOUT_IDS],
            "selection": {"catalog": "ora/environment/benchmarks/robodojo-layout-pools-v1.json",
                          "catalog_sha256": "c" * 64}}


def _identity(index: int) -> dict:
    entry = _entry(LAYOUT_IDS[index])
    return {**entry, "env_seed": entry["layout_id"], "physics_clock_epoch": index}


def _verification(index: int) -> dict:
    """Round 0's measured proof that the loaded scene is its scheduled layout."""
    entry = _entry(LAYOUT_IDS[index])
    return {"schema": layouts.LAYOUT_VERIFICATION_SCHEMA, "passed": True,
            "checks": dict.fromkeys(layouts.LOADED_LAYOUT_CHECKS, True),
            "measured": {"layout_id": entry["layout_id"], "layout_sha256": entry["layout_sha256"],
                         "layout_sha256_planned": entry["layout_sha256"], "env_seed": entry["layout_id"]}}


#: Each round's own simulated time at close: its reset clock plus 20 physics steps.
CLOSE_TIMES = [(500 + 20) * 0.004, (RESTART_CLOCK + 20) * 0.004]


def _offsets() -> list[float]:
    """The annotator offset of each epoch: every earlier epoch's time at close."""
    offsets = [0.0]
    for closed in CLOSE_TIMES:
        offsets.append(offsets[-1] + closed)
    return offsets


def _relaunch(index: int, clock: int) -> dict:
    """A relaunch record whose measured stage, clock and time fields prove the new scene."""
    return {"schema": layouts.RELAUNCH_SCHEMA, "layout_id": LAYOUT_IDS[index],
            "previous_layout_id": LAYOUT_IDS[index - 1], "sim_closed": True, "stage_replaced": True,
            "stage_before": {"stage": "1:anon:a", "env_prim_present": True},
            "stage_after_close": {"stage": "2:anon:b", "env_prim_present": False},
            "physics_clock_before_close": 900, "physics_clock_after_reset": clock,
            "simulation_time_before_close_s": CLOSE_TIMES[index - 1],
            "previous_simulation_time_offset_s": _offsets()[index - 1],
            "simulation_time_offset_s": _offsets()[index], "simulation_time_after_reset_s": clock * 0.004}


def _synchronizations(*, with_offset: bool = True) -> list[dict]:
    """One matched camera synchronization per epoch: annotator = offset + context clock."""
    rows = []
    for index, clock in enumerate((500, RESTART_CLOCK, RESTART_CLOCK)):
        offset = _offsets()[index]
        row = {"schema": "ora.robodojo-render-synchronization.v1", "physics_step": clock,
               "simulation_time_s": clock * 0.004, "matched": True,
               "samples": [{"render_call": 2, "cameras": [{"simulation_time_s": offset + clock * 0.004}] * 3}]}
        if with_offset:
            row["simulation_time_offset_s"] = offset
        rows.append(row)
    return rows


def _scheduled_isolation(index: int, clock: int) -> dict:
    proof = _round_isolation(clock)
    proof["checks"] = dict.fromkeys(layouts.SCHEDULED_ISOLATION_CHECKS, True)
    measured = {key: value for key, value in proof["measured"].items()
                if key not in {"layout_sha256", "layout_sha256_at_start"}}
    entry = _entry(LAYOUT_IDS[index])
    proof["measured"] = {**measured, "layout_id": entry["layout_id"], "layout_file": entry["layout_file"],
                         "layout_sha256": entry["layout_sha256"], "layout_sha256_planned": entry["layout_sha256"],
                         "env_seed": entry["layout_id"], "physics_clock_epoch": index}
    return proof


def _scheduled_evidence(tmp_path: Path, perturbation: str | None = None):
    """One finished scheduled three-round session: 21 -> 37 -> 51, each later round relaunched."""

    run_root = tmp_path / "run"
    native = run_root / "native"
    plan = _plan()
    schedule = _schedule()
    first = schedule["rounds"][0]
    plan["runtime"].update(layout_id=21, layout_file=first["layout_file"], layout_sha256=first["layout_sha256"],
                           layout_schedule=schedule)
    plan["session"] = {"cadence": "persistent_agent_k_rounds_v1", "rounds": 3,
                       "budget_scope": "per_round", "reset_owner": "supervisor"}
    source = {**_source_record(), "layout_id": 21,
              "layout_file": {"path": "/fixture/Assets/" + first["layout_file"], "sha256": first["layout_sha256"]}}
    if perturbation != "source_unhashed":
        source["layout_schedule"] = [{"layout_id": item["layout_id"], "path": "/fixture/Assets/" + item["layout_file"],
                                      "sha256": item["layout_sha256"]} for item in schedule["rounds"]]
    _write(native / "source-manifest.json", source)
    source_hash = _sha(native / "source-manifest.json")
    events: list[dict] = [{"type": "run_started", "run_id": "robodojo-scheduled"}]
    round_records: list[dict] = []
    sequence = 0
    for index in range(3):
        clock = 500 if index == 0 else RESTART_CLOCK
        identity = _identity(index)
        if perturbation == "epoch_not_advanced" and index == 1:
            identity["physics_clock_epoch"] = 0
        reset = {"task": plan["task"], "eval_seed": 0, "seed": 0, **identity, "round_index": index,
                 "reward_registered": True, "physics_steps_before_policy": clock,
                 "simulation_time_offset_s": _offsets()[index],
                 "source_manifest": {"path": "source-manifest.json", "sha256": source_hash}}
        if index == 0:
            reset.update(schema=layouts.SCHEDULED_START_RESET_SCHEMA, scene_reset_mode="initial_launch",
                         layout_verification=_verification(0))
            if perturbation == "round0_unverified":
                reset["layout_verification"]["checks"]["layout_matches_round_schedule"] = False
        else:
            reset.update(
                schema=layouts.SCHEDULED_ROUND_RESET_SCHEMA, scene_reset_mode="simulation_relaunch",
                relaunch=_relaunch(index, clock),
                artifact_sequences_preserved={"observation_sequence": sequence, "action_sequence": sequence,
                                              "motion_probe_sequence": 0},
                round_isolation=_scheduled_isolation(index, clock))
            if perturbation == "wrong_round_layout" and index == 2:
                reset.update(_entry(37), env_seed=37)
            if perturbation == "legacy_isolation" and index == 1:
                reset["round_isolation"] = _round_isolation(clock)
            if perturbation == "old_scene_survived" and index == 1:
                reset["relaunch"]["stage_after_close"]["env_prim_present"] = True
            if perturbation == "relaunch_clock_mismatch" and index == 2:
                reset["relaunch"]["physics_clock_after_reset"] = clock + 1
            if perturbation == "offset_not_chained" and index == 2:
                reset["relaunch"]["simulation_time_before_close_s"] += 0.5
            _write(native / "round-resets" / f"round-{index:04d}.json", reset)
        events.append({"type": "reset", "record": reset, "round_index": index})
        for step in range(2):
            action_id = f"round-{index:04d}-action-{step:04d}"
            journal = native / "controller" / f"action-{sequence:04d}.jsonl"
            journal.parent.mkdir(parents=True, exist_ok=True)
            journal.write_text("".join(json.dumps({
                "action_id": action_id, "segment": "command",
                "physics_clock": clock + step * 10 + offset + 1, "native_action": step + 1,
            }, sort_keys=True) + "\n" for offset in range(10)))
            control = native / "controller" / f"action-{sequence:04d}.json"
            _write(control, {"schema": "ora.robodojo-controller-trace.v1", "action_id": action_id,
                             "samples": [{"native_action": step + 1, "physics_steps": (step + 1) * 10}],
                             "physics_journal": {"path": journal.relative_to(native).as_posix(),
                                                 "sha256": _sha(journal)}})
            receipt = {"schema": "ora.robodojo-sparse-motion-receipt.v1", "action_id": action_id,
                       "action": "move_pose", "round_index": index, "consumed": True, "state_changed": True,
                       "native_actions": 1, "physics_steps": 10,
                       "control_trace": {"path": control.relative_to(native).as_posix(), "sha256": _sha(control)}}
            receipt_path = native / "receipts" / f"action-{sequence:04d}.json"
            _write(receipt_path, receipt)
            events.append({"type": "action_submitted", "action": {"action_id": action_id}})
            events.append({"type": "motion_receipt", "record": {
                "path": receipt_path.relative_to(native).as_posix(), "sha256": _sha(receipt_path),
                "receipt": receipt}})
            sequence += 1
        verdict = {"schema": layouts.SCHEDULED_VERDICT_SCHEMA, "source": "RoboDojo EvalEnv.is_episode_end",
                   "source_manifest_sha256": source_hash, "task": plan["task"], "eval_seed": 0, "seed": 0,
                   **identity, "round_index": index, "official_success": index == 2, "terminal": True,
                   "abort_reason": None, "verdict_basis": "official_terminal_transition", "fault_reason": None,
                   "native_actions": 2, "physics_steps": 20, "reset_physics_steps": clock}
        if perturbation == "verdict_v2" and index == 1:
            verdict["schema"] = "ora.robodojo-official-verdict.v2"
        _write(native / "rounds" / f"round-{index:04d}" / "official-verdict.json", verdict)
        if index == 2:
            _write(native / "official-verdict.json", verdict)
        record = {"schema": "ora.round-result.v1", "round_index": index,
                  "state": "completed", "termination_authority": "environment",
                  "official_success": verdict["official_success"], "verdict": verdict,
                  **round_scene_field(plan, reset)}
        if perturbation == "scene_missing" and index == 1:
            record.pop("scene")
        round_records.append(record)
        events.append({"type": "round_finished", "round_index": index})
    (native / "render-synchronization.jsonl").write_text("".join(
        json.dumps(row) + "\n" for row in _synchronizations(with_offset=perturbation != "render_offset_missing")))
    result = {"schema": "ora.session-result.v1", "state": "completed",
              "official_success": None, "verdict": None, "termination": "session_round_limit",
              "rounds": round_records, "round_count": 3,
              "round_success_count": sum(item["official_success"] is True for item in round_records)}
    return run_root, plan, result, events


@pytest.mark.parametrize(("perturbation", "expected"), [
    (None, []),
    ("wrong_round_layout", ["robodojo_round_reset_invalid:2"]),
    ("legacy_isolation", ["robodojo_round_isolation_invalid:1"]),
    ("epoch_not_advanced", ["robodojo_official_verdict_binding_invalid:1", "robodojo_round_reset_invalid:1"]),
    ("source_unhashed", ["robodojo_layout_schedule_binding_invalid"]),
    ("scene_missing", ["robodojo_round_scene_invalid:1"]),
    ("verdict_v2", ["robodojo_official_verdict_binding_invalid:1"]),
    ("round0_unverified", ["robodojo_layout_identity_invalid"]),
    ("old_scene_survived", ["robodojo_round_reset_invalid:1"]),
    ("relaunch_clock_mismatch", ["robodojo_round_reset_invalid:2"]),
    ("offset_not_chained", ["robodojo_round_reset_invalid:2"]),
    ("render_offset_missing", ["robodojo_render_time_offset_invalid"]),
])
def test_a_scheduled_session_joins_each_round_to_its_own_layout_and_epoch(tmp_path, perturbation, expected):
    run_root, plan, result, events = _scheduled_evidence(tmp_path, perturbation)
    assert sorted(set(_robodojo_evidence_errors(run_root, plan, result, events))) == expected


def test_the_recorded_unscheduled_shape_still_evaluates_and_is_not_a_scheduled_one(tmp_path):
    """g003-g007 sealed evidence: layout 0 every round, v1/v2 records, one clock epoch."""

    run_root, plan, result, events = _multi_round_evidence(tmp_path)
    assert _robodojo_evidence_errors(run_root, plan, result, events) == []
    plan["runtime"]["layout_schedule"] = _schedule()
    errors = set(_robodojo_evidence_errors(run_root, plan, result, events))
    assert {"robodojo_source_manifest_binding_invalid", "robodojo_layout_schedule_binding_invalid",
            "robodojo_round_reset_invalid:1", "robodojo_round_isolation_invalid:1"} <= errors


def test_the_unscheduled_clock_never_goes_back_across_rounds(tmp_path):
    run_root, plan, result, events = _multi_round_evidence(tmp_path)
    resets = [event["record"] for event in events if event["type"] == "reset"]
    # Unscheduled: always one simulation context, whatever an epoch field might say.
    stray = [{**record, "physics_clock_epoch": index} for index, record in enumerate(resets)]
    assert all(layouts.clock_continuity_applies(plan, before, after) for before, after in zip(stray, stray[1:]))
    scheduled = {**plan, "runtime": {**plan["runtime"], "layout_schedule": _schedule()}}
    assert not layouts.clock_continuity_applies(scheduled, stray[0], stray[1])
    assert layouts.clock_continuity_applies(scheduled, resets[0], resets[1])


def test_the_supervisor_compares_each_reset_with_the_planned_layout():
    plan = _plan()
    plan["runtime"]["layout_schedule"] = _schedule()
    for index in range(3):
        require_round_scene(plan, {**_identity(index)}, index)
    with pytest.raises(SupervisorError, match="layout drift"):
        require_round_scene(plan, {**_identity(1), "layout_id": 21}, 2)
    with pytest.raises(SupervisorError, match="outside"):
        require_round_scene(plan, _identity(0), 3)
    require_round_scene(_plan(), {"layout_id": 0}, 1)  # unscheduled: nothing to compare
    assert round_scene_field(_plan(), {"layout_id": 0}) == {}
    scene = round_scene_field(plan, {**_identity(1), "scene_reset_mode": "simulation_relaunch"})["scene"]
    assert scene == {**_identity(1), "scene_reset_mode": "simulation_relaunch"}


def test_the_round_notice_adds_the_neutral_sentence_only_when_the_layout_changes():
    plan = _plan()
    assert layout_change_sentence(plan, 1) == ""  # unscheduled
    plan["runtime"]["layout_schedule"] = _schedule()
    assert layout_change_sentence(plan, 0) == ""  # round 0 has no earlier round
    assert layout_change_sentence(plan, 1) == layout_change_sentence(plan, 2) == LAYOUT_CHANGE_SENTENCE
    plan["runtime"]["layout_schedule"]["rounds"][2]["layout_sha256"] = _entry(21)["layout_sha256"]
    assert layout_change_sentence(plan, 2) == ""
    # No id, position or count reaches the policy.
    assert not any(character.isdigit() for character in LAYOUT_CHANGE_SENTENCE)
    assert "layout" not in LAYOUT_CHANGE_SENTENCE.lower()


def _reach_record(arms: list[str]) -> dict:
    """A smoke object record: the centre and eight footprint points, each with the arms that reached it."""
    points = [{"point": name, "reachable_arms": list(arms)} for name in ("centre", *(f"edge{i}" for i in range(8)))]
    return {"reachable_arms": list(arms), "points": points}


def _smoke_report() -> tuple[dict, dict, dict]:
    """A passing scheduled scene-smoke report, its runtime and its worker source manifest."""
    from ora.environment.benchmarks.robodojo import build_scene_registration
    from ora.orchestrator.robodojo_layout_revalidation import SCHEDULE_CHECKS

    runtime = {"layout_schedule": _schedule(), "native_action_limit": 800}
    registration = build_scene_registration(
        {"bowl_4": "rigid", "bowl_5": "rigid", "bowl_6": "rigid", "camera_stand_7": "geometry"},
        evaluator_registration={"check_list": True, "final_check_list": False, "trigger_check_list": False},
        native_action_limit=800)
    sensors = {name: {"shape": [480, 640], "finite_positive_fraction": 0.9}
               for name in ("head", "left_wrist", "right_wrist")}
    reach = {f"bowl{index}": _reach_record(["right"]) for index in range(3)}
    rounds = []
    for index in (1, 2):
        rounds.append({
            "round_index": index, "layout_id": LAYOUT_IDS[index], "worker_gpu_memory_mib": 9100,
            "restart": {"schema": layouts.SCHEDULED_ROUND_RESET_SCHEMA, "round_index": index, **_identity(index),
                        "scene_reset_mode": "simulation_relaunch", "scene_registration": registration,
                        "physics_steps_before_policy": RESTART_CLOCK, "relaunch": _relaunch(index, RESTART_CLOCK),
                        "simulation_time_offset_s": _offsets()[index],
                        "round_isolation": {"passed": True,
                                            "checks": dict.fromkeys(layouts.SCHEDULED_ISOLATION_CHECKS, True)}},
            "sensors": dict(sensors), "object_reach": dict(reach),
            "alignment": {arm: {"candidates": [{"reachable": True}]} for arm in ("left", "right")},
        })
    report = {"layout_schedule": list(LAYOUT_IDS), "checks": dict.fromkeys(SCHEDULE_CHECKS, True),
              "reset": {"schema": layouts.SCHEDULED_START_RESET_SCHEMA, **_identity(0),
                        "scene_reset_mode": "initial_launch", "layout_verification": _verification(0),
                        "simulation_time_offset_s": 0.0},
              "round0_object_reach": dict(reach), "rounds": rounds}
    source = {"layout_schedule": [{"layout_id": entry["layout_id"], "path": "/assets/" + entry["layout_file"],
                                   "sha256": entry["layout_sha256"]} for entry in _schedule()["rounds"]]}
    return runtime, report, source


@pytest.mark.parametrize("damage, message", [
    (None, None),
    (lambda report, source: report.update(layout_schedule=[21, 37]), "another layout schedule"),
    (lambda report, source: report["checks"].update(scheduled_object_reach=False), "checks incomplete"),
    (lambda report, source: source["layout_schedule"].pop(), "scheduled layout files"),
    (lambda report, source: report["reset"].update(layout_id=0), "round 0"),
    (lambda report, source: report["rounds"][1]["restart"].update(layout_id=37), "round 2 did not relaunch"),
    (lambda report, source: report["rounds"][0]["restart"]["round_isolation"]["checks"].update(
        simulation_relaunched=False), "round 1 isolation"),
    (lambda report, source: report["rounds"][0]["object_reach"]["bowl1"].update(reachable_arms=[]),
     "no reachable top-down"),
    # A centre-only record (a smoke from before footprint reach) cannot certify a lane.
    (lambda report, source: report["rounds"][1]["object_reach"].update(bowl2={"reachable_arms": ["right"]}),
     "round 2 layout objects have no reachable top-down approach over their whole footprint"),
    # The recorded arm must reach every point: a far rim point it missed refutes the summary.
    (lambda report, source: report["rounds"][1]["object_reach"].update(
        bowl2={**_reach_record(["right"]), "points": [*_reach_record(["right"])["points"][:-1],
                                                      {"point": "edge7", "reachable_arms": ["left"]}]}),
     "round 2 layout objects have no reachable top-down"),
    (lambda report, source: report.update(round0_object_reach={}), "round-0 layout objects"),
    (lambda report, source: report["rounds"].pop(), "scheduled rounds missing"),
    (lambda report, source: report["reset"]["layout_verification"].update(passed=False), "round 0"),
    (lambda report, source: report["rounds"][0]["restart"]["relaunch"]["stage_before"].update(env_prim_present=False),
     "round 1 did not relaunch"),
])
def test_a_lane_gate_certifies_only_a_smoke_that_passed_this_schedule(damage, message):
    from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError
    from ora.orchestrator.robodojo_layout_revalidation import validate_scheduled_scene

    runtime, report, source = _smoke_report()
    if damage is None:
        view = validate_scheduled_scene(runtime, report, source, _synchronizations())
        assert view == {"layout_schedule": list(LAYOUT_IDS), "worker_gpu_memory_mib_after_relaunch": [9100, 9100]}
        return
    damage(report, source)
    with pytest.raises(RoboDojoBenchmarkError, match=message):
        validate_scheduled_scene(runtime, report, source, _synchronizations())


def test_a_lane_gate_refuses_cameras_that_did_not_match_on_their_epoch_offset():
    from ora.environment.benchmarks.robodojo import RoboDojoBenchmarkError
    from ora.orchestrator.robodojo_layout_revalidation import validate_scheduled_scene

    runtime, report, source = _smoke_report()
    for synchronizations in (_synchronizations(with_offset=False), []):
        with pytest.raises(RoboDojoBenchmarkError, match="simulation-time offset"):
            validate_scheduled_scene(runtime, report, source, synchronizations)


def test_host_evaluation_and_worker_name_the_same_scheduled_records():
    from ora.environment.backends.robodojo import layout_rounds as worker
    from ora.environment.benchmarks import robodojo_layouts as schedule

    assert layouts.SCHEDULED_ISOLATION_CHECKS == worker.SCHEDULED_ISOLATION_CHECKS
    assert layouts.SCHEDULED_START_RESET_SCHEMA == worker.START_RESET_V2_SCHEMA
    assert layouts.SCHEDULED_ROUND_RESET_SCHEMA == worker.ROUND_RESET_V2_SCHEMA
    assert layouts.SCHEDULED_VERDICT_SCHEMA == worker.VERDICT_V3_SCHEMA
    assert layouts.RELAUNCH_SCHEMA == worker.RELAUNCH_SCHEMA
    assert (layouts.SCENE_RESET_INITIAL, layouts.SCENE_RESET_RELAUNCH) == (
        schedule.SCENE_RESET_INITIAL, schedule.SCENE_RESET_RELAUNCH)
