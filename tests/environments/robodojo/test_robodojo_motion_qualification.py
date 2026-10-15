"""Deterministic layer of the RoboDojo sparse-motion qualification runner."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from ora.contracts.action import validate_runtime_action
from ora.environment.backends.robodojo import session as session_module
from ora.environment.backends.robodojo.motion_qualification import (
    ALL_CASE_IDS,
    MOTION_CASES,
    SCHEMA,
    SEQUENCE_CASES,
    UNREACHABLE_POSITION,
    X5_BASE_MOUNTS_M,
    X5_LINK6_REACH_BOUND_M,
    build_case_arguments,
    build_qualification_record,
    judge_gripper_case,
    judge_motion_case,
    judge_recovery_case,
    run_cases,
    summarize_receipt,
)
from ora.environment.backends.robodojo.common import workspace_violation
from ora.environment.backends.robodojo.session import RobodojoSession
from ora.environment.workspace_profiles import (
    ROBODOJO_ARX_X5_TABLETOP_V1,
    resolve_workspace_profile,
)
from tests.environments.robodojo.test_robodojo_backend import Plant, fake_sdk_candidate_boundary  # noqa: F401


START = {"left": [-0.3, -0.35, 0.92, 0.0, 0.0, 0.707107, 0.707107],
         "right": [0.3, -0.35, 0.92, 0.0, 0.0, 0.707107, 0.707107]}
RECEIPT_SCHEMA = "ora.robodojo-sparse-motion-receipt.v1"


def _receipt(case: Mapping[str, Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """A truthful synthetic receipt for one case's expected happy path."""
    status = sorted(case["expected_status"])[0]
    targets = arguments["targets"]
    arms = {}
    for arm in ("left", "right"):
        targeted = arm in targets
        arms[arm] = {
            "targeted": targeted,
            "pose_error": {"translation_m": 0.002, "orientation_rad": 0.01} if targeted else None,
            "inactive_arm_drift": None if targeted else {"translation_m": 0.0, "orientation_rad": 0.0},
            "segments": [{"segment": name, "reached": True} for name in ("lift", "translate", "descend")]
            if case["action"] == "move_free_space" else [],
        }
    after = {arm: {"pose_xyz_xyzw": (list(targets[arm]["position_m"]) + list(targets[arm]["orientation_xyzw"]))
                   if arm in targets else list(START[arm])} for arm in ("left", "right")}
    return {
        "schema": RECEIPT_SCHEMA, "execution_status": status,
        "motion_strategy": case.get("expected_strategy"), "failure_stage": None,
        "failure_code": case.get("expected_failure_code"),
        "executor": [case["expected_executor"]] if case.get("expected_executor") else None,
        "state_changed": status == "reached", "native_actions": 2 if status == "reached" else 0,
        "arms": arms, "before": {arm: {"pose_xyz_xyzw": list(START[arm])} for arm in START}, "after": after,
        "command_invariant": {"verified": True},
    }


def test_case_arguments_are_valid_v3_actions() -> None:
    for case in MOTION_CASES:
        arguments = build_case_arguments(str(case["case_id"]), START)
        action = validate_runtime_action({
            "schema": "ora-motion-action.v3", "observation_id": "observation-0",
            "action_id": "action-0", "bound_observation_sha256": "0" * 64,
            "action": case["action"], "arguments": arguments,
        })
        expected_arms = {"left", "right"} if case.get("dual_arm") else {"left"}
        assert set(action["arguments"]["targets"]) == expected_arms
    with pytest.raises(ValueError, match="unknown qualification case"):
        build_case_arguments("nope", START)
    assert set(ALL_CASE_IDS) == {case["case_id"] for case in MOTION_CASES} | set(SEQUENCE_CASES)


def test_judges_flag_alias_drift_executor_mismatch_and_inactive_arm_motion() -> None:
    near = next(item for item in MOTION_CASES if item["case_id"] == "orientation_only_near_180deg")
    drifted = _receipt(near, build_case_arguments("orientation_only_near_180deg", START))
    drifted["execution_status"] = "reached"
    drifted["after"]["left"]["pose_xyz_xyzw"][:3] = [-0.3, -0.35, 0.98]
    judgement = judge_motion_case(near, drifted)
    assert judgement["passed"] is False
    assert any("2pi-alias" in note for note in judgement["notes"])
    precise = next(item for item in MOTION_CASES if item["case_id"] == "precision_small_local_6mm")
    wrong = _receipt(precise, build_case_arguments("precision_small_local_6mm", START))
    wrong["executor"] = ["robodojo_sdk_ik_joint_interpolation"]
    wrong["motion_strategy"] = "global"
    judgement = judge_motion_case(precise, wrong)
    assert any("executor" in note for note in judgement["notes"])
    assert any("strategy" in note for note in judgement["notes"])
    translation = next(item for item in MOTION_CASES if item["case_id"] == "free_space_translation_8cm")
    moved = _receipt(translation, build_case_arguments("free_space_translation_8cm", START))
    moved["arms"]["right"]["inactive_arm_drift"] = {"translation_m": 0.02, "orientation_rad": 0.0}
    assert any("while not targeted" in note for note in judge_motion_case(translation, moved)["notes"])
    refusal = next(item for item in MOTION_CASES if item["case_id"] == "contact_approach_large_refused")
    wrote = _receipt(refusal, build_case_arguments("contact_approach_large_refused", START))
    wrote["native_actions"] = 1
    assert any("wrote the world" in note for note in judge_motion_case(refusal, wrote)["notes"])
    assert judge_motion_case(refusal, {"schema": "other"})["passed"] is False
    free = next(item for item in MOTION_CASES if item["case_id"] == "free_space_lift_translate_descend")
    short = _receipt(free, build_case_arguments("free_space_lift_translate_descend", START))
    short["arms"]["left"]["segments"] = [{"segment": "descend", "reached": True}]
    assert any("segments" in note for note in judge_motion_case(free, short)["notes"])
    assert judge_gripper_case("close", {
        "schema": RECEIPT_SCHEMA, "execution_status": "reached", "executor": ["robodojo_sdk_gripper_actuator"],
        "arms": {"left": {"gripper_error": 0.1}}, "command_invariant": {"verified": True},
        "pose_targets_before": {}, "pose_targets_after": {},
    })["notes"] == ["gripper residual 0.1"]
    assert judge_recovery_case({
        "schema": RECEIPT_SCHEMA, "execution_status": "reached", "executor": ["robodojo_sdk_joint_interpolation"],
        "after": {"left": {"pose_xyz_xyzw": [0, 0, 0.9, 0, 0, 0, 1]}},
    }, [0, 0, 0.8, 0, 0, 0, 1])["passed"] is False


def test_the_refusal_target_is_in_envelope_and_out_of_reach() -> None:
    """The two unreachable cases must refuse at the planner, not at the envelope.

    ``_execute_pose_targets`` validates the hard action envelope *before* it
    routes, and ``probe_motion`` raises on an out-of-envelope query, so a
    target outside the live ``robodojo_arx_x5_tabletop_v1`` box would answer
    ``plan_rejected``/``workspace_target_<axis>`` and never ask the SDK IK
    anything.  Pin both halves: inside the envelope, past the X5's reach.
    """

    bounds = resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)["bounds_m"]
    assert workspace_violation(UNREACHABLE_POSITION, bounds) is None
    for axis, index in (("x", 0), ("y", 1), ("z", 2)):
        low, high = bounds[axis]
        assert min(UNREACHABLE_POSITION[index] - low, high - UNREACHABLE_POSITION[index]) >= 0.005
    for base in X5_BASE_MOUNTS_M:
        assert math.dist(UNREACHABLE_POSITION, base) > 1.5 * X5_LINK6_REACH_BOUND_M


def _source_manifest(tmp_path: Path) -> dict[str, Any]:
    """A manifest whose layout file exists, as the round-isolation audit needs."""

    layout = tmp_path / "layout" / "hang_mugs_0.json"
    layout.parent.mkdir(parents=True, exist_ok=True)
    if not layout.is_file():
        layout.write_text('{"objects": []}', encoding="utf-8")
    return {
        "schema": "ora.robodojo-source-manifest.v1",
        "source_revisions": {"robodojo": "1" * 40},
        "layout_file": {
            "path": str(layout),
            "sha256": hashlib.sha256(layout.read_bytes()).hexdigest(),
        },
    }


def test_run_cases_completes_the_full_matrix_against_the_fake_plant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole gate drives the real session and sparse dispatch over the fake SDK."""

    limit = 60
    monkeypatch.setitem(session_module.NATIVE_ACTION_LIMITS, "arrange_largest_number", limit)
    plant = Plant()
    plant.step_lim = limit
    # Drive the matrix under the *live* action envelope the EXP08A resolver
    # hands the worker, from a home link6 pose that sits inside it: the gate
    # once passed 24/24 only because the compatibility runtime's +/-1.5 m box
    # let the refusal cases aim outside any real workspace.
    bounds = resolve_workspace_profile(ROBODOJO_ARX_X5_TABLETOP_V1)["bounds_m"]
    plant.robot_manager.set_home([0.0, -0.35, 0.95, 0.0, 0.0, 0.0])
    # The fake SDK refuses exactly what the real arm cannot reach: a link6
    # target farther than the URDF chain length from both mounted bases.
    plant.robot_manager.unreachable = lambda target: all(
        math.dist([float(value) for value in target[:3]], base) > X5_LINK6_REACH_BOUND_M
        for base in X5_BASE_MOUNTS_M
    )
    session = RobodojoSession(
        config={"workspace_bounds_m": bounds, "action_codec": "motion_action_v3"},
        output_root=tmp_path / "environment",
        environment_factory=lambda config: (
            plant, SimpleNamespace(close=lambda: None), _source_manifest(tmp_path),
        ),
    )
    session._calibration = lambda name: (np.array([[50, 0, 5], [0, 50, 4], [0, 0, 1]]), np.eye(4))
    session.start()
    observation_root = tmp_path / "observations"

    def observe() -> Mapping[str, Any]:
        observation, _images = session.observe(observation_root)
        return {**observation, "observation_sha256": f"{session.observation_sequence:064d}"[:64]}

    def restart_round(index: int) -> Mapping[str, Any]:
        verdict = session.output_root / "official-verdict.json"
        archive = session.output_root / "rounds" / f"round-{index - 1:04d}" / "official-verdict.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        verdict.replace(archive)
        return session.restart_round(index, seed=0)

    results = run_cases(
        observe=observe, execute=session.execute, probe=session.probe_motion, restart_round=restart_round,
        status=session.status, persist_verdict=session.persist_official_verdict,
        current_poses=lambda observation: {
            arm: state["pose_xyz_xyzw"] for arm, state in observation["proprioception"]["end_effectors"].items()
        },
        native_action_limit=limit,
    )
    by_id = {result["case_id"]: result for result in results}
    assert set(by_id) == set(ALL_CASE_IDS)
    failed = {case_id: result["notes"] for case_id, result in by_id.items() if not result["passed"]}
    assert failed == {}
    assert by_id["free_space_lift_translate_descend"]["receipt"]["arms"]["left"]["segments"][0]["segment"] == "lift"
    assert by_id["contact_approach_large_refused"]["receipt"]["native_actions"] == 0
    assert by_id["round_reset_isolation"]["reset"]["round_index"] == 1
    assert by_id["official_terminal_boundary_intact"]["verdict"]["official_success"] is False
    assert by_id["official_terminal_boundary_intact"]["verdict"]["native_actions"] == limit
    assert by_id["official_terminal_boundary_intact"]["receipt"]["round_index"] == 0
    assert by_id["round_reset_isolation"]["receipt"]["round_index"] == 1
    assert session.round_index == 1 and not session.is_terminal
    assert (session.output_root / "rounds/round-0000/official-verdict.json").exists()
    assert (session.output_root / "round-resets/round-0001.json").exists()

    record = build_qualification_record(
        source_commit="a" * 40, host={"execution_host": "test", "cuda_visible_devices": "3"},
        session_identity={"task": "arrange_largest_number", "rounds": 2}, case_results=results,
        evidence={"environment_root": str(session.output_root)},
    )
    assert record["schema"] == SCHEMA and record["passed"] is True and record["missing_cases"] == []
    assert len(record["record_sha256"]) == 64
    json.dumps(record, allow_nan=False)
    partial = build_qualification_record(source_commit="a" * 40, host={}, session_identity={}, case_results=results[:3])
    assert partial["passed"] is False and partial["missing_cases"]
    summary = summarize_receipt(session.last_receipt["receipt"])
    assert "control_trace" not in json.dumps(summary) and summary["schema"] == RECEIPT_SCHEMA


def test_run_cases_records_boundary_errors_instead_of_dropping_cases() -> None:
    calls = {"observe": 0}

    def observe() -> Mapping[str, Any]:
        calls["observe"] += 1
        return {"observation_id": f"o{calls['observe']}", "observation_sha256": "0" * 64,
                "proprioception": {"end_effectors": {arm: {"pose_xyz_xyzw": list(pose)} for arm, pose in START.items()}}}

    def execute(action: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError("worker died")

    results = run_cases(
        observe=observe, execute=execute, probe=lambda request: (_ for _ in ()).throw(RuntimeError("no probe")),
        restart_round=lambda index: (_ for _ in ()).throw(RuntimeError("no reset")),
        status=lambda: {"terminal": False}, persist_verdict=lambda: {"terminal": False, "official_success": None},
        current_poses=lambda observation: {
            arm: state["pose_xyz_xyzw"] for arm, state in observation["proprioception"]["end_effectors"].items()
        },
        native_action_limit=3,
    )
    by_id = {result["case_id"]: result for result in results}
    assert set(by_id) == set(ALL_CASE_IDS)
    assert all(result["passed"] is False for case_id, result in by_id.items()
               if case_id != "official_preterminal_boundary")
    assert by_id["free_space_translation_8cm"]["notes"] == ["execution error: worker died"]
    assert by_id["official_preterminal_boundary"]["passed"] is True
    assert build_qualification_record(source_commit="b" * 40, host={}, session_identity={},
                                      case_results=results)["passed"] is False


def test_rotation_cases_compose_about_the_link6_approach_axis() -> None:
    arguments = build_case_arguments("orientation_only_90deg", {"left": [0, 0, 0.8, 0, 0, 0, 1], "right": START["right"]})
    orientation = arguments["targets"]["left"]["orientation_xyzw"]
    assert orientation == pytest.approx([math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)], abs=1e-6)
    assert arguments["interactions"] == {"left": "free_space"}
