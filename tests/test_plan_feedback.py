"""Rejected candidate causes survive Context without leaking private details."""
import copy
import json

import pytest

from ora.agent.evidence.plan_feedback import public_plan_validation
from ora.agent.native.receipt_projection import (
    ReceiptProjectionContext, resolve_receipt_effect_projector,
)
from ora.agent.context import NativeContextPipeline
from ora.agent.evidence.trajectory import ArtifactStore, _forbidden_path
from ora.agent.providers.native import translate_anthropic_request
from test_lossless_context_runtime import _pipeline as _lossless_pipeline
from test_receipt_projection_failed_arms import _receipt


#: Harness metadata of the observation message; the lossless retention
#: projects only ActionSteps that open with one.
OBSERVATION_METADATA = {"_observation": {"step": 16}, "_session_round": 1}


def _pipeline(tmp_path, retention):
    """The two live retentions: lossless (default) and rewrite."""
    if retention == "action_step_lossless_v4":
        return _lossless_pipeline(tmp_path)
    return NativeContextPipeline("native_causal_context_v1", ArtifactStore(tmp_path),
                                 100_000, 4096, retention=retention)


def validation():
    # Structure minimized from RANDOM-B2-003 receipt295: candidate0 fails
    # planning; candidate1 has planner Success but fails workspace admission.
    return {"status": "rejected", "code": "topk_no_admitted_plan", "details": {
        "private": "SECRET", "candidates": [
            {"candidate_index": 0, "admission": "planner_failed", "status": "Fail",
             "error": "SECRET", "failure_stages": [{"message": "SECRET"}]},
            {"candidate_index": 1, "admission": "plan_rejected", "status": "Success",
             "rejection": {"code": "workspace_path", "details": {"object_gt": "SECRET"},
                           "message": "SECRET"}},
        ]}}


@pytest.mark.parametrize("retention", ["action_step_lossless_v4", "rewrite_compact_v1"])
def test_mixed_candidate_rejection_reaches_both_wire_protocols(tmp_path, retention):
    raw = _receipt(target_arms=["right"], statuses={"right": "Success"})
    raw.update(failure_stage="plan_validation", failure_code="topk_no_admitted_plan",
               execution_status="plan_rejected", state_changed=False, steps_used=0)
    raw["arms"]["right"]["plan_validation"] = validation()
    original = copy.deepcopy(raw)
    effect = resolve_receipt_effect_projector(raw["schema"])(raw, ReceiptProjectionContext(
        "action-7", "move_pose", surface_supports=lambda _: True))
    text = json.dumps(effect)
    assert "workspace_path" in text and "planner_failed" in text
    assert "not an endpoint reachability verdict" in text
    assert "SECRET" not in text and "object_gt" not in text
    assert raw == original
    pipeline = _pipeline(tmp_path, retention)
    message = {"role": "user", "content": [{"type": "text", "text": json.dumps({
        "current_actual": {"actual_sequence": 16}, "last_action_effect": effect})}],
        **OBSERVATION_METADATA}
    projected = pipeline.project(messages=[message], tools=[], observation_step=16)
    request = {"model": "offline", "max_tokens": 64, "messages": projected["messages"]}
    for wire in (request, translate_anthropic_request(request)):
        assert "workspace_path" in json.dumps(wire)
        assert "SECRET" not in json.dumps(wire)
        assert _forbidden_path(wire) is None


def test_bounded_summary_keeps_reason_counts_and_pair_bindings():
    raw = validation()
    raw["details"]["candidates"] *= 10
    raw["details"]["candidates"].append({"candidate_indices": {"left": 2, "right": 1,
        "private_actor": "SECRET"}, "code": "inter_arm_collision", "details": {"contacts": "SECRET"}})
    summary = public_plan_validation(raw)["candidate_summary"]
    assert summary["reported_count"] == 21
    assert len(summary["outcomes"]) == 8 and summary["omitted_outcome_count"] == 13
    assert {x["code"]: x["count"] for x in summary["reason_counts"]} == {
        "planner_failed": 10, "workspace_path": 10, "inter_arm_collision": 1}
    raw["details"]["candidates"] = raw["details"]["candidates"][-1:]
    item = public_plan_validation(raw)["candidate_summary"]["outcomes"][0]
    assert item["candidate_indices"] == {"left": 2, "right": 1}
    assert "SECRET" not in json.dumps(item)


def test_unknown_and_missing_evidence_invent_no_reason():
    raw = validation()
    raw["details"]["candidates"] = [{"rejection": {"code": "SECRET"}}, None]
    summary = public_plan_validation(raw)["candidate_summary"]
    assert summary["outcomes"][0]["code"] == "unknown_rejection"
    assert summary["uninterpretable_count"] == 1
    assert "SECRET" not in json.dumps(summary)
    assert public_plan_validation({"status": "rejected", "code": "topk_no_admitted_plan"}) == {
        "status": "rejected", "code": "topk_no_admitted_plan"}
    assert public_plan_validation({"status": "accepted", "details": {"candidates": [1]}}) == {
        "status": "accepted"}


def test_single_inter_arm_rejection_preserves_robot_measurements_through_wire(tmp_path):
    details = {'stage_index': 171, 'link_pair': [4, 4], 'center_distance_m': .09442456,
        'required_center_distance_m': .095, 'current_center_distance_m': .19677632,
        'planning_center_distance_m': .095, 'execution_center_distance_m': .09,
        'escape_allowed': False, 'actor': 'SECRET', 'object_gt': 'SECRET'}
    raw = _receipt(target_arms=['right'], statuses={'right': 'Success'})
    raw.update(failure_stage='plan_validation', failure_code='inter_arm_collision',
               execution_status='plan_rejected', state_changed=False, steps_used=0)
    raw['arms']['right']['plan_validation'] = {'status': 'rejected', 'code': 'inter_arm_collision',
                                              'details': details, 'message': 'SECRET'}
    effect = resolve_receipt_effect_projector(raw['schema'])(raw, ReceiptProjectionContext(
        'action-7', 'move_pose', surface_supports=lambda _: True))
    pipeline = NativeContextPipeline('native_causal_context_v1', ArtifactStore(tmp_path), 100_000, 4096)
    projected = pipeline.project(messages=[{'role': 'user', 'content': json.dumps({'last_action_effect': effect})}],
                                 tools=[], observation_step=16)
    request = {'model': 'offline', 'max_tokens': 64, 'messages': projected['messages']}
    for wire in (request, translate_anthropic_request(request)):
        text = json.dumps(wire)
        # Every capability is on: the measured centre distance reaches the wire.
        assert '0.0944' in text and '0.095' in text and '171' in text
        assert 'robot_link_sphere_centers_not_end_effector_gap_or_measured_contact' in text
        assert 'SECRET' not in text and 'object_gt' not in text
        assert _forbidden_path(wire) is None
    for invalid in (True, 'SECRET', float('nan'), float('inf'), -1):
        public = public_plan_validation({'status': 'rejected', 'code': 'inter_arm_collision',
                                        'details': {**details, 'center_distance_m': invalid}})
        assert 'center_distance_m' not in public['clearance']


def _planning_diagnostics():
    return {
        "schema": "ora.planning-diagnostics.v1",
        "phase": "ik", "outcome": "no_converged_candidate",
        "candidate_counts": {"returned": 32, "accepted": 0, "requested": None},
        "rejection_counts": {"position_error": 32, "self_collision": None},
        "details_available": True,
        "recovery": ["Inspect the current pose and consider an intermediate target."],
        "unknown_reasons": ["Per-term collision costs were not retained."],
        "evidence_scope": "One bounded solve; this does not prove unreachability.",
    }


def _receipt_with_planning_diagnostics(backend, diagnostic):
    canonical = {"planning_diagnostics": diagnostic,
        "planning_diagnostic_evidence": {"attempts": [{"joint_positions": "PRIVATE_IK_SEEDS"}]},
        "candidate_validation": {"raw_seed": "PRIVATE_IK_SEEDS"},
        "ik_validation_artifact": {"path": "/private/PRIVATE_IK_SEEDS.json"},
        "raw_seed_positions": "PRIVATE_IK_SEEDS"}
    if backend == "robotwin":
        raw = _receipt(target_arms=["right"], statuses={"right": "Fail"})
        raw["arms"]["right"]["planner"].update(canonical)
        return raw
    raw = {"schema": "ora.libero-pro-hybrid-motion-receipt.v2",
        "action_id": "action-7", "action": "move_pose", "motion_status": "refused",
        "feedback": {"planning_failure_reason": "MotionGenStatus.IK_FAIL"},
        "planner": canonical, "terminal_after_action": False}
    if backend == "libero_hybrid":
        return raw
    sparse = {"schema": f"ora.{backend}-sparse-motion-receipt.v1",
        "action_id": "action-7", "action": "move_pose", "execution_status": "planner_failed",
        "failure_stage": "planner", "failure_code": "ik_failed:move",
        "target_arms": ["right"], "terminal_after_action": False,
        "arms": {"right": {"targeted": True, "pose_error": {"translation_m": .02}}},
        "ik_attempts": [{"candidates": "PRIVATE_IK_SEEDS"}]}
    if backend == "libero":
        sparse["segment"] = raw
    else:
        sparse["arms"]["right"].update(canonical)
        # Real Dojo receipts own their summary directly on the arm.
        sparse["arms"]["right"].pop("planner", None)
    return sparse


def _projected_planning_diagnostics(effect, backend):
    detail = effect["detail"]
    if backend == "robotwin":
        return detail["arms"]["right"]["planner"].get("planning_diagnostics")
    if backend == "robodojo":
        return detail["arms"]["right"].get("planning_diagnostics")
    return detail.get("planning_diagnostics")


@pytest.mark.parametrize("backend", ["robotwin", "libero_hybrid", "libero", "robodojo"])
@pytest.mark.parametrize("retention", ["action_step_lossless_v4", "rewrite_compact_v1"])
def test_planning_diagnostics_reach_both_provider_wires_without_canonical_ik_evidence(tmp_path, backend, retention):
    diagnostic = {**_planning_diagnostics(), "raw_seed_positions": "PRIVATE_IK_SEEDS"}
    raw = _receipt_with_planning_diagnostics(backend, diagnostic)
    original = copy.deepcopy(raw)
    effect = resolve_receipt_effect_projector(raw["schema"])(raw, ReceiptProjectionContext(
        "action-7", "move_pose", surface_supports=lambda _: True))
    expected = _planning_diagnostics()
    expected.pop("schema")
    assert _projected_planning_diagnostics(effect, backend) == expected
    assert effect["failure_stage"] == "planner"
    assert raw == original
    pipeline = _pipeline(tmp_path, retention)
    projected = pipeline.project(messages=[{"role": "user", "content": json.dumps({
        "current_actual": {"actual_sequence": 16}, "last_action_effect": effect}),
        **OBSERVATION_METADATA}], tools=[], observation_step=16)
    request = {"model": "offline", "max_tokens": 64, "messages": projected["messages"]}
    for wire in (request, translate_anthropic_request(request)):
        text = json.dumps(wire)
        assert "no_converged_candidate" in text and 'self_collision\\\": null' in text
        assert "PRIVATE_IK_SEEDS" not in text
        assert "planning_diagnostic_evidence" not in text and "candidate_validation" not in text
        assert _forbidden_path(wire) is None


@pytest.mark.parametrize("change", [
    {"phase": "private_phase"}, {"phase": []}, {"outcome": ""},
    {"details_available": 1}, {"candidate_counts": {"returned": True}},
    {"candidate_counts": {"returned": -1}}, {"candidate_counts": {"returned": 1.5}},
    {"candidate_counts": {"returned": float("nan")}},
    {"candidate_counts": {str(i): 0 for i in range(33)}},
    {"rejection_counts": []}, {"recovery": ["hint"] * 5},
    {"recovery": "hint"}, {"unknown_reasons": [None]},
    {"note": "x" * 1201}, {"schema": "unknown.v1"},
])
def test_malformed_planning_diagnostics_are_omitted_from_every_receipt(change):
    from ora.contracts.planning_diagnostics import public_planning_diagnostics

    diagnostic = {**_planning_diagnostics(), **change}
    assert public_planning_diagnostics(diagnostic) == {}
    for backend in ("robotwin", "libero_hybrid", "libero", "robodojo"):
        raw = _receipt_with_planning_diagnostics(backend, diagnostic)
        effect = resolve_receipt_effect_projector(raw["schema"])(raw, ReceiptProjectionContext(
            "action-7", "move_pose", surface_supports=lambda _: True))
        assert _projected_planning_diagnostics(effect, backend) is None
        assert "PRIVATE_IK_SEEDS" not in json.dumps(effect)
        assert effect["failure_stage"] == "planner"


def test_absent_historical_planning_diagnostics_do_not_invent_counts():
    from ora.contracts.planning_diagnostics import public_planning_diagnostics

    for missing in (None, {}, [], "unavailable"):
        assert public_planning_diagnostics(missing) == {}
