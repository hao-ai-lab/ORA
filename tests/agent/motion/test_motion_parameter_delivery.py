"""Execution-owned numbers survive actual receipt and tool-help projections."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ora.agent.composition.public_parameters import bind_public_workspace
from ora.agent.composition.tool_surface import resolve_tool_surface
from ora.agent.native.receipt_projection import public_acceptance_checks, ReceiptProjectionContext
from ora.contracts.acceptance import motion_acceptance_check, validate_motion_acceptance_checks
from ora.environment.controllers.motion_transaction import select_post_global_phase


def context(enabled=True):
    return ReceiptProjectionContext(last_action_id="a1", last_action_public_name="move_pose",
                                    surface_supports=lambda name: enabled and name == "current_public_robot_facts_v1")


@pytest.mark.parametrize("staged,tracking,stage_error,final_error,expected,stages", [
    (False, False, .002, .008, "global_tracking_not_converged", []),
    (True, True, .006, .008, "precontact_global_not_reached", ["precontact_global"]),
    (True, True, .004, .008, "run_local", ["precontact_global", "post_global_final", "local_entry_envelope"]),
    (False, True, .008, .008, "run_local", ["post_global_final", "local_entry_envelope"]),
    (False, True, .001, .001, "not_needed", ["post_global_final"]),
    (False, True, .02, .02, "outside_settle_envelope", ["post_global_final", "local_entry_envelope"]),
])
def test_actual_router_records_only_checks_it_ran(staged, tracking, stage_error, final_error, expected, stages):
    checks = []
    result = select_post_global_phase(staged_contact=staged, tracking_completed=tracking,
        stage_endpoint_errors={"left": {"translation_m": stage_error, "orientation_rad": .02}},
        final_endpoint_errors={"left": {"translation_m": final_error, "orientation_rad": .02}},
        arrival_position_tolerance_m=.005, arrival_orientation_tolerance_rad=.05,
        final_arrival_position_tolerance_m=.003, final_arrival_orientation_tolerance_rad=.03,
        local_translation_envelope_m=.01, local_rotation_envelope_rad=.087,
        acceptance_evidence=checks)
    assert result == expected
    assert [item["stage"] for item in checks] == stages
    projected = public_acceptance_checks({"acceptance_checks": checks}, context())
    assert projected["acceptance_checks"] == checks
    for check in checks:
        bound = {"precontact_global": .005, "post_global_final": .003, "local_entry_envelope": .01}[check["stage"]]
        assert check["tolerance"]["position_m"] == bound
        assert check["claim"] == "numeric_bounds_only_not_motion_or_task_success"


@pytest.mark.parametrize("comparison,within", [("<", False), ("<=", True)])
def test_exact_boundary_comparison_is_not_silently_normalized(comparison, within):
    result = motion_acceptance_check(stage="final_endpoint", comparison=comparison,
        errors={"right": {"position_m": .013, "orientation_rad": .079}},
        position_tolerance_m=.013, orientation_tolerance_rad=.079)
    assert result["arms"]["right"]["within_tolerance"] == {"position_m": within, "orientation_rad": within}
    assert validate_motion_acceptance_checks([result]) == [result]
    result["arms"]["right"]["within_tolerance"]["position_m"] = not within
    with pytest.raises(ValueError, match="comparison differs"):
        validate_motion_acceptance_checks([result])


@pytest.mark.parametrize("bad", ["private", "nan", "negative", "boolean", "zero_limit"])
def test_receipt_numeric_contract_does_not_relax_privacy_or_types(bad):
    error = {"position_m": .001, "orientation_rad": .02}
    limit = .005
    if bad == "private": error["object_pose"] = [1, 2, 3]
    elif bad == "nan": error["position_m"] = float("nan")
    elif bad == "negative": error["position_m"] = -.001
    elif bad == "boolean": error["position_m"] = True
    else: limit = 0
    with pytest.raises(ValueError):
        motion_acceptance_check(stage="final_endpoint", errors={"right": error},
                                position_tolerance_m=limit, orientation_tolerance_rad=.05)


@pytest.mark.parametrize("comparison", ["<", "<="])
@pytest.mark.parametrize("wrapped", [False, True])
def test_libero_persisted_receipt_delivers_the_executed_threshold_after_sparse_wrap(tmp_path, comparison, wrapped):
    from ora.environment.backends.libero_pro.session import LiberoProSession
    receipt = {"schema": "ora.libero-pro-hybrid-motion-receipt.v2", "action_id": "a1",
        "action": "move_pose", "state_changed": False, "motion_status": "not_reached",
        "feedback": {"endpoint_error": {"position_m": .0131, "orientation_rad": .06},
                     "endpoint_tolerance": {"position_m": .013, "orientation_rad": .079},
                     "endpoint_comparison": comparison}}
    session = SimpleNamespace(motion_profile=None, output_root=tmp_path, action_sequence=0,
                              _sparse_receipt_transform=(lambda inner: {"schema": "wrapped", "segment": inner}) if wrapped else None)
    persisted = LiberoProSession._persist_receipt(session, receipt)
    raw = json.loads((tmp_path / "receipts/action-0000.json").read_text())
    assert raw == persisted == session.last_receipt["receipt"]
    projected = public_acceptance_checks(raw, context())
    check = projected["acceptance_checks"][0]
    assert check["comparison"] == comparison
    assert check["tolerance"] == {"position_m": .013, "orientation_rad": .079}
    assert check["arms"]["right"]["within_tolerance"] == {"position_m": False, "orientation_rad": True}


def test_terminal_tracking_uses_its_own_threshold_and_legacy_delivery_is_unchanged():
    record = {"controller_tracking": {"terminal_endpoint_check": {"enabled": True,
        "position_tolerance_m": .0042, "orientation_tolerance_rad": .042,
        "final_errors": {"left": {"translation_m": .0043, "orientation_rad": .01}}}}}
    assert public_acceptance_checks(record, context(False)) == {}
    result = public_acceptance_checks(record, context())
    assert result["acceptance_checks"][0]["tolerance"]["position_m"] == .0042
    assert result["acceptance_checks"][0]["stage"] == "controller_terminal_endpoint"
    assert public_acceptance_checks({}, context())["acceptance_evidence_status"] == "unavailable_in_this_receipt"


@pytest.mark.parametrize("profile", [
    "libero_placement_direct_overhead_v1", "robotwin_toolchain_b2_closed_noservo_v2",
])
def test_workspace_help_uses_actual_admission_box_once_and_does_not_claim_reachability(profile):
    surface = resolve_tool_surface(profile)
    bounds = {"x": [-.333, .222], "y": [-.444, .555], "z": [.123, 1.234]}
    original = copy.deepcopy(bounds)
    bound = bind_public_workspace(surface, bounds)
    descriptions = {d["name"]: d["description"] for d in bound.definitions}
    public = json.loads(descriptions["move_pose"].split("Bound action workspace: ")[1])
    assert public["bounds"] == bounds == original
    assert public["semantics"] == "command_envelope_not_reachability"
    assert sum("Bound action workspace: " in text for text in descriptions.values()) == 1
    # A surface without current_public_robot_facts_v1 is returned unchanged.
    old = resolve_tool_surface("native_atomic_causal_v1")
    assert not old.supports("current_public_robot_facts_v1")
    assert bind_public_workspace(old, bounds) is old
