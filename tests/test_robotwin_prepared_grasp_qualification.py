"""The offline diagnostic checker distinguishes valid stopping from coverage."""
import copy

from experiments.runtime_v1.robotwin_prepared_grasp_qualification import check_case, closed_start_proof, qualification_root
import pytest


def fixture():
    contact = {"position_m": [0., 0., .8], "orientation_xyzw": [0., 0., 0., 1.]}
    pregrasp = {**contact, "position_m": [0., 0., .9]}
    case = {"case_id": "left-empty_offset", "arm": "left", "candidate": {
        "pregrasp_pose": pregrasp, "contact_pose": contact}}
    commitment = {"arm": "left", "poses": {"pregrasp": pregrasp, "approach": contact,
        "lift": {**contact, "position_m": [0., 0., .83]}}}
    receipt = {"committed_sequence": ["open", "pregrasp", "approach", "close", "lift", "verify"],
        "stage_receipts": [{"stage": "open", "status": "executed"},
            {"stage": "pregrasp", "status": "not_reached"}],
        "skill_receipt": {"ledger_released": True}, "status": "aborted", "stop_reason": "pregrasp_not_reached"}
    return case, receipt, commitment


def test_unreachable_prefix_is_correct_stopping_but_incomplete_physical_coverage():
    result = check_case(*fixture(), [])
    assert result["mechanism_checks_passed"]
    assert result["physical_coverage"] == "prefix_only"
    assert result["holding_success_claimed"] is False


@pytest.mark.parametrize("defect", ["continue_after_failure", "wrong_arm", "skip_pregrasp", "missing_baseline_images"])
def test_checker_rejects_evidence_or_order_defects(defect):
    case, receipt, commitment = copy.deepcopy(fixture())
    if defect == "continue_after_failure":
        receipt["stage_receipts"].append({"stage": "approach", "status": "executed"})
    elif defect == "wrong_arm":
        receipt["stage_receipts"][0]["action"] = {"schema": "ora-motion-action.v3", "arguments": {"targets": {"right": "open"}}}
    elif defect == "skip_pregrasp":
        receipt["stage_receipts"][1]["stage"] = "approach"
    else:
        receipt["skill_receipt"]["observation_evidence"] = [{"role": "post_close_before_lift"}]
    assert not check_case(case, receipt, commitment, [])["mechanism_checks_passed"]


def test_checker_cannot_publish_inside_formal_runs(tmp_path):
    with pytest.raises(ValueError, match="outside formal"):
        qualification_root(tmp_path / "runs" / "formal" / "review")


@pytest.mark.parametrize("aperture,consumed,expected", [(0., True, True), (.02, True, False), (0., False, False), (None, True, False)])
def test_closed_setup_uses_public_aperture_not_actuator_normalization(aperture, consumed, expected):
    setup = {"receipt": {"consumed": consumed, "execution_status": "executed"}}
    effector = {"gripper_command_target": "close", "gripper_opening_m": aperture,
                "gripper_state": {"measured_opening_normalized": 2 / 11}}
    assert closed_start_proof(setup, effector)["passed"] is expected
