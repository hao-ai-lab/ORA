"""A staged receipt must fit the delivery ceilings it is measured against.

The arithmetic here is measured with the same serializer the threshold uses,
not estimated. ``body_bytes`` mirrors ``native/session.py``: image payloads and
``_runtime`` are stripped before the tool message is built, so they do not count
toward ``eager_large_result_bytes``.
"""

from __future__ import annotations

import json

import pytest

from ora.agent.tools.robot import ToolError
from ora.agent.tools.skill_receipt import (
    DELIVERY_CEILING_BYTES, MAX_RATIONALE_CHARS, MAX_RECEIPT_IMAGES,
    MAX_STAGE_MEASUREMENTS, MAX_STAGES, RECEIPT_BUDGET_BYTES, SCHEMA,
    assemble, body_bytes, image_bindings, stage, validate_rationale,
)


def _wide_measurements(count):
    """Measurement entries in the shape ``placement_runtime.measurements`` emits."""

    return [
        {
            "name": f"geometry.support.metrics.plane_axes_world[{index}].residual_m",
            "scalar": 0.00123456789,
            "vector": None,
            "text": None,
        }
        for index in range(count)
    ]


def _worst_case_receipt():
    """The largest receipt the closed skills can produce.

    Six stages (place_execute's full sequence) each carrying the per-stage
    measurement cap, a full verification block, the held-object offset, and the
    maximum image bindings.
    """

    return assemble(
        skill="place_execute",
        status="completed",
        rationale="x" * MAX_RATIONALE_CHARS,
        candidate_id="P2",
        observation_id="obs-000000000000000000000000000000",
        observation_sha256="f" * 64,
        stages=[
            stage(
                name, "executed",
                "measured_stage_outcome_with_its_full_diagnostic_vector_attached",
                action_id=f"action-{index}",
                measurements=_wide_measurements(MAX_STAGE_MEASUREMENTS),
            )
            for index, name in enumerate(
                ("transport", "descend", "stop", "release", "settle", "verify")
            )
        ],
        verification={
            "status": "unknown",
            "reason": "visible_surface_motion_hypotheses_are_not_separated",
            "closure_evidence": {
                "schema": "ora.closure-extent-evidence.v1",
                "gripper_opening_m": 0.0301,
                "target_extents_m": [0.121, 0.0502, 0.0321],
                "strength": "not_contradictory",
                "reason": "jaw_separation_lies_inside_the_measured_visible_extent_range_and_proves_nothing",
            },
            "measurements": _wide_measurements(MAX_STAGE_MEASUREMENTS),
        },
        held_object_offset={
            "frame": "world",
            "translation_m": [0.0123, -0.0456, 0.0789],
            "basis": "visible_object_points_median_minus_tcp_position",
            "observation_id": "obs-000000000000000000000000000000",
            "observation_step": 42,
            "claim_boundary": "x" * 300,
        },
        images=[
            {"role": "stage_view", "camera": "wrist", "target_ref": f"target-{index}"}
            for index in range(MAX_RECEIPT_IMAGES)
        ],
        claim_boundary="y" * 400,
    )


def test_the_worst_case_receipt_stays_under_the_measured_delivery_ceiling():
    receipt = _worst_case_receipt()
    measured = body_bytes(receipt)

    assert measured == receipt["delivery"]["body_bytes"]
    assert measured <= RECEIPT_BUDGET_BYTES, (
        f"worst-case receipt is {measured} bytes, over the "
        f"{RECEIPT_BUDGET_BYTES}-byte budget"
    )
    assert measured < DELIVERY_CEILING_BYTES, (
        f"worst-case receipt is {measured} bytes and would trip the "
        f"{DELIVERY_CEILING_BYTES}-byte ceiling"
    )


def test_body_bytes_measures_what_the_threshold_measures():
    """Images and ``_runtime`` are removed before the tool message is built."""

    body = {"schema": SCHEMA, "stages": []}
    with_images = {**body, "images_base64": ["A" * 200_000], "_runtime": {"directive": "x"}}

    assert body_bytes(with_images) == body_bytes(body)
    assert body_bytes(body) == len(
        json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )


def test_an_oversized_receipt_elides_the_oldest_measurements_and_records_it():
    """Nothing is dropped silently, and the newest stage keeps its numbers."""

    receipt = assemble(
        skill="place_execute",
        status="completed",
        rationale="chose the widest clear patch",
        candidate_id="P0",
        observation_id="obs-1",
        observation_sha256="a" * 64,
        stages=[
            stage(name, "executed", "ok", measurements=_wide_measurements(MAX_STAGE_MEASUREMENTS))
            for name in ("transport", "descend", "release", "settle", "verify")
        ],
        claim_boundary="z",
        budget_bytes=4_000,
    )

    assert receipt["delivery"]["body_bytes"] <= 4_000
    assert receipt["delivery"]["stages_with_elided_measurements"] > 0
    elided = [item for item in receipt["stages"] if item["measurements_elided"]]
    assert elided, "an elision must be recorded on the stage it happened to"
    assert elided[0]["stage"] == "transport", "the oldest stage gives up its numbers first"
    # The last stage explains the outcome and is the last to lose anything.
    assert receipt["stages"][-1]["stage"] == "verify"


def test_every_image_index_is_bound_to_its_own_source():
    """Position-based attribution misattributes survivors of per-block eviction."""

    bindings = image_bindings([
        {"role": "stop_close_view", "camera": "wrist", "target_ref": "target-a"},
        {"role": "verify_overlay", "camera": "agentview", "target_ref": "target-b"},
    ])

    assert [item["image_index"] for item in bindings] == [0, 1]
    assert bindings[0]["role"] == "stop_close_view"
    assert bindings[0]["target_ref"] == "target-a"
    assert bindings[1]["camera"] == "agentview"
    # The shared binding shape every perception tool uses.
    assert set(bindings[0]) == {
        "image_index", "role", "camera", "candidate_id", "target_ref", "observation_step",
    }


def test_a_skill_cannot_quietly_spend_the_whole_inline_image_budget():
    with pytest.raises(ToolError):
        image_bindings([{"role": "view"}] * (MAX_RECEIPT_IMAGES + 1))


def test_a_rationale_is_mandatory_and_bounded():
    assert validate_rationale("  the only candidate whose pregrasp probed reachable  ") == (
        "the only candidate whose pregrasp probed reachable"
    )

    for empty in ("", "   ", None, 7):
        with pytest.raises(ToolError) as refused:
            validate_rationale(empty)
        assert refused.value.field == "rationale"

    with pytest.raises(ToolError) as long:
        validate_rationale("x" * (MAX_RATIONALE_CHARS + 1))
    assert long.value.constraints["maximum"] == MAX_RATIONALE_CHARS


def test_the_rationale_travels_with_the_outcome_it_explains():
    receipt = assemble(
        skill="grasp_execute", status="completed",
        rationale="G1 is the only top-down candidate clear of the neighbouring mug",
        candidate_id="G1", observation_id="obs-1", observation_sha256="a" * 64,
        stages=[stage("approach", "executed", "reached")],
        claim_boundary="z",
    )
    assert receipt["rationale"].startswith("G1 is the only top-down candidate")
    assert receipt["candidate_id"] == "G1"


def test_a_skill_declares_no_more_stages_than_the_closed_sequence_allows():
    with pytest.raises(ToolError):
        assemble(
            skill="place_execute", status="completed", rationale="r",
            candidate_id=None, observation_id="o", observation_sha256="a" * 64,
            stages=[stage(f"s{index}", "executed", "ok") for index in range(MAX_STAGES + 1)],
            claim_boundary="z",
        )


def test_an_unknown_stage_or_skill_status_is_refused_rather_than_carried():
    with pytest.raises(ToolError):
        stage("approach", "succeeded", "ok")
    with pytest.raises(ToolError):
        assemble(
            skill="grasp_execute", status="ok", rationale="r", candidate_id=None,
            observation_id="o", observation_sha256="a" * 64, stages=[], claim_boundary="z",
        )


def test_a_completed_sequence_reaches_the_policy_as_its_stages_not_as_unknown():
    """The delivery this module is named for, which nothing here measured.

    Every test above weighs the receipt: its bytes, its images, its stage
    vocabulary. None of them asked whether the policy ever sees it. It did
    not: the native session resolves an effect projector by receipt schema
    and there was none for a skill sequence, so both closed grasps in the
    first B2 run were followed by a request carrying
    ``{action: unknown, status: world_state_unknown}`` and no stage result at
    all, while the tool's own help promised the stages on the next turn.
    """

    from ora.agent.native.receipt_projection import (
        resolve_receipt_effect_projector, ReceiptProjectionContext,
    )

    receipt = {
        "schema": "ora.manipulation-skill-sequence-receipt.v1",
        "action_id": "action-71dedc53a286b2a9",
        "skill": "grasp_execute",
        "candidate_id": "grasp-821d9398bd0a5ae1",
        "status": "completed",
        "stop_reason": "sequence_completed",
        "committed_sequence": ["approach", "close", "verify"],
        "resumed_from": 0,
        "dispatched_stage_actions": 2,
        "stage_receipts": [
            {"stage": "approach", "status": "executed", "reason": "executed",
             "action_id": "action-71dedc53a286b2a9:00-approach",
             "measurements": [{"quantity": "pose_residual_m", "value": 0.004}],
             "receipt": {"schema": "ora.composite-motion-receipt.v1"}},
            {"stage": "close", "status": "executed", "reason": "executed",
             "action_id": "action-71dedc53a286b2a9:01-close",
             "measurements": [],
             "receipt": {"schema": "ora.composite-motion-receipt.v1"}},
            {"stage": "verify", "status": "executed",
             "reason": "verified_against_libero_10:task-00:seed-0:observation-0001",
             "action_id": None,
             "measurements": [
                 {"quantity": "closure_strength", "value": "strong_negative"},
                 {"quantity": "gripper_opening_m", "value": 0.00234956917},
             ],
             "verification": {
                 "strength": "strong_negative",
                 "gripper_opening_m": 0.00234956917,
                 "reason": "jaws_met_with_less_than_the_near_closed_bound_between_them",
                 "claim_boundary": "jaw separation beside visible extents is evidence",
             }},
        ],
    }
    projector = resolve_receipt_effect_projector(receipt["schema"])
    assert projector is not None, "a skill sequence has no effect projector"

    effect = projector(receipt, ReceiptProjectionContext(
        last_action_id="action-71dedc53a286b2a9",
        last_action_public_name="grasp_execute",
    ))

    assert effect["action"] == "grasp_execute"
    assert effect["status"] == "completed"
    assert effect["failure_code"] is None
    stages = effect["detail"]["stages"]
    assert [item["stage"] for item in stages] == ["approach", "close", "verify"]
    assert [item["status"] for item in stages] == ["executed"] * 3
    assert stages[2]["reason"].startswith("verified_against_")
    # The absorbed verdict is the whole point of absorbing the verifier.
    assert effect["detail"]["verification"]["strength"] == "strong_negative"
    assert "claim_boundary" in effect["detail"]["verification"]
    # A measurement a stage took is a measurement the policy gets to read.
    assert stages[0]["measurements"][0]["quantity"] == "pose_residual_m"
    # Resuming is a policy decision, so what a resume would skip is visible.
    assert effect["detail"]["committed_sequence"] == ["approach", "close", "verify"]
    assert effect["detail"]["candidate_id"] == "grasp-821d9398bd0a5ae1"
    # A schema field inside policy-visible content is forbidden by the
    # causal projection; the stage receipts must not smuggle one through.
    # The causal projection forbids schema fields in policy-visible content,
    # and the nested driven receipts are controller evidence, not policy
    # evidence; neither may ride along inside the stages.
    assert "schema" not in json.dumps(effect["detail"])
    assert all("receipt" not in item for item in stages)


def test_a_stopped_sequence_names_the_stage_it_stopped_before():
    from ora.agent.native.receipt_projection import (
        resolve_receipt_effect_projector, ReceiptProjectionContext,
    )

    receipt = {
        "schema": "ora.manipulation-skill-sequence-receipt.v1",
        "action_id": "action-abc",
        "skill": "place_execute",
        "candidate_id": "P0",
        "status": "stopped",
        "stop_reason": "stopped_for_close_view_before_release",
        "committed_sequence": ["transport", "descend", "stop", "release", "settle"],
        "resumed_from": 0,
        "dispatched_stage_actions": 2,
        "stage_receipts": [
            {"stage": "transport", "status": "executed", "reason": "executed",
             "action_id": "action-abc:00-transport", "measurements": [],
             "receipt": {"schema": "ora.composite-motion-receipt.v1"}},
            {"stage": "descend", "status": "not_reached",
             "reason": "endpoint_residual_exceeded",
             "action_id": "action-abc:01-descend", "measurements": [],
             "receipt": {"schema": "ora.composite-motion-receipt.v1"}},
        ],
    }
    effect = resolve_receipt_effect_projector(receipt["schema"])(
        receipt, ReceiptProjectionContext(
            last_action_id="action-abc", last_action_public_name="place_execute",
        ),
    )
    assert effect["status"] == "stopped"
    assert effect["failure_stage"] == "descend"
    assert effect["failure_code"] == "endpoint_residual_exceeded"
    assert effect["detail"]["stop_reason"] == "stopped_for_close_view_before_release"
    # Which stages never ran is what a resume decision turns on.
    assert effect["detail"]["stages_not_reached"] == ["stop", "release", "settle"]
