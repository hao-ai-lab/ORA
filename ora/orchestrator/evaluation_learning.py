"""Learning-eligibility steps: may a valid episode become training or Memory data?

``evaluation.assess_learning_eligibility`` composes these steps in order. A
behaviorally valid run can still be ineligible: a physical receipt breaks its own contract, a backend declared an
execution fault, or a LIBERO-Pro receipt lacks the qualified controller
evidence its OSC action contract requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ora.orchestrator.evidence_io import _receipt_execution_fault

from .evaluation_receipts import _physical_motion_receipts, _referenced_json_artifact
from .spec import json_sha256


def _backend_execution_fault_reasons(run_root: Path) -> list[str]:
    """Name the first receipt a backend itself labelled an execution fault.

    Backend-neutral fault authority: a receipt a backend itself labelled an
    execution fault is infrastructure, not behavior. Such a run can still
    carry a valid official verdict, but the trajectory crossed an unexplained
    physical discontinuity, so it must not become training or Memory
    evidence. Backends that never emit the status are unaffected.
    """

    faulted = sorted(
        path.name for path in run_root.rglob("action-0*.json")
        if _receipt_execution_fault(path)
    )
    if faulted:
        return [f"backend_execution_fault:{faulted[0]}"]
    return []


def _libero_control_reasons(
    plan: Mapping[str, Any], runtime: Mapping[str, Any],
    events: list[Mapping[str, Any]], run_root: Path | None,
) -> list[str]:
    """Bind every LIBERO-Pro receipt to one qualified OSC controller contract."""

    reasons: list[str] = []
    reset_records = [
        event.get("record") for event in events if event.get("type") == "reset"
    ]
    expected_resets = int(plan.get("session", {}).get("rounds", 1))
    if not _libero_workspace_qualified(runtime, reset_records, expected_resets):
        reasons.append("unqualified_action_workspace")
    contract_reasons, expected_contract_sha256, contract_schema = (
        _libero_controller_contract(reset_records, expected_resets)
    )
    reasons.extend(contract_reasons)
    reasons.extend(_libero_source_qualification_reasons(
        reset_records, expected_resets, expected_contract_sha256,
    ))
    receipts = _physical_motion_receipts(events)
    if not receipts:
        reasons.append("no_physical_action_evidence")
    for index, receipt in enumerate(receipts):
        reasons.extend(_libero_receipt_reasons(
            index, receipt, len(receipts), run_root,
            expected_contract_sha256, contract_schema,
        ))
    return reasons


def _libero_workspace_qualified(
    runtime: Mapping[str, Any], reset_records: list[Any], expected_resets: int,
) -> bool:
    """Every reset must carry the plan's hard action-envelope profile."""

    profile = runtime.get("workspace_profile")
    return (
        isinstance(profile, Mapping)
        and profile.get("schema")
        == "ora.action-workspace-profile.v1"
        and profile.get("semantics")
        == "hard_action_envelope_not_reachability"
        and isinstance(profile.get("profile_sha256"), str)
        and profile.get("bounds_m") == runtime.get("workspace_bounds_m")
        and len(reset_records) == expected_resets
        and all(
            isinstance(reset, Mapping)
            and reset.get("workspace_profile") == profile
            and reset.get("workspace_bounds_m")
            == runtime.get("workspace_bounds_m")
            for reset in reset_records
        )
    )


def _libero_controller_contract(
    reset_records: list[Any], expected_resets: int,
) -> tuple[list[str], str | None, str | None]:
    """Return reasons plus the one self-hashed OSC contract every reset shares."""

    reasons: list[str] = []
    contracts = [
        reset.get("controller_action_contract")
        for reset in reset_records if isinstance(reset, Mapping)
    ]
    valid_contracts = [
        contract for contract in contracts
        if isinstance(contract, Mapping)
        and contract.get("schema") in {
            "ora.libero-osc-action-contract.v1",
            "ora.libero-osc-action-contract.v2",
        }
        and contract.get("controller") == "OSC_POSE"
        and contract.get("control_delta") is True
        and contract.get("contract_sha256")
        == json_sha256({
            key: value for key, value in contract.items()
            if key != "contract_sha256"
        })
    ]
    if (
        len(reset_records) != expected_resets
        or len(valid_contracts) != expected_resets
        or len({contract["contract_sha256"] for contract in valid_contracts}) != 1
    ):
        reasons.append("missing_or_invalid_controller_action_contract")
        expected_contract_sha256 = None
        contract_schema = None
    else:
        expected_contract_sha256 = str(valid_contracts[0]["contract_sha256"])
        contract_schema = str(valid_contracts[0]["schema"])
        if len({contract["schema"] for contract in valid_contracts}) != 1:
            reasons.append("mixed_controller_action_contract_schema")
        if contract_schema != "ora.libero-osc-action-contract.v2":
            reasons.append("legacy_control_contract_indeterminate")
    return reasons, expected_contract_sha256, contract_schema


def _libero_source_qualification_reasons(
    reset_records: list[Any], expected_resets: int,
    expected_contract_sha256: str | None,
) -> list[str]:
    """Each reset's controller qualification must bind its source manifest."""

    reasons: list[str] = []
    source_fields = (
        ("robosuite_module", "robosuite_module_sha256"),
        ("controller_config_file", "controller_config_sha256"),
        ("panda_gripper_module", "panda_gripper_module_sha256"),
        ("panda_gripper_file", "panda_gripper_sha256"),
    )
    if len(reset_records) != expected_resets:
        reasons.append("missing_or_invalid_controller_source_qualification")
    for reset_index, reset in enumerate(reset_records):
        qualification = reset.get("qualified_controller_binding") if isinstance(
            reset, Mapping
        ) else None
        source_manifest = reset.get("source_manifest") if isinstance(
            reset, Mapping
        ) else None
        dependencies = source_manifest.get("dependencies") if isinstance(
            source_manifest, Mapping
        ) else None
        dependency_hashes = {
            str(item.get("path")): item.get("sha256")
            for item in dependencies if isinstance(item, Mapping)
            and isinstance(item.get("path"), str)
        } if isinstance(dependencies, list) else {}
        qualified_binding = qualification.get("binding") if isinstance(
            qualification, Mapping
        ) else None
        if not (
            isinstance(qualification, Mapping)
            and qualification.get("schema")
            == "ora.libero-controller-qualification.v1"
            and qualification.get("contract_sha256")
            == expected_contract_sha256
            and isinstance(qualified_binding, Mapping)
            and all(
                isinstance(qualified_binding.get(path_field), str)
                and dependency_hashes.get(qualified_binding[path_field])
                == qualified_binding.get(digest_field)
                for path_field, digest_field in source_fields
            )
        ):
            reasons.append("missing_or_invalid_controller_source_qualification")
            reasons.append(
                "missing_or_invalid_controller_source_qualification:"
                f"{reset_index}"
            )
    return reasons


def _libero_receipt_reasons(
    index: int, receipt: Mapping[str, Any], receipt_count: int,
    run_root: Path | None, expected_contract_sha256: str | None,
    contract_schema: str | None,
) -> list[str]:
    """Judge one LIBERO-Pro physical receipt under the run's OSC contract."""

    receipt_schema = receipt.get("schema")
    sparse_strategy = None
    if (receipt_schema == "ora.libero-sparse-motion-receipt.v1"
            and receipt.get("action") == "wait"):
        # The shared hold has no executor segment or pose tracking: it binds
        # the OSC contract it stepped through and its own hold invariant.
        from ora.environment.controllers.libero_wait import libero_wait_receipt_issues

        return _libero_control_health_reasons(
            index, receipt.get("control_health"), expected_contract_sha256, contract_schema,
            joint_position_motion=False,
        ) + ([f"invalid_wait_receipt:{index}"] if libero_wait_receipt_issues(receipt) else [])
    if receipt_schema == (
        "ora.libero-sparse-motion-receipt.v1"
    ):
        # EXP08A sparse motion v3: the envelope embeds the exact
        # legacy receipt of the executor that ran, bound by the
        # shared action_id; the embedded evidence is judged under
        # the envelope's actual route.
        sparse_strategy = receipt.get("motion_strategy")
        segment = receipt.get("segment")
        if (
            receipt.get("execution_status") == "plan_rejected"
            and receipt.get("state_changed") is False
        ):
            # A recoverable route refusal executed nothing and has
            # no controller evidence to demand. The emitter may
            # still attach a refusal pseudo-segment carrying the
            # route code, so the exemption cannot key on the
            # segment's absence.
            return []
        if (
            not isinstance(segment, Mapping)
            or segment.get("action_id") != receipt.get("action_id")
        ):
            return [f"missing_or_invalid_control_health:{index}"]
        receipt = segment
        receipt_schema = receipt.get("schema")
    reasons: list[str] = []
    feedback = receipt.get("feedback")
    if "operator_stop_requested" in {
        receipt.get("failure_code"),
        feedback.get("stop_reason")
        if isinstance(feedback, Mapping) else None,
    } and index < receipt_count - 1:
        # An operator stop that leaves later actions in the same run
        # is not an operator stopping the run: it is stale stop
        # state polluting execution. Such runs stay behaviorally
        # valid but must not be exported as learning data. A stop on
        # the final physical receipt is a legitimate terminal stop
        # and is not pollution.
        reasons.append(f"operator_stop_execution:{index}")
    health = receipt.get("control_health")
    receipt_action = receipt.get("action")
    hybrid_motion = receipt_schema in {
        "ora.libero-pro-hybrid-motion-receipt.v1",
        "ora.libero-pro-hybrid-motion-receipt.v2",
    }
    global_motion = (
        isinstance(receipt_action, str)
        and receipt_action in {"move_free_space", "recover_motion"}
    ) or sparse_strategy == "global"
    reasons.extend(_libero_control_health_reasons(
        index, health, expected_contract_sha256, contract_schema,
        joint_position_motion=hybrid_motion and global_motion,
    ))
    if contract_schema == "ora.libero-osc-action-contract.v2":
        tracking = receipt.get("tracking_evidence")
        tracking_record = _referenced_json_artifact(run_root, tracking)
        if hybrid_motion and global_motion:
            if not _hybrid_motion_evidence_valid(receipt, run_root):
                reasons.append(
                    f"missing_or_invalid_hybrid_motion_evidence:{index}"
                )
        elif not _pose_tracking_valid(
            receipt, receipt_schema, receipt_action, tracking,
            tracking_record, expected_contract_sha256,
        ):
            reasons.append(
                f"missing_or_invalid_pose_tracking:{index}"
            )
    return reasons


def _libero_control_health_reasons(
    index: int, health: Any, expected_contract_sha256: str | None,
    contract_schema: str | None, *, joint_position_motion: bool,
) -> list[str]:
    """The receipt's control health must match the contract's health schema."""

    expected_health_schema = (
        "ora.libero-control-health.v2"
        if contract_schema == "ora.libero-osc-action-contract.v2"
        else "ora.libero-control-health.v1"
    )
    health_valid = (
        isinstance(health, Mapping)
        and health.get("schema") == expected_health_schema
        and health.get("status") in {"healthy", "degraded", "fault"}
    )
    # Global hybrid motion is executed by JOINT_POSITION, so binding
    # it to the OSC action-contract hash would be a category error.
    # Local servo/gripper receipts still have to bind that exact OSC
    # contract because they execute through the OSC channel.
    if health_valid and not joint_position_motion:
        health_valid = (
            health.get("controller_action_contract_sha256")
            == expected_contract_sha256
        )
    if not health_valid:
        return [f"missing_or_invalid_control_health:{index}"]
    if health.get("status") == "fault":
        return [f"control_health_fault:{index}"]
    return []


def _hybrid_motion_evidence_valid(
    receipt: Mapping[str, Any], run_root: Path | None,
) -> bool:
    """Global JOINT_POSITION motion needs its planner and execution artifacts."""

    if receipt.get("motion_status") == "refused":
        return (
            receipt.get("state_changed") is False
            and isinstance(receipt.get("feedback"), Mapping)
            and receipt["feedback"].get("reachability") is False
        )
    planner = receipt.get("planner")
    execution = receipt.get("execution")
    planner_valid = False
    if isinstance(planner, Mapping):
        if planner.get("world_model") == "none":
            planner_valid = _referenced_json_artifact(
                run_root, planner.get("canonical_artifact")
            ) is not None
        elif (
            planner.get("source") == "motion_probe_plan_ref"
            and isinstance(planner.get("plan_ref"), str)
        ):
            # A probed immutable plan executed verbatim;
            # runs landed before the path re-base recorded
            # the artifact relative to the probe root, so
            # resolution falls back once (bytes stay
            # hash-verified either way).
            reference = planner.get("artifact")
            planner_valid = _referenced_json_artifact(
                run_root, reference,
            ) is not None
            if not planner_valid and isinstance(
                reference, Mapping
            ) and isinstance(reference.get("path"), str):
                planner_valid = _referenced_json_artifact(
                    run_root,
                    {
                        **reference,
                        "path": "motion-probes/"
                        + reference["path"],
                    },
                ) is not None
    return (
        planner_valid
        and isinstance(execution, Mapping)
        and execution.get("controller") == "JOINT_POSITION"
        and _referenced_json_artifact(
            run_root, execution.get("canonical_artifact")
        ) is not None
    )


def _pose_tracking_valid(
    receipt: Mapping[str, Any], receipt_schema: Any, receipt_action: Any,
    tracking: Any, tracking_record: dict[str, Any] | None,
    expected_contract_sha256: str | None,
) -> bool:
    """OSC pose receipts need control-only tracking evidence."""

    pose_action = (
        isinstance(receipt_action, str)
        and receipt_action in {"move_pose", "servo_delta"}
    )
    tracking_contract_valid = False
    if isinstance(tracking_record, Mapping):
        tracking_contract_valid = (
            tracking_record.get("schema")
            == "ora.libero-pose-tracking-evidence.v1"
            and tracking_record.get("policy_visibility")
            == "control_only"
            and tracking_record.get("compiled_pose_sha256")
            == receipt.get("compiled_pose_sha256")
        )
    return (
        receipt_schema in {
            "ora.libero-pro-motion-receipt.v2",
            "ora.libero-pro-hybrid-motion-receipt.v1",
            "ora.libero-pro-hybrid-motion-receipt.v2",
        }
        and tracking_contract_valid
        and (
            not pose_action
            or all(
                isinstance(receipt.get(field), str)
                and len(receipt[field]) == 64
                for field in (
                    "pose_contract_sha256",
                    "compiled_pose_sha256",
                )
            )
        )
    )


def _eligibility_record(reasons: list[str]) -> dict[str, Any]:
    """Assemble the learning-eligibility verdict."""

    eligibility = {
        "schema": "ora.learning-eligibility.v1",
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "policy": "sealed_policy_visible_control_qualified_episode_v1",
    }
    return eligibility
