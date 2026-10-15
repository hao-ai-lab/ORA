"""Quantitative delivery guards: an enforced limit must reach the policy.

`test_tool_help_contract.py` checks that help exists and keeps its shape. These
guards check the *numbers*: a bound the Runtime enforces, a tier the policy
selects by name, and a comparison a public failure code reports must all reach
the model as values it can act on. Each guard names the confirmed defect it
would have caught.

The guards run against the profiles this repository actually ships, not
fixtures, so a new backend or profile version is covered on the day it lands.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Mapping

import pytest

ROOT = Path(__file__).resolve().parents[3]


# Every public motion failure code names a comparison the Runtime made. The
# registry records which measured numbers must accompany it so the policy can
# act, or states why none apply. Adding a code without a decision fails here.
FAILURE_CODE_EVIDENCE = {
    "planner_error": (),          # our software failed; no policy-actionable number
    "planner_failed": (),         # ordinary search failure; the target is the lever
    "invalid_start_state": (),
    "blocked_recovery_direction_unavailable": (),
    "endpoint_residual_exceeded": ("endpoint_error", "endpoint_tolerance"),
    # Cartesian servo stalls carry stall_progress_m against its minimum. The
    # RoboDojo joint stall (environment/backends/robodojo/motion_stall.py)
    # carries last_joint_step_rad and still_native_actions in motion_progress
    # against the rule's joint_step_rad and window_native_actions, which the
    # move_pose help states and recovery_progress.stall_rule repeats.
    "controller_stalled": ("stall_progress_m", "stall_minimum_progress_m",
                           "last_joint_step_rad", "still_native_actions",
                           "joint_step_rad", "window_native_actions"),
    # A jaw command that stopped short: gripper_outcome carries both openings.
    "gripper_stalled_before_target": (
        "measured_opening_normalized", "commanded_opening_normalized",
    ),
    "controller_timeout": ("elapsed_physics_steps", "maximum_physics_steps"),
    "inter_arm_collision": ("current_center_distance_m", "required_center_distance_m"),
    "payload_collision": (),
    "invalid_measured_state": (),
    "joint_target_rejected": (),
    "blocked": ("realized_translation_world_m", "requested_translation_world_m"),
    "contact_stop": ("external_force_magnitude_n", "force_stop_threshold_n"),
    "controller_tracking_error": (
        "maximum_joint_tracking_error_rad", "joint_tracking_error_limit_rad",
    ),
    "safety_guard": ("violations", "workspace_bounds_m"),
    "gripper_not_reached": ("gripper_error", "gripper_targets_effective"),
    "episode_terminal_during_execution": (),  # official lifecycle stop, no numeric comparison
    # RoboDojo recovery with no self-clear unwind order (before motion or from the
    # measured joints mid-recovery): recovery_progress.path_screens carries each
    # screened segment's min_clearance_m and limiting link pair.
    "recovery_self_collision_path": ("min_clearance_m",),
    # A RoboDojo recovery step that would meet the other arm (E11G12-004a):
    # recovery_progress.inter_arm_clearance (before motion) or the refused
    # segment (mid-recovery) carries the sphere clearance found, the clearance
    # required and the limiting link of each arm.
    "recovery_inter_arm_collision_path": ("min_clearance_m", "required_clearance_m", "limiting_links"),
    "execution_fault": (),      # catch-all for an unclassified controller stop
}


def _returned_literals() -> set[str]:
    source = (ROOT / "ora/agent/native/receipt_projection.py").read_text(
        encoding="utf-8",
    )
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "public_motion_failure_code"
    )
    literals = {
        node.value.value for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    passthrough = {
        element.value for node in ast.walk(function)
        if isinstance(node, ast.Compare)
        for comparator in node.comparators
        if isinstance(comparator, ast.Set)
        for element in comparator.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }
    # `raw` is returned unchanged only for the membership sets that guard it.
    return literals | passthrough


def test_every_public_failure_code_declares_the_numbers_that_explain_it() -> None:
    """A code naming a comparison must ship the value and the limit compared.

    `endpoint_residual_exceeded` without the residual and the tolerance cannot
    tell a 16 mm miss against a 15 mm gate from a 90 mm miss, so the policy
    re-sends the identical target instead of nudging once.
    """

    emitted = {
        code for code in _returned_literals()
        # Private planner strings are matched, not emitted.
        if code not in {
            "planner_internal_error", "planner_timeout", "local_servo_stalled",
            "local_servo_max_iterations", "waypoint_tracking_timeout",
            "terminal_endpoint_tracking_timeout", "terminal_tracking_timeout", "execution_step_budget_exhausted",
            "native_action_budget_exhausted",  # normalized to controller_timeout
            "sustained_joint_tracking_error", "waypoint_tracking_error",
        }
    }
    undeclared = sorted(emitted - set(FAILURE_CODE_EVIDENCE))
    assert not undeclared, (
        "these public failure codes ship without deciding which measured "
        f"numbers must accompany them: {undeclared}"
    )


# Every RoboDojo IK candidate rejection names a comparison. The compact
# ik_selection a refused segment carries to the policy must hold the numbers
# behind it, or the entry states why none apply. A rejection code added to
# backends/robodojo/ik.py without a decision here fails.
IK_REJECTION_EVIDENCE = {
    "nonfinite_joint_target": (),         # the solver returned no number to compare
    "sdk_not_feasible_or_converged": (),  # the SDK's own verdict
    "joint_limits": (),                   # per-candidate values stay Control-side in ik_attempts
    "fk_residual_exceeded": (),           # per-candidate residuals stay Control-side in ik_attempts
    "joint_jump": ("limiting_joint", "limiting_joint_delta_rad", "joint_jump_limit_rad"),
    "self_collision_path": (),            # clearances stay Control-side; the recovery text explains
    # The other arm met along the joint path (E11G12-004a): the limiting links,
    # the sphere clearance found and the clearance required.
    "inter_arm_collision_path": ("inter_arm_collision_path_count", "inter_arm_clearance"),
    # A pitch joint (X5 joint2-joint5) carried out of [-pi, pi] or farther out:
    # the joint, its measured and commanded values and the interval.
    "pitch_joint_wrap": ("pitch_joints", "pitch_joint_wrap"),
}


def _ik_rejection_literals() -> set[str]:
    source = (ROOT / "ora/environment/backends/robodojo/ik.py").read_text(encoding="utf-8")
    return {
        node.value.value for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        and any(isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
                and target.slice.value == "rejection" for target in node.targets)
    }


def test_every_robodojo_ik_rejection_declares_the_numbers_that_explain_it() -> None:
    literals = _ik_rejection_literals()
    assert literals, "no IK rejection literal found; this guard would pass vacuously"
    assert literals == set(IK_REJECTION_EVIDENCE), (
        f"undeclared {sorted(literals - set(IK_REJECTION_EVIDENCE))}, "
        f"stale {sorted(set(IK_REJECTION_EVIDENCE) - literals)}"
    )


def test_a_pitch_joint_wrap_refusal_delivers_the_joint_values_and_interval() -> None:
    import math

    from ora.environment.backends.robodojo import ik

    names = [f"joint{index}" for index in range(1, 7)]
    current = [0.0, 0.5, 2.9, 0.0, 0.0, 0.0]
    result = ik.select_candidate_set(
        candidates={"solutions": [[0.0, 0.5, 3.3, 0.0, 0.0, 0.0]], "joint_names": names,
                    "success": [True], "feasible": [True], "revolute": [True] * 6,
                    "pitch": [name in ik.ARX_X5_PITCH_JOINT_NAMES for name in names],
                    "position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.02},
        current=current, limits=[[-10.0, 10.0]] * 6, target=[0, 0, 0, 0, 0, 0, 1],
        fk=lambda q: [0, 0, 0, 0, 0, 0, 1], jump_limit_rad=math.pi)
    summary = result["ik_selection"]
    assert summary["outcome"] == "pitch_joint_wrap"
    assert all(summary.get(field) is not None for field in IK_REJECTION_EVIDENCE["pitch_joint_wrap"])
    evidence = summary["pitch_joint_wrap"]
    assert evidence["joint"] == "joint3" and evidence["current_rad"] == 2.9
    assert evidence["target_rad"] == pytest.approx(3.3)
    assert evidence["canonical_interval_rad"] == pytest.approx([-math.pi, math.pi])


# DS-R06: every maintained default now delivers current calibration.
CAMERA_GEOMETRY_KNOWN_GAPS = set()


def test_current_camera_geometry_reaches_the_native_observation_message(tmp_path):
    """Camera geometry must accompany the matching image, not stay file-only.

    This uses the production snapshot and observation renderer with public,
    nontrivial calibration. A new maintained backend cannot silently inherit
    the gap, and repairing a known backend requires removing its ratchet entry.
    """
    from ora.agent.composition.baseline import load_native_baseline
    from ora.agent.context.projection import causal_provider_value
    from tests.agent.context.test_current_visual_delivery import published, planner_for

    observed = set()
    for backend, instance in load_native_baseline(ROOT)["instances"].items():
        root = tmp_path / backend
        root.mkdir()
        context, _, _, surface, _ = published(root, backend)
        planner = planner_for(root, context, surface)
        message = planner._rewrite_causal_observation_message(context=context, terminal_lifecycle=False)
        text_blocks = [block["text"] for block in message["content"] if block["type"] == "text"]
        assert any(block["type"] == "image_url" for block in message["content"])
        envelope = json.loads(text_blocks[0])
        camera = instance["runtime"]["main_camera"]
        assert envelope["current_actual"]["view_dimensions"][camera] == {"width": 4, "height": 3}
        assert envelope["current_actual"]["public_robot_facts"] == causal_provider_value(
            context["public_robot_facts"])
        # The repair may publish full matrices or camera origin/axes and
        # intrinsics; either must retain these exact public numeric witnesses.
        wire = json.dumps(envelope["current_actual"])
        if not all(number in wire for number in ("50.25", "0.123", "0.234", "1.345")):
            observed.add(backend)
    assert observed == CAMERA_GEOMETRY_KNOWN_GAPS, (
        f"camera geometry delivery gaps changed: observed={sorted(observed)}, "
        f"known={sorted(CAMERA_GEOMETRY_KNOWN_GAPS)}; add a source-bound case for "
        "a new gap, or remove a repaired entry so the guard protects the fix"
    )



def test_current_camera_geometry_reaches_native_slots_v4_through_the_round_slot(tmp_path):
    """D3b moves unchanged calibration into the round slot, never out of the request."""
    from ora.agent.composition.baseline import load_native_baseline
    from ora.agent.context import round_static_facts as slot
    from ora.agent.native.engine import NativeSessionEngine
    from tests.agent.context.test_current_visual_delivery import published, planner_for

    for backend in load_native_baseline(ROOT)["instances"]:
        root = tmp_path / backend
        root.mkdir()
        context, _, _, surface, _ = published(root, backend)
        context = {**context, "remaining_budget": {"session_round": 1}}
        planner = planner_for(root, context, surface)
        planner.engine = NativeSessionEngine()
        planner.actual_sequence = 0
        planner.context_composition = slot.COMPOSITION
        message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
        planner._append_round_static_facts({"event_id": "agent-event-000001"})
        wire = json.dumps([message["content"][0]["text"], planner.messages[-1]["content"][0]["text"]])
        assert all(number in wire for number in ("50.25", "0.123", "0.234", "1.345")), backend


@pytest.mark.parametrize('comparison,within', [('<', False), ('<=', True)])
def test_gripper_acceptance_delivers_exact_comparison_and_detects_mutation(comparison, within):
    from copy import deepcopy
    from ora.contracts.acceptance import gripper_acceptance_check, validate_motion_acceptance_checks
    from ora.agent.native.receipt_projection import public_acceptance_checks, ReceiptProjectionContext
    check = gripper_acceptance_check(errors={'right': .03}, tolerance_normalized=.03, comparison=comparison)
    assert check['arms']['right']['within_tolerance']['opening_normalized'] is within
    context = ReceiptProjectionContext(None, 'set_gripper', lambda _: True)
    assert public_acceptance_checks({'acceptance_checks': [check]}, context)['acceptance_checks'] == [check]
    damaged = deepcopy(check)
    del damaged['tolerance']['opening_normalized']
    with pytest.raises(ValueError):
        validate_motion_acceptance_checks([damaged])
    damaged = deepcopy(check)
    damaged['arms']['right']['within_tolerance']['opening_normalized'] = not within
    with pytest.raises(ValueError):
        validate_motion_acceptance_checks([damaged])
    assert public_acceptance_checks({}, context)['acceptance_evidence_status'] == 'unavailable_in_this_receipt'


@pytest.mark.parametrize('damage', ['missing_target', 'bad_residual', 'bad_command', 'private_field', 'nan', 'bool'])
def test_gripper_feedback_rejects_missing_or_contradictory_evidence(damage):
    from ora.contracts.gripper_feedback import gripper_feedback, validate_gripper_feedback
    feedback = gripper_feedback(requested_opening_normalized=0., drive_target_opening_normalized=.8,
        measured_opening_normalized=1., effective_target_opening_normalized=.01/.054,
        tolerance_normalized=.03, drive_target_source='resolved_sdk_joint_position_target',
        measurement_source='primary_finger_encoder', pose_hold_semantics='preserve_last_applied_drive_target')
    assert validate_gripper_feedback(feedback) == feedback
    if damage == 'missing_target': del feedback['effective_target_opening_normalized']
    elif damage == 'bad_residual': feedback['arrival_check']['residual_normalized'] = 0.
    elif damage == 'bad_command': feedback['requested_command'] = 'open'
    elif damage == 'private_field': feedback['object_gt'] = 'private'
    elif damage == 'nan': feedback['drive_target_opening_normalized'] = float('nan')
    else: feedback['requested_opening_normalized'] = True
    with pytest.raises(ValueError): validate_gripper_feedback(feedback)


def test_unknown_gripper_intent_never_infers_arrival_from_the_drive_target():
    from ora.contracts.gripper_feedback import gripper_feedback
    feedback = gripper_feedback(requested_opening_normalized=None, drive_target_opening_normalized=1.,
        measured_opening_normalized=1., effective_target_opening_normalized=None,
        tolerance_normalized=.03, drive_target_source='resolved_sdk_joint_position_target',
        measurement_source='primary_finger_encoder', pose_hold_semantics='preserve_last_applied_drive_target')
    assert feedback['requested_command'] == 'unknown'
    assert feedback['arrival_check']['residual_normalized'] is None
    assert feedback['arrival_check']['within_tolerance'] is None


def test_no_selectable_surface_publishes_an_aperture_ceiling_below_what_the_compiler_admits():
    """The number a grasp is admitted against and the number it is published under.

    EXP10A raised the compiler's admission threshold from the 0.055 literal to
    the value the pinned ALOHA URDF and config derive, and left the published
    result schema at 0.055. The compiler then admitted candidates its own
    result schema refused, so twenty-one grasp_plan calls across eleven cells
    died with internal_tool_error naming
    ``$result.candidates[0].gripper_opening_m violates result maximum 0.055``.
    Fifteen of the twenty-one killed the top-ranked candidate.

    That is strictly worse than the defect it replaced: the old behaviour
    rejected one candidate with a named aperture reason the policy could act
    on, and the new one returned no candidates at all. One physical quantity
    declared in two places, where changing one cannot make the other complain.
    """

    from ora.agent.composition.profile_catalog import PROFILE_CATALOG
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.tools import grasp
    from ora.agent.composition.tool_surface import ToolSurfaceError

    below = {}
    for profile_id in sorted(PROFILE_CATALOG):
        try:
            surface = resolve_tool_surface(profile_id)
        except ToolSurfaceError:
            continue
        schema = surface.result_schemas.get("grasp_plan")
        if not isinstance(schema, Mapping):
            continue
        candidate = (
            schema.get("$defs", {}).get("candidate", {})
            .get("properties", {}).get("gripper_opening_m", {})
        )
        published = candidate.get("maximum")
        if not isinstance(published, (int, float)):
            continue
        # The threshold is embodiment-derived, so only the surfaces bound to
        # this embodiment are compared against it. A LIBERO Panda surface
        # publishing 0.08 is not this defect.
        if "robotwin" not in profile_id:
            continue
        if published < grasp.ALOHA_NOMINAL_MAXIMUM_OPENING_M:
            below[profile_id] = (surface.revision, published)

    assert below == {}, (
        "these selectable surfaces publish a grasp.plan gripper_opening_m ceiling "
        f"below the {grasp.ALOHA_NOMINAL_MAXIMUM_OPENING_M} m the compiler admits "
        "against, so a candidate the compiler accepts fails the surface's own "
        f"result schema and the whole call returns internal_tool_error: {below}"
    )


#: W2-005: the aperture the compiler admits against is not stated on the card
#: the policy reads. Repairing it needs a new embodiment descriptor and a new
#: environment prompt profile, and the ToolSurface that pins the current pair
#: also serves two other experiment packages, so the gap is registered here
#: rather than closed. Remove the bundle from this set when its card carries
#: the number; frozen bundles that predate the derivation are not listed
#: because the guard only reads the cards current bundles actually deliver.
APERTURE_GAP_BUNDLES = {
    # direct_v4 (integration 0059 unit A0) keeps the same ALOHA card
    # (manual-public-geometry-v3, descriptor v5 session-split): the gap is
    # carried over, not closed, until a new descriptor version states the number.
    f"current_v4/robotwin-aloha-{runtime}-v4.yaml" for runtime in ("native_api",)
}


def _delivered_aloha_cards() -> dict[str, Path]:
    """The ALOHA cards current prompt bundles actually name, not every card on disk.

    A frozen card is frozen: requiring it to carry a number derived later would
    be asking history to change. What matters is the card a fresh consumer gets.
    """

    import re

    root = Path(__file__).resolve().parents[3] / "ora/policy/prompt_bundles"
    delivered: dict[str, Path] = {}
    for bundle in sorted(root.glob("current_*/*.yaml")):
        text = bundle.read_text(encoding="utf-8")
        if "aloha-agilex" not in text:
            continue
        for match in re.findall(
            r"ora/policy/embodiments/aloha-agilex/(manual-[\w.-]+\.md)", text
        ):
            key = f"{bundle.parent.name}/{bundle.name}"
            delivered[key] = root.parents[1] / "policy/embodiments/aloha-agilex" / match
    return delivered


def test_the_embodiment_card_states_the_aperture_the_compiler_admits_against():
    """The number that decides a grasp, on the surface the policy reads.

    EXP10A raised the admission threshold to the value the pinned URDF and
    config derive, and published it through the result schema that caps
    gripper_opening_m. Neither reaches the policy: tool definitions carry only
    input schemas, and the card asserts no aperture at all -- it states the
    closed mesh gap of 4.8964 mm and says that is "not a general minimum grasp
    width", while the joint travel the derivation needs appears nowhere.

    Across the 25 sealed g002 cells, 21 runs made 117 statements about the
    maximum aperture holding 26 distinct values between 50 and 100 mm,
    inferring it from whichever candidate opening they happened to see.
    place_a2b_left concluded "7.8cm opening needed; the gripper max opening
    ~7.5cm per G1 ... too complex" and abandoned a grasp the 94.896 mm jaw
    would have made.

    The guard is one line: the card a current bundle delivers must contain the
    number the compiler admits against.
    """

    from ora.agent.tools import aloha_gripper_geometry as geometry

    aperture = geometry.ALOHA_NOMINAL_MAXIMUM_OPENING_M
    spellings = (f"{aperture}", f"{aperture * 1000:g}", f"{round(aperture * 1000, 3)}")
    delivered = _delivered_aloha_cards()
    assert delivered, "no current bundle names an ALOHA card; the guard reads nothing"

    missing = {
        bundle for bundle, card in delivered.items()
        if not any(s in card.read_text(encoding="utf-8") for s in spellings)
    }
    unexpected = missing - APERTURE_GAP_BUNDLES
    assert not unexpected, (
        f"these current bundles deliver an ALOHA card that does not state the "
        f"{aperture} m aperture the grasp compiler admits against, so a policy "
        f"reading them can only infer it from candidate openings: {sorted(unexpected)}"
    )
    repaired = APERTURE_GAP_BUNDLES - missing
    assert not repaired, (
        f"these bundles now state the aperture: {sorted(repaired)}. Remove them "
        "from APERTURE_GAP_BUNDLES so the guard starts enforcing it."
    )
