"""Fixed prose a tool's description states once (``observation_delivery_trim_v2``).

A result or action effect repeated the same ``claim_boundary``, ``evidence_scope``
or ``note`` sentence on every call: 374 characters of observed-clearance claim on
every motion result, 281 of RoboDojo IK scope on every move. Each entry below is
one such text, copied verbatim from the module that writes it (``producer``),
with the tool whose description layer (``policy/tool_surfaces/delivery_trim_v2``)
states it once and the backends whose v2 surface carries it. The delivered copy
leaves a value out only when it equals an entry's text exactly, in its written
or degree-converted form; any other text under those keys stays.

``tests/agent/context/test_delivery_trim_prose.py`` holds every entry to its producer and to
the composed description of each backend's v2 surface. The agent side keeps
its own copy rather than importing the environment backends that write some of
these texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

#: Result and effect keys whose fixed text a description can state once.
PROSE_KEYS = frozenset({"claim_boundary", "evidence_scope", "note"})


@dataclass(frozen=True)
class FixedProse:
    """One fixed text, where it appears and which description states it."""

    name: str
    key: str
    tool: str
    backends: tuple[str, ...]
    field: str
    producer: str
    text: str


ENTRIES: tuple[FixedProse, ...] = (
    FixedProse(
        "OBSERVED_CLEARANCE", "claim_boundary", "move_pose", ("robodojo", "robotwin", "libero_pro"),
        "compiled_pose.targets.<arm>.observed_clearance.claim_boundary (move_pose, move_to, rotate_wrist and rotate_pitch)",
        "agent/tools/observed_clearance.py CLAIM_BOUNDARY",
        "From the current depth images and their calibration only; points inside the measured "
        "jaws, the held object and the space above them are excluded as robot, other robot "
        "links are not; occluded surfaces and object undersides are unknown; points up to the "
        "footprint's top within radius_m count; not a collision check or a contact "
        "prediction, and the move is submitted unchanged."),
    FixedProse(
        "ROBODOJO_ONE_ARM_IK_SCOPE", "evidence_scope", "move_pose", ("robodojo",),
        "last_action_effect.detail.arms.<arm>.planning_diagnostics.evidence_scope of a one-arm move",
        "environment/backends/robodojo/ik.py INTER_ARM_PATH_EVIDENCE_SCOPE",
        "One bounded SDK IK solve, endpoint FK checks and the SDK collision spheres sampled "
        "along the straight joint interpolation for self-collision and against the other arm "
        "at its measured joints (finger links are not in the sphere model); no scene, table, "
        "payload or task-success proof."),
    FixedProse(
        "ROBODOJO_BOTH_ARMS_IK_SCOPE", "evidence_scope", "move_pose", ("robodojo",),
        "last_action_effect.detail.arms.<arm>.planning_diagnostics.evidence_scope of a two-arm move",
        "environment/backends/robodojo/ik.py PATH_EVIDENCE_SCOPE",
        "One bounded SDK IK solve, endpoint FK checks and the SDK collision spheres sampled "
        "along the straight joint interpolation for self-collision; no scene, table, payload "
        "or task-success proof."),
    FixedProse(
        "ROBODOJO_REACH", "claim_boundary", "move_pose", ("robodojo",),
        "compiled_pose.targets.<arm>.reach_along_request.claim_boundary of an unsolved target",
        "agent/tools/move_pose_reach.py REACH_CLAIM",
        "Read-only IK on the route move_pose takes, from the measured joints, at sampled "
        "points of the straight interpolation to the target; reachability is not proven "
        "monotone between samples, and no path, collision, payload or arrival claim."),
    FixedProse(
        "ROBODOJO_NEAREST_ORIENTATION", "claim_boundary", "move_pose", ("robodojo",),
        "reach_along_request.nearest_reachable_orientation.claim_boundary",
        "agent/composition/reach_orientation_schema.py CLAIM",
        "Read-only IK at each candidate endpoint on the route this move takes, from the "
        "measured joints: the requested orientation turned about axes perpendicular to its "
        "approach axis, holding the nominal jaw centre; no path, collision, payload or "
        "arrival claim, and an orientation outside the sweep may solve."),
    FixedProse(
        "ROBOTWIN_REACH", "claim_boundary", "move_pose", ("robotwin",),
        "compiled_pose.targets.<arm>.reach_along_request.claim_boundary of a solved target",
        "environment/backends/robotwin/reach_inter_arm.py CLAIM_BOUNDARIES[ENDPOINT]",
        "One damped-least-squares IK descent from the measured joints (or a chained query's "
        "earlier solution) with joint limits, then the configuration it reached screened "
        "against the other arm held at its measured joints with move_pose's link-sphere "
        "inter-arm clearance, at that configuration only and not along any path to it: no "
        "other collision model, no route model, and not the 128-seed collision-checked CuRobo "
        "IK move_pose plans through, which may reach the pose in another configuration, so "
        "move_pose can still refuse a reachable pose or reach one reported unreachable."),
    FixedProse(
        "ROBOTWIN_REACH_UNSOLVED", "claim_boundary", "move_pose", ("robotwin",),
        "compiled_pose.targets.<arm>.reach_along_request.claim_boundary of an unsolved target",
        "agent/tools/reach_reference.py SAMPLING_CLAIM + ' ' + the RoboTwin claim",
        "Read-only IK at sampled points of the straight interpolation from the measured pose "
        "to the target; reachability is not proven monotone between samples, and no path, "
        "collision, payload or arrival claim. One damped-least-squares IK descent from the "
        "measured joints (or a chained query's earlier solution) with joint limits, then the "
        "configuration it reached screened against the other arm held at its measured joints "
        "with move_pose's link-sphere inter-arm clearance, at that configuration only and not "
        "along any path to it: no other collision model, no route model, and not the 128-seed "
        "collision-checked CuRobo IK move_pose plans through, which may reach the pose in "
        "another configuration, so move_pose can still refuse a reachable pose or reach one "
        "reported unreachable."),
    FixedProse(
        "LIBERO_REACH", "claim_boundary", "move_pose", ("libero_pro",),
        "compiled_pose.targets.<arm>.reach_along_request.claim_boundary of a solved target",
        "environment/backends/libero_pro/reach_check.py CLAIM_BOUNDARY",
        "One CuRobo v0.7.8 IK solve per pose by the IK solver of the planner that move_pose's "
        "global segment uses, seeded from the measured joints or a chained query's solution, "
        "with joint limits and Panda self-collision only: no table, object or scene model. "
        "The solution's grip site is checked against the 0.005 m / 0.05 rad arrival bound. It "
        "plans no trajectory and does not model the bounded OSC route, so move_pose can still "
        "refuse or fail to reach a pose reported reachable, and a pose reported unreachable "
        "may still be reachable: the seed search is finite."),
    FixedProse(
        "LIBERO_REACH_UNSOLVED", "claim_boundary", "move_pose", ("libero_pro",),
        "compiled_pose.targets.<arm>.reach_along_request.claim_boundary of an unsolved target",
        "agent/tools/reach_reference.py SAMPLING_CLAIM + ' ' + the LIBERO claim",
        "Read-only IK at sampled points of the straight interpolation from the measured pose "
        "to the target; reachability is not proven monotone between samples, and no path, "
        "collision, payload or arrival claim. One CuRobo v0.7.8 IK solve per pose by the IK "
        "solver of the planner that move_pose's global segment uses, seeded from the measured "
        "joints or a chained query's solution, with joint limits and Panda self-collision "
        "only: no table, object or scene model. The solution's grip site is checked against "
        "the 0.005 m / 0.05 rad arrival bound. It plans no trajectory and does not model the "
        "bounded OSC route, so move_pose can still refuse or fail to reach a pose reported "
        "reachable, and a pose reported unreachable may still be reachable: the seed search "
        "is finite."),
    FixedProse(
        "NEAREST_ORIENTATION", "claim_boundary", "move_pose", ("robotwin", "libero_pro"),
        "reach_along_request.nearest_reachable_orientation.claim_boundary",
        "agent/composition/reach_reference_schema.py NEAREST_CLAIM",
        "Read-only IK at each candidate endpoint by the same reach check as "
        "reach_along_request (its ik_model, matches_move_pose_admission and "
        "measured_disagreement apply), from the measured joints: the requested orientation "
        "turned about axes perpendicular to its approach axis, holding the jaw centre; no "
        "path, collision, payload or arrival claim, and an orientation outside the sweep may "
        "solve."),
    FixedProse(
        "LIBERO_PLANNING_NOTE", "note", "move_pose", ("libero_pro",),
        "last_action_effect.detail.planning_diagnostics.note",
        "environment/backends/libero_pro/planning_diagnostics.py (inline)",
        "Counts overlap; raw joint bounds do not isolate self-collision or other constraints. "
        "World collision model is absent."),
    FixedProse(
        "SEGMENTATION", "claim_boundary", "vision_segment", ("robodojo", "robotwin", "libero_pro"),
        "the result's claim_boundary",
        "agent/tools/segmentation_batch.py (inline)",
        "independent point-prompt proposals with a per-point mask budget; one point never "
        "consumes another point's masks, an empty point means no mask at that pixel rather "
        "than an exhausted budget, and omitted_over_* counts every mask withheld; inspect the "
        "proposals and choose semantic identity; crops add no sensor detail"),
    FixedProse(
        "GRIPPER_STALL", "claim_boundary", "set_gripper", ("robodojo", "libero_pro"),
        "the action effect's gripper_outcome.claim_boundary",
        "environment/controllers/gripper_stall.py GRIPPER_STALL_CLAIM_BOUNDARY",
        "stalled_before_target means the jaws stopped short of the commanded opening; it is "
        "consistent with an object between the fingers and is not proof of a grasp"),
    FixedProse(
        "GRASP_PLAN_IK_REFERENCE", "claim_boundary", "grasp_plan", ("robodojo", "robotwin", "libero_pro"),
        "ik_reachability.claim_boundary",
        "agent/tools/reach_check.py GRASP_ANNOTATION_CLAIM",
        "IK reference at planning time for the compiled pregrasp (free_space from the "
        "measured joints) and approach (contact_approach from that pregrasp solution), trying "
        "the exact orientation and its 180-degree jaw flip; not a guarantee. Execution runs "
        "its own backend check, and any world action stales this."),
    FixedProse(
        "GRASP_PLAN_JAW_SYMMETRY", "claim_boundary", "grasp_plan", ("robodojo",),
        "ik_reachability.jaw_symmetry.claim_boundary",
        "agent/composition/grasp_jaw_symmetry_schema.py CLAIM_BOUNDARY",
        "Joint-limit margins of the backend's read-only IK solutions for the published "
        "pregrasp and contact poses at planning time; joint6 is the X5 wrist roll about the "
        "approach axis and is reported only when the measured joints prove it is the binding "
        "joint. An IK reference, not a carry guarantee: execution solves again, and lifting, "
        "carrying or rotating the wrist after the grasp moves every margin."),
    FixedProse(
        "GRASP_EXECUTE", "claim_boundary", "grasp_execute", ("robodojo", "robotwin"),
        "the result's claim_boundary",
        "agent/tools/skill_driver.py CLAIM_BOUNDARY",
        "one committed sequence of trusted executions and the public measurements taken after "
        "it; motion arrival and jaw separation beside visible extents are evidence, not "
        "grasp, placement, retention or official task success"),
    FixedProse(
        "GRASP_EXECUTE_PRECOMMIT", "claim_boundary", "grasp_execute", ("robodojo", "robotwin"),
        "precommit_motion_check.claim_boundary (also in a grasp_pregrasp_unreachable refusal)",
        "agent/tools/grasp_feasibility.py (inline)",
        "Current-state planning/admission for the probed stages, with the selected backend's "
        "collision coverage. Does not certify contact, lift, retention or task success; "
        "execution revalidates after opening."),
    FixedProse(
        "GRASP_VERIFICATION", "claim_boundary", "grasp_execute", ("robodojo", "robotwin"),
        "last_action_effect.detail.verification.claim_boundary",
        "agent/tools/localization.py (inline)",
        "jaw separation beside visible extents is evidence, not a holding verdict; no value "
        "of this pair establishes contact or retention"),
    FixedProse(
        "GRASP_LIFT_PROBE", "claim_boundary", "grasp_execute", ("robodojo", "robotwin"),
        "last_action_effect.detail.verification.lift_probe.claim_boundary",
        "agent/tools/skill_retention.py (inline)",
        "short_lift_evidence_only;not_attachment_or_stable_transport_or_task_success"),
    FixedProse(
        "GRASP_JAW_HOLD", "claim_boundary", "grasp_execute", ("robodojo",),
        "last_action_effect.detail.verification.jaw_hold.claim_boundary",
        "agent/tools/jaw_hold.py CLAIM_BOUNDARY",
        "normalized jaw-encoder readings under the backend drive limiter; not a calibrated "
        "aperture, contact, attachment, retention or task-success verdict"),
    FixedProse(
        "WAIT", "claim_boundary", "wait_ticks", ("robodojo", "robotwin", "libero_pro"),
        "last_action_effect.detail.wait.claim_boundary",
        "contracts/wait.py (inline)",
        "time passed with commands held; drift, settling, slip or scripted scene motion are "
        "measured evidence in the fresh observation, not effects of this action"),
    FixedProse(
        "ROBOTWIN_ALIGN", "claim_boundary", "align_plan", ("robotwin",),
        "candidates[].claim_boundary",
        "environment/controllers/reachability.py (inline)",
        "Inverse kinematics for this arm model only: one damped-least-squares descent from "
        "the seed configuration, with no collision model of any kind. A reachable pose is not "
        "a collision-free path and not a successful motion, and this verdict is not the "
        "backend planner's."),
    FixedProse(
        "ROBODOJO_ALIGN", "claim_boundary", "align_plan", ("robodojo",),
        "candidates[].claim_boundary",
        "environment/backends/robodojo/reachability.py (inline)",
        "SDK IK and FK only; no physics, collision-free path, contact or task-success claim"),
)


def canonical_texts() -> Mapping[str, frozenset[str]]:
    """Every canonical text, by the key it appears under."""
    texts: dict[str, set[str]] = {}
    for entry in ENTRIES:
        texts.setdefault(entry.key, set()).add(entry.text)
    return MappingProxyType({key: frozenset(values) for key, values in texts.items()})


__all__ = ["ENTRIES", "FixedProse", "PROSE_KEYS", "canonical_texts"]
