"""Pure RoboDojo motion/aperture qualification cases and evidence judgements."""
from __future__ import annotations
import math
from typing import Any, Mapping, Sequence
from ora.environment.controllers.robodojo_arx import ARX_X5_CONTRACT
from .sparse_motion import (
    GLOBAL_EXECUTOR, GRIPPER_EXECUTOR, JAW_SYMMETRY_AXIS, LOCAL_EXECUTOR, RECOVERY_EXECUTOR,
    ROUTE_REFUSAL_CODE, SPARSE_RECEIPT_SCHEMA, rotate_about_local_axis,
)

SCHEMA = "ora.robodojo-sparse-motion-qualification.v1"
MOTION_ACTION_V3_SCHEMA = "ora-motion-action.v3"
ROTATION_ONLY_DRIFT_BOUND_M = 0.03
REACH_POSITION_TOLERANCE_M = float(ARX_X5_CONTRACT["position_tolerance_m"])
REACH_ORIENTATION_TOLERANCE_RAD = float(ARX_X5_CONTRACT["orientation_tolerance_rad"])
INACTIVE_ARM_DRIFT_BOUND_M = 0.005
GRIPPER_TOLERANCE = float(ARX_X5_CONTRACT["gripper_tolerance_normalized"])
ARM = "left"
OTHER_ARM = "right"

# --- robodojo_x5_jaw_aperture_probe_v1 ---------------------------------------
#
# The jaw geometry the probe compares its measurement against, read from the
# same pinned Assets checkout the grasp compiler reads
# (a14409d7fae673c00499e01fd88b4457df6351b1, Assets/Robots/x5): joint7/joint8
# are prismatic on link6 at ``xyz = (0.08657, +/-0.0249, -0.00024)`` with axes
# ``+/-y``, and the link7/link8 inner pad faces sit at ``y = -/+0.02449`` in
# their own joint frames. The values are duplicated here rather than imported
# because ``ora/environment`` never imports ``ora/agent``;
# ``test_robodojo_backend`` asserts the two copies agree, so a drift fails the
# suite instead of producing two different "nominal" apertures.
JAW_APERTURE_PROBE_ID = "robodojo_x5_jaw_aperture_probe_v1"
JAW_CALIBRATION_SCHEMA = (
    "ora.robodojo-x5-jaw-aperture-calibration.v1"
)
JAW_COMMAND_JOINT_RANGE_M = tuple(
    float(value) for value in ARX_X5_CONTRACT["gripper_command"]["primary_joint_range"]
)
JAW_MESH_PAD_OFFSET_M = 0.02449
JAW_CLOSED_PAD_GAP_M = 0.0008
JAW_APERTURE_PER_JOINT_M = 2.0
JAW_NOMINAL_MAXIMUM_APERTURE_M = 0.0888
JAW_FINGER_JOINT_LIMIT_M = (0.0, 0.044)
#: Free-air pass bounds. The design spec's intermediate 0.75/0.5/0.25/0.178
#: commands are not expressible: the public ``set_gripper`` contract accepts
#: only ``open`` and ``close``, so a sweep through them would measure commands
#: no policy can ever issue. The probe measures the two endpoints the policy
#: actually commands, twice, and says so in its own record.
JAW_OPEN_APERTURE_TOLERANCE_M = 0.002
JAW_CLOSED_APERTURE_BOUND_M = 0.001
JAW_MIMIC_BOUND_M = 0.0005
JAW_SWEEP_AGREEMENT_M = 0.001
JAW_FINGER_UNDERSHOOT_BOUND_M = 0.0005
JAW_SWEEPS = 2
JAW_APERTURE_CASE_ID = "jaw_aperture_free_air_sweep"
#: What a free-air sweep cannot reach, and therefore what the calibration
#: record must keep declaring unmeasured: pad compliance under load, the
#: contact-limited stall, and the reading the policy would see with an object
#: between the jaws. Those are the only aperture facts a retention judgement
#: could rest on, so the record's ``contact_leg`` stays explicit rather than
#: absent, and ``contact_leg_passed`` gates the calibrated flag on the grasp
#: side.
JAW_CONTACT_LEG_STATUS = "not_measured"
JAW_CONTACT_LEG_REASON = (
    "a contact leg needs a scripted grasp of an object of source-known extent "
    "in a pinned qualification scene; it is specified but not implemented, so "
    "pad compliance and contact-limited stall stay unmeasured"
)
# X5A.urdf's base_link->link6 chain sums to 0.822 m of link translation
# (0.0605 + 0.0447 + 0.264 + 0.2513 + 0.1099 + 0.0912), an upper bound on how
# far link6 can stand from its own base mount at *any* joint vector, and
# env_cfg/robot/dual_x5.yml bolts the two fixed bases to the table's -y edge.
X5_LINK6_REACH_BOUND_M = 0.822
X5_BASE_MOUNTS_M = ((-0.3, -0.45, 0.765), (0.3, -0.45, 0.765))
# The two refusal cases have to reach the *planner*, so the target stays
# inside the live robodojo_arx_x5_tabletop_v1 action envelope
# (x [-0.75, 0.75], y [-0.65, 0.55], z [0.72, 1.40]) with a centimetre of
# margin on every axis, and is still kinematically hopeless: the far top
# corner diagonally opposite the left mount is 1.566 m from it and 1.251 m
# from the right one, over 1.5x the reach bound above and beyond it even if
# the whole 0.14 m jaw chain counted as reach.  The earlier [1.49, 1.49, 1.99]
# only refused at the planner under the robodojo_compat +/-1.5 m
# compatibility envelope; under the resolved profile the hard action envelope
# refuses it first (plan_rejected / workspace_target_x), which proves a
# different thing.
UNREACHABLE_POSITION = [0.74, 0.54, 1.39]
# A ``move_pose`` global route compiles a single ``move`` segment, so this is
# the SDK IK's own refusal code; a hard-envelope refusal would instead answer
# ``plan_rejected`` with ``workspace_target_<axis>``.
PLANNER_IK_REFUSAL_CODE = "ik_failed:move"


def _rotate_approach(orientation: Sequence[float], degrees: float) -> list[float]:
    return rotate_about_local_axis(orientation, JAW_SYMMETRY_AXIS, math.radians(degrees))


def _pose(position: Sequence[float], orientation: Sequence[float]) -> dict[str, list[float]]:
    return {"position_m": [round(float(v), 6) for v in position],
            "orientation_xyzw": [round(float(v), 8) for v in orientation]}


def _target(poses: Mapping[str, Sequence[float]], interaction: str) -> dict[str, Any]:
    return {"targets": {arm: _pose(pose[:3], pose[3:]) for arm, pose in poses.items()},
            "interactions": {arm: interaction for arm in poses}}


def build_case_arguments(case_id: str, poses: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Compile one case's v3 arguments from the measured start poses."""
    left = [float(v) for v in poses[ARM]]
    x, y, z = left[:3]
    quat = left[3:]
    if case_id == "free_space_translation_8cm":
        return _target({ARM: [x - 0.08, y + 0.06, z + 0.04, *quat]}, "free_space")
    if case_id == "free_space_lift_translate_descend":
        return _target({ARM: [x + 0.06, y + 0.08, z - 0.03, *quat]}, "free_space")
    if case_id == "combined_translation_rotation_45deg":
        return _target({ARM: [x, y - 0.06, z + 0.05, *_rotate_approach(quat, 45.0)]}, "free_space")
    if case_id == "orientation_only_small_12deg":
        # Comfortably inside the 15-degree local envelope so passive drift
        # between observation and execution cannot flip the route.
        return _target({ARM: [x, y, z, *_rotate_approach(quat, 12.0)]}, "precision")
    if case_id == "orientation_only_90deg":
        return _target({ARM: [x, y, z, *_rotate_approach(quat, 90.0)]}, "free_space")
    if case_id == "orientation_only_near_180deg":
        return _target({ARM: [x, y, z, *_rotate_approach(quat, 179.0)]}, "free_space")
    if case_id == "precision_small_local_6mm":
        return _target({ARM: [x, y, z + 0.006, *quat]}, "precision")
    if case_id == "contact_approach_small_8mm":
        return _target({ARM: [x, y, z - 0.008, *quat]}, "contact_approach")
    if case_id == "contact_approach_large_refused":
        return _target({ARM: [x, y, z - 0.20, *quat]}, "contact_approach")
    if case_id == "planner_refusal_unreachable":
        return _target({ARM: [*UNREACHABLE_POSITION, *quat]}, "free_space")
    if case_id == "below_tolerance_noop":
        return _target({ARM: [x, y, z, *quat]}, "precision")
    if case_id == "dual_arm_synchronized_lift":
        right = [float(v) for v in poses[OTHER_ARM]]
        return _target({ARM: [x, y, z + 0.04, *quat], OTHER_ARM: [right[0], right[1], right[2] + 0.04, *right[3:]]},
                       "free_space")
    raise ValueError(f"unknown qualification case: {case_id}")


MOTION_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "free_space_translation_8cm", "action": "move_pose",
     "expected_status": {"reached"}, "expected_strategy": "global", "expected_executor": GLOBAL_EXECUTOR},
    {"case_id": "free_space_lift_translate_descend", "action": "move_free_space",
     "expected_status": {"reached"}, "expected_strategy": "global", "expected_executor": GLOBAL_EXECUTOR,
     "expected_segments": ("lift", "translate", "descend")},
    {"case_id": "combined_translation_rotation_45deg", "action": "move_pose",
     "expected_status": {"reached"}, "expected_strategy": "global", "expected_executor": GLOBAL_EXECUTOR},
    {"case_id": "orientation_only_small_12deg", "action": "move_pose",
     "expected_status": {"reached"}, "expected_strategy": "local", "expected_executor": LOCAL_EXECUTOR,
     "rotation_only": True},
    {"case_id": "orientation_only_90deg", "action": "move_pose",
     "expected_status": {"reached", "not_reached"}, "expected_strategy": "global",
     "expected_executor": GLOBAL_EXECUTOR, "rotation_only": True},
    {"case_id": "orientation_only_near_180deg", "action": "move_pose",
     "expected_status": {"reached", "not_reached", "planner_failed", "plan_rejected"},
     "expected_strategy": None, "expected_executor": None, "rotation_only": True},
    {"case_id": "precision_small_local_6mm", "action": "move_pose",
     "expected_status": {"reached"}, "expected_strategy": "local", "expected_executor": LOCAL_EXECUTOR},
    {"case_id": "contact_approach_small_8mm", "action": "move_pose",
     "expected_status": {"reached", "not_reached"}, "expected_strategy": "local",
     "expected_executor": LOCAL_EXECUTOR},
    {"case_id": "contact_approach_large_refused", "action": "move_pose",
     "expected_status": {"plan_rejected"}, "expected_strategy": None, "expected_executor": None,
     "expected_failure_code": ROUTE_REFUSAL_CODE, "expect_no_write": True},
    {"case_id": "planner_refusal_unreachable", "action": "move_pose",
     "expected_status": {"planner_failed"}, "expected_strategy": None, "expected_executor": None,
     "expected_failure_code": PLANNER_IK_REFUSAL_CODE, "expect_no_write": True},
    {"case_id": "below_tolerance_noop", "action": "move_pose",
     "expected_status": {"reached"}, "expected_strategy": None, "expected_executor": None},
    {"case_id": "dual_arm_synchronized_lift", "action": "move_pose",
     "expected_status": {"reached"}, "expected_strategy": "global", "expected_executor": GLOBAL_EXECUTOR,
     "dual_arm": True},
)

# ``JAW_APERTURE_CASE_ID`` is deliberately absent. Its probe, its judge and
# their unit tests are implemented below, but the fake plant has no finger
# bodies, so the case has never been driven end to end and cannot be allowed
# to fail a live gate for a reason no test has exercised. The measured reach
# evidence also puts the aperture off the causal path for the failures that
# motivated it: every refused grasp candidate sat past the arm's dexterous
# workspace boundary, and no aperture check rejected a single candidate.
# Admit the case only together with a fixture that drives it.
SEQUENCE_CASES: tuple[str, ...] = (
    "recover_home_after_matrix",
    "gripper_close_holds_pose",
    "gripper_open_holds_pose",
    "recover_backtrack_after_move",
    "probe_then_plan_ref_execution",
    "plan_ref_survives_gripper_only",
    "plan_ref_expires_after_motion",
    "jaw_flip_variants_probed",
    "probe_consistency_unreachable",
    "official_preterminal_boundary",
    "official_terminal_boundary_intact",
    "round_reset_isolation",
)

ALL_CASE_IDS: tuple[str, ...] = tuple([case["case_id"] for case in MOTION_CASES] + list(SEQUENCE_CASES))


def _pose_error(target: Sequence[float], measured: Sequence[float]) -> tuple[float, float]:
    distance = math.sqrt(sum((float(target[i]) - float(measured[i])) ** 2 for i in range(3)))
    dot = abs(sum(float(target[3 + i]) * float(measured[3 + i]) for i in range(4)))
    return distance, 2.0 * math.acos(min(1.0, dot))


def _translation_drift(receipt: Mapping[str, Any], arm: str) -> float | None:
    before = receipt.get("before", {}).get(arm, {}).get("pose_xyz_xyzw")
    after = receipt.get("after", {}).get(arm, {}).get("pose_xyz_xyzw")
    if not (isinstance(before, list) and isinstance(after, list) and len(before) == 7 and len(after) == 7):
        return None
    return math.sqrt(sum((float(after[i]) - float(before[i])) ** 2 for i in range(3)))


def judge_motion_case(case: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Pass/fail one executed motion case using only public receipt fields."""
    notes: list[str] = []
    if receipt.get("schema") != SPARSE_RECEIPT_SCHEMA:
        return {"passed": False, "notes": [f"unexpected receipt schema: {receipt.get('schema')}"]}
    status = str(receipt.get("execution_status"))
    if status not in case["expected_status"]:
        notes.append(f"status {status} not in {sorted(case['expected_status'])}")
    strategy = case.get("expected_strategy")
    if strategy is not None and status == "reached" and receipt.get("motion_strategy") != strategy:
        notes.append(f"strategy {receipt.get('motion_strategy')} is not {strategy}")
    executor = receipt.get("executor")
    expected_executor = case.get("expected_executor")
    if expected_executor is not None and status == "reached" and (
        not isinstance(executor, list) or expected_executor not in executor
    ):
        notes.append(f"executor {executor} lacks {expected_executor}")
    if case.get("expected_failure_code") is not None and receipt.get("failure_code") != case["expected_failure_code"]:
        notes.append(f"failure_code {receipt.get('failure_code')}")
    refused = status in {"plan_rejected", "planner_failed"}
    if (case.get("expect_no_write") or refused) and (receipt.get("state_changed") or receipt.get("native_actions")):
        notes.append("refusal wrote the world")
    arms = receipt.get("arms") if isinstance(receipt.get("arms"), Mapping) else {}
    targeted = [ARM, OTHER_ARM] if case.get("dual_arm") else [ARM]
    for arm in targeted:
        effect = arms.get(arm) if isinstance(arms.get(arm), Mapping) else {}
        if effect.get("targeted") is not True:
            notes.append(f"{arm} is not marked targeted")
        pose_error = effect.get("pose_error")
        if status == "reached":
            if not isinstance(pose_error, Mapping):
                notes.append(f"{arm} reached without a pose error")
            elif (float(pose_error.get("translation_m", math.inf)) > REACH_POSITION_TOLERANCE_M
                  or float(pose_error.get("orientation_rad", math.inf)) > REACH_ORIENTATION_TOLERANCE_RAD):
                notes.append(f"{arm} reached with residual {pose_error}")
    for arm in (OTHER_ARM,) if not case.get("dual_arm") else ():
        drift = (arms.get(arm) or {}).get("inactive_arm_drift") if isinstance(arms.get(arm), Mapping) else None
        if not isinstance(drift, Mapping):
            notes.append(f"{arm} inactive drift is missing")
        elif float(drift.get("translation_m", math.inf)) > INACTIVE_ARM_DRIFT_BOUND_M:
            notes.append(f"{arm} moved {drift['translation_m']:.4f} m while not targeted")
    expected_segments = case.get("expected_segments")
    if expected_segments and status == "reached":
        executed = [item.get("segment") for item in (arms.get(ARM) or {}).get("segments", [])]
        if not set(executed).issubset(set(expected_segments)) or len(executed) < 2:
            notes.append(f"free-space segments {executed}")
    invariant = receipt.get("command_invariant")
    if not isinstance(invariant, Mapping) or invariant.get("verified") is not True:
        notes.append("command invariant not verified")
    if case.get("rotation_only"):
        drift = _translation_drift(receipt, ARM)
        if drift is None:
            notes.append("rotation-only case lacks before/after poses")
        elif drift > ROTATION_ONLY_DRIFT_BOUND_M:
            notes.append(f"2pi-alias monitor: pure rotation realized {drift:.3f} m translation drift")
    return {"passed": not notes, "notes": notes}


def judge_gripper_case(directive: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
    notes: list[str] = []
    if receipt.get("schema") != SPARSE_RECEIPT_SCHEMA:
        return {"passed": False, "notes": [f"unexpected receipt schema: {receipt.get('schema')}"]}
    if receipt.get("execution_status") != "reached":
        notes.append(f"gripper {directive} ended {receipt.get('execution_status')}")
    if receipt.get("executor") != [GRIPPER_EXECUTOR]:
        notes.append(f"executor {receipt.get('executor')}")
    effect = receipt.get("arms", {}).get(ARM, {}) if isinstance(receipt.get("arms"), Mapping) else {}
    error = effect.get("gripper_error")
    if not isinstance(error, (int, float)) or float(error) > GRIPPER_TOLERANCE:
        notes.append(f"gripper residual {error}")
    invariant = receipt.get("command_invariant")
    if not isinstance(invariant, Mapping) or invariant.get("verified") is not True:
        notes.append("command invariant not verified")
    if receipt.get("pose_targets_before") != receipt.get("pose_targets_after"):
        notes.append("gripper action changed retained pose targets")
    return {"passed": not notes, "notes": notes}


def judge_recovery_case(receipt: Mapping[str, Any], anchor: Sequence[float], *, arm: str = ARM) -> dict[str, Any]:
    notes: list[str] = []
    if receipt.get("schema") != SPARSE_RECEIPT_SCHEMA:
        return {"passed": False, "notes": [f"unexpected receipt schema: {receipt.get('schema')}"]}
    if receipt.get("execution_status") != "reached":
        notes.append(f"recovery ended {receipt.get('execution_status')}")
    if receipt.get("executor") != [RECOVERY_EXECUTOR]:
        notes.append(f"executor {receipt.get('executor')}")
    after = receipt.get("after", {}).get(arm, {}).get("pose_xyz_xyzw")
    if not isinstance(after, list) or len(after) != 7:
        notes.append("recovery lacks an after pose")
    else:
        distance, angle = _pose_error(anchor, after)
        if distance > REACH_POSITION_TOLERANCE_M or angle > REACH_ORIENTATION_TOLERANCE_RAD:
            notes.append(f"recovery residual {distance:.4f} m / {angle:.4f} rad from its anchor")
    return {"passed": not notes, "notes": notes}


def nominal_jaw_aperture_m(gripper_command: float) -> float:
    """Pad-face separation the pinned URDF/mesh geometry predicts for a command."""

    if not 0.0 <= float(gripper_command) <= 1.0:
        raise ValueError("gripper command must lie in [0, 1]")
    low, high = JAW_COMMAND_JOINT_RANGE_M
    joint = low + (high - low) * float(gripper_command)
    return max(
        0.0,
        round(JAW_APERTURE_PER_JOINT_M * joint + JAW_CLOSED_PAD_GAP_M, 6),
    )


def _jaw_axis_world(pose_xyz_xyzw: Sequence[float]) -> list[float]:
    """The controlled link's local +y expressed in world coordinates."""

    x, y, z, w = (float(pose_xyz_xyzw[index]) for index in (3, 4, 5, 6))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError("controlled-link orientation is not a rotation")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # Second column of the rotation matrix for xyzw.
    return [
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
    ]


def measured_pad_face_separation_m(reading: Mapping[str, Any]) -> float:
    """Private measured pad-face gap for one arm's raw jaw reading.

    The two finger bodies' world origins sit on the joint origins, which travel
    along the controlled link's local +/-y as the prismatic joints open. Their
    separation along that axis minus the two mesh pad offsets is the gap
    between the pad faces, and it is the only aperture number in this file that
    is a measurement rather than a prediction.
    """

    bodies = reading.get("finger_body_position_w_m")
    if not isinstance(bodies, Mapping) or len(bodies) != 2:
        raise ValueError("a jaw reading needs exactly two finger body positions")
    axis = _jaw_axis_world(reading["controlled_link_pose_xyz_xyzw"])
    first, second = (list(bodies[name]) for name in sorted(bodies))
    projected = sum(
        (float(first[index]) - float(second[index])) * axis[index]
        for index in range(3)
    )
    return round(abs(projected) - 2.0 * JAW_MESH_PAD_OFFSET_M, 9)


def _two_finger_mean_m(fingers: Sequence[float]) -> float:
    """Both finger joints' mean in the primary coordinate (``common.two_finger_opening``)."""
    command = ARX_X5_CONTRACT["gripper_command"]
    multiplier, offset = float(command["mimic_multiplier"]), float(command["mimic_offset"])
    return (float(fingers[0]) + (float(fingers[1]) - offset) / multiplier) / 2


def derive_jaw_step(
    *, command: float, reading: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one probe step's measured, nominal and error fields."""

    measured = measured_pad_face_separation_m(reading)
    nominal = nominal_jaw_aperture_m(command)
    fingers = [float(value) for value in reading["gripper_finger_qpos"]]
    targets = [
        float(value) for value in reading["applied_gripper_joint_targets"]
    ]
    low, high = JAW_COMMAND_JOINT_RANGE_M
    return {
        "commanded_directive": float(command),
        "gripper_command": float(reading["gripper_command"]),
        "gripper": float(reading["gripper"]),
        "gripper_finger_qpos": fingers,
        "applied_gripper_joint_targets": targets,
        "measured_aperture_m": measured,
        "nominal_aperture_m": nominal,
        "aperture_error_m": round(measured - nominal, 9),
        "mimic_error_m": round(abs(fingers[0] - fingers[1]), 9)
        if len(fingers) >= 2 else None,
        # The public normalized reading must reproduce both finger joints'
        # mean on the command scale (E11G12-001; the second finger mapped back
        # through the X5 mimic); a divergence means the published state and
        # the plant disagree.
        "normalized_reading_error": round(
            float(reading["gripper"]) - (_two_finger_mean_m(fingers) - low) / (high - low), 9,
        ),
        "physics_steps": reading.get("physics_steps"),
        "native_actions": reading.get("native_actions"),
    }


def judge_jaw_aperture_sweeps(
    sweeps: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Pass/fail the free-air legs of ``robodojo_x5_jaw_aperture_probe_v1``.

    Every criterion is measured, none is assumed. A pass certifies the
    actuator, the mimic and the closed floor in free air; it certifies nothing
    about contact, which is why the record keeps a separate contact leg.
    """

    notes: list[str] = []
    if len(sweeps) != JAW_SWEEPS or any(len(sweep) < 2 for sweep in sweeps):
        return {
            "passed": False,
            "notes": [f"expected {JAW_SWEEPS} sweeps of at least two steps"],
        }
    for index, sweep in enumerate(sweeps):
        commands = [float(step["commanded_directive"]) for step in sweep]
        if commands != sorted(commands, reverse=True):
            notes.append(f"sweep {index} is not a descending command sequence")
        apertures = [float(step["measured_aperture_m"]) for step in sweep]
        if any(
            later > earlier + JAW_SWEEP_AGREEMENT_M
            for earlier, later in zip(apertures, apertures[1:])
        ):
            notes.append(f"sweep {index} aperture is not monotone non-increasing")
        for step in sweep:
            command = float(step["commanded_directive"])
            mimic = step.get("mimic_error_m")
            if mimic is None or float(mimic) > JAW_MIMIC_BOUND_M:
                notes.append(
                    f"sweep {index} command {command}: mimic error {mimic}"
                )
            if abs(float(step["normalized_reading_error"])) > GRIPPER_TOLERANCE:
                notes.append(
                    f"sweep {index} command {command}: public normalized reading "
                    f"diverges by {step['normalized_reading_error']}"
                )
            if any(
                value < JAW_FINGER_JOINT_LIMIT_M[0] - JAW_FINGER_UNDERSHOOT_BOUND_M
                for value in step["gripper_finger_qpos"]
            ):
                notes.append(
                    f"sweep {index} command {command}: finger below its own limit"
                )
            if command >= 1.0:
                if abs(float(step["aperture_error_m"])) > JAW_OPEN_APERTURE_TOLERANCE_M:
                    notes.append(
                        f"sweep {index} open aperture error "
                        f"{step['aperture_error_m']}"
                    )
            elif command <= 0.0 and (
                float(step["measured_aperture_m"]) > JAW_CLOSED_APERTURE_BOUND_M
            ):
                notes.append(
                    f"sweep {index} closed aperture "
                    f"{step['measured_aperture_m']} exceeds the closed bound"
                )
    levels = {float(step["commanded_directive"]) for sweep in sweeps for step in sweep}
    for command in sorted(levels):
        measured = [
            float(step["measured_aperture_m"])
            for sweep in sweeps for step in sweep
            if float(step["commanded_directive"]) == command
        ]
        if len(measured) != len(sweeps):
            notes.append(f"command {command} is not measured in every sweep")
        elif max(measured) - min(measured) > JAW_SWEEP_AGREEMENT_M:
            notes.append(
                f"command {command} hysteresis {max(measured) - min(measured):.6f} m"
            )
    return {"passed": not notes, "notes": notes}


def jaw_aperture_calibration(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a passing qualification record into the bindable calibration.

    The output is the exact value
    ``ora/agent/tools/arx_x5_gripper.ARX_X5_APERTURE_CALIBRATION``
    accepts. A record whose aperture case did not pass produces nothing, so a
    nominal number can never be laundered into the grasp compiler as a
    measurement.
    """

    cases = {
        str(case.get("case_id")): case
        for case in record.get("cases", [])
        if isinstance(case, Mapping)
    }
    case = cases.get(JAW_APERTURE_CASE_ID)
    if case is None or not case.get("passed"):
        raise ValueError(
            "no passing jaw-aperture case; the caveat stays and nothing binds"
        )
    steps = [
        step
        for sweep in case.get("sweeps", []) for step in sweep
        if isinstance(step, Mapping)
    ]
    if not steps:
        raise ValueError("a passing jaw-aperture case must carry its steps")
    apertures = [float(step["measured_aperture_m"]) for step in steps]
    contact = case.get("contact_leg")
    contact = dict(contact) if isinstance(contact, Mapping) else {
        "status": JAW_CONTACT_LEG_STATUS, "reason": JAW_CONTACT_LEG_REASON,
    }
    return {
        "schema": JAW_CALIBRATION_SCHEMA,
        "probe_id": JAW_APERTURE_PROBE_ID,
        "record_sha256": str(record["record_sha256"]),
        "source_commit": str(record["source_commit"]),
        "passed": True,
        "measured_maximum_aperture_m": round(max(apertures), 6),
        "measured_closed_aperture_m": round(min(apertures), 6),
        "contact_leg_passed": contact.get("status") == "passed",
        "contact_leg": contact,
        "arms": sorted(case.get("arms", [])),
    }


def summarize_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Public, prompt-safe summary persisted per case."""
    arms = receipt.get("arms") if isinstance(receipt.get("arms"), Mapping) else {}
    summary = {
        "schema": receipt.get("schema"), "action": receipt.get("action"),
        "execution_status": receipt.get("execution_status"), "motion_strategy": receipt.get("motion_strategy"),
        "failure_stage": receipt.get("failure_stage"), "failure_code": receipt.get("failure_code"),
        "executor": receipt.get("executor"), "state_changed": receipt.get("state_changed"),
        "native_actions": receipt.get("native_actions"), "round_index": receipt.get("round_index"),
        "terminal_after_action": receipt.get("terminal_after_action"),
        "command_invariant_verified": (receipt.get("command_invariant") or {}).get("verified")
        if isinstance(receipt.get("command_invariant"), Mapping) else None,
        "arms": {},
    }
    for arm, effect in arms.items():
        if not isinstance(effect, Mapping):
            continue
        summary["arms"][arm] = {
            "targeted": effect.get("targeted"), "pose_error": effect.get("pose_error"),
            "gripper_error": effect.get("gripper_error"), "plan_ref": effect.get("plan_ref"),
            "inactive_arm_drift": effect.get("inactive_arm_drift"),
            "segments": [
                {"segment": item.get("segment"), "reached": item.get("reached"),
                 "native_actions": item.get("native_actions"), "ik_status": item.get("ik_status"),
                 "joint_delta_rad": item.get("joint_delta_rad")}
                for item in effect.get("segments", []) if isinstance(item, Mapping)
            ],
        }
    return summary
