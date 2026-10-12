"""Source-bound ARX X5 parallel-jaw geometry for Contact-GraspNet compilation.

Every number below is read from the pinned RoboDojo Assets checkout
(``Assets/Robots/x5``, revision ``a14409d7fae673c00499e01fd88b4457df6351b1``)
and hashed in :data:`ARX_X5_SOURCE_BINDING`.  Derivation, in the ``link6``
frame the Runtime controls (``controllers/robodojo_arx.ARX_X5_CONTRACT``):

* ``X5A.urdf`` mounts the prismatic finger joints ``joint7``/``joint8`` on
  ``link6`` at ``xyz = (0.08657, ±0.0249, -0.00024)`` with axes ``±y`` and
  limits ``[0, 0.044]``; ``robot_config.yml`` drives ``joint7`` over
  ``gripper_scale [-0.01, 0.044]`` with ``joint8`` as a unit mimic.  The
  finger rails therefore lie along ``+x`` (approach) and open along ``±y``.
* ``link7.STL``/``link8.STL`` (binary STL, vertices in the finger-joint frame)
  have their inner pad faces at ``y = ∓0.02449`` and span
  ``x ∈ [0.0560, 0.0710]``, ``z ∈ [-0.0104, 0.0080]``.  Offsetting by the joint
  origin puts both pad faces at ``|y| = 0.0004`` when ``q = 0`` and the pad
  faces at ``x ∈ [0.1426, 0.1576]`` from the ``link6`` origin; their centre
  ``x = 0.1501`` is the contact baseline.  ``robot_config.yml`` declares
  ``gripper_bias: 0.145`` as the source TCP, 5 mm behind that centre and still
  on the pad.
* Nominal pad-face aperture is ``2 q + 0.0008`` m, so ``q = 0.044`` opens
  ``0.0888`` m and any command below ``0.178`` nominally touches the pads.

The nominal aperture is not a calibrated fact: pad compliance, mimic tracking
and contact-limited closure are unmeasured until a jaw-aperture probe runs.  Grasp evidence
compiled through these identities therefore carries the caveat verbatim.
:func:`jaw_aperture_admission` is the single place that decides which of the
three honest states applies — no record, a free-air-only record, or a record
with a passing contact leg — and the caveat disappears only in the last.

One thing this module's caveat has never been about, and must not be read as:
the ``gripper_opening_m`` published on every compiled candidate is
Contact-GraspNet's Panda-trained width head, passed through unchanged.  Each
identity states that in ``predicted_opening_source``, because calibrating the
X5 jaw would not change that number at all.
"""

from __future__ import annotations

from typing import Any, Mapping


ARX_X5_SOURCE_BINDING = {
    "robodojo_assets_revision": "a14409d7fae673c00499e01fd88b4457df6351b1",
    "asset_root": "Assets/Robots/x5",
    "robot_config_sha256": (
        "9de4f63b84c0f730623c09af7a2d530d84973981397280b6e4b626b50ba6f1c9"
    ),
    "urdf_sha256": (
        "7aaa308a1a5b6a213d914db86ba09ddd6252f25cb498ac9a148bfa6c7166a423"
    ),
    "link7_mesh_sha256": (
        "ec3d40b39cccde0075ca3d16603f14f180fce41a46b446473d384def7a1b0be0"
    ),
    "link8_mesh_sha256": (
        "29f7a8379cad81797e82e4ba099455cc275ac6ac5bb9e264ce46719fd7d2bb60"
    ),
}
ARX_X5_SOURCE_FILES = {
    "robot_config_sha256": "robot_config.yml",
    "urdf_sha256": "X5A.urdf",
    "link7_mesh_sha256": "meshes/link7.STL",
    "link8_mesh_sha256": "meshes/link8.STL",
}
ARX_X5_APERTURE_PROBE_ID = "robodojo_x5_jaw_aperture_probe_v1"
ARX_X5_APERTURE_CAVEAT = (
    "nominal_urdf_mesh_jaw_aperture_uncalibrated_pending_"
    + ARX_X5_APERTURE_PROBE_ID
)
#: A free-air sweep certifies the actuator, the mimic and the closed floor and
#: says nothing about pad compliance, contact-limited stall, or the reading the
#: policy would see with an object between the jaws — the only aperture facts a
#: retention judgement could ever rest on.  A record that passed the free-air
#: legs but carries no passing contact leg therefore keeps a caveat; it just
#: names what is actually still unmeasured instead of claiming the whole
#: aperture is nominal.
ARX_X5_FREE_AIR_ONLY_CAVEAT = (
    "free_air_measured_jaw_aperture_contact_stall_uncalibrated_pending_"
    + ARX_X5_APERTURE_PROBE_ID
    + "_contact_leg"
)
ARX_X5_APERTURE_CALIBRATION_SCHEMA = (
    "ora.robodojo-x5-jaw-aperture-calibration.v1"
)
#: The nominal admission ceiling, held 8.8 mm under the nominal 88.8 mm maximum
#: while nothing has been measured.
ARX_X5_NOMINAL_MAXIMUM_OPENING_M = 0.08
#: A measured maximum is admitted only this far short of itself, so an admitted
#: opening always leaves clearance the measurement itself proved existed.
ARX_X5_APERTURE_ADMISSION_MARGIN_M = 0.004
ARX_X5_GRIPPER_GEOMETRY = {
    "schema": "ora.arx-x5-gripper-geometry.v1",
    "controlled_link": "link6",
    "approach_axis": "link6_local_positive_x",
    "jaw_axis": "link6_local_positive_y",
    "finger_joint_origin_link6_m": {"x": 0.08657, "abs_y": 0.0249, "z": -0.00024},
    "finger_joint_limit_m": [0.0, 0.044],
    "command_joint_range_m": [-0.01, 0.044],
    "mimic_multiplier": 1.0,
    "pad_inner_face_span_link6_x_m": [0.1426, 0.1576],
    "pad_inner_face_span_link6_z_m": [-0.0104, 0.0080],
    "contact_baseline_link6_x_m": 0.1501,
    "source_declared_gripper_bias_m": 0.145,
    "closed_pad_gap_m": 0.0008,
    "aperture_per_joint_metre": 2.0,
    "nominal_maximum_aperture_m": 0.0888,
    "physical_jaw_aperture_calibrated": False,
    "aperture_calibration_probe": ARX_X5_APERTURE_PROBE_ID,
}
CONTACT_GRASPNET_GRIPPER_DEPTH_M = 0.1034
# The categorical projection of a normalized command lives with the RoboDojo
# session that writes the command
# (``environment/backends/robodojo/common.gripper_command_target``), which
# publishes ``gripper_command_target`` in its own proprioception; the policy
# layer must not re-derive a category the Environment already stated, and the
# layering audit forbids importing that module here in any case. This module
# stays the grasp-compilation geometry owner.


def nominal_jaw_aperture_m(gripper_command: float) -> float:
    """Pad-face separation the URDF/mesh geometry predicts for one command."""

    if not 0.0 <= gripper_command <= 1.0:
        raise ValueError("gripper command must lie in [0, 1]")
    low, high = ARX_X5_GRIPPER_GEOMETRY["command_joint_range_m"]
    joint = low + (high - low) * gripper_command
    aperture = (
        ARX_X5_GRIPPER_GEOMETRY["aperture_per_joint_metre"] * joint
        + ARX_X5_GRIPPER_GEOMETRY["closed_pad_gap_m"]
    )
    return max(0.0, round(aperture, 6))


def jaw_aperture_admission(
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Resolve the aperture admission limit and its honest disclosure.

    Three states, and the caveat is dropped in exactly one of them:

    * no bound record — the nominal 80 mm ceiling under the nominal 88.8 mm
      mesh maximum, ``physical_jaw_aperture_calibrated`` false and the
      standing :data:`ARX_X5_APERTURE_CAVEAT`;
    * a passing record whose contact leg is absent or did not pass — the
      *measured* maximum minus :data:`ARX_X5_APERTURE_ADMISSION_MARGIN_M`,
      still ``physical_jaw_aperture_calibrated`` false, and
      :data:`ARX_X5_FREE_AIR_ONLY_CAVEAT`, which names the one thing that is
      still unmeasured rather than pretending nothing was;
    * a passing record with a passing contact leg — the measured limit,
      ``physical_jaw_aperture_calibrated`` true and no caveat at all.

    Nothing here ever substitutes a nominal number for a measured one: an
    absent record keeps the nominal ceiling *and* its caveat together, and a
    record that does not validate raises instead of degrading silently.
    """

    if calibration is None:
        return {
            "maximum_opening_m": ARX_X5_NOMINAL_MAXIMUM_OPENING_M,
            "maximum_opening_source": "nominal_urdf_mesh_geometry",
            "physical_jaw_aperture_calibrated": False,
            "aperture_caveat": ARX_X5_APERTURE_CAVEAT,
        }
    record = validate_jaw_aperture_calibration(calibration)
    measured = float(record["measured_maximum_aperture_m"])
    contact = bool(record["contact_leg_passed"])
    admission = {
        "maximum_opening_m": round(
            measured - ARX_X5_APERTURE_ADMISSION_MARGIN_M, 6,
        ),
        "maximum_opening_source": "measured_" + str(record["probe_id"]),
        "physical_jaw_aperture_calibrated": contact,
        "aperture_calibration_record": {
            "probe_id": str(record["probe_id"]),
            "record_sha256": str(record["record_sha256"]),
            "source_commit": str(record["source_commit"]),
            "measured_maximum_aperture_m": measured,
            "measured_closed_aperture_m": float(
                record["measured_closed_aperture_m"]
            ),
            "contact_leg_passed": contact,
        },
    }
    if not contact:
        admission["aperture_caveat"] = ARX_X5_FREE_AIR_ONLY_CAVEAT
    return admission


def validate_jaw_aperture_calibration(
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Accept only a passing, self-identifying measured aperture record."""

    if not isinstance(calibration, Mapping):
        raise ValueError("jaw-aperture calibration must be a mapping")
    required = (
        "schema", "probe_id", "record_sha256", "source_commit", "passed",
        "measured_maximum_aperture_m", "measured_closed_aperture_m",
        "contact_leg_passed",
    )
    missing = [key for key in required if key not in calibration]
    if missing:
        raise ValueError(
            f"jaw-aperture calibration lacks {sorted(missing)}"
        )
    if calibration["schema"] != ARX_X5_APERTURE_CALIBRATION_SCHEMA:
        raise ValueError("jaw-aperture calibration schema is unknown")
    if calibration["probe_id"] != ARX_X5_APERTURE_PROBE_ID:
        raise ValueError("jaw-aperture calibration probe identity is unknown")
    if calibration["passed"] is not True:
        raise ValueError("a failing jaw-aperture record cannot be bound")
    if not isinstance(calibration["contact_leg_passed"], bool):
        raise ValueError("contact_leg_passed must be a Boolean")
    for key in ("record_sha256", "source_commit"):
        value = calibration[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"jaw-aperture calibration {key} is invalid")
    maximum = calibration["measured_maximum_aperture_m"]
    closed = calibration["measured_closed_aperture_m"]
    for name, value in (("maximum", maximum), ("closed", closed)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"measured {name} aperture must be a number")
    maximum, closed = float(maximum), float(closed)
    # A measured maximum below the admission margin would compile a negative
    # ceiling, and a measured maximum above the nominal mesh maximum means the
    # record and the pinned geometry disagree about the same robot.
    if not (
        ARX_X5_APERTURE_ADMISSION_MARGIN_M
        < maximum
        <= float(ARX_X5_GRIPPER_GEOMETRY["nominal_maximum_aperture_m"])
    ):
        raise ValueError("measured maximum aperture is outside the source geometry")
    if not 0.0 <= closed < maximum:
        raise ValueError("measured closed aperture is not below the maximum")
    return dict(calibration)


def arx_x5_link6_compiler_identities(
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build both arm identities against a measured record when one is bound."""

    admission = jaw_aperture_admission(calibration)
    return {
        arm: {
            "id": f"contact_graspnet_to_robodojo_x5_{arm}_link6_v1",
            "embodiment_descriptor_id": "arx_x5_robodojo_v1",
            "source_pose_convention": (
                "contact_graspnet_panda_gripper_origin_opencv_v2"
            ),
            # The controlled point is the measured link6 pose; end_effector
            # selects the identity-parametrized approach-along-x compilation
            # shared with ALOHA rather than the Panda grip-site mapping.
            "controlled_point": "end_effector",
            "controlled_point_semantics": "robodojo_link6_pose",
            "contact_graspnet_gripper_depth_m": CONTACT_GRASPNET_GRIPPER_DEPTH_M,
            # Contact-GraspNet's Panda origin sits 103.4 mm behind its contact
            # baseline; the X5 pad centre sits 150.1 mm ahead of link6, so
            # link6 lies 46.7 mm behind the model origin along the approach
            # axis.
            "source_origin_to_controlled_point_m": round(
                CONTACT_GRASPNET_GRIPPER_DEPTH_M
                - ARX_X5_GRIPPER_GEOMETRY["contact_baseline_link6_x_m"], 4,
            ),
            "controlled_point_to_contact_baseline_m": (
                ARX_X5_GRIPPER_GEOMETRY["contact_baseline_link6_x_m"]
            ),
            "axis_mapping": {
                "model_approach_z": "end_effector_local_positive_x",
                "model_jaw_x": "end_effector_local_positive_y",
            },
            "arm": arm,
            "source_binding": ARX_X5_SOURCE_BINDING,
            "gripper_geometry": ARX_X5_GRIPPER_GEOMETRY,
            # The published ``gripper_opening_m`` is not an X5 number and is
            # not what the aperture caveat is about: it is Contact-GraspNet's
            # Panda-trained width head, quantized into its own ten bins over
            # the Panda 0--0.08 m range and passed through unchanged. A
            # perfect X5 aperture calibration would not move it by a
            # millimetre, so the field names its own source.
            "predicted_opening_source": (
                "contact_graspnet_panda_width_head_bin"
            ),
            **admission,
        }
        for arm in ("left", "right")
    }


#: No measured record is bound on this commit, so both identities carry the
#: nominal ceiling and the standing caveat. Binding a record is a source
#: change: assign the passing
#: ``ora.robodojo-x5-jaw-aperture-calibration.v1`` projection of a
#: ``robodojo_x5_jaw_aperture_probe_v1`` qualification record here, in the same
#: change that re-measures the two live RoboDojo gates.
ARX_X5_APERTURE_CALIBRATION: dict[str, Any] | None = None
ARX_X5_LINK6_COMPILER_IDENTITIES = arx_x5_link6_compiler_identities(
    ARX_X5_APERTURE_CALIBRATION,
)


__all__ = [
    "ARX_X5_APERTURE_ADMISSION_MARGIN_M", "ARX_X5_APERTURE_CALIBRATION",
    "ARX_X5_APERTURE_CALIBRATION_SCHEMA", "ARX_X5_APERTURE_CAVEAT",
    "ARX_X5_APERTURE_PROBE_ID", "ARX_X5_FREE_AIR_ONLY_CAVEAT",
    "ARX_X5_GRIPPER_GEOMETRY", "ARX_X5_LINK6_COMPILER_IDENTITIES",
    "ARX_X5_NOMINAL_MAXIMUM_OPENING_M", "ARX_X5_SOURCE_BINDING",
    "ARX_X5_SOURCE_FILES", "arx_x5_link6_compiler_identities",
    "jaw_aperture_admission", "nominal_jaw_aperture_m",
    "validate_jaw_aperture_calibration",
]
