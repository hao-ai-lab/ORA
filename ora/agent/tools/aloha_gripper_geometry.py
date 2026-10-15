# Copyright contributors to the Robot Harness Runtime project.
"""ALOHA jaw geometry, derived from the source files it pins by sha256.

Its own module because the number has two consumers that must not drift:
the grasp compiler admits candidates against it, and the published
``grasp.plan`` result schema caps ``gripper_opening_m`` by it. They were
two independent literals once, and raising only the first made the
compiler admit candidates its own result schema then refused, so the whole
call returned ``internal_tool_error`` instead of one named rejection.
``tests/agent/composition/test_parameter_delivery_guards.py`` compares the two.
"""

from __future__ import annotations


ALOHA_SOURCE_BINDING = {
    "robotwin_revision": "c3ddfa8b97d5519efa828b075999bd0006778e5e",
    "base_task_sha256": "448f7152b65cb9102217cf5463aa821d72810ca56f63d5a797ec7bd43e23e101",
    "robot_adapter_sha256": "3dcd80acc8cab489a4c5edb507cc460dab1724be0226b8d5c4c1b218dee605cb",
    "aloha_config_sha256": "a4f043d2d4b44cca5ec47fb498a3194da3f87ff9530dcbfdcee1539a44423d57",
    "aloha_urdf_sha256": "097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a",
}
#: ALOHA jaw aperture, derived from the two files pinned above rather than
#: written as a literal. ``arx5_description_isaac.urdf`` declares ``fl_joint7``
#: and ``fl_joint8`` prismatic on opposite axes (``0 1 0`` and ``0 -1 0``) with
#: travel ``0 .. 0.04765`` m and ``mimic`` multiplier 1.0, so both pads retract
#: together and the gap grows by twice the commanded joint value. ``config.yml``
#: caps the reachable command at ``gripper_scale`` upper 0.045 m, and the mesh
#: gap at the closed reference is the 4.8964 mm the public geometry descriptor
#: publishes. Hence ``2 q + closed_pad_gap``, the shape ARX-X5 already uses in
#: :mod:`ora.agent.tools.arx_x5_gripper`.
#:
#: This is the compiler's admission threshold, and it remains an admission
#: threshold rather than a measured jaw: the tapered pads make the usable
#: opening depth-dependent and one live run measured a free-air closure near
#: 18 mm against the 4.8964 mm the mesh states, so rejections keep disclosing
#: ``compiler_admission_not_measured_physical_aperture``. What changed is the
#: number, not its status. Against the 55 mm literal it replaces, 149 recorded
#: rejections across two recorded RoboTwin batches asked for 55.4 to 291.1 mm; 81 of them,
#: 54%, sat inside this bound and were refused a candidate the arm could have
#: attempted.
ALOHA_GRIPPER_GEOMETRY = {
    "commanded_joint_travel_m": 0.045,
    "urdf_joint_travel_limit_m": 0.04765,
    "mimic_multiplier": 1.0,
    "aperture_per_joint_metre": 2.0,
    "closed_pad_gap_m": 0.0048964,
    "aperture_semantics": "jaw_separation",
    "physical_jaw_aperture_calibrated": False,
}
ALOHA_NOMINAL_MAXIMUM_OPENING_M = round(
    ALOHA_GRIPPER_GEOMETRY["aperture_per_joint_metre"]
    * ALOHA_GRIPPER_GEOMETRY["commanded_joint_travel_m"]
    + ALOHA_GRIPPER_GEOMETRY["closed_pad_gap_m"],
    6,
)
