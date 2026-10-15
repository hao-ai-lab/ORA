from __future__ import annotations

import math

import pytest

from ora.contracts.pose import (
    POSE_FRAME_CONTRACT_SHA256,
    PoseCompilationError,
    compile_absolute_pose,
    compile_pose_request,
)


IDENTITY_POSE = [0.1, -0.2, 0.4, 0.0, 0.0, 0.0, 1.0]


def test_pose_v2_rejects_ambiguous_position_and_orientation() -> None:
    with pytest.raises(PoseCompilationError, match="not both"):
        compile_pose_request(
            {
                "target_xyz_world_m": [0.2, 0.0, 0.5],
                "translation_delta_mm": [1, 2, 3],
            },
            current_pose_xyz_xyzw=IDENTITY_POSE,
        )
    with pytest.raises(PoseCompilationError, match="not both"):
        compile_pose_request(
            {
                "target_orientation_world_xyzw": [0, 0, 0, 1],
                "grip_site_z_axis_world": [0, 0, -1],
            },
            current_pose_xyz_xyzw=IDENTITY_POSE,
        )


def test_pose_v2_world_and_grip_site_delta_are_explicit() -> None:
    world = compile_pose_request(
        {
            "translation_delta_mm": [10, 20, -30],
            "translation_delta_frame": "world",
        },
        current_pose_xyz_xyzw=IDENTITY_POSE,
    )
    assert world["target_pose_xyz_xyzw"][:3] == pytest.approx(
        [0.11, -0.18, 0.37]
    )
    quarter_turn_z = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    local = compile_pose_request(
        {
            "translation_delta_mm": [10, 0, 0],
            "translation_delta_frame": "grip_site",
        },
        current_pose_xyz_xyzw=[0, 0, 0, *quarter_turn_z],
    )
    assert local["translation_world_m"] == pytest.approx([0, 0.01, 0])


def test_pose_v2_single_axis_preserves_current_twist_and_reports_rotation() -> None:
    compiled = compile_pose_request(
        {"grip_site_z_axis_world": [0, 0, -2]},
        current_pose_xyz_xyzw=IDENTITY_POSE,
    )
    axes = compiled["resolved_grip_site_axes_world"]
    assert axes["z"] == pytest.approx([0, 0, -1])
    assert axes["x"] == pytest.approx([1, 0, 0])
    assert compiled["rotation_angle_deg"] == pytest.approx(180.0)


def test_pose_v2_rejects_non_orthogonal_axes() -> None:
    with pytest.raises(PoseCompilationError, match="approximately orthogonal"):
        compile_pose_request(
            {
                "grip_site_z_axis_world": [0, 0, -1],
                "grip_site_x_axis_world": [0, 0.1, -1],
            },
            current_pose_xyz_xyzw=IDENTITY_POSE,
        )


def test_pose_v2_quaternion_sign_and_execution_hash_are_stable() -> None:
    positive = compile_pose_request(
        {"target_orientation_world_xyzw": [0, 0, 1, 0]},
        current_pose_xyz_xyzw=IDENTITY_POSE,
    )
    negative = compile_pose_request(
        {"target_orientation_world_xyzw": [0, 0, -1, 0]},
        current_pose_xyz_xyzw=IDENTITY_POSE,
    )
    assert positive["target_pose_xyz_xyzw"] == negative["target_pose_xyz_xyzw"]
    assert positive["compiled_pose_sha256"] == negative["compiled_pose_sha256"]
    rebuilt = compile_absolute_pose(
        current_pose_xyz_xyzw=IDENTITY_POSE,
        target_pose_xyz_xyzw=positive["target_pose_xyz_xyzw"],
    )
    assert rebuilt["compiled_pose_sha256"] == positive["compiled_pose_sha256"]
    assert rebuilt["pose_contract_sha256"] == POSE_FRAME_CONTRACT_SHA256
    measured = [
        -0.14846308820529638, 0.0006242010878617587, 0.26128132201409593,
        -0.6837433577098534, -0.7291679382873082,
        0.020695362241633857, 0.019515136258282956,
    ]
    awkward = compile_pose_request(
        {
            "target_orientation_world_xyzw": [
                -0.03212002895887295, 0.9990791586146662,
                -0.028433116164182676, 0.0008345457961599807,
            ],
        },
        current_pose_xyz_xyzw=measured,
    )
    awkward_rebuilt = compile_absolute_pose(
        current_pose_xyz_xyzw=measured,
        target_pose_xyz_xyzw=awkward["target_pose_xyz_xyzw"],
    )
    assert awkward_rebuilt["compiled_pose_sha256"] == (
        awkward["compiled_pose_sha256"]
    )
