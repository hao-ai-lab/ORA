"""What the post-action public-observation record measures, and what it never claims.

Every executed action is followed by one `stagnation_observation` record. This
module pins what that record holds, why restoring the retired judgement's
stale arm read would not have produced a working guard, and that the record
reaches the policy through none of the paths that could carry it, the
retained-observation keyframe flag among them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from ora.agent.context.projection import episode_index_keyframe
from ora.contracts import MOTION_ACTION_V3_SCHEMA
from ora.orchestrator.stagnation import StagnationGuard


# The exact per-arm record shapes the two maintained backends write. RoboTwin
# reports the gripper as a categorical command target; LIBERO reports a
# numeric aperture. The retired judgement needed the numeric one.
def _robotwin_arm_state(z: float) -> dict[str, Any]:
    return {
        "arm_drive_target_sha256": "a" * 64,
        "controlled_point_semantics": "parallel_jaw_grip_center_tcp",
        "gripper_command_target": "unknown",
        "gripper_drive_target_sha256": "b" * 64,
        "pose_xyz_xyzw": [0.3, -0.19, z, 0.0, 0.0, 0.71, 0.70],
    }


def _libero_arm_state(z: float, gripper: float) -> dict[str, Any]:
    return {
        "gripper": gripper,
        "pose_xyz_xyzw": [0.3, -0.19, z, 0.0, 0.0, 0.71, 0.70],
    }


def _robotwin_receipt(*, before_z: float, after_z: float) -> dict[str, Any]:
    return {
        "schema": "ora.robotwin-sparse-motion-receipt.v7",
        "action": "set_gripper",
        "before": {"right": _robotwin_arm_state(before_z)},
        "after": {"right": _robotwin_arm_state(after_z)},
    }


def _libero_receipt(*, before_z: float, after_z: float) -> dict[str, Any]:
    return {
        "schema": "ora.libero-sparse-motion-receipt.v1",
        "action": "set_gripper",
        "before": {"right": _libero_arm_state(before_z, 0.976)},
        "after": {"right": _libero_arm_state(after_z, 0.022)},
    }


def _sparse_action(target: str) -> dict[str, Any]:
    return {
        "schema": MOTION_ACTION_V3_SCHEMA,
        "observation_id": "observation-0",
        "action_id": "action-0",
        "action": "set_gripper",
        "arguments": {"targets": {"right": target}},
        "bound_observation_sha256": "0" * 64,
    }


def _identical_images(tmp_path: Path) -> tuple[Path, Path]:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (64, 64), (12, 34, 56)).save(before)
    Image.new("RGB", (64, 64), (12, 34, 56)).save(after)
    return before, after


def test_repairing_the_arm_read_would_not_have_produced_a_working_guard() -> None:
    """Why the retired judgement was not a one-line fix.

    It resolved the acting arm through `arguments["arm"]`, which the sparse
    motion codec does not have, so every lookup returned an empty record.
    Restoring that read would have brought the displacement back on both
    backends but the aperture change only on LIBERO: the RoboTwin receipt
    reports the gripper as a categorical command target. Since the retired
    `no_public_progress` required all three terms, the repaired guard would
    have been alive on one maintained backend and silently constant-false on
    the other.
    """

    arguments = _sparse_action("close")["arguments"]
    assert set(arguments) == {"targets"}
    assert "arm" not in arguments

    robotwin = _robotwin_receipt(before_z=0.94, after_z=1.00)
    for side in ("before", "after"):
        for state in robotwin[side].values():
            assert "gripper" not in state
            assert isinstance(state["gripper_command_target"], str)

    libero = _libero_receipt(before_z=0.94, after_z=1.00)
    for side in ("before", "after"):
        for state in libero[side].values():
            assert isinstance(state["gripper"], float)
            assert "gripper_command_target" not in state


def test_the_record_holds_one_measured_distance_and_no_judgement(
    tmp_path: Path,
) -> None:
    """Nothing derived from the action or the receipt survives in the record.

    The retired judgement needed a per-arm pose pair and a numeric aperture
    pair out of the receipt, neither of which the record now carries. It also
    no longer accepts the action or the receipt at all, so a future reader
    cannot revive the comparison by filling in a parameter.
    """

    before, after = _identical_images(tmp_path)
    guard = StagnationGuard(tmp_path / "stagnation.json")
    records = [
        guard.observe(before_image=before, after_image=after) for _ in range(6)
    ]

    assert all(set(record) == {"head_phash_hamming"} for record in records)
    assert all(record["head_phash_hamming"] == 0 for record in records)
    assert not hasattr(guard, "is_high_confidence")
    assert not hasattr(guard, "streak")

    persisted = json.loads((tmp_path / "stagnation.json").read_text(encoding="utf-8"))
    assert set(persisted) == {"schema", "records"}
    assert persisted["schema"] == "ora.observation-delta.v3"
    assert persisted["records"] == records


def test_an_identical_head_image_is_not_evidence_that_nothing_moved(
    tmp_path: Path,
) -> None:
    """A bit-identical head image and a 6 cm lift produce the same zero.

    The head camera is a whole-scene view; `_phash` reduces it to 64 bits.
    Motion the camera cannot resolve, and motion outside the retained 8x8
    low-frequency block, both leave the hash unchanged.
    """

    before, after = _identical_images(tmp_path)
    guard = StagnationGuard(tmp_path / "stagnation.json")

    still = guard.observe(before_image=before, after_image=after)
    lifted = guard.observe(before_image=before, after_image=after)
    assert still["head_phash_hamming"] == lifted["head_phash_hamming"] == 0


def test_a_changed_scene_raises_the_recorded_distance(tmp_path: Path) -> None:
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    Image.new("RGB", (64, 64), (12, 34, 56)).save(before)
    changed = Image.new("RGB", (64, 64), (12, 34, 56))
    for x in range(64):
        for y in range(32):
            changed.putpixel((x, y), (240, 240, 240))
    changed.save(after)

    guard = StagnationGuard(tmp_path / "stagnation.json")
    record = guard.observe(before_image=before, after_image=after)
    assert record["head_phash_hamming"] > 0


def test_the_retention_keyframe_condition_is_asserted_directly() -> None:
    """The exact disjunction that decides which observations the index keeps.

    The notice disjuncts are gone with the booleans and the routing that fed
    them. Step cadence and a preceding gripper action are what remain, and a
    caller can no longer pass a notice at all.
    """

    gripper_action = {"action": "set_gripper", "arguments": {"targets": {"right": "close"}}}
    legacy_gripper_action = {"action": "target_state", "arguments": {"gripper": 0.0}}
    move_action = {"action": "move_pose", "arguments": {"targets": {"right": {}}}}

    assert episode_index_keyframe(step=0, previous_action=None)
    assert episode_index_keyframe(step=10, previous_action=None)
    assert episode_index_keyframe(step=20, previous_action=move_action)
    assert not episode_index_keyframe(step=7, previous_action=None)
    assert not episode_index_keyframe(step=7, previous_action=move_action)
    assert not episode_index_keyframe(step=7, previous_action="not a mapping")
    assert episode_index_keyframe(step=7, previous_action=gripper_action)
    assert episode_index_keyframe(step=7, previous_action=legacy_gripper_action)

    with pytest.raises(TypeError):
        episode_index_keyframe(
            step=7, notice={"warning": True}, previous_action=None,
        )
