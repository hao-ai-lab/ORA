"""Robotwin motion v2 sparse contract regressions."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pytest
from ora.environment.definitions import resolve_runtime_descriptor
from tests.environments.robotwin.test_robotwin_motion_v2 import (
    transaction_move,
    transaction_plant,
    transaction_session,
)


def test_sparse_motion_admission_is_qualification_table_driven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RoboTwin-only gate is one qualification table, not a name check.

    LIBERO joins a codec row only together with its own pinned-source
    scripted motion qualification (EXP08A D1 path 2); until then the fail
    remains closed for every unqualified environment.
    """

    from ora.environment import definitions as environment_definitions
    from ora.environment.definitions import (
        EnvironmentDefinitionError,
        SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
    )

    assert SPARSE_MOTION_QUALIFIED_ENVIRONMENTS == {
        # libero_pro joined v3 with its passing 14-case sparse-motion
        # qualification on source 932e9bd (record path and hash live next
        # to the table).
        "motion_action_v3": ("robotwin", "libero_pro", "robodojo"),
    }
    # Layered fail-closed: a codec the backend does not implement at all is
    # rejected by the environment's own action-schema set first
    # (RoboDojo implements v3 but not the v4 target state)...
    with pytest.raises(
        EnvironmentDefinitionError, match="unsupported action schema",
    ):
        resolve_runtime_descriptor(
            {"kind": "robodojo", "transport": "local_rpc"},
            action_codec="target_state_v4",
        )
    # ...while an implemented-but-unqualified environment is rejected by the
    # qualification table without touching validation code.
    monkeypatch.setitem(
        environment_definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
        "motion_action_v3", ("robotwin",),
    )
    with pytest.raises(
        EnvironmentDefinitionError, match="no source-bound motion qualification",
    ):
        resolve_runtime_descriptor(
            {"kind": "libero_pro", "transport": "local_rpc"},
            action_codec="motion_action_v3",
        )
    monkeypatch.setitem(
        environment_definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
        "motion_action_v3", ("robotwin", "libero_pro"),
    )

    # The admission itself is table-driven: removing an environment from a
    # codec row rejects it with the qualification message, without touching
    # validation code.
    robotwin_runtime = {
        "kind": "robotwin",
        "transport": "local_rpc",
        "required_revision": "c3ddfa8b97d5519efa828b075999bd0006778e5e",
    }
    resolve_runtime_descriptor(robotwin_runtime, action_codec="motion_action_v3")
    monkeypatch.setitem(
        environment_definitions.SPARSE_MOTION_QUALIFIED_ENVIRONMENTS,
        "motion_action_v3", ("somewhere_else",),
    )
    with pytest.raises(
        EnvironmentDefinitionError, match="no source-bound motion qualification",
    ):
        resolve_runtime_descriptor(
            robotwin_runtime, action_codec="motion_action_v3",
        )


def _joint_writes(plant: Any, arm: str) -> list[Any]:
    return [call for call in plant.calls if call[:2] == ("joint", arm)]


def test_sparse_single_arm_calls_only_selected_planner_and_counts_once(tmp_path: Path) -> None:
    plant = transaction_plant()
    live, calls = transaction_session(tmp_path, plant)
    receipt = live.execute(transaction_move("left"))["receipt"]
    assert receipt["execution_status"] == "reached"
    assert calls == ["left"]
    assert _joint_writes(plant, "left") and not _joint_writes(plant, "right")
    assert receipt["semantic_action_count"] == 1
    assert receipt["arms"]["right"]["planner"]["called"] is False
    assert receipt["arms"]["right"]["inactive_arm_drift"] == {
        "translation_m": 0.0, "orientation_rad": 0.0,
    }
    assert receipt["arms"]["right"]["command_preservation"]["arm_drive_target_unchanged"] is True
    assert receipt["arms"]["left"]["command_preservation"]["gripper_drive_target_unchanged"] is True
    assert receipt["synchronization"] == {
        "requested": False, "planned_from_common_snapshot": False,
        "all_planners_succeeded": True, "all_plans_admitted": True, "executed": True,
    }


def test_sparse_dual_arm_is_zero_execution_when_either_planner_fails(tmp_path: Path) -> None:
    plant = transaction_plant()
    live, calls = transaction_session(tmp_path, plant, failing=("right",))
    receipt = live.execute(transaction_move("left", "right"))["receipt"]
    assert calls == ["left", "right"]
    assert plant.steps == 0 and not plant.calls
    assert receipt["execution_status"] == "planner_failed"
    assert receipt["state_changed"] is False
    assert receipt["steps_used"] == 0
    assert receipt["synchronization"]["all_planners_succeeded"] is False
    assert receipt["semantic_action_count"] == 1


def test_sparse_dual_arm_executes_one_synchronized_dense_update(tmp_path: Path) -> None:
    plant = transaction_plant()
    live, calls = transaction_session(tmp_path, plant)
    receipt = live.execute(transaction_move("left", "right"))["receipt"]
    assert receipt["execution_status"] == "reached"
    assert calls == ["left", "right"]
    # Both arms are written before every shared physics step.
    assert [step for _, _, step in _joint_writes(plant, "left")] == [
        step for _, _, step in _joint_writes(plant, "right")
    ]
    assert receipt["synchronization"] == {
        "requested": True, "planned_from_common_snapshot": True,
        "all_planners_succeeded": True, "all_plans_admitted": True, "executed": True,
    }


def test_sparse_gripper_never_plans_arm_or_exposes_numeric_aperture(tmp_path: Path) -> None:
    from tests.environments.robotwin.test_robotwin_correctness_repair import action as correctness_action

    plant = transaction_plant()
    live, calls = transaction_session(tmp_path, plant)
    receipt = live.execute(correctness_action(arguments={"targets": {"right": "open"}}))["receipt"]
    assert calls == []
    assert not _joint_writes(plant, "left") and not _joint_writes(plant, "right")
    assert set(plant.gripper_writes) == {"right"}
    assert receipt["after"]["right"]["gripper_command_target"] == "open"
    for arm in ("left", "right"):
        assert receipt["arms"][arm]["command_preservation"]["arm_drive_target_unchanged"] is True
    assert "gripper_normalized" not in json.dumps(receipt)
    assert "aperture" not in json.dumps(receipt)


def test_sparse_recovery_reports_selected_arm_residual_and_preservation(tmp_path: Path) -> None:
    from tests.environments.robotwin.test_robotwin_correctness_repair import action as correctness_action

    plant = transaction_plant()
    live, calls = transaction_session(tmp_path, plant)
    live.last_safe_joints["left"] = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    receipt = live.execute(correctness_action(
        "recover_motion", {"targets": {"left": "backtrack_last_safe"}},
    ))["receipt"]
    assert receipt["execution_status"] == "reached"
    assert calls == ["left"]
    assert receipt["arms"]["left"]["pose_error"]["translation_m"] == pytest.approx(0.0, abs=1e-9)
    assert receipt["arms"]["left"]["command_preservation"]["gripper_drive_target_unchanged"] is True
    assert receipt["arms"]["right"]["planner"]["called"] is False


def test_sparse_recovery_does_not_overwrite_previous_pose_checkpoint(tmp_path: Path) -> None:
    from tests.environments.robotwin.test_robotwin_correctness_repair import action as correctness_action

    plant = transaction_plant()
    live, _calls = transaction_session(tmp_path, plant)
    live.last_safe_joints["left"] = [0.02, 0.0, 0.0, 0.0, 0.0, 0.0]
    previous = [-0.1, 0.05, 1.1, 0.0, 0.0, 0.0, 1.0]
    live.last_safe_pose["left"] = list(previous)
    receipt = live.execute(correctness_action(
        "recover_motion", {"targets": {"left": "backtrack_last_safe"}},
    ))["receipt"]
    assert receipt["execution_status"] == "reached"
    assert live.last_safe_pose["left"] == previous


def test_sparse_execution_fault_does_not_create_success_outside_physics_boundary(tmp_path: Path) -> None:
    plant = transaction_plant(success_step=1)

    def fault() -> None:
        raise RuntimeError("dense execution fault")

    plant.scene.step = fault
    live, _calls = transaction_session(tmp_path, plant)
    frames: list[str] = []
    live._capture_operator_frame = lambda *, label: frames.append(label)  # type: ignore[method-assign]
    receipt = live.execute(transaction_move("left"))["receipt"]
    assert receipt["semantic_action_count"] == 1
    # The transaction stops at the failed physics step and reports it as its
    # own guard, never as an evaluated action.
    assert receipt["execution_status"] == "execution_fault"
    assert receipt["controller_tracking"]["abort"]["code"] == "backend_step_fault"
    assert "backend_step_fault" in receipt["execution_fault"]
    assert receipt["official_evaluator_queried"] is True
    assert receipt["terminal_after_action"] is False
    assert plant.predicate_reads == 0
    assert plant.eval_success is False
    assert frames == ["action-0000"]
