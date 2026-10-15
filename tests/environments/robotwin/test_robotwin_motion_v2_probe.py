"""Robotwin motion v2 probe contract regressions."""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pytest
from ora.agent.tools.robot import ToolError
from ora.environment.backends.robotwin.planning import planner_public_summary
from tests.environments.robotwin.test_robotwin_motion_v2 import (
    _probe_public_result,
    primitive_runtime,
)


@pytest.mark.parametrize("corruption", ["missing_query", "wrong_arm", "conflicting_handle"])
def test_probe_registers_no_handles_after_malformed_worker_result(tmp_path: Path, corruption) -> None:
    import copy
    runtime, _observation, _persisted = primitive_runtime(tmp_path)
    runtime._known_plan_refs = {}
    def probe(request):
        result = _probe_public_result(observation_id=request["observation_id"], plan_ref="plan-" + "c" * 32)
        if corruption == "missing_query":
            result["results"] = []
        elif corruption == "wrong_arm":
            result["results"][0]["arm"] = "right"
        else:
            variant = copy.deepcopy(result["results"][0]["variants"][0])
            variant["target_pose_xyzw"][0] += .01
            result["results"][0]["variants"].append(variant)
        return result
    runtime.motion_probe_callback = probe
    with pytest.raises(ToolError) as error:
        runtime._motion_probe({"queries": [{"query_id": "q0", "arm": "left", "pose": {
            "position_m": [-.1, 0., 1.], "orientation_xyzw": [0., 0., 0., 1.],
        }}], "variant_policy": "exact"})
    assert error.value.code == "environment_contract"
    assert runtime._known_plan_refs == {}


def test_motion_probe_consumes_no_action(
    tmp_path: Path,
) -> None:
    runtime, _observation, persisted = primitive_runtime(tmp_path)
    runtime._known_plan_refs = {}
    requested: list[dict[str, Any]] = []
    plan_ref = "plan-" + "a" * 32

    def probe(request: dict[str, Any]) -> dict[str, Any]:
        requested.append(request)
        return _probe_public_result(
            observation_id=str(request["observation_id"]), plan_ref=plan_ref,
        )

    runtime.motion_probe_callback = probe
    result = runtime._motion_probe({
        "queries": [{
            "query_id": "q0", "arm": "left",
            "pose": {
                "position_m": [-0.1, 0.0, 1.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }],
        "variant_policy": "exact",
    })
    assert result["robot_decisions_consumed"] == 0
    assert result["world_state_changed"] is False
    assert persisted == []
    assert requested[0]["queries"][0]["pose"]["orientation_xyzw"] == [
        0.0, 0.0, 0.0, 1.0,
    ]


def test_motion_probe_normalizes_valid_quaternion_and_rejects_bad_norm_at_tool_boundary(
    tmp_path: Path,
) -> None:
    runtime, _observation, persisted = primitive_runtime(tmp_path)
    runtime._known_plan_refs = {}
    requested: list[dict[str, Any]] = []

    def probe(request: dict[str, Any]) -> dict[str, Any]:
        requested.append(request)
        result = _probe_public_result(
            observation_id=str(request["observation_id"]),
            plan_ref="plan-" + "b" * 32,
        )
        result["results"][0]["query_id"] = request["queries"][0]["query_id"]
        return result

    runtime.motion_probe_callback = probe
    result = runtime._motion_probe({
        "queries": [{
            "query_id": "rounded", "arm": "left",
            "pose": {
                "position_m": [-0.1, 0.0, 1.0],
                "orientation_xyzw": [0.0, 0.7071068, 0.0, 0.7071068],
            },
        }],
        "variant_policy": "parallel_jaw_equivalents",
    })
    assert result["robot_decisions_consumed"] == 0
    sent = requested[0]["queries"][0]["pose"]["orientation_xyzw"]
    assert np.linalg.norm(sent) == pytest.approx(1.0, abs=1e-12)
    assert persisted == []

    with pytest.raises(ToolError) as caught:
        runtime._motion_probe({
            "queries": [{
                "query_id": "bad", "arm": "left",
                "pose": {
                    "position_m": [-0.1, 0.0, 1.0],
                    "orientation_xyzw": [-7.09, -3.79, 0.714, 0.7],
                },
            }],
            "variant_policy": "parallel_jaw_equivalents",
        })
    assert caught.value.code == "invalid_arguments"
    assert caught.value.field == "orientation_xyzw"
    assert len(requested) == 1


def test_motion_probe_projection_excludes_action_only_topk_diagnostics() -> None:
    result = _probe_public_result(
        observation_id="observation-0", plan_ref="plan-" + "c" * 32,
    )
    planner = dict(result["results"][0]["variants"][0]["planner"])
    planner.update({
        "position": np.zeros((4, 6), dtype=np.float64),
        "topk_selection": {
            "candidate_count": 3,
            "admitted_count": 1,
            "selected_candidate_index": 2,
            "candidates": [{
                "candidate_index": 2,
                "status": "Success",
                "admission": "accepted",
                "planning_attempt": "trajopt",
            }],
        },
    })
    assert "topk_selection" in planner_public_summary(planner)
    projected = planner_public_summary(
        planner, include_topk_selection=False,
    )
    assert "topk_selection" not in projected
