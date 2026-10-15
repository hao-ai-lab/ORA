"""Direct-pose grasp plans on the maintained surfaces never call reachability probing.

The motion.probe candidate-reference tests left with the last surface exposing
motion.probe; the remaining fixture still binds a probe callback so any call is
recorded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.tools.grasp import GRASP_QUERY_SCHEMA
from ora.agent.tools.runtime import ManipulationToolRuntime
from tests.agent.perception.test_grounding_grasp_runtime import (
    BOUNDS, MANUAL, FakeContactGraspNet, FakeSam3, _publish_object_observation,
)


def _runtime(tmp_path: Path, profile: str, *, arm="right"):
    control, observation_id = _publish_object_observation(tmp_path)
    dual_arm = resolve_tool_surface(profile).supports("dual_arm_grasp_plan_v1")
    camera = "head_camera" if dual_arm else "agentview"
    if dual_arm:
        pointer = control / "current-observation.json"
        current = json.loads(pointer.read_text())
        root = Path(current["source_root"])
        observation_path = root / "observation.json"
        observation = json.loads(observation_path.read_text())
        for original, renamed in (("agentview", "head_camera"), ("wrist", "right_camera")):
            (root / original).rename(root / renamed)
            record = observation["cameras"].pop(original)
            for reference in record["files"].values():
                reference["path"] = reference["path"].replace(original + "/", renamed + "/", 1)
            observation["cameras"][renamed] = record
        shutil.copytree(root / "right_camera", root / "left_camera")
        left = copy.deepcopy(observation["cameras"]["right_camera"])
        for reference in left["files"].values():
            reference["path"] = reference["path"].replace("right_camera/", "left_camera/", 1)
        observation["cameras"]["left_camera"] = left
        effectors = observation["proprioception"]["end_effectors"]
        effectors["left"] = copy.deepcopy(effectors["right"])
        observation_path.chmod(0o600)
        observation_path.write_text(json.dumps(observation))
        current["observation_sha256"] = hashlib.sha256(observation_path.read_bytes()).hexdigest()
        pointer.write_text(json.dumps(current))
    requests = []

    def probe(request):
        requests.append(copy.deepcopy(request))
        return {
            "schema": "ora.motion-probe-result.v1",
            "observation_id": request["observation_id"],
            "world_mutation_sequence": 0,
            "variant_policy": request["variant_policy"],
            "robot_decisions_consumed": 0,
            "world_state_changed": False,
            "results": [{
                "query_id": query["query_id"], "arm": query["arm"],
                "reachable": True, "selected_variant_id": "variant-00-exact-+0deg",
                "plan_ref": "plan-" + "a" * 32,
                "selected_target_pose_xyzw": (
                    query["pose"]["position_m"] + query["pose"]["orientation_xyzw"]
                ),
                "variants": [{
                    "variant_id": "variant-00-exact-+0deg", "reachable": True,
                    "target_pose_xyzw": (
                        query["pose"]["position_m"] + query["pose"]["orientation_xyzw"]
                    ),
                    "endpoint_error": {"translation_m": 0.0, "orientation_rad": 0.0},
                    "trajectory_steps": 120, "trajectory_duration_s": 2.4,
                    "plan_ref": "plan-" + "a" * 32,
                    "planner": {
                        "called": True, "status": "SUCCESS", "stage": "plan_single",
                        "valid_query": True, "attempts": 1, "trajopt_attempts": 1,
                        "used_graph": False, "goalset_index": None, "path_steps": 120,
                        "interpolation_dt_s": 0.02,
                        "errors": {"position": None, "rotation": None, "cspace": None},
                        "timings_s": {key: None for key in (
                            "solve", "ik", "graph", "trajopt", "finetune", "total",
                        )},
                        "error": None,
                    },
                }],
            } for query in request["queries"]],
        }

    sam = FakeSam3()
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL, event_path=tmp_path / "events.jsonl",
        workspace_bounds_m=copy.deepcopy(BOUNDS), state_root=None,
        tool_surface_profile=profile, controlled_end_effector="right",
        segmentation_backend=sam, grounding_backend=sam,
        grasp_backend=FakeContactGraspNet(), motion_probe_callback=probe,
    )
    grounded = runtime.invoke("vision_ground", {
        "camera": camera, "text": "red object", "max_candidates": 1,
    })
    target_ref = grounded["candidates"][0]["target_ref"]
    planned = runtime.invoke("grasp_plan", {
        "target_ref": target_ref, "max_candidates": 1, **({"arm": arm} if dual_arm else {}),
    })
    events = [json.loads(line) for line in runtime.event_path.read_text().splitlines()]
    event = next(item for item in events if item["type"] == "grasp_plan_materialized")
    manifest_path = tmp_path / event["manifest_artifact"]["path"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == GRASP_QUERY_SCHEMA == "ora.grasp-query.v7"
    assert "observation_id" not in manifest
    assert manifest["identity"]["observation_id"] == observation_id
    assert manifest["compiled_candidates"][0]["pregrasp_pose"] == (
        planned["candidates"][0]["pregrasp_pose"]
    )
    requests.clear()
    return runtime, planned, requests, event, manifest_path, manifest


@pytest.mark.parametrize("profile", [
    "libero_placement_direct_overhead_v1", "robotwin_toolchain_b2_closed_noservo_v2",
])
def test_direct_pose_grasp_never_calls_probe(tmp_path, monkeypatch, profile):
    def forbidden(*args, **kwargs):
        raise AssertionError("direct-pose grasp must not call reachability probing")
    monkeypatch.setattr(ManipulationToolRuntime, "_motion_probe", forbidden)
    runtime, planned, requests, *_ = _runtime(tmp_path, profile)
    assert planned["candidates"]
    assert "motion_selection" not in planned
    assert all("motion_probe" not in c and c["compiler_checks"]["reachability"] == "delegated_to_move_pose"
               for c in planned["candidates"])
    from ora.agent.composition import validate_schema_instance
    validate_schema_instance({k: v for k, v in planned.items() if k != "images_base64"},
                             runtime.surface.result_schemas["grasp_plan"])
    assert not requests
