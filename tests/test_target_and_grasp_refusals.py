"""Target, grasp and model-service refusals say who failed and what can change.

EXP11B refusal census B3, B4, D1-D7 and E5 (0059 open-bugs triage rows 2, 9,
10, 20, 28-32, 35). Each test pins one envelope property rather than a message:

* a refused target value states that only a different value can succeed, and
  a malformed or unknown ref names the pattern and the refs that are current;
* a store fault behind a ref is infrastructure and never blames the argument;
* a model service that cannot match its bound identity again is a permanent
  infrastructure fault, a 4xx keeps the service's own ``detail``, and a target
  too thin to propose on names ``target_ref``;
* an id ``grasp_plan`` rejected or did not display is refused as such, with the
  ids that can execute now;
* ``stale_observation`` keeps one meaning: never raised as non-retryable.
"""

from __future__ import annotations

import ast
import inspect
import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from ora.agent.tools import (
    artifact_errors, service_refusals, skill_candidates, skill_geometry, target_refusals,
)
from ora.agent.tools.robot import ToolError
from test_manipulation_skill_driver import _observation_id, _publish_grasp_candidate, _runtime


def _refused(call, *args, **kwargs) -> dict:
    with pytest.raises(ToolError) as caught:
        call(*args, **kwargs)
    return caught.value.as_dict()


# --- target refs (rows 2, 9, 28, 29) ----------------------------------------

def _grounded(tmp_path):
    from test_robodojo_b2_parity import tool_runtime

    runtime = tool_runtime(tmp_path)
    grounded = runtime.invoke("vision_ground", {"camera": "head", "text": "red mug", "max_candidates": 1})
    return runtime, grounded["candidates"][0]["target_ref"]


def test_grasp_plan_refuses_a_stale_ref_with_both_observations(tmp_path):
    runtime, target_ref = _grounded(tmp_path)
    pointer, source, observation = runtime._current()
    runtime._current = lambda: ({**pointer, "observation_id": "observation-later"}, source, observation)
    error = _refused(runtime.invoke, "grasp_plan", {"target_ref": target_ref, "arm": "left"})
    assert error["code"] == "stale_observation" and error["field"] == "target_ref"
    assert error["received"] == target_ref
    assert error["constraints"] == {
        "bound_observation_id": observation["observation_id"],
        "current_observation_id": "observation-later",
    }
    assert error["retryable"] is True and error["retry_requires_changed_arguments"] is True


@pytest.mark.parametrize("value", ["T0", "grasp-a1", "target-XYZ"])
def test_a_malformed_ref_names_the_pattern_and_the_current_refs(tmp_path, value):
    # The align_plan path: its schema publishes no pattern, so the handler states it.
    runtime, target_ref = _grounded(tmp_path)
    control, root, observation, _pose, _gripper = runtime._current_grip_state()
    error = _refused(runtime._current_target_geometry, control=control, root=root,
                     observation=observation, target_ref=value)
    assert error["code"] == "target_ref_unavailable" and error["received"] == value
    assert error["constraints"]["pattern"] == target_refusals.TARGET_REF_PATTERN
    assert error["constraints"]["current_target_refs"] == [target_ref]
    assert error["retry_requires_changed_arguments"] is True and error["infrastructure"] is False


def test_a_well_formed_unknown_ref_is_an_argument_refusal(tmp_path):
    runtime, target_ref = _grounded(tmp_path)
    unknown = "target-" + "0" * 24
    error = _refused(runtime.invoke, "grasp_plan", {"target_ref": unknown, "arm": "left"})
    assert error["code"] == "target_ref_unavailable"
    assert error["constraints"]["current_target_refs"] == [target_ref]
    assert error["retry_requires_changed_arguments"] is True


def test_a_drifted_target_store_is_infrastructure_not_the_argument(tmp_path):
    runtime, target_ref = _grounded(tmp_path)
    index = runtime.artifact_root / "segmented-target-index" / f"{target_ref}.json"
    record = json.loads(index.read_text())
    index.chmod(0o644)
    index.write_text(json.dumps({**record, "manifest_sha256": "0" * 64}))
    error = _refused(runtime.invoke, "grasp_plan", {"target_ref": target_ref, "arm": "left"})
    assert error["code"] == "artifact_hash_mismatch" and error["infrastructure"] is True
    assert error["retryable"] is False and "retry_requires_changed_arguments" not in error


def test_unmeasurable_geometry_asks_for_a_different_ref():
    error = target_refusals.unmeasurable_geometry(target_ref="target-" + "1" * 24, geometry={}).as_dict()
    assert error["retry_requires_changed_arguments"] is True and error["retryable"] is True


# --- model services (rows 20, 30) --------------------------------------------

def _status_error(status: int, body: dict | None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://127.0.0.1:8114/v1/segment-point")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("status", request=request, response=response)


def test_a_4xx_keeps_the_service_detail_as_an_argument_refusal():
    failure = artifact_errors.http_failure(
        _status_error(422, {"detail": "text exceeds 32 tokens"}), service="SAM3", what="request")
    error = service_refusals.refusal(failure, code="grounding_failed").as_dict()
    assert error["code"] == "invalid_arguments" and error["infrastructure"] is False
    assert error["constraints"]["detail"] == "text exceeds 32 tokens"
    assert error["constraints"]["status"] == 422
    assert "text exceeds 32 tokens" in error["message"]


@pytest.mark.parametrize("raised", [
    lambda: _status_error(503, None),
    lambda: httpx.ConnectError("refused", request=httpx.Request("GET", "http://127.0.0.1:1")),
])
def test_a_transient_service_failure_stays_retryable(raised):
    failure = artifact_errors.http_failure(raised(), service="SAM3", what="request")
    error = service_refusals.refusal(failure, code="segmentation_failed").as_dict()
    assert error["code"] == "segmentation_failed" and error["retryable"] is True
    assert error["constraints"]["failure"] == "transient"


def test_identity_drift_after_a_restart_is_permanent_infrastructure():
    from ora.agent.tools.grasp import _require_bound_service_identity

    with pytest.raises(artifact_errors.ModelServiceError) as drifted:
        _require_bound_service_identity(
            health={"instance_id": "new"}, binding={"instance_id": "old"}, service="SAM3")
    error = service_refusals.refusal(drifted.value, code="segmentation_failed").as_dict()
    assert error["code"] == "internal_tool_error" and error["retryable"] is False
    assert error["infrastructure"] is True


def test_the_sam3_client_keeps_the_422_detail(tmp_path):
    import numpy as np

    from ora.agent.tools.grasp import Sam3ApiSegmenter

    client = Sam3ApiSegmenter()
    object.__setattr__(client, "_client", httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(422, json={"detail": "point_uv outside image"}))))
    with pytest.raises(artifact_errors.ModelServiceError) as refused:
        client.segment(rgb_u8=np.zeros((4, 4, 3), dtype=np.uint8),
                       depth_m=np.ones((4, 4)), point_uv=(1, 1))
    assert refused.value.failure == "request" and refused.value.detail == "point_uv outside image"


def test_a_target_too_thin_to_propose_on_names_target_ref():
    from ora.agent.tools.grasp import PcaTopDownGraspProposer

    with pytest.raises(artifact_errors.TargetInsufficient) as thin:
        PcaTopDownGraspProposer().propose(
            scene_points_camera_m=[[0, 0, 1]] * 4, target_points_camera_m=[[0, 0, 1]] * 4,
            camera_to_world=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            target_ref="target-x", workspace_bounds_m={"x": [-1, 1], "y": [-1, 1], "z": [0, 2]},
            max_candidates=1)
    error = service_refusals.refusal(thin.value, code="grasp_proposal_failed", target_ref="target-x").as_dict()
    assert (error["field"], error["received"]) == ("target_ref", "target-x")
    assert error["retry_requires_changed_arguments"] is True and error["infrastructure"] is False


# --- grasp_execute candidates (rows 10, 31, 32) -------------------------------

def _withholding_manifest(runtime, observation_id: str) -> None:
    root = runtime.artifact_root / "grasp-queries" / "grasp-query-withheld"
    root.mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "query_ref": "grasp-query-withheld",
        "identity": {"observation_id": observation_id, "target_ref": "target-1"},
        "compiled_candidates": [],
        "diagnostics": {"rejected": [{"candidate_id": "grasp-r1", "reasons": ["aperture"]}],
                        "compiler_rejected": [{"candidate_id": "grasp-c1", "reason": "ik"}]},
        "candidate_selection": {"pool": [{"candidate_id": "grasp-a1"}, {"candidate_id": "grasp-p9"}],
                                "displayed_candidate_ids": ["grasp-a1"]},
    }))


@pytest.mark.parametrize("candidate_id,disposition", [
    ("grasp-r1", "rejected"), ("grasp-c1", "rejected_by_compiler"), ("grasp-p9", "not_displayed"),
])
def test_a_withheld_candidate_is_refused_as_withheld(tmp_path, candidate_id, disposition):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    _withholding_manifest(runtime, _observation_id(runtime))
    error = _refused(runtime.invoke, "grasp_execute", {"candidate_id": candidate_id, "rationale": "chosen"})
    assert error["code"] == "candidate_unavailable" and error["received"] == candidate_id
    assert error["constraints"]["disposition"] == disposition
    assert error["constraints"]["valid_candidate_ids"] == ["grasp-a1"]
    assert error["retry_requires_changed_arguments"] is True


def test_an_unknown_or_stale_candidate_says_only_a_changed_id_helps(tmp_path):
    runtime = _runtime(tmp_path)
    _publish_grasp_candidate(runtime, observation_id=_observation_id(runtime))
    unknown = _refused(runtime.invoke, "grasp_execute", {"candidate_id": "grasp-zz", "rationale": "x"})
    assert unknown["constraints"]["valid_candidate_ids"] == ["grasp-a1"]
    assert unknown["retry_requires_changed_arguments"] is True
    host = SimpleNamespace(artifact_root=runtime.artifact_root, surface=runtime.surface)
    stale = _refused(skill_candidates._grasp_candidate, host, "grasp-a1", "observation-later")
    assert stale["code"] == "stale_candidate_observation"
    assert stale["retry_requires_changed_arguments"] is True


def test_a_blank_rationale_is_refused_before_any_candidate_lookup(tmp_path):
    runtime = _runtime(tmp_path)
    error = _refused(runtime.invoke, "grasp_execute", {"candidate_id": "grasp-unknown", "rationale": "   "})
    assert error["field"] == "rationale"


def test_unusable_candidate_geometry_names_the_candidate_and_is_retryable():
    error = _refused(skill_geometry._pose, None, "contact_pose", "grasp-a1")
    assert error["code"] == "candidate_geometry_unavailable"
    assert (error["field"], error["received"]) == ("candidate_id", "grasp-a1")
    assert error["retryable"] is True and error["retry_requires_changed_arguments"] is True


def test_grasp_plan_rejects_a_candidate_whose_lift_leaves_the_workspace():
    bounds = {"x": [-1, 1], "y": [-1, 1], "z": [0, 1.0]}
    raw = [{"candidate_id": "low"}, {"candidate_id": "high"}]
    compiled = [{"candidate_id": "low", "contact_pose": {"position_m": [0, 0, 0.9]}},
                {"candidate_id": "high", "contact_pose": {"position_m": [0, 0, 0.99]}}]
    kept_raw, kept, rejected = skill_geometry.split_liftable(raw, compiled, 0.03, bounds)
    assert [item["candidate_id"] for item in kept_raw] == ["low"] == [item["candidate_id"] for item in kept]
    assert rejected[0]["candidate_id"] == "high" and "1.0200 m" in rejected[0]["reason"]


# --- one meaning for stale_observation (row 35) -------------------------------

def test_stale_observation_is_never_raised_as_non_retryable():
    from ora.agent import tools

    package = sys.modules[tools.__name__]
    offenders = []
    for path in sorted(__import__("pathlib").Path(inspect.getfile(package)).parent.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ToolError"):
                continue
            keywords = {kw.arg: kw.value for kw in node.keywords}
            code, retryable = keywords.get("code"), keywords.get("retryable")
            if (isinstance(code, ast.Constant) and code.value == "stale_observation"
                    and isinstance(retryable, ast.Constant) and retryable.value is False):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders
