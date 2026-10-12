"""Historical grasp baselines retain identity through actual provider inputs."""

import base64
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ora.agent.context.skill_observations import (
    materialize_skill_observations, resolve_skill_images,
)
from ora.agent.context.action_views import PostActionObservationError
from test_servo_observation_delivery import _observation, _published_servo


def _baseline(root, cameras=("agentview", "wrist")):
    observation, source = _observation(root, cameras, sequence=9)
    receipt = {
        "schema": "ora.manipulation-skill-sequence-receipt.v1",
        "skill": "grasp_execute", "status": "completed", "action_id": "action-1",
        "committed_sequence": ["open", "pregrasp", "approach", "close", "lift", "verify"],
        "stage_receipts": [],
        "skill_receipt": {"observation_evidence": [{
            "role": "post_close_before_lift", "arm": "right",
            "observation_id": observation["observation_id"], "observation_root": str(source),
            "observation_sha256": hashlib.sha256((source / "observation.json").read_bytes()).hexdigest(),
        }]},
    }
    return receipt, source


def test_baseline_prefers_selected_arm_and_rejects_drift_and_escape(tmp_path):
    cameras = ("head_camera", "front_camera", "left_camera", "right_camera")
    receipt, source = _baseline(tmp_path, cameras)
    args = dict(receipt=receipt, run_root=tmp_path, workspace=tmp_path / "workspace", camera_names=cameras)
    images = materialize_skill_observations(**args)
    assert [i["camera"] for i in images] == ["head_camera", "right_camera"]
    assert len(resolve_skill_images({"skill_observations": images}, root=tmp_path / "workspace")) == 2
    raw = (source / "observation.json").read_bytes()
    (source / "observation.json").write_bytes(raw + b" ")
    with pytest.raises(PostActionObservationError, match="hash"):
        materialize_skill_observations(**args)
    (source / "observation.json").write_bytes(raw)
    receipt["skill_receipt"]["observation_evidence"][0]["observation_root"] = str(tmp_path.parent)
    with pytest.raises(PostActionObservationError, match="escapes"):
        materialize_skill_observations(**args)


@pytest.mark.parametrize("retention", ["action_step_lossless_v4", "rewrite_compact_v1"])
def test_native_provider_request_includes_labeled_historical_baseline(tmp_path, retention):
    from ora.agent.composition import resolve_tool_surface
    from ora.agent.native.session import NativePlanner
    from ora.agent.providers.native import OpenAICompatibleProvider, translate_anthropic_request

    context, _, _ = _published_servo(tmp_path)
    receipt, source = _baseline(tmp_path)
    planner = NativePlanner.__new__(NativePlanner)
    planner.engine = SimpleNamespace()
    planner.run_root, planner.supports_images = tmp_path, True
    surface = resolve_tool_surface("native_atomic_causal_v1")
    surface = replace(surface, capabilities=(*surface.capabilities, "prepared_grasp_execution_v2"),
                      view_config={"automatic_actual_views": ["agentview", "wrist"]})
    planner.tools = SimpleNamespace(surface=surface, controlled_end_effector="right")
    current = planner._actual_snapshot(context)
    planner.last_actual_snapshot = {**current, "actual_sequence": 1}
    planner.actual_sequence = 1
    planner.pending_receipt_body = receipt
    planner.pending_receipt = {"action_id": "action-1"}
    planner._current_observation_id = lambda: context["post_action_observation"]["observation_id"]
    planner.context_retention = retention
    message = planner._causal_observation_message(context=context, terminal_lifecycle=False)
    assert "historical intermediate frame" in json.dumps(message)
    request = {"model": "offline", "max_tokens": 64, "messages": [message]}
    for body in (OpenAICompatibleProvider.prepare_request(request), translate_anthropic_request(request)):
        serialized = json.dumps(body)
        for camera in ("agentview", "wrist"):
            assert base64.b64encode((source / camera / "rgb.png").read_bytes()).decode() in serialized
    from ora.agent.context.projection import _causal_projection_violations
    assert _causal_projection_violations(message["content"]) == []
