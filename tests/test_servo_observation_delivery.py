"""Fresh action-bound Servo views and exact multi-image runtime inputs (offline)."""

from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image

from ora.agent.context.action_views import (
    PostActionObservationError,
    materialize_post_action_observation,
    resolve_post_action_images,
)
from ora.agent.contracts import PolicyTurnInput


def _observation(root, cameras, *, sequence=1):
    source = root / f"source-{sequence}"
    source.mkdir()
    records = {}
    for index, name in enumerate(cameras):
        directory = source / name
        directory.mkdir()
        path = directory / "rgb.png"
        Image.new("RGB", (12, 10), (sequence, index * 30, 100)).save(path)
        records[name] = {
            "width": 12, "height": 10,
            "files": {"rgb_lossless_png": {
                "path": f"{name}/rgb.png",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }},
        }
    observation = {
        "schema": "ora-observation.v1",
        "observation_id": f"observation-{sequence}", "sequence": sequence,
        "captured_at_unix_s": 123.0 + sequence,
        "cameras": records, "proprioception": {"end_effectors": {}},
    }
    (source / "observation.json").write_text(json.dumps(observation))
    return observation, source


def _project(root, observation, source, *, defaults, roles, arms=("right",), mode="delta"):
    return materialize_post_action_observation(
        run_root=root, workspace=root / "workspace", observation=observation,
        observation_root=source, policy_state={"step": observation["sequence"],
            "state_sha256": "state-hash", "observation_id": observation["observation_id"]},
        previous_action={"schema": "ora-servo-action.v1", "action": "servo",
            "action_id": "servo-1", "observation_id": "observation-0",
            "arguments": {"mode": mode, **({"arms": list(arms)} if mode == "hold"
                else {"targets": {arm: {} for arm in arms}})}},
        view_config={"automatic_actual_views": list(defaults), "rgb_resolution": "native"},
        camera_roles=roles,
    )


@pytest.mark.parametrize("cameras,defaults,roles,arms,expected", [
    (["agentview", "wrist"], ["agentview"], {"agentview": "external", "wrist": "right_wrist"}, ["right"], ["agentview", "wrist"]),
    (["head", "left_wrist", "right_wrist"], ["head"], {"left_wrist": "left_wrist", "right_wrist": "right_wrist"}, ["left", "right"], ["head", "left_wrist", "right_wrist"]),
    (["observer_camera", "left_camera", "right_camera"], ["observer_camera", "left_camera"], {"left_camera": "left_wrist", "right_camera": "right_wrist"}, ["left"], ["observer_camera", "left_camera"]),
])
@pytest.mark.parametrize("mode", ["delta", "hold"])
def test_fresh_defaults_plus_selected_wrist_deduplicate(tmp_path, cameras, defaults, roles, arms, expected, mode):
    observation, source = _observation(tmp_path, cameras)
    projected = _project(tmp_path, observation, source, defaults=defaults, roles=roles, arms=arms, mode=mode)
    assert projected["observation_id"] == observation["observation_id"]
    assert projected["action_id"] == "servo-1"
    assert projected["captured_at_unix_s"] == 124.0
    assert projected["status"] == "complete"
    assert [image["camera"] for image in projected["images"]] == expected
    actual = resolve_post_action_images({"post_action_observation": projected}, root=tmp_path, path_key="run_path")
    for original, delivered in zip(projected["images"], actual):
        assert delivered["path"].read_bytes() == (source / original["camera"] / "rgb.png").read_bytes()
    # Same pixels after hold still receive a new observation-bound artifact identity.
    later, later_source = _observation(tmp_path, cameras, sequence=2)
    next_projected = _project(tmp_path, later, later_source, defaults=defaults, roles=roles, arms=arms, mode="hold")
    assert next_projected["images"][0]["path"] != projected["images"][0]["path"]


def test_missing_wrist_is_explicit_but_default_failure_and_stale_images_fail_closed(tmp_path):
    observation, source = _observation(tmp_path, ["head"])
    projected = _project(tmp_path, observation, source, defaults=["head"], roles={"left_wrist": "left_wrist"}, arms=["left"])
    assert projected["status"] == "degraded"
    assert projected["missing_wrist_views"] == [{"arm": "left", "camera": "left_wrist", "reason": "current_camera_unavailable"}]
    (source / "head/rgb.png").unlink()
    with pytest.raises(PostActionObservationError, match="default"):
        _project(tmp_path, observation, source, defaults=["head"], roles={}, arms=["left"])


def test_post_action_image_bytes_are_verified_and_legacy_is_unchanged(tmp_path):
    assert resolve_post_action_images({}, root=tmp_path) == []
    observation, source = _observation(tmp_path, ["head"])
    projected = _project(tmp_path, observation, source, defaults=["head"], roles={})
    delivered = tmp_path / "workspace" / projected["images"][0]["path"]
    delivered.chmod(0o644)
    delivered.write_bytes(b"different")
    with pytest.raises(PostActionObservationError, match="hash"):
        resolve_post_action_images({"post_action_observation": projected}, root=tmp_path / "workspace")


def test_unreadable_wrist_disclosure_never_exposes_host_paths(tmp_path):
    observation, source = _observation(tmp_path, ["head", "wrist"])
    path = source / "wrist/rgb.png"
    path.write_bytes(b"not an image")
    observation["cameras"]["wrist"]["files"]["rgb_lossless_png"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "observation.json").write_text(json.dumps(observation))
    record = _project(tmp_path, observation, source, defaults=["head"], roles={"wrist": "right_wrist"})
    assert record["missing_wrist_views"] == [{"arm": "right", "camera": "wrist", "reason": "current_rgb_decode_or_read_failed"}]
    assert str(tmp_path) not in json.dumps(record)


def test_policy_turn_immutable_multi_image_extension_preserves_legacy_kwargs(tmp_path):
    args = dict(task_id="task", task_instruction="Act", context={}, image_path=None,
        observation_id="obs", observation_step=1)
    assert "image_paths" not in PolicyTurnInput(**args).runtime_kwargs()
    paths = [tmp_path / "head.png", tmp_path / "wrist.png"]
    turn = PolicyTurnInput(**args, image_paths=paths)
    paths.clear()
    assert turn.runtime_kwargs()["image_paths"] == (tmp_path / "head.png", tmp_path / "wrist.png")


def _published_servo(tmp_path):
    from ora.environment.synthetic import SyntheticSession
    from ora.agent.evidence.policy_state import PolicyStateStore
    from ora.agent.context.codex import ContextProjector
    from test_native_causal_runtime import PROMPT, MANUAL, PROJECT

    environment = SyntheticSession(tmp_path / "environment", "servo", 0,
        main_camera="agentview", camera_names=("agentview", "wrist"))
    environment.start()
    observation, _ = environment.observe(tmp_path / "source")
    source = tmp_path / "source" / observation["observation_id"]
    store = PolicyStateStore(tmp_path, "servo-test", "plan-test")
    store.publish(observation=observation, observation_root=source, cause="mutation",
        task_instruction="Inspect fresh motion.", previous_public_action=None, remaining_budget={})
    projector = ContextProjector(tmp_path / "workspace", tmp_path / "control", PROMPT, MANUAL, PROJECT,
        main_camera="agentview", run_root=tmp_path,
        view_config={"automatic_actual_views": ["agentview"], "rgb_resolution": "native"},
        camera_roles={"wrist": "right_wrist"})
    pointer = projector.publish(observation=observation, observation_root=source,
        task_instruction="Inspect fresh motion.", remaining_budget={}, policy_state=store.steps[-1],
        previous_action={"schema": "ora-servo-action.v1", "action": "servo",
            "action_id": "action-1", "observation_id": "earlier-observation",
            "arguments": {"mode": "hold", "arms": ["right"]}})
    return pointer, projector, store
