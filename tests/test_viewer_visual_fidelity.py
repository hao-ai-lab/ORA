"""Original visual evidence is the default; frame sampling never skips physics."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from viewer import trajectory as video


@pytest.mark.parametrize("backend", ["robotwin", "libero_pro", "robodojo"])
def test_review_uses_original_images_even_with_replay_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    run = tmp_path / "runs" / backend
    (run / "policy-state/steps").mkdir(parents=True)
    (run / "native/controller-replay").mkdir(parents=True)
    # Deliberately unusable: default review must never need replay setup.
    (run / "native/controller-replay/manifest.json").write_text("{}")
    (run / "plan.json").write_text(json.dumps({"runtime": {"kind": backend}}))
    Image.new("RGB", (32, 32), "red").save(run / "original.png")
    (run / "policy-state/steps/step-0000.json").write_text(json.dumps({
        "step": 0, "observation": {"id": "original-0"},
        "cameras": {"agentview": {"files": {
            "rgb_lossless_png": {"path": "original.png"},
        }}},
    }))
    (run / "policy-state/manifest.json").write_text(json.dumps({"steps": [
        {"step": 0, "path": "policy-state/steps/step-0000.json"},
    ]}))

    def forbidden(*_args, **_kwargs):
        pytest.fail("default review attempted simulator reconstruction")

    monkeypatch.setattr(video, "_controller_replay_source", forbidden)
    trajectory = video.compile_trajectory(run)
    assert trajectory["video_source"]["kind"] == "original_observation_sequence"
    assert video.pending_replay_source(run) is None
    output, manifest = video.compile_trajectory_video(run, allow_replay_compile=False)
    assert output.is_file()
    assert manifest["visual_fidelity"] == "original_frames_only"
    assert manifest["video_source_kind"] == "original_observation_sequence"
    assert manifest["controller_replay"] is False
    assert manifest["frame_count"] == 1
    assert video.compile_trajectory_video(run)[0] == output

    # A stale render cannot silently accept an image changed after projection.
    Image.new("RGB", (32, 32), "blue").save(run / "original.png")
    with pytest.raises(RuntimeError, match="hash drifted"):
        video._render_trajectory_frame(run, trajectory, 0)
    assert video.trajectory_video_status(run)["cached"] is False


def test_robotwin_frame_sampling_executes_every_target_and_settle_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ora.environment import replay_worker as worker
    from ora.environment.backends.robotwin import session

    applied = []
    physics = []
    captures = []
    environment = SimpleNamespace()
    environment.scene = SimpleNamespace(step=lambda: physics.append("step"))
    environment._update_render = lambda: None

    def dense_action(targets, save_freq):
        assert save_freq == 1
        for target in targets:
            applied.append(target)
            environment.scene.step()
            environment._update_render()

    environment.take_dense_action = dense_action
    fake_session = SimpleNamespace(
        start=lambda: None, close=lambda: None,
        _require_environment=lambda: environment,
    )

    class Writer:
        sequence = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def append(self, _images, **_kwargs):
            captures.append(len(physics))
            self.sequence += 1

    monkeypatch.setattr(session, "RoboTwinLiveSession", lambda **_kwargs: fake_session)
    monkeypatch.setattr(worker, "_FrameWriter", Writer)
    monkeypatch.setattr(worker, "_robotwin_replay_needs_planner", lambda _: True)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: {})
    monkeypatch.setattr(worker, "_recorded_control_step_count", lambda *_: 7)
    monkeypatch.setattr(worker, "_robotwin_control_sequence", lambda *_: list(range(7)))
    monkeypatch.setattr(worker, "_robotwin_images", lambda _: {})
    worker._replay_robotwin(
        plan={"runtime": {"task_config": "test", "required_revision": "source",
                          "workspace_bounds_m": []},
              "task": "test", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"camera_panels": [], "entries": [{
            "replayable": True, "controller": "robotwin_dense_joint_targets",
            "execution": {"settle_steps": 2},
        }]},
        manifest_path=tmp_path / "manifest.json", project_root=tmp_path,
        output_root=tmp_path,
        sampling={"mode": "action_keyframes_v1", "max_frames_per_action": 4,
                  "hold_steps_per_sample": 8},
    )
    assert applied == list(range(7))
    assert len(physics) == 9
    assert captures == [0, 1, 4, 6, 9]


@pytest.mark.parametrize("controller", ["libero_osc_pose", "libero_joint_position"])
def test_libero_legacy_replay_frames_use_each_actual_step_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, controller: str,
) -> None:
    """Direct worker steps must not render an unchanged outer reset cache."""
    import numpy as np
    from ora.environment import replay_worker as worker
    from ora.environment.backends.libero_pro import session as backend

    def observation(number):
        return {"agentview_image": np.full((2, 3, 3), number, dtype=np.uint8),
                "robot0_eye_in_hand_image": np.full((2, 3, 3), number + 10, dtype=np.uint8)}

    actual_steps = []
    frames = []
    environment = SimpleNamespace(current_raw_obs=[observation(0)])
    current = observation(0)

    def setter(name, value):
        nonlocal current
        if name == "harness_replay_command":
            actual_steps.append(value)
            current = observation(len(actual_steps))

    environment.env = SimpleNamespace(set_env_attr=setter)
    fake_session = SimpleNamespace(
        start=lambda: None, close=lambda: None,
        _require_environment=lambda: environment,
        _raw_observation=lambda: environment.current_raw_obs[0],
        _worker_attr=lambda name: current if name == "harness_replay_observation" else None,
    )

    class Writer:
        sequence = 0

        def __init__(self, *_args, **_kwargs):
            pass

        def append(self, images, **kwargs):
            frames.append((kwargs["phase"], int(images["agentview"][0, 0, 0]),
                           int(images["wrist"][0, 0, 0])))
            self.sequence += 1

    arrays = {"commands": np.zeros((2, 8 if controller == "libero_joint_position" else 7))}
    if controller == "libero_joint_position":
        arrays["refresh_osc_command"] = np.zeros((1, 7))
    monkeypatch.setattr(backend, "LiberoProSession", lambda **_kwargs: fake_session)
    monkeypatch.setattr(worker, "_FrameWriter", Writer)
    monkeypatch.setattr(worker, "_required_environment_path", lambda *_: tmp_path)
    monkeypatch.setattr(worker, "_libero_qualified_controller_binding", lambda *_: {})
    monkeypatch.setattr(worker, "_artifact_arrays", lambda *_: arrays)
    worker._replay_libero(
        plan={"runtime": {"suite": "libero_10", "task_id": 0,
                          "expected_task_language": "test",
                          "required_libero_pro_revision": "l", "required_data_revision": "d",
                          "workspace_bounds_m": []},
              "task": "test", "seed": 0, "budgets": {"max_sim_steps": 100}},
        manifest={"camera_panels": ["agentview", "wrist"], "entries": [{
            "replayable": True, "controller": controller,
        }]}, manifest_path=tmp_path / "manifest.json", output_root=tmp_path,
    )
    assert frames[:3] == [("reset", 0, 10), ("control", 1, 11), ("control", 2, 12)]
    if controller == "libero_joint_position":
        assert frames[3:] == [("refresh", 3, 13)]
    assert len(actual_steps) == len(frames) - 1
