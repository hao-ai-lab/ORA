"""Legacy RoboTwin reset media observes synchronized poses without new physics."""
from types import SimpleNamespace

import pytest

from ora.environment.backends.robotwin.session import RoboTwinLiveSession


@pytest.mark.parametrize("world,execution", [(True, False), (False, True), (True, True)])
def test_legacy_reset_capture_refreshes_scene_even_without_shared_recorder(world, execution):
    calls = []
    environment = SimpleNamespace(
        robot=SimpleNamespace(
            left_camera=SimpleNamespace(get_pose=lambda: "left-pose"),
            right_camera=SimpleNamespace(get_pose=lambda: "right-pose")),
        cameras=SimpleNamespace(update_wrist_camera=lambda *poses: calls.append(poses)),
        scene=SimpleNamespace(update_render=lambda: calls.append("render")),
        _update_render=lambda: pytest.fail("task renderer would randomize lighting"),
    )
    session = RoboTwinLiveSession.__new__(RoboTwinLiveSession)
    session.execution_capture = {"enabled": False}
    session.operator_world_video, session.operator_execution_video = world, execution
    session._require_environment = lambda: environment
    session._capture_operator_frame = lambda **kw: calls.append(("world", kw))
    session._capture_operator_execution_frame = lambda **kw: calls.append(("execution", kw))
    session._record_operator_capture_failure = lambda exc: pytest.fail(str(exc))
    session._capture_operator_reset_frames(label="round-reset")
    assert calls[:2] == [("left-pose", "right-pose"), "render"]
    assert calls[2] == ("world", {"label": "round-reset"})
    assert calls[3:] == ([("execution", {"action": None, "phase": "reset", "control_step_index": None})]
                          if execution else [])


def test_disabled_legacy_media_does_not_touch_the_renderer():
    session = RoboTwinLiveSession.__new__(RoboTwinLiveSession)
    session.operator_world_video = session.operator_execution_video = False
    session._require_environment = lambda: pytest.fail("no legacy media requested")
    session._capture_operator_reset_frames(label="reset")


def test_reset_render_failure_is_auxiliary_and_never_captures_stale_pixels():
    errors = []
    session = RoboTwinLiveSession.__new__(RoboTwinLiveSession)
    session.operator_world_video, session.operator_execution_video = True, True
    def unavailable():
        raise RuntimeError("camera unavailable")
    session._require_environment = unavailable
    session._record_operator_capture_failure = errors.append
    session._capture_operator_frame = lambda **kw: pytest.fail("stale capture")
    session._capture_operator_execution_frame = lambda **kw: pytest.fail("stale capture")
    session._capture_operator_reset_frames(label="reset")
    assert len(errors) == 1 and str(errors[0]) == "camera unavailable"
