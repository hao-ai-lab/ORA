"""Historical semantic re-enactment must not regenerate new Servo control."""

from copy import deepcopy

import pytest

from ora.environment import historical_libero_replay_worker as worker


@pytest.mark.parametrize("payload", [
    {"schema": "ora-servo-action.v1", "action": "servo"},
    {"schema": "ora-servo-action.v1", "action": "move_pose"},
    {"schema": "ora-action.v3", "action": "servo"},
])
@pytest.mark.parametrize("prefix", [[], [{"action": {
    "schema": "ora-action.v3", "action": "move_pose",
}}]])
def test_historical_worker_rejects_servo_before_any_simulator_construction(
    tmp_path, monkeypatch, payload, prefix,
):
    from ora.environment.backends.libero_pro import session

    job = {
        "schema": worker.JOB_SCHEMA,
        "run_root": str(tmp_path / "canonical"), "output_root": str(tmp_path / "derived"),
        "plan": {"task": "test", "seed": 0, "budgets": {"max_sim_steps": 100},
            "runtime": {"suite": "libero_10", "task_id": 0, "expected_task_language": "test",
                "required_rlinf_revision": "a" * 40, "required_libero_pro_revision": "b" * 40,
                "required_data_revision": "c" * 40,
                "workspace_bounds_m": {"x": [-1, 1], "y": [-1, 1], "z": [0, 2]}}},
        "runtime_paths": {key: str(tmp_path / key)
            for key in ("rlinf_root", "libero_pro_root", "data_root")},
        "validation_thresholds": {},
        "sampling": {"mode": "action_keyframes_v1", "max_frames_per_action": 4},
        "actions": [*deepcopy(prefix), {"action": payload}],
    }
    monkeypatch.setattr(worker, "_load", lambda path: job)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported Servo must be rejected before simulator construction")

    monkeypatch.setattr(session, "LiberoProSession", forbidden)
    with pytest.raises(worker.HistoricalReplayError, match="Servo.*recorded.controller"):
        worker.run(tmp_path / "unread-job.json")
    assert not (tmp_path / "derived").exists()
    assert not (tmp_path / "derived.tmp").exists()


@pytest.mark.parametrize("payload", [
    {"schema": "ora-action.v3", "action": "move_pose"},
    {"schema": "ora-action.v4", "action": "target_state"},
    {"schema": "ora-motion-action.v1", "action": "servo_delta"},
])
def test_historical_action_guard_preserves_legacy_payload_bytes(payload):
    entries = [{"action_sequence": 0, "action": deepcopy(payload)}]
    original = deepcopy(entries)
    worker._validate_semantic_actions(entries)
    assert entries == original


@pytest.mark.parametrize("entry", [None, {}, {"action": None}, {"action": []}])
def test_historical_action_guard_rejects_malformed_entries(entry):
    with pytest.raises(worker.HistoricalReplayError, match="malformed"):
        worker._validate_semantic_actions([entry])
