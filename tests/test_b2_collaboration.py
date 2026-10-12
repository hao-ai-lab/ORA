"""Local B2 model-service operator admission and direct LIBERO driver contracts."""
from __future__ import annotations

import copy
import json
import time

import pytest

from ora.agent.model_services import operator as services


def test_report_binds_content_adapter_endpoint_and_age(tmp_path, monkeypatch):
    monkeypatch.setenv("B2_SAM_PORT", "18114")
    monkeypatch.setenv("B2_GRASP_PORT", "18115")
    health = {key: "fixture" for key in services.FIELDS}
    report = {"schema": services.REPORT_SCHEMA, "created_unix_s": time.time(), "passed": True,
              "adapter_sha256": services.grounding_grasp_adapter_sha256(),
              "checks": {k: True for k in ("single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup")},
              "services": {name: {"endpoint": endpoint, "health": health}
                           for name, endpoint in services.endpoints().items()}}
    path = tmp_path / (services.digest(report) + ".json")
    path.write_text(json.dumps(report))
    bound = services.binding(path, "test-surface", ["test-capability"])
    assert len(bound["services"]) == 2
    assert bound["services"]["segmentation"]["tool_surface"] == "test-surface"
    monkeypatch.setenv("B2_SAM_PORT", "18116")
    with pytest.raises(ValueError, match="endpoint"):
        services.binding(path, "test-surface", [])
    monkeypatch.setenv("B2_SAM_PORT", "18114")
    changed = copy.deepcopy(report)
    changed["created_unix_s"] -= 86401
    stale = tmp_path / (services.digest(changed) + ".json")
    stale.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="stale"):
        services.binding(stale, "test-surface", [])
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed"):
        services.binding(path, "test-surface", [])


@pytest.mark.parametrize("marker", ["leases/robotwin--slot0.json", "active-run.json"])
def test_stop_refuses_resource_lease_before_touching_service(tmp_path, monkeypatch, marker):
    runtime_state = tmp_path / "runtime"
    lease = runtime_state / marker
    lease.parent.mkdir(parents=True)
    lease.write_text('{}')
    monkeypatch.setattr(services, "terminate_service_process", lambda record: pytest.fail("must not signal"))
    with pytest.raises(ValueError, match="leases exist"):
        services.lifecycle("stop", root=tmp_path, state=tmp_path / "services",
                           runtime_state_root=runtime_state)


def test_stop_requires_explicit_shared_runtime_state(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "terminate_service_process", lambda record: pytest.fail("must not signal"))
    with pytest.raises(ValueError, match="shared Runtime state root"):
        services.lifecycle("stop", root=tmp_path, state=tmp_path / "services")


def test_readiness_allows_multiple_candidates_for_one_point(tmp_path, monkeypatch):
    import numpy as np
    model_health = {key: "fixture" for key in services.FIELDS}
    model_health.update(adapter_sha256=services.grounding_grasp_adapter_sha256(),
                        warmup_complete=True, load_count=1)
    closed = []

    class Client:
        def require_ready(self):
            return dict(model_health)
        def segment(self, **kwargs):
            return np.ones((128, 128), dtype=bool), {}
        def segment_points(self, **kwargs):
            assert kwargs["points_uv"] == [[64, 64]]
            return [{"candidate": i} for i in range(3)], {"query_results": [{"error_code": None}]}
        def ground(self, **kwargs):
            return [], {}
        def close(self):
            closed.append(self)
    monkeypatch.setattr(services, "clients", lambda: {name: Client() for name in services.NAMES})
    path = services.qualify(tmp_path)
    record = json.loads(path.read_text())
    assert record["checks"]["batch_point_wire"] is True
    assert len(closed) == 2
    assert path.name == services.digest(record) + ".json"


def test_services_start_passes_gpu_and_strips_credentials(tmp_path, monkeypatch):
    commands = []
    for prefix in ("SAM", "GRASP"):
        for field, value in {"PYTHON": "/fake/python", "ROOT": "/fake/source", "CHECKPOINT": "/fake/checkpoint", "GPU": "3"}.items():
            monkeypatch.setenv(f"B2_{prefix}_{field}", value)
    monkeypatch.setenv("B2_GRASP_CONFIG", "/fake/config")
    monkeypatch.setenv("DEEPSEEK_LOW_API_KEY", "test-secret-must-not-leak")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("B2_SAM_PORT", "18114")
    monkeypatch.setenv("B2_GRASP_PORT", "18115")

    class Process:
        pid = 111
        def __init__(self, command, **kwargs):
            commands.append((command, kwargs))
    monkeypatch.setattr(services.subprocess, "Popen", Process)
    monkeypatch.setattr(services, "capture_service_process_identity", lambda *a: {"fixture": True})
    result = services.lifecycle("start", root=tmp_path, state=tmp_path / "state")
    assert set(result) == set(services.NAMES)
    assert len(commands) == 2
    for command, kw in commands:
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert "DEEPSEEK_LOW_API_KEY" not in kw["env"]
        assert kw["env"]["HF_HUB_OFFLINE"] == "1"
        assert kw["start_new_session"] is True


def test_direct_libero_reset_official_termination_and_no_auto_reset():
    import numpy as np
    from ora.environment.backends.libero_pro.direct import DirectLiberoEnvironment
    class Simulator:
        def __init__(self):
            self.unwrapped = self
            self.commands = []
            self.done = False
            self.closed = False
        def seed(self, value): self.seed_value = value
        def reset(self): self.commands.clear()
        def set_init_state(self, value): self.state = value
        def step(self, action):
            self.commands.append(action.copy())
            return {"image": np.zeros((2, 2, 3))}, 0.0, self.done, {}
        def close(self): self.closed = True
    sim = Simulator()
    env = DirectLiberoEnvironment(factory=lambda: sim, init_states=["a", "b"],
        first_reset_id=10, reset_id=11, seed=7, task_language="task", max_sim_steps=2)
    env.reset()
    assert sim.state == "b" and sim.seed_value == 7
    assert len(sim.commands) == 15
    assert all(list(a) == [0, 0, 0, 0, 0, 0, -1] for a in sim.commands)
    assert env.get_env_attr("state") == ["b"]
    env.set_env_attr("state", "changed")
    assert sim.state == "changed"
    _, _, done, truncated, _ = env.step(np.zeros((1, 7)))
    assert not done[0] and not truncated[0]
    _, _, done, truncated, _ = env.step(np.zeros((1, 7)))
    assert not done[0] and truncated[0]
    sim.done = True
    assert env.step(np.zeros((1, 7)))[2][0]
    assert len(sim.commands) == 18  # no terminal auto-reset
    env.reset(reset_state_ids=np.array([10]))
    assert sim.state == "a" and env._elapsed_steps == 0
    with pytest.raises(ValueError, match="another task"):
        env.reset(reset_state_ids=np.array([12]))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.full((1, 7), np.nan))
    env.close()
    assert sim.closed


def test_services_reject_changed_configuration_for_live_owned_process(tmp_path, monkeypatch):
    for field, value in {"PYTHON": "/fake/python", "ROOT": "/fake/source",
                         "CHECKPOINT": "/fake/checkpoint", "GPU": "3"}.items():
        monkeypatch.setenv(f"B2_SAM_{field}", value)
    state = tmp_path / "state"
    state.mkdir()
    (state / "segmentation.json").write_text(json.dumps({"pid": 111,
        "command": ["old-command"], "gpu": "2", "source_root": str(tmp_path)}))
    monkeypatch.setattr(services, "service_process_identity_matches", lambda record: True)
    monkeypatch.setattr(services.subprocess, "Popen", lambda *a, **k: pytest.fail("must not start"))
    with pytest.raises(ValueError, match="configuration changed"):
        services.lifecycle("start", root=tmp_path, state=state)
