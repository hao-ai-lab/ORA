"""Resident perception services: readiness reports, binding and lifecycle."""
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


def _record(pid=4242):
    return {"pid": pid, "command": ["python", "-m", "service"], "log": "service.log",
            "process_identity": {"process_group_id": pid, "boot_id": "boot"}}


def test_identity_state_separates_uninspectable_from_drift(monkeypatch):
    from ora.agent.model_services import process

    monkeypatch.setattr(process, "process_alive", lambda pid: True)

    def denied(pid, command, **kwargs):
        raise PermissionError(13, "Permission denied", f"/proc/{pid}/exe")

    monkeypatch.setattr(process, "capture_service_process_identity", denied)
    assert process.service_process_identity_state(_record()) == process.IDENTITY_UNINSPECTABLE
    assert process.service_process_identity_matches(_record()) is False
    with pytest.raises(RuntimeError, match="another container, sandbox or user namespace") as refused:
        process.terminate_service_process(_record())
    assert "process group 4242" in str(refused.value)
    monkeypatch.setattr(process, "capture_service_process_identity",
                        lambda pid, command, **kwargs: {"process_group_id": pid, "boot_id": "other"})
    assert process.service_process_identity_state(_record()) == process.IDENTITY_DRIFTED
    with pytest.raises(RuntimeError, match="drifted identity"):
        process.terminate_service_process(_record())
    monkeypatch.setattr(process, "process_alive", lambda pid: False)
    assert process.service_process_identity_state(_record()) == process.IDENTITY_EXITED
    assert process.terminate_service_process(_record()) is True


def test_stop_reports_a_refused_service_and_still_stops_the_other(tmp_path, monkeypatch):
    state = tmp_path / "services"
    state.mkdir()
    for name, pid in (("segmentation", 11), ("grasp_proposals", 12)):
        (state / f"{name}.json").write_text(json.dumps(_record(pid)))

    def terminate(record):
        if record["pid"] == 11:
            raise RuntimeError("cannot inspect")
        return True

    monkeypatch.setattr(services, "terminate_service_process", terminate)
    result = services.lifecycle("stop", root=tmp_path, state=state, runtime_state_root=tmp_path / "runtime")
    assert result["segmentation"] == {"stopped": False, "pid": 11, "refused": "cannot inspect"}
    assert result["grasp_proposals"] == {"stopped": True}


def test_start_names_an_uninspectable_running_service_instead_of_a_port_error(tmp_path, monkeypatch):
    state = tmp_path / "services"
    state.mkdir()
    (state / "segmentation.json").write_text(json.dumps(_record()))
    monkeypatch.setattr(services, "service_process_identity_state", lambda record: services.IDENTITY_UNINSPECTABLE)
    for prefix in ("SAM", "GRASP"):
        for field in ("PYTHON", "ROOT", "CHECKPOINT", "GPU"):
            monkeypatch.setenv(f"B2_{prefix}_{field}", "x")
    with pytest.raises(ValueError, match="segmentation: resident service pid 4242 is alive"):
        services.lifecycle("start", root=tmp_path, state=state)
