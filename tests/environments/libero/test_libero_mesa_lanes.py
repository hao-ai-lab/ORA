"""Software-rendered (Mesa llvmpipe) LIBERO lanes on the cluster fixture host profile."""
from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from ora.environment import graphics_probe
from ora.environment.rpc import WORKER_ENVIRONMENT_ALLOWLIST, _worker_environment
from ora.orchestrator import preflight
from ora.orchestrator.resources import ExecutionResourceError, resolve_execution_resource

MESA_VENDOR = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"


def _lane(name: str) -> dict:
    return resolve_execution_resource(name, runtime_kind="libero_pro", required_host="test-cluster")


def test_a_mesa_lane_selects_the_software_vendor_and_lists_gpu_zero_for_egl():
    binding = _lane("libero-mesa-g3s1")
    assert binding["slot_id"] == "lpm-g3s1"
    assert binding["worker_environment"] == {
        "CUDA_VISIBLE_DEVICES": "3,0", "MUJOCO_EGL_DEVICE_ID": "0", "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": MESA_VENDOR,
        "LIBGL_ALWAYS_SOFTWARE": "1", "LP_NUM_THREADS": "2",
    }
    assert _lane("libero-mesa-g0s0")["worker_environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    # The NVIDIA-EGL lanes keep their exact binding.
    assert _lane("libero-g3s1")["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "3", "MUJOCO_EGL_DEVICE_ID": "3"}


def test_the_worker_receives_every_mesa_variable():
    environment = _worker_environment(preflight.PROJECT_ROOT, _lane("libero-mesa-g5s0")["worker_environment"])
    assert environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == MESA_VENDOR
    assert environment["LIBGL_ALWAYS_SOFTWARE"] == "1" and environment["LP_NUM_THREADS"] == "2"
    assert {"__EGL_VENDOR_LIBRARY_FILENAMES", "LIBGL_ALWAYS_SOFTWARE", "LP_NUM_THREADS"} <= WORKER_ENVIRONMENT_ALLOWLIST


@pytest.mark.parametrize("environment,message", [
    ({"__EGL_VENDOR_LIBRARY_FILENAMES": "50_mesa.json"}, "absolute files"),
    ({"CUDA_VISIBLE_DEVICES": "3,3"}, "distinct numeric GPUs"),
    ({"CUDA_VISIBLE_DEVICES": "GPU-abc"}, "distinct numeric GPUs"),
    ({"LD_PRELOAD": "/tmp/x.so"}, "untrusted name"),
])
def test_lane_environments_stay_closed(monkeypatch, tmp_path, environment, message):
    host = tmp_path / "hosts" / "test-host.yaml"
    host.parent.mkdir()
    host.write_text(
        "schema: ora.execution-host.v1\nid: test-host\nssh_alias: null\n"
        "execution_marker: {environment_variable: ORA_EXECUTION_HOST, required_value: test-host}\n"
        "project_root: /x\nexecution_resources:\n  lane:\n    runtime_kind: libero_pro\n"
        "    resource_class: libero_pro_sim\n    slot_id: s\n    minimum_free_memory_mib: 1\n"
        f"    worker_environment: {environment!r}\n".replace("'", '"'))
    from ora.orchestrator import resources
    monkeypatch.setattr(resources, "HOSTS_ROOT", host.parent)
    with pytest.raises(ExecutionResourceError, match=message):
        resolve_execution_resource("lane", runtime_kind="libero_pro", required_host="test-host")


def test_the_graphics_probe_reports_an_explicit_vendor_file(monkeypatch):
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_FILENAMES", MESA_VENDOR)
    monkeypatch.setenv("__EGL_VENDOR_LIBRARY_DIRS", "/nonexistent")
    assert graphics_probe._vendor_icd_files() == [MESA_VENDOR]


def test_execution_resource_checks_the_lane_compute_gpu(monkeypatch):
    binding = _lane("libero-mesa-g3s1")
    plan = {"runtime": {"kind": "libero_pro", "required_host": "test-cluster", "resource_binding": binding}}
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(preflight.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout="0, 90000\n3, 120000\n", stderr=""))
    check = preflight._execution_resource(plan)
    assert check["passed"] and check["detail"]["gpu_index"] == 3
