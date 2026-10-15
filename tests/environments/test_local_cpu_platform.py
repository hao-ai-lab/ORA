"""Local worker graphics and Unix socket constraints across desktop platforms."""
import json
import os
from pathlib import Path

import pytest

from ora.environment import graphics_probe, rpc
from ora.orchestrator import host_storage, paths, resources, run_scratch, shm_segments


@pytest.fixture
def graphics_environment(monkeypatch):
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("PYOPENGL_PLATFORM", raising=False)
    return monkeypatch


def test_darwin_native_graphics_never_selects_egl(graphics_environment):
    graphics_environment.setattr(graphics_probe.platform, "system", lambda: "Darwin")
    graphics_probe.configure_libero_graphics()
    state = graphics_probe.probe()
    assert state["backend"] == "cgl"
    assert state["environment"]["PYOPENGL_PLATFORM"] == "darwin"
    assert state["usable"] and state["egl_devices"] is None


def test_darwin_rejects_inherited_linux_loader(graphics_environment):
    graphics_environment.setattr(graphics_probe.platform, "system", lambda: "Darwin")
    graphics_environment.setenv("PYOPENGL_PLATFORM", "egl")
    with pytest.raises(ValueError, match="contradicts"):
        graphics_probe.configure_libero_graphics()
    assert not graphics_probe.probe()["usable"]


def test_linux_preserves_default_egl_and_explicit_software_rendering(graphics_environment):
    graphics_environment.setattr(graphics_probe.platform, "system", lambda: "Linux")
    graphics_probe.configure_libero_graphics()
    assert graphics_probe.os.environ["MUJOCO_GL"] == "egl"
    graphics_environment.setenv("MUJOCO_GL", "osmesa")
    graphics_environment.delenv("PYOPENGL_PLATFORM")
    graphics_probe.configure_libero_graphics()
    assert graphics_probe.probe()["usable"]
    assert graphics_probe.os.environ["PYOPENGL_PLATFORM"] == "osmesa"


def test_unknown_graphics_binding_is_not_reported_usable(graphics_environment):
    graphics_environment.setenv("MUJOCO_GL", "not-a-backend")
    assert not graphics_probe.probe()["usable"]


def test_desktop_socket_budget_uses_short_fallback(tmp_path, monkeypatch):
    # The platform temporary directory stands for a short root such as macOS's
    # $TMPDIR; pytest's own tmp_path may itself exceed the 103-byte budget.
    short_root = Path("/tmp/ora-socket-budget")
    monkeypatch.setattr(rpc, "MAX_UNIX_SOCKET_PATH_BYTES", 103)
    monkeypatch.setattr(rpc.tempfile, "gettempdir", lambda: str(short_root))
    session = rpc.LocalRpcEnvironmentSession(
        python_executable=Path("/usr/bin/python3"), project_root=tmp_path,
        output_root=tmp_path / ("deep-desktop-run-" * 8), worker_config={},
    )
    assert session.socket_root_fallback
    assert session.socket_root == short_root.resolve()
    assert len(str(session.socket_path).encode()) <= 103
    assert session.process is None


def _local_plan():
    return {"runtime": {"kind": "libero_pro", "required_host": "local-cpu"},
            "policy": {"kind": "native_api"}}


def _local_cache(monkeypatch, cache):
    expanduser = Path.expanduser
    monkeypatch.setattr(Path, "expanduser", lambda path: (
        cache if str(path) == "~/.cache/ora" else expanduser(path)))


@pytest.mark.parametrize("marker", [False, True])
def test_local_cpu_scratch_profile_fits_deep_mac_checkout(monkeypatch, marker):
    cache = Path("/Users/example/.cache/ora")
    _local_cache(monkeypatch, cache)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", Path(
        "/Users/example-user/code/projects/embodied-agents/ora-agent/outputs"))
    monkeypatch.setattr(host_storage, "MAX_UNIX_SOCKET_PATH_BYTES", 103)
    if marker:
        monkeypatch.setenv(host_storage.EXECUTION_HOST_ENV, "local-cpu")
    else:
        monkeypatch.delenv(host_storage.EXECUTION_HOST_ENV, raising=False)
    monkeypatch.setattr(host_storage, "_capacity", lambda path, floor: {
        "path": str(path), "filesystem_probe": str(cache.parent),
        "free_bytes": 10 * host_storage.GIB, "free_gib": 10,
        "minimum_free_gib": floor, "disabled": False, "passed": True,
    })
    monkeypatch.setattr(host_storage, "_writable", lambda path: True)
    monkeypatch.setattr(run_scratch, "census", lambda base: {
        "base": str(base), "present": False,
    })

    profile = resources.load_host_profile("local-cpu")
    assert profile["runtime_storage"]["tmp_root"] == "~/.cache/ora"
    config = host_storage.storage_config(_local_plan())
    check = host_storage._tmpdir_check(config, lambda name, passed, detail: {
        "check": name, "passed": passed, "detail": detail,
    })
    assert check["passed"] is True
    assert config["tmp_base"] == cache
    assert check["detail"]["source"] == "host_profile:local-cpu"
    assert check["detail"]["host_id"] == "local-cpu"
    assert check["detail"]["host_source"] == ("execution_marker" if marker else "required_host")
    assert check["detail"]["rpc_socket_fallback_bytes"] <= 103
    assert run_scratch.socket_fallback_bytes(run_scratch.default_base()) > 103


@pytest.mark.host_storage_real
@pytest.mark.parametrize("cleanup", ["discard", "release"])
def test_local_cpu_profile_stages_and_cleans_owned_scratch(tmp_path, monkeypatch, cleanup):
    cache = tmp_path / "cache" / "ora"
    _local_cache(monkeypatch, cache)
    monkeypatch.delenv(host_storage.EXECUTION_HOST_ENV, raising=False)
    monkeypatch.setattr(paths, "OUTPUTS_ROOT", tmp_path / "deep" / "checkout" / "outputs")
    shm = tmp_path / "shm"
    shm.mkdir()
    monkeypatch.setattr(shm_segments, "SHM_ROOT", shm)

    run_id = "ora-local-scratch-" + cleanup
    staged = host_storage.stage_launch(_local_plan(), run_id)
    record = staged["runtime_scratch"]
    directory = Path(record["tmp_root"])
    assert directory == cache / run_scratch.run_key(run_id)
    assert directory.is_dir() and not directory.is_relative_to(paths.OUTPUTS_ROOT)
    assert record["source"] == "host_profile:local-cpu"
    assert record["host_id"] == "local-cpu"
    assert host_storage.scratch_environment(staged) == {
        name: str(directory) for name in ("TMPDIR", "TMP", "TEMP")}
    assert host_storage.bind_scratch_owner(staged, os.getpid()) is True
    owner = json.loads((directory / run_scratch.OWNER_FILE).read_text())
    assert (owner["run_id"], owner["supervisor_pid"]) == (run_id, os.getpid())
    sibling = cache / "unrelated"
    sibling.mkdir()

    if cleanup == "discard":
        assert host_storage.discard_staged(staged) is True
    else:
        run_root = tmp_path / "run"
        run_root.mkdir()
        (run_root / "launch.json").write_text(json.dumps({
            "run_id": run_id, "runtime_scratch": record,
        }))
        assert host_storage.release_run_scratch(run_root) is True
    assert not directory.exists()
    assert sibling.is_dir()
