"""RoboDojo startup probes tolerate a slow shared disk without hiding a hang."""
import subprocess
from pathlib import Path

from ora.environment.backends.robodojo import graphics_compat
from ora.orchestrator import robodojo_qualification


def test_worker_and_host_readback_share_one_bound():
    assert graphics_compat.READBACK_TIMEOUT_S == robodojo_qualification.GRAPHICS_READBACK_TIMEOUT_S
    # Observed 34.6 s during a writeback storm; the worker start limit is 600 s.
    assert 60 <= graphics_compat.READBACK_TIMEOUT_S <= 300


def test_activate_uses_the_bound(monkeypatch, tmp_path):
    seen = {}

    def fake_run(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(graphics_compat, "installation", lambda root: (None, {}))
    monkeypatch.setattr(graphics_compat.subprocess, "run", fake_run)
    try:
        graphics_compat.activate(Path(tmp_path))
    except subprocess.TimeoutExpired:
        pass
    assert seen["timeout"] == graphics_compat.READBACK_TIMEOUT_S


def test_host_inspection_uses_the_bound(tmp_path):
    seen = {}

    def runner(*args, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    python = tmp_path / "env/bin/python"
    try:
        robodojo_qualification.inspect_graphics_compatibility(python, runner=runner)
    except Exception:
        pass
    assert seen["timeout"] == robodojo_qualification.GRAPHICS_READBACK_TIMEOUT_S


def test_worker_environment_keeps_the_host_proxy(monkeypatch, tmp_path):
    """Uncached Isaac materials are fetched from S3; this host reaches it only via a proxy."""
    from ora.environment.rpc import _worker_environment
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8989")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("DEEPSEEK_LOW_API_KEY", "secret")
    environment = _worker_environment(tmp_path)
    assert environment["HTTPS_PROXY"] == "http://127.0.0.1:8989"
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert "DEEPSEEK_LOW_API_KEY" not in environment


def test_graphics_compat_directory_may_be_relocated_behind_a_link(tmp_path):
    """The compatibility files stay contained when the directory itself is a link."""
    real = tmp_path / "ram" / "graphics-compat"
    real.mkdir(parents=True)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "graphics-compat").symlink_to(real)
    root = (runtime / "graphics-compat").resolve()
    probe = (root / "layer.json").resolve()
    assert probe.is_relative_to(root)
    import inspect
    assert '(runtime_directory / "graphics-compat").resolve()' in inspect.getsource(graphics_compat.installation)


def test_worker_environment_drops_a_credentialed_or_remote_proxy(monkeypatch, tmp_path):
    from ora.environment.rpc import _worker_environment
    for name in ("HTTP_PROXY", "http_proxy", "https_proxy"):
        monkeypatch.delenv(name, raising=False)
    for value in ("http://user:secret@127.0.0.1:8989", "http://secret-proxy.invalid:3128"):
        monkeypatch.setenv("HTTPS_PROXY", value)
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        environment = _worker_environment(tmp_path)
        assert "HTTPS_PROXY" not in environment and "NO_PROXY" not in environment
