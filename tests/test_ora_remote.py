"""Local SSH browser coordination without a simulator, provider or SSH server."""

import io
import json
import shlex
import threading

import pytest

from ora.orchestrator.ora_remote import open_when_ready, ssh_command


def test_ssh_forward_is_loopback_and_preserves_literal_remote_arguments():
    repo = "/space path/$(never-run)`also-never`"
    run = "/runs/special ' quote; $(never-run)"
    command, url = ssh_command("person@gpu-host", repo, ["watch", run], local_port=40201, remote_port=40202)
    assert command[command.index("-L") + 1] == "127.0.0.1:40201:127.0.0.1:40202"
    assert "ExitOnForwardFailure=yes" in command and "-tt" in command
    remote = command[-1]
    assert shlex.split(remote.split(" && exec ", 1)[0]) == ["cd", "--", repo]
    args = shlex.split(remote.split(" && exec ", 1)[1])
    assert args[:3] == ["./scripts/ora", "watch", run]
    assert args[-2:] == ["--world-public-url", url]
    assert "--confirm-launch" not in args
    assert args[args.index("--world-browser") + 1] == "off"


@pytest.mark.parametrize("host,repo,args", [
    ("-oProxyCommand=bad", "/repo", ["watch", "/run"]),
    ("gpu", "relative", ["watch", "/run"]),
    ("gpu", "/repo", ["launch", "/run"]),
    ("gpu", "/repo", ["watch", "/run", "--world-view=off"]),
])
def test_invalid_launcher_bindings_are_refused(host, repo, args):
    with pytest.raises(ValueError):
        ssh_command(host, repo, args, local_port=40201, remote_port=40202)


def test_only_healthy_world_service_opens_local_browser_once():
    calls, opened, warnings = [], [], []
    responses = [b'{"service":"wrong","status":"ready"}',
                 json.dumps({"service": "ora-live-world", "status": "ready"}).encode()]

    def fetch(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(responses.pop(0))

    assert open_when_ready("http://127.0.0.1:40201/live", threading.Event(), timeout=2,
                           fetch=fetch, opener=lambda url: opened.append(url) or True, report=warnings.append)
    assert len(calls) == 2 and calls[0][0].endswith("/health")
    assert opened == ["http://127.0.0.1:40201/live"] and warnings == []


def test_failed_or_closed_transport_does_not_open_browser_or_relaunch():
    opened, warnings = [], []
    stopped = threading.Event()

    def fetch(*args, **kwargs):
        stopped.set()
        raise OSError("SSH detached")

    assert not open_when_ready("http://127.0.0.1:40201/live", stopped, timeout=1,
                               fetch=fetch, opener=opened.append, report=warnings.append)
    assert opened == [] and warnings == []


def test_remote_launcher_help_needs_no_ssh_connection(capsys):
    from ora.orchestrator.ora_remote import main
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    assert "SSH hostname" in capsys.readouterr().out


def test_browser_os_failure_is_reported_without_retrying_tabs():
    opened, warnings = [], []

    def opener(url):
        opened.append(url)
        raise OSError("browser launcher failed")

    assert not open_when_ready("http://127.0.0.1:40201/live", threading.Event(), timeout=1,
                               fetch=lambda *args, **kwargs: io.BytesIO(b'{"service":"ora-live-world","status":"ready"}'),
                               opener=opener, report=warnings.append)
    assert len(opened) == len(warnings) == 1


def test_detaching_terminates_only_owned_ssh_process(monkeypatch):
    from ora.orchestrator import ora_remote
    calls = []

    class Child:
        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if timeout is None:
                raise KeyboardInterrupt
            return 0

        def terminate(self):
            calls.append(("terminate",))

    monkeypatch.setattr(ora_remote, "available_port", lambda requested: 40201)
    monkeypatch.setattr(ora_remote.subprocess, "Popen", lambda command: calls.append(command) or Child())
    monkeypatch.setattr(ora_remote, "open_when_ready", lambda *args, **kwargs: None)
    assert ora_remote.main(["gpu-host", "--repo", "/repo", "--", "interactive", "--spec", "/task.json"]) == 130
    assert sum(isinstance(call, list) for call in calls) == 1
    assert ("terminate",) in calls
    assert "--confirm-launch" not in calls[0][-1]
