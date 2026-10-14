"""Open ORA on an SSH host with its world view in the local browser.

This launcher forwards one loopback observer port. It never launches a task
itself, retries an SSH session, or adds launch confirmation to the supplied
ORA command. The detached Supervisor retains the run when this UI detaches.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import PurePosixPath
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

_local_http = build_opener(ProxyHandler({})).open


def available_port(requested: int = 0) -> int:
    if isinstance(requested, bool) or not 0 <= requested <= 65535:
        raise ValueError("local port must be between 0 and 65535")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", requested))
        return listener.getsockname()[1]


def ssh_command(host: str, repo: str, command: list[str], *, local_port: int,
                remote_port: int, timeout: float = 30) -> tuple[list[str], str]:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:\[\]-]*", host):
        raise ValueError("host must be an SSH hostname or configured alias, optionally user@host")
    if not PurePosixPath(repo).is_absolute() or any(c in repo for c in "\x00\r\n"):
        raise ValueError("--repo must be an absolute path on the SSH host")
    if not command or command[0] not in {"watch", "interactive"}:
        raise ValueError("supply -- watch RUN or -- interactive --spec PATH")
    if any(arg.split("=", 1)[0].startswith("--world-") for arg in command):
        raise ValueError("the remote launcher owns --world-* options; use its port options instead")
    if any("\x00" in arg for arg in command):
        raise ValueError("command arguments must not contain NUL")
    if not all(isinstance(p, int) and not isinstance(p, bool) and 1 <= p <= 65535
               for p in (local_port, remote_port)):
        raise ValueError("forwarded ports must be between 1 and 65535")
    url = f"http://127.0.0.1:{local_port}/live"
    remote_args = ["./scripts/ora", *command, "--world-view", "auto", "--world-port", str(remote_port),
                   "--world-browser", "off", "--world-public-url", url]
    remote = f"cd -- {shlex.quote(repo)} && exec {shlex.join(remote_args)}"
    return ["ssh", "-tt", "-o", "ExitOnForwardFailure=yes", "-o", f"ConnectTimeout={max(1, int(timeout))}",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-L", f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}", "--", host, remote], url


def open_when_ready(url: str, stopped: threading.Event, *, timeout: float,
                    opener=webbrowser.open_new_tab, fetch=_local_http, report=None) -> bool:
    """Confirm the forwarded observer, then request one local browser tab.

    HTTP readiness is not proof of a displayed frame. The page's decode ACK
    determines that separate state, which ORA displays in the terminal.
    """
    report = report or (lambda message: print(f"ora-remote: {message}", file=sys.stderr))
    health = url.removesuffix("/live") + "/health"
    deadline = time.monotonic() + timeout
    while not stopped.is_set() and time.monotonic() < deadline:
        try:
            with fetch(health, timeout=0.5) as response:
                raw = response.read(4097)
            if len(raw) > 4096:
                raise ValueError("unexpected health response")
            value = json.loads(raw)
            if value.get("service") != "ora-live-world" or value.get("status") != "ready":
                raise ValueError("forwarded service is not the ORA world observer")
            if stopped.is_set():
                return False
            try:
                opened = opener(url)
            except (OSError, webbrowser.Error) as exc:
                report(f"Browser did not open ({exc}); open {url} locally.")
                return False
            if not opened:
                report(f"Browser did not open; open {url} locally. Waiting for the page's first frame.")
                return False
            return True
        except (OSError, URLError, ValueError, AttributeError):
            stopped.wait(0.1)
    if not stopped.is_set():
        report(f"World window is not reachable yet at {url}; check SSH forwarding and the remote CLI. "
               "The SSH session will not be restarted.")
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ora-remote", description=__doc__,
                                     epilog="Example: ora-remote gpu-host --repo /path/to/checkout -- watch /path/to/run")
    parser.add_argument("host", help="SSH hostname or configured alias (normal SSH authentication/config)")
    parser.add_argument("--repo", required=True, help="absolute remote checkout with scripts/ora")
    parser.add_argument("--local-port", type=int, default=0, help="local loopback port (default: choose a free port)")
    parser.add_argument("--remote-port", type=int, help="remote loopback port (default: use chosen local port)")
    parser.add_argument("--connect-timeout", type=float, default=30,
                        help="seconds to wait for the forwarded observer (default 30)")
    return parser


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if "--" not in values:
        if "--help" in values or "-h" in values:
            parser.parse_args(values)
        parser.error("separate launcher options from the ORA command with --")
    boundary = values.index("--")
    args = parser.parse_args(values[:boundary])
    try:
        if not 0 < args.connect_timeout <= 300:
            raise ValueError("connect timeout must be finite and in (0, 300] seconds")
        local_port = available_port(args.local_port)
        command, url = ssh_command(args.host, args.repo, values[boundary + 1:], local_port=local_port,
                                   remote_port=local_port if args.remote_port is None else args.remote_port,
                                   timeout=args.connect_timeout)
        print(f"World window: {url} (opens after the observer responds)", flush=True)
        child = subprocess.Popen(command)
        stopped = threading.Event()
        browser = threading.Thread(target=open_when_ready, args=(url, stopped),
                                   kwargs={"timeout": args.connect_timeout}, daemon=True, name="ora-local-browser")
        browser.start()
        try:
            return child.wait()
        except KeyboardInterrupt:
            # Terminate only this SSH UI; never signal the remote Supervisor.
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            return 130
        finally:
            stopped.set()
            browser.join(timeout=0.7)
    except (OSError, ValueError) as exc:
        print(f"ora-remote: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
