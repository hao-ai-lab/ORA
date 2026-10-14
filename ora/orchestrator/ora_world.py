"""Optional read-only world observer for ORA; never owns a Runtime or simulator."""

from __future__ import annotations

import json
import os
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_WORLD_VIEW = {
    "view": "auto", "port": 0, "browser": "auto", "bind_host": "127.0.0.1",
    "human_live_watch": True, "human_interactive": True, "json_implicit": False,
    "sealed_watch_auto": False, "connected_requires_decoded_frame": True,
}


def add_world_options(parser) -> None:
    parser.add_argument("--world-view", choices=("auto", "off"), default=DEFAULT_WORLD_VIEW["view"],
                        help="automatically show the exact live run's cameras (default: human terminal only)")
    parser.add_argument("--world-port", type=int, default=DEFAULT_WORLD_VIEW["port"], help="loopback world-view port; 0 chooses a free port")
    parser.add_argument("--world-browser", choices=("auto", "off"), default=DEFAULT_WORLD_VIEW["browser"],
                        help="open the local world view after its server is ready")
    parser.add_argument("--world-public-url", help="browser-facing /live URL, for example a local SSH-forwarded URL")


def observer_factory(args, *, enabled: bool):
    """Keep headless defaults quiet, including direct JSON protocol consumers."""
    if not 0 <= args.world_port <= 65535:
        raise ValueError("world-port must be in [0, 65535]")
    public_url = args.world_public_url
    if public_url:
        parsed = urllib.parse.urlsplit(public_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
                or parsed.path != "/live" or parsed.query or parsed.fragment
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)):
            raise ValueError("world-public-url must be an HTTP(S) /live URL with a valid port and no credentials, query or fragment")
    if args.world_view == "off" or not enabled:
        return None

    def create(*, root: Path | None, finished: bool, emit: Callable[[Mapping[str, Any]], None]):
        if finished:
            emit({"op": "world_view", "phase": "replay", "connected": False, "visible": False,
                  "url": None, "message": "Live world view is off for a finished-run transcript replay."})
            return None
        return WorldObserver(emit, root=root, port=args.world_port, browser=args.world_browser,
                             public_url=public_url)
    return create


class WorldObserver:
    """A background observer; UI polling never waits for HTTP or image decoding."""

    def __init__(self, emit: Callable[[Mapping[str, Any]], None], *, root: Path | None = None,
                 port: int = 0, browser: str = "auto", public_url: str | None = None,
                 server_factory=None, open_browser=webbrowser.open_new_tab,
                 health_check=None, environment=None) -> None:
        self.emit = emit
        self.port, self.browser, self.public_url = port, browser, public_url
        self.server_factory, self.open_browser = server_factory, open_browser
        self.health_check = health_check or self._health
        self.environment = dict(os.environ if environment is None else environment)
        self._lock = threading.Lock()
        self._root, self._generation = root, 0
        self._stop = threading.Event()
        self._snapshot: dict[str, Any] = {"op": "world_view", "phase": "starting", "connected": False,
                                          "visible": False, "url": public_url, "message": "Starting world view…"}
        self._last: str | None = None
        self._server = None
        self._browser_message = ""
        self._browser_thread = None
        self._server_close_lock = threading.Lock()
        self._server_close_claimed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="ora-world-view")
        self._thread.start()

    @staticmethod
    def _health(url: str) -> bool:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=0.5) as response:
            value = json.loads(response.read(4096))
            return response.status == 200 and value.get("service") == "ora-live-world" and value.get("status") == "ready"

    def bind(self, root: Path | None) -> None:
        with self._lock:
            self._root = root
            self._generation += 1
            self._snapshot = {"op": "world_view", "phase": "waiting", "connected": False, "visible": False,
                              "url": self.public_url or self._snapshot.get("url"),
                              "run_id": root.name if root else None, "message": "Waiting for the selected run's cameras…"}
        self.poll()

    def poll(self) -> None:
        with self._lock:
            snapshot = dict(self._snapshot)
        encoded = json.dumps(snapshot, sort_keys=True)
        if encoded != self._last:
            self._last = encoded
            self.emit(snapshot)

    def _run(self) -> None:
        server = None
        try:
            if self.server_factory is None:
                from viewer.live_world import WorldViewServer
                factory = WorldViewServer
            else:
                factory = self.server_factory
            server = factory(host=DEFAULT_WORLD_VIEW["bind_host"], port=self.port).start()
            with self._lock:
                self._server = server
                root, generation = self._root, self._generation
            server.bind(root)
            remote = bool(self.environment.get("SSH_CONNECTION") or self.environment.get("SSH_TTY"))
            if self.browser == "auto" and not remote:
                self._browser_thread = threading.Thread(target=self._open_browser_when_ready,
                                                        args=(server.url, generation), daemon=True,
                                                        name="ora-world-browser")
                self._browser_thread.start()
            elif remote and not self.public_url:
                with self._lock:
                    self._browser_message = "SSH session: forward this port and open the URL on your computer."
            while not self._stop.is_set():
                with self._lock:
                    root, current = self._root, self._generation
                if current != generation:
                    server.bind(root)
                    generation = current
                state = dict(server.status())
                with self._lock:
                    if current == self._generation:
                        self._snapshot = {**state, "op": "world_view", "url": self.public_url or server.url,
                                          "browser_message": self._browser_message}
                self._stop.wait(0.2)
        except Exception as exc:
            with self._lock:
                self._snapshot = {"op": "world_view", "phase": "unavailable", "connected": False,
                                  "visible": False, "url": self.public_url,
                                  "message": f"World view unavailable: {type(exc).__name__}: {exc}"}
        finally:
            if server is not None:
                self._close_server(server)

    def _open_browser_when_ready(self, service_url: str, generation: int) -> None:
        # OS browser launchers can block. Keep them outside camera polling and shutdown.
        ready = False
        for _ in range(4):
            if self._stop.is_set():
                return
            try:
                ready = self.health_check(service_url.rsplit("/", 1)[0] + "/health")
            except (OSError, ValueError):
                pass
            if ready or self._stop.wait(0.1):
                break
        message = ""
        if ready and not self._stop.is_set():
            try:
                opened = self.open_browser(self.public_url or service_url)
            except Exception:
                opened = False
            if not opened:
                message = "Open this URL in your browser."
        elif not ready:
            message = "World-view health check failed; open the URL when ready."
        with self._lock:
            if not self._stop.is_set() and generation == self._generation:
                self._browser_message = message

    def _close_server(self, server) -> None:
        with self._server_close_lock:
            if self._server_close_claimed:
                return
            self._server_close_claimed = True
        server.close()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            server = self._server
        if server is not None:
            self._close_server(server)
        self._thread.join(timeout=1)


def watch_observer(args):
    """The world view of `ora watch`: on in a human terminal, or when a public URL is given."""
    import sys

    human = sys.stdout.isatty() and args.emit is None
    explicit = getattr(args, "world_public_url", None) is not None
    return observer_factory(args, enabled=(human and not args.plain) or explicit)
