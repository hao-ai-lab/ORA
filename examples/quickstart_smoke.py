"""Run the documented quickstart against a deterministic local HTTP model.

No real API, credential, simulator or GPU is used. The CLI still resolves,
preflights, leases, supervises, evaluates and seals a Native episode.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def smoke() -> None:
    calls = []

    class Model(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(request)
            suffix = "read_manual" if len(calls) == 1 else "submit_action"
            from ora.agent.providers.tool_names import tool_alias
            name = tool_alias("robot." + suffix)
            assert name in {tool["function"]["name"] for tool in request["tools"]}
            arguments = {"section": "Actions"} if len(calls) == 1 else {
                "arm": "left", "pose": {"position_m": [0, 0, 1.02], "orientation_xyzw": [0, 0, 0, 1]},
            }
            body = json.dumps({"id": f"smoke-{len(calls)}", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": None, "tool_calls": [{"id": f"call-{len(calls)}",
                    "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50, "total_tokens": 550}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="ora-quickstart-smoke-") as temporary:
            environment = {key: value for key, value in os.environ.items()
                           if not key.startswith("ORA_") and key not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
            environment.update(ORA_API_KEY="local-smoke-placeholder", ORA_MODEL="smoke-model",
                ORA_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                ORA_WORKSPACE_ROOT=temporary)
            result = subprocess.run([sys.executable, "-m", "ora.orchestrator.cli", "run", "--env", "synthetic"],
                env=environment, capture_output=True, text=True, timeout=150)
            if result.returncode:
                logs = [path.read_text() for path in Path(temporary).rglob("supervisor.log")]
                raise RuntimeError(result.stdout + result.stderr + "\n".join(logs))
            roots = list(Path(temporary).glob("outputs/quickstart/runs/*"))
            assert len(roots) == 1, roots
            root = roots[0]
            outcome = json.loads((root / "result.json").read_text())
            assert outcome["state"] == "completed" and outcome["official_success"] is None, outcome
            assert (root / "seal.json").is_file()
            assert len(calls) == 2
            print("Quickstart passed: two HTTP model calls, one accepted action, sealed Synthetic evidence.")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    smoke()
