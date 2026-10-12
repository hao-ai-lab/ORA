"""Local resident perception service lifecycle and portable readiness evidence."""
from __future__ import annotations

import concurrent.futures
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from .identity import grounding_grasp_adapter_sha256
from .process import (capture_service_process_identity, isolated_service_environment,
                      service_process_identity_matches, terminate_service_process)

NAMES = {"segmentation": "SAM", "grasp_proposals": "GRASP"}
FIELDS = ("instance_id", "pid", "started_unix_s", "adapter_sha256", "model",
          "load_count", "warmup_complete")
REPORT_SCHEMA = "ora.perception-readiness.v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def endpoints() -> dict[str, str]:
    ports = {name: int(os.environ.get(f"B2_{prefix}_PORT", default))
             for (name, prefix), default in zip(NAMES.items(), ("8114", "8115"))}
    if len(set(ports.values())) != 2 or any(not 1024 <= port <= 65535 for port in ports.values()):
        raise ValueError("service ports must be distinct integers in 1024..65535")
    return {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}


def clients(*, timeout_s: float = 60):
    from ora.agent.tools.grasp import Sam3ApiSegmenter, ContactGraspNetApiProposer
    urls = endpoints()
    return {"segmentation": Sam3ApiSegmenter(endpoint=urls["segmentation"], timeout_s=timeout_s),
            "grasp_proposals": ContactGraspNetApiProposer(endpoint=urls["grasp_proposals"], timeout_s=timeout_s)}


def health() -> dict[str, Any]:
    def inspect(item):
        name, client = item
        try:
            return name, {"ready": True, "health": client.require_ready()}
        except Exception as exc:
            return name, {"ready": False, "error": type(exc).__name__}
        finally:
            client.close()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        return dict(pool.map(inspect, clients(timeout_s=5).items()))


def lifecycle(
    command: str, *, root: Path, state: Path, runtime_state_root: Path | None = None,
) -> dict[str, Any]:
    """Start asynchronously; never replace a listener or signal an unowned PID."""
    state.mkdir(parents=True, exist_ok=True)
    with (state / "operator.lock").open("a") as lock, ExitStack() as stack:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if command == "status":
            return health()
        if command == "stop":
            if runtime_state_root is None:
                raise ValueError("stop requires the shared Runtime state root")
            # Shared resident models may serve any experiment. Be conservative
            # around active/staged work, including another linked checkout.
            lease_root = runtime_state_root / "leases"
            lease_root.mkdir(parents=True, exist_ok=True)
            registry_lock = stack.enter_context((lease_root / ".registry.lock").open("a+"))
            fcntl.flock(registry_lock, fcntl.LOCK_EX)
            leases = list(lease_root.glob("*.json"))
            if (runtime_state_root / "active-run.json").exists():
                leases.append(runtime_state_root / "active-run.json")
            if leases:
                raise ValueError("Runtime leases exist; stop/reconcile runs before stopping shared services")
        result = {}
        for name, prefix in NAMES.items():
            record_path = state / f"{name}.json"
            record = json.loads(record_path.read_text()) if record_path.exists() else None
            if command == "stop":
                if record:
                    result[name] = {"stopped": terminate_service_process(record)}
                continue
            # Refuse occupied ports, including listeners we did not launch.
            import socket
            port = int(endpoints()[name].rsplit(":", 1)[1])
            values = {}
            for field in ("PYTHON", "ROOT", "CHECKPOINT", "GPU"):
                key = f"B2_{prefix}_{field}"
                values[field] = os.environ.get(key, "")
                if not values[field]:
                    raise ValueError(f"missing {key}")
            cmd = [values["PYTHON"], "-m", "ora.agent.model_services.grounding_grasp",
                   "--service", name, "--port", str(port), "--source-root", values["ROOT"],
                   "--checkpoint", values["CHECKPOINT"]]
            if name == "grasp_proposals":
                cmd += ["--config", os.environ["B2_GRASP_CONFIG"]]
            if record and service_process_identity_matches(record):
                if (record.get("command") != cmd or record.get("gpu") != values["GPU"]
                        or record.get("source_root") != str(root)):
                    raise ValueError(f"{name} configuration changed; stop owned services before restarting")
                result[name] = {"status": "already_started", "pid": record["pid"]}
                continue
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
            env = isolated_service_environment(worker_environment={"CUDA_VISIBLE_DEVICES": values["GPU"]},
                extra={"PYTHONPATH": str(root), "PYOPENGL_PLATFORM": "egl"})
            log_path = state / f"{name}-{time.time_ns()}.log"
            with log_path.open("xb") as log:
                process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            try:
                identity = capture_service_process_identity(process.pid, cmd)
            except Exception:
                # This is still our freshly-created Popen, never an old PID file.
                process.terminate()
                process.wait(timeout=10)
                raise
            record = {"pid": process.pid, "command": cmd, "process_identity": identity,
                      "log": str(log_path), "source_root": str(root), "gpu": values["GPU"]}
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            result[name] = {"status": "starting", "pid": process.pid, "log": str(log_path)}
        return result


def qualify(directory: Path) -> Path:
    """Exercise shared wire formats on a generated RGB rectangle; no simulator."""
    import numpy as np
    rgb = np.full((128, 128, 3), 32, dtype=np.uint8)
    rgb[40:92, 48:84] = [220, 48, 32]
    depth = np.ones((128, 128), dtype=np.float32)
    tools = clients()
    try:
        before = {name: client.require_ready() for name, client in tools.items()}
        sam = tools["segmentation"]
        mask, _ = sam.segment(rgb_u8=rgb, depth_m=depth, point_uv=(64, 64))
        masks, batch = sam.segment_points(rgb_u8=rgb, depth_m=depth, points_uv=[[64, 64]])
        grounded, _ = sam.ground(rgb_u8=rgb, text="red rectangle", max_candidates=3)
        after = {name: client.require_ready() for name, client in tools.items()}
        if any(before[name][key] != after[name][key] for name in tools for key in FIELDS):
            raise ValueError("service identity changed during readiness probes")
        if any(h["adapter_sha256"] != grounding_grasp_adapter_sha256() for h in after.values()):
            raise ValueError("service adapter differs from this checkout")
        checks = {"single_point_wire": mask.shape == (128, 128),
                  "batch_point_wire": bool(masks) and len(batch["query_results"]) == 1
                  and batch["query_results"][0]["error_code"] is None, "text_wire": isinstance(grounded, list),
                  "grasp_warmup": after["grasp_proposals"]["warmup_complete"] is True}
        if not all(checks.values()):
            raise ValueError("perception readiness wire probes failed")
        report = {"schema": REPORT_SCHEMA, "created_unix_s": time.time(), "passed": True,
                  "adapter_sha256": grounding_grasp_adapter_sha256(), "checks": checks,
                  "services": {n: {"endpoint": endpoints()[n], "health": h} for n, h in after.items()},
                  "claim_boundary": "Generated-image wire checks and model warmup only; no segmentation quality, grasp geometry or behavioral qualification."}
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (digest(report) + ".json")
        with path.open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        return path
    finally:
        for client in tools.values():
            client.close()


def binding(path: Path, surface: str, capabilities: list[str]) -> dict[str, Any]:
    """Offline binding; the shared launch preflight verifies live identity again."""
    record = json.loads(path.read_text())
    expected_checks = {"single_point_wire", "batch_point_wire", "text_wire", "grasp_warmup"}
    if (path.name != digest(record) + ".json" or record.get("schema") != REPORT_SCHEMA
            or record.get("passed") is not True
            or not 0 <= time.time() - record.get("created_unix_s", 0) < 86400
            or record.get("adapter_sha256") != grounding_grasp_adapter_sha256()
            or set(record.get("checks", {})) != expected_checks
            or not all(v is True for v in record["checks"].values())
            or set(record.get("services", {})) != set(NAMES)):
        raise ValueError("service report is stale, changed or from another adapter")
    children = {}
    for name in NAMES:
        item = record["services"][name]
        if item["endpoint"] != endpoints()[name]:
            raise ValueError("configured service endpoint differs from readiness report")
        children[name] = {"schema": "ora." + ("segmentation" if name == "segmentation" else "grasp-proposal") + "-service-binding.v1",
            "tool_surface": surface, "endpoint": item["endpoint"],
            **{key: item["health"][key] for key in FIELDS},
            "qualification": {"status": "passed", "sha256": digest(record),
                              "created_unix_s": record["created_unix_s"]}}
    return {"schema": "ora.model-service-set-binding.v2",
            "capability_scope": capabilities, "services": children}
