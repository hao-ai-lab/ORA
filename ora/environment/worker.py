"""Isolated environment process serving one EnvironmentSession over local RPC."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

from ora.contracts.refusal import EnvironmentRefusal
from .rpc import (
    KIT_BACKENDS, MAX_MESSAGE_BYTES, PID_COLLISION_FILE, PID_SEGMENT_COLLISION_EXIT, RPC_SCHEMA,
)

#: carb names its string-interning segment and semaphore by this PID. Only
#: KIT_BACKENDS workers are checked, so every other worker starts as before.
#: Read at call time, so a test worker can point it at a temporary directory.
SHM_ROOT = Path("/dev/shm")


class WorkerProtocolError(RuntimeError):
    """A local client request violated the sealed environment protocol."""


def kit_pid_collision(config: Mapping[str, Any], *, pid: int, uid: int,
                      root: Path | None = None) -> str | None:
    """Name a carb segment for this PID that Kit could not create, if any.

    Another user's stale segment cannot be removed and made Kit abort at start
    (run 683e36fb73af); the caller exits for a relaunch under a new PID. This
    user's segment for this PID is stale by construction (the PID is ours now)
    and is removed.
    """

    if config.get("backend") not in KIT_BACKENDS:
        return None
    root = SHM_ROOT if root is None else root
    for name in (f"carb-RStringInternals-{pid}", f"sem.carb-RStringInternals-{pid}"):
        path = root / name
        try:
            owner = os.lstat(path).st_uid
        except FileNotFoundError:
            continue
        except OSError:
            return None
        if owner != uid:
            return name
        try:
            path.unlink()
        except FileNotFoundError:
            continue
        except OSError:
            return name
    return None


def _error_envelope(error: Exception) -> dict[str, Any]:
    if isinstance(error, EnvironmentRefusal):
        return error.as_dict()
    return {"kind": "fault", "type": type(error).__name__, "message": str(error)}


def _load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != RPC_SCHEMA:
        raise WorkerProtocolError("invalid worker configuration")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _build_session(config: Mapping[str, Any]) -> Any:
    backend = config.get("backend")
    output_root = Path(str(config["output_root"])).resolve()
    shared_root = Path(str(config["shared_root"])).resolve()
    if not _inside(output_root, shared_root):
        raise WorkerProtocolError("environment output root escapes the run workspace")
    if backend == "synthetic":
        from .synthetic import SyntheticSession

        return SyntheticSession(output_root, str(config["task"]), int(config["seed"]))
    if backend == "robodojo":
        from ora.environment.backends.robodojo.session import RobodojoSession

        return RobodojoSession(config=config, output_root=output_root)
    if backend == "robotwin":
        from ora.environment.backends.robotwin.session import RoboTwinLiveSession

        return RoboTwinLiveSession(
            robotwin_root=Path(str(config["robotwin_root"])),
            output_root=output_root,
            execution_capture=config.get("execution_capture"),
            task=str(config["task"]),
            actual_seed=int(config["seed"]),
            episode=int(config.get("episode", 0)),
            task_config=str(config["task_config"]),
            required_revision=str(config["required_revision"]),
            workspace_bounds_m=config["workspace_bounds_m"],
            max_sim_steps=int(config["max_sim_steps"]),
            operator_world_video=bool(config.get("operator_world_video", True)),
            operator_execution_video=bool(
                config.get("operator_execution_video", False)
            ),
            environment_signal_policy=str(
                config.get("environment_signal_policy", "lifecycle_only")
            ),
            action_codec=str(config.get("action_codec") or "runtime_action_v3"),
            control_profile=config.get("control_profile"),
            public_robot_geometry=config.get("public_robot_geometry"),
            # The reach check's other-arm screen (0059 M6); the session
            # validates the value, so it passes through uncoerced.
            reach_check_inter_arm_screen=config.get("reach_check_inter_arm_screen", False),
        )
    if backend == "libero_pro":
        from ora.environment.backends.libero_pro.session import LiberoProSession

        return LiberoProSession(
            environment_driver=str(config.get("environment_driver", "direct_libero_v1")),
            libero_pro_root=Path(str(config["libero_pro_root"])),
            data_root=Path(str(config["data_root"])),
            output_root=output_root,
            execution_capture=config.get("execution_capture"),
            task=str(config["task"]),
            seed=int(config["seed"]),
            suite=str(config["suite"]),
            task_id=int(config["task_id"]),
            robustness_case=(
                str(config["robustness_case"])
                if config.get("robustness_case") is not None else None
            ),
            expected_task_language=str(config["expected_task_language"]),
            required_libero_pro_revision=str(config["required_libero_pro_revision"]),
            required_data_revision=str(config["required_data_revision"]),
            workspace_bounds_m=config["workspace_bounds_m"],
            workspace_profile=config.get("workspace_profile"),
            qualified_controller_binding=config.get(
                "qualified_controller_binding"
            ),
            max_sim_steps=int(config["max_sim_steps"]),
            servo_max_steps=int(config.get("servo_max_steps", 80)),
            action_codec=str(config.get("action_codec") or "runtime_action_v3"),
            motion_planner_config=config.get("motion_planner"),
            motion_profile_config=config.get("motion_profile"),
            public_robot_geometry=config.get("public_robot_geometry"),
        )
    raise WorkerProtocolError(f"unsupported environment backend: {backend!r}")


def _dispatch(session: Any, config: Mapping[str, Any], method: str, params: Mapping[str, Any]) -> tuple[Any, bool]:
    if method == "ping":
        return {"ready": True}, False
    if method == "start":
        return session.start(), False
    if method == "restart_round":
        round_index = params.get("round_index")
        if (
            isinstance(round_index, bool)
            or not isinstance(round_index, int)
            or round_index <= 0
        ):
            raise WorkerProtocolError("restart_round.round_index must be positive")
        restart = getattr(session, "restart_round", None)
        if not callable(restart):
            raise WorkerProtocolError("environment backend does not support round reset")
        seed = params.get("seed")
        if (
            seed is not None
            and (
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or seed < 0
            )
        ):
            raise WorkerProtocolError(
                "restart_round.seed must be a non-negative integer"
            )
        return restart(round_index, seed=seed), False
    if method == "observe":
        destination = Path(str(params.get("destination", ""))).resolve()
        shared_root = Path(str(config["shared_root"])).resolve()
        if not _inside(destination, shared_root):
            raise WorkerProtocolError("observation destination escapes the run workspace")
        observation, images = session.observe(destination)
        if config.get("backend") == "libero_pro":
            from ora.environment.backends.libero_pro.render_guard import observation_corruption
            from ora.environment.backends.libero_pro.session import LiberoProError

            corruption = observation_corruption(images)
            if corruption is not None:
                raise LiberoProError(f"offscreen render corrupted: {corruption}")
        return {
            "observation": observation,
            "images": [str(path.resolve()) for path in images],
        }, False
    if method == "begin_action_sequence":
        from ora.contracts import validate_runtime_action
        action = params.get("action")
        if not isinstance(action, Mapping):
            raise WorkerProtocolError("begin_action_sequence.action must be an object")
        action = validate_runtime_action(action)
        if action["schema"] not in {"ora-skill-sequence-action.v1", "ora-action.v4"}:
            raise WorkerProtocolError("action sequence requires a supported parent")
        begin = getattr(session, "begin_action_sequence", None)
        if callable(begin):
            begin(action)
        return {"supported": callable(begin)}, False
    if method == "end_action_sequence":
        action_id = params.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise WorkerProtocolError("end_action_sequence requires action_id")
        end = getattr(session, "end_action_sequence", None)
        if callable(end):
            end(action_id)
        return {"supported": callable(end)}, False
    if method == "execute":
        action = params.get("action")
        if not isinstance(action, Mapping):
            raise WorkerProtocolError("execute.action must be an object")
        return session.execute(action), False
    if method == "align_plan":
        request = params.get("request")
        align = getattr(session, "align_plan", None)
        if not isinstance(request, Mapping) or not callable(align):
            raise WorkerProtocolError("environment backend does not support alignment planning")
        # False is should_close, not a mutation flag: this call is read-only and
        # the worker must stay alive for the next one.
        return align(request), False
    if method == "probe_motion":
        request = params.get("request")
        probe = getattr(session, "probe_motion", None)
        if not isinstance(request, Mapping) or not callable(probe):
            raise WorkerProtocolError("environment backend does not support motion probing")
        return probe(request), False
    if method == "measure_jaw_state":
        # Scripted-qualification only. It reads private finger-body poses, so
        # it stays off every policy route: no ToolSurface names it and the
        # coordinator never calls it.
        measure = getattr(session, "measure_jaw_state", None)
        if params or not callable(measure):
            raise WorkerProtocolError(
                "environment backend does not support jaw-state measurement"
            )
        return measure(), False
    if method == "status":
        return session.status(), False
    if method == "is_terminal":
        return bool(session.is_terminal), False
    if method == "abort":
        reason = params.get("reason")
        if not isinstance(reason, str) or not reason:
            raise WorkerProtocolError("abort.reason must be non-empty")
        session.abort(reason)
        return None, False
    if method == "persist_official_verdict":
        return session.persist_official_verdict(), False
    if method == "finalize_operator_video":
        return session.finalize_operator_video(), False
    if method == "close":
        session.close()
        return None, True
    raise WorkerProtocolError(f"unsupported environment method: {method!r}")


def serve(socket_path: Path, config_path: Path) -> int:
    config = _load_config(config_path)
    session_id = config.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise WorkerProtocolError("missing environment session id")
    collision = kit_pid_collision(config, pid=os.getpid(), uid=os.getuid())
    if collision is not None:
        print(f"worker PID {os.getpid()} collides with {SHM_ROOT / collision}; "
              "exiting for a relaunch", file=sys.stderr, flush=True)
        try:  # the launcher records the segment name with the relaunch
            (socket_path.parent / PID_COLLISION_FILE).write_text(json.dumps(
                {"worker_pid": os.getpid(), "segment": collision}), encoding="utf-8")
        except OSError:
            pass
        return PID_SEGMENT_COLLISION_EXIT
    session = _build_session(config)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    previous_umask = os.umask(0o077)
    try:
        server.bind(str(socket_path))
    finally:
        os.umask(previous_umask)
    server.listen(1)
    should_close = False
    try:
        connection, _ = server.accept()
        with connection:
            reader = connection.makefile("r", encoding="utf-8", newline="\n")
            writer = connection.makefile("w", encoding="utf-8", newline="\n")
            with reader, writer:
                while not should_close:
                    line = reader.readline(MAX_MESSAGE_BYTES + 1)
                    if not line:
                        break
                    response: dict[str, Any]
                    request_id: Any = None
                    try:
                        if len(line.encode()) > MAX_MESSAGE_BYTES:
                            raise WorkerProtocolError("request exceeds size limit")
                        request = json.loads(line)
                        request_id = request.get("request_id")
                        if request.get("schema") != RPC_SCHEMA:
                            raise WorkerProtocolError("RPC schema mismatch")
                        if request.get("session_id") != session_id:
                            raise WorkerProtocolError("environment session identity mismatch")
                        method = request.get("method")
                        params = request.get("params")
                        if not isinstance(method, str) or not isinstance(params, Mapping):
                            raise WorkerProtocolError("invalid RPC request shape")
                        result, should_close = _dispatch(session, config, method, params)
                        response = {
                            "schema": RPC_SCHEMA,
                            "session_id": session_id,
                            "request_id": request_id,
                            "ok": True,
                            "result": result,
                        }
                    except Exception as exc:
                        if not isinstance(exc, EnvironmentRefusal):
                            traceback.print_exc()
                        response = {
                            "schema": RPC_SCHEMA,
                            "session_id": session_id,
                            "request_id": request_id,
                            "ok": False,
                            "error": _error_envelope(exc),
                        }
                    writer.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
                    writer.flush()
    finally:
        try:
            session.close()
        except Exception:
            traceback.print_exc()
        server.close()
        socket_path.unlink(missing_ok=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one isolated environment session.")
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return serve(args.socket.resolve(), args.config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
