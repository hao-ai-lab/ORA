"""Transport-neutral contract for one trusted external-world episode."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any, Mapping, Protocol, cast, runtime_checkable


ENVIRONMENT_SESSION_SCHEMA = "ora.environment-session.v1"
ENVIRONMENT_STATUS_SCHEMA = "ora.environment-status.v1"
_STATUS_FIELDS = {
    "schema", "phase", "terminal", "reason", "source",
    "policy_visibility", "signals",
}
_STABLE_STATUS_VALUE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")


class EnvironmentContractError(RuntimeError):
    """An environment backend does not implement the required session surface."""


def validate_environment_status(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one control-only lifecycle/recoverability projection.

    The status envelope is deliberately narrow.  Backends may use privileged
    state internally, but they may emit only typed codes, never object poses,
    contacts, task source or success-predicate details.
    """

    status = dict(value)
    if set(status) != _STATUS_FIELDS:
        raise EnvironmentContractError("environment status has an invalid projection")
    if status.get("schema") != ENVIRONMENT_STATUS_SCHEMA:
        raise EnvironmentContractError("environment status schema is invalid")
    terminal = status.get("terminal")
    phase = status.get("phase")
    reason = status.get("reason")
    if not isinstance(terminal, bool):
        raise EnvironmentContractError("environment status terminal must be boolean")
    if phase not in {"running", "terminal"} or terminal != (phase == "terminal"):
        raise EnvironmentContractError("environment status phase is inconsistent")
    if terminal:
        if not isinstance(reason, str) or not _STABLE_STATUS_VALUE.fullmatch(reason):
            raise EnvironmentContractError("terminal environment status requires a reason")
    elif reason is not None:
        raise EnvironmentContractError("running environment status reason must be null")
    if status.get("policy_visibility") != "control_only":
        raise EnvironmentContractError("environment status must remain control-only")
    if (
        not isinstance(status.get("source"), str)
        or not _STABLE_STATUS_VALUE.fullmatch(status["source"])
    ):
        raise EnvironmentContractError("environment status source must be non-empty")
    signals = status.get("signals")
    if not isinstance(signals, list):
        raise EnvironmentContractError("environment status signals must be a list")
    for signal in signals:
        if not isinstance(signal, Mapping) or set(signal) != {
            "code", "severity", "source", "policy_visibility",
        }:
            raise EnvironmentContractError("environment signal has an invalid projection")
        if (
            not isinstance(signal.get("code"), str)
            or not _STABLE_STATUS_VALUE.fullmatch(signal["code"])
            or signal.get("severity") not in {"warning", "terminal"}
            or not isinstance(signal.get("source"), str)
            or not _STABLE_STATUS_VALUE.fullmatch(signal["source"])
            or signal.get("policy_visibility") != "control_only"
        ):
            raise EnvironmentContractError("environment signal fields are invalid")
    return status


@runtime_checkable
class EnvironmentSession(Protocol):
    """One exclusive, trusted writer for one external-world episode.

    Implementations may be an in-process simulator, an RPC client, or a real
    robot.  The Harness only depends on this lifecycle and never receives the
    backend's privileged state, task implementation, or success predicate.

    ``observe`` publishes immutable, privilege-clean artifacts beneath the
    supplied destination. ``execute`` applies one validated action and returns
    a durable receipt. Only the environment supplies the official verdict.
    """

    def start(self) -> Mapping[str, Any]:
        """Acquire the environment and reset/start the episode."""

    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        """Publish a fresh public observation and return display image paths."""

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        """Apply one action bound to the current observation and persist its receipt."""

    def status(self) -> Mapping[str, Any]:
        """Return a typed control-only status without privileged state details."""

    @property
    def is_terminal(self) -> bool:
        """Whether the trusted environment boundary has ended the episode."""

    def abort(self, reason: str) -> None:
        """Stop further state changes because of a trusted control-plane reason."""

    def persist_official_verdict(self) -> dict[str, Any]:
        """Persist and return the backend's official outcome record."""

    def finalize_operator_video(self) -> dict[str, Any] | None:
        """Finalize optional operator-only visual evidence."""

    def close(self) -> None:
        """Release all simulator, transport, sensor, or robot resources."""


@runtime_checkable
class RestartableEnvironmentSession(EnvironmentSession, Protocol):
    """Opt-in extension for multiple independent rounds in one live session."""

    def restart_round(
        self, round_index: int, *, seed: int | None = None,
    ) -> Mapping[str, Any]:
        """Reset a later round, optionally rebinding its public experiment seed."""


def validate_environment_session(session: object) -> EnvironmentSession:
    """Fail fast without evaluating lifecycle properties on an idle backend."""

    missing = []
    for name in (
        "start",
        "observe",
        "execute",
        "status",
        "is_terminal",
        "abort",
        "persist_official_verdict",
        "finalize_operator_video",
        "close",
    ):
        try:
            inspect.getattr_static(session, name)
        except AttributeError:
            missing.append(name)
    if missing:
        detail = ", ".join(missing) if missing else "incompatible attributes"
        raise EnvironmentContractError(
            f"environment backend does not satisfy {ENVIRONMENT_SESSION_SCHEMA}: {detail}"
        )
    return cast(EnvironmentSession, session)


def validate_restartable_environment_session(
    session: EnvironmentSession,
) -> RestartableEnvironmentSession:
    try:
        inspect.getattr_static(session, "restart_round")
    except AttributeError as exc:
        raise EnvironmentContractError(
            "environment backend does not support persistent multi-round reset"
        ) from exc
    return cast(RestartableEnvironmentSession, session)
