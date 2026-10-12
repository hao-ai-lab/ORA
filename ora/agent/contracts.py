"""Transport-neutral contracts for one policy lifecycle.

The trusted Orchestrator supplies an immutable public turn and receives a
typed outcome.  Provider-specific sessions may retain their own context, but
they never receive privileged environment state or execute world mutations.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Protocol, Sequence, TypeAlias


class PolicySessionError(RuntimeError):
    """A policy session cannot make or persist a valid decision."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class PolicySessionCapabilities:
    """Backend facts used for orchestration, never policy coaching."""

    runtime_kind: Literal["scripted", "managed_coder", "native"]
    context_owner: Literal["none", "managed", "harness"]
    tool_transport: Literal["in_process", "mcp", "direct"]
    request_visibility: Literal["none", "project_exact", "wire_exact"]
    supports_images: bool
    supports_interrupt: bool


@dataclass(frozen=True)
class PolicyTurnInput:
    """Immutable, privilege-clean input for one observation-scoped policy turn."""

    task_id: str
    task_instruction: str
    context: Mapping[str, Any]
    image_path: Path | None
    observation_id: str | None
    observation_step: int | None
    budget: Mapping[str, Any] = field(default_factory=dict)
    interaction_phase: str = "observation_ready"
    fresh_observation_required: bool = False
    allow_reminder: bool = True
    terminal_lifecycle: bool = False
    task_sha256: str = ""
    image_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        task = self.task_instruction.strip()
        if not self.task_id or not task:
            raise PolicySessionError("policy turn requires task identity and exact text")
        if self.observation_step is not None and self.observation_step < 0:
            raise PolicySessionError("policy observation step must be non-negative")
        object.__setattr__(self, "task_instruction", task)
        object.__setattr__(
            self,
            "task_sha256",
            self.task_sha256 or hashlib.sha256(task.encode("utf-8")).hexdigest(),
        )
        object.__setattr__(self, "context", _freeze(self.context))
        object.__setattr__(self, "budget", _freeze(self.budget))
        if self.image_path is not None:
            object.__setattr__(self, "image_path", Path(self.image_path))
        object.__setattr__(self, "image_paths", tuple(Path(path) for path in self.image_paths))

    def runtime_kwargs(self) -> dict[str, Any]:
        """Project the typed turn into the runtime-owned invocation shape."""
        value = {
            "task_instruction": self.task_instruction,
            "context": _thaw(self.context),
            "image_path": self.image_path,
            "allow_reminder": self.allow_reminder,
        }
        if self.terminal_lifecycle:
            value["terminal_lifecycle"] = True
        if self.image_paths:
            value["image_paths"] = self.image_paths
        return value


@dataclass(frozen=True)
class PolicyRuntimeEvent:
    """One trusted outer-loop fact delivered identically to every policy backend."""

    kind: Literal[
        "fresh_observation",
        "environment_receipt",
        "official_verdict",
        "run_termination",
    ]
    payload: Mapping[str, Any]
    observation_id: str | None = None
    observation_step: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))

    def mutable_payload(self) -> dict[str, Any]:
        return _thaw(self.payload)


@dataclass(frozen=True)
class ActionProposal:
    action: Mapping[str, Any] | None
    model_calls: int
    details: Mapping[str, Any] = field(default_factory=dict)
    kind: Literal["action"] = "action"

    def as_record(self) -> dict[str, Any]:
        return _thaw(self.details)


@dataclass(frozen=True)
class NoAction:
    reason: str
    model_calls: int
    details: Mapping[str, Any] = field(default_factory=dict)
    kind: Literal["no_action"] = "no_action"

    def as_record(self) -> dict[str, Any]:
        return _thaw(self.details)


@dataclass(frozen=True)
class LifecycleRequest:
    reason: str
    model_calls: int
    details: Mapping[str, Any] = field(default_factory=dict)
    kind: Literal["lifecycle"] = "lifecycle"

    def as_record(self) -> dict[str, Any]:
        return _thaw(self.details)


PolicyOutcome: TypeAlias = ActionProposal | NoAction | LifecycleRequest


def policy_outcome_from_mapping(value: Mapping[str, Any]) -> PolicyOutcome:
    raw = copy.deepcopy(dict(value))
    attempts = raw.get("attempts", 1)
    model_calls = int(attempts) if isinstance(attempts, int) and attempts >= 0 else 1
    if raw.get("submitted") is True:
        action = raw.get("action")
        return ActionProposal(
            action=copy.deepcopy(dict(action)) if isinstance(action, Mapping) else None,
            model_calls=model_calls,
            details=_freeze(raw),
        )
    reason = str(raw.get("termination", "policy_no_action"))
    if reason == "policy_finished_episode":
        return LifecycleRequest(reason=reason, model_calls=model_calls, details=_freeze(raw))
    return NoAction(reason=reason, model_calls=model_calls, details=_freeze(raw))


class PolicySession(Protocol):
    capabilities: PolicySessionCapabilities

    def start(self) -> None: ...
    def act(self, turn: PolicyTurnInput) -> PolicyOutcome: ...
    def accept_runtime_event(self, event: PolicyRuntimeEvent) -> None: ...
    def interrupt(self, reason: str = "operator_interrupt") -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class SummaryResult:
    """Explicit result from an independently configured summary call."""

    content: str
    source_refs: Sequence[Mapping[str, Any]]
    source_set_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ModelProvider(Protocol):
    def complete(
        self, *, request: Mapping[str, Any], model_call_id: str,
    ) -> Mapping[str, Any]: ...
    def close(self) -> None: ...


class ContextProjectionPipeline(Protocol):
    def project(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        observation_step: int,
    ) -> Mapping[str, Any]: ...


class TrajectorySink(Protocol):
    def append(self, event_type: str, **fields: Any) -> Mapping[str, Any]: ...


class ToolGateway(Protocol):
    def definitions(self) -> Sequence[Mapping[str, Any]]: ...
    def execute(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...


class SummaryProvider(Protocol):
    def summarize(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        source_refs: Sequence[Mapping[str, Any]],
    ) -> SummaryResult: ...
