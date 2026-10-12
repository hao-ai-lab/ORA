"""Stable fault taxonomy across provider, tool, environment and evidence layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping


FaultLayer = Literal[
    "provider", "context", "tool", "protocol", "environment", "evidence",
    "operator", "orchestrator",
]
WorldStateCertainty = Literal["unchanged", "changed", "unknown"]
RecoveryPoint = Literal["same_turn", "next_observation", "operator", "never"]


@dataclass(frozen=True)
class HarnessFault:
    code: str
    layer: FaultLayer
    retryable: bool
    world_state_certainty: WorldStateCertainty
    recoverable_at: RecoveryPoint
    requires_fresh_observation: bool
    fatal: bool
    cause_event_id: str | None = None
    message: str | None = None
    schema: str = "ora.harness-fault.v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "code": self.code,
            "layer": self.layer,
            "retryable": self.retryable,
            "world_state_certainty": self.world_state_certainty,
            "recoverable_at": self.recoverable_at,
            "requires_fresh_observation": self.requires_fresh_observation,
            "fatal": self.fatal,
            "cause_event_id": self.cause_event_id,
            "message": self.message,
        }


class HarnessFaultError(RuntimeError):
    def __init__(self, fault: HarnessFault) -> None:
        super().__init__(fault.message or fault.code)
        self.fault = fault


def classify_harness_fault(
    exc: BaseException,
    *,
    phase: str,
    layer: FaultLayer | None = None,
    cause_event_id: str | None = None,
) -> HarnessFault:
    """Classify without authorizing an implicit retry or mutation replay."""

    if isinstance(exc, HarnessFaultError):
        return exc.fault
    if layer is None:
        origin = f"{exc.__class__.__module__}.{exc.__class__.__name__}".casefold()
        if "provider" in origin:
            layer = "provider"
        elif any(token in origin for token in ("context", "composition", "prompt")):
            layer = "context"
        elif any(token in origin for token in ("tool", "mcp", "openeta")):
            layer = "tool"
        elif "environment" in origin or "rpc" in origin:
            layer = "environment"
        elif any(token in origin for token in ("trajectory", "evidence", "state")):
            layer = "evidence"
        elif "interactionstate" in origin:
            layer = "protocol"
        elif isinstance(exc, (KeyboardInterrupt, InterruptedError)):
            layer = "operator"
        else:
            layer = "orchestrator"
    unknown = phase in {"executing", "world_state_unknown"}
    confirmed_change = phase == "fresh_observation_required"
    certainty: WorldStateCertainty = (
        "unknown" if unknown else "changed" if confirmed_change else "unchanged"
    )
    requires_fresh = unknown or confirmed_change
    return HarnessFault(
        code=(
            "mutation_outcome_unknown" if unknown
            else "post_mutation_observation_failure" if confirmed_change
            else f"{layer}_failure"
        ),
        layer=layer,
        retryable=False,
        world_state_certainty=certainty,
        recoverable_at=(
            "operator" if unknown else "next_observation" if confirmed_change else "never"
        ),
        requires_fresh_observation=requires_fresh,
        fatal=True,
        cause_event_id=cause_event_id,
        message=str(exc),
    )
