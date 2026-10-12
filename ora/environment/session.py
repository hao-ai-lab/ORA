"""Descriptor-validating façade around an existing EnvironmentSession."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .base import EnvironmentSession, validate_environment_session

from .definitions import EnvironmentRuntimeDescriptor, resolve_runtime_descriptor


@dataclass
class DescribedEnvironmentSession:
    delegate: EnvironmentSession
    planned_descriptor: EnvironmentRuntimeDescriptor
    runtime_config: Mapping[str, Any]
    action_codec: str | None = None

    def __post_init__(self) -> None:
        self.delegate = validate_environment_session(self.delegate)
        self._descriptor = self.planned_descriptor

    def describe(self) -> Mapping[str, Any]:
        return self._descriptor.as_dict()

    def start(self) -> Mapping[str, Any]:
        reset = self.delegate.start()
        observed = reset.get("public_environment_context")
        if observed is not None:
            actual = resolve_runtime_descriptor(
                self.runtime_config,
                action_codec=self.action_codec,
                observed_public_context=(
                    observed if isinstance(observed, Mapping) else {"invalid": observed}
                ),
            )
            if actual.as_dict() != self.planned_descriptor.as_dict():
                raise RuntimeError("environment runtime descriptor drift")
            self._descriptor = actual
        return reset

    def restart_round(
        self, round_index: int, *, seed: int | None = None,
    ) -> Mapping[str, Any]:
        restart = getattr(self.delegate, "restart_round", None)
        if not callable(restart):
            raise RuntimeError("environment backend does not support round reset")
        reset = restart(round_index, seed=seed)
        observed = reset.get("public_environment_context")
        if observed is not None:
            actual = resolve_runtime_descriptor(
                self.runtime_config,
                action_codec=self.action_codec,
                observed_public_context=(
                    observed if isinstance(observed, Mapping) else {"invalid": observed}
                ),
            )
            if actual.as_dict() != self.planned_descriptor.as_dict():
                raise RuntimeError("environment runtime descriptor drift after reset")
            self._descriptor = actual
        return reset

    def observe(self, destination: Path) -> tuple[dict[str, Any], list[Path]]:
        return self.delegate.observe(destination)

    def begin_action_sequence(self, action: Mapping[str, Any]) -> None:
        begin = getattr(self.delegate, "begin_action_sequence", None)
        if callable(begin):
            begin(action)

    def end_action_sequence(self, action_id: str) -> None:
        end = getattr(self.delegate, "end_action_sequence", None)
        if callable(end):
            end(action_id)

    def execute(self, action: Mapping[str, Any]) -> dict[str, Any]:
        return self.delegate.execute(action)

    def align_plan(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Forward the read-only inverse-kinematics reachability extension."""

        align = getattr(self.delegate, "align_plan", None)
        if not callable(align):
            raise RuntimeError("environment backend does not support alignment planning")
        return dict(align(request))

    def probe_motion(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Forward the opt-in read-only motion-planning extension."""

        probe = getattr(self.delegate, "probe_motion", None)
        if not callable(probe):
            raise RuntimeError("environment backend does not support motion probing")
        return dict(probe(request))

    def status(self) -> Mapping[str, Any]:
        return self.delegate.status()

    @property
    def is_terminal(self) -> bool:
        return self.delegate.is_terminal

    def abort(self, reason: str) -> None:
        self.delegate.abort(reason)

    def persist_official_verdict(self) -> dict[str, Any]:
        return self.delegate.persist_official_verdict()

    def finalize_operator_video(self) -> dict[str, Any] | None:
        return self.delegate.finalize_operator_video()

    def close(self) -> None:
        self.delegate.close()
