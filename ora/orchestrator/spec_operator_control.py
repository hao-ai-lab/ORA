"""Resolve and validate the optional, explicitly interactive RunPlan binding."""

from __future__ import annotations

from typing import Any, Mapping

from .operator_control import OperatorControlError, normalize_operator_control
from .spec_values import RunSpecError


def resolve_operator_control(spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return normalize_operator_control(
            spec.get("operator_control"), experiment=spec.get("experiment", ""),
            policy_kind=spec.get("policy", {}).get("kind", ""),
        )
    except OperatorControlError as exc:
        raise RunSpecError(str(exc)) from exc


def validate_persisted_operator_control(plan: Mapping[str, Any]) -> None:
    if "operator_control" in plan and resolve_operator_control(plan) != plan["operator_control"]:
        raise RunSpecError("persisted RunPlan operator control is not normalized")
