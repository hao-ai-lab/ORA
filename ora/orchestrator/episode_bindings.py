"""Preflight binding consumption and environment-owned round comparability."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .policy_finish import admitted_abort_round

class SupervisorError(RuntimeError):
    """The trusted causal episode loop cannot continue."""


def _qualified_runtime_binding(
    run_root: Path, runtime_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load the exact external control binding accepted by preflight."""

    kind = runtime_config.get("kind")
    if kind not in {"libero_pro", "robodojo"}:
        return None
    if kind == "robodojo":
        try:
            record = json.loads((run_root / "preflight.json").read_text(encoding="utf-8"))
            matching = [
                check for check in record["checks"]
                if check.get("check") == "robodojo_infrastructure_qualification"
            ]
            if len(matching) != 1 or matching[0].get("passed") is not True:
                raise ValueError("missing qualification")
            binding = matching[0]["detail"]
            # Either admitted contract: the combined single record, or the
            # card/task layered view, whose merged projection carries the
            # same source/layout/configuration/device/instruction fields the
            # worker rechecks against its own live readback at reset.
            from ora.environment.benchmarks.robodojo import (
                QUALIFICATION_SCHEMA,
            )
            from .robodojo_qualification import LAYERED_SCHEMA
            from .robodojo_revalidation import SCHEMA as REVALIDATION_SCHEMA

            if binding.get("schema") not in {QUALIFICATION_SCHEMA, LAYERED_SCHEMA, REVALIDATION_SCHEMA}:
                raise ValueError("invalid qualification schema")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise SupervisorError("RoboDojo run lacks its passing infrastructure binding") from exc
        return dict(binding)
    path = run_root / "preflight.json"
    try:
        preflight = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisorError(
            "LIBERO run lacks its passing controller preflight binding"
        ) from exc
    checks = preflight.get("checks") if isinstance(preflight, Mapping) else None
    matches = [
        item for item in checks if isinstance(item, Mapping)
        and item.get("check") == "libero_controller_action_contract"
    ] if isinstance(checks, list) else []
    detail = matches[0].get("detail") if len(matches) == 1 else None
    if (
        len(matches) != 1
        or matches[0].get("passed") is not True
        or not isinstance(detail, Mapping)
        or detail.get("schema")
        != "ora.libero-controller-qualification.v1"
    ):
        raise SupervisorError(
            "LIBERO controller preflight binding is absent or invalid"
        )
    return dict(detail)


ROUND_COMPARABILITY_FIELDS = (
    "round_end_basis", "protocol_comparable", "upstream_ending_condition",
)




def _round_comparability(verdict: Mapping[str, Any] | None) -> dict[str, Any]:
    """Mirror the environment's own round-ending classification, or nulls."""

    source = verdict if isinstance(verdict, Mapping) else {}
    return {field: source.get(field) for field in ROUND_COMPARABILITY_FIELDS}


def robodojo_official_complete(
    official: Any, round_results: Sequence[Mapping[str, Any]], *,
    multi_round: bool, round_limit: int,
) -> bool:
    """Whether a RoboDojo run actually carries its official Boolean outcome.

    A round the Supervisor stopped on its own budget or on a policy lifecycle
    request now carries the standing official predicate — false, with the
    verdict disclosing that the simulator never declared a terminal
    transition — exactly as LIBERO reports a truncated episode. Only an
    infrastructure fault, or a backend that produced no verdict at all, keeps
    the Boolean null and the run incomplete. A persistent session keeps its
    aggregate verdict null by contract, so the rule is applied at the round
    boundary; applying it to the aggregate would mark every completed K-round
    RoboDojo session incomplete. A round an abort admitted on recorded
    infrastructure evidence ended (``policy_finish.admitted_abort_round``) has
    no official Boolean by design: it is an invalid sample that ends only
    itself, so it does not make the session incomplete.
    """

    if not multi_round:
        return isinstance(official, bool)
    return len(round_results) == round_limit and all(
        isinstance(record.get("official_success"), bool) or admitted_abort_round(record)
        for record in round_results
    )
