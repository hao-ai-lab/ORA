"""Preflight binding of a loopback gateway's client variables.

``scripts/ora preflight|launch`` runs this before the shared
preflight: for a native_api plan whose Provider profile reaches a loopback
gateway (``endpoint_transport: loopback_http_v1``, ``gateway_client_binding: true``), it binds the gateway's
client variables into this process environment
(``agent.providers.gateway_client``), so the provider-marker check and the
launched child see them. The result gains a ``provider_gateway_client`` check
whose detail names the variables, the files and their presence, never a
value; a missing or disagreeing file or variable fails preflight with that
name. Plans on any other transport get no check and no change.

It wraps ``preflight.run_preflight`` rather than editing it, because
``preflight.py`` is a hash-bound launch source.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping, MutableMapping

from ora.agent.providers.gateway_client import GatewayClientError, bind_gateway_client, uses_gateway
from .checks import check as _check

CHECK = "provider_gateway_client"


def gateway_check(plan: Mapping[str, Any], environ: MutableMapping[str, str] | None = None) -> dict[str, Any] | None:
    """Bind the plan's gateway client into ``environ`` (default ``os.environ``); None when not a gateway plan."""
    policy = plan.get("policy") or {}
    if policy.get("kind") != "native_api" or not uses_gateway(policy):
        return None
    try:
        receipt = bind_gateway_client(policy, os.environ if environ is None else environ)
    except (GatewayClientError, OSError) as exc:
        return _check(CHECK, False, {"error": str(exc), "endpoint_transport": policy.get("endpoint_transport")})
    return _check(CHECK, True, receipt)


def gateway_preflight(plan: Mapping[str, Any], run: Callable[[Mapping[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """``run(plan)`` after the gateway binding, with the binding's check first and in ``passed``."""
    check = gateway_check(plan)
    result = run(plan)
    if check is None:
        return result
    return {**result, "passed": bool(result["passed"] and check["passed"]), "checks": [check, *result["checks"]]}
