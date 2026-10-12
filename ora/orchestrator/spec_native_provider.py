"""native_api Provider fields of a RunSpec: protocol, recovery and endpoint.

``resolve_run_spec`` delegates here for the Provider half of a native_api
policy; context and the ToolSurface allowlist stay in ``spec.py``. The
two phases keep the order and error text of the inline block they replace, and
``spec.py`` still checks the tool allowlist between them, so a RunSpec with
several invalid fields reports the same first error as before. Both functions
set defaults on, and normalize, the caller's own copy of ``policy`` in place,
exactly as that block did.
"""

from __future__ import annotations

import re
from typing import Any, Mapping
from urllib.parse import urlsplit

from ora.orchestrator.spec_values import RunSpecError

#: The native_api continuation-replay selectors this tree accepts.
#: visible_only_v1 composes and evaluates only at the commits that composed it.
CONTINUATION_REPLAYS = frozenset({"exact_v1"})


def resolve_native_provider_protocol(
    policy: dict[str, Any], runtime_spec: Any,
) -> None:
    """Validate the wire protocol, auth, replay and recovery selectors."""
    if policy["provider"] not in {"openai_compatible", "anthropic_messages"}:
        raise RunSpecError(
            "native_api policy.provider must be openai_compatible or anthropic_messages"
        )
    auth_scheme = policy.get("auth_scheme")
    if auth_scheme is not None and auth_scheme not in {"bearer", "x_api_key"}:
        raise RunSpecError(
            "native_api policy.auth_scheme must be bearer or x_api_key"
        )
    if policy["provider"] == "openai_compatible" and auth_scheme not in {
        None, "bearer",
    }:
        raise RunSpecError(
            "openai_compatible native_api policies require bearer auth"
        )
    output_token_parameter = policy.get("output_token_parameter")
    if output_token_parameter is not None and (
        output_token_parameter not in {"max_tokens", "max_completion_tokens"}
        or policy["provider"] != "openai_compatible"
    ):
        raise RunSpecError(
            "native_api policy.output_token_parameter must be max_tokens or "
            "max_completion_tokens on an openai_compatible provider"
        )
    tool_strictness = policy.get("tool_strictness")
    if tool_strictness is not None and (
        tool_strictness != "explicit_false" or policy["provider"] != "openai_compatible"
    ):
        raise RunSpecError(
            "native_api policy.tool_strictness must be explicit_false on an "
            "openai_compatible provider"
        )
    if policy.get("reasoning_field") not in {None, "reasoning_content", "reasoning"} or (
        policy.get("reasoning_field") is not None and policy["provider"] != "openai_compatible"
    ):
        raise RunSpecError(
            "native_api policy.reasoning_field must be reasoning_content or reasoning "
            "on an openai_compatible provider"
        )
    continuation_replay = policy.setdefault(
        "provider_continuation_replay", "exact_v1",
    )
    # RoboDojo never passes through baseline expansion, so an omitted
    # selector left it on legacy retries; it inherits the maintained one.
    if isinstance(runtime_spec, Mapping) and runtime_spec.get("kind") == "robodojo":
        policy.setdefault("provider_recovery", "same_credential_deadline_v1")
    recovery = policy.get("provider_recovery", "legacy_v1")
    if recovery not in {"legacy_v1", "same_credential_deadline_v1"}:
        raise RunSpecError("unsupported native_api policy.provider_recovery")
    window = policy.get("provider_recovery_window_s")
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, (int, float))
        or not 30 <= window <= 1800 or recovery != "same_credential_deadline_v1"
    ):
        raise RunSpecError(
            "native_api policy.provider_recovery_window_s must be 30..1800 seconds "
            "with provider_recovery same_credential_deadline_v1"
        )
    if continuation_replay not in CONTINUATION_REPLAYS:
        raise RunSpecError(
            "native_api policy.provider_continuation_replay must be exact_v1"
        )


def resolve_native_provider_endpoint(policy: dict[str, Any]) -> None:
    """Validate credential names, the endpoint and its transport-bound options."""
    for field, default in (
        ("base_url_env", "NATIVE_MODEL_BASE_URL"),
        ("api_key_env", "NATIVE_MODEL_API_KEY"),
    ):
        policy.setdefault(field, default)
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", str(policy[field])):
            raise RunSpecError(f"policy.{field} must be an uppercase environment name")
    endpoint = policy.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise RunSpecError("native_api policy.endpoint must be an explicit URL")
    endpoint_transport = policy.setdefault("endpoint_transport", "https_v1")
    if endpoint_transport not in {"https_v1", "loopback_http_v1"}:
        raise RunSpecError("unsupported native_api policy.endpoint_transport")
    if "gateway_client_binding" in policy and not isinstance(policy["gateway_client_binding"], bool):
        raise RunSpecError("policy.gateway_client_binding must be boolean")
    if policy.get("gateway_client_binding") is True and endpoint_transport != "loopback_http_v1":
        raise RunSpecError("gateway client binding requires loopback_http_v1")
    parsed_endpoint = urlsplit(endpoint)
    common_endpoint_invalid = (
        not parsed_endpoint.netloc
        or parsed_endpoint.username is not None
        or parsed_endpoint.password is not None
        or parsed_endpoint.query
        or parsed_endpoint.fragment
    )
    if common_endpoint_invalid:
        if endpoint_transport == "https_v1":
            raise RunSpecError(
                "native_api policy.endpoint must be HTTPS without credentials, "
                "query or fragment"
            )
        raise RunSpecError(
            "native_api loopback endpoint must not contain credentials, query or fragment"
        )
    if endpoint_transport == "https_v1" and parsed_endpoint.scheme != "https":
        raise RunSpecError(
            "native_api https_v1 endpoint must use HTTPS"
        )
    if endpoint_transport == "loopback_http_v1" and not (
        parsed_endpoint.scheme == "http"
        and parsed_endpoint.hostname in {"127.0.0.1", "localhost", "::1"}
    ):
        raise RunSpecError(
            "native_api loopback_http_v1 endpoint must be HTTP loopback"
        )
    policy["endpoint"] = endpoint.rstrip("/")
    session_affinity = policy.get("provider_session_affinity")
    if session_affinity not in {None, "x_session_id_run_v1"}:
        raise RunSpecError("unsupported policy.provider_session_affinity")
    # Both wire protocols send the same X-Session-ID (a non-secret run identity)
    # to a loopback gateway or over HTTPS: a self-hosted DeepSeek router keys
    # replica affinity by it, and without it every episode of a task shared one
    # replica (EXP12A g003).
    if session_affinity is not None and endpoint_transport not in {"loopback_http_v1", "https_v1"}:
        raise RunSpecError(
            "provider session affinity is admitted only for a loopback gateway or an HTTPS endpoint"
        )
    proxy_env = policy.setdefault("proxy_env", [])
    if (
        not isinstance(proxy_env, list)
        or len(proxy_env) != len(set(proxy_env))
        or any(
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            for name in proxy_env
        )
    ):
        raise RunSpecError("policy.proxy_env must contain unique environment names")
    provider_profile = policy.get("provider_profile")
    if provider_profile is not None and (
        not isinstance(provider_profile, str) or not provider_profile
    ):
        raise RunSpecError("policy.provider_profile must be non-empty when set")
    qualification_schema = policy.get("provider_qualification_schema")
    if qualification_schema is not None and qualification_schema != (
        "ora.native-provider-probe.v1"
    ):
        raise RunSpecError("unsupported policy.provider_qualification_schema")
    qualification_version = policy.get("provider_qualification_version")
    if qualification_version is not None and qualification_version != 1:
        raise RunSpecError("unsupported policy.provider_qualification_version")
    if "model_env" in policy and not re.fullmatch(
        r"[A-Z_][A-Z0-9_]*", str(policy["model_env"])
    ):
        raise RunSpecError("policy.model_env must be an uppercase environment name")
