"""Source-tracked native-provider catalogs and non-persistent wire probes."""

from __future__ import annotations

import base64
import json
import re
import secrets
import struct
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

import yaml

from ora.agent.providers.openai_wire_options import (
    REASONING_FIELDS, TOOL_STRICTNESS_MODES, validate_request_options,
)
from ora.agent.providers.continuation import (
    PROVIDER_CONTINUATION_MESSAGE_KEY,
)


CATALOG_SCHEMA = "ora.native-provider-catalog.v1"
PROBE_SCHEMA = "ora.native-provider-probe.v1"
PROVIDER_PROTOCOLS = {"anthropic_messages", "openai_compatible"}
BUILTIN_PROVIDER_CATALOG = Path(__file__).with_name("builtin-profiles.yaml")
PROFILE_ADMISSIONS = {"registered_only", "text_tool_qualified", "vision_qualified"}
PROFILE_USAGE_SCOPES = {
    "agent_plan_api", "general_api", "supported_coding_tools_only",
}
ENDPOINT_TRANSPORTS = {"https_v1", "loopback_http_v1"}
SESSION_AFFINITY_MODES = {"x_session_id_run_v1"}
#: How an OpenAI-compatible profile names the output-token budget on the wire.
OUTPUT_TOKEN_PARAMETERS = {"max_tokens", "max_completion_tokens"}


def _red_png_data_url(size: int = 256) -> str:
    raw = b"".join(b"\x00" + bytes((255, 0, 0)) * size for _ in range(size))

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class NativeProviderProfileError(RuntimeError):
    """A native provider catalog or selected profile is invalid."""


def load_provider_profiles(path: Path) -> dict[str, dict[str, Any]]:
    """Load and fail closed on one source-tracked provider catalog."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise NativeProviderProfileError(f"cannot load native provider catalog: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != CATALOG_SCHEMA:
        raise NativeProviderProfileError("invalid native provider catalog")
    profiles = value.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise NativeProviderProfileError("native provider catalog must contain profiles")
    required = {
        "protocol", "endpoint", "model", "model_env", "base_url_env",
        "api_key_env", "proxy_env", "reasoning_effort", "context_window_tokens",
        "max_output_tokens", "stream", "supports_images", "qualification_schema",
        "qualification_version",
    }
    output: dict[str, dict[str, Any]] = {}
    for raw_profile_id, raw_profile in profiles.items():
        if (
            not isinstance(raw_profile_id, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", raw_profile_id)
            or not isinstance(raw_profile, Mapping)
        ):
            raise NativeProviderProfileError(
                "native provider profile identifiers and values must be objects"
            )
        profile = dict(raw_profile)
        missing = sorted(required - set(profile))
        optional = {
            "admission", "auth_scheme", "endpoint_transport",
            "max_inline_images_per_request", "output_token_parameter",
            "session_affinity", "source_url", "tool_strictness", "usage_scope",
            "reasoning_field", "recovery_window_s", "gateway_client_binding", "request_options",
        }
        unexpected = sorted(set(profile) - required - optional)
        if missing:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} is missing {missing}"
            )
        if unexpected:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has unexpected fields {unexpected}"
            )
        if profile["protocol"] not in PROVIDER_PROTOCOLS:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has unsupported protocol"
            )
        auth_scheme = profile.get("auth_scheme")
        if auth_scheme is not None and auth_scheme not in {"bearer", "x_api_key"}:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has unsupported auth_scheme"
            )
        if profile["protocol"] == "openai_compatible" and auth_scheme not in {
            None, "bearer",
        }:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has incompatible auth_scheme"
            )
        admission = profile.get("admission")
        if admission is not None and admission not in PROFILE_ADMISSIONS:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid admission"
            )
        usage_scope = profile.get("usage_scope")
        if usage_scope is not None and usage_scope not in PROFILE_USAGE_SCOPES:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid usage_scope"
            )
        source_url = profile.get("source_url")
        if source_url is not None:
            parsed_source = urlsplit(str(source_url))
            if parsed_source.scheme != "https" or not parsed_source.netloc:
                raise NativeProviderProfileError(
                    f"provider profile {raw_profile_id} has invalid source_url"
                )
        if "gateway_client_binding" in profile and not isinstance(profile["gateway_client_binding"], bool):
            raise NativeProviderProfileError(f"provider profile {raw_profile_id} has invalid gateway_client_binding")
        endpoint_transport = profile.get("endpoint_transport", "https_v1")
        if profile.get("gateway_client_binding") is True and endpoint_transport != "loopback_http_v1":
            raise NativeProviderProfileError(f"provider profile {raw_profile_id} gateway binding requires loopback_http_v1")
        if endpoint_transport not in ENDPOINT_TRANSPORTS:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid endpoint_transport"
            )
        endpoint = urlsplit(str(profile["endpoint"]))
        common_endpoint_valid = (
            bool(endpoint.netloc)
            and endpoint.username is None and endpoint.password is None
            and not endpoint.query and not endpoint.fragment
        )
        endpoint_valid = (
            common_endpoint_valid
            and (
                (endpoint_transport == "https_v1" and endpoint.scheme == "https")
                or (
                    endpoint_transport == "loopback_http_v1"
                    and endpoint.scheme == "http"
                    and endpoint.hostname in {"127.0.0.1", "localhost", "::1"}
                )
            )
        )
        if not endpoint_valid:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has an unsafe endpoint"
            )
        if "endpoint_transport" in profile:
            profile["endpoint_transport"] = endpoint_transport
        output_token_parameter = profile.get("output_token_parameter")
        if output_token_parameter is not None and (
            output_token_parameter not in OUTPUT_TOKEN_PARAMETERS
            or profile["protocol"] != "openai_compatible"
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid output_token_parameter"
            )
        tool_strictness = profile.get("tool_strictness")
        if tool_strictness is not None and (
            tool_strictness not in TOOL_STRICTNESS_MODES
            or profile["protocol"] != "openai_compatible"
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid tool_strictness"
            )
        reasoning_field = profile.get("reasoning_field")
        if reasoning_field is not None and (
            reasoning_field not in REASONING_FIELDS or profile["protocol"] != "openai_compatible"
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid reasoning_field"
            )
        if profile.get("request_options") is not None:
            if profile["protocol"] != "openai_compatible":
                raise NativeProviderProfileError(
                    f"provider profile {raw_profile_id} names request_options on a non-OpenAI protocol"
                )
            try:
                profile["request_options"] = validate_request_options(profile["request_options"])
            except ValueError as exc:
                raise NativeProviderProfileError(
                    f"provider profile {raw_profile_id} has invalid request_options: {exc}"
                ) from exc
        recovery_window_s = profile.get("recovery_window_s")
        if recovery_window_s is not None and (
            isinstance(recovery_window_s, bool) or not isinstance(recovery_window_s, (int, float))
            or not 30 <= float(recovery_window_s) <= 1800
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid recovery_window_s (30..1800 s)"
            )
        # Either protocol may bind the closed affinity mode; both clients send
        # the same X-Session-ID, to a loopback gateway or an HTTPS endpoint.
        session_affinity = profile.get("session_affinity")
        if session_affinity is not None and session_affinity not in (
            SESSION_AFFINITY_MODES
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid session_affinity"
            )
        if session_affinity is not None and endpoint_transport not in ENDPOINT_TRANSPORTS:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} limits session affinity to loopback or HTTPS endpoints"
            )
        for field in ("model_env", "base_url_env", "api_key_env"):
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", str(profile[field])):
                raise NativeProviderProfileError(
                    f"provider profile {raw_profile_id} has an invalid {field}"
                )
        proxy_env = profile["proxy_env"]
        if (
            not isinstance(proxy_env, list)
            or len(proxy_env) != len(set(proxy_env))
            or any(
                not isinstance(name, str)
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
                for name in proxy_env
            )
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid proxy_env"
            )
        for field in ("model", "reasoning_effort"):
            if not isinstance(profile[field], str) or not profile[field]:
                raise NativeProviderProfileError(
                    f"provider profile {raw_profile_id} has invalid {field}"
                )
        for field in ("context_window_tokens", "max_output_tokens"):
            amount = profile[field]
            if isinstance(amount, bool) or not isinstance(amount, int) or amount <= 0:
                raise NativeProviderProfileError(
                    f"provider profile {raw_profile_id} has invalid {field}"
                )
        if profile["context_window_tokens"] <= profile["max_output_tokens"] + 2048:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has no safe input window"
            )
        if not isinstance(profile["stream"], bool) or not isinstance(
            profile["supports_images"], bool
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid capabilities"
            )
        image_limit = profile.get("max_inline_images_per_request")
        if image_limit is not None and (
            isinstance(image_limit, bool)
            or not isinstance(image_limit, int)
            or image_limit <= 0
            or profile["supports_images"] is not True
        ):
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has invalid max_inline_images_per_request"
            )
        if profile["qualification_schema"] != PROBE_SCHEMA:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has unsupported qualification schema"
            )
        if profile["qualification_version"] != 1:
            raise NativeProviderProfileError(
                f"provider profile {raw_profile_id} has unsupported qualification version"
            )
        output[raw_profile_id] = profile
    return output


def load_builtin_provider_profiles() -> dict[str, dict[str, Any]]:
    """Load reusable registrations without consulting an experiment archive."""

    return load_provider_profiles(BUILTIN_PROVIDER_CATALOG)


def select_provider_profile(
    path: Path = BUILTIN_PROVIDER_CATALOG, *, profile_id: str | None = None,
    default_profile_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Prefer explicit selection, then package default, then catalog default."""

    profiles = load_provider_profiles(path)
    selected = profile_id or default_profile_id
    if selected is None:
        selected = yaml.safe_load(path.read_text(encoding="utf-8")).get("default_profile")
    if not isinstance(selected, str) or selected not in profiles:
        raise NativeProviderProfileError(
            f"unknown provider profile: {selected}; choose one of {sorted(profiles)}"
        )
    profile = profiles[selected]
    if profile.get("admission") == "registered_only":
        raise NativeProviderProfileError(
            f"provider profile {selected} is registration-only and has no live admission"
        )
    if profile.get("usage_scope") == "supported_coding_tools_only":
        raise NativeProviderProfileError(
            f"provider profile {selected} is restricted to supported coding tools"
        )
    return selected, profile


def _safe_error(exc: Exception, redactions: Sequence[str]) -> str:
    detail = str(exc)
    for value in redactions:
        if value:
            detail = detail.replace(value, "[REDACTED]")
    return detail[:1000]


def _tool_call(message: Any) -> tuple[str | None, str | None, dict[str, Any] | None]:
    if not isinstance(message, Mapping):
        return None, None, None
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], Mapping):
        return None, None, None
    call = calls[0]
    function = call.get("function")
    if not isinstance(function, Mapping):
        return None, None, None
    arguments = function.get("arguments", "{}")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, None, None
    if not isinstance(arguments, Mapping):
        return None, None, None
    call_id = call.get("id")
    name = function.get("name")
    return (
        call_id if isinstance(call_id, str) else None,
        name if isinstance(name, str) else None,
        dict(arguments),
    )


def probe_native_provider(
    provider: Any, *, model: str, tool_definitions: Sequence[Mapping[str, Any]],
    tool_name: str, tool_arguments: Mapping[str, Any], include_images: bool,
    reasoning_effort: str = "provider_default", stream: bool = True,
    redactions: Sequence[str] = (),
) -> dict[str, Any]:
    """Probe the configured transport, one tool schema, pairing and image semantics.

    The returned record contains only bounded diagnostics. Exact requests, raw
    responses and credentials are deliberately not retained.
    """

    definitions = [dict(item) for item in tool_definitions]
    matching = [
        item for item in definitions
        if isinstance(item.get("function"), Mapping)
        and item["function"].get("name") == tool_name
    ]
    if len(matching) != 1:
        raise NativeProviderProfileError(
            f"probe tool {tool_name!r} must have exactly one definition"
        )
    checks: list[dict[str, Any]] = []
    request_options = (
        {} if reasoning_effort == "provider_default"
        else {"reasoning_effort": reasoning_effort}
    )
    transport_options = {
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }

    marker = f"NATIVE-HARNESS-{secrets.token_hex(8)}"
    try:
        response = provider.complete(model_call_id="probe-text", request={
            "model": model,
            "messages": [{"role": "user", "content": f"Reply with exactly this marker: {marker}"}],
            "max_tokens": 256,
            **transport_options,
            **request_options,
        })
        message = response.get("message", {})
        content = message.get("content") if isinstance(message, Mapping) else None
        passed = isinstance(content, str) and marker in content
        checks.append({
            "check": "streaming_text", "passed": passed,
            "detail": {
                "marker_observed": passed,
                "reasoning_observed": bool(
                    message.get("reasoning_content") if isinstance(message, Mapping) else None
                ),
                "usage_observed": isinstance(response.get("usage"), Mapping),
                "finish_reason": response.get("finish_reason"),
            },
        })
    except Exception as exc:
        checks.append({
            "check": "streaming_text", "passed": False,
            "detail": {"error": _safe_error(exc, redactions)},
        })

    tool_message: dict[str, Any] | None = None
    try:
        response = provider.complete(model_call_id="probe-tool", request={
            "model": model,
            "messages": [{
                "role": "user",
                "content": (
                    f"Call {tool_name} exactly once with these JSON arguments: "
                    f"{json.dumps(dict(tool_arguments), ensure_ascii=False, sort_keys=True)}"
                ),
            }],
            "tools": matching,
            "tool_choice": "auto",
            "max_tokens": 512,
            **transport_options,
            **request_options,
        })
        message = response.get("message", {})
        call_id, returned_name, returned_arguments = _tool_call(message)
        passed = (
            call_id is not None and returned_name == tool_name
            and returned_arguments == dict(tool_arguments)
        )
        if passed:
            tool_message = dict(message)
        continuation = (
            message.get(PROVIDER_CONTINUATION_MESSAGE_KEY)
            if isinstance(message, Mapping) else None
        )
        checks.append({
            "check": "tool_schema", "passed": passed,
            "detail": {
                "tool_name": returned_name,
                "arguments_valid": returned_arguments == dict(tool_arguments),
                "finish_reason": response.get("finish_reason"),
                "provider_continuation_observed": isinstance(
                    continuation, Mapping
                ),
                "provider_continuation_sha256": (
                    continuation.get("state_sha256")
                    if isinstance(continuation, Mapping) else None
                ),
            },
        })
    except Exception as exc:
        checks.append({
            "check": "tool_schema", "passed": False,
            "detail": {"error": _safe_error(exc, redactions)},
        })

    continuity_marker = f"TOOL-RESULT-{secrets.token_hex(8)}"
    if tool_message is None:
        checks.append({
            "check": "tool_result_continuity", "passed": False,
            "detail": {"error": "tool response was unavailable"},
        })
    else:
        call_id, _name, _arguments = _tool_call(tool_message)
        try:
            response = provider.complete(model_call_id="probe-continuity", request={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Call {tool_name} exactly once with these JSON arguments: "
                            f"{json.dumps(dict(tool_arguments), ensure_ascii=False, sort_keys=True)}"
                        ),
                    },
                    tool_message,
                    {
                        "role": "tool", "tool_call_id": call_id, "name": tool_name,
                        "content": json.dumps({
                            "ok": False,
                            "error": {
                                "code": "probe_only",
                                "message": "Diagnostic probe; no mutation was executed.",
                                "retryable": False,
                            },
                        }, sort_keys=True),
                    },
                    {
                        "role": "user",
                        "content": f"Acknowledge the diagnostic result with exactly: {continuity_marker}",
                    },
                ],
                "tools": matching,
                "tool_choice": "none",
                "max_tokens": 256,
                **transport_options,
                **request_options,
            })
            message = response.get("message", {})
            content = message.get("content") if isinstance(message, Mapping) else None
            passed = isinstance(content, str) and continuity_marker in content
            checks.append({
                "check": "tool_result_continuity", "passed": passed,
                "detail": {
                    "marker_observed": passed,
                    "finish_reason": response.get("finish_reason"),
                },
            })
        except Exception as exc:
            checks.append({
                "check": "tool_result_continuity", "passed": False,
                "detail": {"error": _safe_error(exc, redactions)},
            })

    if include_images:
        try:
            response = provider.complete(model_call_id="probe-image", request={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Name the dominant color in this image in one word."},
                        {"type": "image_url", "image_url": {"url": _red_png_data_url()}},
                    ],
                }],
                "max_tokens": 256,
                **transport_options,
                **request_options,
            })
            message = response.get("message", {})
            content = message.get("content") if isinstance(message, Mapping) else None
            answer = content.strip().lower() if isinstance(content, str) else ""
            passed = answer == "red"
            checks.append({
                "check": "image_content", "passed": passed,
                "detail": {
                    "answer": answer[:80] if answer else None,
                    "semantic_answer_verified": passed,
                    "finish_reason": response.get("finish_reason"),
                },
            })
        except Exception as exc:
            checks.append({
                "check": "image_content", "passed": False,
                "detail": {"error": _safe_error(exc, redactions)},
            })

    return {
        "schema": PROBE_SCHEMA,
        "passed": bool(checks) and all(check["passed"] for check in checks),
        "model": model,
        "reasoning_effort": reasoning_effort,
        "stream": stream,
        "tool": tool_name,
        "images_required": include_images,
        "checks": checks,
    }
