"""The Native provider admission probe behind ``preflight._native_provider_candidate``.

One candidate credential runs the marker stage, the optional image stage and
the optional continuity tool round trip, in that order, against one provider
that is always closed afterwards. ``ProbeRun`` carries the current stage name:
a stage is entered before its request is built, so an exception anywhere is
reported against the stage that was running. ``provider_marker_verdict``
derives the ``provider_marker`` detail from the stage responses, and
``probe_failure_detail`` the detail of a stage that raised.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any, Mapping

from .provider_image_answer import image_answer_word as _image_answer_word, names_red as _names_red
from .provider_probe_diagnostics import ProviderProbeDiagnostics


_NATIVE_PROVIDER_PROBE_SCHEMA = "ora.native-provider-probe.v1"

# A solid-red 64x64 PNG; the image stage asks for its dominant colour.
_RED_IMAGE_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJ"
    "AAAAX0lEQVR4nO3PQQ0AIBDAMMC/50MEj4ZkVbDt"
    "WX87OuBVA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNa"
    "A1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNa"
    "A1oDWgNaA9oFUoUBf3Xr7AgAAAAASUVORK5CYII="
)


class ProbeRun:
    """The stage cursor, stage telemetry and provider of one candidate probe."""

    def __init__(self, probes: ProviderProbeDiagnostics, credential_pool: list[str]) -> None:
        self.probes = probes
        self.credential_pool = credential_pool
        self.stage = "provider-build"
        self.provider: Any = None

    def complete(self, **kwargs: Any) -> Any:
        """Send one request for the current stage and record its response."""

        self.probes.begin(self.stage)
        return self.probes.observe(
            self.stage, self.provider.complete(**kwargs), self.credential_pool,
        )


@dataclass
class ProbeOutcome:
    """The stage responses a completed probe hands to the verdict."""

    response: Any
    image_response: Any
    continuity_response: Any
    continuity_call_valid: Any
    continuity_marker: str | None



# Marker and image answers share one output budget: low-effort reasoning models
# (self-hosted DeepSeek) spent all of 256 tokens on reasoning before
# the one-word image answer.
PROBE_ANSWER_BUDGET = 1024
# A sampled answer can miss without the Provider being unhealthy (a paraphrased
# marker, an answer that ran into the budget). Those response-level misses re-run
# the whole suite once on the same credential; HTTP, transport,
# malformed-response and credential-echo failures never do.
RESPONSE_RETRY_CATEGORIES = frozenset({"output_cap", "semantic_mismatch"})


def response_level_miss(detail: Mapping[str, Any]) -> bool:
    """Whether a failed probe suite failed only on a sampled answer."""
    probes = detail.get("probes")
    record = probes.get(detail.get("stage")) if isinstance(probes, Mapping) else None
    return isinstance(record, Mapping) and record.get("failure_category") in RESPONSE_RETRY_CATEGORIES

def probe_request_options(policy: Mapping[str, Any]) -> dict[str, Any]:
    return (
        {}
        if policy.get("reasoning_effort") == "provider_default"
        else {"reasoning_effort": policy["reasoning_effort"]}
    )


def run_probe_stages(
    run: ProbeRun, policy: Mapping[str, Any], *, api_key: str, marker: str,
    request_options: Mapping[str, Any],
) -> ProbeOutcome:
    """Build the probe provider, run every stage in order and always close it."""

    from ora.agent.providers import build_native_provider

    provider = build_native_provider(
        policy["provider"], base_url=policy["endpoint"], api_key=api_key,
        auth_scheme=policy.get("auth_scheme"),
        continuation_replay=policy.get(
            "provider_continuation_replay", "exact_v1",
        ),
        # The probe names its output budget as the session will.
        output_token_parameter=policy.get("output_token_parameter"),
        tool_strictness=policy.get("tool_strictness"),
        reasoning_field=policy.get("reasoning_field"),
        session_affinity_id=(
            f"ora:preflight:{secrets.token_hex(8)}"
            if policy.get("provider_session_affinity")
            == "x_session_id_run_v1"
            else None
        ),
    )
    provider.preflight_probe = True
    run.provider = provider
    try:
        return _probe_stages(run, policy, marker, request_options)
    finally:
        provider.close()


def _probe_stages(
    run: ProbeRun, policy: Mapping[str, Any], marker: str,
    request_options: Mapping[str, Any],
) -> ProbeOutcome:
    """Marker, then image when supported, then continuity when qualified."""

    run.stage = "preflight-marker"
    response = run.complete(
        model_call_id="preflight-marker",
        request=_marker_request(policy, marker, request_options),
    )
    message = response.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    marker_observed = isinstance(content, str) and marker in content
    run.probes.semantic(run.stage, marker_observed)
    image_response = None
    if marker_observed and policy.get("supports_images") is True:
        run.stage = "preflight-image"
        image_response = run.complete(
            model_call_id="preflight-image",
            request=_image_request(policy, request_options),
        )
    image_passed = _image_stage_passed(policy, image_response)
    if image_response is not None:
        run.probes.semantic(run.stage, image_passed)
    continuity_response = None
    continuity_call_valid = None
    continuity_marker = None
    if marker_observed and image_passed and policy.get("provider_qualification_schema") == (
        _NATIVE_PROVIDER_PROBE_SCHEMA
    ):
        continuity_call_valid, continuity_marker, continuity_response = (
            _continuity_stages(run, policy, request_options)
        )
    return ProbeOutcome(
        response=response, image_response=image_response,
        continuity_response=continuity_response,
        continuity_call_valid=continuity_call_valid,
        continuity_marker=continuity_marker,
    )


def _stream_fields(policy: Mapping[str, Any], request_options: Mapping[str, Any]) -> dict[str, Any]:
    """The ``stream``, request-option and usage-stream fields every stage sends."""

    return {
        "stream": bool(policy["stream"]),
        **request_options,
        **(
            {"stream_options": {"include_usage": True}}
            if policy["stream"] else {}
        ),
    }


def _marker_request(
    policy: Mapping[str, Any], marker: str, request_options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": policy["model"],
        "messages": [{
            "role": "user",
            "content": f"Reply with exactly this marker: {marker}",
        }],
        # Reasoning models may spend part of this budget before
        # emitting the requested marker. Keep the probe small,
        # but leave enough room to avoid a false-negative
        # length stop from otherwise healthy providers.
        "max_tokens": PROBE_ANSWER_BUDGET,
        **_stream_fields(policy, request_options),
    }


def _image_request(
    policy: Mapping[str, Any], request_options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": policy["model"],
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Name the dominant color in this image in one word.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _RED_IMAGE_URL},
                },
            ],
        }],
        # Keep this aligned with the text marker. Providers
        # with visible/private reasoning may otherwise
        # consume a tiny allowance before emitting the
        # one-word semantic answer.
        "max_tokens": PROBE_ANSWER_BUDGET,
        **_stream_fields(policy, request_options),
    }


def _image_stage_passed(policy: Mapping[str, Any], image_response: Any) -> Any:
    image_message = (
        image_response.get("message")
        if isinstance(image_response, Mapping) else None
    )
    image_content = (
        image_message.get("content") if isinstance(image_message, Mapping) else None
    )
    return (
        policy.get("supports_images") is not True
        or _names_red(_image_answer_word(image_content))
    )


def _continuity_stages(
    run: ProbeRun, policy: Mapping[str, Any], request_options: Mapping[str, Any],
) -> tuple[Any, str, Any]:
    """Run the tool round trip; return the call validity, marker and resume response."""

    continuity_token = f"CONTINUITY-{secrets.token_hex(6)}"
    continuity_marker = f"CONTINUED-{secrets.token_hex(6)}"
    continuity_user = {
        "role": "user",
        "content": (
            "Call preflight_echo exactly once with token "
            f"{continuity_token}. Do not answer in text."
        ),
    }
    run.stage = "preflight-continuity-tool"
    first_continuity = run.complete(
        model_call_id="preflight-continuity-tool",
        request=_continuity_tool_request(policy, continuity_user, request_options),
    )
    first_message, continuity_call, continuity_call_valid = _continuity_call(
        first_continuity, continuity_token,
    )
    run.probes.semantic(run.stage, bool(continuity_call_valid))
    if not continuity_call_valid:
        raise RuntimeError(
            "provider continuity qualification did not return one complete tool call"
        )
    run.stage = "preflight-continuity-resume"
    continuity_response = run.complete(
        model_call_id="preflight-continuity-resume",
        request=_continuity_resume_request(
            policy, request_options, continuity_user=continuity_user,
            first_message=first_message, continuity_call=continuity_call,
            continuity_token=continuity_token, continuity_marker=continuity_marker,
        ),
    )
    return continuity_call_valid, continuity_marker, continuity_response


def _continuity_tool_request(
    policy: Mapping[str, Any], continuity_user: Mapping[str, Any],
    request_options: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": policy["model"],
        "messages": [continuity_user],
        "tools": [{
            "type": "function",
            "function": {
                "name": "preflight_echo",
                "description": "Echo one qualification token.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "token": {"type": "string"},
                    },
                    "required": ["token"],
                    "additionalProperties": False,
                },
            },
        }],
        "tool_choice": "auto",
        "max_tokens": 512,
        **_stream_fields(policy, request_options),
    }


def _continuity_call(first_continuity: Any, continuity_token: str) -> tuple[Any, Any, Any]:
    """Return the first message, its single tool call and whether that call is valid."""

    first_message = first_continuity.get("message")
    raw_calls = (
        first_message.get("tool_calls")
        if isinstance(first_message, Mapping) else None
    )
    continuity_call = (
        raw_calls[0]
        if isinstance(raw_calls, list) and len(raw_calls) == 1
        and isinstance(raw_calls[0], Mapping) else None
    )
    function = (
        continuity_call.get("function")
        if isinstance(continuity_call, Mapping) else None
    )
    try:
        arguments = json.loads(str(function.get("arguments")))
    except (AttributeError, json.JSONDecodeError):
        arguments = None
    continuity_call_valid = (
        first_continuity.get("finish_reason") == "tool_calls"
        and isinstance(continuity_call, Mapping)
        and isinstance(continuity_call.get("id"), str)
        and continuity_call["id"]
        and isinstance(function, Mapping)
        and function.get("name") == "preflight_echo"
        and arguments == {"token": continuity_token}
    )
    return first_message, continuity_call, continuity_call_valid


def _continuity_resume_request(
    policy: Mapping[str, Any], request_options: Mapping[str, Any], *,
    continuity_user: Mapping[str, Any], first_message: Any, continuity_call: Any,
    continuity_token: str, continuity_marker: str,
) -> dict[str, Any]:
    return {
        "model": policy["model"],
        "messages": [
            continuity_user,
            dict(first_message),
            {
                "role": "tool",
                "tool_call_id": continuity_call["id"],
                "name": "preflight_echo",
                "content": json.dumps(
                    {"token": continuity_token},
                    sort_keys=True,
                ),
            },
            {
                "role": "user",
                "content": (
                    "The tool round trip is complete. Reply with "
                    f"exactly {continuity_marker}"
                ),
            },
        ],
        "tools": [],
        "max_tokens": 512,
        **_stream_fields(policy, request_options),
    }


def probe_failure_detail(run: ProbeRun, exc: Exception) -> dict[str, Any]:
    """Record the raising stage's failure and return the refused check's detail."""

    status = getattr(exc, "http_status", None)
    run.probes.failure(run.stage, status)
    return {
        "probes": run.probes.records,
        "stage": run.stage, "error_type": type(exc).__name__,
        "http_status": status if type(status) is int else None,
    }


def provider_marker_verdict(
    policy: Mapping[str, Any], run: ProbeRun, marker: str, outcome: ProbeOutcome,
) -> tuple[Any, dict[str, Any]]:
    """Return the ``provider_marker`` pass value and detail of a completed probe."""

    message = outcome.response.get("message")
    content = message.get("content") if isinstance(message, Mapping) else None
    image = _image_findings(policy, outcome.image_response)
    continuity_required, continuity_passed = _continuity_verdict(policy, outcome)
    if outcome.continuity_response is not None:
        run.probes.semantic(run.stage, bool(continuity_passed))
    passed = (
        isinstance(content, str) and marker in content
        and image["passed"] and continuity_passed
    )
    return passed, {
        "model": policy["model"],
        "provider": policy["provider"],
        "stage": run.stage,
        "provider_profile": policy.get("provider_profile"),
        "reasoning_effort": policy.get("reasoning_effort"),
        "endpoint_matches_plan": True,
        "probes": run.probes.records,
        "marker_observed": isinstance(content, str) and marker in content,
        "image_request_required": policy.get("supports_images") is True,
        "image_response_observed": image["response_observed"],
        "image_semantics_verified": (
            image["semantics_verified"]
            if policy.get("supports_images") is True else None
        ),
        "image_answer": image["answer"][:80] if image["response_observed"] else None,
        "image_finish_reason": image["finish_reason"],
        "image_reasoning_observed": bool(image["reasoning"]),
        "image_usage_observed": isinstance(image["usage"], Mapping),
        "continuity_required": continuity_required,
        "continuity_tool_call_valid": outcome.continuity_call_valid,
        "continuity_passed": continuity_passed,
        "session_affinity": policy.get("provider_session_affinity"),
    }


def _image_findings(policy: Mapping[str, Any], image_response: Any) -> dict[str, Any]:
    """What the image stage's response showed, read in a fixed field order."""

    image_message = (
        image_response.get("message")
        if isinstance(image_response, Mapping) else None
    )
    image_content = (
        image_message.get("content") if isinstance(image_message, Mapping) else None
    )
    image_reasoning = (
        image_message.get("reasoning_content")
        if isinstance(image_message, Mapping) else None
    )
    image_finish_reason = (
        image_response.get("finish_reason")
        if isinstance(image_response, Mapping) else None
    )
    image_usage = (
        image_response.get("usage")
        if isinstance(image_response, Mapping) else None
    )
    image_answer = _image_answer_word(image_content)
    image_response_observed = bool(image_answer)
    image_semantics_verified = _names_red(image_answer)
    image_passed = (
        policy.get("supports_images") is not True
        or image_semantics_verified
    )
    return {
        "reasoning": image_reasoning, "finish_reason": image_finish_reason,
        "usage": image_usage, "answer": image_answer,
        "response_observed": image_response_observed,
        "semantics_verified": image_semantics_verified, "passed": image_passed,
    }


def _continuity_verdict(
    policy: Mapping[str, Any], outcome: ProbeOutcome,
) -> tuple[bool, Any]:
    """Return whether continuity was required and whether it passed."""

    continuity_required = (
        policy.get("provider_qualification_schema")
        == _NATIVE_PROVIDER_PROBE_SCHEMA
    )
    continuity_response = outcome.continuity_response
    continuity_message = (
        continuity_response.get("message")
        if isinstance(continuity_response, Mapping) else None
    )
    continuity_content = (
        continuity_message.get("content")
        if isinstance(continuity_message, Mapping) else None
    )
    continuity_passed = (
        not continuity_required
        or (
            outcome.continuity_call_valid is True
            and isinstance(continuity_response, Mapping)
            and continuity_response.get("finish_reason") == "stop"
            and isinstance(continuity_content, str)
            and outcome.continuity_marker in continuity_content
        )
    )
    return continuity_required, continuity_passed
