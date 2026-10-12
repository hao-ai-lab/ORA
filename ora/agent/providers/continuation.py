"""Opaque provider-owned state required to continue one assistant turn.

The Harness keeps a provider-neutral transcript for evidence and export, but
some model protocols return additional state that must be replayed verbatim on
the next call.  This module gives that state one content-addressed envelope and
binds it to the canonical assistant message that it accompanies.  Provider
adapters own the native payload schema; the shared Runtime only validates the
envelope and preserves it while the bound assistant message remains in active
context.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from ora.agent.contracts import PolicySessionError
from ora.agent.evidence.trajectory import canonical_bytes


PROVIDER_CONTINUATION_STATE_SCHEMA = (
    "ora.provider-continuation-state.v1"
)
PROVIDER_CONTINUATION_MESSAGE_KEY = "_provider_continuation_state"
REPLAY_POLICY = "required_unmodified"


def canonical_assistant_replay_view(
    message: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the provider-neutral assistant fields bound by a continuation.

    Harness metadata and normalized reasoning text are deliberately excluded.
    Reasoning remains trace evidence, while the provider-native payload is the
    only authorized source for replaying signed/encrypted reasoning blocks.
    """

    return {
        str(key): copy.deepcopy(value)
        for key, value in message.items()
        if not str(key).startswith("_") and key != "reasoning_content"
    }


def canonical_assistant_sha256(message: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_bytes(canonical_assistant_replay_view(message))
    ).hexdigest()


def strip_provider_continuation(
    message: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove internal continuation state from one canonical message."""

    return {
        str(key): copy.deepcopy(value)
        for key, value in message.items()
        if key != PROVIDER_CONTINUATION_MESSAGE_KEY
    }


@dataclass(frozen=True)
class ProviderContinuationState:
    """Validated, content-addressed envelope for opaque provider state."""

    provider_protocol: str
    model: str
    native_kind: str
    native_payload: Mapping[str, Any]
    canonical_message_sha256: str
    response_id: str | None
    state_sha256: str

    @staticmethod
    def _unsigned_body(
        *,
        provider_protocol: str,
        model: str,
        native_kind: str,
        native_payload: Mapping[str, Any],
        canonical_message_sha256: str,
        response_id: str | None,
    ) -> dict[str, Any]:
        return {
            "schema": PROVIDER_CONTINUATION_STATE_SCHEMA,
            "provider_protocol": provider_protocol,
            "model": model,
            "native_kind": native_kind,
            "native_payload": copy.deepcopy(dict(native_payload)),
            "canonical_message_sha256": canonical_message_sha256,
            "response_id": response_id,
            "replay_policy": REPLAY_POLICY,
        }

    @classmethod
    def create(
        cls,
        *,
        provider_protocol: str,
        model: str,
        native_kind: str,
        native_payload: Mapping[str, Any],
        canonical_message: Mapping[str, Any],
        response_id: str | None = None,
    ) -> "ProviderContinuationState":
        for label, value in (
            ("provider_protocol", provider_protocol),
            ("model", model),
            ("native_kind", native_kind),
        ):
            if not isinstance(value, str) or not value:
                raise PolicySessionError(
                    f"provider continuation {label} must be non-empty text"
                )
        if response_id is not None and (
            not isinstance(response_id, str) or not response_id
        ):
            raise PolicySessionError(
                "provider continuation response_id must be non-empty text"
            )
        if not isinstance(native_payload, Mapping):
            raise PolicySessionError(
                "provider continuation native_payload must be an object"
            )
        message_sha256 = canonical_assistant_sha256(canonical_message)
        body = cls._unsigned_body(
            provider_protocol=provider_protocol,
            model=model,
            native_kind=native_kind,
            native_payload=native_payload,
            canonical_message_sha256=message_sha256,
            response_id=response_id,
        )
        return cls(
            provider_protocol=provider_protocol,
            model=model,
            native_kind=native_kind,
            native_payload=copy.deepcopy(dict(native_payload)),
            canonical_message_sha256=message_sha256,
            response_id=response_id,
            state_sha256=hashlib.sha256(canonical_bytes(body)).hexdigest(),
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any],
    ) -> "ProviderContinuationState":
        expected_keys = {
            "schema", "provider_protocol", "model", "native_kind",
            "native_payload", "canonical_message_sha256", "response_id",
            "replay_policy", "state_sha256",
        }
        if set(value) != expected_keys:
            raise PolicySessionError(
                "provider continuation state fields are not canonical"
            )
        if value.get("schema") != PROVIDER_CONTINUATION_STATE_SCHEMA:
            raise PolicySessionError("unsupported provider continuation schema")
        if value.get("replay_policy") != REPLAY_POLICY:
            raise PolicySessionError("unsupported provider continuation replay policy")
        provider_protocol = value.get("provider_protocol")
        model = value.get("model")
        native_kind = value.get("native_kind")
        native_payload = value.get("native_payload")
        message_sha256 = value.get("canonical_message_sha256")
        response_id = value.get("response_id")
        state_sha256 = value.get("state_sha256")
        for label, item in (
            ("provider_protocol", provider_protocol),
            ("model", model),
            ("native_kind", native_kind),
        ):
            if not isinstance(item, str) or not item:
                raise PolicySessionError(
                    f"provider continuation {label} must be non-empty text"
                )
        if not isinstance(native_payload, Mapping):
            raise PolicySessionError(
                "provider continuation native_payload must be an object"
            )
        if (
            not isinstance(message_sha256, str)
            or len(message_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in message_sha256
            )
        ):
            raise PolicySessionError(
                "provider continuation lacks canonical message identity"
            )
        if response_id is not None and (
            not isinstance(response_id, str) or not response_id
        ):
            raise PolicySessionError("provider continuation has invalid response_id")
        if (
            not isinstance(state_sha256, str)
            or len(state_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in state_sha256
            )
        ):
            raise PolicySessionError("provider continuation lacks state identity")
        body = cls._unsigned_body(
            provider_protocol=provider_protocol,
            model=model,
            native_kind=native_kind,
            native_payload=native_payload,
            canonical_message_sha256=message_sha256,
            response_id=response_id,
        )
        expected = hashlib.sha256(canonical_bytes(body)).hexdigest()
        if state_sha256 != expected:
            raise PolicySessionError("provider continuation state hash mismatch")
        return cls(
            provider_protocol=provider_protocol,
            model=model,
            native_kind=native_kind,
            native_payload=copy.deepcopy(dict(native_payload)),
            canonical_message_sha256=message_sha256,
            response_id=response_id,
            state_sha256=state_sha256,
        )

    def validate_binding(
        self,
        *,
        message: Mapping[str, Any],
        provider_protocol: str,
        model: str,
    ) -> None:
        if self.provider_protocol != provider_protocol:
            raise PolicySessionError(
                "provider continuation protocol does not match the active adapter"
            )
        if self.model != model:
            raise PolicySessionError(
                "provider continuation model does not match the active request"
            )
        if canonical_assistant_sha256(message) != self.canonical_message_sha256:
            raise PolicySessionError(
                "provider continuation is not bound to this assistant message"
            )

    def to_dict(self) -> dict[str, Any]:
        body = self._unsigned_body(
            provider_protocol=self.provider_protocol,
            model=self.model,
            native_kind=self.native_kind,
            native_payload=self.native_payload,
            canonical_message_sha256=self.canonical_message_sha256,
            response_id=self.response_id,
        )
        body["state_sha256"] = self.state_sha256
        return body
