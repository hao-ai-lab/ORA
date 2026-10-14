"""Human messages delivered at a Native request boundary on the policy thread.

The orchestrator owns command admission and the inbox. Native records only the
delivered public text and preserves its provenance in the model conversation.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from ora.agent.contracts import PolicySessionError


OPERATOR_MESSAGE_PREFIX = "<operator_message>\n"


class NativeOperatorMessagesMixin:
    def bind_operator_decision_callback(self, callback: Callable[..., None]) -> None:
        if not callable(callback) or getattr(self, "_operator_decision_callback", None) is not None:
            raise PolicySessionError("Native operator callback must be bound once")
        self._operator_decision_callback = callback

    def poll_operator_messages(self, *, observation_step: int, before_model_call_id: str) -> None:
        callback = getattr(self, "_operator_decision_callback", None)
        if callback is not None:
            callback(observation_step=observation_step, before_model_call_id=before_model_call_id)

    def record_operator_message(self, payload: Mapping[str, Any]) -> None:
        if self.closed or self.interrupted:
            raise PolicySessionError("Native policy session is not active")
        if (
            payload.get("type") not in {"steer", "follow_up"}
            or not isinstance(payload.get("id"), str)
            or not isinstance(payload.get("message"), str)
            or not payload["message"].strip()
            or payload.get("run_id") != self.run_id
            or type(payload.get("round_index")) is not int
            or type(payload.get("observation_step")) is not int
            or not isinstance(payload.get("before_model_call_id"), str)
        ):
            raise PolicySessionError("Native operator message is malformed")
        event = self._append(
            "operator_message", observation_step=payload["observation_step"],
            visibility="exact", payload=dict(payload),
        )
        self.messages.append({
            "role": "user",
            "content": OPERATOR_MESSAGE_PREFIX + json.dumps({
                "id": payload["id"],
                "message": payload["message"],
                "instruction": (
                    "Human operator guidance for the current benchmark task. "
                    "The original task and official evaluator remain unchanged; "
                    "all robot actions still use the advertised tools."
                ),
            }, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n</operator_message>",
            "_protected_context": True,
            "_context_source": f"operator_message_{payload['id']}",
            "_source_event_ids": [str(event["event_id"])],
        })
        self.durability.barrier()  # Acknowledgement follows durable input evidence.

    def record_run_termination(self, result: Mapping[str, Any]) -> None:
        self._append(
            "run_terminated",
            visibility="derived",
            payload={
                "state": result.get("state"),
                "termination": result.get("termination"),
                "decisions": result.get("decisions"),
                "model_calls": result.get("model_calls"),
            },
        )
