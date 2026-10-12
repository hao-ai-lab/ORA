"""Validation of already canonical Anthropic history."""
from __future__ import annotations
import base64
from typing import Any, Mapping

def _has_complete_anthropic_thinking_signature(
    body: Mapping[str, Any],
) -> bool:
    """Return whether a wire request replays one complete encoded signature."""

    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "thinking":
                continue
            signature = block.get("signature")
            if not isinstance(signature, str) or not signature:
                continue
            padded = signature + "=" * (-len(signature) % 4)
            for alternate in (None, b"-_"):
                try:
                    base64.b64decode(
                        padded.encode("ascii"),
                        altchars=alternate,
                        validate=True,
                    )
                except (UnicodeEncodeError, ValueError):
                    continue
                return True
    return False
