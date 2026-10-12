"""Cache transition evidence when L1/L2 rewrite an older read-only result."""
from __future__ import annotations

from typing import Any, Sequence


def transition_record(
    *, projection_epoch: int, from_epoch: int, before: int, after: int,
    event_ids: Sequence[str], pre_pressure: int, post_pressure: int,
    pre_bytes: int, post_bytes: int, inline_images: int, media_bytes: int,
) -> dict[str, Any]:
    """The lifecycle transition a deferred eager externalization announces.

    Its image and media counts are equal on both sides on purpose: this pass
    externalizes read-only *results*, not media, so nothing visual leaves. What
    changes is the text of already-delivered results, and that is precisely
    what the prefix check would otherwise refuse without an epoch.
    """

    return {
        "schema": "ora.context-projection-transition.v4",
        "kind": "lifecycle_read_only_externalization",
        "projection_epoch": projection_epoch,
        "from_epoch": from_epoch,
        "to_epoch": projection_epoch,
        "reasons": ["action_step_left_protected_near_window"],
        "externalized_action_steps_before": before,
        "externalized_action_steps_after": after,
        "externalized_event_ids": list(event_ids),
        "semantic_loss": False,
        "pre_inline_images": inline_images,
        "post_inline_images": inline_images,
        "pre_media_bytes": media_bytes,
        "post_media_bytes": media_bytes,
        "pre_estimated_input_tokens": pre_pressure,
        "post_estimated_input_tokens": post_pressure,
        "pre_request_utf8_bytes": pre_bytes,
        "post_request_utf8_bytes": post_bytes,
    }
