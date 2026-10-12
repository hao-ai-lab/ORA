"""When the lossless Native Context ages older ActionSteps (Context v4-v6).

v4 moves every lifecycle frontier one ActionStep at every fresh observation:
the oldest inline images, then (one step later) the read-only results and the
bulky observation body of an older step are rewritten in place. A Provider
whose prompt cache only reuses an exactly extended request (measured on the
OpenAI/Codex backend) therefore falls back to the static prefix on roughly
every call that follows a physical action.

v5 keeps the v4 tails as minimums and ages in batches. A batch is due once
``tail_aging_stride_action_steps`` ActionSteps are pending beyond either
minimum tail; it then moves the read-only and media frontiers together to
exactly those tails, and any admitted L4 pressure pass joins the same epoch.
A reset releases prior-Episode media at once, as v4 does. Between batches the
request is a strict extension of the previous one. With stride 1 a batch is
due exactly when v4 would move a frontier, so v5 reproduces v4 byte for byte.
v6 ages exactly as v5 does; it changes only the L5 watermark.

L4 pressure is evaluated only while no aging is pending: the slack a stride
keeps inline is a cache-stability allowance, not pressure, and must not turn
the stride into a second rewrite inside each batch. Because L4 never moves
beyond the read-only frontier, deferring its evaluation to the next batch
delays at most ``stride - 1`` ActionSteps of lossless externalization; the
Provider-anchored L5/L6 layers are not deferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .management import TAIL_AGING_SCHEMAS, TAIL_AGING_STRIDE_FIELD

V4_MEDIA_POLICY = "current_round_action_step_tail_v1"
STRIDE_MEDIA_POLICY = "current_round_action_step_tail_stride_v1"
TAIL_AGING_RECORD_SCHEMA = "ora.context-tail-aging.v1"


@dataclass(frozen=True)
class TailAging:
    """One projection's aging decision; frontiers count ActionSteps."""

    schema: str
    stride: int
    read_only_target: int
    media_target: int
    read_only_through: int
    media_through: int
    reasons: tuple[str, ...]
    aged: bool

    @property
    def pending(self) -> bool:
        """Whether older steps stay unaged beyond a minimum tail."""

        return (
            self.read_only_through < self.read_only_target
            or self.media_through < self.media_target
        )

    @property
    def media_policy(self) -> str:
        return (
            STRIDE_MEDIA_POLICY if self.schema in TAIL_AGING_SCHEMAS
            else V4_MEDIA_POLICY
        )

    def manifest_fields(self) -> dict[str, Any]:
        """Projection-manifest fields; v4 evidence keeps its exact field set."""

        if self.schema not in TAIL_AGING_SCHEMAS:
            return {}
        return {
            "resolved_tail_aging_stride_action_steps": self.stride,
            "tail_aging": {
                "schema": TAIL_AGING_RECORD_SCHEMA,
                "stride_action_steps": self.stride,
                "aged": self.aged,
                "reasons": list(self.reasons),
                "read_only_target_action_steps": self.read_only_target,
                "media_target_action_steps": self.media_target,
                "pending_read_only_action_steps": (
                    self.read_only_target - self.read_only_through
                ),
                "pending_media_action_steps": (
                    self.media_target - self.media_through
                ),
                "pressure_admission": (
                    "deferred_to_tail_aging" if self.pending else "evaluated"
                ),
            },
        }


def plan_tail_aging(
    *, management: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
    completed: int, near_tail: int, read_only_active: int, media_active: int,
    allow_transition: bool,
) -> TailAging:
    """Decide whether this projection ages, and to which frontiers.

    ``read_only_active`` and ``media_active`` are the numbers of leading
    completed ActionSteps already aged. The targets are the v4 frontiers: the
    shared causal/read-only near window and the current-Episode media tail.
    """

    stride = int(management.get(TAIL_AGING_STRIDE_FIELD, 1))
    current_round = inventory[-1]["session_round"]
    current_round_completed = sum(
        item.get("complete") is True
        and item.get("session_round") == current_round
        for item in inventory
    )
    read_only_target = max(0, completed - near_tail)
    media_target = completed - min(
        int(management["inline_media_tail_action_steps"]),
        current_round_completed,
    )
    reasons: list[str] = []
    if read_only_target - read_only_active >= stride:
        reasons.append("read_only_stride_reached")
    if media_target - media_active >= stride:
        reasons.append("media_stride_reached")
    if (
        media_active < completed
        and inventory[media_active].get("session_round") != current_round
    ):
        reasons.append("prior_episode_media_released")
    aged = allow_transition and bool(reasons)
    return TailAging(
        schema=str(management["schema"]), stride=stride,
        read_only_target=read_only_target, media_target=media_target,
        read_only_through=(
            max(read_only_active, read_only_target) if aged else read_only_active
        ),
        media_through=max(media_active, media_target) if aged else media_active,
        reasons=tuple(reasons), aged=aged,
    )


__all__ = [
    "STRIDE_MEDIA_POLICY",
    "TAIL_AGING_RECORD_SCHEMA",
    "TailAging",
    "V4_MEDIA_POLICY",
    "plan_tail_aging",
]
