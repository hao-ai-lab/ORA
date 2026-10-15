# Copyright contributors to the Robot Harness Runtime project.
"""Where each context layer answers for the model it is bound to.

L4 externalizes losslessly at a share of the usable input window, with the
absolute ``trigger_estimated_tokens`` as a floor. Every shipped baseline
carries 48,000, chosen against a 200K context; read against a 1M model it is
4.7% of usable input instead of the 24% it was picked to be, so the share
governs above the floor.

L5 summarizes one closed Episode at an absolute Provider-input watermark. From
Context management v6 it is
``episode_checkpoint_trigger_tokens`` alone, 150,000 in every maintained
configuration, and no longer scales with the window: v5's
``max(tokens, 0.60 x hard)`` put it at 618,086 tokens on a 1M DeepSeek window
(1,030,144 usable), where it effectively never fired. v4/v5 keep that formula
here so their recorded runs, frozen archives and explicit historical
treatments read back with the exact threshold they ran with.

The property that matters is order: the lossless answer should be reached
before the summarizing one. In token terms that holds only up to about 620K of
usable input (L4's share reaches 150,000 at 618,600); at a 1M window L4's token
trigger is about 250,000 (249,792 at 1,030,144 usable), above L5. In practice
L4 still precedes L5 there, because its 24-inline-image trigger fires far
earlier than either token trigger. The v6 validator only checks the
window-independent part: the L4 floor below the absolute L5. L6 is unchanged
(80% of usable input, target 65%).
"""

from __future__ import annotations

from typing import Any, Mapping

from .management import WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS

#: The share of usable input at which L4 externalizes, recovered from the
#: absolute ``trigger_estimated_tokens`` every baseline carries: 48000 against
#: the 197952 usable tokens of the 200K window it was chosen for. Keeping the
#: share rather than the count is what makes the layer mean the same thing on a
#: model with a different window.
ESTIMATE_PRESSURE_WINDOW_SHARE = 48_000 / (200_000 - 2_048)

#: ``resolved_episode_checkpoint_trigger_formula`` in projection evidence.
ABSOLUTE_EPISODE_TRIGGER_FORMULA = "absolute_tokens"
WINDOW_SCALED_EPISODE_TRIGGER_FORMULA = (
    "max(absolute_tokens, int(hard_input_tokens * ratio))"
)


def estimate_pressure_trigger(
    management: Mapping[str, Any], hard_input_tokens: int,
) -> int:
    """Where L4 externalizes losslessly.

    The absolute count stays as a floor, so a 200K binding keeps exactly the
    behaviour its figure was picked for and nothing changes there; the share
    governs above it. Below about 620K of usable input this sits below
    :func:`episode_checkpoint_trigger`; above it the image trigger, not this
    one, keeps L4 ahead of L5 (see the module docstring).
    """

    return max(
        int(management["trigger_estimated_tokens"]),
        int(hard_input_tokens * ESTIMATE_PRESSURE_WINDOW_SHARE),
    )


def episode_checkpoint_trigger(
    management: Mapping[str, Any], hard_input_tokens: int,
) -> int:
    """Where L5 closes an Episode: absolute from v6, window-scaled in v4/v5."""

    tokens = int(management["episode_checkpoint_trigger_tokens"])
    if management.get("schema") in WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS:
        return max(
            tokens,
            int(
                hard_input_tokens
                * float(management["episode_checkpoint_trigger_ratio"])
            ),
        )
    return tokens


def episode_checkpoint_trigger_formula(management: Mapping[str, Any]) -> str:
    """The formula :func:`episode_checkpoint_trigger` applied, for evidence."""

    if management.get("schema") in WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS:
        return WINDOW_SCALED_EPISODE_TRIGGER_FORMULA
    return ABSOLUTE_EPISODE_TRIGGER_FORMULA


__all__ = [
    "ABSOLUTE_EPISODE_TRIGGER_FORMULA",
    "ESTIMATE_PRESSURE_WINDOW_SHARE",
    "WINDOW_SCALED_EPISODE_TRIGGER_FORMULA",
    "episode_checkpoint_trigger",
    "episode_checkpoint_trigger_formula",
    "estimate_pressure_trigger",
]
