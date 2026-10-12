"""Capability-gated delivery rules for every projected action effect.

The receipt-schema registry in :mod:`.receipt_projection` returns each
projector wrapped here, so one rule set covers every backend's envelope
without a per-backend branch. Order matters:

1. the D1 summary line is computed from the projected effect, before anything
   is moved or converted, so it states exactly what that effect states;
2. D3a points feedback the current facts own at its owner;
3. D2 restates every angle in degrees;
4. ``observation_delivery_trim_v2`` leaves out what carries no information
   (:mod:`ora.agent.context.delivery_trim`), on the degree form;
5. the summary is attached at the head (``action_summary`` sorts after
   ``action`` in the session's ``sort_keys`` serialization).

A surface declaring none of the four capabilities receives the projector's
own object, byte for byte.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..context.delivery_trim import CAPABILITY as TRIM, trim_action_effect
from ..context.payload_dedup import CAPABILITY as DEDUP, point_effect_to_facts
from ..context.policy_units import CAPABILITY as DEGREES, policy_degrees
from .action_summary import CAPABILITY as SUMMARY, SUMMARY_KEY, action_summary

DELIVERY_CAPABILITIES = (SUMMARY, DEDUP, DEGREES, TRIM)


def deliver_action_effect(effect: Any, supports: Callable[[str], bool]) -> Any:
    """Apply the declared delivery rules to one projected effect."""
    declared = {capability for capability in DELIVERY_CAPABILITIES if supports(capability)}
    if not declared or not isinstance(effect, Mapping):
        return effect
    line = action_summary(effect) if SUMMARY in declared else None
    delivered: dict[str, Any] = point_effect_to_facts(effect) if DEDUP in declared else dict(effect)
    if DEGREES in declared:
        delivered = policy_degrees(delivered)
    if TRIM in declared:
        delivered = trim_action_effect(delivered)
    if line is not None:
        delivered[SUMMARY_KEY] = line
    return delivered


def delivered_projector(projector: Callable[..., Any] | None) -> Callable[..., Any] | None:
    """Wrap one registered projector with the delivery rules its context declares."""
    if projector is None:
        return None

    def project(record: Mapping[str, Any], context: Any) -> Any:
        effect = projector(record, context)
        supports = getattr(context, "supports", None)
        return deliver_action_effect(effect, supports) if callable(supports) else effect

    project.__wrapped__ = projector  # type: ignore[attr-defined]
    return project


__all__ = ["DELIVERY_CAPABILITIES", "deliver_action_effect", "delivered_projector"]
