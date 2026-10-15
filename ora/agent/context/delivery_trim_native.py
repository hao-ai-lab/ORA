"""Delivery-trim rules for the Native observation envelope (``observation_delivery_trim_v2``).

These rules edit parts only the Native session assembles: the
``current_views`` block and the envelope's ``terminal_lifecycle``. They follow
the contract of :mod:`.delivery_trim`: one named field, left out only at its
stated value, with the information still in the same request.
"""

from __future__ import annotations

from typing import Any, Mapping

#: ``session._post_action_view_feedback``: the label of the view block.
VIEW_BLOCKS = ("current_views",)


def _without(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


# -- Envelope constants -------------------------------------------------------------

def trim_view_freshness(current_actual: Any, *, paired: bool) -> Any:
    """``fresh_for_this_action`` goes when it equals ``comparison.available``.

    Both say whether this observation answers an action: the view set is
    action-bound exactly when an action-input observation is retained. Run
    before the round's static-facts reduction, so the slot never holds it.
    """
    if not isinstance(current_actual, Mapping):
        return current_actual
    result = dict(current_actual)
    for label in VIEW_BLOCKS:
        block = current_actual.get(label)
        if isinstance(block, Mapping) and block.get("fresh_for_this_action") is paired:
            result[label] = _without(block, ("fresh_for_this_action",))
    return result


def trim_observation_envelope(envelope: Any) -> Any:
    """The envelope without ``terminal_lifecycle: false``."""
    if not isinstance(envelope, Mapping):
        return envelope
    result = dict(envelope)
    if envelope.get("terminal_lifecycle") is False:
        del result["terminal_lifecycle"]
    return result


__all__ = ["VIEW_BLOCKS", "trim_observation_envelope", "trim_view_freshness"]
