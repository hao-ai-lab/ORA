"""Static facts once per round, in a protected slot (``round_static_facts_slot_v1``).

Decided 2026-10-04, amending the 2026-09-10 rule that every Native observation
carries the complete current calibration and robot facts: fixed-camera
calibration and the static robot facts go once per round into a protected
message, dynamic content (a wrist camera's calibration while its arm moves) is
still sent with every observation, and a changed value is always sent inline.

``native_slots_v4`` is ``native_slots_v3`` with this capability (and
``tool_image_call_labels_v1``, which only names a tool image's call).

* **Slot.** The first observation of a Session round captures every
  ``current_actual`` field except the identity keys (:data:`IDENTITY_KEYS`)
  into one user message appended right after that observation. It is appended
  once and never edited, so the request stays a strict extension of the last.
* **Observations.** Each observation of the round, the first included, omits a
  field whose delivered value is byte-identical to the slot's and lists it in
  ``current_actual.static_facts.unchanged``; every other field is inline. The
  split is by value, never by a producer flag such as ``calibration_dynamic``
  (LIBERO does not set it). A field is compared as a whole first; a mapping that
  differs is compared key by key, so a moving wrist camera keeps its unchanged
  intrinsics in the slot. A field is listed only when naming it is shorter than
  restating it, and only under keys that are plain identifiers, so every path
  in ``unchanged`` is unambiguous.
* **Absent listing.** On a surface declaring ``observation_delivery_trim_v2``
  the reduction omits every byte-identical field without naming it, carries
  the rest, and lists under ``static_facts.absent`` only the slot fields that
  have no current value (the slot's rule says so: :data:`ABSENT_RULE`). A
  mapping with a key that is not a plain identifier is carried whole.
* **Aging.** The slot is ``_protected_context``. Tail aging (L1-L4) never
  rewrites the current round's slot. The previous round's slot is externalized
  to a recoverable pointer when its round's media are released (L3, at reset),
  and an emergency checkpoint (L6) that absorbs the round's first step carries
  the current round's slot forward. L5 capsules cover closed rounds only.

:func:`restore` rebuilds the full ``current_actual`` from an observation and
its slot; it is the check that nothing is lost.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from ora.agent.contracts import PolicySessionError
from ora.agent.memory.artifacts import recoverable_artifact_pointer

CAPABILITY = "round_static_facts_slot_v1"
COMPOSITION = "native_slots_v4"
STATIC_FACTS_KEY = "static_facts"
SLOT_KEY = "round_static_facts"
MESSAGE_FLAG = "_round_static_facts"
REF_KEY = "_round_static_facts_ref"
ORIGINAL_KEY = "_round_static_facts_original"
EXTERNALIZED_KEY = "_lossless_round_static_facts_externalized"
ARTIFACT_KIND = "context_round_static_facts"

#: ``current_actual`` keys that identify the observation and are never moved.
IDENTITY_KEYS = frozenset({"actual_sequence", "observation_step", "step", STATIC_FACTS_KEY})

RULE = (
    "An observation in this round omits each current_actual field it lists in "
    "static_facts.unchanged; that field's current value is byte-identical to its "
    "value here. A field the observation carries inline is the current value; a "
    "value here that the observation does not list is the round-start value."
)

#: The two listings. ``unchanged`` names each omitted field; ``absent``
#: (``observation_delivery_trim_v2``) omits every byte-identical field without
#: naming it and names only the slot fields that have no current value.
LISTING_UNCHANGED = "unchanged"
LISTING_ABSENT = "absent"
ABSENT_RULE = (
    "An observation in this round leaves out each current_actual field whose current "
    "value is byte-identical to its value here. A field it carries is the current "
    "value; a mapping it carries replaces only the keys it names, unless one of its "
    "keys is not a plain identifier, in which case it is the whole current value. "
    "static_facts.absent lists each field here that has no current value."
)
RULES = {LISTING_UNCHANGED: RULE, LISTING_ABSENT: ABSENT_RULE}

_SEGMENT = re.compile(r"[A-Za-z0-9_]+")
_MISSING = object()
_OMIT = object()


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def capture(current_actual: Mapping[str, Any]) -> dict[str, Any]:
    """The slot's fields: a copy of ``current_actual`` without its identity keys."""
    return {key: copy.deepcopy(value) for key, value in current_actual.items()
            if key not in IDENTITY_KEYS}


def slot_message(*, round_index: int, actual_sequence: int,
                 fields: Mapping[str, Any], listing: str = LISTING_UNCHANGED) -> dict[str, Any]:
    """The protected message that states one round's facts."""
    body = {SLOT_KEY: {
        "round": round_index, "captured_at_actual_sequence": actual_sequence,
        "rule": RULES[listing], "current_actual": copy.deepcopy(dict(fields)),
    }}
    return {
        "role": "user",
        "content": [{"type": "text", "text": _dumps(body)}],
        "_protected_context": True,
        "_context_source": f"round_static_facts_{round_index}",
        MESSAGE_FLAG: {"round": round_index},
    }


def _worth_naming(path: str, value: Any) -> bool:
    """Whether omitting ``"key": value`` saves more bytes than listing ``path``."""
    key = path.rsplit(".", 1)[-1]
    saved = len(_dumps(key).encode()) + 2 + len(_dumps(value).encode()) + 2
    return saved > len(_dumps(path).encode()) + 2


def _reduce(value: Any, base: Any, path: str, unchanged: list[str]) -> Any:
    if base is not _MISSING and _dumps(value) == _dumps(base) and _worth_naming(path, value):
        unchanged.append(path)
        return _OMIT
    if isinstance(value, Mapping) and isinstance(base, Mapping):
        result = {}
        for key, child in value.items():
            reduced = (_reduce(child, base.get(key, _MISSING), f"{path}.{key}", unchanged)
                       if _SEGMENT.fullmatch(str(key)) else copy.deepcopy(child))
            if reduced is not _OMIT:
                result[key] = reduced
        return result
    return copy.deepcopy(value)


def _identifier_keys(*mappings: Mapping[str, Any]) -> bool:
    return all(_SEGMENT.fullmatch(str(key)) for mapping in mappings for key in mapping)


def _reduce_absent(value: Any, base: Any, path: str, absent: list[str]) -> Any:
    if base is not _MISSING and _dumps(value) == _dumps(base):
        return _OMIT
    if isinstance(value, Mapping) and isinstance(base, Mapping) and _identifier_keys(value, base):
        result = {}
        for key, child in value.items():
            reduced = _reduce_absent(child, base.get(key, _MISSING), f"{path}.{key}", absent)
            if reduced is not _OMIT:
                result[key] = reduced
        absent.extend(f"{path}.{key}" for key in base if key not in value)
        # Everything left is in the slot or in ``absent``.
        return result if result else _OMIT
    return copy.deepcopy(value)


def _reduce_listing_absent(current_actual: Mapping[str, Any], fields: Mapping[str, Any], *,
                           round_index: int) -> dict[str, Any]:
    absent: list[str] = []
    result: dict[str, Any] = {}
    for key, value in current_actual.items():
        if key in IDENTITY_KEYS or not _SEGMENT.fullmatch(str(key)):
            result[key] = copy.deepcopy(value)
            continue
        reduced = _reduce_absent(value, fields.get(key, _MISSING), key, absent)
        if reduced is not _OMIT:
            result[key] = reduced
    absent.extend(key for key in fields
                  if key not in current_actual and key not in IDENTITY_KEYS
                  and _SEGMENT.fullmatch(str(key)))
    listing: dict[str, Any] = {"round_slot": round_index}
    if absent:
        listing[LISTING_ABSENT] = sorted(absent)
    result[STATIC_FACTS_KEY] = listing
    return result


def reduce(current_actual: Mapping[str, Any], fields: Mapping[str, Any], *,
           round_index: int, listing: str = LISTING_UNCHANGED) -> dict[str, Any]:
    """``current_actual`` without the fields the round's slot already states."""
    if listing == LISTING_ABSENT:
        return _reduce_listing_absent(current_actual, fields, round_index=round_index)
    if listing != LISTING_UNCHANGED:
        raise PolicySessionError(f"unknown round static-facts listing {listing!r}")
    unchanged: list[str] = []
    result: dict[str, Any] = {}
    for key, value in current_actual.items():
        if key in IDENTITY_KEYS or not _SEGMENT.fullmatch(str(key)):
            result[key] = copy.deepcopy(value)
            continue
        reduced = _reduce(value, fields.get(key, _MISSING), key, unchanged)
        if reduced is not _OMIT:
            result[key] = reduced
    result[STATIC_FACTS_KEY] = {"round_slot": round_index, "unchanged": sorted(unchanged)}
    return result


def _merge(base: Any, value: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(value, Mapping) and _identifier_keys(base, value):
        merged = dict(base)
        for key, child in value.items():
            merged[key] = _merge(base[key], child) if key in base else copy.deepcopy(child)
        return merged
    return copy.deepcopy(value)


def _restore_absent(current_actual: Mapping[str, Any], fields: Mapping[str, Any],
                    absent: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(dict(fields))
    for key, value in current_actual.items():
        if key == STATIC_FACTS_KEY:
            continue
        result[key] = (_merge(result[key], value)
                       if key in result and key not in IDENTITY_KEYS and _SEGMENT.fullmatch(str(key))
                       else copy.deepcopy(value))
    for path in absent:
        *parents, leaf = path.split(".")
        target: Any = result
        for step in parents:
            target = target[step]
        del target[leaf]
    return result


def restore(current_actual: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """The full ``current_actual``, under whichever listing the observation carries.

    ``unchanged``: each listed field put back from the slot. ``absent``: the
    slot, with every field the observation carries merged over it and each
    absent field removed.
    """
    listing = current_actual.get(STATIC_FACTS_KEY)
    if isinstance(listing, Mapping) and LISTING_UNCHANGED not in listing:
        return _restore_absent(current_actual, fields, listing.get(LISTING_ABSENT, ()))
    result = copy.deepcopy(dict(current_actual))
    listing = result.pop(STATIC_FACTS_KEY, None)
    for path in (listing or {}).get("unchanged", ()):
        steps = path.split(".")
        source: Any = fields
        for step in steps:
            source = source[step]
        target = result
        for step in steps[:-1]:
            target = target.setdefault(step, {})
        target[steps[-1]] = copy.deepcopy(source)
    return result


# -- Context lifecycle ---------------------------------------------------------


def is_slot(message: Mapping[str, Any]) -> bool:
    return isinstance(message.get(MESSAGE_FLAG), Mapping)


def slot_round(message: Mapping[str, Any]) -> Any:
    return message[MESSAGE_FLAG].get("round") if is_slot(message) else None


def materialize(message: dict[str, Any], *, catalog: Any, round_index: int | None,
                action_step: int | None) -> dict[str, Any] | None:
    """Store a slot's exact text as a recoverable Artifact (lossless aging)."""
    if not is_slot(message):
        return None
    content = message.get("content")
    if not (isinstance(content, list) and len(content) == 1 and isinstance(content[0], Mapping)
            and isinstance(content[0].get("text"), str)):
        raise PolicySessionError("round static facts slot is malformed")
    original = content[0]["text"]
    ref = catalog.put_bytes(
        original.encode("utf-8"), kind=ARTIFACT_KIND, suffix=".json",
        media_type="application/json", round_index=round_index, action_step=action_step,
    )
    message[REF_KEY] = ref
    message[ORIGINAL_KEY] = original
    return ref


def _marker(message: Mapping[str, Any]) -> str:
    return json.dumps({
        SLOT_KEY: {"round": slot_round(message)},
        "recoverable_round_static_facts": recoverable_artifact_pointer(message[REF_KEY]),
        "round_static_facts_externalized": True,
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _current_round(messages: Sequence[Mapping[str, Any]]) -> Any:
    for message in reversed(messages):
        if message.get("role") == "user" and isinstance(message.get("_observation"), Mapping):
            return message.get("_session_round")
    return None


def externalize_prior_rounds(
    messages: list[dict[str, Any]], *, through_action_steps: int,
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    """L3: a slot of an earlier round whose step's media aged becomes a pointer.

    The current round's slot is never rewritten, whatever the media frontier.
    Copy-on-write, as in the other lossless passes.
    """
    output = list(messages)
    current_round = _current_round(output)
    changed: list[int] = []
    refs: list[dict[str, Any]] = []
    action_step = -1
    for index, message in enumerate(output):
        if message.get("role") == "user" and isinstance(message.get("_observation"), Mapping):
            action_step += 1
        if (not is_slot(message) or not 0 <= action_step < through_action_steps
                or slot_round(message) == current_round
                or not isinstance(message.get(REF_KEY), Mapping)):
            continue
        replaced = dict(message)
        replaced["content"] = [{"type": "text", "text": _marker(message)}]
        replaced[EXTERNALIZED_KEY] = True
        output[index] = replaced
        changed.append(index)
        refs.append(copy.deepcopy(dict(message[REF_KEY])))
    return output, changed, refs


def invariant_content(message: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """A slot's original content, after checking its current text is exact."""
    ref = message.get(REF_KEY)
    if not is_slot(message) or not isinstance(ref, Mapping):
        return None
    original = message.get(ORIGINAL_KEY)
    if not isinstance(original, str) or hashlib.sha256(
        original.encode("utf-8")
    ).hexdigest() != ref.get("sha256") or len(original.encode("utf-8")) != ref.get("bytes"):
        raise PolicySessionError("round static facts Artifact identity drift")
    expected = _marker(message) if message.get(EXTERNALIZED_KEY) is True else original
    if message.get("content") != [{"type": "text", "text": expected}]:
        raise PolicySessionError("round static facts differ from their recoverable source")
    return [{"type": "text", "text": original}]


def current_round_slots(
    absorbed: Sequence[Mapping[str, Any]], *, current_round: Any,
) -> list[dict[str, Any]]:
    """L6: the current round's slot, carried out of a summarized span."""
    return [copy.deepcopy(dict(message)) for message in absorbed
            if is_slot(message) and slot_round(message) == current_round]


__all__ = [
    "ABSENT_RULE", "ARTIFACT_KIND", "CAPABILITY", "COMPOSITION", "IDENTITY_KEYS", "LISTING_ABSENT",
    "LISTING_UNCHANGED", "MESSAGE_FLAG", "RULE", "RULES",
    "SLOT_KEY", "STATIC_FACTS_KEY", "capture", "current_round_slots", "externalize_prior_rounds",
    "invariant_content", "is_slot", "materialize", "reduce", "restore", "slot_message",
    "slot_round",
]
