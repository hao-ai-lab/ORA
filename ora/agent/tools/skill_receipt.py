"""Staged receipts for the closed manipulation skills, inside both ceilings.

A closed skill runs several world-changing stages and reports once. That report
is the only place a later reader — a person auditing the run, or a student being
trained on the trajectory — can see what happened between the decision and the
outcome. It is also the single largest tool result the surface produces, so it
has to be built against the two limits the sealed run configuration sets rather
than assembled optimistically and truncated afterwards.

Delivery ceilings, as measured rather than as assumed
-----------------------------------------------------

**Bytes.** ``eager_large_result_bytes`` is 50,000 in the sealed EXP08A
configuration. It is compared against
``len(json.dumps(visible, ensure_ascii=False, sort_keys=True).encode("utf-8"))``
where ``visible`` is the result with ``image_base64``/``images_base64`` and
``_runtime`` removed (``ora/agent/native/session.py``). So **image
payloads do not count** toward this ceiling; only the JSON body does.

Above the ceiling the sealed ``action_step_lossless_v4`` retention replaces the
body with a recoverable-artifact pointer plus a 2,048-byte preview
(``eager_preview_bytes``). It does *not* tell the model to repeat the call —
that wording belongs to the legacy ``_spill_large_tool_results`` path, which is
unreachable under this configuration. Either way a receipt that trips the
ceiling is worse than a smaller one, because the stage detail a reader needs is
exactly what the preview cuts.

``budget_bytes`` therefore bounds the receipt by construction, and
``assemble`` elides the oldest stages' measurements first if a receipt would
still exceed it, recording that it did so. Nothing is dropped silently.

**Images.** L3 keeps every image in the open ActionStep and the configured
completed-step tail of the current Episode. Older current-Episode and all
prior-Episode images become exactly recoverable pointers. The maintained value
keeps two completed visual steps plus the open step. ``trigger_inline_images``
is a soft L4 pressure trigger of 24, not an image-retention target. L4 may act
on remaining read-only bodies under pressure. The independent per-block window
(``max_inline_images_per_request``) remains refused for this retention.

The per-image binding is kept anyway, and is not defensive clutter: it is the
only correct convention under the retentions where the per-block window *is*
live, and position-based attribution silently misattributes survivors there. So
every ``image_index`` names its own source, and the skill contributes at most
``MAX_RECEIPT_IMAGES`` of the 24.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .robot import ToolError


SCHEMA = "ora.skill-receipt.v1"

#: The sealed ``eager_large_result_bytes``. A receipt is built to stay under it.
DELIVERY_CEILING_BYTES = 50_000

#: Headroom under the ceiling. The receipt is one field of the tool result, and
#: the caller adds observation identity and its own claim boundary around it.
RECEIPT_BUDGET_BYTES = 40_000

#: A staged skill contributes at most this many current-ActionStep images.
MAX_RECEIPT_IMAGES = 4

#: Per-stage measurement cap. ``measurements`` entries are four-key objects, so
#: this bounds the dominant term in the receipt's size.
MAX_STAGE_MEASUREMENTS = 24

#: The most stages any closed skill declares (place_execute's six).
MAX_STAGES = 6

#: The rationale is the only channel through which a student reading this
#: trajectory learns *why* a candidate was chosen: the provider's reasoning is
#: not in the trajectory. It is mandatory and bounded.
MAX_RATIONALE_CHARS = 200

STAGE_STATUSES = frozenset({
    "executed", "stopped", "refused", "not_reached", "unavailable", "skipped", "terminated",
})

TERMINAL_STATUSES = frozenset({"completed", "stopped", "aborted", "terminated"})


def validate_rationale(value: Any) -> str:
    """Require a short, non-empty reason for a world-changing skill call.

    Every mutation tool on the redesigned surface takes one. It is a first-class
    output rather than a formality: the trajectory records the tool call and its
    result, never the provider's reasoning, so without this field a reader can
    see *what* was chosen and never *why*.
    """

    if not isinstance(value, str) or not value.strip():
        raise ToolError(
            "rationale is required: state briefly why this candidate was chosen",
            code="invalid_arguments", field="rationale",
        )
    text = value.strip()
    if len(text) > MAX_RATIONALE_CHARS:
        raise ToolError(
            f"rationale must be at most {MAX_RATIONALE_CHARS} characters",
            code="invalid_arguments", field="rationale",
            received=len(text), constraints={"maximum": MAX_RATIONALE_CHARS},
        )
    return text


def stage(
    name: str, status: str, reason: str, *,
    action_id: str | None = None,
    measurements: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """One executed stage of a closed skill, with its own outcome and evidence."""

    if not isinstance(name, str) or not name:
        raise ToolError("a stage needs a name", code="internal_tool_error", retryable=False)
    if status not in STAGE_STATUSES:
        raise ToolError(
            f"unknown stage status: {status}", code="internal_tool_error", retryable=False,
        )
    items = [dict(item) for item in measurements]
    elided = max(0, len(items) - MAX_STAGE_MEASUREMENTS)
    return {
        "stage": name,
        "status": status,
        "reason": reason,
        "action_id": action_id,
        "measurements": items[:MAX_STAGE_MEASUREMENTS],
        "measurements_elided": elided,
    }


def image_bindings(sources: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Bind each ``image_index`` to its own source, never to its position.

    Under the retentions where the per-block inline-image window is live, images
    are dropped one block at a time and survivors are not renumbered, so "the
    i-th image belongs to the i-th stage" misattributes them. The binding is
    carried in the result body, which is a different message from the image
    blocks and is not touched by that window.
    """

    from .perception_feedback import image_label

    if len(sources) > MAX_RECEIPT_IMAGES:
        raise ToolError(
            f"a skill receipt carries at most {MAX_RECEIPT_IMAGES} images",
            code="internal_tool_error", retryable=False,
        )
    bindings = []
    for index, source in enumerate(sources):
        bindings.append(image_label(
            index, str(source["role"]),
            camera=str(source.get("camera") or ""),
            target_ref=str(source.get("target_ref") or ""),
            candidate_id=str(source.get("candidate_id") or ""),
            step=source.get("observation_step"),
        ))
    return bindings


def body_bytes(result: Mapping[str, Any]) -> int:
    """Measure a result the way the delivery ceiling measures it.

    Image payloads and ``_runtime`` are removed before the tool message is
    built, so they are excluded here too. Anything else is a size claim about a
    different object than the one the threshold sees.
    """

    visible = {
        key: value for key, value in result.items()
        if key not in {"image_base64", "images_base64", "_runtime"}
    }
    return len(json.dumps(visible, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def assemble(
    *, skill: str, status: str, rationale: str, candidate_id: str | None,
    observation_id: str, observation_sha256: str,
    stages: Sequence[Mapping[str, Any]],
    verification: Mapping[str, Any] | None = None,
    held_object_offset: Mapping[str, Any] | None = None,
    images: Sequence[Mapping[str, Any]] = (),
    claim_boundary: str,
    budget_bytes: int = RECEIPT_BUDGET_BYTES,
) -> dict[str, Any]:
    """Build one staged receipt that fits the byte budget by construction.

    If the assembled receipt would still exceed ``budget_bytes``, the oldest
    stages' measurements are dropped first — the newest stage is the one whose
    numbers explain the outcome — and each drop is recorded in that stage's
    ``measurements_elided``. A receipt never silently trips the ceiling, and it
    never quietly loses evidence either.
    """

    if status not in TERMINAL_STATUSES:
        raise ToolError(
            f"unknown skill status: {status}", code="internal_tool_error", retryable=False,
        )
    if len(stages) > MAX_STAGES:
        raise ToolError(
            f"a closed skill declares at most {MAX_STAGES} stages",
            code="internal_tool_error", retryable=False,
        )
    receipt = {
        "schema": SCHEMA,
        "skill": skill,
        "status": status,
        # Echoed so the reason travels with the outcome it explains.
        "rationale": validate_rationale(rationale),
        "candidate_id": candidate_id,
        "observation_id": observation_id,
        "observation_sha256": observation_sha256,
        "stages": [dict(item) for item in stages],
        "verification": dict(verification) if verification is not None else None,
        "held_object_offset": dict(held_object_offset) if held_object_offset is not None else None,
        "image_mapping": image_bindings(images),
        "delivery": {
            "ceiling_bytes": DELIVERY_CEILING_BYTES,
            "budget_bytes": int(budget_bytes),
            "body_bytes": 0,
            "stages_with_elided_measurements": 0,
        },
        "claim_boundary": claim_boundary,
    }
    for index in range(len(receipt["stages"])):
        if body_bytes(receipt) <= budget_bytes:
            break
        item = receipt["stages"][index]
        dropped = len(item["measurements"])
        if not dropped:
            continue
        item["measurements"] = []
        item["measurements_elided"] = item.get("measurements_elided", 0) + dropped
        receipt["delivery"]["stages_with_elided_measurements"] += 1
    # Recording the size changes the size: the field grows from one digit to
    # five. Iterate to a fixed point so the reported number is the size of the
    # receipt that actually carries it, and settle on the larger value at a
    # digit boundary rather than oscillating.
    for _ in range(4):
        measured = body_bytes(receipt)
        if receipt["delivery"]["body_bytes"] == measured:
            break
        receipt["delivery"]["body_bytes"] = max(measured, receipt["delivery"]["body_bytes"])
    return receipt
