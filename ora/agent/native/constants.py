"""Stable Native protocol identifiers shared by the session and TurnLoop."""

MUTATION_TOOL = "robot.submit_action"
NATIVE_SLOTS_V3 = "native_slots_v3"
#: ``native_slots_v3`` plus ``round_static_facts_slot_v1``
#: (``agent/context/round_static_facts.py``) and ``tool_image_call_labels_v1``.
NATIVE_SLOTS_V4 = "native_slots_v4"
#: Compositions with the ``native_slots_v3`` causal slot layout.
NATIVE_CAUSAL_SLOTS = frozenset({NATIVE_SLOTS_V3, NATIVE_SLOTS_V4})
#: A tool-returned image message names its tool and call id beside its kind
#: (the bare causal-slot label left parallel calls ambiguous).
TOOL_IMAGE_CALL_LABELS = "tool_image_call_labels_v1"
#: The Context capabilities each composition adds to ``native_slots_v3``.
COMPOSITION_CONTEXT_CAPABILITIES = {
    NATIVE_SLOTS_V4: ("round_static_facts_slot_v1", TOOL_IMAGE_CALL_LABELS),
}
