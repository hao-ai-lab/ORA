"""Closed configuration contracts for recoverable-first Native Context v4-v6."""

from __future__ import annotations

import copy
from typing import Any, Callable, Mapping


CONTEXT_MANAGEMENT_V4_SCHEMA = "ora.native-context-management.v4"
CONTEXT_MANAGEMENT_V5_SCHEMA = "ora.native-context-management.v5"
CONTEXT_MANAGEMENT_V6_SCHEMA = "ora.native-context-management.v6"
LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS = frozenset({
    CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_SCHEMA,
    CONTEXT_MANAGEMENT_V6_SCHEMA,
})
#: Schemas whose v4 tails are minimums that age in strides (v5 onward).
TAIL_AGING_SCHEMAS = frozenset({
    CONTEXT_MANAGEMENT_V5_SCHEMA, CONTEXT_MANAGEMENT_V6_SCHEMA,
})
#: Schemas whose L5 watermark is max(tokens, ratio x hard window). Maintained
#: configurations no longer select them; HEAD keeps the formula so recorded
#: v4/v5 runs, frozen archives and explicit historical treatments keep the
#: exact threshold they ran with.
WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS = frozenset({
    CONTEXT_MANAGEMENT_V4_SCHEMA, CONTEXT_MANAGEMENT_V5_SCHEMA,
})
TAIL_AGING_STRIDE_FIELD = "tail_aging_stride_action_steps"
EPISODE_TRIGGER_RATIO_FIELD = "episode_checkpoint_trigger_ratio"

# This ledger is the machine-readable owner of the v4 field set and field-level
# semantics. Callers own explicit treatment values; the lifecycle reference
# renders the reviewed field meanings and one historical value set.
CONTEXT_MANAGEMENT_V4_FIELD_SPECS: dict[str, dict[str, str]] = {
    "schema": {
        "layer": "Contract",
        "phase": "RunSpec and pipeline construction",
        "authority": "exact literal",
        "kind": "schema",
        "meaning": "Select the closed v4 configuration and reject mixed versions.",
    },
    "causal_exact_tail_action_steps": {
        "layer": "Guard",
        "phase": "every projection",
        "authority": "Harness ActionStep inventory",
        "kind": "positive_int",
        "meaning": "Completed ActionSteps whose causal and read-only bodies stay exact through L1-L6; media has its own current-Episode tail.",
    },
    "inline_media_tail_action_steps": {
        "layer": "L3",
        "phase": "every media lifecycle projection",
        "authority": "Harness ActionStep and Session-round inventory",
        "kind": "nonnegative_int",
        "meaning": "Completed steps in the current Episode whose media remains inline; earlier and prior-Episode media becomes recoverable Pointer-only.",
    },
    "eager_large_result_bytes": {
        "layer": "L1",
        "phase": "after a result leaves the protected near window",
        "authority": "exact UTF-8 byte length",
        "kind": "positive_int",
        "meaning": "Strict byte threshold for eager result externalization.",
    },
    "eager_preview_bytes": {
        "layer": "L1/L4",
        "phase": "recoverable result replacement",
        "authority": "exact UTF-8 byte slice",
        "kind": "positive_int",
        "meaning": "Maximum inline preview bytes retained beside a Pointer.",
    },
    "eager_exact_deduplication": {
        "layer": "L2",
        "phase": "after repeated results leave the protected near window",
        "authority": "content SHA-256 and exact bytes",
        "kind": "required_true",
        "meaning": "Require beneficial byte-exact duplicate replacement.",
    },
    "trigger_inline_images": {
        "layer": "L4",
        "phase": "post-L3 pressure admission",
        "authority": "Harness inline image count",
        "kind": "positive_int",
        "meaning": "Run L4 when images remaining after the fixed current-Episode media tail are strictly above this soft threshold.",
    },
    "trigger_estimated_tokens": {
        "layer": "L4",
        "phase": "post-L3 pressure admission",
        "authority": "model-dependent Harness estimate",
        "kind": "positive_int",
        "meaning": "Run L4 when estimated request tokens are strictly above this value.",
    },
    "trigger_provider_input_ratio": {
        "layer": "L4",
        "phase": "post-L3 pressure admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Run L4 when exact Provider input reaches this hard-window ratio.",
    },
    "episode_checkpoint_trigger_tokens": {
        "layer": "L5",
        "phase": "closed-Episode admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "positive_int",
        "meaning": "Absolute lower bound in the L5 max(tokens, ratio) watermark.",
    },
    "episode_checkpoint_trigger_ratio": {
        "layer": "L5",
        "phase": "closed-Episode admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Hard-window lower bound in the L5 max(tokens, ratio) watermark.",
    },
    "emergency_fallback_ratio": {
        "layer": "L6",
        "phase": "post-L5 emergency admission",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Admit L6 only at or above this hard-window ratio.",
    },
    "emergency_target_ratio": {
        "layer": "L6",
        "phase": "next Provider report after L6",
        "authority": "same-epoch Provider input tokens",
        "kind": "ratio",
        "meaning": "Nonterminal post-L6 request-size target.",
    },
    "checkpoint_target_estimated_tokens": {
        "layer": "L5/L6",
        "phase": "compactor generation",
        "authority": "compactor output target",
        "kind": "positive_int",
        "meaning": "Requested semantic checkpoint output size.",
    },
    "checkpoint_max_estimated_tokens": {
        "layer": "L5/L6",
        "phase": "checkpoint telemetry",
        "authority": "Provider output usage, else Harness estimate",
        "kind": "positive_int",
        "meaning": "Soft checkpoint-size maximum; exceeding it is nonterminal telemetry.",
    },
}

CONTEXT_MANAGEMENT_V4_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V4_FIELD_SPECS
)

# v5 is v4 plus one field: the v4 tails become minimums and older ActionSteps
# age in batches, so a Provider prefix cache sees one history rewrite per
# stride instead of one per ActionStep. The entries whose meaning changes are
# restated; every other v4 field keeps its exact v4 semantics.
CONTEXT_MANAGEMENT_V5_FIELD_SPECS: dict[str, dict[str, str]] = {
    **CONTEXT_MANAGEMENT_V4_FIELD_SPECS,
    "schema": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["schema"],
        "meaning": "Select the closed v5 configuration and reject mixed versions.",
    },
    "causal_exact_tail_action_steps": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["causal_exact_tail_action_steps"],
        "meaning": "Minimum completed ActionSteps whose causal and read-only bodies stay exact through L1-L6; older bodies age only at a stride event.",
    },
    "inline_media_tail_action_steps": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["inline_media_tail_action_steps"],
        "meaning": "Minimum completed current-Episode steps whose media stays inline; older media ages only at a stride event, and a reset releases prior-Episode media at once.",
    },
    "trigger_inline_images": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["trigger_inline_images"],
        "meaning": "Run L4 when images remaining after the minimum current-Episode media tail are strictly above this soft threshold; not evaluated while stride-deferred aging is pending.",
    },
    "trigger_estimated_tokens": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["trigger_estimated_tokens"],
        "meaning": "Run L4 when estimated request tokens are strictly above this value; not evaluated while stride-deferred aging is pending.",
    },
    "trigger_provider_input_ratio": {
        **CONTEXT_MANAGEMENT_V4_FIELD_SPECS["trigger_provider_input_ratio"],
        "meaning": "Run L4 when exact Provider input reaches this hard-window ratio; not evaluated while stride-deferred aging is pending.",
    },
    TAIL_AGING_STRIDE_FIELD: {
        "layer": "L1-L4",
        "phase": "every lifecycle projection",
        "authority": "Harness ActionStep and Session-round inventory",
        "kind": "positive_int",
        "meaning": "Age older ActionSteps only once this many are pending beyond a minimum tail, then all at once to exactly the minimum tails; 1 reproduces v4 byte for byte.",
    },
}

CONTEXT_MANAGEMENT_V5_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V5_FIELD_SPECS
)

# v6 is v5 with an absolute L5 watermark: the ratio
# field is gone, so L5 no longer scales with the bound window. Under v5 a 1M
# window put L5 at 0.60 x 1,030,144 = 618,086 tokens, where it effectively
# never fired. Every other v5 field keeps its exact v5 semantics.
CONTEXT_MANAGEMENT_V6_FIELD_SPECS: dict[str, dict[str, str]] = {
    **{
        field: spec for field, spec in CONTEXT_MANAGEMENT_V5_FIELD_SPECS.items()
        if field != EPISODE_TRIGGER_RATIO_FIELD
    },
    "schema": {
        **CONTEXT_MANAGEMENT_V5_FIELD_SPECS["schema"],
        "meaning": "Select the closed v6 configuration and reject mixed versions.",
    },
    "episode_checkpoint_trigger_tokens": {
        **CONTEXT_MANAGEMENT_V5_FIELD_SPECS["episode_checkpoint_trigger_tokens"],
        "meaning": "Absolute L5 watermark: admit one closed Episode Capsule once same-epoch Provider input reaches this many tokens, whatever the bound window.",
    },
}

CONTEXT_MANAGEMENT_V6_FIELDS = frozenset(
    CONTEXT_MANAGEMENT_V6_FIELD_SPECS
)


class ContextManagementConfigError(ValueError):
    """The closed v4-v6 Context configuration is incomplete or inconsistent."""


def validate_context_management_v4(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v4 object."""

    return _validate_closed(
        value, version="v4", schema=CONTEXT_MANAGEMENT_V4_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V4_FIELD_SPECS,
    )


def validate_context_management_v5(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v5 object."""

    return _validate_closed(
        value, version="v5", schema=CONTEXT_MANAGEMENT_V5_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V5_FIELD_SPECS,
    )


def validate_context_management_v6(value: Any) -> dict[str, Any]:
    """Validate and normalize the one closed Context Management v6 object."""

    return _validate_closed(
        value, version="v6", schema=CONTEXT_MANAGEMENT_V6_SCHEMA,
        specs=CONTEXT_MANAGEMENT_V6_FIELD_SPECS,
    )


_VALIDATORS: dict[str, Callable[[Any], dict[str, Any]]] = {
    CONTEXT_MANAGEMENT_V4_SCHEMA: validate_context_management_v4,
    CONTEXT_MANAGEMENT_V5_SCHEMA: validate_context_management_v5,
    CONTEXT_MANAGEMENT_V6_SCHEMA: validate_context_management_v6,
}


def validate_context_management(value: Any) -> dict[str, Any]:
    """Validate the lossless-retention object against the schema it names."""

    if not isinstance(value, Mapping):
        raise ContextManagementConfigError("context_management must be an object")
    schema = value.get("schema")
    validator = _VALIDATORS.get(schema) if isinstance(schema, str) else None
    if validator is None:
        raise ContextManagementConfigError("unsupported context_management schema")
    return validator(value)


def _validate_closed(
    value: Any, *, version: str, schema: str,
    specs: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    fields = frozenset(specs)
    if not isinstance(value, Mapping):
        raise ContextManagementConfigError(
            f"context_management {version} must be an object"
        )
    if set(value) != fields:
        missing = sorted(fields - set(value))
        extra = sorted(set(value) - fields)
        raise ContextManagementConfigError(
            f"context_management {version} fields drifted; "
            f"missing={missing}, extra={extra}"
        )
    if value.get("schema") != schema:
        raise ContextManagementConfigError(
            f"unsupported context_management {version} schema"
        )

    normalized = copy.deepcopy(dict(value))
    for field, spec in specs.items():
        kind = spec["kind"]
        item = normalized[field]
        if kind == "schema":
            continue
        if kind == "positive_int":
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be positive"
                )
        elif kind == "nonnegative_int":
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be non-negative"
                )
        elif kind == "required_true":
            if item is not True:
                raise ContextManagementConfigError(
                    f"context_management.{field} must be true"
                )
        elif kind == "ratio":
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not 0 < float(item) < 1
            ):
                raise ContextManagementConfigError(
                    f"context_management.{field} must be in (0,1)"
                )
            normalized[field] = float(item)
        else:  # pragma: no cover - the static ledger is reviewed with tests.
            raise ContextManagementConfigError(
                f"unsupported context_management field kind: {kind}"
            )

    if normalized["causal_exact_tail_action_steps"] != 3:
        raise ContextManagementConfigError(
            "context_management.causal_exact_tail_action_steps must equal 3"
        )
    if normalized["eager_preview_bytes"] >= normalized["eager_large_result_bytes"]:
        raise ContextManagementConfigError(
            "eager preview must be smaller than the large-result threshold"
        )
    if (
        normalized["checkpoint_target_estimated_tokens"]
        > normalized["checkpoint_max_estimated_tokens"]
    ):
        raise ContextManagementConfigError(
            "checkpoint target must not exceed its configured soft maximum"
        )
    _check_layer_order(normalized, version=version, schema=schema)
    return normalized


def _check_layer_order(
    normalized: Mapping[str, Any], *, version: str, schema: str,
) -> None:
    """Window-independent order of the L4, L5 and L6 watermarks.

    v4/v5 order the L5 ratio below the L6 target. v6 has no L5 ratio; its
    absolute L5 must lie above the absolute L4 estimate floor, which is the
    order at the 200K window both figures were chosen for. Above that window
    the order depends on the bound model (see ``thresholds``).
    """

    if schema in WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS:
        if not (
            normalized[EPISODE_TRIGGER_RATIO_FIELD]
            < normalized["emergency_target_ratio"]
            <= normalized["trigger_provider_input_ratio"]
            < normalized["emergency_fallback_ratio"]
        ):
            raise ContextManagementConfigError(
                f"Context {version} ratios require episode trigger < emergency target "
                "<= pressure trigger < emergency fallback"
            )
        return
    if not (
        normalized["emergency_target_ratio"]
        <= normalized["trigger_provider_input_ratio"]
        < normalized["emergency_fallback_ratio"]
    ):
        raise ContextManagementConfigError(
            f"Context {version} ratios require emergency target "
            "<= pressure trigger < emergency fallback"
        )
    if not (
        normalized["trigger_estimated_tokens"]
        < normalized["episode_checkpoint_trigger_tokens"]
    ):
        raise ContextManagementConfigError(
            f"Context {version} requires the L4 estimate floor "
            "(trigger_estimated_tokens) below the absolute L5 watermark "
            "(episode_checkpoint_trigger_tokens)"
        )


__all__ = [
    "CONTEXT_MANAGEMENT_V4_FIELDS",
    "CONTEXT_MANAGEMENT_V4_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V4_SCHEMA",
    "CONTEXT_MANAGEMENT_V5_FIELDS",
    "CONTEXT_MANAGEMENT_V5_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V5_SCHEMA",
    "CONTEXT_MANAGEMENT_V6_FIELDS",
    "CONTEXT_MANAGEMENT_V6_FIELD_SPECS",
    "CONTEXT_MANAGEMENT_V6_SCHEMA",
    "ContextManagementConfigError",
    "EPISODE_TRIGGER_RATIO_FIELD",
    "LOSSLESS_CONTEXT_MANAGEMENT_SCHEMAS",
    "TAIL_AGING_SCHEMAS",
    "TAIL_AGING_STRIDE_FIELD",
    "WINDOW_SCALED_EPISODE_TRIGGER_SCHEMAS",
    "validate_context_management",
    "validate_context_management_v4",
    "validate_context_management_v5",
    "validate_context_management_v6",
]
