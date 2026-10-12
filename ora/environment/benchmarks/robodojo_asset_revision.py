"""Reviewed task-specific asset selection; host-side, no simulator imports.

2026-10-06: the user authorized the upstream replacement swap_T layouts.
All other Assets blobs are identical. Old task records and installations stay
intact; only fresh swap_T runs select the successor asset checkout.
"""
from __future__ import annotations

OLD_ASSET_REVISION = "a14409d7fae673c00499e01fd88b4457df6351b1"
SWAP_T_ASSET_REVISION = "35efbc7dedfdbeeb6e95fb749bd885d73d483e41"
SWAP_T_ASSET_ROOT_ENV = "ROBODOJO_SWAP_T_ASSET_ROOT"


def asset_revision_for(task: str) -> str:
    return SWAP_T_ASSET_REVISION if task == "swap_T" else OLD_ASSET_REVISION


def unchanged_card_assets(original: dict, current: dict, *, measured_task: str) -> bool:
    """The one reviewed revision transition; it cannot reuse a swap_T card.

The exact Git tree comparison is recorded in the v8 compatibility review:
165 changed blobs, exclusively swap_T layouts, no added/deleted Asset paths.
The card's measured task must therefore be different. Scene/layout reuse is
separately required by revalidation and is never granted by this predicate.
"""
    if original == current:
        return True
    from .robodojo import NATIVE_ACTION_LIMITS
    return (measured_task in NATIVE_ACTION_LIMITS and measured_task != "swap_T"
            and original.get("assets") == OLD_ASSET_REVISION
            and current.get("assets") == SWAP_T_ASSET_REVISION
            and {k: v for k, v in original.items() if k != "assets"}
            == {k: v for k, v in current.items() if k != "assets"})
