"""Choose which compiled grasp candidates the display cap shows.

``materialize_grasp_query`` caps the backend's candidates at ``max_candidates``
before compiling them unless a surface passes a ``candidate_selection``
(``{"identity": {..., "pool_limit": n}, "select": callable}``). Then the backend
is asked for a pool of up to ``pool_limit``, every pool candidate is compiled,
``select`` annotates and orders the whole pool, and only then is the display cap
applied here. Nothing below the cap is dropped except by the cap.

Kept out of ``grasp.py`` (past its size ratchet) on purpose: the query module
only threads the selection through; the ordering contract lives here and in
``reach_check.select_grasp_candidates``.
"""
from __future__ import annotations

from typing import Any, Mapping

SCHEMA = "ora.grasp-candidate-selection.v1"


def apply_candidate_selection(
    candidates: list[dict[str, Any]], compiled_candidates: list[dict[str, Any]],
    selection: Mapping[str, Any] | None, max_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any] | None]:
    """Order the whole compiled pool, then cap it.

    ``rank`` stays the backend's preference, renumbered 0..k-1 among the
    displayed candidates, so overlay labels and the ``rank`` field keep their
    released meaning; the list order carries the selection. The record is
    private manifest evidence; its ``summary`` is what the result publishes.
    """
    if selection is None:
        return candidates, compiled_candidates, None
    chosen = selection["select"]([dict(item) for item in compiled_candidates])
    ordered = [dict(item) for item in chosen["ordered"]]
    if sorted(str(item["candidate_id"]) for item in ordered) != sorted(
        str(item["candidate_id"]) for item in compiled_candidates
    ):
        raise ValueError("grasp candidate selection must keep every pool candidate")
    shown = ordered[:max_candidates]
    raw_by_id = {str(item["candidate_id"]): item for item in candidates}
    backend_rank = {str(item["candidate_id"]): int(item["rank"]) for item in compiled_candidates}
    by_backend = sorted((str(item["candidate_id"]) for item in shown), key=backend_rank.__getitem__)
    rank = {candidate_id: index for index, candidate_id in enumerate(by_backend)}
    displayed_raw = [{**raw_by_id[str(item["candidate_id"])], "rank": rank[str(item["candidate_id"])]}
                     for item in shown]
    displayed = [{**item, "rank": rank[str(item["candidate_id"])]} for item in shown]
    record = {
        "schema": SCHEMA,
        "identity": dict(selection["identity"]),
        "summary": dict(chosen["summary"]),
        "pool": [{"candidate_id": item["candidate_id"],
                  "backend_rank": backend_rank[str(item["candidate_id"])],
                  "ik_reachability": item.get("ik_reachability")} for item in ordered],
        "displayed_candidate_ids": [item["candidate_id"] for item in displayed],
        "evidence": dict(chosen.get("evidence", {})),
    }
    return displayed_raw, displayed, record
