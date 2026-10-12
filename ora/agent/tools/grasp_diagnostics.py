# Copyright contributors to the Robot Harness Runtime project.
#
# The policy-visible diagnostics of a multi-family grasp backend.
#
# A single-family backend returns the service's own diagnostics and the shared
# materializer projects them. A backend that runs two families has to decide
# what one set of diagnostics means, and the first attempt decided it by
# building a fresh dictionary: the request identity and the rejection ledger
# ended up nested under ``learned`` and three keys appeared that the result
# contract forbade. Every RoboTwin ``grasp_plan`` call then failed its own
# schema with a non-retryable error, after the backend had already compiled
# candidates -- the geometry was found and thrown away at the format gate.
#
# This module holds that decision in one place so the next family added to a
# proposer has somewhere to be added, and so the assembly can be tested
# against the resolved result schema without standing a backend up.
"""Compose one grasp backend's diagnostics from its per-family parts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

_WARNING = (
    "all learned and public-RGB-D candidates still require motion "
    "probing and visual review; none proves contact or retention"
)


#: A family's own diagnostics, minus the one key the union above already holds.
#: Keeping the per-family ledgers as well made the block unbounded -- the result
#: schema had to declare both as bare objects, which is the failure mode the
#: authoring guide names -- and shipped the same rejection twice in one result.
def _family(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in diagnostics.items() if key != "rejected"}


def hybrid_grasp_diagnostics(
    *,
    learned_diagnostics: Mapping[str, Any],
    geometric_diagnostics: Mapping[str, Any],
    learned: Sequence[Any],
    geometric: Sequence[Any],
    selected: Sequence[Any],
) -> dict[str, Any]:
    """The two families' diagnostics under the shape the contract declares.

    ``request_id`` and ``rejected`` belong at the top level, where a
    single-family backend puts them, because that is where the result schema
    requires them and where a policy reads them. The ledger is the union: a
    candidate either family rejected is a candidate this call considered and
    did not return, and the reason it went is the ledger's whole purpose.

    Each family's own diagnostics stay, named for the family. They are a
    computed quantity and this surface does not withhold those; the counts
    say how many each family offered before selection, which no consumer can
    recover from the returned candidates alone.
    """

    rejected: list[Any] = []
    for source in (learned_diagnostics, geometric_diagnostics):
        entries = source.get("rejected")
        if isinstance(entries, list):
            rejected.extend(entries)
    return {
        "model": dict(learned_diagnostics.get("model", {})),
        "request_id": str(learned_diagnostics.get("request_id") or ""),
        "rejected": rejected,
        "returned_candidate_count": len(selected),
        "learned": _family(learned_diagnostics),
        "public_rgbd_topdown": _family(geometric_diagnostics),
        "candidate_family_counts": {
            "learned_available": len(learned),
            "public_rgbd_topdown_available": len(geometric),
            "selected": len(selected),
        },
        "warning": _WARNING,
    }


__all__ = ["hybrid_grasp_diagnostics"]
