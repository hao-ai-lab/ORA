"""Resolve observation-bound grasp/place candidates for committed skills."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import tool_names
from .robot import ToolError


def _advertised(host: Any, tool: str) -> str:
    """The spelling this host's surface shows the policy for one tool.

    These error messages direct a policy to another tool by name, which is the
    shape that misleads when the name is not the one the policy holds.
    `canonicalise_prose` rewrites help text and schemas at composition; a
    ToolError built at call time is not composed, so the spelling is resolved
    here instead. A surface that does not declare the tool gets the literal
    back, naming what the message meant.
    """

    surface = getattr(host, "surface", None)
    definitions = getattr(surface, "definitions", ()) if surface is not None else ()
    return tool_names.as_advertised(
        [str(item.get("name")) for item in definitions], tool,
    )


def _grasp_candidate(host: Any, candidate_id: str, observation_id: str) -> dict[str, Any]:
    """Find one compiled grasp candidate produced from the current observation.

    Candidate identities are content hashes of the proposal, so the lookup is
    unambiguous; what the scan enforces is the binding. A candidate compiled
    from an earlier observation describes geometry the robot has since moved
    through, and executing against it is the exact mistake a committed sequence
    must not be able to make quietly.

    An id ``grasp_plan`` published only as rejected, or kept in its pool but
    not displayed, is refused as such, with the ids that can execute now
    "no candidate carries that id" was false for those.
    """

    root = Path(host.artifact_root) / "grasp-queries"
    stale = False
    current: list[str] = []
    disposition: dict[str, Any] | None = None
    for manifest_path in sorted(root.glob("*/manifest.json")) if root.is_dir() else ():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bound = str(manifest.get("identity", {}).get("observation_id")) == observation_id
        if bound:
            current.extend(str(item.get("candidate_id")) for item in manifest.get("compiled_candidates", ()))
            disposition = disposition or _withheld(manifest, candidate_id)
        for candidate in manifest.get("compiled_candidates", ()):
            if str(candidate.get("candidate_id")) != candidate_id:
                continue
            if not bound:
                stale = True
                continue
            compiler = manifest.get("identity", {}).get("compiler")
            compiler = compiler if isinstance(compiler, Mapping) else {}
            return {
                "candidate": dict(candidate),
                "target_ref": str(manifest["identity"]["target_ref"]),
                "query_ref": str(manifest["query_ref"]),
                # The arm a candidate was compiled for is a property of the
                # plan, not of the moment it executes in. A dual-arm backend
                # names it here because it has a choice to record; a
                # single-arm one leaves it out and the observation resolves it.
                "arm": str(compiler["arm"]) if compiler.get("arm") else None,
            }
    valid = {"current_observation_id": observation_id, "valid_candidate_ids": sorted(set(current))}
    if disposition is not None:
        raise ToolError(
            f"candidate_id {candidate_id} was {_WITHHELD[disposition['disposition']]} by "
            f"{_advertised(host, 'grasp_plan')} on the current observation and is not executable",
            code="candidate_unavailable", field="candidate_id", received=candidate_id,
            retryable=True, retry_requires_changed_arguments=True,
            constraints={**disposition, **valid},
        )
    raise ToolError(
        "grasp candidate was compiled from an earlier observation" if stale else
        "no grasp candidate carries that candidate_id",
        code="stale_candidate_observation" if stale else "candidate_unavailable",
        field="candidate_id", received=candidate_id, retryable=True,
        retry_requires_changed_arguments=True, constraints=valid,
    )


_WITHHELD = {
    "rejected": "rejected", "rejected_by_compiler": "rejected at compilation",
    "not_displayed": "kept in the pool but not displayed",
}


def _withheld(manifest: Mapping[str, Any], candidate_id: str) -> dict[str, Any] | None:
    """Why one current ``grasp_plan`` call did not publish ``candidate_id``, if it did not."""

    diagnostics = manifest.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    for key, disposition in (("rejected", "rejected"), ("compiler_rejected", "rejected_by_compiler")):
        for entry in diagnostics.get(key) or ():
            if isinstance(entry, Mapping) and str(entry.get("candidate_id")) == candidate_id:
                reasons = entry.get("reasons", entry.get("reason"))
                return {"disposition": disposition,
                        "reasons": list(reasons) if isinstance(reasons, list) else [reasons]}
    selection = manifest.get("candidate_selection")
    if isinstance(selection, Mapping) and any(
        isinstance(item, Mapping) and str(item.get("candidate_id")) == candidate_id
        for item in selection.get("pool") or ()
    ):
        return {"disposition": "not_displayed",
                "displayed_candidate_ids": list(selection.get("displayed_candidate_ids") or ())}
    return None


def _place_candidate(host: Any, candidate_id: str, observation_id: str) -> dict[str, Any]:
    """Find one place candidate produced from the current observation.

    ``place.plan`` numbers its candidates per call (``P0``, ``P1``), so unlike a
    grasp identity these are not unique on their own. Two assessments of the
    current observation that both publish ``P0`` are a genuine ambiguity and are
    refused by name rather than resolved by recency: picking the newer one would
    silently execute a candidate the policy did not look at.
    """

    root = Path(host.artifact_root) / "placement"
    matches: list[dict[str, Any]] = []
    stale = False
    for record_path in sorted(root.glob("place-check-*.json")) if root.is_dir() else ():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        computation = record.get("computation", {})
        for candidate in computation.get("candidates", ()):
            if str(candidate.get("candidate_id")) != candidate_id:
                continue
            if str(record.get("identity", {}).get("observation_id")) != observation_id:
                stale = True
                continue
            matches.append({
                "candidate": dict(candidate),
                "assessment_ref": record_path.stem,
                "arm": record.get("identity", {}).get("arguments", {}).get("arm"),
                "phase": str(computation.get("phase") or record["identity"]["arguments"].get("phase", "")),
            })
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ToolError(
            "several current place assessments publish that candidate_id; call "
            f"{_advertised(host, 'place_plan')} once and execute from that "
            "call's candidates",
            code="ambiguous_candidate_id", field="candidate_id",
            received=candidate_id, retryable=True,
            constraints={"matching_assessments": sorted(
                item["assessment_ref"] for item in matches
            )},
        )
    raise ToolError(
        "place candidate was assessed from an earlier observation" if stale else
        "no place candidate carries that candidate_id",
        code="stale_candidate_observation" if stale else "candidate_unavailable",
        field="candidate_id", received=candidate_id, retryable=True,
        retry_requires_changed_arguments=True,
    )
