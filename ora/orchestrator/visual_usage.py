"""Separate automatic observation receipt, fresh reads and Artifact reopens."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def visual_usage(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_event = {event.get("event_id"): event for event in events if event.get("event_id")}
    buckets = {name: {"proposed": 0, "results_ok": 0, "results_failed": 0, "image_events": 0}
               for name in ("observe_current", "observe_historical_artifact", "observe_unclassified",
                            "artifact_read", "other_visual_tools")}

    def bucket(proposal: Mapping[str, Any]) -> str:
        payload = proposal.get("payload", {})
        name = payload.get("name")
        if name == "artifact.read":
            return "artifact_read"
        if name != "observe":
            return "other_visual_tools"
        args = payload.get("arguments")
        if not isinstance(args, Mapping):
            return "observe_unclassified"
        ref = args.get("ref")
        if ref is None:
            return "observe_current"
        return "observe_historical_artifact" if isinstance(ref, str) and ref.startswith(
            "artifact://"
        ) else "observe_unclassified"

    unknown_images = 0
    for event in events:
        kind = event.get("type")
        if kind == "tool_call_proposed":
            name = bucket(event)
            # Only image-producing tools are counted in this residual bucket;
            # their proposals/results are unavailable until an image joins.
            if name != "other_visual_tools":
                buckets[name]["proposed"] += 1
        elif kind == "tool_result":
            proposal = by_event.get(event.get("causal_parent_id"), {})
            if proposal.get("type") != "tool_call_proposed":
                continue
            name = bucket(proposal)
            if name != "other_visual_tools":
                key = "results_ok" if event.get("payload", {}).get("ok") is True else "results_failed"
                buckets[name][key] += 1
        elif kind == "tool_image":
            result = by_event.get(event.get("causal_parent_id"), {})
            # Multiple images are emitted as a causal chain after one result.
            # Follow image ancestors, never a reused wire call ID.
            visited: set[str] = set()
            while result.get("type") == "tool_image":
                identity = result.get("event_id")
                if not isinstance(identity, str) or identity in visited:
                    result = {}
                    break
                visited.add(identity)
                result = by_event.get(result.get("causal_parent_id"), {})
            proposal = by_event.get(result.get("causal_parent_id"), {})
            if result.get("type") != "tool_result" or proposal.get("type") != "tool_call_proposed":
                unknown_images += 1
            else:
                buckets[bucket(proposal)]["image_events"] += 1
    other_images = buckets.pop("other_visual_tools")["image_events"]
    return {
        "basis": "native_agent_events_and_causal_parent_ids",
        "automatic_observation_events": sum(event.get("type") == "observation_received" for event in events),
        **buckets, "other_tool_image_events": other_images,
        "unjoined_tool_image_events": unknown_images,
        "interpretation": "Event counts, not attention, fresh camera captures, or repeated Provider image deliveries.",
    }
