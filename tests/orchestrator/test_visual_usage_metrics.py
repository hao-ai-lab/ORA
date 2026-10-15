"""Image reopens are not fresh observes; reused Provider call IDs cannot misjoin."""
from ora.orchestrator.visual_usage import visual_usage


def test_visual_usage_joins_event_identity_and_separates_reads():
    events = [{"event_id": "o", "type": "observation_received", "payload": {}}]
    for i, (name, arguments, ok) in enumerate([
        ("observe", {}, True),
        ("observe", {"ref": "artifact://session/old"}, True),
        ("observe", {"ref": "artifact://session/missing"}, False),
        ("vision_ground", {}, True),
        ("artifact.read", {"ref": "artifact://session/text"}, True),
    ]):
        events.extend([
            {"event_id": f"p{i}", "type": "tool_call_proposed", "tool_call_id": "reused",
             "payload": {"name": name, "arguments": arguments}},
            {"event_id": f"r{i}", "type": "tool_result", "tool_call_id": "reused",
             "causal_parent_id": f"p{i}", "payload": {"name": name, "ok": ok}},
        ])
        if ok and name != "artifact.read":
            events.append({"event_id": f"i{i}", "type": "tool_image", "causal_parent_id": f"r{i}"})
    events.append({"event_id": "second-historical", "type": "tool_image", "causal_parent_id": "i1"})
    usage = visual_usage(events)
    assert usage["automatic_observation_events"] == 1
    assert usage["observe_current"] == {"proposed": 1, "results_ok": 1, "results_failed": 0, "image_events": 1}
    assert usage["observe_historical_artifact"] == {"proposed": 2, "results_ok": 1, "results_failed": 1, "image_events": 2}
    assert usage["artifact_read"]["results_ok"] == 1
    assert usage["other_tool_image_events"] == 1
    assert usage["unjoined_tool_image_events"] == 0


def test_unjoined_image_is_explicit_and_never_guessed_from_tool_call_id():
    assert visual_usage([{"type": "tool_image", "tool_call_id": "reused"}])["unjoined_tool_image_events"] == 1
