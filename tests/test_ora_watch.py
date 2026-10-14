"""``ora watch``: ORA's read-only terminal transcript over a Native run's agent events."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import ora
from ora.agent.native.tui import EventTail, ReplayClock, watch
from ora.agent.native.tui import cli as ora_cli
from ora.agent.native.tui import follow as watch_module

RESULT = {
    "candidates": [
        {"candidate_id": "T0", "backend_score": 0.91,
         "localization": {"center_xyz_m": [0.1, -0.2, 0.8], "height_above_support_m": 0.104}},
        {"candidate_id": "T1", "backend_score": 0.12, "localization": {}},
    ],
}
REFUSAL = {"ok": False, "error": {"code": "workspace_bounds", "message": "target pose is outside the configured action workspace"}}
RECEIPT = {"receipt": {"execution_status": "reached", "after": {"right": {"pose_xyz_xyzw": [0.1, -0.2, 0.95, 0, 0, 0, 1]}},
                       "arms": {"right": {"pose_error": {"translation_m": 0.0042}}}}}


def _artifact(root: Path, name: str, value: dict) -> dict:
    path = root / "native" / "agent-artifacts" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {"path": path.relative_to(root).as_posix(), "kind": name, "media_type": "application/json"}


def _events(root: Path) -> list[dict]:
    rows = [
        ("session_started", {"model": "acme/open-model-1", "reasoning_effort": "low", "provider_protocol": "openai_chat_completions",
                             "context_window_tokens": 262144}, None, None),
        ("provider_request", {"model": "open-model-1"}, None, None),
        ("provider_response", {"message": {"role": "assistant", "content": "I will  find the\nbowl first.",
                                           "reasoning_content": "SECRET-REASONING must never be shown"},
                               "usage": {"input_tokens": 30537, "output_tokens": 1210}}, None, None),
        ("tool_call_proposed", {"name": "vision_ground", "effect": "read_only",
                                "arguments": {"camera": "head", "text": "black bowl", "max_candidates": 2}}, None, "c1"),
        ("tool_result", {"name": "vision_ground", "ok": True}, _artifact(root, "ground", RESULT), "c1"),
        ("tool_image", {"image_index": 0}, {"path": "native/agent-artifacts/a.png"}, "c1"),
        ("tool_image", {"image_index": 1}, {"path": "native/agent-artifacts/b.png"}, "c1"),
        ("provider_request", {}, None, None),
        ("provider_response", {"message": {"role": "assistant", "content": ""},
                               "usage": {"input_tokens": 41000, "output_tokens": 300}}, None, None),
        ("tool_call_proposed", {"name": "move_pose", "effect": "mutation",
                                "arguments": {"targets": {"right": {"position_m": [0.1, -0.2, 1.4]}}}}, None, "c2"),
        ("tool_result", {"name": "move_pose", "ok": False}, _artifact(root, "refusal", REFUSAL), "c2"),
        ("tool_call_proposed", {"name": "move_pose", "effect": "mutation",
                                "arguments": {"targets": {"right": {"position_m": [0.1, -0.2, 0.95]}}}}, None, "c3"),
        ("tool_result", {"name": "move_pose", "ok": True}, _artifact(root, "submitted", {"_runtime": {"action_id": "a1"}}), "c3"),
        ("action_proposed", {"action_id": "a1"}, None, "c3"),
        ("environment_receipt", {"action_id": "a1"}, _artifact(root, "receipt", RECEIPT), None),
        ("context_compactor_attempt", {"call_kind": "context_compactor"}, None, None),
        ("context_compactor_response", {"latency_s": 29.4}, None, None),
        ("official_verdict", {"official_success": False}, None, None),
        ("tool_call_proposed", {"name": "check_task", "effect": "read_only", "arguments": {}}, None, "c4"),
        ("tool_result", {"name": "check_task", "ok": True}, _artifact(root, "check", {"success": True, "task_success": True}), "c4"),
        ("official_verdict", {"official_success": True}, None, None),
        ("run_terminated", {"state": "completed", "termination": "session_round_limit"}, None, None),
        ("session_closed", {"reason": "closed"}, None, None),
    ]
    return [
        {"seq": index, "type": kind, "payload": payload, "artifact_ref": ref, "tool_call_id": call,
         "unix_s": 1000.0 + 2.0 * index}
        for index, (kind, payload, ref, call) in enumerate(rows)
    ]


def _run(tmp_path: Path, *, state: str | None = "completed") -> Path:
    root = tmp_path / "exp-robodojo-abc--20261012T000000Z"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps({
        "task": "pour_by_language", "experiment": "exp-test", "task_instruction": "Pour the water into the black bowl.",
        "runtime": {"kind": "robodojo"}, "session": {"rounds": 2},
    }))
    events = _events(root)
    (root / "agent-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    if state:
        (root / "status.json").write_text(json.dumps({"state": state}))
    return root


EXPECTED = """\
   ▄▄██▄▄     ORA v-test  open robot-use agent
   █▄██▄█     open-model-1 · low effort · openai_chat_completions
  ▄▄ ██ ▄▄    RoboDojo / pour_by_language · exp-test
  ▀      ▀    exp-robodojo-abc--20261012T000000Z · finished run

❯ Pour the water into the black bowl.

● I will find the bowl first.

● vision_ground(text="black bowl", camera="head", max_candidates=2)
  ⎿  T0  score 0.91  centre (0.100, -0.200, 0.800) m  height 104 mm
     T1  score 0.12
     2 images to the model

● move_pose(right: position_m=[0.1, -0.2, 1.4])
  ⎿  error workspace_bounds: target pose is outside the configured action workspace

● move_pose(right: position_m=[0.1, -0.2, 0.95])
  ⎿  receipt  reached  end effector (0.100, -0.200, 0.950) m  error 4.2 mm

✻ Context compacted · 29 s

● Round 1/2 · official verdict (RoboDojo evaluator): failure

── round 2/2: the world is reset ──

● check_task()
  ⎿  success true · task_success true

● Round 2/2 · official verdict (RoboDojo evaluator): success

── run completed · session round limit ──
▸▸ round 2/2 · context 41.0k / 262.1k · last call 2s ↓ 300 · open-model-1 · low effort
"""


def _inventory(root: Path) -> dict[str, tuple[int, int]]:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*"))}


def test_sealed_run_renders_the_transcript_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_module, "_version", lambda: "v-test")
    root = _run(tmp_path)
    before = _inventory(root)
    out = io.StringIO()
    assert watch(root, stream=out, plain=True, speed=0, width=100) == 0
    text = out.getvalue()
    assert text == EXPECTED
    assert "SECRET-REASONING" not in text
    assert _inventory(root) == before


def test_cli_prints_plain_transcript_when_not_a_terminal(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(watch_module, "_version", lambda: "v-test")
    root = _run(tmp_path)
    assert ora_cli.main(["watch", str(root), "--plain"]) == 0
    assert "● Round 2/2 · official verdict (RoboDojo evaluator): success" in capsys.readouterr().out
    assert ora_cli.main(["watch", str(tmp_path / "missing/run")]) == 2


def test_replay_draws_a_live_status_with_recorded_elapsed_time(tmp_path):
    root = _run(tmp_path)
    clock = {"now": 0.0}
    out = io.StringIO()
    assert watch(root, stream=out, plain=False, speed=4.0, width=100,
                 now=lambda: clock["now"], sleep=lambda s: clock.__setitem__("now", clock["now"] + max(s, 0.01))) == 0
    text = out.getvalue()
    assert "Thinking… (1s" in text            # two recorded seconds per event, replayed at x4
    assert "Moving on the robot…" in text and "Compacting context…" in text
    assert "\x1b[?25h" in text                # the cursor is restored


def test_live_run_is_followed_until_the_session_closes(tmp_path):
    root = _run(tmp_path, state=None)
    events = (root / "agent-events.jsonl").read_text().splitlines(keepends=True)
    (root / "agent-events.jsonl").write_text("".join(events[:4]))
    out = io.StringIO()

    def append_rest():
        for line in events[4:]:
            time.sleep(0.005)
            with (root / "agent-events.jsonl").open("a") as stream:
                stream.write(line[:15])
                stream.flush()
                stream.write(line[15:])

    writer = threading.Thread(target=append_rest)
    writer.start()
    assert watch(root, stream=out, plain=True, poll=0.02, width=100) == 0
    writer.join()
    text = out.getvalue()
    assert "· live" in text
    assert "● Round 2/2 · official verdict (RoboDojo evaluator): success" in text


def test_event_tail_holds_back_a_torn_line(tmp_path):
    path = tmp_path / "agent-events.jsonl"
    path.write_text('{"type": "a"}\n{"type": ')
    tail = EventTail(path)
    assert [event["type"] for event in tail.read()] == ["a"]
    with path.open("a") as stream:
        stream.write('"b"}\n')
    assert [event["type"] for event in tail.read()] == ["b"]
    assert tail.read() == []


def test_replay_clock_shortens_long_gaps_but_keeps_recorded_time():
    clock = ReplayClock([100.0, 102.0, 702.0], speed=2.0, max_gap=30.0)
    assert clock.due == [0.0, 1.0, 16.0]
    assert clock.recorded_at(0.5) == 101.0
    assert clock.recorded_at(8.5) == 402.0
    assert clock.recorded_at(99.0) == 702.0


def test_importing_the_watcher_leaves_the_native_loop_unloaded():
    probe = ("import sys, ora.agent.native.tui, ora.orchestrator.ora_cli; "
             "print(sorted(m for m in ('numpy', 'ora.agent.native.session') if m in sys.modules))")
    root = Path(ora.__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", probe], cwd=root, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"


def test_jsonl_emits_the_operations_the_pi_front_end_draws(tmp_path):
    root = _run(tmp_path)
    out = io.StringIO()
    assert watch(root, stream=out, emit="jsonl", speed=0) == 0
    ops = [json.loads(line) for line in out.getvalue().splitlines()]
    kinds = [op["op"] for op in ops]
    assert kinds[:3] == ["header", "user", "assistant"]
    assert ops[0]["model"] == "open-model-1" and ops[0]["rounds"] == 2 and ops[0]["mode"] == "finished run"
    assert ops[2]["text"] == "I will  find the\nbowl first."
    by_kind = {kind: [op for op in ops if op["op"] == kind] for kind in set(kinds)}
    assert [op["name"] for op in by_kind["tool_call"]] == ["vision_ground", "move_pose", "move_pose", "check_task"]
    refusal = by_kind["tool_result"][1]
    assert refusal["error"] and refusal["done"] and refusal["lines"][0].startswith("error workspace_bounds")
    assert by_kind["tool_result"][2] == {"op": "tool_result", "id": "c3", "lines": [], "error": False, "done": False}
    assert by_kind["receipt"][0]["id"] == "c3"
    assert by_kind["tool_images"] == [{"op": "tool_images", "id": "c1", "count": 2}]
    assert [op["success"] for op in by_kind["verdict"]] == [False, True]
    assert kinds[-2:] == ["footer", "end"] and ops[-1]["ended"] == "completed · session round limit"
    assert ops[-2]["input_total"] == 71537 and ops[-2]["context_window"] == 262144
    assert "SECRET-REASONING" not in out.getvalue()


def test_condensed_results_name_empty_plans_and_hide_digests():
    from ora.agent.native.tui.condense import result_lines

    assert result_lines("grasp_plan", {"candidates": [], "status": "no_candidate"}) == ["no candidates (no_candidate)"]
    query = {"camera": "agentview", "step": 0, "source_state_sha256": "2e" * 32, "points": [
        {"pixel_uv": [80, 100], "valid": True, "world_xyz_m": [-0.216, -0.1627, 1.026], "depth_median_m": 1.0476},
        {"pixel_uv": [80, 110], "valid": False},
    ]}
    assert result_lines("geometry_query", query) == [
        "[80, 100] → (-0.216, -0.163, 1.026) m  depth 1.048 m", "[80, 110] → invalid"]
    generic = result_lines("observe", {"camera": "head", "source_state_sha256": "2e" * 32, "digest": "ab" * 32})
    assert generic == ['camera "head"']


def test_an_operator_message_starts_its_own_block(tmp_path, monkeypatch):
    monkeypatch.setattr(watch_module, "_version", lambda: "v-test")
    root = _run(tmp_path)
    events = [json.loads(line) for line in (root / "agent-events.jsonl").read_text().splitlines()]
    events.insert(7, {"seq": 99, "type": "operator_message", "unix_s": 1013.5, "artifact_ref": None,
                      "tool_call_id": None, "payload": {"id": "steer-1", "message": "Look at the bowl first."}})
    (root / "agent-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    out = io.StringIO()
    assert watch(root, stream=out, plain=True, speed=0, width=100) == 0
    lines = out.getvalue().splitlines()
    index = lines.index("❯ Look at the bowl first.")
    assert lines[index - 1] == "" and lines[index - 2].strip() == "2 images to the model"
