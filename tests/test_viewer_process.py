"""Recorded process ordering, round identity and CPU-only presentation media."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from viewer import app as viewer
from viewer.process_timeline import compile_process_timeline, tool_images_by_step
from viewer.process_video import PresentationPendingError, VideoOptions, presentation_video, render_plan
from viewer.registry import register_run, registry
from viewer.review_media import derived_root, find_execution_media
from viewer.rounds import render_round_matrix, round_ledger
from viewer.trajectory import compile_trajectory


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image(root: Path, name: str, color: str) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color).save(path)
    return {"path": name, "sha256": _hash(path), "media_type": "image/png"}


@pytest.fixture
def process_run(tmp_path: Path) -> Path:
    root = tmp_path / "runs/process-k2"
    _write(root / "plan.json", {
        "experiment": "process-test", "config": "K2", "task": "pick", "seed": 0,
        "source": {"commit": "source-1"}, "evidence_tier": "behavioral",
        "runtime": {"kind": "libero_pro", "suite": "libero_10", "task_id": 0},
        "policy": {"kind": "native_api", "model": "test"},
        "session": {"cadence": "persistent_agent_k_rounds_v1", "rounds": 2},
    })
    _write(root / "status.json", {"state": "completed"})
    _write(root / "result.json", {
        "state": "completed", "official_success": None, "round_count": 2,
        "session": {"cadence": "persistent_agent_k_rounds_v1", "rounds": 2},
        "rounds": [
            {"round_index": 0, "state": "completed", "official_success": False, "decisions": 1,
             "model_calls": 2, "policy_state_steps": {"start": 0, "end_exclusive": 2}},
            {"round_index": 1, "state": "completed", "official_success": True, "decisions": 1,
             "model_calls": 1, "policy_state_steps": {"start": 2, "end_exclusive": 4}},
        ],
    })
    rgb = _image(root, "images/observation.png", "#404040")
    overlay = _image(root, "images/grounding.png", "#389063")
    query_result = root / "native/query.json"
    _write(query_result, {"candidates": [{"id": "T0"}, {"id": "T1"}]})
    tool_ref = {"path": "native/query.json", "sha256": _hash(query_result), "media_type": "application/json"}
    manifest, runtime, agent = [], [], []
    for step in range(4):
        relative = f"policy-state/steps/step-{step:04d}.json"
        _write(root / relative, {
            "step": step, "cause": "reset" if step % 2 == 0 else "after_action",
            "task_instruction": "Pick the cup.", "observation": {"id": f"obs-{step}"},
            "main_camera": "agentview", "cameras": {"agentview": {"files": {"rgb_lossless_png": rgb}}},
        })
        manifest.append({"step": step, "path": relative})
        runtime.append({"type": "observation", "observation_id": f"obs-{step}", "policy_state": {"step": step}})
        if step % 2:
            continue
        action = {"action_id": f"action-{step}", "action": "move_pose", "arguments": {"position_m": [0, 0, 0.5]}}
        receipt = {"action_id": f"action-{step}", "consumed": True, "pose_reached": True}
        runtime.extend([{"type": "action_submitted", "action": action}, {"type": "motion_receipt", "record": receipt}])
        agent.append({"type": "provider_response", "observation_step": step, "model_call_id": f"model-{step}",
                      "payload": {"message": {"content": "Locate the cup.", "reasoning_content": "Recorded reasoning. " * 300}}})
        for call_index in range(2 if step == 0 else 1):
            call = f"tool-{step}-{call_index}"
            agent.extend([
                {"type": "tool_call_proposed", "observation_step": step, "tool_call_id": call,
                 "model_call_id": f"model-{step}", "payload": {"name": "vision.ground", "arguments": {"text": f"cup query {call_index}"}}},
                {"type": "tool_result", "observation_step": step, "tool_call_id": call,
                 "model_call_id": f"model-{step}", "payload": {"name": "vision.ground", "ok": True}, "artifact_ref": tool_ref},
                {"type": "tool_image", "observation_step": step, "tool_call_id": call,
                 "model_call_id": f"model-{step}", "artifact_ref": overlay, "visibility": "exact"},
            ])
            if call_index == 0 and step == 0:
                agent.append({"type": "provider_response", "observation_step": step, "model_call_id": "model-retry",
                              "payload": {"message": {"content": "Inspect another candidate."}}})
        agent.extend([
            {"type": "action_proposed", "observation_step": step, "payload": {"action_id": f"action-{step}"}},
            {"type": "environment_receipt", "observation_step": step, "payload": {"action_id": f"action-{step}"}},
        ])
    for index, event in enumerate(agent):
        event.update(seq=index, event_id=f"event-{index}", causal_parent_id=f"event-{index-1}" if index else None)
    _write(root / "policy-state/manifest.json", {"steps": manifest})
    _jsonl(root / "events.jsonl", runtime)
    _jsonl(root / "agent-events.jsonl", agent)
    return root


def _execution_media(root: Path) -> tuple[Path, Path]:
    cache = derived_root(root)
    cache.mkdir(parents=True, exist_ok=True)
    video = cache / "trajectory-review-fixture.mp4"
    colors = ["#000000", "#dc2525", "#d92525", "#000000", "#2555dc", "#2555d9"]
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1920x960", "-r", "10", "-i", "-", "-an", "-c:v", "libx264", "-threads", "1",
        "-pix_fmt", "yuv420p", str(video),
    ], stdin=subprocess.PIPE)
    for color in colors:
        process.stdin.write(Image.new("RGB", (1920, 960), color).tobytes())
    process.stdin.close()
    assert process.wait() == 0
    timeline = video.with_suffix(".timeline.jsonl")
    _jsonl(timeline, [{"frame_index": index, "action_id": None if index in {0, 3} else f"action-{0 if index < 3 else 2}",
                       "phase": "reset" if index == 0 else "round_reset" if index == 3 else "control"}
                      for index in range(6)])
    _write(video.with_suffix(".json"), {
        "schema": "ora.viewer-trajectory-video.v2", "run_id": root.name,
        "continuous_execution": True, "video_source_kind": "same_session_robotwin_control_frames",
        "source_hashes": compile_trajectory(root)["source_hashes"], "frame_count": 6,
        "fps": 10, "resolution": [1920, 960], "video_sha256": _hash(video),
        "execution_timeline": {"path": timeline.name, "sha256": _hash(timeline)},
    })
    return video, timeline


def test_process_keeps_interleaved_calls_and_duplicate_image_events(process_run: Path) -> None:
    process = compile_process_timeline(process_run)
    first = [event for event in process["events"] if event.get("state_step") == 0]
    assert [event["phase"] for event in first] == [
        "round_start", "observation", "decision", "perception", "decision", "perception", "execution", "outcome",
    ]
    queries = [event for event in first if event["phase"] == "perception"]
    assert [event["arguments"]["text"] for event in queries] == ["cup query 0", "cup query 1"]
    assert queries[0]["images"][0]["sha256"] == queries[1]["images"][0]["sha256"]
    assert queries[0]["source_refs"] != queries[1]["source_refs"]
    assert process["coverage"]["reasoning_responses"] == 2
    assert process["unmapped_events"] == []
    trajectory = compile_trajectory(process_run)
    assert sum(image["kind"] == "tool_image" for image in trajectory["steps"][0]["images"]) == 1
    assert trajectory["summary"]["official_success"] is None


@pytest.mark.parametrize("damage", ["hash", "missing", "escape"])
def test_process_does_not_render_unverified_tool_images(process_run: Path, damage: str) -> None:
    path = process_run / "agent-events.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    images = [event for event in records if event["type"] == "tool_image"]
    for event in images:
        if damage == "hash":
            event["artifact_ref"]["sha256"] = "0" * 64
        elif damage == "missing":
            event["artifact_ref"]["path"] = "missing.png"
        else:
            event["artifact_ref"]["path"] = "../../outside.png"
    _jsonl(path, records)
    process = compile_process_timeline(process_run)
    perception = [event for event in process["events"] if event["phase"] == "perception"]
    assert all(not event["images"] and event["issues"] for event in perception)
    assert tool_images_by_step(process_run, records) == {}


def test_round_matrix_preserves_unrun_incomplete_and_source_identity(process_run: Path) -> None:
    result = json.loads((process_run / "result.json").read_text())
    result["session"]["rounds"] = 4
    result["rounds"][1].update(state="incomplete", official_success=None)
    _write(process_run / "result.json", result)
    ledger = round_ledger(process_run)
    assert [row["outcome"] for row in ledger["rounds"]] == ["failure", "incomplete", "not_run", "not_run"]
    assert ledger["scored_rounds"] == 1
    assert ledger["aggregate_official_success"] is None
    html = render_round_matrix([register_run(process_run)])
    assert "?round=2#review-player" in html
    assert "?round=3#review-player" not in html
    assert "source-1" in html and process_run.name in html

    result["rounds"][0]["decisions"] = "<img src=x onerror=alert(1)>"
    _write(process_run / "result.json", result)
    html = render_round_matrix([register_run(process_run)])
    assert "<img src=x" not in html
    assert "&lt;img src=x" in html


def test_live_rounds_use_only_published_notices_and_finished_events(process_run: Path) -> None:
    result = json.loads((process_run / "result.json").read_text())
    (process_run / "result.json").unlink()
    _write(process_run / "status.json", {"state": "running"})
    _jsonl(process_run / "events.jsonl", [
        {"type": "observation", "policy_state": {"step": 0},
         "context": {"remaining_budget": {"session_round": 1}}},
        {"type": "round_finished", "round_index": 0, "result": result["rounds"][0]},
        {"type": "observation", "policy_state": {"step": 2},
         "context": {"remaining_budget": {"session_round": 2}}},
    ])
    ledger = round_ledger(process_run)
    assert [row["outcome"] for row in ledger["rounds"]] == ["failure", "running"]
    assert all(row["provisional"] for row in ledger["rounds"])
    assert ledger["completed_rounds"] == 1


def test_legacy_execution_timeline_recovers_rounds_by_action_id(process_run: Path) -> None:
    video, _timeline = _execution_media(process_run)
    selected, manifest = presentation_video(process_run)
    assert selected == video
    assert [(item["round_index"], item["start_frame"], item["end_frame"]) for item in manifest["chapters"]] == [
        (0, 0, 1), (0, 1, 3), (1, 3, 4), (1, 4, 6),
    ]
    assert manifest["round_ledger"]["first_success_round"] == 2


def test_execution_cache_rejects_source_or_timeline_drift(process_run: Path) -> None:
    video, timeline = _execution_media(process_run)
    trajectory = compile_trajectory(process_run)
    timeline.write_text(timeline.read_text() + "{}\n")
    assert find_execution_media(process_run, trajectory) is None
    _execution_media(process_run)
    _write(process_run / "plan.json", {"source": {"commit": "changed"}})
    assert find_execution_media(process_run, compile_trajectory(process_run)) is None
    assert video.is_file()


def test_round_selection_and_reasoning_are_part_of_independent_caches(process_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = {str(path.relative_to(process_run)): _hash(path) for path in process_run.rglob("*") if path.is_file()}
    def no_simulator(*args, **kwargs):
        raise AssertionError("process rendering must not invoke a simulator")
    monkeypatch.setattr("viewer.trajectory.ensure_controller_replay", no_simulator)
    options = VideoOptions(mode="process", round_index=1, card_seconds=0.5)
    first, manifest = presentation_video(process_run, options, allow_compile=True)
    assert {item["round_index"] for item in manifest["chapters"]} == {1}
    assert manifest["execution_source"] is None
    same, saved = presentation_video(process_run, options)
    assert same == first and saved == manifest
    with pytest.raises(PresentationPendingError):
        presentation_video(process_run, VideoOptions(mode="process", round_index=1, reasoning=True, card_seconds=0.5))
    other, _ = presentation_video(process_run, VideoOptions(mode="process", round_index=1, reasoning=True, card_seconds=0.5), allow_compile=True)
    assert other != first
    assert original == {str(path.relative_to(process_run)): _hash(path) for path in process_run.rglob("*") if path.is_file()}
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,nb_frames", "-of", "json", str(first)], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert (stream["width"], stream["height"], int(stream["nb_frames"])) == (1920, 960, manifest["frame_count"])


def test_single_round_execution_export_preserves_color_and_duration(process_run: Path) -> None:
    _execution_media(process_run)
    video, manifest = presentation_video(process_run, VideoOptions(round_index=1), allow_compile=True)
    assert manifest["frame_count"] == 3
    assert manifest["duration_s"] == pytest.approx(0.3)
    sample = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", "select=eq(n\\,2)", "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True)
    image = Image.frombytes("RGB", (1920, 960), sample.stdout)
    red, _green, blue = image.getpixel((100, 500))
    assert blue > red + 80  # R2 blue; never the earlier R1 red frames.


@pytest.mark.parametrize("kwargs", [
    {"round_index": -1}, {"round_index": True}, {"mode": "invented"},
    {"card_seconds": float("nan")}, {"card_seconds": float("inf")},
    {"card_seconds": 0}, {"mode": "execution", "reasoning": True},
])
def test_video_options_reject_ambiguous_or_unbounded_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        VideoOptions(**kwargs)


def test_web_projection_never_starts_a_compiler(process_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viewer, "run_root", lambda _run_id: process_run)
    def no_process(*args, **kwargs):
        raise AssertionError("read-only web requests must not spawn encoders or simulators")
    monkeypatch.setattr(subprocess, "Popen", no_process)
    client = viewer.app.test_client()
    process = client.get(f"/api/run/{process_run.name}/process")
    assert process.status_code == 200
    pending = client.get(f"/api/run/{process_run.name}/presentation-video?video_mode=process&round=2&reasoning=1")
    assert pending.status_code == 503
    assert "--include-reasoning" in pending.json["compile_hint"]
    assert "--round 2" in pending.json["compile_hint"]
    assert not derived_root(process_run).exists()
    assert client.get(f"/api/run/{process_run.name}/presentation-video?round=0").status_code == 400
    assert client.get(f"/api/run/{process_run.name}/presentation-video?round=3").status_code == 400
    assert client.get(f"/run/{process_run.name}/presentation.mp4?reasoning=2").status_code == 400


def test_process_reading_time_does_not_overwrite_latency(process_run: Path) -> None:
    process = compile_process_timeline(process_run)
    options = VideoOptions(mode="process", card_seconds=3, reasoning=True)
    plan = render_plan(process, None, options)
    decisions = [segment for segment in plan if segment["phase"] == "decision"]
    assert all(segment["end_frame"] - segment["start_frame"] == 30 for segment in decisions)
    assert decisions[0]["reasoning"] == "Recorded reasoning. " * 300
    assert decisions[0]["latency_s"] is None


def test_empty_round_retains_order_without_fabricating_execution(process_run: Path) -> None:
    result = json.loads((process_run / "result.json").read_text())
    result["session"]["rounds"] = 3
    result["rounds"][1]["round_index"] = 2
    result["rounds"].insert(1, {
        "round_index": 1, "state": "completed", "official_success": True,
        "decisions": 0, "policy_state_steps": {"start": 2, "end_exclusive": 2},
    })
    _write(process_run / "result.json", result)
    events = compile_process_timeline(process_run)["events"]
    assert [event["round_index"] for event in events if event["phase"] == "round_start"] == [0, 1, 2]
    assert [event["phase"] for event in events if event["round_index"] == 1] == ["round_start", "round_end"]


def test_result_registration_omits_unused_trace_projection_without_poisoning_cache(
    process_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write(process_run / "evaluation.json", {"status": "valid"})
    _write(process_run / "seal.json", {})
    checks = []
    def check(root):
        checks.append(root)
        return {"cached": True}
    monkeypatch.setattr("viewer.trajectory.trajectory_video_status", check)
    assert registry([process_run], inspect_video=False)[0].video_cached is None
    assert checks == []
    assert registry([process_run])[0].video_cached is True
    assert checks == [process_run]


@pytest.mark.parametrize("route", ["/?view=all", "/costs"])
def test_run_and_cost_lists_do_not_project_video_traces(
    process_run: Path, monkeypatch: pytest.MonkeyPatch, route: str,
) -> None:
    _write(process_run / "evaluation.json", {"status": "valid"})
    _write(process_run / "seal.json", {})
    monkeypatch.setattr(viewer, "viewer_run_roots", lambda: [process_run])
    def forbidden_projection(root):
        raise AssertionError("list pages must not project full video traces")
    monkeypatch.setattr("viewer.trajectory.trajectory_video_status", forbidden_projection)
    with viewer.app.test_client() as client:
        response = client.get(route)
    assert response.status_code == 200
    if route.startswith("/?"):
        body = response.get_data(as_text=True)
        assert "check in run" in body
        assert "not compiled" not in body
        assert f"/run/{process_run.name}" in body


def test_discovery_cache_lifetime_starts_after_slow_scan(
    process_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    clock = [0.0]
    scans = []
    monkeypatch.setattr(viewer, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    monkeypatch.setattr(viewer, "_RUN_ROOT_CACHE", {})
    monkeypatch.setitem(viewer.app.config, "VIEWER_OUTPUT_ROOTS", [process_run])
    def slow_scan(roots):
        scans.append(tuple(roots))
        clock[0] += 8.0
        return [process_run]
    monkeypatch.setattr(viewer, "discover_run_roots", slow_scan)
    assert viewer.viewer_run_roots() == [process_run]
    assert viewer.viewer_run_roots() == [process_run]
    assert len(scans) == 1
    clock[0] += viewer._RUN_ROOT_TTL_S + 1
    assert viewer.viewer_run_roots() == [process_run]
    assert len(scans) == 2


def test_exact_video_compile_scopes_registration_and_rejects_ambiguous_ids(
    process_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = process_run.parent / "unrelated"
    _write(other / "plan.json", {"experiment": "unrelated"})
    inspected = []
    def selected_registry(roots, *, inspect_video):
        assert not inspect_video
        inspected.extend(roots)
        return registry(roots, inspect_video=False)
    monkeypatch.setattr(viewer, "build_registry", selected_registry)
    monkeypatch.setattr(viewer, "presentation_video", lambda *args, **kwargs: (Path("video.mp4"), {"frame_count": 1}))
    assert viewer.compile_registered_videos(
        [process_run.parent.parent], scope="formal", run_ids=[process_run.name],
        video_options=VideoOptions(mode="process"),
    ) == (1, 1)
    assert inspected == [process_run]
    duplicate = process_run.parent.parent / "another/runs" / process_run.name
    _write(duplicate / "plan.json", {})
    with pytest.raises(ValueError, match="ambiguous"):
        viewer.compile_registered_videos([process_run.parent.parent], scope="formal", run_ids=[process_run.name])


def test_review_rejects_historical_reconstructed_cache_but_forensic_keeps_it(process_run: Path) -> None:
    video, timeline = _execution_media(process_run)
    path = video.with_suffix('.json')
    manifest = json.loads(path.read_text())
    trace = process_run / 'native/controller-replay/manifest.json'
    _write(trace, {'schema': 'ora.controller-replay-manifest.v1', 'entries': []})
    manifest.update(controller_replay=True, video_source_kind='viewer_derived_controller_replay',
                    replay_mode='review', controller_replay_manifest_sha256=_hash(trace))
    _write(path, manifest)
    trajectory = compile_trajectory(process_run)
    assert find_execution_media(process_run, trajectory) is None
    # The process source selector must also reject the old reconstruction.
    with pytest.raises(PresentationPendingError):
        presentation_video(process_run, VideoOptions(), allow_compile=False)
    forensic = video.with_name('trajectory-forensic-fixture.mp4')
    video.rename(forensic)
    manifest['replay_mode'] = 'forensic'
    _write(forensic.with_suffix('.json'), manifest)
    path.unlink()
    assert find_execution_media(process_run, trajectory, replay_mode='forensic')['path'] == forensic
    assert find_execution_media(process_run, trajectory) is None


def test_finalized_capture_priority_resampled_chapters_and_original_pixel_binding(process_run: Path) -> None:
    import numpy as np
    from ora.environment.execution_capture import ExecutionCapture
    from viewer.trajectory import compile_trajectory_video
    from viewer.process_video import _execution_header

    legacy_video, _ = _execution_media(process_run)
    assert find_execution_media(process_run, compile_trajectory(process_run))['path'] == legacy_video
    capture = ExecutionCapture(process_run / 'native', backend='libero_pro',
                               identity={'run_id': process_run.name}, cameras=['agentview', 'wrist'])
    pixels = np.full((16, 16, 3), 42, dtype=np.uint8)
    rows = [
        (0, 0., 'reset', None), (0, .2, 'control', 'action-0'),
        (0, .3, 'control', 'native-skill-part'), (0, .4, 'terminal', None),
        (1, 0., 'reset', None), (1, .1, 'control', 'action-2'), (1, .2, 'terminal', None),
    ]
    for index, (round_index, at, phase, action_id) in enumerate(rows):
        capture.capture(lambda: {'agentview': pixels, 'wrist': pixels}, round_index=round_index,
                        action_id=action_id, action_sequence=index if action_id else None,
                        action='move_pose' if action_id else None, sim_step=round(at * 100),
                        sim_time_s=at, step_unit='test', phase=phase, force=True)
    assert capture.close()['status'] == 'complete'
    trajectory = compile_trajectory(process_run)
    # A finalized capture supersedes the previously cached legacy source.
    assert find_execution_media(process_run, trajectory) is None
    video, manifest = compile_trajectory_video(process_run, allow_replay_compile=False)
    assert manifest['source_frame_count'] == 7 and manifest['frame_count'] == 8
    media = find_execution_media(process_run, trajectory)
    assert media is not None and media['path'] == video
    assert media['source_kind'] == 'same_session_execution_capture'
    assert media['chapters'][-1]['end_frame'] == 8
    assert {chapter['round_index'] for chapter in media['chapters']} == {0, 1}
    native = next(chapter for chapter in media['chapters'] if chapter['action_id'] == 'native-skill-part')
    assert native['state_step'] is None and native['policy_action_mapped'] is False
    assert any(chapter['capture_phase'] == 'terminal' for chapter in media['chapters'])
    cached, result = presentation_video(process_run, VideoOptions(), allow_compile=False)
    assert cached == video
    assert result['execution_source']['visual_fidelity'] == 'original_frames_only'
    assert result['execution_source']['unmapped_policy_action_chapters'][0]['action_id'] == 'native-skill-part'
    from viewer.process_video import _manifest
    process = compile_process_timeline(process_run, trajectory)
    options = VideoOptions(mode='process')
    process_manifest = _manifest(process, media, options, render_plan(process, media, options))
    assert process_manifest['unplaced_execution_chapters'][0]['action_id'] == 'native-skill-part'
    assert 'captured_control_actions_without_policy_join' in process_manifest['issues']
    # Recropping into a round/process video keeps the correctly recorded native label.
    frame = Image.new('RGB', (1920, 960), '#abcdef')
    assert _execution_header(frame, {}, native, media).tobytes() == frame.tobytes()
    image = next((capture.root / 'frames').glob('*.png'))
    image.write_bytes(b'changed original image')
    with pytest.raises(RuntimeError, match='hash drifted'):
        find_execution_media(process_run, trajectory)


def test_replay_cache_identity_includes_backend_implementation(tmp_path: Path, monkeypatch) -> None:
    from viewer import replay

    manifest = tmp_path / 'native/controller-replay/manifest.json'
    _write(manifest, {'schema': 'ora.controller-replay-manifest.v1', 'entries': []})
    monkeypatch.setattr(replay, '_replay_implementation_hashes', lambda: {'worker': 'a', 'libero_session': 'old'})
    first, _ = replay.replay_identity(tmp_path)
    monkeypatch.setattr(replay, '_replay_implementation_hashes', lambda: {'worker': 'a', 'libero_session': 'new'})
    assert replay.replay_identity(tmp_path)[0] != first
