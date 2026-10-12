"""Offline, source-bound process videos composed from existing media and evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image, ImageDraw, ImageFont, ImageOps

from viewer.process_timeline import compile_process_timeline, verified_artifact
from viewer.review_media import derived_root, file_hash, find_execution_media
from viewer.rounds import load_object


PRESENTATION_SCHEMA = "ora.viewer-presentation-video.v1"
RENDERER_VERSION = "process-video-original-capture-2"
FRAME_SIZE = (1920, 960)
FPS = 10


class PresentationPendingError(RuntimeError):
    """An explicit offline media compile is needed; the web route stays read-only."""


@dataclass(frozen=True)
class VideoOptions:
    mode: str = "execution"
    round_index: int | None = None
    reasoning: bool = False
    context: bool = False
    card_seconds: float = 2.5
    replay_mode: str = "review"

    def __post_init__(self) -> None:
        if self.mode not in {"execution", "process"} or self.replay_mode not in {"review", "forensic"}:
            raise ValueError("unsupported video mode")
        if self.round_index is not None and (type(self.round_index) is not int or self.round_index < 0):
            raise ValueError("round must be a positive one-based integer")
        if type(self.reasoning) is not bool or type(self.context) is not bool:
            raise ValueError("reasoning/context must be Boolean")
        if not isinstance(self.card_seconds, (int, float)) or isinstance(self.card_seconds, bool) or not math.isfinite(self.card_seconds) or not 0.5 <= self.card_seconds <= 15:
            raise ValueError("card seconds must be between 0.5 and 15")
        if self.mode == "execution" and (self.reasoning or self.context):
            raise ValueError("reasoning/context are process-video options")


def compile_hint(run_id: str, options: VideoOptions) -> str:
    # Run IDs are passed as argv by the CLI. Quote here only for operator copy.
    import shlex

    args = ["./scripts/compile-viewer-videos", "--run-id", run_id, "--video-mode", options.mode]
    if options.round_index is not None:
        args.extend(["--round", str(options.round_index + 1)])
    if options.reasoning:
        args.append("--include-reasoning")
    if options.context:
        args.append("--include-context")
    if options.card_seconds != 2.5:
        args.extend(["--card-seconds", str(options.card_seconds)])
    if options.replay_mode != "review":
        args.extend(["--replay-mode", options.replay_mode])
    return shlex.join(args)


def _source_projection(media: Mapping[str, Any] | None) -> dict | None:
    if media is None:
        return None
    return {
        "video": media["path"].name, "video_sha256": media["sha256"],
        "manifest_sha256": file_hash(media["manifest_path"]),
        "timeline_sha256": file_hash(media["timeline_path"]) if media.get("timeline_path") else None,
        "kind": media["source_kind"], "fps": media["fps"],
        "visual_fidelity": media["manifest"].get("visual_fidelity") or (
            "unverified_simulator_reconstruction" if media["source_kind"].startswith("viewer_derived_")
            else "original_frames_only"
        ),
        "capture_status": media["manifest"].get("capture_status"),
        "unmapped_policy_action_chapters": [
            {key: chapter.get(key) for key in ("round_index", "action_id", "start_frame", "end_frame")}
            for chapter in media.get("chapters", [])
            if chapter.get("action_id") and chapter.get("policy_action_mapped") is False
        ],
        "resolution": media["resolution"], "frame_count": media["frame_count"],
    }


def _selected_rounds(process: Mapping[str, Any], options: VideoOptions) -> list[dict]:
    rows = process["round_ledger"]["rounds"]
    if options.round_index is not None:
        rows = [row for row in rows if row["round_index"] == options.round_index]
        if not rows or rows[0]["outcome"] == "not_run":
            raise ValueError("requested round has no recorded evidence")
    return rows


def render_plan(process: Mapping[str, Any], media: Mapping[str, Any] | None, options: VideoOptions) -> list[dict]:
    """Assign display frames separately from original event/execution time."""
    _selected_rounds(process, options)
    by_action = {item["action_id"]: item for item in (media or {}).get("chapters", []) if item.get("action_id")}
    source_events = process["events"]
    if options.mode == "execution":
        if media is None:
            raise PresentationPendingError("No verified execution video is cached.")
        source_events = [{**chapter, "phase": "execution", "title": "Execution", "source_refs": [], "images": []}
                         for chapter in media["chapters"]]
    latest_images: list[dict] = []
    plans: list[dict] = []
    output_frame = 0
    for event in source_events:
        if options.round_index is not None and event.get("round_index") != options.round_index:
            continue
        phase = event["phase"]
        if phase == "context" and not options.context:
            continue
        if options.mode == "process" and event.get("covered_by_execution"):
            continue
        if phase in {"round_start", "round_end"}:
            latest_images = []
        if phase == "observation":
            latest_images = event.get("images") or []
        elif event.get("images"):
            latest_images = event["images"]
        images = event.get("images") or latest_images
        clip = event if options.mode == "execution" else by_action.get(event.get("action_id")) if phase == "execution" else None
        if clip is not None:
            count = max(1, math.ceil((clip["end_frame"] - clip["start_frame"]) / media["fps"] * FPS - 1e-8))
            plans.append({**event, "display_images": [], "source_start_frame": clip["start_frame"],
                          "source_end_frame": clip["end_frame"], "start_frame": output_frame,
                          "end_frame": output_frame + count, "kind": "clip"})
            output_frame += count
            continue
        pages = max(1, math.ceil(len(images) / 4))
        for page in range(pages):
            seconds = options.card_seconds
            if phase == "observation":
                seconds = min(seconds, 1.0)
            elif phase == "outcome":
                seconds = min(seconds, 1.5)
            count = max(1, round(seconds * FPS))
            plans.append({**event, "display_images": images[page * 4:(page + 1) * 4],
                          "image_page": page + 1, "image_pages": pages,
                          "start_frame": output_frame, "end_frame": output_frame + count, "kind": "card"})
            output_frame += count
    if not plans:
        raise ValueError("there are no recorded events for the requested video")
    return plans


def _manifest(process: Mapping[str, Any], media: Mapping[str, Any] | None, options: VideoOptions, plans: list[dict]) -> dict:
    chapters = [{key: item.get(key) for key in (
        "key", "phase", "round_index", "state_step", "action_id", "title", "source_refs", "issues",
        "start_frame", "end_frame", "source_start_frame", "source_end_frame", "image_page", "image_pages",
    )} for item in plans]
    for chapter in chapters:
        chapter.update(start_s=chapter["start_frame"] / FPS, end_s=chapter["end_frame"] / FPS)
    mapped_ranges = {(item.get("source_start_frame"), item.get("source_end_frame"))
                     for item in plans if item["kind"] == "clip"}
    unplaced = [
        {key: chapter.get(key) for key in ("round_index", "action_id", "start_frame", "end_frame")}
        for chapter in (media or {}).get("chapters", [])
        if chapter.get("action_id") and chapter.get("policy_action_mapped") is False
        and (options.round_index is None or chapter.get("round_index") == options.round_index)
        and (chapter["start_frame"], chapter["end_frame"]) not in mapped_ranges
    ] if options.mode == "process" else []
    return {
        "schema": PRESENTATION_SCHEMA, "renderer_version": RENDERER_VERSION,
        "run_id": process["run_id"], "options": asdict(options), "fps": FPS,
        "frame_count": plans[-1]["end_frame"], "duration_s": plans[-1]["end_frame"] / FPS,
        "resolution": list(FRAME_SIZE), "round_ledger": process["round_ledger"],
        "chapters": chapters, "source_hashes": process["source_hashes"],
        "execution_source": _source_projection(media), "coverage": process["coverage"],
        "issues": [*process["issues"], *(["captured_control_actions_without_policy_join"] if unplaced else [])],
        "unmapped_events": process["unmapped_events"],
        "unplaced_execution_chapters": unplaced,
        "execution_coverage": (
            "exact_policy_action_joins_only; unplaced_control_frames_remain_in_full_execution_video"
            if unplaced else "all_selected_mapped_execution_chapters"
        ),
        "timing": "reading_cards_and_resampled_execution; not_wall_clock_time",
        "evidence_boundary": "read_only_viewer_derived_not_sealed_run_evidence",
        "compile_hint": compile_hint(process["run_id"], options),
    }


def presentation_video(
    root: Path, options: VideoOptions = VideoOptions(), *, allow_compile: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, dict]:
    """Serve cached media or explicitly compile with CPU-only FFmpeg/Pillow.

    Never invokes a Provider, simulator, grounding service or planning tool.
    """
    from viewer.replay import _exclusive_replay_lock
    from viewer.trajectory import compile_trajectory

    root = root.resolve()
    trajectory = compile_trajectory(root)
    process = compile_process_timeline(root, trajectory)
    _selected_rounds(process, options)
    media = find_execution_media(root, trajectory, replay_mode=options.replay_mode)
    if options.mode == "execution" and options.round_index is None and media is not None:
        # Full execution retains the existing media exactly. Chapter metadata
        # adds round navigation even when historical burned-in labels omit it.
        manifest = {
            "schema": PRESENTATION_SCHEMA, "run_id": root.name, "options": asdict(options),
            "fps": media["fps"], "frame_count": media["frame_count"],
            "duration_s": media["frame_count"] / media["fps"], "resolution": media["resolution"],
            "video_sha256": media["sha256"], "chapters": media["chapters"],
            "round_ledger": process["round_ledger"], "execution_source": _source_projection(media),
            "source_hashes": process["source_hashes"], "coverage": process["coverage"],
            "evidence_boundary": "existing_execution_media_with_derived_chapters", "status": "ready",
        }
        return media["path"], manifest
    if options.mode == "execution" and media is None:
        raise PresentationPendingError(compile_hint(root.name, options))
    plans = render_plan(process, media, options)
    manifest = _manifest(process, media, options, plans)
    identity_input = {"process": process, "manifest": manifest, "renderer_sha256": file_hash(Path(__file__))}
    identity = hashlib.sha256(json.dumps(identity_input, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache = derived_root(root)
    output = cache / f"presentation-{options.mode}-{identity[:20]}.mp4"
    manifest_path = output.with_suffix(".json")
    saved = load_object(manifest_path)
    if output.is_file() and saved.get("identity_sha256") == identity and file_hash(output) == saved.get("video_sha256"):
        return output, saved
    if not allow_compile:
        raise PresentationPendingError(compile_hint(root.name, options))
    cache.mkdir(parents=True, exist_ok=True)
    with _exclusive_replay_lock(output):
        saved = load_object(manifest_path)
        if output.is_file() and saved.get("identity_sha256") == identity and file_hash(output) == saved.get("video_sha256"):
            return output, saved
        with tempfile.TemporaryDirectory(prefix="presentation-", dir=cache) as temporary:
            temp = Path(temporary)
            encoded = temp / "video.mp4"
            _encode(root, process, media, options, plans, encoded, progress=progress)
            manifest.update(identity_sha256=identity, video_sha256=file_hash(encoded), status="ready")
            temp_manifest = temp / "manifest.json"
            temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(encoded, output)
            os.replace(temp_manifest, manifest_path)
    return output, manifest


@lru_cache(maxsize=24)
def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    face = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{face}", size)


def _text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], *, width: int, height: int,
          size: int = 23, fill: str = "#f0f0f0", bold: bool = False) -> bool:
    """Pixel-based wrapping, including unspaced strings; report clipping."""
    font = _font(size, bold=bold)
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        line = ""
        for char in paragraph:
            if draw.textlength(line + char, font=font) > width and line:
                split = line.rfind(" ")
                if split > len(line) // 2:
                    lines.append(line[:split])
                    line = line[split + 1:] + char
                else:
                    lines.append(line)
                    line = char
            else:
                line += char
        lines.append(line)
    maximum = max(1, height // (size + 8))
    clipped = len(lines) > maximum
    for index, line in enumerate(lines[:maximum]):
        if index == maximum - 1 and clipped:
            line = line[:-2] + "…"
        draw.text((xy[0], xy[1] + index * (size + 8)), line, font=font, fill=fill)
    return clipped


def render_card(root: Path, process: Mapping[str, Any], segment: Mapping[str, Any], options: VideoOptions) -> Image.Image:
    canvas = Image.new("RGB", FRAME_SIZE, "#0a0a0a")
    draw = ImageDraw.Draw(canvas)
    ink, muted = "#f0f0f0", "#aaaaaa"
    phase = segment["phase"]
    round_index = segment.get("round_index")
    round_label = f"ROUND {round_index + 1} / {process['round_ledger']['planned_rounds']}" if isinstance(round_index, int) else "SESSION"
    draw.line((30, 100, 1890, 100), fill="#c6011f", width=3)
    _text(draw, f"{round_label}   ·   {phase.replace('_', ' ').upper()}", (32, 24), width=1750, height=65, size=34, bold=True)
    images = segment.get("display_images") or []
    _text(draw, str(process.get("instruction") or "Episode"), (32, 124), width=1180, height=80, size=28)
    if images:
        columns = 1 if len(images) == 1 else 2
        rows = math.ceil(len(images) / columns)
        box_width, box_height = 1190 // columns, 640 // rows
        for index, item in enumerate(images):
            x = 32 + index % columns * box_width
            y = 218 + index // columns * box_height
            path, issue = verified_artifact(root, item)
            if path is None:
                _text(draw, f"Image unavailable\n{issue}", (x + 10, y + 45), width=box_width - 20, height=120, size=22)
                continue
            with Image.open(path) as source:
                picture = ImageOps.contain(source.convert("RGB"), (box_width - 18, box_height - 60), Image.Resampling.LANCZOS)
            canvas.paste(picture, (x + (box_width - picture.width) // 2, y + (box_height - 48 - picture.height) // 2))
            _text(draw, f"{item.get('label') or 'recorded image'} · {item.get('visibility') or 'visibility unrecorded'}",
                  (x, y + box_height - 44), width=box_width - 18, height=38, size=15, fill=muted)
    else:
        headline = segment.get("title") or phase.replace("_", " ")
        _text(draw, str(headline), (54, 310), width=1090, height=210, size=68, bold=True)
        if phase == "round_end":
            row = segment.get("round") or {}
            label = str(row.get("outcome") or "unscored").replace("_", " ").upper()
            _text(draw, label, (54, 560), width=1090, height=100, size=60, bold=True,
                  fill="#7fb493" if row.get("official_success") is True else ink)
    draw.line((1260, 130, 1260, 865), fill="#333333", width=1)
    _text(draw, str(segment.get("title") or phase), (1300, 136), width=574, height=100, size=30, bold=True)
    text = str(segment.get("text") or "No public textual response recorded.")
    reasoning = segment.get("reasoning") if options.reasoning else None
    if reasoning:
        text = (text[:240] + "…" if len(text) > 240 else text)
        text += "\n\nAPI-visible reasoning (recorded excerpt)\n" + reasoning
    if phase == "execution":
        text = "ACTION-BOUNDARY FALLBACK\nNo verified continuous execution clip.\n\n" + text
    if phase == "outcome":
        receipt = segment.get("receipt") or {}
        text = "\n".join(f"{key}: {receipt[key]}" for key in (
            "execution_status", "status", "reached", "consumed", "aperture_mm", "terminal_after_action",
        ) if receipt.get(key) is not None) or text
    clipped = _text(draw, text, (1300, 248), width=570, height=480, size=21)
    if clipped or reasoning:
        _text(draw, "Excerpt · complete record in Viewer", (1300, 748), width=570, height=40, size=16, fill=muted)
    if segment.get("accepted") is False:
        _text(draw, "Provider response rejected", (1300, 791), width=570, height=35, size=18)
    elif segment.get("ok") is False:
        _text(draw, "Tool returned an error", (1300, 791), width=570, height=35, size=18)
    elif segment.get("issues"):
        _text(draw, "Evidence gap · inspect original event", (1300, 791), width=570, height=35, size=18)
    draw.line((32, 884, 1888, 884), fill="#333333", width=1)
    source = next((ref for ref in segment.get("source_refs", []) if ref.get("stream")), {})
    address = f"{source['stream']}:{source['source_line']}" if source else segment["key"]
    _text(draw, f"{address}  ·  {segment.get('model_call_id') or ''}", (32, 903), width=1190, height=40, size=17, fill=muted)
    _text(draw, "VIEWER EDIT · reading time, not wall clock", (1300, 903), width=585, height=40, size=16, fill=muted)
    return canvas


def _execution_header(frame: Image.Image, process: Mapping[str, Any], segment: Mapping[str, Any], media: Mapping[str, Any]) -> Image.Image:
    if frame.size != FRAME_SIZE:
        canvas = Image.new("RGB", FRAME_SIZE, "#0a0a0a")
        frame = ImageOps.contain(frame, FRAME_SIZE, Image.Resampling.LANCZOS)
        canvas.paste(frame, ((1920 - frame.width) // 2, (960 - frame.height) // 2))
        frame = canvas
        # Older observation-video layouts have no 640px right sidebar.
        return frame
    if media["source_kind"] == "same_session_execution_capture":
        return frame
    draw = ImageDraw.Draw(frame)
    draw.rectangle((1282, 0, 1919, 125), fill="#0a0a0a")
    label = str(media["source_kind"]).replace("viewer_derived_", "DERIVED ").replace("_", " ").upper()
    _text(draw, label, (1310, 18), width=580, height=40, size=16, fill="#aaaaaa")
    ri = segment.get("round_index")
    prefix = f"ROUND {ri + 1}/{process['round_ledger']['planned_rounds']}" if isinstance(ri, int) else "SESSION"
    actions = [event for event in process["events"] if event["phase"] == "execution" and event.get("round_index") == ri]
    ordinal = next((i + 1 for i, action in enumerate(actions) if action.get("action_id") == segment.get("action_id")), None)
    label = f"{prefix} · ACTION {ordinal}/{len(actions)}" if ordinal else f"{prefix} · RESET"
    _text(draw, label, (1310, 58), width=580, height=50, size=29, bold=True)
    return frame


def _read_exact(stream, count: int) -> bytes:
    parts = bytearray()
    while len(parts) < count:
        block = stream.read(count - len(parts))
        if not block:
            raise RuntimeError("execution video ended before its declared frame range")
        parts.extend(block)
    return bytes(parts)


def _encode(root: Path, process: Mapping[str, Any], media: Mapping[str, Any] | None, options: VideoOptions,
            plans: list[dict], output: Path, *, progress: Callable[[str], None] | None) -> None:
    encoder = decoder = None
    if media is not None and any(segment["kind"] == "clip" for segment in plans):
        source_info = _probe(media["path"])
        if (source_info["frames"] != media["frame_count"] or source_info["resolution"] != media["resolution"]
                or not math.isclose(source_info["fps"], media["fps"], rel_tol=1e-6)):
            raise ValueError("execution MP4 does not match its recorded timeline dimensions/rate/count")
    with tempfile.TemporaryFile() as encode_log, tempfile.TemporaryFile() as decode_log:
        try:
            encoder = subprocess.Popen([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", "1920x960", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-threads", "2",
                "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
            ], stdin=subprocess.PIPE, stderr=encode_log)
            source_index = -1
            source_bytes = b""
            previous_progress = -1
            for index, segment in enumerate(plans):
                if segment["kind"] == "card":
                    rendered = render_card(root, process, segment, options).tobytes()
                    for _ in range(segment["end_frame"] - segment["start_frame"]):
                        encoder.stdin.write(rendered)
                else:
                    if media is None:
                        raise RuntimeError("execution segment has no source media")
                    dimensions = media.get("resolution")
                    if not isinstance(dimensions, list) or len(dimensions) != 2 or dimensions not in ([1920, 960], [1280, 720]):
                        raise ValueError("execution video layout is unsupported")
                    if decoder is None:
                        first_source_frame = segment["source_start_frame"]
                        decoder = subprocess.Popen([
                            "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-i", str(media["path"]),
                            "-an", "-vf", f"select=gte(n\\,{first_source_frame})", "-vsync", "0",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-threads", "2", "-",
                        ], stdout=subprocess.PIPE, stderr=decode_log)
                        source_index = first_source_frame - 1
                    source_frame_bytes = dimensions[0] * dimensions[1] * 3
                    for local in range(segment["end_frame"] - segment["start_frame"]):
                        wanted = min(segment["source_end_frame"] - 1, segment["source_start_frame"] + int(local * media["fps"] / FPS))
                        if wanted < source_index:
                            raise ValueError("execution clips must retain source order")
                        while source_index < wanted:
                            source_bytes = _read_exact(decoder.stdout, source_frame_bytes)
                            source_index += 1
                        frame = Image.frombytes("RGB", tuple(dimensions), source_bytes)
                        frame = _execution_header(frame, process, segment, media)
                        encoder.stdin.write(frame.tobytes())
                percent = (index + 1) * 10 // len(plans)
                if progress and percent != previous_progress:
                    previous_progress = percent
                    progress(f"{percent * 10}% · {index + 1}/{len(plans)} presentation segments")
            encoder.stdin.close()
            if encoder.wait() != 0:
                encode_log.seek(0)
                raise RuntimeError("video encode failed: " + encode_log.read().decode(errors="replace")[-1600:])
            encoded_info = _probe(output)
            if encoded_info["frames"] != plans[-1]["end_frame"] or encoded_info["resolution"] != list(FRAME_SIZE) or encoded_info["fps"] != FPS:
                raise RuntimeError("encoded video does not match its chapter timeline")
        finally:
            for process_handle in (decoder, encoder):
                if process_handle is not None:
                    if process_handle.poll() is None:
                        process_handle.kill()
                    process_handle.wait()
                    for stream in (process_handle.stdin, process_handle.stdout):
                        if stream and not stream.closed:
                            stream.close()


def _probe(path: Path) -> dict:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(completed.stdout)["streams"][0]
    return {"frames": int(stream["nb_frames"]), "fps": float(Fraction(stream["avg_frame_rate"])),
            "resolution": [stream["width"], stream["height"]]}
