"""Bounded-batch ``mark_point``: one call, up to eight pixels, one overlay.

Two things are being defended here at once.

The first is the *new* batched input form and the module it now lives in.  A
measurement pass over 222 sealed runs found 379 model turns that issued several
``mark_point`` calls against one unchanged observation; at a median 258,193
provider input tokens per request against 1,476 output, each of those extra
turns bought a re-read of the whole context for one pixel.  The batch collapses
them into one turn, and the tests below pin the two delivery ceilings that
decide whether that is worth anything: the 50,000-byte result threshold above
which the projection layer keeps only a 2,048-byte prefix and tells the model to
call again, and the per-block image eviction that makes an index-position image
convention unsafe.

The second is the *old* parallel-call aggregation --
``begin_read_only_feedback_batch`` / ``finish_read_only_feedback_batch`` and the
generic ``_execute_aggregated_read_only_batch`` that drives them.  A proposal to
batch the input claimed those three become dead and can be deleted for a net
-450 lines.  The four aggregation tests below exist to answer that claim with
evidence rather than with reading: they drive the path end to end, through the
Native session, on a surface that declares ``batched_mark_overlay_v1``.  They
pass today, which is the proof that the functions are reachable -- the
Synthetic Native surface ``native_atomic_causal_v1`` declares that
capability -- so nothing was deleted.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ora.agent.composition.tool_surface import (
    PROFILE_CATALOG,
    validate_schema_instance,
)
from ora.agent.context import ContextProjector
from ora.agent.providers.native import FakeProvider
from ora.agent.tools import mark_point
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError
from ora.environment import SyntheticSession

ROOT = Path(__file__).resolve().parents[1]
MANUAL = ROOT / "ora/policy/embodiments/aloha-agilex/manual-v1.md"
PROMPT = ROOT / "tests/data/prompts/startup-prompt-v1.md"
AGENTS = ROOT / "ora/policy/planners/native/project-v1.md"
BOUNDS = {"x": [-0.6, 0.6], "y": [-0.4, 0.4], "z": [0.7, 1.45]}

# The aggregation path is only armed on a surface whose hash-bound manifest
# declares it, so a reachability test must name one of those surfaces.
AGGREGATION_CAPABILITY = "batched_mark_overlay_v1"
AGGREGATING_PROFILE = "native_atomic_causal_v1"


def _runtime(
    tmp_path: Path, *, profile: str = "native_atomic_causal_v1",
) -> tuple[ManipulationToolRuntime, Path, Mapping[str, Any]]:
    source = tmp_path / "source"
    session = SyntheticSession(
        tmp_path / "native", "openeta_synthetic", 0,
        main_camera="agentview", camera_names=("agentview", "wrist"),
    )
    session.start()
    observation, _ = session.observe(source)
    control = tmp_path / "control"
    projector = ContextProjector(
        tmp_path / "workspace", control, PROMPT, MANUAL, AGENTS,
        expose_manual=False, main_camera="agentview",
    )
    projector.publish(
        observation=observation,
        observation_root=source / observation["observation_id"],
        task_instruction="test",
        remaining_budget={"robot_decisions": 2, "model_calls": 8},
        policy_state={
            "step": 0, "path": "unused",
            "sha256": "unused", "state_sha256": "unused",
        },
    )
    runtime = ManipulationToolRuntime(
        control_root=control, manual_path=MANUAL,
        event_path=tmp_path / "events.jsonl",
        workspace_bounds_m=BOUNDS, state_root=None,
        tool_surface_profile=profile,
    )
    pointer = json.loads(
        (tmp_path / "workspace/inputs/current-context.json").read_text(
            encoding="utf-8",
        )
    )
    return runtime, source, pointer


def _assistant_tools(calls: Sequence[tuple[str, str, dict]]) -> dict:
    return {
        "role": "assistant", "content": "", "tool_calls": [
            {
                "id": call_id, "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, sort_keys=True),
                },
            }
            for call_id, name, arguments in calls
        ],
    }


# --- Capture what the three aggregation functions do before deleting them.
def test_a_registered_surface_still_declares_the_aggregation_capability() -> None:
    """The gate is a manifest capability, so the manifests are the evidence."""

    declaring = sorted(
        profile
        for profile, path in PROFILE_CATALOG.items()
        if AGGREGATION_CAPABILITY in json.loads(
            path.read_text(encoding="utf-8"),
        ).get("capabilities", [])
    )
    assert declaring, (
        "no registered profile declares "
        f"{AGGREGATION_CAPABILITY}; only then is the aggregation path dead"
    )
    assert AGGREGATING_PROFILE in declaring


def test_begin_and_finish_render_one_overlay_for_parallel_single_calls(
    tmp_path: Path,
) -> None:
    """This is the behavior the deletion proposal called unreachable."""

    runtime, _source, _pointer = _runtime(
        tmp_path, profile=AGGREGATING_PROFILE,
    )
    calls = [
        {"point_id": "P0", "view": "agentview", "x": 256, "y": 256},
        {"point_id": "P1", "view": "agentview", "x": 250, "y": 250},
    ]
    assert runtime.begin_read_only_feedback_batch("mark_point", calls) is True

    results = [runtime.invoke("mark_point", call) for call in calls]
    assert [item["status"] for item in results] == ["solved", "solved"]
    assert all("images_base64" not in item for item in results), (
        "an active aggregation must suppress the per-call overlay"
    )

    feedback = runtime.finish_read_only_feedback_batch("mark_point", results)
    assert list(feedback) == ["images_base64"]
    assert len(feedback["images_base64"]) == 1


def test_a_single_call_outside_the_aggregation_still_carries_its_overlay(
    tmp_path: Path,
) -> None:
    """The suppression is scoped to an active batch, not to the profile."""

    runtime, _source, _pointer = _runtime(
        tmp_path, profile=AGGREGATING_PROFILE,
    )
    solved = runtime.invoke("mark_point", {
        "point_id": "P0", "view": "agentview", "x": 256, "y": 256,
    })
    assert len(solved["images_base64"]) == 1


def test_the_native_session_reaches_the_aggregation_from_one_response(
    tmp_path: Path,
) -> None:
    """Two parallel read-only calls in one response drive the generic path."""

    from native_synthetic_spec import synthetic_spec
    from ora.orchestrator.coordinator import run_episode
    from ora.orchestrator.spec import resolve_run_spec

    spec = synthetic_spec(seed=0)
    spec["policy"]["context_retention"] = "rewrite_compact_v1"
    assert spec["policy"]["tool_surface"]["profile"] == AGGREGATING_PROFILE
    plan = resolve_run_spec(spec, root=ROOT)
    root = tmp_path / "run"
    root.mkdir()
    (root / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    provider = FakeProvider([
        {"message": _assistant_tools([
            ("mark-a", "mark_point", {
                "point_id": "P0", "view": "wrist", "x": 2, "y": 1,
            }),
            ("mark-b", "mark_point", {
                "point_id": "P1", "view": "wrist", "x": 1, "y": 1,
            }),
        ])},
        {"message": _assistant_tools([
            ("finish-1", "finish_episode", {
                "outcome": "failure", "reason": "aggregation probe complete",
                "attempts": ["marked two points in one response"],
            }),
        ])},
    ])
    run_episode(run_root=root, plan=plan, run_id="mark-point-aggregation",
                provider_override=provider)

    events = [
        json.loads(line)
        for line in (root / "agent-events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    combined = [
        event for event in events
        if event.get("type") == "tool_call_proposed"
        and event.get("payload", {}).get("visual_feedback") == "combined"
    ]
    assert len(combined) == 2, (
        "both parallel mark_point calls must be routed through the "
        "aggregated read-only batch"
    )


def _batched(runtime: ManipulationToolRuntime) -> ManipulationToolRuntime:
    """Arm the batch capability with the contract a shipped surface resolves to.

    The synthetic fixture cannot construct an explicit-motion-v3 runtime -- that
    profile requires a verified robot-geometry observation and a locked Servo
    binding -- so the capability and result schema are lifted onto this runtime
    instead. They are read back out of the resolved default surface rather than
    from the module constant, so every end-to-end assertion below is made
    against what a policy will actually be handed.
    """

    from ora.agent.composition.tool_surface import resolve_tool_surface

    shipped = resolve_tool_surface("libero_placement_direct_overhead_v1")
    runtime.surface = dataclasses.replace(
        runtime.surface,
        capabilities=(
            *runtime.surface.capabilities, mark_point.MARK_POINT_BATCH_CAPABILITY,
        ),
        result_schemas={
            **runtime.surface.result_schemas,
            "mark_point": shipped.result_schemas["mark_point"],
        },
    )
    return runtime


def _schema_nodes(node: Any, path: str = "$") -> list[tuple[str, Mapping[str, Any]]]:
    found: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(node, Mapping):
        found.append((path, node))
        for key in ("properties", "items"):
            child = node.get(key)
            if isinstance(child, Mapping) and key == "items":
                found.extend(_schema_nodes(child, f"{path}[]"))
            elif isinstance(child, Mapping):
                for name, value in child.items():
                    found.extend(_schema_nodes(value, f"{path}.{name}"))
    return found


# --- The two limits that decide whether a batched result arrives at all.
def test_eight_maximal_marks_stay_far_under_the_fifty_kilobyte_cliff() -> None:
    """Measure the largest result the schema permits, do not estimate it.

    Above ``eager_large_result_bytes`` (50,000 in the sealed run
    configuration) the projection layer keeps only an ``eager_preview_bytes``
    prefix -- 2,048 bytes, which at these entry sizes would not reach the
    second mark -- and tells the model to call again. A batch that trips that
    is worse than no batch: the model pays for the call and gets back its
    envelope. So the bound is measured against a result built by saturating
    every declared maximum, not estimated.
    """

    schema = mark_point.MARK_POINT_BATCH_RESULT_SCHEMA
    limit = mark_point.MARK_POINT_MAX_BATCH
    name = "n" * 64
    view = "v" * 32
    marks = [
        {
            "source_index": index,
            "point_id": name,
            "point_uv": [9999, 9999],
            "status": "verified",
            "error_code": "invalid_camera_calibration",
            "error_message": "e" * 200,
            "xyz_m": [-0.12345678901234567] * 3,
            "views": [view] * 8,
            "next_views": [view] * 3,
            "colour_rgb": [255, 255, 255],
            "verification_distance_m": 0.024999999999999998,
            "quality_method": "local_supported_depth_v1",
            "quality_status": "measured",
            "quality_depth_m": 1.2345678901234567,
            "quality_window_radius_px": 99,
            "quality_valid_samples": 9999,
            "quality_supporting_samples": 9999,
            "quality_rejected_samples": 9999,
            "quality_max_support_deviation_m": 0.12345678901234567,
            "action_workspace_inside": False,
            "action_workspace_use": (
                "measurement_only_outside_action_envelope"
            ),
        }
        for index in range(limit)
    ]
    maximal = {
        "success": True,
        "status": "partial",
        "view": view,
        "observation_id": "o" * 64,
        "mark_count": limit,
        "solved_count": limit,
        "failed_count": 0,
        "marks": marks,
        "image_mapping": [
            {
                "image_index": index,
                "role": "mark_batch_panel",
                "camera": view,
                "source_index": index,
                "candidate_id": None,
                "target_ref": None,
                "observation_step": 999999,
            }
            for index in range(limit)
        ],
        "measurement_contract": "c" * 512,
    }
    # A strawman maximum proves nothing: the worst case has to be a result
    # the contract would actually accept.
    validate_schema_instance(maximal, schema)

    encoded = len(json.dumps(maximal, ensure_ascii=False).encode("utf-8"))
    per_mark = len(json.dumps(marks[0], ensure_ascii=False).encode("utf-8"))
    # 1,398 B per entry x 8 = 11,184, plus 8 image bindings at 181 B =
    # 1,448, plus an 797 B envelope: 13,457 B, 26.9% of the threshold.
    assert per_mark <= 1_400, per_mark
    assert encoded <= 13_500, encoded
    # Three times the headroom, so a later field addition cannot quietly
    # walk the result over the cliff without this failing first.
    assert encoded * 3 < 50_000, encoded


def test_a_realistic_eight_mark_result_costs_about_a_tenth_of_the_cliff(
    tmp_path: Path,
) -> None:
    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": [f"p{index}" for index in range(8)],
        "points_uv": [[250 + index, 250 + index] for index in range(8)],
    })
    visible = {
        key: value for key, value in result.items()
        if key not in {"images_base64", "image_base64", "_runtime"}
    }
    assert len(json.dumps(visible, ensure_ascii=False).encode("utf-8")) < 8_000


def test_eight_marks_cost_one_image_slot_not_eight(
    tmp_path: Path,
) -> None:
    """The inline-image window is 24 wide with a measured median of 12."""

    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": [f"p{index}" for index in range(8)],
        "points_uv": [[250 + index, 250 + index] for index in range(8)],
    })
    assert len(result["images_base64"]) == 1
    assert len(result["image_mapping"]) == 1


def test_attribution_survives_the_image_being_dropped(
    tmp_path: Path,
) -> None:
    """Images are evicted one block at a time, so colour lives in the body.

    ``image_mapping`` never claims a ``source_index`` for a panel that
    depicts several marks: a reader that assumed "image i belongs to input
    i" would misattribute every survivor once one block was replaced.
    """

    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": ["a", "b", "c"],
        "points_uv": [[250, 250], [251, 251], [252, 252]],
    })
    assert result["image_mapping"][0]["source_index"] is None
    colours = [mark["colour_rgb"] for mark in result["marks"]]
    assert all(colour is not None for colour in colours)
    assert len({tuple(colour) for colour in colours}) == len(colours), (
        "two marks drawn in one colour cannot be told apart in the overlay"
    )
    # The colours are in the result body, which no image eviction touches.
    body = json.dumps({
        key: value for key, value in result.items() if key != "images_base64"
    })
    assert all(json.dumps(colour) in body for colour in colours)


# --- Eight is bound to the colour table, not chosen.
def test_the_bound_is_the_length_of_the_colour_table() -> None:
    assert mark_point.MARK_POINT_MAX_BATCH == len(
        mark_point.MARK_POINT_BATCH_COLOURS
    )
    assert mark_point.MARK_POINT_MAX_BATCH == 8


def test_the_schemas_publish_that_same_bound() -> None:
    limit = mark_point.MARK_POINT_MAX_BATCH
    schema = mark_point.batch_input_schema(["agentview", "wrist"])
    assert schema["properties"]["point_ids"]["maxItems"] == limit
    assert schema["properties"]["points_uv"]["maxItems"] == limit
    result = mark_point.MARK_POINT_BATCH_RESULT_SCHEMA
    assert result["properties"]["marks"]["maxItems"] == limit
    assert result["properties"]["mark_count"]["maximum"] == limit


def test_a_ninth_pixel_is_refused_with_the_reason(
    tmp_path: Path,
) -> None:
    runtime = _batched(_runtime(tmp_path)[0])
    with pytest.raises(ToolError) as refused:
        runtime.invoke("mark_point", {
            "view": "agentview",
            "point_ids": [f"p{index}" for index in range(9)],
            "points_uv": [[250, 250]] * 9,
        })
    assert refused.value.code == "invalid_arguments"
    assert refused.value.field == "point_ids"
    assert refused.value.received == 9
    assert refused.value.constraints["maximum"] == 8
    assert "colour" in str(refused.value)


def test_the_overlay_never_reuses_a_colour_within_one_call(
    tmp_path: Path,
) -> None:
    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": [f"p{index}" for index in range(8)],
        "points_uv": [[250 + index, 250 + index] for index in range(8)],
    })
    drawn = [tuple(mark["colour_rgb"]) for mark in result["marks"]]
    assert len(set(drawn)) == 8
    assert set(drawn) == set(mark_point.MARK_POINT_BATCH_COLOURS)


# --- One bad pixel costs its own entry and nothing else.
def test_marks_match_the_request_in_length_and_order(
    tmp_path: Path,
) -> None:
    runtime = _batched(_runtime(tmp_path)[0])
    names = ["rim_left", "rim_right", "handle"]
    pixels = [[250, 250], [251, 252], [253, 254]]
    result = runtime.invoke("mark_point", {
        "view": "agentview", "point_ids": names, "points_uv": pixels,
    })
    assert [mark["point_id"] for mark in result["marks"]] == names
    assert [mark["point_uv"] for mark in result["marks"]] == pixels
    assert [mark["source_index"] for mark in result["marks"]] == [0, 1, 2]
    assert result["mark_count"] == 3


def test_an_unmeasurable_pixel_fails_only_its_own_entry(
    tmp_path: Path,
) -> None:
    """All-or-nothing would make a batch strictly worse than three calls."""

    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": ["good_a", "off_frame", "good_b"],
        "points_uv": [[250, 250], [99999, 99999], [252, 252]],
    })
    assert result["success"] is True
    assert result["status"] == "partial"
    assert result["solved_count"] == 2 and result["failed_count"] == 1
    first, bad, last = result["marks"]
    assert first["status"] == "solved" and first["xyz_m"] is not None
    assert last["status"] == "solved" and last["xyz_m"] is not None
    assert bad["status"] == "failed"
    assert bad["xyz_m"] is None
    # The batch reports exactly the code the single-pixel form raises for
    # the same condition; error_message carries what the code does not.
    assert bad["error_code"] == "invalid_arguments"
    # The entry names the click and the view's size (census C3).
    assert "is outside the 512x512 agentview view" in bad["error_message"]
    assert bad["source_index"] == 1, (
        "a failed entry keeps its place so the survivors are not renumbered"
    )


def test_every_entry_carries_its_depth_evidence_including_failures(
    tmp_path: Path,
) -> None:
    """The single-pixel form drops this on its error path; a batch cannot.

    A model correcting several pixels at once has to distinguish "this
    pixel saw the far plane" from "this pixel straddles a silhouette",
    because the corrections differ.
    """

    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": ["solid", "off_frame"],
        "points_uv": [[250, 250], [99999, 99999]],
    })
    solved, failed = result["marks"]
    quality_keys = [
        "quality_method", "quality_status", "quality_depth_m",
        "quality_window_radius_px", "quality_valid_samples",
        "quality_supporting_samples", "quality_rejected_samples",
        "quality_max_support_deviation_m",
    ]
    for mark in (solved, failed):
        assert all(key in mark for key in quality_keys), mark

    assert solved["quality_status"] == "measured"
    assert solved["quality_method"] == "local_supported_depth_v1"
    assert solved["quality_depth_m"] is not None
    assert solved["quality_supporting_samples"] >= 3
    assert solved["quality_max_support_deviation_m"] is not None

    # This pixel never reached the estimator, so it says so rather than
    # inventing counters.
    assert failed["quality_status"] == "unmeasured"
    assert failed["quality_method"] == "not_measured"
    assert failed["quality_depth_m"] is None


def test_a_far_plane_pixel_reports_what_the_estimator_saw(
    tmp_path: Path,
) -> None:
    """A rejected depth still has diagnostics; the single form discards them."""

    import numpy as np

    from ora.agent.tools.mark_point import robust_depth_measurement

    calibration = {"valid_depth_range_m": {"near": 0.1, "far": 10.0}}
    sink: dict[str, Any] = {}
    with pytest.raises(ToolError) as far_plane:
        robust_depth_measurement(
            np.full((5, 5), 9.9, dtype=np.float32),
            source_x=2, source_y=2, calibration=calibration,
            quality_sink=sink,
        )
    assert far_plane.value.code == "invalid_depth"
    assert sink["quality_status"] == "partial"
    assert sink["quality_depth_m"] == pytest.approx(9.9, abs=1e-6)

    edge = np.full((5, 5), 3.0, dtype=np.float32)
    edge[1:4, 1:4] = 1.0
    sink = {}
    with pytest.raises(ToolError) as discontinuity:
        robust_depth_measurement(
            edge, source_x=2, source_y=2, calibration=calibration,
            quality_sink=sink,
        )
    assert discontinuity.value.code == "depth_discontinuity"
    assert sink["quality_rejected_samples"] > 0, (
        "the counter is what tells the model to aim further inside"
    )


def test_omitting_the_sink_leaves_the_estimator_exactly_as_it_was() -> None:
    import numpy as np

    from ora.agent.tools.mark_point import robust_depth_measurement

    calibration = {"valid_depth_range_m": {"near": 0.1, "far": 10.0}}
    measured, quality = robust_depth_measurement(
        np.ones((5, 5), dtype=np.float32), source_x=2, source_y=2,
        calibration=calibration,
    )
    assert measured == pytest.approx(1.0)
    assert quality["supporting_samples"] == 25
    assert quality["method"] == "local_supported_depth_v1"


# --- A malformed call is rejected as a call; a bad pixel is not.
def test_mismatched_lengths_reject_the_whole_call(
    tmp_path: Path,
) -> None:
    runtime = _batched(_runtime(tmp_path)[0])
    with pytest.raises(ToolError) as rejected:
        runtime.invoke("mark_point", {
            "view": "agentview",
            "point_ids": ["a", "b"],
            "points_uv": [[250, 250]],
        })
    assert rejected.value.code == "invalid_arguments"


def test_a_repeated_name_rejects_the_whole_call(
    tmp_path: Path,
) -> None:
    """Two entries naming one point would verify it against itself."""

    runtime = _batched(_runtime(tmp_path)[0])
    with pytest.raises(ToolError) as rejected:
        runtime.invoke("mark_point", {
            "view": "agentview",
            "point_ids": ["rim", "rim"],
            "points_uv": [[250, 250], [251, 251]],
        })
    assert rejected.value.code == "invalid_arguments"
    assert "distinct" in str(rejected.value)


def test_a_point_cloud_view_stays_on_the_single_pixel_form(
    tmp_path: Path,
) -> None:
    """Axis accumulation is order dependent across views, so it cannot batch."""

    runtime = _batched(_runtime(tmp_path)[0])
    with pytest.raises(ToolError) as rejected:
        runtime.invoke("mark_point", {
            "view": "pointcloud_top",
            "point_ids": ["a"], "points_uv": [[250, 250]],
        })
    assert rejected.value.code == "invalid_arguments"
    assert rejected.value.field == "view"
    assert rejected.value.received == "pointcloud_top"


def test_a_malformed_pixel_rejects_the_call_before_measuring_anything(
    tmp_path: Path,
) -> None:
    runtime = _batched(_runtime(tmp_path)[0])
    with pytest.raises(ToolError) as rejected:
        runtime.invoke("mark_point", {
            "view": "agentview",
            "point_ids": ["a", "b"],
            "points_uv": [[250, 250], [250.5, 250]],
        })
    assert rejected.value.code == "invalid_arguments"
    assert not (tmp_path / "control/openeta-points.json").is_file(), (
        "a rejected call must not leave half a batch measured"
    )


# --- Every surface that has not opted in behaves exactly as before.
def test_the_four_field_call_is_unchanged(tmp_path: Path) -> None:
    runtime, _source, _pointer = _runtime(tmp_path)
    solved = runtime.invoke("mark_point", {
        "point_id": "P0", "view": "agentview", "x": 256, "y": 256,
    })
    assert solved["success"] is True
    assert solved["status"] == "solved"
    assert solved["point_id"] == "P0"
    assert len(solved["xyz_m"]) == 3
    assert solved["views"] == ["agentview"]
    assert len(solved["images_base64"]) == 1
    assert "marks" not in solved and "source_index" not in solved


def test_one_pixel_on_an_opted_in_surface_is_a_one_entry_batch(
    tmp_path: Path,
) -> None:
    """A surface cannot publish both result shapes under one schema.

    Admitting the four-field result and the batched result from one
    contract would need a top-level ``oneOf``, which the closed-surface
    rule forbids. So a surface that declares the batch capability expresses
    a single pixel as a batch of one, and every surface that does not
    declare it keeps the four-field form untouched. The runtime accepts
    both; only the published contract has to choose.
    """

    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview", "point_ids": ["only"], "points_uv": [[256, 256]],
    })
    assert result["mark_count"] == 1 and result["status"] == "solved"
    assert result["marks"][0]["point_id"] == "only"
    assert len(result["images_base64"]) == 1
    assert result["image_mapping"][0]["source_index"] == 0, (
        "a panel showing exactly one mark can name which one it is"
    )


def test_a_surface_without_the_capability_refuses_the_batch(
    tmp_path: Path,
) -> None:
    runtime, _source, _pointer = _runtime(tmp_path)
    assert not runtime.surface.supports(
        mark_point.MARK_POINT_BATCH_CAPABILITY
    )
    with pytest.raises(ToolError) as refused:
        runtime.invoke("mark_point", {
            "view": "agentview",
            "point_ids": ["a"], "points_uv": [[250, 250]],
        })
    assert refused.value.code == "invalid_arguments"
    assert refused.value.constraints["accepted_fields"] == [
        "point_id", "view", "x", "y",
    ]


#: The active default surfaces, and the only ones that may publish the batch.
def _maintained_generation() -> set[str]:
    """Every profile composed from the `explicit_motion_v3` components.

    Derived, not listed. The closed manipulation arm hash-reuses these exact
    files, so it inherits whatever the generation publishes without appearing
    in any literal -- and a name list is edited by whoever adds a sibling,
    which is the edit that gets forgotten.
    """
    import json as _json

    from ora.agent.composition.tool_surface import PROFILE_CATALOG as _CATALOG

    found: set[str] = set()
    for profile, path in _CATALOG.items():
        if path.parent.name == "explicit_motion_v3":
            found.add(profile)
            continue
        manifest = _json.loads(path.read_text(encoding="utf-8"))
        refs = [manifest.get("base_definitions") or "", manifest.get("additional_definitions") or ""]
        refs += list((manifest.get("components") or {}).values())
        if any("explicit_motion_v3/" in str(ref) for ref in refs):
            found.add(profile)
    return found


BATCH_PROFILES = _maintained_generation()


def test_exactly_the_active_default_surfaces_declare_the_batch() -> None:
    """The surface JSON is owned elsewhere and binds components by sha256.

    Every profile of the maintained generation publishes the batched form, so
    fresh configurations on either backend inherit it and so does the closed
    sibling that hash-reuses the same components; every other registered
    profile is frozen or source-bound and keeps the four-field form it was
    sealed with.
    """

    declaring = {
        profile
        for profile, path in PROFILE_CATALOG.items()
        if mark_point.MARK_POINT_BATCH_CAPABILITY in json.loads(
            path.read_text(encoding="utf-8"),
        ).get("capabilities", [])
    }
    assert declaring == _maintained_generation()


@pytest.mark.parametrize("profile", sorted(BATCH_PROFILES))
def test_a_declaring_surface_binds_exactly_the_published_contract(profile: str) -> None:
    """The schemas have one owner; a surface may bind them but not retype them."""

    from ora.agent.composition.tool_surface import resolve_tool_surface

    surface = resolve_tool_surface(profile)
    definition = next(item for item in surface.definitions if item["name"] == "mark_point")
    views = definition["inputSchema"]["properties"]["view"]["enum"]
    assert definition["inputSchema"] == mark_point.batch_input_schema(views)
    assert surface.result_schemas["mark_point"] == mark_point.MARK_POINT_BATCH_RESULT_SCHEMA
    assert not set(views) & set(mark_point.POINTCLOUD_VIEWS), (
        "a point-cloud view accumulates axis constraints across views and the batch "
        "refuses it, so publishing one would advertise a call the runtime rejects"
    )


def test_a_declaring_surface_refuses_the_single_pixel_form(tmp_path: Path) -> None:
    """It is not served silently, because its result would fail the bound schema.

    The published contract is the batched one. Measuring a single pixel through
    the historical path would return the four-field result, which the surface's
    own result schema rejects -- and a result-schema failure reaches the policy
    as a non-retryable internal fault, not as something it can correct. The
    refusal names the three fields this surface accepts instead.
    """

    runtime = _batched(_runtime(tmp_path)[0])
    with pytest.raises(ToolError) as refused:
        runtime.invoke("mark_point", {
            "point_id": "P0", "view": "agentview", "x": 250, "y": 250,
        })
    assert refused.value.code == "invalid_arguments"
    assert refused.value.constraints["accepted_fields"] == [
        "view", "point_ids", "points_uv",
    ]
    assert refused.value.retryable is True
    # The same pixel, spelled as the batch of one the contract publishes.
    measured = runtime.invoke("mark_point", {
        "view": "agentview", "point_ids": ["P0"], "points_uv": [[250, 250]],
    })
    assert measured["mark_count"] == 1 and measured["marks"][0]["point_id"] == "P0"


def test_an_unrecognised_shape_still_names_the_four_fields(
    tmp_path: Path,
) -> None:
    runtime, _source, _pointer = _runtime(tmp_path)
    with pytest.raises(ToolError, match="point_id, view, x and y"):
        runtime.invoke("mark_point", {"point_id": "P0", "view": "agentview"})


# --- The closed-surface rules, checked rather than asserted in prose.
@pytest.mark.parametrize("which", ["input", "result"])

def test_no_reference_indirection_anywhere(which: str) -> None:
    """A dangling $ref cost run H6-012 an entire tool call."""

    schema = (
        mark_point.batch_input_schema(["agentview", "wrist"])
        if which == "input" else mark_point.MARK_POINT_BATCH_RESULT_SCHEMA
    )
    encoded = json.dumps(schema)
    assert "$ref" not in encoded and "$defs" not in encoded
    assert "oneOf" not in schema and "anyOf" not in schema

@pytest.mark.parametrize("which", ["input", "result"])

def test_every_object_closes_and_every_array_is_bounded(
    which: str,
) -> None:
    schema = (
        mark_point.batch_input_schema(["agentview", "wrist"])
        if which == "input" else mark_point.MARK_POINT_BATCH_RESULT_SCHEMA
    )
    for path, node in _schema_nodes(schema):
        types = node.get("type")
        types = types if isinstance(types, list) else [types]
        if "object" in types:
            assert node.get("additionalProperties") is False, path
        if "array" in types:
            assert isinstance(node.get("maxItems"), int), path

@pytest.mark.parametrize("which", ["input", "result"])

def test_every_property_describes_its_units_and_frame(
    which: str,
) -> None:
    schema = (
        mark_point.batch_input_schema(["agentview", "wrist"])
        if which == "input" else mark_point.MARK_POINT_BATCH_RESULT_SCHEMA
    )
    for path, node in _schema_nodes(schema):
        if path == "$":
            continue
        description = node.get("description")
        assert isinstance(description, str) and description.strip(), path


def test_the_input_carries_no_arrays_of_objects() -> None:
    """Fixed-length numeric vectors and strings; never a record array."""

    schema = mark_point.batch_input_schema(["agentview", "wrist"])
    for path, node in _schema_nodes(schema):
        if path == "$":
            continue
        types = node.get("type")
        types = types if isinstance(types, list) else [types]
        assert "object" not in types, path


def test_the_input_view_enum_is_closed_and_excludes_point_clouds() -> None:
    schema = mark_point.batch_input_schema(["agentview", "wrist"])
    assert schema["properties"]["view"]["enum"] == ["agentview", "wrist"]
    with pytest.raises(ToolError):
        mark_point.batch_input_schema([])
    with pytest.raises(ToolError):
        mark_point.batch_input_schema(["agentview", "agentview"])


# --- The point of the change: the same evidence, one turn instead of eight.
def test_one_batched_call_stores_what_eight_single_calls_store(
    tmp_path: Path,
) -> None:
    """Across the sealed cohort, 379 turns paid ~258k input tokens each to
    issue the second and later ``mark_point`` call of one observation. The
    batch is only worth taking if it measures the same thing.
    """

    pixels = [[250 + index, 250 + index] for index in range(8)]
    names = [f"p{index}" for index in range(8)]

    one_at_a_time, _source, _pointer = _runtime(tmp_path / "serial")
    for name, (u, v) in zip(names, pixels):
        one_at_a_time.invoke("mark_point", {
            "point_id": name, "view": "agentview", "x": u, "y": v,
        })
    serial = json.loads(
        (tmp_path / "serial/control/openeta-points.json").read_text(
            encoding="utf-8",
        )
    )

    batched = _batched(_runtime(tmp_path / "batch")[0])
    result = batched.invoke("mark_point", {
        "view": "agentview", "point_ids": names, "points_uv": pixels,
    })
    together = json.loads(
        (tmp_path / "batch/control/openeta-points.json").read_text(
            encoding="utf-8",
        )
    )

    assert {name: serial[name]["xyz_m"] for name in names} == {
        name: together[name]["xyz_m"] for name in names
    }
    assert [mark["xyz_m"] for mark in result["marks"]] == [
        serial[name]["xyz_m"] for name in names
    ]
    assert all(
        together[name]["sources"][0]["measurement_quality"]["method"]
        == "local_supported_depth_v1"
        for name in names
    )


def test_a_disagreeing_second_view_still_fails_only_its_entry(
    tmp_path: Path,
) -> None:
    """Cross-view verification keeps its 25 mm rule inside a batch."""

    runtime = _batched(_runtime(tmp_path)[0])
    runtime.invoke("mark_point", {
        "view": "agentview", "point_ids": ["rim"], "points_uv": [[256, 256]],
    })
    result = runtime.invoke("mark_point", {
        "view": "wrist",
        "point_ids": ["rim", "fresh"],
        "points_uv": [[2, 1], [250, 250]],
    })
    rim, fresh = result["marks"]
    assert rim["point_id"] == "rim"
    assert rim["status"] in {"verified", "failed"}
    if rim["status"] == "failed":
        assert rim["error_code"] == "inconsistent_mark"
        # The entry keeps the measured distance and the bound (census C3).
        assert "m, more than the 0.025 m maximum" in rim["error_message"]
    else:
        assert rim["verification_distance_m"] is not None
        assert rim["verification_distance_m"] <= 0.025
    assert fresh["status"] == "solved", (
        "a disagreement on one name must not cost the other its measurement"
    )


def test_re_marking_a_solved_pixel_reports_the_measurement_it_reuses(
    tmp_path: Path,
) -> None:
    """"Solved" and "unmeasured" together would be a contradiction.

    Marking the exact pixel of a point already solved in this observation
    short-circuits before the depth estimator runs, so the entry is solved and
    the estimator reported nothing. The honest answer is the quality of the
    measurement being reused.
    """

    runtime = _batched(_runtime(tmp_path)[0])
    first = runtime.invoke("mark_point", {
        "view": "agentview", "point_ids": ["rim"], "points_uv": [[256, 256]],
    })
    again = runtime.invoke("mark_point", {
        "view": "agentview", "point_ids": ["rim"], "points_uv": [[256, 256]],
    })
    reused = again["marks"][0]
    assert reused["status"] == "solved"
    assert reused["xyz_m"] == first["marks"][0]["xyz_m"]
    assert reused["quality_status"] == "measured"
    assert reused["quality_method"] == "local_supported_depth_v1"
    assert reused["quality_depth_m"] == first["marks"][0]["quality_depth_m"]


def test_a_panel_drawing_one_surviving_mark_names_which_one(
    tmp_path: Path,
) -> None:
    """Two of three pixels fail, so the overlay is unambiguous and says so."""

    runtime = _batched(_runtime(tmp_path)[0])
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": ["off_a", "good", "off_b"],
        "points_uv": [[99999, 1], [250, 250], [1, 99999]],
    })
    assert result["status"] == "partial" and result["solved_count"] == 1
    assert result["image_mapping"][0]["source_index"] == 1


def test_the_delivered_contract_survives_the_perception_feedback_layer(
    tmp_path: Path,
) -> None:
    """``enrich`` runs after the handler and must not break the result schema."""

    runtime = _batched(_runtime(tmp_path)[0])
    runtime.surface = dataclasses.replace(
        runtime.surface,
        capabilities=(
            *runtime.surface.capabilities, "public_rgbd_perception_feedback_v2",
        ),
    )
    result = runtime.invoke("mark_point", {
        "view": "agentview",
        "point_ids": ["a", "b"], "points_uv": [[250, 250], [251, 251]],
    })
    assert result["measurement_contract"]
    assert len(result["image_mapping"]) == 1, (
        "the batch supplies its own mapping; enrich must not add a "
        "position-indexed one over the top"
    )
    assert result["image_mapping"][0]["role"] == "mark_batch_panel"


def test_the_declared_error_codes_are_exactly_the_ones_an_entry_can_carry(
) -> None:
    """A code outside the enum would fail the result the model never sees.

    The runtime decides per-entry versus per-call by this same set, so the two
    cannot be tightened independently.
    """

    from ora.agent.tools.mark_point_batch import _ENTRY_ERROR_CODES

    declared = {
        code
        for code in mark_point.MARK_POINT_BATCH_RESULT_SCHEMA["properties"][
            "marks"
        ]["items"]["properties"]["error_code"]["enum"]
        if code is not None
    }
    # invalid_camera_calibration left the entry codes (EXP11B census C3): a
    # published calibration is infrastructure and fails the whole call. The
    # frozen result schema still declares it; only a successor schema drops it.
    assert declared == set(_ENTRY_ERROR_CODES) | {"invalid_camera_calibration"}
    assert "invalid_camera_calibration" not in _ENTRY_ERROR_CODES


def test_a_failure_that_is_not_about_one_pixel_fails_the_whole_call(
    tmp_path: Path,
) -> None:
    """The observation disappearing is not one entry's problem."""

    runtime = _batched(_runtime(tmp_path)[0])
    (tmp_path / "control/current-observation.json").unlink()
    with pytest.raises(ToolError) as failed:
        runtime.invoke("mark_point", {
            "view": "agentview",
            "point_ids": ["a", "b"], "points_uv": [[250, 250], [251, 251]],
        })
    assert failed.value.code == "environment_unavailable"
