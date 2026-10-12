"""Per-point mask ledgers, bounded delivery, source-bound artifacts and service wire."""

import base64
import copy
import hashlib
import io
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from ora.agent.tools import tool_names
from PIL import Image

from ora.agent.composition import resolve_tool_surface, validate_schema_instance
from ora.agent.composition.tool_surface import PROFILE_CATALOG
from ora.agent.model_services.resident import SegmentService, _encode_npy
from ora.agent.tools import grasp, segmentation_batch as batch
from ora.agent.tools.robot import ToolError


#: A released v1 vision.segment result schema (the frozen explicit-motion LIBERO
#: component) the reviewed v2 delta below is applied to.
RELEASED_V1_RESULTS = (Path(__file__).resolve().parents[1]
                       / "ora/policy/tool_surfaces/explicit_motion_v3/libero-results.json")
#: The kept default surfaces; each binds the v2 batch contract.
DEFAULT_PROFILES = ("robotwin_toolchain_b2_closed_noservo_v2", "libero_placement_direct_overhead_v1",
                    "robodojo_placement_closed_skills_v2")

# Delivery ceilings read from the sealed run configuration
# (`native_policy.context_management` in `tests/data/default-expectations.json`).
EAGER_LARGE_RESULT_BYTES = 50000
TRIGGER_INLINE_IMAGES = 24
# Measured over 252 sealed vision.segment results under
# outputs/*/runs/*/openeta-tool-artifacts/segmentation-batches/*/manifest.json:
# a 1-point 1-candidate result is 3,387 B and a 3-candidate result is 7,275 B,
# so the fixed header is ~1,313 B and each delivered candidate costs ~1,944 B.
# The per-point ledger adds shared_with_source_indexes (~35 B) per candidate.
SEALED_FIXED_BYTES = 1313
SEALED_BYTES_PER_CANDIDATE = 1944 + 35
SEALED_BYTES_PER_QUERY = 220


def result_schema_v2(released=None):
    """Exact delta the ToolSurface result component must apply for the per-point ledger.

    Built from a released v1 schema so the change is reviewable as a patch
    rather than as a retyped 600-line schema. `released` defaults to a released
    v1 component; a caller may pass its own v1 schema, whose backend camera enum
    and image binding the patch must carry through rather than flatten.
    """
    schema = copy.deepcopy(released) if released is not None else json.loads(
        RELEASED_V1_RESULTS.read_text())["vision.segment"]
    properties = schema["properties"]
    properties["schema"]["const"] = batch.SCHEMA
    properties["status"]["enum"] = ["candidates", "partial", "no_candidate"]
    properties["query_count"] = {"type": "integer", "minimum": 1, "maximum": batch.MAX_POINTS}

    properties["candidates"]["maxItems"] = batch.MAX_RETURNED_CANDIDATES
    candidate = properties["candidates"]["items"]
    candidate["properties"].pop("rank")
    candidate["required"].remove("rank")
    candidate["properties"]["position"] = {
        "type": "integer", "minimum": 0, "maximum": batch.MAX_RETURNED_CANDIDATES - 1}
    candidate["properties"]["shared_with_source_indexes"] = {
        "type": "array", "maxItems": batch.MAX_POINTS,
        "items": {"type": "integer", "minimum": 0, "maximum": batch.MAX_POINTS - 1}}
    candidate["required"] += ["position", "shared_with_source_indexes"]

    properties.pop("query_results")
    properties["queries"] = {"type": "array", "maxItems": batch.MAX_POINTS, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["query_index", "point_uv", "status", "error_code", "raw_candidate_count",
                     "candidate_count", "candidate_ids", "omitted_over_point_limit",
                     "omitted_over_call_cap", "preview_image_index", "previews_omitted"],
        "properties": {
            "query_index": {"type": "integer", "minimum": 0, "maximum": batch.MAX_POINTS - 1},
            "point_uv": {"type": "array", "minItems": 2, "maxItems": 2,
                         "items": {"type": "integer", "minimum": 0}},
            "status": {"type": "string", "enum": ["candidates", "no_candidate", "failed"]},
            "error_code": {"type": ["string", "null"]},
            "raw_candidate_count": {"type": "integer", "minimum": 0},
            "candidate_count": {"type": "integer", "minimum": 0, "maximum": batch.MAX_CANDIDATES},
            "candidate_ids": {"type": "array", "maxItems": batch.MAX_CANDIDATES,
                              "items": {"type": "string"}},
            "omitted_over_point_limit": {"type": "integer", "minimum": 0},
            "omitted_over_call_cap": {"type": "integer", "minimum": 0,
                                      "maximum": batch.MAX_CANDIDATES},
            "preview_image_index": {"type": ["integer", "null"], "minimum": 1,
                                    "maximum": batch.MAX_PREVIEW_IMAGES},
            "previews_omitted": {"type": "integer", "minimum": 0, "maximum": 1}}}}

    properties["selection"] = {"type": "object", "additionalProperties": False,
        "required": ["method", "mask_iou_threshold", "max_candidates_per_point",
                     "max_returned_candidates", "raw_candidate_count", "usable_candidate_count",
                     "distinct_candidate_count", "duplicate_count", "returned_count",
                     "omitted_over_point_limit_total", "omitted_over_call_cap_total",
                     "rejected_candidates", "semantic_selection"],
        "properties": {
            "method": {"type": "string", "const": batch.SELECTION_METHOD},
            "mask_iou_threshold": {"type": "number"},
            "max_candidates_per_point": {"type": "integer", "minimum": 1,
                                         "maximum": batch.MAX_CANDIDATES},
            "max_returned_candidates": {"type": "integer", "minimum": 1,
                                        "maximum": batch.MAX_RETURNED_CANDIDATES},
            "raw_candidate_count": {"type": "integer", "minimum": 0},
            "usable_candidate_count": {"type": "integer", "minimum": 0},
            "distinct_candidate_count": {"type": "integer", "minimum": 0},
            "duplicate_count": {"type": "integer", "minimum": 0},
            "returned_count": {"type": "integer", "minimum": 0},
            "omitted_over_point_limit_total": {"type": "integer", "minimum": 0},
            "omitted_over_call_cap_total": {"type": "integer", "minimum": 0},
            "rejected_candidates": {"type": "array", "maxItems": 24, "items": {
                "type": "object", "additionalProperties": False,
                "required": ["query_index", "source_index", "reason"],
                "properties": {"query_index": {"type": "integer", "minimum": 0},
                               "source_index": {"type": "integer", "minimum": 0},
                               "reason": {"type": "string"}}}},
            "semantic_selection": {"type": "string", "const": "agent"}}}

    properties["candidate_previews"]["maxItems"] = batch.MAX_PREVIEW_IMAGES
    preview = properties["candidate_previews"]["items"]
    preview["properties"].pop("rank")
    preview["required"].remove("rank")
    preview["properties"]["position"] = {
        "type": "integer", "minimum": 0, "maximum": batch.MAX_RETURNED_CANDIDATES - 1}
    preview["required"].append("position")
    properties["image_order"]["maxItems"] = batch.MAX_PREVIEW_IMAGES + 1
    # A profile that already binds an image mapping keeps its own camera
    # vocabulary; only the count changes, because a call now returns at most one
    # overview plus MAX_PREVIEW_IMAGES crops.
    mapping = properties.get("image_mapping")
    if mapping is None:
        properties["image_mapping"] = {"type": "array", "maxItems": batch.MAX_PREVIEW_IMAGES + 1,
            "items": {"type": "object", "additionalProperties": False,
                "required": ["image_index", "role", "camera", "candidate_id", "target_ref",
                             "observation_step"],
                "properties": {"image_index": {"type": "integer", "minimum": 0},
                               "role": {"type": "string"}, "camera": {"type": "string"},
                               "candidate_id": {"type": "string"}, "target_ref": {"type": "string"},
                               "observation_step": {"type": ["integer", "null"], "minimum": 0}}}}
    else:
        mapping["maxItems"] = batch.MAX_PREVIEW_IMAGES + 1

    schema["required"] = [name if name != "query_results" else "queries"
                          for name in schema["required"]] + ["query_count"]
    return schema


def batch_surface():
    """The kept RoboTwin surface of the public-run fixture; it ships the v2 batch contract."""
    from test_robotwin_placement_adapter import PROFILE
    surface = resolve_tool_surface(PROFILE)
    assert surface.supports(batch.CAPABILITY)
    return surface


def rectangle(x0, y0, x1, y1, size=96):
    mask = np.zeros((size, size), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def raw_candidate(q, c, point, mask, score):
    return {"query_index": q, "source_index": c, "point_uv": point,
            "mask": mask, "backend_score": score}


def test_a_low_scoring_point_keeps_its_only_mask_against_higher_scoring_neighbours():
    """Regression for the pooled-budget defect measured on the sealed cohort.

    Point 0 asks about a small object whose mask scores below every mask of
    points 1 and 2. One pooled budget of three spent every slot on the two
    louder points and reported point 0 as a normally completed query holding
    nothing, which reads as "no graspable surface at that pixel". Rank 0 of
    every point is now admitted before any point's rank 1.
    """
    points = [[20, 20], [50, 50], [80, 80]]
    raw = [raw_candidate(0, 0, points[0], rectangle(15, 15, 25, 25), .40),
           raw_candidate(1, 0, points[1], rectangle(45, 45, 55, 55), .90),
           raw_candidate(1, 1, points[1], rectangle(44, 44, 58, 58), .85),
           raw_candidate(2, 0, points[2], rectangle(75, 75, 85, 85), .80)]
    selected, report = batch.select_masks(raw, points_uv=points, shape=(96, 96), max_candidates=3)
    owners = {source["query_index"] for item in selected for source in item["sources"]}
    assert owners == {0, 1, 2}, "every point that produced a usable mask keeps one"
    ledger = report["queries"]
    assert [entry["candidate_ids"] for entry in ledger] == [["T0"], ["T1", "T3"], ["T2"]]
    assert all(entry["omitted_over_call_cap"] == 0 for entry in ledger)
    assert report["returned_count"] == 4 and report["omitted_over_call_cap_total"] == 0


def test_one_point_never_spends_another_points_budget_when_the_call_cap_binds():
    """Eight points each holding three masks: the reserve survives, the surplus competes."""
    points = [[8 + 11 * q, 8 + 11 * q] for q in range(8)]
    raw = [raw_candidate(q, c, point, rectangle(point[0] - 4 - c, point[1] - 4 - c,
                                                point[0] + 4 + c, point[1] + 4 + c), .9 - q * .05 - c * .01)
           for q, point in enumerate(points) for c in range(3)]
    selected, report = batch.select_masks(raw, points_uv=points, shape=(96, 96), max_candidates=3)
    ledger = report["queries"]
    assert len(selected) == batch.MAX_RETURNED_CANDIDATES
    assert all(entry["candidate_ids"] for entry in ledger), "the reserve is never starved"
    assert all(entry["omitted_over_point_limit"] == 0 for entry in ledger)
    assert report["omitted_over_call_cap_total"] == 24 - batch.MAX_RETURNED_CANDIDATES
    assert sum(entry["omitted_over_call_cap"] for entry in ledger) == 12
    # Every point can read its own loss and tell it apart from an empty pixel.
    assert {len(entry["candidate_ids"]) for entry in ledger} == {1, 2}
    assert report["preview_count"] == batch.MAX_PREVIEW_IMAGES
    assert sum(entry["previews_omitted"] for entry in ledger) == 8 - batch.MAX_PREVIEW_IMAGES


def test_the_delivery_cap_always_reserves_one_mask_for_every_queried_point():
    assert batch.MAX_RETURNED_CANDIDATES >= batch.MAX_POINTS
    points = [[10 * q + 5, 20] for q in range(8)]
    raw = [raw_candidate(q, 0, p, rectangle(p[0] - 3, 17, p[0] + 3, 23), .5) for q, p in enumerate(points)]
    with pytest.raises(ValueError, match="reserve one mask"):
        batch.select_masks(raw, points_uv=points, shape=(96, 96), max_candidates=3, max_returned=7)


def test_worst_case_delivered_result_stays_under_the_eager_large_result_threshold():
    """Measured sealed-cohort byte costs, not an estimate of the fixture's size."""
    worst = (SEALED_FIXED_BYTES
             + batch.MAX_RETURNED_CANDIDATES * SEALED_BYTES_PER_CANDIDATE
             + batch.MAX_POINTS * SEALED_BYTES_PER_QUERY)
    assert worst < .7 * EAGER_LARGE_RESULT_BYTES, (
        "a result above eager_large_result_bytes is replaced by a 2,048-byte stub")
    assert batch.MAX_PREVIEW_IMAGES + 1 <= TRIGGER_INLINE_IMAGES // 4


def test_shared_masks_cost_one_artifact_and_are_reported_under_every_point():
    points = [[20, 20], [22, 22], [80, 80]]
    shared = rectangle(12, 12, 32, 32)
    raw = [raw_candidate(0, 0, points[0], shared, .8),
           raw_candidate(1, 0, points[1], shared, .7),
           raw_candidate(2, 0, points[2], rectangle(75, 75, 85, 85), .6)]
    selected, report = batch.select_masks(raw, points_uv=points, shape=(96, 96), max_candidates=3)
    assert len(selected) == 2 and report["duplicate_count"] == 1
    assert selected[0]["sources"] == [{"query_index": 0, "source_index": 0},
                                      {"query_index": 1, "source_index": 0}]
    ledger = report["queries"]
    assert ledger[0]["candidate_ids"] == ledger[1]["candidate_ids"] == ["T0"]
    assert ledger[0]["preview_image_index"] == ledger[1]["preview_image_index"] == 1
    assert report["preview_count"] == 2


def test_component_filter_and_bad_positive_mask_are_disclosed():
    point = [20, 20]
    remote = rectangle(65, 65, 70, 70)
    mask = rectangle(15, 15, 25, 25) | remote
    selected, report = batch.select_masks([raw_candidate(0, 0, point, mask, .8),
        raw_candidate(0, 1, point, remote, .9)], points_uv=[point], shape=(96, 96), max_candidates=3)
    assert len(selected) == 1
    assert selected[0]["normalization"]["raw_component_count"] == 2
    assert selected[0]["normalization"]["retained_fraction"] == .8
    assert not selected[0]["mask"][67, 67]
    assert report["rejected_candidates"] == [{"query_index": 0, "source_index": 1,
                                             "reason": "mask_excludes_positive_point"}]


class BatchBackend:
    backend_id, backend_version = "batch-fixture", "1"

    def __init__(self, fail=(), empty=(), excluded=()):
        self.calls = 0
        self.fail, self.empty, self.excluded = set(fail), set(empty), set(excluded)

    def configuration(self):
        return {"fixture": "independent-point-proposals"}

    def segment_points(self, *, rgb_u8, depth_m, points_uv):
        self.calls += 1
        raw, queries = [], []
        for q, point in enumerate(points_uv):
            failed, produced = q in self.fail, 0
            if not failed and q not in self.empty:
                x, y = point
                mask = (rectangle(2, 2, 6, 6) if q in self.excluded
                        else rectangle(x - 5, y - 5, x + 6, y + 6))
                raw.extend([raw_candidate(q, i, point, mask, .8 - i * .1) for i in range(2)])
                produced = 2
            queries.append({"query_index": q, "point_uv": point,
                "raw_candidate_count": produced,
                "error_code": "point_inference_failed" if failed else None})
        return raw, {"query_results": queries, "image_encodings": 1}


def public_run(tmp_path, **kwargs):
    from test_robotwin_placement_adapter import RoboTwinPublicRun
    run = RoboTwinPublicRun(tmp_path, bind_submitted=True)
    run.runtime.surface = batch_surface()
    run.runtime.segmentation_backend = BatchBackend(**kwargs)
    return run


def test_public_batch_preserves_partial_success_and_reuses_bound_artifacts(tmp_path):
    run = public_run(tmp_path, fail=[1])
    arguments = {"camera": "head_camera", "points_uv": [[40, 40], [50, 50], [60, 60]]}
    result = run.runtime.invoke("vision_segment", arguments)
    assert result["status"] == "partial" and len(result["candidates"]) == 2
    assert len(result["images_base64"]) == 3
    assert result["queries"][1]["status"] == "failed"
    assert result["queries"][1]["error_code"] == "point_inference_failed"
    assert result["selection"]["duplicate_count"] == 2
    assert result["image_order"] == ["overview", "T0", "T1"]
    assert result["semantic_identity_verified"] is False
    assert not (run.runtime.control_root / "submitted-action.json").exists()
    for candidate in result["candidates"]:
        geometry, state = run.runtime._target_geometry_and_state(target_ref=candidate["target_ref"])
        assert geometry["observation_id"] == result["observation_id"] and state["step"] == 0
    repeated = run.runtime.invoke("vision_segment", arguments)
    assert repeated["cache_reused"] is True and run.runtime.segmentation_backend.calls == 1
    assert repeated["candidates"] == result["candidates"]
    candidate = result["candidates"][0]
    _, mask_path, _ = grasp.load_segmented_target(artifact_root=run.runtime.artifact_root, target_ref=candidate["target_ref"])
    mask_path.chmod(0o644)
    mask_path.write_bytes(b"corrupted")
    with pytest.raises(ToolError, match="hash"):
        run.runtime.invoke("vision_segment", arguments)


def test_an_empty_point_names_why_it_is_empty_and_never_blames_the_budget(tmp_path):
    """The two facts the pooled ledger could not tell apart, now separately named."""
    run = public_run(tmp_path, empty=[1], excluded=[2])
    result = run.runtime.invoke("vision_segment",
        {"camera": "head_camera", "points_uv": [[40, 40], [50, 50], [60, 60]]})
    absent, rejected = result["queries"][1], result["queries"][2]
    assert absent["status"] == "no_candidate" and absent["error_code"] == "backend_returned_no_mask"
    assert rejected["status"] == "no_candidate"
    assert rejected["error_code"] == "mask_excludes_positive_point"
    for entry in result["queries"]:
        assert entry["omitted_over_call_cap"] == 0, "an empty point is never a budget loss here"
    assert result["status"] == "partial"
    assert result["selection"]["omitted_over_call_cap_total"] == 0


def test_bounded_delivery_holds_on_a_full_eight_point_call(tmp_path):
    run = public_run(tmp_path)
    points = [[12 + 9 * q, 12 + 9 * q] for q in range(8)]
    result = run.runtime.invoke("vision_segment", {"camera": "head_camera", "points_uv": points})
    assert len(result["images_base64"]) <= batch.MAX_PREVIEW_IMAGES + 1
    assert len(result["candidates"]) <= batch.MAX_RETURNED_CANDIDATES
    assert all(entry["candidate_ids"] for entry in result["queries"])
    visible = {k: v for k, v in result.items() if k != "images_base64"}
    assert len(json.dumps(visible).encode()) < EAGER_LARGE_RESULT_BYTES
    previewed = [entry["preview_image_index"] for entry in result["queries"]
                 if entry["preview_image_index"] is not None]
    assert previewed == sorted(set(previewed))
    assert max(previewed) < len(result["images_base64"])


def test_empty_pool_is_inspectable_and_history_never_becomes_current_geometry(tmp_path):
    run = public_run(tmp_path, fail=[0, 1])
    result = run.runtime.invoke("vision_segment", {"camera": "head_camera", "points_uv": [[30, 30], [60, 60]]})
    assert result["status"] == "no_candidate" and result["candidates"] == []
    assert len(result["images_base64"]) == 1
    run.runtime.segmentation_backend = BatchBackend()
    run.runtime.invoke("set_gripper", {"targets": {"right": "open"}})
    run.publish({"right": "open"})
    old = run.runtime.invoke("vision_segment", {"camera": "left_camera", "step": 0, "points_uv": [[48, 48]]})
    ref = old["candidates"][0]["target_ref"]
    assert old["observation_id"] != run.runtime._current_grip_state()[0]["observation_id"]
    with pytest.raises(ToolError, match="stale"):
        run.runtime.invoke("grasp_plan", {"arm": "right", "target_ref": ref})
    run.runtime.reset_round(1)
    run.publish(reset=True)
    with pytest.raises(ToolError, match="current round"):
        run.runtime.invoke("vision_segment", {"camera": "left_camera", "step": 0, "points_uv": [[48, 48]]})


@pytest.mark.parametrize("points,maximum", [([], 3), ([[1, 1]] * 9, 3), ([[True, 2]], 3),
    ([[1.0, 2]], 3), ([[96, 2]], 3), ([[1, 2]], 4), ([[1, 2]], True)])
def test_invalid_batch_parameters_fail_before_any_backend_query(tmp_path, points, maximum):
    run = public_run(tmp_path)
    with pytest.raises(ToolError):
        run.runtime.invoke("vision_segment", {"camera": "head_camera", "points_uv": points,
                                               "max_candidates": maximum})
    assert run.runtime.segmentation_backend.calls == 0


def test_service_encodes_once_and_never_accumulates_point_prompts():
    calls, encodings = [], []
    torch = SimpleNamespace(inference_mode=nullcontext, autocast=lambda *_a, **_k: nullcontext(),
        bfloat16="bf16", cuda=SimpleNamespace(synchronize=lambda: None))
    marker = object()

    def set_image(image):
        encodings.append(image.size)
        return marker

    def predict(state, **kwargs):
        assert state is marker
        points = kwargs["point_coords"]
        assert points.shape == (1, 2) and kwargs["point_labels"].tolist() == [1]
        assert "mask_input" not in kwargs
        calls.append(points.copy())
        if int(points[0, 0]) == 10:
            raise RuntimeError("one point failed")
        masks = np.ones((3, 32, 32), dtype=np.uint8)
        return masks, np.array([.6, .9, .8]), None

    service = object.__new__(SegmentService)
    service.torch, service.processor, service.model = torch, SimpleNamespace(set_image=set_image), SimpleNamespace(predict_inst=predict)
    service.model_identity, service.requests = {"test": "fake"}, 0
    image = io.BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    result = service.infer_batch({"schema": "ora.model-service.segment-points-request.v1",
        "image_png_base64": base64.b64encode(image.getvalue()).decode(), "points_uv": [[5, 5], [10, 10], [15, 15]]})
    assert len(encodings) == 1 and len(calls) == 3 and result["image_encodings"] == 1
    assert [len(q["candidates"]) for q in result["queries"]] == [3, 0, 3]
    assert result["queries"][1]["error"]["code"] == "point_inference_failed"
    assert service.requests == 1


def test_sam_batch_client_uses_one_request_and_validates_query_identity():
    requests, drift = [], []
    identity = {"schema": "ora.model-service-response-identity.v1",
                "instance_id": "fixture", "pid": 1, "started_unix_s": 1., "adapter_sha256": "a" * 64}

    def response(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/segment-points"
        queries = [{"query_index": q + int(bool(drift)), "point_uv": p, "error": None,
                    "candidates": [{"source_index": 0, "score": .9,
                        "mask_npy_base64": _encode_npy(np.ones((32, 32), dtype=np.uint8))}]}
                   for q, p in enumerate(body["points_uv"])]
        return httpx.Response(200, json={"schema": "ora.model-service.segment-points.v1",
            "request_id": "query-1", "model": grasp.SAM3_MODEL_IDENTITY, "service_identity": identity,
            "queries": queries, "image_encodings": 1, "metrics": {"queue_ms": 0, "inference_ms": 1, "total_ms": 1}})

    segmenter = grasp.Sam3ApiSegmenter()
    segmenter._client.close()
    object.__setattr__(segmenter, "_client", httpx.Client(transport=httpx.MockTransport(response)))
    try:
        args = dict(rgb_u8=np.zeros((32, 32, 3), np.uint8), depth_m=np.ones((32, 32)), points_uv=[[5, 5], [10, 10]])
        raw, metadata = segmenter.segment_points(**args)
        assert len(requests) == 1 and len(raw) == 2 and metadata["image_encodings"] == 1
        drift.append(True)
        with pytest.raises(ValueError, match="identity"):
            segmenter.segment_points(**args)
    finally:
        segmenter.close()


# --- Two published contracts, one runtime path.
class GoldenBackend:
    """The fixture backend the released-v1 golden was recorded against."""

    backend_id, backend_version = "released-v1-golden", "1"

    def configuration(self):
        return {"fixture": "released-v1-golden"}

    def segment_points(self, *, rgb_u8, depth_m, points_uv):
        raw, queries = [], []
        for q, (x, y) in enumerate(points_uv):
            produced = 0
            if q != 2:
                for i in range(2):
                    mask = np.zeros(depth_m.shape, dtype=bool)
                    # Wider than tall: a square mask gives two equal in-plane
                    # eigenvalues, so the principal axis was whichever one the
                    # host's LAPACK returned and the golden held on one host only.
                    mask[y - 6 - i:y + 7 + i, x - 6 - i:x + 10 + i] = True
                    raw.append({"query_index": q, "source_index": i, "point_uv": [x, y],
                                "mask": mask, "backend_score": .9 - q * .2 - i * .05})
                produced = 2
            queries.append({"query_index": q, "point_uv": [x, y],
                            "raw_candidate_count": produced,
                            "error_code": None if produced else "point_inference_failed"})
        return raw, {"query_results": queries, "image_encodings": 1}


def golden_source(root: Path) -> dict:
    """Rebuild the exact RGB-D source the golden was recorded from.

    Every identity in the result is content-addressed -- source file hashes,
    point list, backend configuration -- so the same bytes in the same shape
    reproduce the same `query_ref` and `target_ref`s from any directory.
    """
    camera_root = root / "policy-observations/episode-000_observation-0003/agentview"
    camera_root.mkdir(parents=True, exist_ok=True)
    rgb = np.full((64, 64, 3), [48, 52, 60], dtype=np.uint8)
    rgb[24:40, 24:40] = [210, 48, 32]
    rgb[44:58, 44:58] = [40, 160, 90]
    Image.fromarray(rgb).save(camera_root / "rgb.png", format="PNG")
    depth = np.full((64, 64), 1.2, dtype=np.float32)
    depth[24:40, 24:40] = 0.8
    depth[44:58, 44:58] = 0.95
    np.save(camera_root / "depth-m.npy", depth, allow_pickle=False)
    (camera_root / "calibration.json").write_text(json.dumps({
        "intrinsic_cv": [[400.0, 0.0, 32.0], [0.0, 400.0, 32.0], [0.0, 0.0, 1.0]],
        "extrinsic_cv": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "cam2world_gl": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        "depth_unit": "metres",
    }, sort_keys=True), encoding="utf-8")
    return {"artifact_root": root, "observation_id": "episode-000:observation-0003",
            "camera": "agentview", "rgb_path": camera_root / "rgb.png",
            "depth_path": camera_root / "depth-m.npy",
            "calibration_path": camera_root / "calibration.json",
            "points_uv": [[30, 30], [50, 50], [12, 12]], "max_candidates": 3}


def test_the_same_fixture_under_v2_keeps_every_points_leading_mask(tmp_path):
    """Every point keeps its own leading mask under the per-point ledger."""
    manifest, _paths, cached = batch.materialize_batch(
        backend=GoldenBackend(), **golden_source(tmp_path))
    result = batch.public_result(manifest, cache_reused=cached)
    assert result["schema"] == "ora.point-segmentation-batch.v2"
    assert [entry["candidate_ids"] for entry in result["queries"]] == [
        ["T0", "T2"], ["T1", "T3"], []]
    assert result["queries"][2]["error_code"] == "point_inference_failed"


def test_the_depth_profile_extends_only_the_v2_batch_identity_and_cards(tmp_path):
    """grounding_candidate_depth_profile_v1 on the per-point batch.

    The query identity gains the renderer, so a profiled card never reuses a
    released one; target refs, the ledger and every published preview key are
    unchanged, and the crops are the released pixels above the new strip.
    """
    from ora.agent.tools import grounding_preview

    options = {"renderer": grounding_preview.PROFILE_RENDERER, "support_line": True}
    released_manifest, released_paths, _ = batch.materialize_batch(
        backend=GoldenBackend(), **golden_source(tmp_path / "released"))
    manifest, paths, cached = batch.materialize_batch(
        backend=GoldenBackend(), depth_profile=options, **golden_source(tmp_path / "profiled"))
    assert "depth_profile" not in released_manifest["identity"]
    assert manifest["identity"] == {**released_manifest["identity"], "depth_profile": options}
    assert manifest["query_ref"] != released_manifest["query_ref"]
    assert [c["target_ref"] for c in manifest["candidates"]] == [
        c["target_ref"] for c in released_manifest["candidates"]]
    assert manifest["queries"] == released_manifest["queries"]
    released_result = batch.public_result(released_manifest, cache_reused=False)
    result = batch.public_result(manifest, cache_reused=cached)
    assert [set(item) for item in result["candidate_previews"]] == [
        set(item) for item in released_result["candidate_previews"]]
    assert {tuple(item["render_dimensions"]) for item in result["candidate_previews"]} == {(1024, 800)}
    validate_schema_instance(grasp.project_target_frame(copy.deepcopy(result), declared=False),
                             result_schema_v2())
    for released_path, path, record in zip(released_paths[1:], paths[1:], manifest["candidate_previews"]):
        released_card = np.asarray(Image.open(released_path).convert("RGB"))
        card = np.asarray(Image.open(path).convert("RGB"))
        assert card.shape == (800, 1024, 3) and np.array_equal(card[:576], released_card)
        assert record["depth_profile"]["status"] == "rendered"
    # The released overview is untouched.
    assert paths[0].read_bytes() == released_paths[0].read_bytes()


@pytest.mark.parametrize("capability,expected", [
    (batch.CAPABILITY, "segmentation_batch"),
    ("some_other_capability", None),
])
def test_the_declared_capability_alone_selects_the_contract(capability, expected):
    surface = SimpleNamespace(supports=lambda name: name == capability)
    contract = batch.select_contract(surface)
    assert (None if contract is None else contract.__name__.rsplit(".", 1)[-1]) == expected


@pytest.mark.parametrize("profile", DEFAULT_PROFILES)
def test_max_candidates_help_states_the_budget_it_now_is(profile):
    """Its meaning changed from a pooled total to a per-point budget."""
    surface = resolve_tool_surface(profile)
    definition = tool_names.definition_for(surface.definitions, "vision_segment")
    description = definition["inputSchema"]["properties"]["max_candidates"]["description"]
    assert "per-point" in description.lower() and "not a total" in description.lower()
    # The released text said "Total maximum distinct masks after pooling ... not a
    # per-point limit", which is now the opposite of what the runtime does.
    assert "pool" not in description.lower()
    assert "per-point" in definition["description"].lower()
    assert "TOTAL" not in definition["description"]
    positions = [definition["description"].index(marker) for marker in (
        "Use when:", "Effect:", "Returns:", "Does not:", "On error:")]
    assert positions == sorted(positions)
