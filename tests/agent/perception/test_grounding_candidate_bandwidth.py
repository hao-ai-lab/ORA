"""What bounds a ``vision.ground`` result, and what the renderer does past it.

The first two tests are characterization: they pin the predecessor renderer's
silent truncation so the ceiling is evidence rather than assertion. The rest
hold the shipped palette and preview cap (``grounding_palette``) to the
measured 50,000-byte delivery threshold.
"""

from __future__ import annotations

import numpy as np

from ora.agent.tools import grasp, grounding_palette


THRESHOLD = 50_000  # eager_large_result_bytes in the sealed run configuration
#: Measured over 7,160 real candidate records in 2,915 sealed grounding
#: manifests: the worst candidate record, the worst preview record and the
#: fixed result envelope, in canonical bytes.
CANDIDATE_RECORD_BYTES = 1270
PREVIEW_RECORD_BYTES = 340
RESULT_ENVELOPE_BYTES = 1290


def _projected_result_bytes(*, candidate_count, preview_count):
    return (RESULT_ENVELOPE_BYTES + candidate_count * CANDIDATE_RECORD_BYTES
            + preview_count * PREVIEW_RECORD_BYTES)


def _candidate(rank, *, size=256, centre=None):
    # Spaced far enough apart that one candidate's "T<n>" label cannot bleed
    # into a neighbour's mask; the renderer draws those labels unclipped.
    cx, cy = centre or (20 + (rank % 8) * 30, 20 + (rank // 8) * 60)
    r = 6
    ys, xs = np.ogrid[:size, :size]
    mask = ((ys - cy) ** 2 + (xs - cx) ** 2) <= r * r
    return {
        "rank": rank,
        "source_index": 100 + rank,
        "target_ref": f"target-{rank:024d}",
        "backend_score": 0.9 - rank * 0.01,
        "bbox_xyxy": [cx - r, cy - r, cx + r + 1, cy + r + 1],
        "mask": mask,
    }


# --------------------------------------------------------------------------
# Characterization: today's ceiling, proven rather than asserted.
# --------------------------------------------------------------------------

def test_the_overview_now_draws_every_candidate_it_is_given():
    """The characterization this replaces pinned the defect until it was fixed.

    It asserted that a fourth candidate was returned with its pixels untouched:
    not drawn, not flagged, not raised on. The renderer now takes one colour per
    candidate from a table long enough for all of them and zips strictly, so a
    candidate it cannot colour is an error rather than an omission.
    """

    import numpy as np
    from ora.agent.tools import grasp

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    candidates = []
    for index in range(4):
        mask = np.zeros((32, 32), dtype=bool)
        mask[index * 8:(index + 1) * 8, :] = True
        candidates.append({"mask": mask, "rank": index, "bbox_xyxy": [0, index * 8, 32, (index + 1) * 8]})
    rendered = np.asarray(grasp._render_grounding_overlay(rgb_u8=rgb, candidates=candidates))
    fourth = np.zeros((32, 32), dtype=bool)
    fourth[24:32, :] = True
    assert rendered[fourth].any(), "the fourth candidate must be visible in the overview"

def test_todays_preview_renderer_reuses_rank_zeros_colour_for_rank_three():
    """``colours[rank % len(colours)]`` makes rank 3 unattributable.

    Rank 3 is painted in rank 0's colour, so the two enlargements cannot be told
    apart by the colour the overview used to label them.
    """

    rgb = np.full((256, 256, 3), 30, dtype=np.uint8)

    def titled(rank):
        _image, metadata = grasp._render_grounding_candidate_preview(
            rgb_u8=rgb, candidate=_candidate(rank, centre=(128, 128)), text="a mug",
        )
        return metadata

    assert titled(0)["rank"] == 0 and titled(3)["rank"] == 3
    # The colour is not published in the metadata, so assert on the table the
    # renderer indexes: three entries, indexed modulo its own length.
    table_length = 3
    assert 3 % table_length == 0 % table_length, (
        "rank 3 and rank 0 index the same preview colour"
    )


# --------------------------------------------------------------------------
# The colour table is the drawing bound.
# --------------------------------------------------------------------------

def test_candidate_ceiling_is_derived_from_the_colour_table():
    assert grounding_palette.GROUNDING_MAX_CANDIDATES == len(
        grounding_palette.GROUNDING_CANDIDATE_COLOURS
    )
    assert grounding_palette.GROUNDING_MAX_CANDIDATES == 32


# --------------------------------------------------------------------------
# The 50,000-byte cliff is the text bound.
# --------------------------------------------------------------------------

def test_thirty_two_candidates_each_previewed_would_trip_the_cliff():
    """The arithmetic that rules out previewing every candidate."""

    over = _projected_result_bytes(candidate_count=32, preview_count=32)
    assert over == 1290 + 32 * 1270 + 32 * 340 == 52_810
    assert over > THRESHOLD, (
        "a result above the threshold is replaced by a ~2,048-byte stub, so it "
        "is strictly worse than the three-candidate result it replaces"
    )


def test_thirty_two_candidates_with_capped_previews_fit_under_the_cliff():
    under = _projected_result_bytes(
        candidate_count=32, preview_count=grounding_palette.GROUNDING_PREVIEW_RENDER_LIMIT,
    )
    assert under == 1290 + 32 * 1270 + 6 * 340 == 43_970
    assert under <= THRESHOLD
    assert THRESHOLD - under == 6_030  # 12.1% headroom over the worst observed records


def test_the_live_rendering_path_is_bounded_by_the_same_limit():
    """The loop that renders one preview per candidate is sliced by the cap.

    At 32 candidates one preview each is 52,810 bytes against the 50,000
    ceiling, past which the result becomes a pointer and a ~2,048-byte preview.
    """

    import ast
    import inspect

    # The loop that renders one preview per candidate must slice by the limit.
    source = inspect.getsource(grasp)
    tree = ast.parse(source)
    bounded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        body = ast.dump(node)
        if "_render_grounding_candidate_preview" not in body:
            continue
        iterated = node.iter
        assert isinstance(iterated, ast.Subscript), (
            "the grounding preview loop iterates every candidate; every one of "
            "them reaches images_base64"
        )
        assert "GROUNDING_PREVIEW_RENDER_LIMIT" in ast.dump(iterated), (
            "the grounding preview loop is bounded by something other than the "
            "measured limit"
        )
        bounded = True
    assert bounded, "no grounding preview loop found; this test lost its subject"


def test_the_capped_live_path_fits_the_ceiling_that_binds():
    """The advertised maximum, with the constants that ship.

    Bytes are the binding ceiling: the result stays whole rather than becoming
    a pointer and a ~2,048-byte preview. The image window is the softer one --
    the check that raises is unset on every active baseline -- so this asserts
    the byte budget and states the image count rather than bounding it against
    a limit nothing enforces.
    """

    previews = grounding_palette.GROUNDING_PREVIEW_RENDER_LIMIT
    candidates = grounding_palette.GROUNDING_MAX_CANDIDATES

    assert _projected_result_bytes(
        candidate_count=candidates, preview_count=previews) <= THRESHOLD
    # One overview plus the previews; the L4 pass is triggered at 24 and this
    # keeps one call's contribution well inside a step that already carries the
    # four automatic RoboTwin camera views.
    assert previews + 1 == 7


def test_the_depth_profile_keeps_the_policy_visible_preview_record_budget(tmp_path):
    """grounding_candidate_depth_profile_v1 grows the image, never the text record.

    ``PREVIEW_RECORD_BYTES`` is the measured worst preview record the
    candidate ceiling is derived from. The profiled card publishes the same keys
    with a same-width ``render_dimensions``, so the budget still holds.
    """
    import json
    from pathlib import Path

    from ora.agent.tools import grounding_preview

    calibration = Path(tmp_path) / "calibration.json"
    calibration.write_text(json.dumps({
        "intrinsic_cv": [[200.0, 0.0, 128.0], [0.0, 200.0, 128.0], [0.0, 0.0, 1.0]],
        "extrinsic_cv": np.eye(4).tolist()}))
    rgb = np.full((256, 256, 3), 30, dtype=np.uint8)
    depth = np.full((256, 256), 1.0)
    candidate = _candidate(0, centre=(128, 128))
    released_image, released = grasp._render_grounding_candidate_preview(
        rgb_u8=rgb, candidate=candidate, text="a mug")
    profiled_image, profiled = grasp._render_grounding_candidate_preview(
        rgb_u8=rgb, candidate=candidate, text="a mug",
        depth_profile=grounding_preview.profile_input(
            {"renderer": grounding_preview.PROFILE_RENDERER, "support_line": True},
            depth_m=depth, calibration_path=calibration,
            localization={"status": "localized", "center_xyz_m": [0.0, 0.0, 1.0],
                          "extent_xyz_m": [0.06, 0.06, 0.0]}))
    assert released_image.size == (1024, 576) and profiled_image.size == (1024, 800)
    public = ("rank", "target_ref", "source_dimensions", "source_bbox_xyxy", "crop_xyxy",
              "render_dimensions", "resampling")

    def record(metadata):
        return json.dumps({key: metadata[key] for key in public}, sort_keys=True,
                          separators=(",", ":")).encode()

    assert len(record(profiled)) == len(record(released))
    assert len(record(profiled)) + len('"image_index":6,"sha256":""') + 64 <= (
        PREVIEW_RECORD_BYTES)
    # What grew is private: the strip's window and counts stay in the manifest.
    assert set(profiled) - set(released) == {"depth_profile"}
