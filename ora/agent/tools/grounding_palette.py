"""One colour table, and the ceiling on grounding candidates derived from it.

Two three-entry tuples used to sit inline in the two rendering paths and
disagree about what happens past the third candidate: the overview zipped
non-strictly and drew nothing, while the enlarged preview indexed modulo
three and painted the fourth in the first one's colour. Neither raised, so
how much a policy could be shown was decided by the length of a tuple
nobody read as a bound.
"""

from __future__ import annotations

#: Grounding overlay colours, one per candidate, and the ceiling derived from
#: the table rather than restated beside it. Two three-entry tuples used to sit
#: inline in the two rendering paths and disagree about what happens past the
#: third candidate: the overview zipped non-strictly and drew nothing, while the
#: enlarged preview indexed modulo three and painted the fourth in the first
#: one's colour. Neither raised, so the ceiling on how much a policy could be
#: shown was the length of a tuple nobody read as a bound.
#:
#: Chosen by farthest-point sampling of the RGB cube, restricted to colours that
#: read against a photograph -- a peak channel of at least 150 so a mask is not
#: mistaken for shadow, and a floor of at most 200 so it is not mistaken for
#: glare. Minimum pairwise separation is 86.7, against the 46.5 an evenly spaced
#: hue walk of the same size achieves.
#: The first three entries are the released palette, byte-for-byte. Widening
#: the table may only append; changing an entry a released rendering already
#: used would change the images sealed runs were shown.
GROUNDING_CANDIDATE_COLOURS = (
    (48, 220, 96), (64, 180, 255), (255, 190, 48),
    (187, 255, 255), (255, 0, 0), (0, 0, 204), (0, 255, 0),
    (255, 0, 255), (255, 255, 0), (0, 204, 204), (153, 119, 119),
    (102, 102, 255), (102, 255, 102), (255, 136, 221), (255, 255, 136),
    (102, 153, 0), (136, 0, 170), (0, 153, 85), (221, 119, 0),
    (255, 34, 119), (153, 17, 51), (51, 85, 153), (136, 255, 0),
    (255, 153, 102), (102, 187, 187), (0, 102, 255), (0, 255, 102),
    (68, 255, 255), (85, 0, 255), (187, 68, 221), (170, 187, 51),
    (187, 204, 170),
)
GROUNDING_MAX_CANDIDATES = len(GROUNDING_CANDIDATE_COLOURS)

#: How many candidate previews one call may put in front of the model.
#:
#: The binding ceiling is bytes, not images. 32 candidate records at the worst
#: observed 1,270 canonical bytes, plus 32 preview records at 340 and the 1,290
#: envelope, is 52,810 against `eager_large_result_bytes` of 50,000 -- past which
#: the whole result is replaced by a pointer and a ~2,048-byte preview, which is
#: strictly worse than the three-candidate result it replaced. Capped at six the
#: same request is 43,970, with 12.1% headroom.
#:
#: The image side is a cost, not a cliff, and an earlier version of this comment
#: had it wrong. `trigger_inline_images` is 24, but it admits the L4
#: externalization pass and evicts nothing; the only hard image ceiling was
#: `max_inline_images_per_request`, which no retention of this tree accepts.
#: 33 inline images rewrites the cached prefix; it does not end the episode.
#:
#: Six is 24, less the p99 of images already produced in one observation step
#: over 324 sealed runs (17), less the overview this call also returns.
#:
#: Raising `max_candidates` still widens what the policy may choose between. It
#: does not widen what it is shown: past this rank a candidate arrives as a
#: record with its localization and its colour in the overview, without a
#: close-up of its own.
GROUNDING_PREVIEW_RENDER_LIMIT = 6


def grounding_candidate_colour(rank: int) -> tuple[int, int, int]:
    """The colour for one candidate, refusing past the table rather than wrapping.

    Wrapping is what made a fourth candidate unattributable: it appeared in the
    enlarged preview wearing the first candidate's colour while the overview
    drew it in none at all.
    """

    if not 0 <= rank < GROUNDING_MAX_CANDIDATES:
        raise ValueError(
            f"grounding candidate rank {rank} is outside the "
            f"{GROUNDING_MAX_CANDIDATES}-colour palette"
        )
    return GROUNDING_CANDIDATE_COLOURS[rank]
