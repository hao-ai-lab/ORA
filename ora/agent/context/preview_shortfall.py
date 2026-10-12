# Copyright contributors to the Robot Harness Runtime project.
"""What an externalized preview is missing, stated in the record itself.

Its own module because the projection it sits in is already oversized and the
question it answers is separable: the preview is a byte prefix of a larger
result, and nothing in the record said so. Across 619 runs and 63,954 tool
calls the tool that opens the full result was called twice, both in one run
and both correctly, so the recovery path works and was simply never known to
be needed.
"""

from __future__ import annotations

from typing import Any, Mapping


def _preview_shortfall(
    content: str, preview: str, value: Any,
) -> dict[str, Any]:
    """Say what the preview is missing, so a reader knows to open the pointer.

    The record already carries the pointer and the tool that opens it, and one
    run proved that is enough to use it correctly, paging included. It was used
    twice in 63,954 tool calls, because nothing said it was needed: the preview
    is the first ``eager_preview_bytes`` of the content, cut wherever those
    bytes end, so a three-candidate result arrives as one whole candidate, half
    of the second, and no sign that a third exists. This names the shortfall
    rather than changing what is kept.

    ``preview_truncated`` is false when the whole content fit, and false when a
    ToolSurface's own bounded summary replaced the slice -- that summary is
    complete for what it covers, and ``preview_is_summary`` says the reader is
    looking at a summary rather than the result. ``item_counts`` reports the
    length of every top-level list in the original, so a reader counts what
    exists instead of what survived the cut.
    """

    total = len(content.encode("utf-8"))
    kept = len(preview.encode("utf-8"))
    summarized = bool(preview) and preview != content[:len(preview)]
    shortfall: dict[str, Any] = {
        "preview_truncated": (not summarized) and kept < total,
        "preview_bytes": kept,
        "result_bytes": total,
    }
    if summarized:
        shortfall["preview_is_summary"] = True
    if isinstance(value, Mapping):
        counts = {
            key: len(item) for key, item in value.items()
            if isinstance(item, list)
        }
        if counts:
            shortfall["item_counts"] = counts
    return shortfall
