"""Value normalization and distribution summaries for Runtime metrics.

`_number` and `_integer` accept only finite non-negative JSON numbers, `_round`
fixes six decimals, and `summarize` reports count, total, mean, p50, p95, min
and max. `summarize_signed` keeps negative values and counts them. `metrics`, `metrics_provider_usage` and `metrics_native_projection`
share them; `metrics` re-exports every name.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return None if number is None or not number.is_integer() else int(number)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize(values: Iterable[float | int]) -> dict[str, Any]:
    """Return one deterministic distribution summary for non-negative values."""
    clean = [float(value) for value in values if _number(value) is not None]
    if not clean:
        return {
            "count": 0, "total": None, "mean": None, "p50": None,
            "p95": None, "min": None, "max": None,
        }
    ordered = sorted(clean)
    return {
        "count": len(ordered),
        "total": _round(sum(ordered)),
        "mean": _round(statistics.fmean(ordered)),
        "p50": _round(_percentile(ordered, 0.50)),
        "p95": _round(_percentile(ordered, 0.95)),
        "min": _round(ordered[0]),
        "max": _round(ordered[-1]),
    }


def summarize_signed(values: Iterable[float | int]) -> dict[str, Any]:
    """Summarize finite values of either sign; ``increased_count`` counts the negatives.

    Used for per-pass byte reductions, where a negative value is a pass that
    grew the request and must stay visible instead of being dropped.
    """
    clean = [
        float(value) for value in values
        if not isinstance(value, bool) and isinstance(value, (int, float))
        and math.isfinite(float(value))
    ]
    if not clean:
        return {**summarize(()), "increased_count": 0}
    ordered = sorted(clean)
    return {
        "count": len(ordered),
        "total": _round(sum(ordered)),
        "mean": _round(statistics.fmean(ordered)),
        "p50": _round(_percentile(ordered, 0.50)),
        "p95": _round(_percentile(ordered, 0.95)),
        "min": _round(ordered[0]),
        "max": _round(ordered[-1]),
        "increased_count": sum(value < 0 for value in ordered),
    }
