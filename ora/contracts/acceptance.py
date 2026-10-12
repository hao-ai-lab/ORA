"""Closed numeric motion evidence; no controller, task or verdict authority."""
from __future__ import annotations

import math
from typing import Any, Mapping


STAGES = {"precontact_global", "post_global_final", "local_entry_envelope",
          "final_endpoint", "controller_terminal_endpoint", "local_solver"}


def motion_acceptance_check(*, stage: str, errors: Mapping[str, Mapping[str, Any]],
                            position_tolerance_m: float, orientation_tolerance_rad: float,
                            comparison: str = "<=") -> dict[str, Any]:
    """Pair actual stage residuals with its exact source-owned thresholds.

    This is a numeric comparison only: convergence, other guards and official
    task termination remain independent requirements. No thresholds live here.
    """
    if stage not in STAGES or comparison not in {"<", "<="}:
        raise ValueError("motion acceptance stage or comparison is invalid")
    if not isinstance(errors, Mapping) or not errors or set(errors) - {"left", "right"}:
        raise ValueError("motion acceptance requires selected public arms")
    limits = {"position_m": position_tolerance_m, "orientation_rad": orientation_tolerance_rad}
    if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in limits.values()):
        raise ValueError("motion acceptance requires positive finite thresholds")
    arms = {}
    for arm, source in errors.items():
        if not isinstance(source, Mapping) or set(source) not in (
            {"translation_m", "orientation_rad"}, {"position_m", "orientation_rad"},
        ):
            raise ValueError("motion acceptance residual fields differ")
        residual = {"position_m": source.get("position_m", source.get("translation_m")),
                    "orientation_rad": source["orientation_rad"]}
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in residual.values()):
            raise ValueError("motion acceptance residual must be finite and nonnegative")
        within = {key: (x < limits[key] if comparison == "<" else x <= limits[key])
                  for key, x in residual.items()}
        arms[arm] = {"residual": residual, "within_tolerance": within}
    return {"stage": stage, "comparison": comparison, "tolerance": limits, "arms": arms,
            "claim": "numeric_bounds_only_not_motion_or_task_success"}


def validate_motion_acceptance_checks(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > 12:
        raise ValueError("motion acceptance checks must be a bounded list")
    checked = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"stage", "comparison", "tolerance", "arms", "claim"}:
            raise ValueError("motion acceptance check fields differ")
        arms, tolerance = item["arms"], item["tolerance"]
        if item["stage"] == "gripper_endpoint":
            if (not isinstance(arms, Mapping) or not isinstance(tolerance, Mapping)
                    or set(tolerance) != {"opening_normalized"}
                    or any(not isinstance(v, Mapping) or set(v) != {"residual", "within_tolerance"}
                           or not isinstance(v["residual"], Mapping)
                           or set(v["residual"]) != {"opening_normalized"} for v in arms.values())):
                raise ValueError("gripper acceptance fields differ")
            canonical = gripper_acceptance_check(
                errors={arm: v["residual"]["opening_normalized"] for arm, v in arms.items()},
                tolerance_normalized=tolerance["opening_normalized"], comparison=item["comparison"])
            if canonical != item:
                raise ValueError("gripper acceptance comparison differs from its source numbers")
            checked.append(canonical)
            continue
        if not isinstance(arms, Mapping) or not isinstance(tolerance, Mapping) or set(tolerance) != {"position_m", "orientation_rad"}:
            raise ValueError("motion acceptance arm or tolerance fields differ")
        if any(not isinstance(v, Mapping) or set(v) != {"residual", "within_tolerance"} for v in arms.values()):
            raise ValueError("motion acceptance residual fields differ")
        canonical = motion_acceptance_check(stage=item["stage"], comparison=item["comparison"],
            errors={k: v["residual"] for k, v in arms.items()},
            position_tolerance_m=tolerance["position_m"], orientation_tolerance_rad=tolerance["orientation_rad"])
        if canonical != item:
            raise ValueError("motion acceptance comparison differs from its source numbers")
        checked.append(canonical)
    return checked


def gripper_acceptance_check(*, errors: Mapping[str, float], tolerance_normalized: float,
                             comparison: str = "<=") -> dict[str, Any]:
    """The same residual/bound contract for a controller that checks opening."""
    if (not isinstance(errors, Mapping) or not errors or set(errors) - {"left", "right"}
            or comparison not in {"<", "<="}
            or type(tolerance_normalized) not in (int, float)
            or not math.isfinite(tolerance_normalized) or tolerance_normalized <= 0):
        raise ValueError("gripper acceptance bound or arms are invalid")
    arms = {}
    for arm, error in errors.items():
        if type(error) not in (int, float) or not math.isfinite(error) or error < 0:
            raise ValueError("gripper acceptance residual must be finite and nonnegative")
        within = error < tolerance_normalized if comparison == "<" else error <= tolerance_normalized
        arms[arm] = {"residual": {"opening_normalized": error},
                     "within_tolerance": {"opening_normalized": within}}
    return {"stage": "gripper_endpoint", "comparison": comparison,
            "tolerance": {"opening_normalized": tolerance_normalized}, "arms": arms,
            "claim": "numeric_bounds_only_not_motion_or_task_success"}
