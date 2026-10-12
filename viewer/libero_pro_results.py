"""Source-backed, protocol-aware published LIBERO-Pro result registry.

The values in this module are literature metadata for the read-only Viewer.
They are not local Robot Harness run evidence and must never be mixed into the
run registry or used to infer an experiment verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any


RESULTS_SCHEMA = "ora.libero-pro-published-results.v1"
REVIEWED_AT = "2026-08-16"

CELL_ORDER = (
    "Spat-T", "Spat-S", "Obj-T", "Obj-S",
    "Goal-T", "Goal-S", "L10-T", "L10-S",
)


@dataclass(frozen=True)
class PublishedResult:
    """One paper-reported result under one exact evaluation cohort."""

    key: str
    method: str
    cohort: str
    reported_overall: float
    scores: tuple[float | None, ...]
    method_kind: str
    adaptation: str
    source_url: str
    source_label: str
    note: str = ""

    @property
    def derived_overall(self) -> float:
        return round(fmean(score for score in self.scores if score is not None), 1)

    @property
    def common_six(self) -> float:
        return round(fmean(score for score in self.scores[:6] if score is not None), 1)

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["scores"] = [
            {"cell": cell, "score": score}
            for cell, score in zip(CELL_ORDER, self.scores, strict=True)
        ]
        value["derived_overall"] = self.derived_overall
        value["common_six"] = self.common_six
        return value


COHORTS: tuple[dict[str, Any], ...] = (
    {
        "key": "harness_8x10",
        "label": "Harness paper rerun",
        "badge": "同文同协议 · 8 cells × 10 seeds",
        "scope": "Spatial / Object / Goal / LIBERO-10 × Task / Swap",
        "trials": "800 trials per method",
        "protocol": (
            "Each cell is 10 tasks × 10 seeds. The paper authors reran the "
            "frozen pi_RLinf checkpoint and both Harness planners under this "
            "eight-cell protocol."
        ),
        "comparison": (
            "The three rows below are the cleanest apples-to-apples island. "
            "Harness uses one reference seed for bootstrapping and retrieves "
            "task/global memory; pi_RLinf is the direct frozen-VLA baseline."
        ),
    },
    {
        "key": "aspire_6x50",
        "label": "ASPIRE held-out evaluation",
        "badge": "同文同任务 · 6 cells × 50 seeds",
        "scope": "Spatial / Object / Goal × Task / Position",
        "trials": "3,000 held-out trials per method",
        "protocol": (
            "Ten tasks per cell and held-out seeds 1–50. ASPIRE learns on "
            "seeds 51–65 and evaluates one generated program per task; its "
            "CaP-Agent0 baseline regenerates a program per evaluation seed."
        ),
        "comparison": (
            "ASPIRE versus the CaP-Agent0 row is source-internal and uses the "
            "same task/seed cells, but the deployment computation differs by "
            "design. The paper rounds their overalls to 72% and 18%."
        ),
    },
    {
        "key": "rho_6x50",
        "label": "RHO / CaP-Bench evaluation",
        "badge": "同一评测面 · 6 cells × 50 trials",
        "scope": "Spatial / Object / Goal × Task / Position",
        "trials": "3,000 held-out trials per method",
        "protocol": (
            "Each cell has 10 tasks × 50 trials. RHO searches for 63 "
            "generations on unperturbed LIBERO, never trains on LIBERO-Pro, "
            "and deploys one selected repository under single-turn S4."
        ),
        "comparison": (
            "RHO and the reconstructed CaP-Agent0 aggregate use the same "
            "3,000 evaluation instances, but deliberately use different "
            "deployment regimes: RHO is S4 with no online LLM edits; "
            "CaP-Agent0 is the stronger multi-turn M4 result."
        ),
    },
    {
        "key": "rats_6x10",
        "label": "RATs held-out evaluation",
        "badge": "同文同协议 · 6 cells × 10 initializations",
        "scope": "Spatial / Object / Goal × Task / Position",
        "trials": "600 held-out trials per method",
        "protocol": (
            "Sixty held-out task cells with 10 initializations each. RATs "
            "first performs 50 play iterations, then freezes and retrieves its "
            "learned skill library at evaluation."
        ),
        "comparison": (
            "RATs versus its own CaP-Agent0 baseline is apples-to-apples. The "
            "same baseline name scores differently in the ASPIRE cohort, "
            "which is direct evidence that cross-paper percentages do not "
            "form one leaderboard."
        ),
    },
)


RESULTS = (
    PublishedResult(
        key="harness_vla_cc",
        method="Harness VLA (Claude Code)",
        cohort="harness_8x10",
        reported_overall=82.4,
        scores=(94, 80, 88, 90, 87, 87, 71, 62),
        method_kind="Agent planner + frozen pi0.5 VLA primitive",
        adaptation="One reference seed bootstrapping; task/global memory enabled",
        source_url="https://arxiv.org/html/2607.08448#S3.T3",
        source_label="Harness VLA · Table 3",
    ),
    PublishedResult(
        key="harness_vla_codex",
        method="Harness VLA (Codex)",
        cohort="harness_8x10",
        reported_overall=72.1,
        scores=(81, 69, 94, 91, 75, 66, 52, 49),
        method_kind="Agent planner + frozen pi0.5 VLA primitive",
        adaptation="One reference seed bootstrapping; task/global memory enabled",
        source_url="https://arxiv.org/html/2607.08448#S3.T3",
        source_label="Harness VLA · Table 3",
    ),
    PublishedResult(
        key="pi_rlinf",
        method="pi_RLinf (direct frozen VLA)",
        cohort="harness_8x10",
        reported_overall=50.0,
        scores=(42, 59, 71, 78, 45, 42, 49, 14),
        method_kind="Frozen pi0.5-SFT checkpoint",
        adaptation="No Harness planner or Harness memory",
        source_url="https://arxiv.org/html/2607.08448#S3.T3",
        source_label="Harness VLA · Table 3",
    ),
    PublishedResult(
        key="aspire",
        method="ASPIRE",
        cohort="aspire_6x50",
        reported_overall=72.0,
        scores=(60, 51, 95, 98, 45, 81, None, None),
        method_kind="Continual code-as-policy skill discovery",
        adaptation="Per-task learning on seeds 51–65; persistent skill library",
        source_url="https://arxiv.org/html/2607.00272#A1.T2",
        source_label="ASPIRE · Table 2",
        note="Published overall is rounded to 72%; six displayed cells average 71.7%.",
    ),
    PublishedResult(
        key="cap_agent0_aspire",
        method="CaP-Agent0 (ASPIRE protocol)",
        cohort="aspire_6x50",
        reported_overall=18.0,
        scores=(14, 12, 18, 22, 17, 26, None, None),
        method_kind="Training-free code-as-policy agent",
        adaptation="Regenerates per seed with test-time reasoning and retries",
        source_url="https://arxiv.org/html/2607.00272#A1.T2",
        source_label="ASPIRE · Table 2",
        note="Published overall is rounded to 18%; six displayed cells average 18.2%.",
    ),
    PublishedResult(
        key="rho",
        method="RHO",
        cohort="rho_6x50",
        reported_overall=45.0,
        scores=(61.6, 53.0, 37.2, 12.2, 55.6, 50.6, None, None),
        method_kind="Evolved repository-as-policy",
        adaptation="63 training generations on unperturbed LIBERO; S4 deployment",
        source_url="https://arxiv.org/html/2606.16458#S4.SS1.SSS2",
        source_label="RHO · Section 4.1 / Appendix A.2.7",
    ),
    PublishedResult(
        key="cap_agent0_rho",
        method="CaP-Agent0 (RHO comparator)",
        cohort="rho_6x50",
        reported_overall=18.17,
        scores=(14, 12, 18, 22, 17, 26, None, None),
        method_kind="Training-free multi-turn code-as-policy agent",
        adaptation="M4 deployment with visual differencing and corrective LLM calls",
        source_url="https://arxiv.org/html/2606.16458#S4.SS1.SSS2",
        source_label="RHO · comparator reconstruction",
        note="The paper reconstructs 545/3,000 = 18.17% from CaP-X cells.",
    ),
    PublishedResult(
        key="rats",
        method="RATs",
        cohort="rats_6x10",
        reported_overall=43.8,
        scores=(31, 29, 63, 61, 36, 43, None, None),
        method_kind="Play-time robotics agent teams",
        adaptation="50 play iterations; frozen learned skills at evaluation",
        source_url="https://arxiv.org/html/2606.19419#S4.T1",
        source_label="Playful Agentic Robot Learning · Table 1",
    ),
    PublishedResult(
        key="cap_agent0_rats",
        method="CaP-Agent0 (RATs protocol)",
        cohort="rats_6x10",
        reported_overall=23.2,
        scores=(23, 13, 31, 27, 16, 29, None, None),
        method_kind="Training-free code-as-policy agent",
        adaptation="No play-learned skill library",
        source_url="https://arxiv.org/html/2606.19419#S4.T1",
        source_label="Playful Agentic Robot Learning · Table 1",
    ),
)


OPENETA_BOUNDARY = {
    "method": "OpenETA",
    "result": None,
    "label": "No formal LIBERO-Pro result",
    "protocol": (
        "The OpenETA paper formally evaluates standard LIBERO only: Spatial, "
        "Object, Goal and Long / LIBERO-10. It explicitly places LIBERO-Pro "
        "outside scope and says a separate preregistered manifest is required."
    ),
    "source_url": "https://arxiv.org/html/2608.03924#S5.SS1.SSS1",
    "source_label": "OpenETA · Section 5.1.1",
}


def published_results_payload() -> dict[str, Any]:
    """Return the literature registry as a JSON-safe read-only projection."""

    rows = [result.payload() for result in RESULTS]
    return {
        "schema": RESULTS_SCHEMA,
        "reviewed_at": REVIEWED_AT,
        "cells": list(CELL_ORDER),
        "cohorts": [
            {
                **cohort,
                "methods": [
                    row for row in rows if row["cohort"] == cohort["key"]
                ],
            }
            for cohort in COHORTS
        ],
        "openeta": dict(OPENETA_BOUNDARY),
        "evidence_boundary": {
            "kind": "paper_reported_literature_metadata",
            "contains_local_rollouts": False,
            "global_ranking_valid": False,
            "comparability": "within_cohort_only",
        },
    }
