"""The expert-unsolved RoboTwin seed: a labelled exception to seed admission.

A RoboTwin cell normally runs only on a seed its expert solved
(``robotwin_evaluation.seed_admission``). An external case list (the RPent
leaderboard's 250 C2R pairs) can name seeds our expert never solves; scoring
on the same denominator needs those cells run anyway. Such a RunSpec keeps
``qualification.required: false`` and states why:

    qualification:
      required: false
      exception:
        kind: robotwin_expert_unsolved_seed_v1
        reason: <non-empty text>
        evidence: [outputs/<experiment>/qualification/expert-unsolved/<file>.json, ...]

Every evidence file is an official seed manifest bound by sha256 at resolve
time; preflight re-reads it and passes ``seed_qualification`` only when each
one shows the expert failed or raised on this task, seed, scene configuration
and revision (``robotwin_evaluation.expert_unsolved``). An unstable scene, a
conflicting seed or an accepted seed never qualifies. The exception stays in
the plan, so every reader can separate these cells from admitted ones.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

EXPERT_UNSOLVED_KIND = "robotwin_expert_unsolved_seed_v1"
MAX_EVIDENCE = 10


class QualificationExceptionError(ValueError):
    """A qualification exception is malformed."""


def resolve_exception(qualification: Mapping[str, Any],
                      bind: Callable[[Any, str], dict[str, Any]]) -> dict[str, Any] | None:
    """The bound exception of a resolved ``qualification``, or None when it states none.

    ``bind(path, label)`` binds one generated resource under the experiment's
    outputs (``spec._generated_resource``) and returns ``{path, sha256, bytes}``.
    """
    exception = qualification.get("exception")
    if exception is None:
        return None
    if qualification.get("required") is not False:
        raise QualificationExceptionError("qualification.exception requires qualification.required: false")
    if not isinstance(exception, Mapping) or set(exception) != {"kind", "reason", "evidence"}:
        raise QualificationExceptionError("qualification.exception fields must be kind, reason and evidence")
    if exception["kind"] != EXPERT_UNSOLVED_KIND:
        raise QualificationExceptionError(f"qualification.exception.kind must be {EXPERT_UNSOLVED_KIND}")
    if not isinstance(exception["reason"], str) or not exception["reason"].strip():
        raise QualificationExceptionError("qualification.exception.reason must be non-empty text")
    evidence = exception["evidence"]
    if not isinstance(evidence, list) or not 1 <= len(evidence) <= MAX_EVIDENCE:
        raise QualificationExceptionError(f"qualification.exception.evidence must list 1-{MAX_EVIDENCE} manifests")
    return {"kind": EXPERT_UNSOLVED_KIND, "reason": exception["reason"],
            "evidence": [bind(item, f"qualification.exception.evidence[{index}]")
                         for index, item in enumerate(evidence)]}


def check_exception(plan: Mapping[str, Any], workspace_root: Path) -> tuple[bool, dict[str, Any]]:
    """Preflight's verdict on a plan's exception: every evidence manifest unchanged and expert-unsolved."""
    from ora.orchestrator.robotwin_evaluation import expert_unsolved
    from ora.orchestrator.spec_values import file_sha256

    exception = plan["qualification"]["exception"]
    runtime = plan.get("runtime", {})
    if runtime.get("kind") != "robotwin":
        return False, {"exception": exception["kind"], "error": "only a RoboTwin cell may run an unsolved seed"}
    verdicts = []
    for item in exception["evidence"]:
        path = workspace_root / item["path"]
        try:
            unchanged = file_sha256(path) == item["sha256"]
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, {"exception": exception["kind"], "error": f"{item['path']}: {exc}"}
        passed, detail = expert_unsolved(manifest, task=plan["task"], seed=plan["seed"],
                                         task_config=runtime.get("task_config"),
                                         revision=runtime.get("required_revision"))
        verdicts.append({"path": item["path"], "unchanged": unchanged, "expert_unsolved": passed, **detail})
    passed = all(verdict["unchanged"] and verdict["expert_unsolved"] for verdict in verdicts)
    return passed, {"exception": exception["kind"], "reason": exception["reason"], "evidence": verdicts}


__all__ = ["EXPERT_UNSOLVED_KIND", "MAX_EVIDENCE", "QualificationExceptionError", "check_exception",
           "resolve_exception"]
