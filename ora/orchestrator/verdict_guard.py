"""Guard the recorded official verdict against the policy-visible checker.

The trusted termination reason, the policy-visible ``check_task`` record and
the persisted official verdict are three projections of one environment
predicate. When two of them disagree the Harness is broken, not the robot
task: on 2026-09-08 a RoboTwin ``hanging_mug`` round terminated with reason
``official_success``, told the policy ``check_task`` was true, forced a
``finish_episode`` success claim and then recorded ``official_success:
false``. The policy spent its next round reporting the contradiction instead
of moving the robot, which was the correct reading of a Harness defect.

A divergence is therefore raised here, at the round boundary, as an
infrastructure fault. It is never a behavioral verdict: the affected round
carries no official success or failure, exactly like any other trusted
infrastructure fault.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .faults import HarnessFault


#: Infrastructure termination reason and fault code for the contradiction.
OFFICIAL_VERDICT_CHECKER_DIVERGENCE = "official_verdict_checker_divergence"
DIVERGENCE_SCHEMA = "ora.official-verdict-checker-divergence.v1"
#: The checker snapshot ``check_task`` reads and ``publish`` writes.
TASK_CHECK_FILE = "policy-task-check.json"
#: Append-only control record of every contradiction the guard raised.
DIVERGENCE_FILE = "official-verdict-divergence.jsonl"


def official_verdict_divergence(
    verdict: Mapping[str, Any] | None,
    *,
    task_check: Mapping[str, Any] | None,
    round_index: int | None = None,
) -> dict[str, Any] | None:
    """Return a divergence record when checker and verdict disagree.

    Both signals must be present and Boolean for the comparison to mean
    anything: an environment without an official predicate (Synthetic M0)
    records ``official_success: null``, and a ToolSurface without the
    faithful checker writes no policy-visible record at all. Neither is a
    contradiction, so both return ``None``.

    The comparison is deliberately asymmetric, and the asymmetry follows from
    the coordinator's loop order. The checker record is a *snapshot*, written
    by ``publish`` from the ``status()`` reason at the last policy boundary;
    the verdict is read at the termination boundary that follows it. Two
    things can therefore move between them, and both move the same way:

    * a late latch -- the predicate first holds during the final commanded
      motion, after the snapshot was written; and
    * a shadowed latch -- the trusted ``abort_reason`` takes precedence over
      ``official_success`` in ``status()``, so a snapshot taken after an
      abort reports ``task_success: false`` for an episode that did latch.

    In both, the snapshot says false and the record says true, and the record
    is right: the latch never clears. Reporting that as an infrastructure
    fault would null a round the published protocol counts as a success,
    which is exactly what this guard exists to prevent. The reverse -- the
    policy was *told* the task was complete and the record denies it -- is
    the real contradiction, the one that compelled a ``finish_episode``
    success claim against a recorded failure, and it is the only one raised.
    """

    if not isinstance(verdict, Mapping) or not isinstance(task_check, Mapping):
        return None
    official = verdict.get("official_success")
    checker = task_check.get("task_success")
    if not isinstance(official, bool) or not isinstance(checker, bool):
        return None
    if official == checker:
        return None
    if official and not checker:
        # A late or shadowed latch, not a contradiction. The coordinator
        # republishes the checker from the persisted verdict so the next
        # policy-visible read agrees with the record.
        return None
    return {
        "schema": DIVERGENCE_SCHEMA,
        "reason": OFFICIAL_VERDICT_CHECKER_DIVERGENCE,
        "round_index": round_index,
        "policy_visible_task_success": checker,
        "recorded_official_success": official,
        "checker_source": task_check.get("source"),
        "verdict_source": verdict.get("source"),
        "official_success_rule": verdict.get("official_success_rule"),
        "predicate_ever_held": verdict.get("eval_success_seen"),
        "predicate_held_at_persist": verdict.get("predicate_held_at_persist"),
        "environment_status_reason": (
            verdict.get("environment_status", {}).get("reason")
            if isinstance(verdict.get("environment_status"), Mapping)
            else None
        ),
    }


def verdict_divergence_fault(divergence: Mapping[str, Any]) -> HarnessFault:
    """Type the divergence as a non-retryable trusted-boundary fault."""

    return HarnessFault(
        code=OFFICIAL_VERDICT_CHECKER_DIVERGENCE,
        layer="environment",
        retryable=False,
        # The world is exactly where the terminal round left it; what is
        # unknown is which projection of the predicate to believe, and no
        # further observation resolves that.
        world_state_certainty="unchanged",
        recoverable_at="operator",
        requires_fresh_observation=False,
        fatal=True,
        message=(
            "the policy-visible task checker reported task_success="
            f"{divergence.get('policy_visible_task_success')!r} while the "
            "recorded official verdict reported official_success="
            f"{divergence.get('recorded_official_success')!r}"
        ),
    )


def _optional_json(path: Path) -> dict[str, Any] | None:
    """Read a trusted control record, or nothing when it is absent."""

    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class RoundVerdictGuard:
    """Persist each official verdict and guard it against the checker snapshot.

    The recorded verdict, the trusted termination reason and the
    policy-visible ``check_task`` record are three projections of one
    environment predicate. A disagreement is a Harness defect, so it is raised
    here, at the round boundary, as an infrastructure fault -- never as a
    behavioral verdict, and never as something a policy has to discover
    mid-run.

    Order matters. The guard compares the record against the checker snapshot
    the policy actually saw, so it runs first; only then is the checker
    republished from the record. On a real contradiction the snapshot is left
    exactly as written, because it is the evidence.

    The coordinator owns the file helpers, the live port and the auxiliary
    failure ledger; they are passed in so this module holds no run state beyond
    the current round's divergence.
    """

    def __init__(
        self, *, runtime: Any, control: Path, events: Path, surface: Any,
        write_json: Callable[[Path, Any], None],
        append: Callable[[Path, Mapping[str, Any]], None],
        publish_live: Callable[[str, Mapping[str, Any]], None],
        record_failure: Callable[[str, BaseException], None],
    ) -> None:
        self._runtime = runtime
        self._control = control
        self._events = events
        self._surface = surface
        self._write_json = write_json
        self._append = append
        self._publish_live = publish_live
        self._record_failure = record_failure
        #: The current round's contradiction, or ``None``.
        self.divergence: dict[str, Any] | None = None

    def reset(self) -> None:
        """Start a new round with no recorded contradiction."""

        self.divergence = None

    def persist(self, round_index: int | None) -> dict[str, Any]:
        """Persist the official verdict, then compare or republish."""

        record = self._runtime.persist_official_verdict()
        divergence = official_verdict_divergence(
            record,
            task_check=_optional_json(self._control / TASK_CHECK_FILE),
            round_index=round_index,
        )
        if divergence is None:
            try:
                self._republish_task_check(record)
            except Exception as exc:
                self._record_failure("verdict_checker_guard:republish", exc)
            return record
        self.divergence = divergence
        fault = verdict_divergence_fault(divergence)
        try:
            self._append(self._control / DIVERGENCE_FILE, divergence)
            self._append(self._events, {
                "type": "runtime_fault",
                "fault": fault.as_dict(),
                "divergence": divergence,
                "round_index": round_index,
                "time": time.time(),
            })
        except Exception as exc:
            self._record_failure("verdict_checker_guard:record", exc)
        self._publish_live("runtime_fault", fault.as_dict())
        return record

    def _republish_task_check(self, record: Mapping[str, Any]) -> None:
        """Re-project the persisted verdict into the policy-visible checker.

        ``publish`` writes the checker from the ``status()`` reason at a policy
        boundary, and ``status()`` gives a trusted abort reason precedence over
        ``official_success``. A predicate that first holds during the final
        commanded motion, or that latched under an abort, therefore leaves the
        last snapshot stale-false while the persisted record says true. Any
        terminal policy turn -- and the ``finish_episode`` rule that reads the
        same file -- must see the predicate the verdict was scored from, so the
        record republishes it.
        """

        official = record.get("official_success")
        if (
            not self._surface.supports("faithful_check_task")
            or not isinstance(official, bool)
        ):
            return
        self._write_json(self._control / TASK_CHECK_FILE, {
            "schema": "ora.openeta-task-check.v1",
            "terminal": True,
            "task_success": official,
            "source": str(record.get("source") or "official_verdict"),
            "projected_from": "persisted_official_verdict",
            "updated_unix_s": time.time(),
        })


__all__ = [
    "DIVERGENCE_FILE",
    "DIVERGENCE_SCHEMA",
    "OFFICIAL_VERDICT_CHECKER_DIVERGENCE",
    "RoundVerdictGuard",
    "TASK_CHECK_FILE",
    "official_verdict_divergence",
    "verdict_divergence_fault",
]
