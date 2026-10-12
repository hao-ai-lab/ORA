"""Source-bound RoboDojo receipt, round-isolation and official-verdict joins."""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Mapping
from .evidence_io import EvaluationError, _hash, _load, _receipt_execution_fault
from . import robodojo_layout_evaluation as layouts
from .policy_finish import admitted_abort_round

ROBODOJO_VERDICT_SCHEMA = "ora.robodojo-official-verdict.v2"
ROBODOJO_ROUND_ISOLATION_SCHEMA = "ora.robodojo-round-isolation.v1"
#: The restore facts every RoboDojo round-reset record must carry as measured
#: True. The names are the session's own; a backend that stops publishing one
#: fails this join instead of quietly narrowing the guarantee.
ROBODOJO_ROUND_ISOLATION_CHECKS = (
    "arm_home_restored",
    "carried_channels_cleared",
    "episode_counters_zeroed",
    "failure_latch_cleared",
    "layout_identity_unchanged",
    "official_lifecycle_rebound",
    "official_terminal_cleared",
)
ROBODOJO_TERMINAL_BASIS = "official_terminal_transition"
ROBODOJO_POLICY_BUDGET_BASIS = "standing_official_predicate_at_policy_budget"




def _robodojo_verdict_untrusted(
    verdict: Mapping[str, Any], record: Mapping[str, Any] | None,
) -> bool:
    """Whether a RoboDojo verdict may not be read as the official Boolean.

    Two provenances carry a Boolean. ``official_terminal_transition`` is the
    simulator's own terminal transition and must claim ``terminal`` true.
    ``standing_official_predicate_at_policy_budget`` is a round the Supervisor
    stopped on its own budget: the benchmark never declared a terminal
    transition, so ``terminal`` stays false, and the Boolean is the standing
    predicate the last ``RewardManager.step`` evaluated. Any other provenance,
    a null Boolean, or a Boolean on a round an infrastructure fault ended is
    refused: a broken worker is never published as a task failure. The fault is
    joined twice — the backend's own ``fault_reason`` and, when the round or
    session record carries it, the Supervisor's termination authority — so
    neither side alone can publish a fault as a behavioral result.
    """

    if not isinstance(record, Mapping) or not isinstance(
        record.get("official_success"), bool
    ):
        return True
    if record.get("termination_authority") == "infrastructure":
        return True
    if not isinstance(verdict.get("official_success"), bool):
        return True
    if verdict.get("fault_reason") is not None:
        return True
    basis = verdict.get("verdict_basis")
    if basis == ROBODOJO_TERMINAL_BASIS:
        return (
            verdict.get("terminal") is not True
            or verdict.get("abort_reason") is not None
        )
    if basis == ROBODOJO_POLICY_BUDGET_BASIS:
        return verdict.get("terminal") is not False
    return True


def _robodojo_round_verdict_errors(
    native: Path, result: Mapping[str, Any], runtime: Mapping[str, Any],
    expected_identity: Mapping[str, Any], source_hash: str, *,
    rounds: int, completed: bool, reset_clocks: list[int],
    native_totals: list[int], physics_totals: list[int],
    identities: list[Mapping[str, Any]] | None = None,
    verdict_schema: str = ROBODOJO_VERDICT_SCHEMA,
) -> list[str]:
    """Join every persistent-session RoboDojo verdict to its own round.

    The Supervisor copies each round's ``native/official-verdict.json`` into
    ``native/rounds/round-XXXX/`` before rotating the live file away, and the
    final round's live file is never rotated. The archived copy is therefore
    the per-round authority, and the live file is only the final round's
    fallback when that best-effort copy failed. A verdict counts its own
    round's native actions, physics steps and reset clock, so joining it
    against session totals would reject a correct K-round run. A scheduled
    session's verdicts each carry their own round's layout identity.
    """

    errors: list[str] = []
    records = result.get("rounds")
    if not isinstance(records, list):
        records = []
    live = native / "official-verdict.json"
    if not completed and not (live.exists() or any(
        (native / "rounds" / f"round-{index:04d}" / "official-verdict.json").exists()
        for index in range(rounds)
    )):
        return errors  # A session that never reached a verdict has none to join.
    limit = runtime.get("native_action_limit", 200)
    for index in range(rounds):
        record = records[index] if index < len(records) else None
        round_completed = (
            isinstance(record, Mapping) and record.get("state") == "completed"
        )
        if isinstance(record, Mapping) and admitted_abort_round(record):
            # An admitted infrastructure abort persists no verdict for its
            # round (policy_finish); there is nothing to join.
            continue
        path = native / "rounds" / f"round-{index:04d}" / "official-verdict.json"
        if not path.is_file() and index == rounds - 1:
            path = live
        try:
            if path.is_symlink() or not path.is_file():
                raise EvaluationError("official verdict is missing or unsafe")
            verdict = _load(path)
        except (EvaluationError, OSError):
            if completed or round_completed:
                errors.append(f"robodojo_official_verdict_missing:{index}")
            continue
        native_actions = native_totals[index] if index < len(native_totals) else None
        physics_steps = physics_totals[index] if index < len(physics_totals) else None
        reset_clock = reset_clocks[index] if index < len(reset_clocks) else None
        identity = identities[index] if identities else expected_identity
        if (
            verdict.get("schema") != verdict_schema
            or verdict.get("source") != "RoboDojo EvalEnv.is_episode_end"
            or verdict.get("source_manifest_sha256") != source_hash
            or verdict.get("round_index") != index
            or any(
                verdict.get(key) != expected
                for key, expected in identity.items()
            )
            or not isinstance(record, Mapping)
            or record.get("verdict") != verdict
            or record.get("official_success") != verdict.get("official_success")
        ):
            errors.append(f"robodojo_official_verdict_binding_invalid:{index}")
        if completed and (
            type(verdict.get("native_actions")) is not int
            or native_actions is None
            or verdict["native_actions"] != native_actions
            or native_actions > limit
            or type(verdict.get("physics_steps")) is not int
            or verdict["physics_steps"] != physics_steps
            or verdict.get("reset_physics_steps") != reset_clock
        ):
            errors.append(f"robodojo_official_counter_mismatch:{index}")
        if completed and _robodojo_verdict_untrusted(verdict, record):
            errors.append("missing_trusted_boolean_verdict")
    last = native / "rounds" / f"round-{rounds - 1:04d}" / "official-verdict.json"
    if completed and last.is_file() and live.is_file():
        try:
            if _hash(last) != _hash(live):
                raise EvaluationError("archived and live final verdicts differ")
        except (EvaluationError, OSError):
            errors.append(f"robodojo_official_verdict_binding_invalid:{rounds - 1}")
    return errors


def _robodojo_worker_receipts(native: Path, action: Mapping[str, Any], value: Mapping[str, Any]):
    """Resolve trusted causal wrappers to ordered, hash-bound worker leaves.

    Composite and skill records live in the causal log. Every worker leaf must
    still match its own immutable file; the tree join verifies parent action,
    committed stage order, child identity and observation binding first.
    """
    from ora.agent.evidence.command_tree import command_leaves

    resolved: dict[str, tuple[Path, Mapping[str, Any]]] = {}

    def resolve(record):
        if not isinstance(record, Mapping):
            raise EvaluationError("receipt record is missing")
        if record.get("schema") in {
            "ora.composite-motion-receipt.v1",
            "ora.manipulation-skill-sequence-receipt.v1",
        }:
            return record
        relative = record.get("path")
        if not isinstance(relative, str) or not relative:
            raise EvaluationError("receipt path is missing")
        path = native / relative
        if (Path(relative).is_absolute() or ".." in Path(relative).parts
                or path.is_symlink() or not path.resolve().is_relative_to(native)):
            raise EvaluationError("receipt path is unsafe")
        if _hash(path) != record.get("sha256") or _load(path) != record.get("receipt"):
            raise EvaluationError("receipt bytes differ from the causal trace")
        receipt = record["receipt"]
        identity = receipt.get("action_id")
        if not isinstance(identity, str) or identity in resolved:
            raise EvaluationError("duplicate or missing worker action identity")
        resolved[identity] = (path, receipt)
        return receipt

    parent = resolve(value)
    if parent.get("action_id") != action.get("action_id"):
        raise EvaluationError("receipt action identity differs from its submitted action")
    if parent.get("schema") in {
        "ora.composite-motion-receipt.v1",
        "ora.manipulation-skill-sequence-receipt.v1",
    }:
        # Materialize before yielding so a broken later child cannot make a
        # partially validated tree contribute to the trusted counter totals.
        leaves = list(command_leaves(action, parent, resolve))
        return [resolved[child["action_id"]] for child, _ in leaves]
    return [resolved[parent["action_id"]]]


def _robodojo_evidence_errors(
    run_root: Path, plan: Mapping[str, Any], result: Mapping[str, Any],
    events: list[Mapping[str, Any]],
) -> list[str]:
    """Verify the single-layout official boundary without importing Isaac.

    Runtime completion, task failure and a lost worker are distinct facts.
    Neither a missing verdict nor RoboDojo's pre-terminal success default is
    admissible as a completed episode. Reset, verdict and actual issued
    receipts remain joined to their immutable source artifacts.

    A persistent K-round session resets the same plant once per round, so
    every counter join below is per round. The PhysX clock keeps advancing
    across a round reset while the wrapper's own native and physics counters
    restart at zero, and the backend writes one official verdict per round.
    Joining a later round's physics journal against round 0's reset clock, or
    the final verdict against the sum of every receipt in the session, would
    reject a correct multi-round run, so the rounds are walked separately
    instead of summed. A single-round run keeps exactly the previous joins.
    """

    errors: list[str] = []
    completed = result.get("state") == "completed"
    native = (run_root / "native").resolve()
    runtime = plan.get("runtime", {})
    session = plan.get("session")
    multi_round = isinstance(session, Mapping)
    rounds = session.get("rounds") if multi_round else 1
    resets = [event.get("record") for event in events if event.get("type") == "reset"]
    if not resets and not completed:
        return errors  # A worker that died before reset has no task verdict.
    if (
        type(rounds) is not int or rounds < 1
        or len(resets) != rounds
        or not all(isinstance(record, Mapping) for record in resets)
        or (multi_round and resets[0].get("round_index") != 0)
    ):
        return ["robodojo_reset_evidence_invalid"]
    reset = resets[0]
    identities = layouts.round_identities(plan, rounds)
    start_schema, round_reset_schema, verdict_schema = layouts.record_schemas(plan)
    isolation_names = layouts.isolation_check_names(plan, ROBODOJO_ROUND_ISOLATION_CHECKS)
    reference = reset.get("source_manifest")
    source_path = native / "source-manifest.json"
    try:
        if source_path.is_symlink() or not source_path.is_file():
            raise EvaluationError("source manifest is missing or unsafe")
        source = _load(source_path)
        source_hash = _hash(source_path)
    except (EvaluationError, OSError):
        return ["robodojo_source_manifest_invalid"]
    if (
        reset.get("schema") != start_schema
        or reset.get("reward_registered") is not True
        or not isinstance(reference, Mapping)
        or reference.get("path") != "source-manifest.json"
        or reference.get("sha256") != source_hash
        or source.get("schema") != "ora.robodojo-source-manifest.v1"
    ):
        errors.append("robodojo_source_manifest_binding_invalid")
    expected_identity = {
        "task": plan.get("task"),
        "eval_seed": runtime.get("eval_seed", plan.get("seed")),
        "layout_id": runtime.get("layout_id"),
    }
    if any(
        expected is None or reset.get(key) != expected or source.get(key) != expected
        for key, expected in expected_identity.items()
    ) or any(reset.get(key) != expected for key, expected in identities[0].items()) or (
        layouts.scheduled_reset_invalid(plan, reset, 0)
    ):
        errors.append("robodojo_layout_identity_invalid")
    errors.extend(layouts.source_schedule_errors(plan, source))
    # The resolved IsaacLab device must be the declared one (CPU-physics lane).
    from .robodojo_physics_attestation import physics_attestation_errors
    errors.extend(physics_attestation_errors(runtime, source))
    revision_fields = {
        "robodojo": "required_revision",
        "isaaclab": "required_isaaclab_revision",
        "curobo": "required_curobo_revision",
        "xpolicylab": "required_xpolicylab_revision",
        "assets": "required_asset_revision",
    }
    expected_revisions = {key: runtime.get(field) for key, field in revision_fields.items()}
    if (
        any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
            for value in expected_revisions.values())
        or source.get("source_revisions") != expected_revisions
        or source.get("required_revisions") != expected_revisions
    ):
        errors.append("robodojo_source_revision_binding_invalid")
    layout = source.get("layout_file")
    expected_layout = runtime.get("layout_file")
    configuration_fields = {
        "sensor_profile": "sensor_profile", "physics_dt_s": "physics_dt",
        "control_hz": "control_hz", "official_native_action_limit": "native_action_limit",
    }
    if (
        not isinstance(layout, Mapping)
        or not isinstance(layout.get("path"), str) or not layout.get("path")
        or not isinstance(layout.get("sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", layout["sha256"]) is None
        or layout.get("sha256") != runtime.get("layout_sha256")
        or not isinstance(expected_layout, str) or not expected_layout
        or Path(expected_layout).is_absolute() or ".." in Path(expected_layout).parts
        or not Path(layout["path"]).as_posix().endswith("/" + expected_layout)
        or type(source.get("num_envs")) is not int or source.get("num_envs") != 1
        or source.get("action_codec") != plan.get("policy", {}).get("action_codec")
        or any(
            runtime.get(field) is None or source.get(key) != runtime.get(field)
            for key, field in configuration_fields.items()
        )
    ):
        errors.append("robodojo_source_configuration_invalid")
    reset_clocks: list[int] = []
    for record in resets:
        clock = record.get("physics_steps_before_policy")
        if type(clock) is not int or clock < 0:
            errors.append("robodojo_reset_clock_invalid")
            clock = 0
        reset_clocks.append(clock)
    sequence_names = (
        "observation_sequence", "action_sequence", "motion_probe_sequence",
    )
    previous_sequences: Mapping[str, Any] | None = None
    for index, record in enumerate(resets[1:], start=1):
        archived = native / "round-resets" / f"round-{index:04d}.json"
        try:
            if archived.is_symlink() or not archived.is_file():
                raise EvaluationError("round reset record is missing or unsafe")
            persisted: Any = _load(archived)
        except (EvaluationError, OSError):
            persisted = None
        sequences = record.get("artifact_sequences_preserved")
        round_reference = record.get("source_manifest")
        # The backend's own measured restore proof. RoboDojo owes one for
        # every round, so an absent, failing or layout-drifting block is a
        # rejected round rather than a silently unproven one: this is the
        # join that refuses a round whose reset record does not prove the
        # world came back and no channel kept a previous round's value.
        isolation = record.get("round_isolation")
        isolation_checks = (
            isolation.get("checks") if isinstance(isolation, Mapping) else None
        )
        isolation_measured = (
            isolation.get("measured") if isinstance(isolation, Mapping) else None
        )
        if (
            not isinstance(isolation, Mapping)
            or isolation.get("schema") != ROBODOJO_ROUND_ISOLATION_SCHEMA
            or isolation.get("passed") is not True
            or isolation.get("failures")
            or not isinstance(isolation_checks, Mapping)
            or not set(isolation_names).issubset(isolation_checks)
            or any(
                isolation_checks.get(name) is not True
                for name in isolation_names
            )
            or not isinstance(isolation_measured, Mapping)
            or isolation_measured.get("stale_channels")
            or isolation_measured.get("reward_registered") is not True
            or layouts.isolation_layout_errors(plan, isolation_measured, index)
            or isolation_measured.get("native_action_limit") != runtime.get(
                "native_action_limit"
            )
            or isolation_measured.get("step_lim") != runtime.get(
                "native_action_limit"
            )
            or isolation_measured.get("physics_steps_before_policy") != record.get(
                "physics_steps_before_policy"
            )
        ):
            errors.append(f"robodojo_round_isolation_invalid:{index}")
        if (
            persisted != dict(record)
            or record.get("schema") != round_reset_schema
            or layouts.scheduled_reset_invalid(plan, record, index, resets[index - 1])
            or record.get("round_index") != index
            or record.get("reward_registered") is not True
            or not isinstance(round_reference, Mapping)
            or round_reference.get("path") != "source-manifest.json"
            or round_reference.get("sha256") != source_hash
            or any(
                record.get(key) != expected
                for key, expected in identities[index].items()
            )
            or not isinstance(sequences, Mapping)
            or any(
                type(sequences.get(name)) is not int or sequences[name] < 0
                or (
                    previous_sequences is not None
                    and sequences[name] < previous_sequences[name]
                )
                for name in sequence_names
            )
        ):
            errors.append(f"robodojo_round_reset_invalid:{index}")
        if isinstance(sequences, Mapping) and all(
            type(sequences.get(name)) is int for name in sequence_names
        ):
            previous_sequences = sequences
    native_total = physics_total = receipt_index = seen_resets = 0
    reset_clock = reset_clocks[0]
    current_round = 0
    round_native_totals: list[int] = []
    round_physics_totals: list[int] = []
    submitted_actions = [
        event.get("action", {})
        for event in events if event.get("type") == "action_submitted"
        and isinstance(event.get("action"), Mapping)
    ]
    for event in events:
        kind = event.get("type")
        if kind == "reset":
            # A round reset restarts the wrapper's counters and rebinds the
            # PhysX clock, so every later join belongs to the new round.
            seen_resets += 1
            if seen_resets > 1:
                round_native_totals.append(native_total)
                round_physics_totals.append(physics_total)
                current_round = seen_resets - 1
                reset_clock = reset_clocks[current_round]
                native_total = physics_total = 0
            continue
        if kind != "motion_receipt":
            continue
        record = event.get("record")
        try:
            if not isinstance(record, Mapping):
                raise EvaluationError("receipt record is missing")
            if receipt_index >= len(submitted_actions):
                raise EvaluationError("receipt has no submitted action")
            worker_receipts = _robodojo_worker_receipts(
                native, submitted_actions[receipt_index], record,
            )
            for path, receipt in worker_receipts:
                if (
                    receipt.get("round_index") is not None
                    and receipt.get("round_index") != current_round
                ):
                    raise EvaluationError("receipt round differs from the causal trace")
                native_count, physics_count = receipt.get("native_actions"), receipt.get("physics_steps")
                if (
                    type(native_count) is not int
                    or not 0 <= native_count <= 20
                    or type(physics_count) is not int or physics_count != native_count * 10
                ):
                    raise EvaluationError("receipt native and physics counters are invalid")
                control = receipt.get("control_trace")
                if (
                    receipt.get("schema") not in {
                        "ora.robodojo-motion-receipt.v1",
                        "ora.robodojo-sparse-motion-receipt.v1",
                    }
                    or not isinstance(control, Mapping)
                    or not isinstance(control.get("path"), str) or not control.get("path")
                ):
                    raise EvaluationError("controller trace reference is missing")
                control_relative = Path(control["path"])
                control_path = native / control_relative
                if (
                    control_relative.is_absolute() or ".." in control_relative.parts
                    or control_path.is_symlink() or not control_path.resolve().is_relative_to(native)
                    or _hash(control_path) != control.get("sha256")
                ):
                    raise EvaluationError("controller trace is unsafe or differs from its receipt")
                control_record = _load(control_path)
                if (
                    control_record.get("schema") != "ora.robodojo-controller-trace.v1"
                    or not isinstance(receipt.get("action_id"), str)
                    or control_record.get("action_id") != receipt["action_id"]
                ):
                    raise EvaluationError("controller trace action identity differs")
                samples = control_record.get("samples")
                if not isinstance(samples, list) or len(samples) != native_count:
                    raise EvaluationError("controller trace length differs from its native action count")
                for index, sample in enumerate(samples):
                    if (
                        not isinstance(sample, Mapping)
                        or type(sample.get("native_action")) is not int
                        or sample["native_action"] != native_total + index + 1
                        or type(sample.get("physics_steps")) is not int
                        or sample["physics_steps"] != physics_total + (index + 1) * 10
                    ):
                        raise EvaluationError("controller trace counters are not contiguous")
                journal = control_record.get("physics_journal")
                if not isinstance(journal, Mapping) or not isinstance(journal.get("path"), str) or not journal.get("path"):
                    raise EvaluationError("controller physics journal reference is missing")
                journal_relative = Path(journal["path"])
                journal_path = native / journal_relative
                if (
                    journal_relative.is_absolute() or ".." in journal_relative.parts
                    or journal_path.is_symlink() or not journal_path.resolve().is_relative_to(native)
                    or _hash(journal_path) != journal.get("sha256")
                ):
                    raise EvaluationError("controller physics journal is unsafe or its hash differs")
                journal_rows = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
                if len(journal_rows) != physics_count:
                    raise EvaluationError("physics journal length differs from its receipt")
                for index, row in enumerate(journal_rows):
                    if (
                        not isinstance(row, Mapping) or row.get("action_id") != receipt["action_id"]
                        or type(row.get("physics_clock")) is not int
                        or row["physics_clock"] != reset_clock + physics_total + index + 1
                        or type(row.get("native_action")) is not int
                        or row["native_action"] != native_total + index // 10 + 1
                    ):
                        raise EvaluationError("physics journal clock or action identity differs")
                native_total += native_count
                physics_total += physics_count
                if _receipt_execution_fault(path):
                    errors.append("robodojo_backend_execution_fault")
        except (EvaluationError, OSError, ValueError, UnicodeError, TypeError, KeyError):
            errors.append("robodojo_motion_receipt_binding_invalid")
        finally:
            receipt_index += 1
    round_native_totals.append(native_total)
    round_physics_totals.append(physics_total)
    if multi_round:
        for index in range(1, rounds):
            # The PhysX clock never rewinds within one simulation context, so
            # a later in-place reset must start at or after the last physics
            # step the previous round journaled. A scheduled round relaunches
            # the simulation, whose new context starts its clock again.
            if layouts.clock_continuity_applies(plan, resets[index - 1], resets[index]) and reset_clocks[index] < (
                reset_clocks[index - 1] + round_physics_totals[index - 1]
            ):
                errors.append(f"robodojo_round_reset_invalid:{index}")
        errors.extend(_robodojo_round_verdict_errors(
            native, result, runtime, expected_identity, source_hash,
            rounds=rounds, completed=completed, reset_clocks=reset_clocks,
            native_totals=round_native_totals,
            physics_totals=round_physics_totals,
            identities=identities, verdict_schema=verdict_schema,
        ))
        errors.extend(layouts.round_scene_errors(plan, result, resets))
        errors.extend(layouts.render_time_file_errors(plan, native, resets))
        return errors
    verdict_path = native / "official-verdict.json"
    if not completed and result.get("verdict") is None and not verdict_path.exists():
        return errors
    try:
        if verdict_path.is_symlink():
            raise EvaluationError("official verdict is unsafe")
        verdict = _load(verdict_path)
    except (EvaluationError, OSError):
        return [*errors, "robodojo_official_verdict_missing"]
    if (
        verdict.get("schema") != ROBODOJO_VERDICT_SCHEMA
        or verdict.get("source") != "RoboDojo EvalEnv.is_episode_end"
        or verdict.get("source_manifest_sha256") != source_hash
        or result.get("verdict") != verdict
        or any(verdict.get(key) != expected for key, expected in expected_identity.items())
        or result.get("official_success") != verdict.get("official_success")
    ):
        errors.append("robodojo_official_verdict_binding_invalid")
    if completed and (
        type(verdict.get("native_actions")) is not int
        or verdict["native_actions"] != native_total
        or native_total > runtime.get("native_action_limit", 200)
        or type(verdict.get("physics_steps")) is not int
        or verdict["physics_steps"] != physics_total
        or verdict.get("reset_physics_steps") != reset_clock
    ):
        errors.append("robodojo_official_counter_mismatch")
    # One rule for both cadences: the same budget-stopped episode cannot be
    # valid as round 0 of a K3 session and invalid as a single-episode run.
    if completed and _robodojo_verdict_untrusted(verdict, result):
        errors.append("missing_trusted_boolean_verdict")
    return errors
