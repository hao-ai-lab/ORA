# Task contracts and execution feedback

Task text, private predicate review, public completion clauses and action feedback have different owners. Keep them separate. Use the [task review template](../templates/task-contract-review.md). Current formats are described in [configuration and evidence](../reference/configuration-and-evidence.md).

## Review a task

1. Bind backend/task, source revision, original instruction, predicate and helpers, reset/randomization and relevant asset/frame hashes.
2. Decompose every AND/OR, quantifier, object role, ordering, orientation/spatial relation, holding/release, contact/joint and timing requirement. Read called helpers too.
3. Establish physical meaning from asset geometry and coordinate transforms. A quaternion alone does not identify the printed face; a connected mask does not establish a hinge.
4. Separate initial state from final requirements. A random starting orientation plus a fixed final orientation does not require a fixed half-turn. An expert waypoint is not automatically a task requirement.
5. Identify when official success is checked or latched. Qualification hold, arrival, release, stable support and official success are distinct facts.
6. Map every public clause to its source or original task intent. Explain each withheld condition, unobservable requirement or evaluator gap; do not silently omit it.
7. Compare saved successful states from the exact source/assets/configuration with the proposed wording. Explain any conflict or missing evidence.
8. Review relevant near misses: wrong orientation, still-held objects, partial object sets or actions never executed. Reuse existing records; this review does not require launching new simulation.
9. Verify rendered text, resolved RunPlan and actual first/per-round provider delivery separately. Ensure private review JSON never enters the prompt.

A hash match or passing static test does not establish semantic correctness. Missing or contradictory evidence remains `needs_review`.

## Public projection

The renderer uses reviewed `public_clauses`; it must not concatenate the entire private review. Choose only conditions the task requires:

```text
Public completion contract for this task:

- <Which publicly identifiable objects must satisfy which observable relation.>
- <If needed: which visible face points toward which explicit reference.>
- <If needed: holding/release or another state, at the required time.>
```

Do not disclose hidden coordinates, thresholds, GT, private object IDs, predicate code or expert trajectories. The conservative renderer checks vocabulary/numbers, but lexical acceptance cannot prove absence of semantic leakage. If original task intent is stronger than its evaluator, retain the intent and record the scoring gap rather than inventing an official requirement.

Sensor-view improvements must be task-independent and respect the selected benchmark. Changing scene lighting changes the benchmark. Do not add task-specific object locations, view crops or search tactics to public task text. A model failing to recognize an object is a behavioral outcome; wrong or withheld interface information is a harness defect.

## Feedback projection

Generate feedback from approved fields/enums, not raw exception text or recursive private details. Each report joins the exact action, arm, round and before/after observations.

| Item | Required distinction |
|---|---|
| Stage | Input validation, planning, trajectory admission, execution and endpoint check are separate. |
| Facts | Submitted versus executed, state changed versus unchanged, arrived versus not arrived. |
| Reason | Stable public code; preserve candidate/arm-specific categories and counts. |
| Recovery | An available alternative direction/intermediate target/replan path, without promising success or weakening gates. |
| Uncertainty | Planner failure does not prove universal unreachability; predicted collision is not measured contact. |
| Bounds | Limit text and disclose omissions; remove private paths, object identities, hidden poses and stacks. |

`ora/agent/evidence/plan_feedback.py` bounds displayed candidate outcomes to eight while retaining counts for reported candidates. Unknown categories become `unknown_rejection`. For example, `planner_failed` means no admitted trajectory was produced; `workspace_path` means a generated path exceeded the public workspace before execution, not necessarily that its endpoint was outside.

## Audit commands

```bash
./scripts/dev/audit-task-contracts template --backend robotwin --task rotate_qrcode
./scripts/dev/audit-task-contracts audit
./scripts/dev/audit-task-contracts render --record PATH_TO_REVIEW_RECORD
./scripts/dev/audit-task-contracts --help
```

Templates start as `needs_review`; these commands do not launch a provider or simulator. Rendering is refused until its review requirements pass. Validate source bindings, semantic review, actual Native delivery and physical validity separately. A change to the source/assets/contract must state which evidence is invalidated; preserve sealed originals.
