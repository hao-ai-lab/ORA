# Qualification authoring and evidence reuse

Separate development qualification from fresh launch preflight. A launch should not require hours of full suites, repeated simulator initialization or Cartesian sweeps without a concrete dependency failure. These are authoring requirements; each operator's current implementation and declared coverage still determine what it proves.

## Purpose and coverage

| Purpose | Coverage | Launch prerequisite |
|---|---|---|
| `launch_smoke` | A few representative mechanisms, one execution each for deterministic cases. | Reuse valid evidence rather than rerunning every launch. |
| `targeted_regression` | Cases affected by a specific failure or dependency change. | Only the missing/invalidated evidence. |
| `development_diagnostic` | A stated capability question, wider matrix or repeatability study. | Not automatically required for ordinary launches. |

Ordinary coverage uses `representative-v1` and one repeat. A full development matrix or repeats require a separately justified diagnostic. Repeats need prior nondeterminism evidence, an estimate of interest, a fixed sample count and a stop rule. Different poses or directions are separate cases, not repeated measurements. Prefer offline fixtures for deterministic contracts and error branches; use simulation for physical questions fixtures cannot prove.

## Bind actual dependencies

A producer declares the inputs it executes: implementation and called contracts, consumed controller/profile content, simulator/numerical dependencies, selected assets/device conditions, task/reset/seed, initial state, parameters, windows, arms and budgets. Explain why each affects the case. Do not substitute the whole checkout or current HEAD for this dependency closure.

Record a reproducible `producer_dependency_hash`, `case_key` and separate `checker_hash`. Keep producer, checker and launch source commits as distinct provenance. A documentation change, profile label or scheduler change alone does not invalidate physical evidence. A changed control cadence or physics implementation does. Cache reliable content inventories within their real invalidation boundary.

When old evidence lacks bindings, reconstruct them from its saved source/configuration/original files where possible. Mark only the unconfirmed dependencies/cases as missing. Do not infer equivalence from a filename or version label.

## Separate producer and checker

The producer writes original commands, measurements and observations. The checker reads them without implicitly initializing a simulator or replaying actions. If a checker or tolerance changes, revalidate existing originals first. Run additional physical cases only for measurements the originals do not contain.

Write new reports with original case/file hashes, producer identity, current checker identity and reasons. Preserve old failures, interruptions and originals. Record passed, failed and incomplete per case. A failure blocks its dependent capability; independent healthy lanes can continue. A shared controller/environment failure stops the affected set.

Plans and attempts retain immutable identities. A new plan may reference valid cases from several old plans and execute only gaps. Independent reset/state/resource/evidence identities are required; one worktree per case is not.

## Prepare a reviewable schedule

Before simulation, print and save purpose, coverage policy, dependency closure and hashes, case IDs/keys/parameters/repeats, reused evidence and invalidation reasons, resource inventory, explicit lane bindings, shared setup and task assignments.

Estimate wall time from the actual parallel schedule: shared setup + slowest remaining lane + checking/finalization + known waiting. Include a finite `estimated_wall_minutes`, its historical/static basis and uncertainty. Do not assume every GPU is free or put zero for unknown cost. Serial execution needs a dependency or measured resource reason.

A plan over 15 minutes is a `development_diagnostic`; splitting the same plan into smaller chunks does not change its total cost. This label does not authorize more coverage or repeats. First consider reuse, representative cases and better scheduling.

## Minimum record

```yaml
purpose: targeted_regression
coverage_policy: representative-v1
producer_source_commit: <original execution source>
dependency_closure: <actual implementation, environment and case inputs>
producer_dependency_hash: <content hash>
checker_hash: <independent checker identity>
cases:
  - case_id: <mechanism>
    case_key: <dependencies plus case parameters>
    repeats: 1
reuse: <original references or concrete invalidation reasons>
parallel_execution: <available slots, bindings and distribution>
estimated_wall_minutes: <measured or conservatively estimated value>
estimate_basis: <setup, case costs, reuse, parallelism and uncertainty>
```

Check that fresh launch credentials, services, resources and admission are current. Physical qualification reuse does not waive those short checks. Preserve exact source/seed/evidence identities and report only the coverage actually obtained.
