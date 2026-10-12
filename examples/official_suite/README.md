# Official suite: LIBERO-Pro core and RoboTwin at official seed scale

Namespace `official-suite`. This example composes the two benchmarks' official
evaluations on the default baseline: a RunSpec names no tool surface, prompt
bundle or Context, so the default resolver composes the default catalog's
LIBERO-Pro and RoboTwin instances. It selects only the cell, the Provider
profile, budgets, the resident-service binding and the qualification evidence.

## What a cell is

One episode on one seed or initial state, single-attempt admission, no
persistent session:

- **LIBERO-Pro:** `benchmarks/libero_pro.official_core_evaluation()` -- the
  sixteen core suites (Obj/Pos/Sem/Task x four families) x ten tasks x initial
  states 0-49, 8,000 cells at the pinned code and data revisions. The
  instruction is the perturbation's BDDL language from the pinned task catalog.
- **RoboTwin:** per task and scene configuration, every expert-accepted seed a
  sealed seed manifest lists (`robotwin_evaluation prepare --accepted-per-task
  100 --task-config demo_clean|demo_randomized`, then `qualify`). The
  instruction is the public per-task instruction in
  `ora/environment/benchmarks/robotwin-instructions-v1.json`;
  `place_a2b_left` and `place_a2b_right` name the
  objects the seed spawns and take theirs from `--episode-names FILE`
  (`{"episodes": {"<task>:s<seed>": {"instruction": ...}}}`), without which
  those cells are refused.

Budgets are 80 decisions / 200 model calls / 10,000 native steps / 6 h per
episode. The LIBERO arm uses the default
experiment Provider (Low); the RoboTwin arm uses High. `--provider-profile` overrides either, for example with a profile
for a self-hosted model.

`build_spec` writes `runtime.task` for a RoboTwin cell, which binds the
reviewed completion contract; without it the plan records
`task_contract.delivered: false`. Single cells and other cell lists compose
through the maintained `ora/orchestrator/benchmark_spec.py`
([RunSpec inputs](../../docs/reference/configuration-and-evidence.md#new-robotwin-libero-pro-and-libero-runspecs)).

## Commands

```bash
uv run --locked python examples/official_suite/run.py cells --arm libero_pro
uv run --locked python examples/official_suite/run.py cells --arm robotwin \
  --task-config demo_clean --seed-manifests <qualify-output>/seed-manifests
uv run --locked python examples/official_suite/run.py services qualify
uv run --locked python examples/official_suite/run.py compose --arm libero_pro \
  --config lp-libero_object_swap-t00-s0 --lane <resource-profile> \
  --service-report <report.json>
uv run --locked python examples/official_suite/run.py queue --arm robotwin \
  --task-config demo_randomized --seed-manifests <dir> --accepted-limit 10 \
  --lanes <profile>... --group-id pilot-1 --service-report <report.json>
```

`cells` lists a selection; `compose` writes one RunSpec; `queue` launches a
selection across lanes, one attempt per cell, and stops at the first refusal.
`--resume` skips cells earlier groups of this cohort already sealed. Every
launch still passes the shared preflight: host marker, services, Provider
probe and the seed manifest (RoboTwin).

## Evidence boundary

Runs land under `outputs/official-suite/`. A result is reported per
benchmark row with the differences from the leaderboards named (no
demonstrations, RGB-D plus tools, pose targets, Harness budgets).
No result exists yet.
