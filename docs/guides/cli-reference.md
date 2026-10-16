# ORA CLI reference

`ora` is the generic operator surface for an already composed
RunSpec and an indexed run. From a source checkout, use the repository wrapper:

```bash
./scripts/ora --help
```

The wrapper runs `.venv/bin/python` from the checkout when it exists, else
`$PYTHON`, else `python3`. If your environment lives elsewhere (for example via
`UV_PROJECT_ENVIRONMENT`), use `uv run --locked ora ...` or set `PYTHON` to that
environment's interpreter. An installed wheel exposes the same command as `ora`. Experiment
CLIs remain the normal place to select a config, task and seed; they delegate
the resulting RunSpec and run operations to this generic surface. The internal
`_supervise` child-process entry is deliberately excluded from the public
command set and must never be invoked by an operator.

## Public commands

The table order is canonical and is checked against the parser in pytest.
“Writes” describes local Runtime evidence or process effects; it does not mean
that a command sends a robot action unless explicitly stated.

| Command | Syntax | Writes or external effect | Purpose |
|---|---|---|---|
| `run` | `run --env synthetic`; `run --env libero\|robotwin\|robodojo --task TASK [--seed N] [--task-config demo_randomized\|demo_clean] [--candidates N] [--yes] [--requalify] [--prepare-only] [--confirm-launch] [--provider-profile ID]` | Synthetic: calls the configured model and writes a sealed episode. GPU: may start the perception services, run a confirmed one-time qualification, writes a RunSpec under `outputs/ora-interactive/specs/`; with `--confirm-launch` and a picker selection launches one run | `synthetic` runs the bounded, simulator-free Native quickstart. A GPU environment prepares one task on the shipped `single-gpu` host and opens `ora interactive` on it ([guide](../getting-started/single-gpu.md)). |
| `doctor` | `doctor [--env synthetic\|robotwin\|libero_pro\|robodojo\|local] [--root PATH] [--json]` | None | Discover Python, ORA, credential presence, simulator interpreters, asset directories and graphics libraries without starting a simulator. For `robodojo` it also loads `libvulkan.so.1` in the simulator interpreter (honouring `LD_LIBRARY_PATH`) and, on drivers that need it (595.x), verifies the Vulkan compatibility-layer receipt. |
| `local` | `local [--suite S] [--task-id N] [--seed N] [--root PATH] [--confirm-launch] [--world-view auto\|off]` | With `--confirm-launch` and a picker selection, launches one run under `ora-interactive` | Open the local CPU MuJoCo LIBERO task picker ([guide](../getting-started/local.md)); uses the bundled `local-cpu` host profile and the DeepSeek Low provider. |
| `setup` | `setup [--root PATH] [--python PATH] [--graphics auto\|cgl\|glfw\|egl\|osmesa] [--json]`; `setup --env libero\|robotwin\|robodojo\|perception [--env …] [--prefix DIR] [--cuda-home DIR] [--cuda-arch X.Y] [--dry-run] [--json]` | Without `--env`: pinned LIBERO sources and data, the CPU simulator environment, `npm ci` in `ora-tui/`. With `--env`: the single-GPU stacks under the prefix and their non-credential `.env` keys | Install what `ora local` or `ora run --env` needs; starts no simulator, service or model call. |
| `interactive` | `interactive --spec PATH [--spec PATH ...] [--confirm-launch] [--emit jsonl]` | With `--confirm-launch` and a picker selection, launches the selected task under `ora-interactive`; writes operator messages to the run's inbox | Pick a prepared Native benchmark task and steer it while it runs ([guide](../usage/interactive.md)). |
| `validate` | `validate RUN_SPEC` | None | Parse, validate and content-bind one RunSpec; print its plan identity. |
| `plan` | `plan RUN_SPEC` | None | Print the resolved content-addressed RunPlan, including policy, Runtime, budgets, hashes and the launch confirmation code (the last six hex digits of the plan sha256). |
| `context` | `context RUN_SPEC` | Temporary files only | Preview the exact project-owned context, composition and tools without starting an episode. Native provider requests are inspectable. |
| `preflight` | `preflight RUN_SPEC` | Writes a content-addressed preflight record | Run source, host, environment and provider admission gates without launching an episode. A live provider check may be billable when required by the RunSpec. |
| `launch` | `launch RUN_SPEC [--yes]` | Creates a run and starts its supervisor | Repeat preflight, require the plan confirmation code, atomically acquire the RunPlan's execution-resource lease, enforce optional plan-bound launch admission and launch one episode. A `single_attempt_cell` claim is consumed by this generic path, including when called outside an experiment wrapper. Unprofiled and physical Runtime plans retain host-wide exclusivity; explicitly qualified simulator slots may coexist. `--yes` is only for an already reviewed higher-level workflow. |
| `retry` | `retry RUN_ID [--yes]` | Creates a distinct retry run | Retry only an unsealed, incomplete attempt that published zero robot decisions and whose RunPlan permits retry. A `single_attempt_cell` plan rejects retry. It never overwrites or resumes the old run directory. |
| `status` | `status [RUN_ID]` | None | Print one status document or summarize every active resource lease and the complete indexed local run registry. |
| `workspace` | `workspace` | None | Show the Git-common evidence root, state root, run index and legacy discovery roots used across worktrees. |
| `providers` | `providers [--json]` | None | List reusable credential-free provider registrations, admission status, usage scope and declared capabilities. |
| `baselines` | `baselines [--json]`, `baselines --explain [--environment KIND] [--json]` | None | Inspect the Native robot default catalog, its source hash, per-environment tools, prompts and prerequisites; never probe or start services. `--explain` derives from code why the default is selected, its per-environment resolution chain, the maintained baselines, and each tool's effect, handler and proposal route to the backend. |
| `experiments` | `experiments [FAMILY] [--json]` | None | Inspect stable experiment families, generation parents and historical aliases; never launch or reset admission. |
| `lock` | `lock [--check]` | Writes `policy.lock` (without `--check`) | Regenerate `policy.lock`, the sorted `sha256sum`-format inventory of every tracked policy asset and of the path-loaded assets outside `ora/policy/`. `--check` writes nothing, lists assets that are missing from, extra in or changed against the lock, and exits 1 on any drift; CI runs it before the tests. Run it after editing, adding or deleting an asset (a new asset counts once it is `git add`ed) and commit the result. |
| `revalidate` | `revalidate [RUN_DIR ...] [--run-list LIST.json] --output-dir OUT [--workers N]` | Writes separate reports only | Review exact completed run directories with checker/source hashes, original/new evaluations and differences; preserve all original files and defer active prefixes. |
| `watch` | `watch RUN_ID\|RUN_DIR [--speed S] [--ui auto\|pi\|ansi\|plain] [--raw]` | None | Follow a live run, or replay a finished one, as ORA's terminal transcript: the model's visible words, condensed tool results, refusals, receipts and per-round verdicts. A finished run replays on recorded time (`--speed 0` prints at once). `--ui pi` draws with pi-tui (needs Node 20+ and `npm ci` in `ora-tui/`); `auto` uses it in a terminal when installed, else the built-in ANSI view. `--raw` prints the canonical causal events as JSON lines instead. Read-only: it writes nothing under the run and takes no lease; `Ctrl-C` stops watching, not the run. |
| `stop` | `stop RUN_ID` | Sends `SIGTERM` to one verified process group | Request graceful termination only when the exact run ID, its resource lease and process command agree; sibling slots are unaffected. |
| `metadata` | `metadata RUN_ID` | None | Print the run's exact context/environment manifest. |
| `state` | `state RUN_ID [--step N]` | None | Verify the policy-state hash chain and print the latest or selected public state. |
| `metrics` | `metrics RUN_ID` | Rebuilds context audit and Runtime metrics | Deterministically derive context, token, cache, image and latency measurements from raw evidence. Run before evaluation and sealing. |
| `ownership` | `ownership RUN_ID` | Rebuilds the behavioral-ownership audit (`ownership-audit.json`) | Deterministically count Language-Model decisions, explicit robot actions, hidden planner/controller work and re-observation boundaries from append-only evidence. Selection and delegation only — no competence or success claim. |
| `evaluate` | `evaluate RUN_ID` | Writes deterministic evaluation evidence | Validate the completed trace and bind the official environment result without asking the policy to judge success. |
| `seal` | `seal RUN_ID [--backfill]` | Writes the seal and makes the inventory immutable | Seal only an evaluator-valid run; sealed evidence cannot be retried or rewritten. Landing attempts this automatically (mode `landing`, refusals recorded as `seal-refusal.json`); `--backfill` seals an old unsealed run with the current evaluator while pinning its recorded source identity and marking the seal as a backfill. |
| `export` | `export RUN_ID [--destination PATH]` | Writes a transfer archive | Export one sealed run with its verified inventory. The default destination is the experiment workspace's `exports/` directory. |
| `trace-validate` | `trace-validate RUN_ID` | None | Validate the canonical Agent trace, request hashes and artifact graph; exit 2 when invalid. |
| `export-sft` | `export-sft RUN_ID [--destination PATH]` | Writes a training projection | Export eligible Native model/tool calls as OpenAI tool-call JSONL from canonical request evidence. This projection is not the canonical trace. |

## Normal lifecycle

Use the owning experiment CLI to compose the RunSpec, then follow the generic
control lifecycle:

```bash
./scripts/ora validate path/to/run-spec.yaml
./scripts/ora plan path/to/run-spec.yaml
./scripts/ora context path/to/run-spec.yaml
./scripts/ora preflight path/to/run-spec.yaml
./scripts/ora launch path/to/run-spec.yaml

./scripts/ora status RUN_ID
./scripts/ora watch RUN_ID
./scripts/ora metrics RUN_ID
./scripts/ora evaluate RUN_ID
./scripts/ora seal RUN_ID
./scripts/ora export RUN_ID
```

`launch` is the only command in that sequence that starts an episode. A
preflight can still contact the configured provider and therefore may require
explicit billable confirmation in the owning experiment CLI. Use `stop` only
for the exact active run; do not replace it with a broad process kill.

For Native live-marker policies, `preflight`, `launch` and `retry` can probe
the configured standby credential pool. HTTP 401, 402, 403 or 429 advances to
the next distinct key, with at most one complete qualification attempt per
candidate. Other errors stop admission. A successful launch pins the qualified
credential in the child environment; reports contain indices, never secrets.
`launch` and `retry` require their own fresh qualification, not a previously
saved report. See [configuration and evidence](../reference/configuration-and-evidence.md).

## What “tested” means

The locked pytest suite checks every public command name, their parser order,
their repository-wrapper `--help` entry, their dispatch branch and this table.
It also tests the underlying deterministic RunSpec, context, preflight,
workspace/index, policy-state, metrics, evaluation, sealing, archive and SFT
contracts with fake or Synthetic boundaries.

Pytest does **not** launch a paid model or a live LIBERO/RoboTwin simulator.
Consequently, `launch`, `retry`, `watch` and `stop` have deterministic dispatcher,
parser and help coverage. Supervisor leases, run indexing and fake-process
boundaries are tested separately, but the exact real host/process/environment
path is qualified only by a source-bound experiment preflight or live
qualification. Passing pytest means the CLI wiring and fail-closed contracts
are intact; it is not evidence that a particular provider, host or robot task
is currently qualified.

## RoboTwin evaluation preparation

The maintained source-checkout entry is
`uv run --locked python -m ora.orchestrator.robotwin_evaluation`.
It is separate from the generic episode commands above:

- `configs [--outer-seed N] [--task-config demo_clean|demo_randomized]`
  prints the pinned 50-task evaluation protocol for one scene configuration
  without writing evidence, querying GPUs or starting a simulator.
- `prepare --robotwin-root PATH --robotwin-python PATH --tasks TASK...
  --slots SLOT... --candidate-count N [--accepted-per-task K]
  [--task-config demo_clean|demo_randomized] [--outer-seed N]
  [--start-offsets K... | --segments TASK:OFFSET[:COUNT]...]
  [--process-attempt-limit N]
  [--purpose targeted_regression|development_diagnostic]` checks the exact
  clean upstream revision (the vendored install the Runtime loads), the
  selected scene configuration's official settings, task roster, unseen
  texture pool and current capacity. It hashes declared dependencies and
  writes an immutable plan under the shared
  `outputs/runtime-v1/robotwin-evaluation/` workspace (under
  `ORA_OUTPUTS_ROOT` when set, not the node-local workspace). Tasks, slots and
  candidate count have no experiment defaults. It prepares 50 configurations
  but schedules only the explicitly named cases: per task, the first K seeds
  from the official start seed that the expert accepts (K defaults to 1; the
  official protocol is 100), trying at most N candidates. Accepted seeds are
  bound to the scene configuration and revision, so `demo_clean` and
  `demo_randomized` are separate scans. `--start-offsets K...` splits long
  tasks into segments: every (task, offset) is its own case and lane, scanning
  N seeds from the official start plus K (set `--accepted-per-task` equal to
  the candidate count to scan the whole segment). Case ids are `task` for
  offset 0 and `task@<first seed>` otherwise; they name the case's output and
  manifest. `--segments TASK:OFFSET[:COUNT]...` lists the cases explicitly
  instead, for tasks that need different numbers of segments; COUNT (at most
  N) sets one segment's length, so the unscanned remainder of stopped segments
  can be tiled exactly. `--process-attempt-limit` (default 8) is how many
  seeds one oracle process scans before re-executing itself; plans written
  before it existed use 1. One seed per process re-creates the CUDA context,
  renderer and planner warm-up every seed, which serialized on the NVIDIA
  driver at ~100 workers per node. The interpreter path is kept as given, not
  resolved (a venv's `bin/python` is a symlink), and a GPU whose `nvidia-smi`
  row reads `[N/A]` is left out of capacity.
- `qualify --plan PATH --confirm QUALIFY-ROBOTWIN:HASH` checks the saved plan,
  exact clean Harness/upstream sources, dependency hashes and the host marker
  of the plan's `required_host` (set by `prepare --required-host`, default the
  current marker; a plan without one accepts any registered marker),
  rechecks GPU capacity, acquires resource leases and runs isolated oracle lanes
  in parallel. Lanes start 3 s apart, and a lease claim refused only because
  the GPU probe could not be read (`device_probe_unavailable`) is retried with
  backoff. Each lane records its commands, logs and private evidence. A
  candidate is rejected and the lane continues when the scene is unstable, the
  expert's plan or the official predicate fails (`oracle_failure`), or the
  expert's own task code raises (`oracle_exception`, innermost frame under
  RoboTwin `envs/`; upstream `collect_data.py` treats any `play_once` exception
  as a failed seed). An exception from anywhere else (CUDA, CuRobo, SAPIEN, the
  Harness) is `infra_failure` and stops that task's lane. A seed whose expert
  run gives no result for 2 h (`SEED_WATCHDOG_S`; 3 h, `SEED_WATCHDOG_FIRST_S`,
  for a worker process's first seed, which carries its start-up) is recorded in
  the worker's ledger as rejected with classification `oracle_timeout`
  (`recorded_by: robotwin_evaluation.seed_watchdog`, with `limit_s` and
  `first_in_process`); the worker is stopped and restarted and resumes at the
  next seed, and the lane result lists those seeds in `oracle_timeouts`. There
  is no other per-case timeout.
  It seals one `seed-manifests/<task>.json` per scanned task from the worker's
  ledger: accepted seeds and rejected candidates with their classification in
  requested-seed order, `complete` when the target was reached, per-seed
  evidence under `seeds[<seed>]` (`reset_predicate_false`, `oracle_success`,
  `post_success_physics_hold`, measured `terminal_stability_steps`),
  aggregates derived from those records (`unstable_accepted_seeds`,
  `required_stability_steps`) and `manifest_sha256`, so a RunSpec can name it
  as its `qualification.manifest`. Sealing leaves out `infra_failure` and
  `invalid_artifact` records (the worker resumes at that seed) and timeouts
  recorded under a shorter bound than the one now in force (earlier builds used
  30/90 min, which stopped sound slow seeds), so those seeds count as
  unscanned. Launch preflight admits a seed of this
  schema when the manifest's task, `task_config` and `robotwin_revision` match
  the RunSpec, the digest verifies, and the seed's own reset/oracle evidence
  holds; the post-success hold is evidence, never admission, because upstream
  collects such seeds. Exit 0 means every requested case
  passed; exit 2 reports incomplete/failing cases. This does not authorize or
  launch a policy episode.
- `merge --task-config C --inputs <qualification dirs or manifests> --output DIR
  [--accepted-per-task N] [--conflicting-outcomes refuse|reject]` joins each task's segments: walking seeds upward
  from the official start, it keeps the first N accepted seeds of the
  contiguous scanned prefix, exactly what one sequential scan collects.
  Segments must share the RoboTwin revision and the task's own dependency
  closure (shared inputs, the checker and that task's `envs/<task>.py`, read
  from each segment's `prepared-plan.json`; a bare manifest falls back to the
  plan-wide producer hash, which differs whenever two plans schedule different
  tasks). A seed recorded with two outcomes is refused unless
  `--conflicting-outcomes reject` is given, which counts a seed accepted in one
  segment and rejected in another as rejected (classification
  `conflicting_outcomes`, listed in `conflicting_seeds`) and keeps the earliest
  segment's record when re-scans agree. Expert scene stability is not
  reproducible run to run, so the official suite manifests use `reject`. A gap before N leaves
  the manifest incomplete with `next_unscanned_seed`. The merged manifest keeps
  the seed-manifest schema, so `seed_admission` reads it like any `qualify`
  manifest, and lists its segments in `merged_from`; `merge-report.json`
  summarizes the tasks. Exit 2 if any task is incomplete.
- `seal --inputs <qualification dirs>` seals interim manifests from a scan that
  is still running into `seed-manifests-interim/` (the final `seed-manifests/`
  is untouched); a torn final ledger line is skipped. `merge` reads the interim
  folder when no final one exists, so a small `--accepted-per-task` gives a
  pilot list that is a prefix of the final one. Neither command authorizes or
  launches a policy episode.

Outer seed defaults to zero, hence requested seeds begin at 100000.
The exclusive limit is start plus the explicit candidate count. Preparation
reports the actual longest-lane estimate (540 seconds per candidate plus
90 seconds setup/finalization); estimates over 15 minutes are explicitly
`development_diagnostic`, never an implicit per-launch requirement.
Saved preparations do not automatically discover/reuse historical evidence.
Do not reschedule already valid cases solely because HEAD changed; identify
missing or dependency-invalidated cases before requesting a new preparation.
Historical experimental plan schemas/confirmation codes are not accepted by
this entry. No fixed GPU assignment, provider or policy binding is inherited.

## RoboTwin and LIBERO-Pro RunSpec builder

`uv run --locked python -m ora.orchestrator.benchmark_spec` composes
one RoboTwin or LIBERO-Pro cell on the default baseline
([RunSpec inputs](../reference/configuration-and-evidence.md#fresh-benchmark-specifications));
RoboDojo cells use `ora.orchestrator.robodojo_spec`.

- `tasks robotwin|libero_pro` prints the RoboTwin roster with each public
  instruction and seed-named flag, or each core suite's ten BDDL languages.
- `spec|compose robotwin TASK --seed N [--task-config demo_randomized|demo_clean]
  --qualification MANIFEST [--episode-names FILE]` and `spec|compose libero_pro
  SUITE TASK_ID --seed N`, each with `--experiment NS --resource-profile SLOT
  --service-report REPORT --cohort ID [--host HOST] [--provider-profile ID]
  [--provider-catalog PATH] [--config ID] [--milestone M] [--output PATH]`.
  `spec` prints the RunSpec; `compose` writes `outputs/NS/specs/<config>.json`
  (or `--output`), imports the RoboTwin seed manifest under
  `outputs/NS/qualification/seed-manifests/` and prints the spec path. A seed
  the manifest does not admit, a slot the host does not register for the
  backend, or a stale or re-pointed readiness report is refused before anything
  is written. It starts no simulator, service or Provider and grants no launch.

### Generic command maintenance

When adding, renaming or removing a public command, update all of the following
in the same change:

1. `PUBLIC_COMMANDS` and `build_parser()` in
   `ora/orchestrator/cli.py`;
2. the command table above;
3. the command's deterministic contract or dispatcher test; and
4. `scripts/README.md` when the operator workflow changes.

Do not add another shell wrapper merely to fill in a task, config or milestone.
Those selections belong to the explicit experiment CLI so the chosen study
identity remains visible.


### Startup binding and refusal diagnostics

Generic launch and preflight share the controller-source check with the RoboTwin
factory and select resident endpoints from the same RunPlan bindings as policy
tools. Missing experiment-wrapper environment setup no longer redirects a bound
SAM3/grasp client to a default port. Conflicting explicit endpoint overrides still
fail identity validation. A duplicate single-attempt claim prints its cohort/cell,
original run/plan and creation time to stderr and returns a failure exit code;
missing historical owner metadata is explicitly unavailable.

`./scripts/model-services --help` lists the resident-service commands
(`start`, `status`, `stop`, `qualify`); `examples/official_suite/run.py
--help` lists the official-suite cell, compose and queue commands.
`uv run --locked python -m ora.orchestrator.inspect_export --help`
lists sealed-run export options; this command is an evidence exporter.
