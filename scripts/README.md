# Script and command inventory

`ora experiments [FAMILY] [--json]` inspects source-tracked generation lineage and exact-source historical aliases; see [experiment identity](../docs/reference/experiment-identity.md).

This is the canonical inventory for human-operated repository commands. A
script stays here only when it adds a stable command surface or unique safety
logic. Thin per-task or per-milestone launch aliases are intentionally
forbidden: operators must be able to see the experiment, policy and milestone
in the command they run.

## Retained repository scripts

`scripts/audit-task-contracts` is the offline task-contract review command.
`template --backend KIND --task TASK` emits a pending private review record;
`render --record PATH` emits only reviewed public clauses; `audit` inventories
the entire source roster. Add `--require-reviewed` to fail on incomplete or
unregistered tasks, and `--source-root/--evidence-root` to verify external
dependency/witness hashes. It writes stdout, calls no provider/simulator and
never activates or launches a treatment. See the
[authoring guide](../docs/guides/task-contract-and-feedback-authoring.md).

| Entry | Purpose | Inputs | Output or effect | Run where |
|---|---|---|---|---|
| `scripts/docs` | Preview or strictly build the MkDocs Material documentation in an isolated, locked uv environment. | `serve` (default), `build`, or `--help`; optional MkDocs arguments. | Local preview at port 8000 or static output under `.cache/docs/site/`; no publish, push, model or simulator calls. | Repository root while writing or reviewing documentation. |
| `scripts/compare-model-visible` | Prove that two commits let the model see the same thing. Each commit's tree is extracted with `git archive` into a scratch directory under `$TMPDIR`, committed once into a throwaway repository (because `resolve_run_spec` runs `git rev-parse HEAD` and `git status`), and resolved by that tree's own `resolve_run_spec` in a child process with an isolated `ORA_WORKSPACE_ROOT`. The fixed matrix is 7 offline Native RunSpecs: default `direct_v3` and frozen `direct_v2`, each on LIBERO-Pro and RoboTwin, and three RoboDojo cells (the default `direct_v4` instance from a RunSpec naming no surface, `robodojo_toolchain_b2_closed_noservo_v4`, and `robodojo_placement_closed_skills_v1` with its bundle and semantic manifest). For each cell it computes the five `model_visible_sha256` items (advertised composition, prompt-layer texts, embodiment card and projected facts, observation projection, masked policy/runtime/budgets without execution placement) with `ora/orchestrator/model_visible.py` from this checkout, so a base commit that predates `policy_identity.model_visible_sha256` is compared the same way; card rendering uses each tree's own code. | `BASE CANDIDATE` commits; optional `--json REPORT` and `--keep` to retain the scratch trees. | One `same`/`different`/`unresolved_at_both` line per cell with the differing tools, layers or JSON paths, and a summary; exit 0 when nothing differs, 1 on any difference, 2 when nothing could be compared. Writes only its scratch directory; never touches the checkout, the shared Runtime workspace, a provider or a simulator. Both trees run in this checkout's `.venv`. | Repository root before integrating a refactor or asset-format change that must not change model-visible content (open-source readiness plan §5.1). |
| `scripts/audit-prompt-bundle` | Create a closed draft prompt-bundle manifest or statically audit one complete bundle: layer ownership, nine-node embodiment format, camera inventory and exact source identity, without composing a live run or calling a provider. | `template` with a bundle id, or `audit` with a bundle manifest path. | A draft YAML manifest or a finite JSON audit verdict on stdout; never edits policy assets or RunPlans. | Repository root while authoring or reviewing prompt bundles. |
| `scripts/refresh-vendored-pins` | Recompute the deterministic install revision of the vendored RoboTwin planner chain and rewrite the current installer constant, v10 control-profile `source_binding` and v10 registry hash idempotently; frozen v1–v9 and experiment pins are preserved. | Optional `--robotwin-git` source object database and `--project-root` pin destination; reads the current vendored payload bytes. | Updated current pin constants/bindings, or no change when already current; writes temporary Git objects in a disposable bare repository and reads the external clone through alternates without changing its objects, index, refs or files. | Repository root after editing `environment/backends/robotwin/vendored/`. |
| `scripts/setup-robodojo` | Prepare the pinned isolated RoboDojo runtime through `ora/environment/backends/robodojo/installation/setup_runtime.py`. | Setup CLI arguments, including explicit source/install roots and host settings. | Creates or verifies the simulator installation outside Git; does not launch a formal episode. | Qualified RoboDojo host during environment setup. |
| `scripts/model-services` | Start, inspect, stop and qualify the resident SAM3 / Contact-GraspNet services through the shared model-services operator (`ora.orchestrator.model_services_cli`). | `start`, `status`, `stop` or `qualify`; optional `--state DIR` (default `model-services` under the configured output storage); selectors from `B2_SAM_*` / `B2_GRASP_*`. | Process identities or readiness JSON; `qualify` prints the hash-named readiness report a launch binds; `status` exits 1 until both services are ready; `stop` refuses while Runtime leases exist. | A host that runs live RoboTwin or LIBERO-Pro episodes, before `preflight`/`launch`. |
| `scripts/run-report` | Read-only stop-reason and cost report for sealed or in-progress runs on every backend (`python -m ora.orchestrator.run_report`): per attempt (round), per cohort, per (model, task) and in total it reports official success, validity (evaluation status, learning eligibility and post-seal corrections under `<namespace>/diagnostics/corrections/`), termination and its authority, the ending class, the policy finish outcome and give_up category, decisions, model calls, wall minutes, the tool-call mix, refused tool calls by tool and code, provider tokens and cache share, an API-equivalent cost from `ora/orchestrator/run_report_pricing.yaml` (unpriced models null) and request/observation context sizes. Not yet a `ora` subcommand: a later unit wires it into `cli.py`. | Run directories, or `--namespace NS` with repeatable `--cohort ID`; optional `--json FILE`, `--markdown FILE`, `--format markdown\|json` for stdout, `--pricing FILE`, repeatable `--corrections FILE` or `--no-corrections`, `--jobs N`. | Markdown on stdout, or the JSON (`ora.run-report.v1`) and Markdown files; refuses an output path inside a source run; never writes inside a run or starts a simulator or provider; exits 1 when a run directory cannot be read. | Any checkout that can read the run directories, during or after an experiment. |
| `scripts/ora` | Repository-local wrapper for `ora.orchestrator.cli`. It selects `.venv/bin/python` when present and otherwise uses `PYTHON`/`python3`; it adds no hidden behavior. | An Orchestrator subcommand and RunSpec or exact `run_id`; `workspace` reports the shared evidence/state roots, `providers` lists reusable credential-free registrations, `state RUN_ID [--step N]` reads verified public state, `metrics RUN_ID` rebuilds Runtime measurements, and the wire commands validate/export opted-in managed-Agent traffic. | Resolves the Git-common Runtime workspace, atomically claims the RunPlan's bound execution-resource slot, lists every active slot lease, or plans, launches, inspects, stops, evaluates, seals and exports one exact run without affecting independent slots. | Repository root; live launch only on the qualified host. |
| `scripts/ora-remote` | Run `ora` on a remote host over SSH and show its terminal UI and world view locally (`python -m ora.orchestrator.ora_remote`). | `HOST --repo /remote/checkout [--local-port N] [--remote-port N] [--connect-timeout S] -- watch RUN \| interactive --spec ...`; `HOST` may be an SSH config alias. | Opens one SSH session with a loopback tunnel, waits for the forwarded viewer, then opens the local browser; adds no launch confirmation and does not reconnect or restart a run. | Your computer, against a host where the run executes. |
| `scripts/compile-embodiment-card` | Create a strict draft template, validate one policy-safe embodiment descriptor, preview the exact model-visible card, or deterministically compile the card and hash manifest. It imports no simulator and never changes a RunPlan or prompt profile. | `template`, `validate`, `preview` or `compile`; an explicit descriptor path and, for generated files, an explicit output path. | A draft JSON descriptor, stdout preview, or `manual.md` plus `manifest.json`; drifted outputs are not replaced unless `--replace` is explicit. | Repository root; generated compilations normally go under ignored `outputs/embodiment-cards/`. |
| `scripts/open-viewer` | Select the project interpreter, start the one read-only Viewer and open an allowlisted start page. Automatic discovery combines the Git-common shared outputs root, live linked-worktree legacy roots and nested worktree snapshots; formal/attempt/qualification/diagnostic filters remain evidence-only. | `--page runs\|libero\|libero-pro\|robotwin` selects the initial browser page; optional `--host`, `--port` and repeatable replacement `--outputs-root` arguments retain their existing meaning. `ORA_WORKSPACE_ROOT` selects the shared physical workspace and `ORA_EXTRA_OUTPUT_ROOTS` adds copied collections. | One aggregated local Viewer process and one browser tab. The LIBERO atlas reads the preview manifest but never launches a simulator or preview build. | Local analysis machine. |
| `scripts/compile-viewer-videos` | Select the same locked interpreter as the Viewer and invoke its canonical batch compiler without starting a web server. Existing current-identity videos are verified and reused; missing or source-drifted videos are compiled. Default execution review uses original images without simulator startup; only explicit `--replay-mode forensic` reconstructs physics. | Optional `--video-scope formal\|libero`, repeatable `--outputs-root PATH`, repeatable exact `--run-id`, `--video-mode execution\|process`, one-based `--round`, `--include-reasoning`, `--include-context` and `--card-seconds 0.5..15`. Defaults to formal runs; exact IDs override that filter. | A finite per-run `OK`/`FAIL` report and content-keyed MP4/manifest pairs under `viewer-derived/`; exits non-zero if any selected video fails. Process mode composes recorded cards and verified cached execution clips with CPU-only encoding; it never launches a simulator/provider. | Local analysis machine after runs are fetched or before trajectory review. |
| `scripts/compile-libero-previews` | Build or verify the complete source-pinned seed-0 Gallery image set without executing robot actions. The web process never invokes it. | Pinned `LIBERO_PRO_PYTHON`, `LIBERO_PRO_ROOT` and `LIBERO_PRO_DATA_ROOT` through environment variables or explicit flags; optional `--jobs 1..8` and `--status`. | 130 original LIBERO, 160 core LIBERO-Pro and 280 BDDL-driven robustness 256×256 `agentview` PNGs plus one atomic hash manifest below ignored `outputs/viewer-derived/libero-pro-previews/`. Interrupted content-addressed builds reuse the prior 480 identities and resume safely. | Qualified local analysis/GPU host; no provider credentials or model calls. |

Use `./scripts/audit-prompt-bundle --help`,
`./scripts/compare-model-visible --help`,
`./scripts/ora --help`,
`./scripts/compile-embodiment-card --help`,
`./scripts/open-viewer --help`,
`./scripts/compile-viewer-videos --help`,
`./scripts/compile-libero-previews --help` and `./scripts/run-report --help`
for the current argument surface.

`ora lock` regenerates `policy.lock` at the repository root: one
`<sha256>  <path>` line per tracked policy asset, sorted by path, in
`sha256sum` format, so `sha256sum -c policy.lock` also verifies a checkout.
It covers every tracked file under `ora/policy/` (except Python
code) plus the path-loaded assets that live elsewhere; the list and the loader
behind each entry are `COVERAGE` in `ora/orchestrator/policy_lock.py`.
Only `git ls-files` entries count, so `git add` a new asset before
regenerating. `ora lock --check` writes nothing, reports missing,
extra and changed entries and exits 1 on drift; CI runs it before any test. At run time a loader only
checks that an asset reference is listed in the lock, then hashes the bytes it
reads; each RunPlan records the lock's sha256 as
`policy_identity.policy_lock_sha256`. `ora lock --strip-pins`
rewrites every asset this tree composes to name other assets by path only
(including embodiment descriptors' repository sources and wording reviews'
`descriptor_sha256`), keeping each file's formatting, refusing a stale pin,
regenerates the compiled card manifests of rewritten descriptors and then the
lock; with `--check` it lists what it would change and exits 1
([path-only asset references](../docs/reference/configuration-and-evidence.md#path-only-asset-references)).

`ora baselines [--json]` inspects the provider-independent Native robot
default catalog without contacting a provider or model service. `plan` and
`context` inspect the exact task-bound expansion; neither authorizes launch.
`ora baselines --explain [--environment KIND] [--json]` explains that
default for a newcomer: why a baseline is selected, the per-environment
resolution chain (surface, policy defaults, runtime template and the keys a
RunSpec may still choose, prompt layers, resident services), the frozen
baselines, and for every tool its effect, its handler as `module:qualname` and
`file:line`, and how a state-changing tool's proposal reaches the backend. It is
derived from the code, read-only and writes no file.

The public generic subcommands, their exact syntax, local effects, lifecycle
order and automated/live validation boundary are maintained in
[`docs/guides/cli-reference.md`](../docs/guides/cli-reference.md). The parser,
repository-wrapper help for every command and that table are checked together.

Open the Viewer directly:

```bash
./scripts/open-viewer
./scripts/open-viewer --page libero
./scripts/ora workspace
./scripts/ora providers
./scripts/ora providers --json
./scripts/compile-embodiment-card preview ora/policy/embodiments/franka-panda/descriptor-public-geometry-v8-controller-feedback-v1-session-split-v1.json
./scripts/compile-viewer-videos
./scripts/compile-viewer-videos --video-scope libero
./scripts/compile-viewer-videos --outputs-root /path/to/copied/outputs
./scripts/compile-libero-previews --status
./scripts/run-report --namespace /path/to/outputs/<experiment> \
  --cohort <cohort-a> --cohort <cohort-b> \
  --json report.json --markdown report.md
```

Live GPU, Provider and simulator qualifications remain exact-source
experiment gates outside generic CI; the workflow has read-only repository
permission and no credentials.

## Experiment and specialist entry points

These Python files are executable command surfaces even though they do not
live in `scripts/`:

| Entry | Purpose | Intended caller |
|---|---|---|
| `examples/official_suite/run.py` | List the official LIBERO-Pro core and RoboTwin seed-manifest cells, compose one cell's RunSpec on the default baseline, qualify the resident services, or queue a selection across lanes with one attempt per cell. | Official-suite operator on a registered host. |
| `experiments/runtime_v1/run.py` | Operate the first-class Runtime v1 experiment package through the standard discover/context/plan/preflight/launch/observe/evaluate/seal lifecycle, including M0-M5 prerequisite gates, read-only LIBERO control audit and the source-pinned Pose/Frame v2 Panda qualification matrix. | Runtime qualification operator. |
| `ora/orchestrator/cli.py` | Canonical generic RunSpec/episode-control implementation behind `scripts/ora` and the experiment CLIs. | Wrappers and advanced debugging; prefer the higher-level experiment CLI for normal work. |
| `ora/environment/backends/robotwin/qualification.py` | Run the trusted RoboTwin expert/oracle seed qualification loop. | EXP00/runtime composition commands; direct use only when debugging qualification. |
| `viewer/app.py` | Start the unified local read-only experiment config catalog, exact run-composition inspector, trajectory/raw-event viewers, generic run registry and 50-task RoboTwin Gallery on `127.0.0.1:5012`. | Local operator before task inspection or after runs are fetched. |

The installed equivalents declared in `pyproject.toml` are `ora`,
`robotwin-qualify`, and `ora-viewer`.

`ora/environment/worker.py` is an internal child-process entry point launched only
by `LocalRpcEnvironmentSession`; it is not an additional operator command.
Its config is per-run state under ignored outputs; its short-lived Unix socket
lives under ignored `.harness-runtime/env/` and is removed on close.

## Normal paths

Runtime qualification:

```bash
uv run --locked python experiments/runtime_v1/run.py configs
uv run --locked python experiments/runtime_v1/run.py explain M0
uv run --locked python experiments/runtime_v1/run.py context M0 --policy codex
uv run --locked python experiments/runtime_v1/run.py plan M0 --policy scripted
uv run --locked python experiments/runtime_v1/run.py preflight M0 --policy scripted
uv run --locked python experiments/runtime_v1/run.py control-audit
uv run --locked python experiments/runtime_v1/run.py libero-pose-control
```

The generic `ora providers` command reads the shared packaged
registry, not a frozen experiment catalog. Its Volcengine Agent Plan GLM-5.2
entry records the independently qualified Anthropic Messages text/tool route,
official 1M context and 128K output limits, and environment-variable names
without storing credentials. It makes no image-capability claim; behavioral
robot use still starts on a new source-bound experiment branch.

## Removed aliases

The repository deliberately removed `scripts/run-exp00` and
`scripts/run-harness-m0` through `scripts/run-harness-m5`. They only forwarded
arguments to an existing CLI, created seven extra entry points to maintain,
and made direct launch commands easier to run without first inspecting the
plan. Do not recreate equivalent aliases.

## Maintenance rule

- Add a script only when it contains unique safety, portability or integration
  behavior that cannot live in an existing CLI.
- Add or update its row here in the same change.
- Give Python scripts an `argparse` description/docstring and shell scripts a
  header comment.
- Delete an entry when its unique behavior moves into a canonical CLI.
- `tests/test_control_plane_cli.py::test_documented_script_and_test_module_inventory_is_current`
  prevents undocumented executable or test surfaces from entering the tree.
