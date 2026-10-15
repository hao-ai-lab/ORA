# Scripts

Repository-local commands. Each one selects the project interpreter
(`.venv/bin/python` when present) and adds no hidden behaviour; run `--help` on
any of them for the full argument surface. Run them from the repository root.

## User commands

| Script | What it does |
|---|---|
| `scripts/ora` | Wrapper for the `ora` CLI (`ora/orchestrator/cli.py`): setup, local and single-GPU runs, interactive sessions, watch, plan/preflight/launch/evaluate/seal, `lock`, `baselines`, `providers`, `workspace`. See [`docs/guides/cli-reference.md`](../docs/guides/cli-reference.md). |
| `scripts/ora-remote` | Run `ora` on a remote host over SSH and show its terminal UI and world view locally through a loopback tunnel (`python -m ora.orchestrator.ora_remote`). |
| `scripts/open-viewer` | Start the read-only local Viewer (`viewer/app.py`) and open a start page (`--page runs\|libero\|libero-pro\|robotwin`). |
| `scripts/model-services` | Start, inspect, stop and qualify the resident SAM3 and Contact-GraspNet services (`start`, `status`, `stop`, `qualify`). `stop` refuses while Runtime leases exist. |
| `scripts/setup-robodojo` | Prepare the pinned isolated RoboDojo runtime outside Git; it does not launch an episode. |
| `scripts/run-report` | Read-only stop-reason and cost report over sealed or in-progress runs (`python -m ora.orchestrator.run_report`), as Markdown or `ora.run-report.v1` JSON. Prices come from `ora/orchestrator/run_report_pricing.yaml`. |
| `scripts/docs` | Preview (`serve`) or strictly build (`build`) the MkDocs documentation in its own locked environment. |

## Developer tools (`scripts/dev/`)

Authoring and maintenance tools. None calls a provider or starts a simulator.

| Script | What it does |
|---|---|
| `scripts/dev/compare-model-visible` | `BASE CANDIDATE`: resolve a fixed matrix of offline Native RunSpecs at both commits (each in a scratch copy of its tree) and report which tools, prompt layers or projected fields the model would see differently. Exit 0 when nothing differs. Run it before integrating a change to prompts, tools, observation projection or composition. |
| `scripts/dev/audit-prompt-bundle` | `template` a draft prompt-bundle manifest or `audit` one bundle: layer ownership, embodiment format, cameras and source identity. See [`docs/guides/prompt-bundle-authoring.md`](../docs/guides/prompt-bundle-authoring.md). |
| `scripts/dev/audit-task-contracts` | Task-contract review: `template` a pending review record, `render` the reviewed public clauses, `audit` the roster (`--require-reviewed` to fail on gaps). See [`docs/guides/task-contract-and-feedback-authoring.md`](../docs/guides/task-contract-and-feedback-authoring.md). |
| `scripts/dev/compile-embodiment-card` | `template`, `validate`, `preview` or `compile` an embodiment descriptor and its model-visible card. |
| `scripts/dev/refresh-vendored-pins` | Recompute the install revision of the vendored RoboTwin planner after editing `ora/environment/backends/robotwin/vendored/` and rewrite the current pins; older pins are kept. |
| `scripts/dev/compile-viewer-videos` | Compile or verify the Viewer's execution and process videos without starting the web server. |
| `scripts/dev/compile-libero-previews` | Build or verify the seed-0 LIBERO/LIBERO-Pro gallery previews offscreen; needs the pinned LIBERO-Pro interpreter and data (`--status` only reports). |

## Other entry points

| Entry | Purpose |
|---|---|
| `ora/orchestrator/cli.py` | The implementation behind `scripts/ora` and the installed `ora` command. |
| `experiments/runtime_v1/run.py` | The M0-M5 Runtime qualification ladder (`configs`, `explain`, `plan`, `preflight`, `launch`, `control-audit`). |
| `examples/official_suite/run.py` | List, compose and queue the official LIBERO-Pro and RoboTwin suite cells. |
| `ora/environment/backends/robotwin/qualification.py` | RoboTwin expert/oracle seed qualification (installed as `robotwin-qualify`). |
| `viewer/app.py` | The Viewer server behind `scripts/open-viewer` (installed as `ora-viewer`). |

`ora/environment/worker.py` is the isolated simulator worker the Runtime
launches; it is not an operator command.

## Adding a script

Add a script only when it carries behaviour that cannot live in the `ora` CLI.
Put user commands in `scripts/` and maintenance tools in `scripts/dev/`, give it
a `--help` description, and add its row here in the same change:
`tests/cli/test_control_plane_cli.py::test_documented_script_and_test_module_inventory_is_current`
fails on any undocumented script.
