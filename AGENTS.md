# ORA: contributor and agent guide

This repository is a thin harness that runs a language-model agent as the
policy of a simulated robot. The agent sees only public observations and tools;
an isolated simulator backend executes its actions; only the benchmark's own
official boundary decides success. Every run is planned, executed, evaluated
and sealed with hash-bound evidence.

## Layout

- `ora/orchestrator/`: RunSpec/RunPlan resolution, defaults, host
  profiles, leases, episode loop, evaluation, sealing and the CLI.
- `ora/agent/`: the Native policy (direct OpenAI-compatible or
  Anthropic provider loop), context composition, policy-visible tools and
  evidence formats; `agent/model_services/` holds the resident perception
  services.
- `ora/contracts/`: the versioned action envelope shared by agent
  and environment.
- `ora/environment/`: transport-neutral sessions, Synthetic, and the
  isolated RoboTwin, LIBERO/LIBERO-Pro and RoboDojo backends (loaded only by
  the worker process under each simulator's own interpreter).
- `ora/policy/`: prompt bundles, ToolSurfaces, embodiment and
  environment assets, the default baseline catalog. `policy.lock` pins them.
- `experiments/runtime_v1/`: the M0-M5 Runtime qualification ladder.
- `examples/`: the official-suite launcher and the host profile template.
- `viewer/`: the read-only local run viewer.
- `tests/`: the deterministic CPU-only suite, grouped by area (`agent/`,
  `environments/<backend>/`, `orchestrator/`, `cli/`, `viewer/`,
  `model_services/`, `repository/`; shared helpers in `support/`).
  `tests/README.md` describes each area; `tests/test-manifest.json` states
  each module's contract.
- `scripts/`: user commands (`ora`, `ora-remote`, `open-viewer`, ...);
  `scripts/dev/` holds authoring and maintenance tools. `scripts/README.md`
  lists both.

## Setup and checks

```bash
uv sync --all-extras --locked
uv run --locked ora doctor
uv run --locked python -m pytest -q tests      # add -n auto with pytest-xdist
uv run --locked python -m pytest -q tests/agent/motion   # one area
./scripts/ora lock --check           # policy.lock is current
./scripts/ora baselines --explain    # how the default composes
uv run --locked python experiments/runtime_v1/run.py configs
./scripts/open-viewer
```

Python 3.11 and `uv.lock` are canonical; do not regenerate the lock as a side
effect of another change. Simulators run in their own interpreters, named by
environment variables (`ROBOTWIN_PYTHON`, `LIBERO_PRO_PYTHON`,
`ROBODOJO_PYTHON`). A live run needs a registered execution host: copy
`examples/hosts/example-host.yaml` to `ora/orchestrator/hosts/<id>.yaml`
(or a directory named by `ORA_HOSTS_DIR`) and export
`ORA_EXECUTION_HOST=<id>`. Generated runs and credentials stay
outside Git under the ignored `outputs/` workspace.

## Invariants

- The policy never receives privileged information: no object ground truth,
  segmentation, contacts, task source, expert code or success predicate.
  `observation.dump_state` is read-only; `robot.submit_action` (or a surface's
  renamed proposal tool) is the only state-changing tool and always passes the
  Supervisor.
- Success comes only from the selected benchmark's official boundary
  (RoboTwin `check_success`, LIBERO termination, RoboDojo's task success).
  Synthetic runs report `official_success: null`.
- What the model sees is identity-checked. Any change to prompts, tools,
  observation projection or composition must be deliberate: run
  `scripts/dev/compare-model-visible <base> HEAD` and explain every difference.
- Policy assets are referenced by path and pinned by `policy.lock`. After
  adding or changing a file under `ora/policy/`, run
  `./scripts/ora lock` and commit the lock in the same change.
- A new feature becomes the default for the maintained consumers it applies
  to (the baseline catalog and default resolvers), with its tests and the
  matching `docs/reference/` page updated in the same change.
- Keep changes minimal and in the surrounding style; prefer the smallest test
  that exercises the changed behavior.

## Documentation

`docs/reference/README.md` describes the current APIs, defaults, authority
boundaries and evidence model; `docs/architecture.md` and `docs/code-map.md`
describe structure; `docs/guides/` holds authoring guides (tool surfaces,
prompt bundles, task contracts, environment compatibility, qualification).

The documentation site uses MkDocs Material. Run `./scripts/docs serve` for a
local preview and `./scripts/docs build` for strict validation. Register public
pages in `mkdocs.yml`. See `docs/developer/documentation.md`. Documentation
dependencies have their own hash-locked requirements; do not change the
runtime's `uv.lock` for them.
