# Tests

The deterministic test suite. Every test runs on a CPU-only machine: no
provider, simulator, GPU, resident model service or physical robot is started.
Backend tests drive the real session code against fake plants, fake SDKs and
recorded numeric witnesses, so they check contracts, not simulator physics.

[`test-manifest.json`](test-manifest.json) states, for every test module, the
contract it protects. Read a module's entry there before changing an expected
value; `tests/cli/test_control_plane_cli.py` keeps the manifest equal to the
files on disk and every module inside one of the directories below.

## Layout

| Directory | What it covers |
|---|---|
| `tests/agent/native/` | The Native policy session: the provider loop, OpenAI-compatible and Anthropic wire translation, streaming recovery and deadlines, credentials and gateway binding, multi-round sessions, operator stop, evidence writing and SFT export. |
| `tests/agent/context/` | What the model sees each turn: Context v4-v6 projection and lossless aging, observation and image delivery, public robot facts, delivery trimming and de-duplication, round static facts, receipt and feedback projection, and the privileged-field audit. |
| `tests/agent/motion/` | Motion tools: pose and frame compilation, `move_pose`/`move_to`/rotations, reach checks and nearest reachable orientation, observed clearance, wait and gripper actions, the motion transaction and repair admission. |
| `tests/agent/perception/` | Perception and manipulation geometry: segmentation, grounding and grasp candidates, grasp review and rejection schemas, placement geometry and depth, target localization and support height, retention, `mark_point`, and their refusals. |
| `tests/agent/tools/` | The tool runtime itself: argument and schema validation before dispatch, error envelopes and infrastructure classification, the restricted `compute` evaluator, closed skills and skill sequences, finish/give-up/abort outcomes. |
| `tests/agent/composition/` | How a run's policy is composed: the default baseline, ToolSurface manifests and published help, prompt bundles, embodiment descriptors and cards, task contracts, published limits, backend parity and model-visible identity. |
| `tests/environments/` | Backend-neutral environment code: the isolated worker RPC, execution capture, local CPU platform selection and control-profile capabilities. Synthetic is exercised through the session, orchestrator and quickstart tests rather than its own directory. |
| `tests/environments/libero/` | The direct LIBERO/LIBERO-Pro driver: reset and termination, camera resolution, CuRobo planning diagnostics and reach checks, gripper feedback, wait, robustness variants and software-rendered lanes. |
| `tests/environments/robotwin/` | The RoboTwin backend: dual-arm motion v2 and the vendored planner, IK feedback, contact recovery, inter-arm screening, placement adapter, evaluation setting, seed scans and the vendored install pin. |
| `tests/environments/robodojo/` | The RoboDojo backend and its ARX X5 embodiment: session and SDK boundary, IK continuity, inter-arm admission, recovery, layouts and rounds, the closed-skill default surface, the worker kernel set, qualification and the operator/installer. |
| `tests/orchestrator/` | RunSpec/RunPlan resolution and the episode lifecycle: preflight, leases and resource concurrency, host storage, workspace and run identity, evaluation, verdict guards, sealing, run reports, the official suite and suite queue, and the `experiments/runtime_v1` qualification ladder. |
| `tests/cli/` | User commands: the `ora` CLI surface and its documentation, `ora local`/`setup`/`run` on one GPU, `ora interactive`, `ora watch`, `ora-remote`, the world view, `quickstart` and `baselines --explain`. |
| `tests/viewer/` | The read-only Viewer: run discovery, controller replay and caches, process timelines and videos, pricing and visual-fidelity labels. |
| `tests/model_services/` | The resident SAM3 / Contact-GraspNet services: the operator, `scripts/model-services` and the Contact-GraspNet input policy. |
| `tests/repository/` | Repository guards: `policy.lock`, path-only asset references, packaging, the module-size ratchet, undefined globals, dead tool-runtime state and retired import paths. |
| `tests/support/` | Shared helpers, not tests: Native RunSpecs, placement, observation and delivery fixtures, test-local ToolSurface copies, a scratch-tree `policy.lock` writer and synthetic run and journal writers. |

A module named after a backend (`test_robodojo_*`, `test_robotwin_*`,
`test_libero_*`) lives under that backend's directory even when the code it
covers sits in `ora/orchestrator/` or `ora/agent/`.

## Running

```bash
uv run --locked pytest -q                                   # everything
uv run --locked pytest -q -n auto                           # in parallel (pytest-xdist)
uv run --locked pytest -q tests/environments/robodojo/      # one area
uv run --locked pytest -q tests/agent/motion/test_reach_check.py   # one module
```

The suite is collected from `tests/` (`[tool.pytest.ini_options]` in
`pyproject.toml`) and the repository root is on `sys.path`, so a module imports
helpers as `tests.support.<name>` and, where it reuses another module's
fixture, names that module by its full package path (for example
`tests.environments.robodojo.test_robodojo_backend`). CI runs
`./scripts/ora lock --check` before the suite; a stale `policy.lock` also fails
`tests/repository/test_policy_lock.py`.

## External inputs

Nothing in the suite needs a simulator or a GPU. It does need `ffmpeg` and
`ffprobe` on `PATH` for the Viewer video export tests (CI installs them; on
Ubuntu, `sudo apt-get install ffmpeg`). A few cases depend on optional local
inputs and skip without them:

- `tar` and `zstd` for `tests/agent/native/test_request_archive.py`;
- the pinned RoboDojo source and Assets (`ROBODOJO_ASSET_ROOT`, default
  `~/robodojo-runtime/assets/Assets`) for some layout, X5 IK and CPU-physics
  cases under `tests/environments/robodojo/`;
- a separately pinned Inspect runtime for one case of
  `tests/orchestrator/test_inspect_export.py`.

Live simulator, provider and GPU qualification is a separate activity (see
[`docs/guides/qualification-authoring.md`](../docs/guides/qualification-authoring.md)),
not part of this suite.

## Fixtures and data

- `conftest.py` points `ORA_WORKSPACE_ROOT` at a temporary directory before any
  `ora` module is imported, binds the fixture execution hosts through
  `ORA_HOSTS_DIR=tests/data/hosts`, stubs the host-storage probes (opt out with
  the `host_storage_real` marker) and provides the `robodojo_activation` and
  `ready_model_services` fixtures.
- `data/hosts/`: the `test-host`, `test-cluster` and `test-arm` execution host
  profiles.
- `data/prompts/` and `data/robodojo/*.md`: small prompt and manual layers the
  runtime tests compose instead of the shipped bundles.
- `data/delivery_trim/`: per-backend observation and delivered-text goldens for
  the delivery trim.
- `data/robodojo/*-receipts.json`: recorded RoboDojo receipts the per-arm
  translation and retract-diagnostic projections are checked against.
- `data/default-expectations.json`: the expected default baseline and
  per-backend instances the default-activation checks compare against.
- `data/identity-registry-fixture.json`: a synthetic experiment lineage for the
  run-identity tests.

## Adding a test

Put the module in the directory of the code it protects, add its row to
`test-manifest.json` with a one-sentence contract, and move a helper that more
than one module needs into `tests/support/`. Prefer a behavioural assertion
over a source-layout one, and the smallest check that exercises the change.
