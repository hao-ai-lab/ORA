# Test suite navigation

Start with the changed contract, then run the modules that own it.
`tests/test-manifest.json` records every module's contract; `tests/README.md`
describes the area directories (`tests/agent/`, `tests/environments/<backend>/`,
`tests/orchestrator/`, `tests/cli/`, `tests/viewer/`, `tests/model_services/`,
`tests/repository/`) and the shared helpers in `tests/support/`. This guide
explains how to choose and split checks.

## Choose a check

| Check | Command | Evidence boundary |
|---|---|---|
| Single module | `uv run --locked pytest -q tests/<area>/test_<name>.py` | The changed contract only |
| One area | `uv run --locked pytest -q tests/<area>/` | Every module of that area, for example `tests/environments/robodojo/` |
| Bare suite | `uv run --locked pytest -q` | All deterministic tests |

Inspect CPU, memory, existing jobs and simulator slots before scheduling checks.
Use available parallel CPU capacity for independent offline cases. Fixture-based
robot checks do not certify simulator physics or live service readiness.

## Families split by protected contract

| Family | Area | Base contract | Children |
|---|---|---|---|
| `test_control_plane` | `tests/orchestrator/` | Planning, leases, milestone/snapshot gates and sealing | `preflight`, `workspace`; `cli` in `tests/cli/`, `viewer` in `tests/viewer/` |
| `test_runtime_v1` | `tests/orchestrator/` | Lifecycle and session/resource contracts | `state`, `metrics`, `tools`, `libero` |
| `test_native_harness` | `tests/agent/native/` | Composition and shared recovery | `anthropic`, `openai`, `sft`, `loop` |
| `test_lossless_context_runtime` | `tests/agent/context/` | Recoverable projection | `cost`, `triggers` |
| `test_robotwin_motion_v2` | `tests/environments/robotwin/` | Motion execution | `grasp`, `probe`, `profiles`, `sparse`, `surface` |

Each child has its own manifest contract. Shared fixtures and constants keep a
single owner: either the base module, which children import by its package
path (`from tests.agent.native.test_native_harness import ...`), or a module in
`tests/support/` when several areas use it. Do not duplicate constants into
children: an asset update must not leave independent copies of its expected
hash behind.

For another split or move, compare the complete test-function/decorator AST set
and the per-module pytest collection before and after. Exercise cross-module
helper consumers as well as the moved tests. A collection count alone cannot
detect replacement of one test by another.

## Timing and parallel execution

Splitting improves navigation and gives file-based schedulers smaller work
units. It does not itself demonstrate faster wall time: uneven file costs can
still put most work in one shard.

Use the current environment's installed runner and bounded timing evidence.
When diagnosing a slow check, inspect subprocess startup, installed client
versions and shard imbalance before changing the test. Report actual worker
count, selected cases, elapsed time and failures. A transient timeout or an
installed-client incompatibility must be distinguished from a behavior regression
with a bounded reproduction, not an assumed historical explanation.

## Pinned assertions and readability

The module-size, parameter-delivery and ownership tests are ratchets. A
failure requires identifying which contract changed before editing an expected
value. Frozen profile hashes and tool inventories are deliberate
boundaries. Fix a stale probe when its implementation no longer checks the
contract; fix product code when it violates the existing contract.

`tests/repository/test_module_size_ratchet.py` governs product Python modules, not test-file
length. Names and docstrings describe the executable test; the manifest states
the module contract. Prefer behavior checks over source-layout assertions unless
the source itself is the artifact under test, such as delivered prompt text.
