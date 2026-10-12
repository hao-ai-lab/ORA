# Test suite navigation

Start with the changed contract, then run the modules that own it.
`tests/test-manifest.json` records every module's contract; `tests/README.md`
describes the modules. This guide explains the current organization.

## Choose a check

| Check | Command | Evidence boundary |
|---|---|---|
| Single module | `uv run --locked pytest -q tests/test_<name>.py` | The changed contract only |
| Bare suite | `uv run --locked pytest -q` | All deterministic tests |

Inspect CPU, memory, existing jobs and simulator slots before scheduling checks.
Use available parallel CPU capacity for independent offline cases. Fixture-based
robot checks do not certify simulator physics or live service readiness.

## Five families split by protected contract

| Family | Base contract | Children |
|---|---|---|
| `test_control_plane` | Planning, leases, milestone/snapshot gates and sealing | `preflight`, `viewer`, `workspace`, `cli` |
| `test_runtime_v1` | Lifecycle and session/resource contracts | `state`, `metrics`, `tools`, `libero` |
| `test_native_harness` | Composition and shared recovery | `anthropic`, `openai`, `sft`, `loop` |
| `test_lossless_context_runtime` | Recoverable projection | `triggers` |
| `test_robotwin_motion_v2` | Motion execution | `grasp`, `probe`, `profiles`, `sparse`, `surface` |

Each child has its own manifest contract. Shared fixtures and
constants retain a single owner in the base module; children explicitly import
what they use. Existing external helper imports continue to work. Test bodies
and parametrization come from the current development candidate, including its
new Provider and Context regressions, rather than from older split files.

For another split, compare the complete test-function/decorator AST set and
pytest case collection before and after. Exercise cross-module helper consumers
as well as the moved tests. A collection count alone cannot detect replacement
of one test by another. Do not duplicate constants into children: an asset update
must not leave independent copies of its expected hash behind.

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

`test_module_size_ratchet.py` governs product Python modules, not test-file
length. Names and docstrings describe the executable test; the manifest states
the module contract. Prefer behavior checks over source-layout assertions unless
the source itself is the artifact under test, such as delivered prompt text.
