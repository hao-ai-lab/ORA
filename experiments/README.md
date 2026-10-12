# Experiment packages

`experiments/` holds the one maintained experiment package, the shared Runtime
qualification ladder. It is not a second Runtime implementation tree and not a
registry of studies: reusable behavior belongs under `ora/`, and
generated runs, services, Store state and videos stay under ignored `outputs/`
workspaces.

| Package | Role |
|---|---|
| `runtime_v1/` | M0–M5 qualification for the shared Runtime; not a behavioral study matrix. See its [README](runtime_v1/README.md). |

The research study and archive packages that used to live here were removed
from this tree. The official-protocol LIBERO-Pro and RoboTwin suite launcher is
the example [`examples/official_suite/`](../examples/official_suite/README.md).

[Experiment identity](../docs/reference/experiment-identity.md) separates
experiment families, branches, generations and exact runs; the lineage catalog
is `ora/orchestrator/identity-registry.json` (shipped empty; register
your own families there).

## Lifecycle rules

- A study that changes prompt, tools, observation projection, context policy,
  sandbox, session cadence or execution semantics starts on its own branch with
  its own package and output workspace.
- Run artifacts bind the exact experiment commit.
- Never import experiment code from maintained `ora/` modules.
- Generated evidence never merges into Git.
