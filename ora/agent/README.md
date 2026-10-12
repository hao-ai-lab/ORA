# Agent package map

`ora.agent` contains the Native policy runtime, a scripted policy for
tests and the quickstart, and the shared capabilities they compose.

## Executable policy runtimes

| Runtime | Owner |
|---|---|
| Harness-owned direct provider loop | `native/` |
| Scripted action list | `scripted.py` |

`providers/` contains direct provider adapters used by Native; it does not own
a Session lifecycle.

`providers/attempts.py` supplies redacted transport-failure records without
filesystem or Session access. `native/provider_calls.py` binds those records to
the active request and persists failed-attempt evidence before retry backoff.
The normal tool loop uses this recorder; partial responses never enter its tool
dispatcher.

## Shared capability domains

| Directory | Responsibility |
|---|---|
| `composition/` | Versioned prompt, tool and embodiment composition |
| `context/` | Context inventory, projection, pressure and epoch lifecycle |
| `memory/` | Recoverable Context Artifact storage, readers and openers (the package name predates the removal of Task Memory) |
| `model_services/` | Isolated resident-model services shared by experiments |
| `tools/` | Policy-visible tool implementations |
| `evidence/` | Policy-state and trajectory formats |
| `sessions/` | Shared managed-session state helpers |
| `wire/` | Managed protocol capture and export |

Flat `active_memory.py`, `file_memory.py`, `memory.py` and
`recoverable_artifacts.py` modules are retired. Recoverable Artifact storage
lives in `agent/memory/`. There is no Task Memory and no post-round or
post-session Agent-state phase.

The three runtimes intentionally remain siblings for now.


`context/action_views.py` owns post-action image selection across all three
policy runtimes. `evidence/motion_feedback.py` projects measured effects
without importing simulator backends. The environment owns actual execution.
