# Code map

| Owner | Location |
|---|---|
| Plan resolution, defaults and source identity | `ora/orchestrator/spec.py`, `defaults.py`, `spec_source.py` |
| CLI, admission, leases and supervision | `ora/orchestrator/cli.py`, `preflight.py`, `lease_claim.py` |
| Terminal watch, interactive sessions and local CPU entry | `ora/agent/native/tui/` and `ora-tui/` (transcript and Pi front end); `ora/orchestrator/ora_cli.py`, `ora_backend.py`, `ora_catalog.py`, `operator_control.py`, `ora_local.py`, `ora_local_setup.py`, `ora_world.py`, `ora_remote.py`; `viewer/live_world.py`; `ora/agent/native/operator_messages.py` |
| Episode coordination and official evaluation | `ora/orchestrator/coordinator.py`, `evaluation.py` |
| Native policy lifecycle and provider/tool loop | `ora/agent/native/session.py`, `tool_loop.py` |
| Provider protocols and credentials | `ora/agent/providers/` |
| Policy composition, embodiment and tool contracts | `ora/agent/composition/` |
| Context projection, compaction and delivery | `ora/agent/context/` |
| Recoverable context artifact storage | `ora/agent/memory/` (historical package name; no Task Memory) |
| Policy-visible tools and closed skills | `ora/agent/tools/` |
| Resident perception services | `ora/agent/model_services/` |
| Trace and policy-state evidence | `ora/agent/evidence/` |
| Action envelopes and shared validation | `ora/contracts/` |
| Transport-neutral sessions and workers | `ora/environment/` |
| Host-safe benchmark identities and task catalogs | `ora/environment/benchmarks/` |
| Host-safe controller contracts | `ora/environment/controllers/` |
| Simulator implementations | `ora/environment/backends/{robotwin,libero_pro,robodojo}/` |
| Versioned prompt/tool/environment assets | `ora/policy/`; integrity in `policy.lock` |
| Runtime qualification ladder | `experiments/runtime_v1/` |
| One-cell and official-suite examples | `examples/`, `benchmark_spec.py`, `robodojo_spec.py` |
| Read-only run registry and media inspection | `viewer/` |

Use the explicit canonical subpackages for imports. The package root does not re-export implementation classes. Add shared behavior to its owner rather than to a launcher or benchmark-specific example. [Contributor instructions](../AGENTS.md) describe the required checks.
