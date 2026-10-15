# Native context assembly

The Native policy owns a canonical message history and translates it at the provider boundary. ORA supplies project instructions, the task, robot manual, advertised tools, current public state and selected tool history. No managed coding-agent scaffolding is part of this distribution.

## Composition

Policy layers are source files bound by path and hash in the RunPlan. The typed v3 context channels are `harness_system`, `project_instructions`, `episode_task` and `tool_manual`. The retained legacy context channels are `codex_project`, `episode_prefix` and `tool_manual`; the first name remains a serialization field and does not select a Codex adapter.

The default robot compositions use the versioned Native causal slots. `native_slots_v4` adds protected round static facts to v3. The active baseline catalog determines the prompt bundle, ToolSurface, embodiment and context retention for each environment. Resolution validates their compatibility together.

## History and artifacts

Context projection retains causal action/observation relationships. Large tool results and media can be stored as recoverable context artifacts, with content hashes and bounded readers. The lossless retention policy distinguishes recoverable evidence from what is presently inlined into a request. Compaction and rollover are recorded, including the exact delivered request and usage evidence.

The context budget must leave room for model output. Settings that alter projection, protected facts, thresholds or tool delivery change model-visible behavior and require a deliberate comparison. Do not silently summarize away a refusal's constraints or an action's units.

## Code and verification

- Composition: `ora/agent/composition/`, `ora/orchestrator/spec_composition.py`.
- Projection and request assembly: `ora/agent/context/`, `ora/agent/native/request_assembly.py`.
- Artifact catalog and readers: `ora/agent/memory/artifacts.py`, `artifact_reader.py`.
- Delivery trace: `ora/agent/evidence/trajectory.py`, `ora/orchestrator/context_manifest.py`.

Use `ora context RUN_SPEC` to inspect the resolved context without launching. `scripts/dev/compare-model-visible BASE CANDIDATE` compares advertised tools, prompt text, embodiment facts, observation projection and settings. A package/schema migration can change identities without changing task behavior; report those changes explicitly rather than calling the raw digests equal.
