# Public interfaces

Import from the explicit `ora` subpackages. There are no flat compatibility re-exports or managed Codex/Claude Code adapters.

## Operator and composition

| Interface | Owner | Contract |
|---|---|---|
| `ora` | `ora.orchestrator.cli` | Validate, plan, preview context, preflight, launch, inspect, evaluate, seal and export. |
| `ora doctor` | `ora.orchestrator.doctor` | Read-only Python/distribution, credential-presence, interpreter/asset and graphics-library discovery. Never prints key values or starts a simulator. |
| `ora run --env synthetic` | `ora.orchestrator.quickstart` | Bounded Native API demonstration through normal launch gates; waits for completed, sealed evidence. |
| `load_run_spec`, `resolve_run_spec`, `validate_run_plan` | `ora.orchestrator.spec` | Source/asset-bound plan resolution and persisted plan validation. |
| `resolve_tool_surface`, `PROFILE_CATALOG` | `ora.agent.composition` | Registered tool definitions, schemas, roles, capabilities and effect contracts. |
| Default baseline catalog | `ora.agent.composition.baseline`, `ora.orchestrator.defaults` | Select complete maintained compositions for fresh robot consumers. |
| `run_episode` | `ora.orchestrator.coordinator` | Execute a resolved plan with policy/environment sessions; operator launches use the supervised CLI admission path. |

`benchmark_spec` composes one RoboTwin, LIBERO or LIBERO-Pro cell; `robodojo_spec` composes a reviewed RoboDojo cell. `examples/official_suite` demonstrates suite expansion. These select task/provider/resources and use the default resolver for policy composition.

## Policy and tools

`ora.agent.contracts` defines policy-session contracts. `ora.agent.native` implements the direct API lifecycle and loop. `ora.agent.providers` translates canonical OpenAI-shaped requests at the OpenAI-compatible or Anthropic Messages wire boundary. Credentials are environment inputs, not RunSpec values.

Tool runtime ownership is `ora.agent.tools`. A ToolSurface is the authority for advertised names and schemas; callers must not infer tools from module names. Observation, perception and geometry tools are read-only. Mutation tools submit proposals through the host mutation port and Supervisor. Composed tools are checked/compiled from the versioned library; their compiled identities must match their sources.

`observation.dump_state` reads public current/historical state. `robot.submit_action` is the generic mutation boundary; selected surfaces use their declared motion/gripper/skill names without bypassing supervision. Argument refusals include their public constraints. Grasp/placement helpers may provide derived geometry, never privileged simulator labels.

`ora.agent.context` owns context projection and delivery. `ora.agent.memory` retains recoverable context artifact storage under its historical package name; it provides no task-file or cross-episode Memory service.

## Environment boundary

`ora.environment` defines transport-neutral sessions, descriptors, registry and Synthetic. The `local_rpc` worker uses the selected simulator's explicit interpreter. Backend implementation lives in `ora.environment.backends.robotwin`, `libero_pro` or `robodojo` and is loaded in the isolated worker.

`ora.environment.benchmarks` holds host-safe task identities, source versions and suite expansion. `ora.environment.controllers` holds shared controller/safety contracts and source-bound profiles. Simulator modules do not belong in host-safe planning or action-contract imports.

Only the environment's official boundary supplies the boolean task verdict. Synthetic has no such boolean. A model's `finish_episode` or natural-language claim cannot override official evidence.

## Evidence and viewer

`ora.agent.evidence` owns canonical request/response, tool, action, state and artifact records. `ora.orchestrator.evaluation` validates and seals a run; `revalidation` writes independent later reports. Exported SFT conversations are derived training projections, not the canonical trace.

The viewer reads indexed runs, trajectories and media. Original captured images/video and reconstructed replay are distinct evidence types. Rendering or cache reconstruction never changes the recorded verdict or seal. [Viewer documentation](../../viewer/README.md) describes the local read-only interface.

Schema identifiers use `ora.*`. The distribution is `ora-agent`; Python imports and commands use `ora`. Older evidence namespaces are not silently aliased by this release.

## Structured tool errors and repeated identical requests

Public refusals identify the failed stage and stable code, expose the allowed public constraint and describe recovery supported by the interface. Private paths, object state and exception stacks remain outside policy feedback. Retrying an identical invalid request does not make it admissible; the Native loop records tool results and recovery evidence without pretending a refused proposal executed.

## Tool description consistency

Canonical tool help, closed input schemas, public result schemas and actual handlers must agree. Help states when to select the tool, effects, immediate/deferred evidence, limits, non-guarantees and recovery. Both provider projections derive from those same definitions. `compute` performs admitted pure geometry/calculation; policy-visible numeric conversion must not rely on undocumented mental arithmetic. See the ToolSurface guide and description consistency tests.

## Published limits and stated refusals

Every numeric constraint in help/refusals has a source owner and actual consumer. Publish the same units, frame, inequalities and applicable conditions that the admission check enforces. Derived values must be computed by the responsible helper. A missing or uncalibrated quantity remains unknown; do not present nominal geometry as a measured fact or weaken an execution gate to match prose.
