# Runtime lifecycle

## Plan

`load_run_spec` and `resolve_run_spec` in `ora.orchestrator.spec` validate inputs, select defaults, bind assets and produce a RunPlan. A plan binds the exact source commit, policy composition, benchmark revisions, provider settings, resources and budgets. `validate_run_plan` checks persisted identity before launch.

The Native policy sees public observations, task instructions, robot manuals and advertised tools. It never receives object ground truth, contact state, task source, expert code or the private success predicate. Perception results are computed from public camera data.

## Admit and launch

`ora preflight RUN_SPEC` checks the clean source, execution host, simulator/runtime, provider, resident services, resources and required qualification. `ora launch RUN_SPEC` repeats those checks and requires the plan's confirmation code; `--yes` records explicit non-interactive confirmation.

The launch path atomically claims the exact execution resource and any bound cell admission. Qualified simulator slots may coexist when their source-tracked resource bindings allow it; unprofiled plans are host-exclusive. The supervisor rechecks source identity before opening sessions. A newer Git commit alone does not establish or invalidate physical qualification: its declared dependencies determine reuse.

`ora run --env synthetic` invokes that same launch path for a bounded simulator-free demonstration. The command itself authorizes this one Synthetic episode; it does not launch a benchmark task.

## Execute

The episode loop opens one policy session and one environment session. Native requests and tools use the resolved contracts. Read-only tools cannot mutate robot state. Every mutation crosses the Supervisor as a versioned action proposal before the environment executes it. The environment returns measured receipts and fresh observations; the policy cannot turn an attempted action into an official success claim.

The v3 atomic action envelope separates pose motion from gripper motion. Target-state v4 accepts pose/gripper independently and stages a combined proposal as pose-first, conditional-gripper execution. Motion-action v3 supports the declared motion interface. Physical contact can change measured state without itself being an infrastructure fault.

Budgets bound decisions, model calls, simulation work and wall time. A persistent multi-round session must be selected explicitly. Cross-episode Task Memory is disabled; recoverable context artifacts remain episode-local evidence.

## Land, inspect and seal

The environment's official boundary supplies success: RoboTwin `check_success`, LIBERO termination or RoboDojo's task verdict. Synthetic reports `official_success: null`.

On landing the supervisor derives metrics and context audits, evaluates evidence and attempts sealing. An invalid trace receives a seal refusal rather than a fabricated seal. Provider request storage can be compacted losslessly after sealing. Resource leases and per-run scratch are released for this exact run.

`ora status`, `watch`, `metadata`, `state`, `metrics` and `ownership` inspect recorded state. `ora stop RUN_ID` verifies the selected run's process identity before signaling it. Retry is restricted to an unsealed incomplete attempt before any robot action was dispatched. Sealed evidence is never rewritten by revalidation; reports are written separately.

[Configuration and evidence](configuration-and-evidence.md) describes paths and identities. The [CLI reference](../guides/cli-reference.md) lists every command.

## RoboDojo physics device

`ora.environment.benchmarks.robodojo_physics` maps each reviewed task to its physics device. Articulation, conveyor and fluid scenes in its CPU table use CPU physics; other reviewed scenes use `cuda:0`. Rendering still has its own bound GPU. Composers, scene probes and qualification must use the same task/device/layout mapping; a rigid CUDA scene's qualification does not establish CPU-scene correctness.
