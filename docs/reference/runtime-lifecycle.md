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

With `operator_control`, the episode loop checks the run's inbox before each Native model call. A steering message enters the conversation at that boundary; it cannot alter a request in flight, execute a tool or override Supervisor admission. An abort uses the normal stop path below.

Budgets bound decisions, model calls, simulation work and wall time. A persistent multi-round session must be selected explicitly. Cross-episode Task Memory is disabled; recoverable context artifacts remain episode-local evidence.

## Land, inspect and seal

The environment's official boundary supplies success: RoboTwin `check_success`, LIBERO termination or RoboDojo's task verdict. Synthetic reports `official_success: null`.

On landing the supervisor derives metrics and context audits, evaluates evidence and attempts sealing. An invalid trace receives a seal refusal rather than a fabricated seal. Provider request storage can be compacted losslessly after sealing. Resource leases and per-run scratch are released for this exact run. The Supervisor belongs to the process (or cluster job step) that launched it; an interactive client reports the run finished only after the Supervisor has exited, so keep that process open until then.

`ora status`, `watch`, `metadata`, `state`, `metrics` and `ownership` inspect recorded state. `ora stop RUN_ID` verifies the selected run's process identity before signaling it. Retry is restricted to an unsealed incomplete attempt before any robot action was dispatched. Sealed evidence is never rewritten by revalidation; reports are written separately.

[Configuration and evidence](configuration-and-evidence.md) describes paths and identities. The [CLI reference](../guides/cli-reference.md) lists every command.

## Single-GPU entry

`ora run --env libero|robotwin|robodojo --task TASK` (`ora_run`) runs these
stages for one task on the shipped `single-gpu` host and stops at the first
failure:

1. `ora_env_file.load_into` loads `.env` without overriding or blanking the
   process environment; the host marker is bound for this process only (a
   different marker is refused), an unset `DEEPSEEK_BASE_URL` defaults to the
   profile endpoint and `preflight.provider_marker_configuration` runs offline.
   A dirty checkout is refused.
2. `ora_run_qualification` decides reuse with preflight's own validators:
   `benchmark_spec.admitted_seed_manifest` for a RoboTwin seed manifest, and
   `robodojo_qualification.load_qualification` / `load_card_qualification` for
   a RoboDojo card/task pair, plus GPU 0's `nvidia-smi` UUID and the worker
   interpreter readback. Missing records are produced by `robotwin_evaluation
   prepare/qualify` or `robodojo_lane_qualification card/task` after an
   explicit confirmation (`--yes` off a terminal).
3. `ora_run_services.ensure_services` starts the resident services only when
   no listener is ready and binds the newest readiness report whose recorded
   service health equals the running instances and is under 20 h old
   (`REPORT_REUSE_MAX_AGE_S`), else writes a new one. It runs after stage 2, so
   a long scan cannot age the report out.
4. The RunSpec has experiment `ora-interactive`, the host's `<env>-gpu0`
   slot, the readiness report and a new single-attempt cohort
   `ora-run-<env>-<UTC time>`. `ora interactive` repeats full preflight and
   launches only with `--confirm-launch` and a picker selection.

`ora setup --env` (`ora_gpu_setup`) prepares the inputs: a step is done when
its real state holds (a clean checkout at the pinned revision) or its marker
records the same recipe digest; package markers also bind the target
virtualenv's `pyvenv.cfg`, so a recreated environment is reinstalled. RoboDojo's
`graphics-compat` is done when its receipt exists. Each command runs in its own
process group, killed as a group on timeout.

## RoboDojo physics device

`ora.environment.benchmarks.robodojo_physics` maps each reviewed task to its physics device. Articulation, conveyor and fluid scenes in its CPU table use CPU physics; other reviewed scenes use `cuda:0`. Rendering still has its own bound GPU. Composers, scene probes and qualification must use the same task/device/layout mapping; a rigid CUDA scene's qualification does not establish CPU-scene correctness.
