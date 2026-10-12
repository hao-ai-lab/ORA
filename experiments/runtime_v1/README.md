# Runtime v1 qualification ladder

`runtime_v1` checks the shared ORA Runtime from Synthetic M0 through RoboTwin M5. It is a development qualification example, not a behavioral benchmark matrix. It uses the Native API policy and the shared default `deepseek_flash_openai_low_16k_v1` provider registration; no managed coding-agent runtime is required.

| Milestone | Task | Scope |
|---|---|---|
| M0 | `synthetic_tool_boundary` | In-process observation/action/receipt; official success is null. |
| M1 | `click_alarmclock` | One safe upward motion through the isolated worker; infrastructure only. |
| M2 | `click_alarmclock` | Diagnostic button interaction on qualified seeds. |
| M3 | `open_laptop` | Behavioral articulated-object interaction. |
| M4 | `place_a2b_right` | Behavioral grasp and placement. |
| M5 | `open_microwave` | Final behavioral acceptance. |

`experiment.yaml` owns tasks, seed choices, budgets, source pins and prompt layers. `run.py` composes specs and delegates to the shared CLI. Generated specs, gates, seed records and episodes live under ignored `outputs/harness-runtime-v1/`, or its selected outputs root.

## Inspect without launching

```bash
uv run --locked python experiments/runtime_v1/run.py configs
uv run --locked python experiments/runtime_v1/run.py explain M0
uv run --locked python experiments/runtime_v1/run.py context M0 --policy native_api
```

Context preview does not require milestone gates or local seed qualification. It is not evidence that the host or provider is ready. For the shortest real-model demonstration, use the [Synthetic quickstart](../../docs/quickstart.md).

## Qualify and run

For RoboTwin, install the [pinned simulator](../../docs/environments/robotwin.md), configure a registered execution host and the selected provider's environment variables. M2–M5 require local trusted-expert seed qualification on that source and host. This can start simulator jobs; inspect the command and capacity before requesting it:

```bash
uv run --locked python experiments/runtime_v1/run.py qualify M2
```

Choose a seed accepted by the resulting manifest and also declared for that milestone. Historical seed summaries do not qualify a new installation.

Each milestone first runs with `scripted`, then `native_api`. A completed evaluator-valid sealed scripted run opens Native at the same milestone; a Native gate opens the next milestone. Gate records bind the exact source commit. Physical seed qualification is a separate prerequisite; do not confuse a stale milestone gate with invalidation of all physical evidence.

```bash
uv run --locked python experiments/runtime_v1/run.py launch M0 --policy scripted
uv run --locked python experiments/runtime_v1/run.py seal SCRIPTED_RUN_ID
uv run --locked python experiments/runtime_v1/run.py launch M0 --policy native_api
uv run --locked python experiments/runtime_v1/run.py seal NATIVE_RUN_ID
```

Only a valid completed Native M5 run with `official_success: true` records final behavioral acceptance. Lower infrastructure gates may be valid with a false behavioral verdict. The source and run identities of previous evidence remain unchanged.

The extra `control-audit` and `libero-pose-control` commands support specific controller diagnostics; inspect `--help` and their module contracts before allocating a simulator. They are not required by the Synthetic quickstart.
