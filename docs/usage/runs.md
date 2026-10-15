# Run and inspect

Start with the [Synthetic first episode](../quickstart.md), or compose a robot
RunSpec using an [environment guide](../environments/README.md). Operator
commands share the plan, preflight, Supervisor, evaluation, and seal boundary.

## Inspect before launching

From the repository root:

```bash
./scripts/ora validate path/to/run-spec.yaml
./scripts/ora plan path/to/run-spec.yaml
./scripts/ora context path/to/run-spec.yaml
```

These inspect the specification, resolved plan, and policy context. The
[CLI reference](../guides/cli-reference.md) describes preflight and launch,
including live provider checks and qualification. Synthetic's bounded quickstart
performs its normal launch sequence.

## Follow one run

Replace `RUN_ID` with the exact ID printed at launch:

```bash
./scripts/ora status RUN_ID
./scripts/ora watch RUN_ID
./scripts/ora metrics RUN_ID
```

`ora watch` shows the run as a terminal transcript: what the model said,
which tools it called with condensed results, refusals, action receipts and
each round's official verdict. It follows a live run and replays a finished
one on recorded time (`--speed 0` prints it at once). For the pi-tui front
end, install Node 20+ and run `npm ci` in `ora-tui/` once; without it `ora
watch` uses its built-in terminal view. Both views head the transcript with
ORA's mark, a pixel robot arm, in teal (`#4ecdc4`), the accent that also marks
the spinner and the `▸▸` footer. `ora watch RUN_ID --raw` prints the
canonical causal events as JSON lines instead.

A live `ora watch` in a terminal also opens a read-only browser window bound
to the run's recorded camera frames (`--world-view off` disables it; a
finished-run replay leaves it off). When the run executes on another machine,
`./scripts/ora-remote HOST --repo /path/on/HOST -- watch RUN_DIR` runs ORA there
over SSH and forwards the world view to a local browser. To steer a run while
it works, see [interactive sessions](interactive.md).

Use `ora stop RUN_ID` to stop that active run. Avoid broad process kills that
could stop unrelated sessions or services.

## Read the evidence

| File / directory | Records |
|---|---|
| `plan.json` | Resolved configuration and source/asset identities |
| Canonical events and provider evidence | Requests, replies, tools, proposals, and receipts |
| `result.json` | Episode outcome and counters |
| `evaluation.json` | Deterministic validation and environment verdict |
| `seal.json` | Hash-bound inventory of accepted evidence |

Synthetic lands under `outputs/quickstart/runs/<run-id>/` by default. Robot
placement follows the workspace and execution namespace. See
[configuration and evidence](../reference/configuration-and-evidence.md).

## Open the viewer

```bash
./scripts/open-viewer
```

The local read-only viewer joins the plan, conversation, trajectory, original
media, and official verdict. Diagnostic reconstructed media is labeled separately.

[Viewer guide](../../viewer/README.md)

## Export a sealed run

```bash
./scripts/ora trace-validate RUN_ID
./scripts/ora export RUN_ID
./scripts/ora export-sft RUN_ID
```

The transfer archive and training projection serve different purposes.
Canonical evidence and the seal remain authoritative; export does not change
the verdict or original evidence.
