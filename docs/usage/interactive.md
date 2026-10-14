# Interactive sessions

`ora interactive` lets you pick a prepared benchmark task, watch ORA work on
it, and send it a message while it runs, the way you would talk to a coding
agent. `ora local` (see [Run ORA on your computer](../getting-started/local.md))
opens the same interface for its local task.

```bash
./scripts/ora interactive --spec path/to/run-spec.yaml --confirm-launch
```

Repeat `--spec` to offer several tasks. Each must be a Native RunSpec for
LIBERO, RoboTwin or RoboDojo. The selected task keeps its benchmark
instruction, official evaluator, seed or layout, provider, resources and
policy; a typed message never replaces the task. Selecting a task and passing
`--confirm-launch` are both required before the normal preflight and launch.

## Keys

| Key | Effect |
|---|---|
| ↑/↓, Enter in the picker | Select and start a task |
| Enter in the editor | Steer the active run: the message reaches the model before its next call |
| Alt+Enter | Queue a follow-up for the next task you start |
| Tab | Switch between the picker and the editor when no run is active |
| Esc | Stop the active run through the normal stop path |
| Ctrl-C / Ctrl-D | Leave the interface; the run keeps going |

Queued, delivered and rejected messages are shown above the editor. A message
is delivered at a model-call boundary: it never interrupts a request in
flight or moves the robot directly.

## What a message changes

A run that received a message is human-assisted. Its official verdict is
kept, but it is marked `human_intervention: true` with an
`operator_message_count`, excluded from autonomous success rates, and not used
for automatic learning or SFT export. A stopped run ends `incomplete` with
the operator as termination authority. Use `ora watch RUN_ID` to inspect any
run without being able to send input.

## Finishing

After the environment ends the episode, ORA writes metrics, evaluates, seals
and releases the run's resources. The status line shows `Landing…` until
that is done. Keep the terminal (and, on a cluster, the job step that runs
ORA) open until the run is reported finished: closing it earlier can stop
the landing midway.
