# Run ORA on your computer (CPU)

`ora local` runs one LIBERO task on your own machine with a CPU MuJoCo
simulator: no GPU, CUDA, CuRobo or perception service. The model is called
through the DeepSeek API, so you need an API key and network access. Your
provider bills the calls.

This is a reduced composition for trying ORA, not the benchmark setup. Its
tools are bounded end-effector moves, gripper open/close, RGB-D observation and
point marking. It has no motion planner, segmentation or learned grasp service,
so do not compare its results with the full server composition.

## Requirements

- Git, [uv](https://docs.astral.sh/uv/getting-started/installation/) and
  Node.js 20 or newer (with npm) on `PATH`.
- A Python 3.11 interpreter. `ora setup` does not download one; install it
  with `uv python install 3.11` if needed.
- macOS 13 or newer on Apple Silicon (native OpenGL), or Linux with the
  system OSMesa library for offscreen software rendering.
- About 2 GB of disk for the pinned LIBERO sources, assets and the simulator's
  own Python environment.

## Install

From a clone of this repository:

```bash
uv sync --all-extras --locked --python 3.11
./scripts/ora setup
```

`ora setup` downloads the pinned LIBERO-Pro code and task data, creates an
isolated CPU simulator environment and installs the terminal UI (`ora-tui/`).
It starts no simulator and calls no API. `--root PATH` (or `ORA_LOCAL_ROOT`)
chooses the installation directory; the default is your user data directory.

## Check

```bash
export DEEPSEEK_LOW_API_KEY=...        # DEEPSEEK_API_KEY is accepted too
./scripts/ora doctor --env local
```

`doctor --env local` checks the sources, simulator environment, graphics
binding, credential presence and terminal UI without starting the simulator or
calling the API. It prints what is missing and the next command.

## Run

```bash
./scripts/ora local --confirm-launch
```

The task picker opens with LIBERO `libero_goal` task 0, seed 0 ("open the
middle drawer of the cabinet"); press Enter to start it. `--suite`,
`--task-id` and `--seed` select another original LIBERO task. Without
`--confirm-launch` you can browse but not start a task.

A browser window follows the run's recorded camera frames: the scene stays
still while the model thinks and moves only during robot actions. While the
run is active you can steer it from the terminal; see
[interactive sessions](../usage/interactive.md). Runs land under
`outputs/ora-interactive/runs/`.

## Status

This path is new. Platform selection and the composition are covered by
offline tests; it has not yet been validated end to end on a Mac. Please
report the output of `./scripts/ora doctor --env local --json` with any issue.
