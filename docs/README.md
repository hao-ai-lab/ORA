# ORA documentation

<p class="ora-intro">Run a language model as a robot policy. ORA supplies the robot manual, observation tools, supervised actions, and a record of every episode.</p>

<p class="ora-meta">Open Robot-Use Agents · Python 3.11 · OpenAI-compatible and Anthropic APIs</p>

<div class="ora-next" markdown>
[Run your first episode](quickstart.md){ .md-button .md-button--primary }
[Choose a robot environment](environments/README.md){ .md-button }
</div>

## Start here

<div class="grid cards" markdown>

-   **Getting started**

    Install the locked Python environment and connect your model API. The first
    episode uses Synthetic: no simulator, model download, or GPU.

    [Installation →](getting-started/installation.md)

-   **Robot environments**

    Set up LIBERO / LIBERO-Pro, RoboTwin, or RoboDojo in an isolated simulator
    environment, then compose and run one task.

    [Environment guides →](environments/README.md)

-   **Runs and evidence**

    Inspect the plan, model calls, executed actions, official result, and seal.
    Browse trajectories in the local read-only viewer.

    [Run and inspect →](usage/runs.md)

-   **Build on ORA**

    Read the architecture and Python interfaces. Extend a backend, tool surface,
    or prompt bundle using the authoring guides.

    [Architecture →](architecture.md) · [Python API →](api/README.md)

</div>

## First episode

From a source checkout, install ORA and check the environment:

```bash
uv sync --all-extras --locked
uv run --locked ora doctor
```

Set `ORA_API_KEY` in your shell, then run:

```bash
uv run --locked ora run --env synthetic
```

This calls a real model and saves a sealed episode under
`outputs/quickstart/runs/<run-id>/`. Synthetic verifies the agent/runtime
connection; its `official_success` is `null`.

[Complete quickstart and API configuration](quickstart.md)

## Choose an environment

| Environment | Robot / simulator | Guide |
|---|---|---|
| LIBERO / LIBERO-Pro | Panda manipulation in MuJoCo | [Install and run](environments/libero.md) |
| RoboTwin | Dual-arm manipulation in SAPIEN | [Install and run](environments/robotwin.md) |
| RoboDojo | Isaac Sim task environments | [Install and run](environments/robodojo.md) |

The simulators, assets, perception weights, and GPU requirements are installed
separately from ORA. [Supported setups](support.md) records the tested scope.

## Find what you need

| I want to… | Read |
|---|---|
| Use a self-hosted model or Anthropic | [Model APIs](getting-started/model-apis.md) |
| Fix an installation or run problem | [Troubleshooting](getting-started/troubleshooting.md) |
| Inspect a run or export evidence | [User guide](usage/runs.md) |
| Understand the action and trust boundaries | [Runtime lifecycle](reference/runtime-lifecycle.md) |
| Configure a benchmark task | [Configuration](reference/configuration-and-evidence.md) |
| Change tools or robot instructions | [Tool surfaces](guides/tool-surface-authoring.md) · [Prompt bundles](guides/prompt-bundle-authoring.md) |
| Contribute or maintain these docs | [Contributing](../CONTRIBUTING.md) · [Documentation guide](developer/documentation.md) |
