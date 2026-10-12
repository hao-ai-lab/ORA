# ORA: Open Robot-Use Agents

ORA runs a language model as a robot policy: it reads a manual, inspects public observations, and proposes actions through a supervised tool interface.

[Documentation](docs/README.md) · [Quickstart](docs/quickstart.md) · [Supported environments](docs/support.md) · [Contributing](CONTRIBUTING.md)

## Try it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/hao-ai-lab/ora-agent.git && cd ora-agent
uv sync --locked
uv run --locked ora doctor
```

Set `ORA_API_KEY` in your shell, then run:

```bash
uv run --locked ora run --env synthetic
```

This calls a real model API and saves a sealed episode under `outputs/quickstart/runs/`. It needs no simulator, model download or GPU. The default endpoint is OpenAI Chat Completions with `gpt-4.1-mini`; [the quickstart](docs/quickstart.md) explains other OpenAI-compatible and Anthropic endpoints, including self-hosted models. API usage is billed by your provider.

Synthetic checks the agent/runtime connection. It has no manipulation benchmark score: `official_success` is `null`.

## Choose an environment

| Environment | Install and run | Verdict |
|---|---|---|
| LIBERO / LIBERO-Pro | [Panda installation and one episode](docs/environments/libero.md) | Official LIBERO termination |
| RoboTwin | [Dual-arm installation and one episode](docs/environments/robotwin.md) | Official `check_success` |
| RoboDojo | [Isaac installation and one episode](docs/environments/robodojo.md) | Official task success |

Each simulator has its own interpreter. Torch, simulator packages, robot assets and perception weights are installed separately from ORA's locked Python environment. See the [support matrix](docs/support.md) for the tested configurations.

## How it works

The Native agent connects directly to your model API. Prompt bundles deliver the task, robot manual and tool contracts. Tools expose observations and bounded action proposals; the Supervisor checks and dispatches each mutation. The simulator supplies the task verdict. ORA records requests, responses, actions and receipts for inspection and export.

[Architecture](docs/architecture.md) · [Current interfaces](docs/reference/README.md) · [Evidence viewer](viewer/README.md)

There are no managed coding-agent adapters, VLA services or cross-episode Task Memory in this distribution. Recoverable context artifacts retain large tool results within an episode.

## Develop

```bash
uv sync --all-extras --locked
uv run --locked python -m pytest tests
./scripts/ora lock --check
```

The distribution is named `ora-agent`; the Python import and command are `ora`. Use a source checkout for launches so plans can bind a clean Git commit. A PyPI release and the accompanying blog are pending.

To install with a coding agent, give it this instruction:

> Read AGENTS.md and docs/quickstart.md. Install the locked ORA environment, run ora doctor, and help me configure my existing model API without printing or committing credentials. Run the Synthetic quickstart before installing a simulator.

Licensed under [Apache-2.0](LICENSE). Bundled third-party code retains the notices in [NOTICE](NOTICE).
