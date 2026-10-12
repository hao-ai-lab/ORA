# First episode

You need Git, uv, Python 3.11 and a model API key. ORA installs its Python dependencies from `uv.lock`; this example downloads no model or simulator assets and uses no GPU.

```bash
git clone https://github.com/hao-ai-lab/ora-agent.git && cd ora-agent
uv sync --locked
uv run --locked ora doctor
```

The repository must be clean: launches bind the exact source commit. Keep local configuration in shell variables or an ignored `.env` file; ORA does not automatically source that file.

## Model API

For the default OpenAI endpoint, set `ORA_API_KEY` or `OPENAI_API_KEY` in your shell. The default model is [gpt-4.1-mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), which accepts image inputs and tool calls.

For another OpenAI-compatible API, set these non-secret values and supply its key as `ORA_API_KEY`:

```bash
export ORA_BASE_URL=https://your-model-server.example/v1
export ORA_MODEL=your-served-model-id
export ORA_PROVIDER=openai_compatible
```

A local HTTP server is accepted only on loopback. Remote endpoints require HTTPS. For a private certificate authority, set `SSL_CERT_FILE` to a trust bundle containing that CA. Do not disable certificate verification.

For Anthropic Messages, set `ORA_API_KEY` or `ANTHROPIC_API_KEY`, plus:

```bash
export ORA_PROVIDER=anthropic_messages
export ORA_BASE_URL=https://api.anthropic.com
export ORA_MODEL=your-available-model-id
```

`ORA_REASONING_EFFORT` optionally selects the provider's reasoning setting. A text-only endpoint can use `ORA_SUPPORTS_IMAGES=0` for Synthetic; the robot benchmarks need the observation capabilities their selected surface declares. Keys are read from the environment and never stored in RunSpecs.

## Run

```bash
uv run --locked ora run --env synthetic
```

The model reads a robot manual and moves a Synthetic end effector once. The normal resolver, preflight, resource lease, supervisor, evaluator and seal are used. Running this command explicitly starts the bounded demonstration; it does not require a separate confirmation prompt.

The episode has a 120-second budget, at most six model calls, and at most 2,048 output tokens per call. Actual latency and input-token use depend on your provider and the tool calls it chooses. The command prints the episode directory and call count. It exits successfully only after a completed episode has been sealed.

Look under `outputs/quickstart/runs/<run-id>/` for `plan.json`, `result.json`, `seal.json`, canonical events and archived provider evidence. `official_success: null` is expected: Synthetic tests infrastructure and has no benchmark success predicate.

```bash
./scripts/open-viewer
```

The viewer discovers the saved run. [Choose a robot environment](support.md) after this example works.

## Troubleshooting

| Symptom | Check |
|---|---|
| Missing key / 401 | Export the key expected by the chosen provider. `ora doctor` reports presence only; it does not authenticate. |
| 429 or provider unavailable | Check the provider's capacity and billing. The run remains bounded; inspect its `supervisor.log`. |
| Source is dirty | Commit intentional source changes; keep credentials and generated files ignored. |
| Context overflow | Use a model with the declared window or reduce the task/context in a separate RunSpec. This quickstart declares 32,768 tokens. |
| Episode appears stalled | Inspect the printed run directory and `ora status`; `ora stop <run-id>` stops that exact run. |
| No GPU, EGL or simulator | None is needed here. For robot episodes, run `ora doctor --env <environment>` and follow that environment's installation page. |

CI runs the same CLI against a deterministic local HTTP model with `python examples/quickstart_smoke.py`; it does not spend API credits. This is separate from acceptance against a real model API.
