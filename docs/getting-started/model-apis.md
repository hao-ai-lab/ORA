# Model APIs

ORA's Native agent calls a model directly through OpenAI-compatible Chat
Completions or Anthropic Messages. There is no managed coding-agent session
between ORA and the endpoint.

## First episode

The [quickstart API section](../quickstart.md#model-api) contains the complete
setup for OpenAI, self-hosted compatible endpoints, and Anthropic.

| Variable | Purpose |
|---|---|
| `ORA_API_KEY` | Credential; falls back to the protocol's usual API-key variable |
| `ORA_PROVIDER` | `openai_compatible` or `anthropic_messages` |
| `ORA_BASE_URL` | Endpoint URL |
| `ORA_MODEL` | Model identifier served by that endpoint |
| `ORA_REASONING_EFFORT` | Optional reasoning setting supported by the endpoint |
| `ORA_SUPPORTS_IMAGES` | Set to `0` for a text-only Synthetic endpoint |

Keep keys in the process environment. ORA does not automatically load `.env`
files. Credentials do not belong in RunSpecs or Git.

## Robot episodes

Robot recipes select a provider profile when composing a RunSpec. Inspect
registered profiles without contacting a service:

```bash
uv run --locked ora providers
```

The shared default is `deepseek_flash_openai_low_16k_v1`; the standalone
Synthetic quickstart has its own explicit default. A robot endpoint needs the
image, tool-call, and continuation capabilities declared by its surface.
Registration and doctor discovery do not establish those capabilities; live
preflight checks them.

Read [configuration](../reference/configuration-and-evidence.md) and
[host setup](../environments/common.md) before a robot launch.

## Self-hosted endpoints

Use the model ID exposed by the service. Plain HTTP is allowed only on loopback;
remote services need HTTPS. Set `SSL_CERT_FILE` to the trust bundle for a private
certificate authority. Keys stay in the environment even for local servers.

See [troubleshooting](troubleshooting.md) for 401, 429, TLS, and capability failures.
