"""Loopback gateway client binding before preflight.

Fake gateway directories under tmp stand in for ~/.config/cliproxyapi-*; the
real files are never read. Every assertion that touches a record, an error or
printed output also checks that no fake credential value appears in it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ora.agent.providers import gateway_client as gateway
from ora.agent.providers.profiles import load_builtin_provider_profiles
from ora.orchestrator import provider_gateway

KIMI_KEY, CODEX_KEY, OTHER_KEY = "fake-kimi-client-key", "fake-codex-client-key", "fake-unaccepted-key"
SECRETS = (KIMI_KEY, CODEX_KEY, OTHER_KEY)
PROFILES = load_builtin_provider_profiles()
_LOOPBACK = {
    "endpoint_transport": "loopback_http_v1", "gateway_client_binding": True,
    "proxy_env": ["NO_PROXY", "no_proxy"], "reasoning_effort": "high", "stream": True,
    "supports_images": True, "session_affinity": "x_session_id_run_v1",
    "qualification_schema": "ora.native-provider-probe.v1", "qualification_version": 1,
}
# Two test-local gateway profiles: an Anthropic-protocol account pool and an
# OpenAI-compatible gateway whose endpoint carries /v1.
KIMI = {"kind": "native_api", **_LOOPBACK, "protocol": "anthropic_messages",
        "endpoint": "http://127.0.0.1:8318", "model": "k3", "model_env": "NATIVE_MODEL_ID",
        "base_url_env": "NATIVE_MODEL_BASE_URL", "api_key_env": "NATIVE_MODEL_API_KEY",
        "context_window_tokens": 262144, "max_output_tokens": 8192}
ASTRA = {"kind": "native_api", **_LOOPBACK, "protocol": "openai_compatible",
         "endpoint": "http://127.0.0.1:8319/v1", "model": "gpt-6-astra",
         "model_env": "CODEX_PROXY_MODEL_ID", "base_url_env": "CODEX_PROXY_CHAT_BASE_URL",
         "api_key_env": "CODEX_PROXY_API_KEY", "context_window_tokens": 262144,
         "max_output_tokens": 16384, "output_token_parameter": "max_completion_tokens",
         "tool_strictness": "explicit_false", "auth_scheme": "bearer"}
LOW = {"kind": "native_api", **PROFILES["deepseek_flash_openai_low_16k_v1"]}


def _gateway(root: Path, name: str, port: int, keys: list[str], client: str | None) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "config.yaml").write_text(json.dumps({"host": "127.0.0.1", "port": port, "api-keys": keys}))
    if client is not None:
        (directory / "client.env").write_text(client)
    return directory


@pytest.fixture(autouse=True)
def no_real_home(tmp_path, monkeypatch):
    """Path.home() is a tmp directory in every test, so no real gateway file can ever be read."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(gateway.CONFIG_ROOT_ENV, str(home / "unset-gateway-root"))
    return home


@pytest.fixture
def gateways(tmp_path, monkeypatch):
    root = tmp_path / "config"
    _gateway(root, "cliproxyapi-kimi", 8318, [KIMI_KEY],
             f"export NATIVE_MODEL_API_KEY={KIMI_KEY}\nNATIVE_MODEL_BASE_URL=http://127.0.0.1:8318\n"
             "NATIVE_MODEL_ID=k3\nNO_PROXY=127.0.0.1,localhost\nUNRELATED=$(never-run)\n")
    _gateway(root, "cliproxyapi-codex", 8319, [CODEX_KEY],
             f"CODEX_PROXY_BASE_URL=http://127.0.0.1:8319\nCODEX_PROXY_API_KEY='{CODEX_KEY}'\n")
    _gateway(root, "unrelated-tool", 8318, [OTHER_KEY], "NATIVE_MODEL_API_KEY=x\n")  # not a cliproxyapi-* dir
    monkeypatch.setenv(gateway.CONFIG_ROOT_ENV, str(root))
    for name in (*gateway.PROXY_VARIABLES, *gateway.NO_PROXY_VARIABLES):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    return root


def _clean(value) -> None:
    text = json.dumps(value) if not isinstance(value, str) else value
    assert not any(secret in text for secret in SECRETS), text


def test_the_kimi_pool_binds_its_client_variables_and_records_only_names(gateways):
    environ: dict[str, str] = {"NATIVE_MODEL_API_KEY": "stale"}
    receipt = gateway.bind_gateway_client(KIMI, environ)
    assert environ == {"NATIVE_MODEL_API_KEY": KIMI_KEY, "NATIVE_MODEL_BASE_URL": "http://127.0.0.1:8318",
                       "NATIVE_MODEL_ID": "k3", "NO_PROXY": "127.0.0.1,localhost"}
    assert receipt == {
        "schema": gateway.RECEIPT_SCHEMA, "gateway_config": str(gateways / "cliproxyapi-kimi/config.yaml"),
        "client_file": str(gateways / "cliproxyapi-kimi/client.env"),
        "bound_variables": {"NATIVE_MODEL_API_KEY": "client_file", "NATIVE_MODEL_BASE_URL": "client_file",
                            "NATIVE_MODEL_ID": "client_file", "NO_PROXY": "client_file"},
        "api_key_accepted_by_gateway": True, "absent_from_client_file": ["no_proxy"],
    }
    _clean(receipt)


def test_the_codex_gateway_takes_its_key_from_the_client_and_url_and_model_from_the_profile(gateways):
    environ = {"CODEX_PROXY_API_KEY": "stale", "CODEX_PROXY_MODEL_ID": "gpt-6.1-sol",
               "CODEX_PROXY_CHAT_BASE_URL": "http://elsewhere/v1"}
    receipt = gateway.bind_gateway_client(ASTRA, environ)
    assert environ == {"CODEX_PROXY_API_KEY": CODEX_KEY, "CODEX_PROXY_MODEL_ID": "gpt-6-astra",
                       "CODEX_PROXY_CHAT_BASE_URL": "http://127.0.0.1:8319/v1"}
    assert receipt["bound_variables"] == {"CODEX_PROXY_API_KEY": "client_file",
                                          "CODEX_PROXY_CHAT_BASE_URL": "profile_endpoint",
                                          "CODEX_PROXY_MODEL_ID": "profile_model"}
    assert receipt["absent_from_client_file"] == ["CODEX_PROXY_CHAT_BASE_URL", "CODEX_PROXY_MODEL_ID",
                                                  "NO_PROXY", "no_proxy"]
    _clean(receipt)


@pytest.mark.parametrize("change, message", [
    (lambda root: (root / "cliproxyapi-kimi/client.env").unlink(), "gateway client file is missing: .*client.env"),
    (lambda root: (root / "cliproxyapi-kimi/client.env").write_text("NATIVE_MODEL_ID=k3\n"),
     "client.env does not assign NATIVE_MODEL_API_KEY"),
    (lambda root: (root / "cliproxyapi-kimi/client.env").write_text(f"NATIVE_MODEL_API_KEY={OTHER_KEY}\n"),
     "NATIVE_MODEL_API_KEY in .*client.env is not an api-key of .*cliproxyapi-kimi/config.yaml"),
    (lambda root: (root / "cliproxyapi-kimi/client.env").write_text(
        f"NATIVE_MODEL_API_KEY={KIMI_KEY}\nNATIVE_MODEL_BASE_URL=http://127.0.0.1:9999\n"),
     "assigns NATIVE_MODEL_BASE_URL to another endpoint"),
    (lambda root: (root / "cliproxyapi-kimi/client.env").write_text(
        f"NATIVE_MODEL_API_KEY={KIMI_KEY}\nNATIVE_MODEL_ID=k2\n"), "assigns NATIVE_MODEL_ID to another model"),
    (lambda root: (root / "cliproxyapi-kimi/client.env").write_text(f"NATIVE_MODEL_API_KEY=${KIMI_KEY}\n"),
     "line 1 must assign NATIVE_MODEL_API_KEY one literal value"),
    (lambda root: (root / "cliproxyapi-kimi/config.yaml").unlink(), "no loopback gateway configuration serves "
                                                                    "http://127.0.0.1:8318"),
    (lambda root: _gateway(root, "cliproxyapi-kimi-copy", 8318, [KIMI_KEY], None),
     "more than one gateway configuration serves"),
])
def test_a_missing_or_disagreeing_file_or_variable_is_refused_by_name_without_its_value(gateways, change, message):
    change(gateways)
    environ = {"KEEP": "unchanged"}
    with pytest.raises(gateway.GatewayClientError, match=message) as caught:
        gateway.bind_gateway_client(KIMI, environ)
    _clean(str(caught.value))
    assert environ == {"KEEP": "unchanged"}
    check = provider_gateway.gateway_check({"policy": KIMI}, environ)
    assert check["check"] == "provider_gateway_client" and check["passed"] is False
    _clean(check)


def test_an_http_proxy_must_exempt_the_gateway_host(gateways):
    environ = {"HTTP_PROXY": "http://proxy.invalid:3128"}
    with pytest.raises(gateway.GatewayClientError, match="does not exempt the gateway host 127.0.0.1"):
        gateway.bind_gateway_client(ASTRA, environ)
    assert environ == {"HTTP_PROXY": "http://proxy.invalid:3128"}
    gateway.bind_gateway_client(ASTRA, {**environ, "no_proxy": "localhost,127.0.0.1"})
    # The Kimi client file supplies its own NO_PROXY exemption.
    gateway.bind_gateway_client(KIMI, dict(environ))


def test_profiles_on_other_transports_are_untouched():
    environ = {"DEEPSEEK_LOW_API_KEY": "kept"}
    assert gateway.uses_gateway(LOW) is False and gateway.bind_gateway_client(LOW, environ) is None
    assert environ == {"DEEPSEEK_LOW_API_KEY": "kept"}
    assert provider_gateway.gateway_check({"policy": LOW}, environ) is None
    assert provider_gateway.gateway_check({"policy": {**KIMI, "kind": "claude_code"}}, environ) is None
    result = {"passed": True, "checks": [{"check": "provider_marker", "passed": True, "detail": {}}]}
    assert provider_gateway.gateway_preflight({"policy": LOW}, lambda plan: result) is result


def test_the_default_root_is_the_users_config_directory(monkeypatch, no_real_home):
    monkeypatch.delenv(gateway.CONFIG_ROOT_ENV)
    assert gateway.config_root() == no_real_home / ".config"
    assert gateway.config_root({gateway.CONFIG_ROOT_ENV: "~/gateways"}) == no_real_home / "gateways"


def test_the_root_comes_from_the_process_not_the_mapping_being_bound(gateways, tmp_path):
    """A caller-supplied mapping cannot point the loader at another directory."""
    elsewhere = tmp_path / "elsewhere"
    _gateway(elsewhere, "cliproxyapi-kimi", 8318, [OTHER_KEY], f"NATIVE_MODEL_API_KEY={OTHER_KEY}\n")
    environ = {gateway.CONFIG_ROOT_ENV: str(elsewhere)}
    gateway.bind_gateway_client(KIMI, environ)
    assert environ["NATIVE_MODEL_API_KEY"] == KIMI_KEY


def test_preflight_binds_before_the_provider_check_and_records_no_value(gateways, monkeypatch, tmp_path, capsys):
    import os

    from ora.orchestrator import cli

    for name in ("NATIVE_MODEL_API_KEY", "NATIVE_MODEL_BASE_URL", "NATIVE_MODEL_ID", "NO_PROXY"):
        # setenv records the original state, so the binding below is undone after the test.
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    seen = {}

    def run_preflight(plan):
        seen["key_bound_before_marker"] = os.environ.get("NATIVE_MODEL_API_KEY") == KIMI_KEY
        return {"schema": "ora.preflight.v1", "plan_id": "p", "plan_sha256": "s",
                "passed": True, "checks": [{"check": "provider_marker", "passed": True, "detail": "fixture"}]}
    monkeypatch.setattr(cli, "run_preflight", run_preflight)
    monkeypatch.setattr(cli.paths, "workspace_from_plan", lambda plan: tmp_path / "workspace")
    plan = {"plan_sha256": "s", "policy": KIMI}
    result = cli._preflight(plan)
    assert seen == {"key_bound_before_marker": True} and result["passed"] is True
    assert [check["check"] for check in result["checks"]] == ["provider_gateway_client", "provider_marker"]
    _clean(capsys.readouterr().out)
    _clean((tmp_path / "workspace/preflight/s.json").read_text())
    (gateways / "cliproxyapi-kimi/client.env").unlink()
    monkeypatch.delenv("NATIVE_MODEL_API_KEY")
    blocked = cli._preflight(plan)
    assert blocked["passed"] is False and blocked["checks"][0]["passed"] is False
    assert "gateway client file is missing" in blocked["checks"][0]["detail"]["error"]
    assert "preflight: BLOCKED" in capsys.readouterr().out


def test_direct_loopback_api_uses_supplied_credentials_without_gateway_files():
    direct = {**KIMI, "gateway_client_binding": False}
    environment = {"NATIVE_MODEL_API_KEY": "local-model-key"}
    assert gateway.uses_gateway(direct) is False
    assert gateway.bind_gateway_client(direct, environment) is None
    assert environment == {"NATIVE_MODEL_API_KEY": "local-model-key"}
    assert provider_gateway.gateway_check({"policy": direct}, environment) is None
