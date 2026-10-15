"""Bind a loopback gateway's client variables for the Provider profile a plan selected.

A native_api policy with ``gateway_client_binding: true`` (the
Kimi K3 account pool, the Codex gateway) reaches a local gateway that keeps the
upstream accounts itself. Its client credentials live with the gateway, not in
the shared ``.env``: ``<root>/<gateway>/config.yaml`` names the host, port and
accepted ``api-keys``, and ``<root>/<gateway>/client.env`` holds the client's
literal ``NAME=value`` assignments. ``<root>`` is ``~/.config`` unless
``ORA_GATEWAY_CONFIG_ROOT`` names another directory, and a gateway
directory is any ``cliproxyapi-*`` there.

``bind_gateway_client`` selects the one gateway whose ``config.yaml`` serves the
policy's endpoint (``http://host:port``, or that root plus ``/v1``) and binds,
into the given environment, exactly the policy's own variables:

- ``api_key_env``: from ``client.env``; it must be one of the gateway's keys;
- ``base_url_env``: the policy endpoint (a ``client.env`` value must equal it);
- ``model_env``: the policy model (a ``client.env`` value must equal it);
- ``proxy_env`` names that ``client.env`` assigns.

An HTTP proxy that does not exempt the gateway host is refused, since the key
would leave the host. The receipt names variables, files and their presence;
no value ever enters it, an error message or a log. Profiles on any other
transport are not touched.

Maintained port of an earlier launcher's ``load_provider_environment``.
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
from typing import Any, Mapping, MutableMapping
from urllib.parse import urlsplit

import yaml

RECEIPT_SCHEMA = "ora.provider-gateway-client.v1"
LOOPBACK_TRANSPORT = "loopback_http_v1"
CONFIG_ROOT_ENV = "ORA_GATEWAY_CONFIG_ROOT"
GATEWAY_GLOB = "cliproxyapi-*"
PROXY_VARIABLES = ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
NO_PROXY_VARIABLES = ("NO_PROXY", "no_proxy")


class GatewayClientError(ValueError):
    """A loopback gateway's client binding is missing or disagrees with the profile; no value is named."""


def uses_gateway(policy: Mapping[str, Any]) -> bool:
    """True for a native_api policy that reaches a loopback gateway."""
    return (policy.get("kind", "native_api") == "native_api"
            and policy.get("endpoint_transport") == LOOPBACK_TRANSPORT
            and policy.get("gateway_client_binding") is True)


def config_root(environ: Mapping[str, str] | None = None) -> Path:
    """``ORA_GATEWAY_CONFIG_ROOT`` of ``environ`` (default ``os.environ``), else ``~/.config``."""
    environ = os.environ if environ is None else environ
    raw = environ.get(CONFIG_ROOT_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".config"


def _endpoint_root(endpoint: str) -> str:
    parts = urlsplit(endpoint.rstrip("/"))
    return f"{parts.scheme}://{parts.hostname}:{parts.port}"


def _gateway_root(config: Mapping[str, Any]) -> str:
    return f"http://{config.get('host')}:{config.get('port')}"


def find_gateway(endpoint: str, root: Path) -> tuple[Path, dict[str, Any]]:
    """The one gateway directory under ``root`` whose ``config.yaml`` serves ``endpoint``."""
    wanted = _endpoint_root(endpoint)
    if endpoint.rstrip("/") not in (wanted, f"{wanted}/v1"):
        raise GatewayClientError(f"loopback endpoint {endpoint} is neither a gateway root nor its /v1")
    found = []
    for config in sorted(root.glob(f"{GATEWAY_GLOB}/config.yaml")):
        try:
            value = yaml.safe_load(config.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise GatewayClientError(f"gateway configuration is unreadable: {config} ({type(exc).__name__})") from None
        if isinstance(value, Mapping) and _gateway_root(value) == wanted:
            found.append((config.parent, dict(value)))
    if not found:
        raise GatewayClientError(
            f"no loopback gateway configuration serves {wanted}: none of {root}/{GATEWAY_GLOB}/config.yaml "
            f"names that host and port (set {CONFIG_ROOT_ENV} for another root)")
    if len(found) > 1:
        raise GatewayClientError(f"more than one gateway configuration serves {wanted}: "
                                 + ", ".join(str(path / 'config.yaml') for path, _ in found))
    return found[0]


def client_assignments(path: Path, names: set[str]) -> dict[str, str]:
    """The literal ``NAME=value`` lines of ``path`` for ``names`` only; values are never echoed."""
    if not path.is_file():
        raise GatewayClientError(f"gateway client file is missing: {path}")
    values = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        key, sep, text = raw.strip().removeprefix("export ").partition("=")
        if key not in names:
            continue
        try:
            words = shlex.split(text, comments=True)
        except ValueError:
            words = []
        if not sep or len(words) != 1 or "$" in text or "`" in text:
            raise GatewayClientError(f"{path} line {number} must assign {key} one literal value")
        values[key] = words[0]
    return values


def _proxy_exempts(environ: Mapping[str, str], host: str) -> bool:
    if not any(environ.get(name) for name in PROXY_VARIABLES):
        return True
    exempt = {item.strip() for name in NO_PROXY_VARIABLES for item in environ.get(name, "").split(",")}
    return bool(exempt & {host, "*"})


def bind_gateway_client(policy: Mapping[str, Any], environ: MutableMapping[str, str] | None = None, *,
                        root: Path | None = None) -> dict[str, Any] | None:
    """Bind the policy's gateway client variables into ``environ``; return a value-free receipt.

    Returns None for a policy that does not reach a loopback gateway. Raises
    :class:`GatewayClientError` naming the missing or disagreeing file or
    variable; ``environ`` is changed only when every check passes.
    """
    if not uses_gateway(policy):
        return None
    environ = os.environ if environ is None else environ
    # The gateway root is process configuration (os.environ), never read from
    # the mapping being bound, so a caller-supplied mapping cannot redirect it.
    directory, config = find_gateway(str(policy["endpoint"]), config_root() if root is None else root)
    client = directory / "client.env"
    key_name, base_name, model_name = (str(policy[field]) for field in ("api_key_env", "base_url_env", "model_env"))
    proxy_names = [str(name) for name in policy.get("proxy_env", [])]
    values = client_assignments(client, {key_name, base_name, model_name, *proxy_names})
    if key_name not in values:
        raise GatewayClientError(f"{client} does not assign {key_name}")
    if values[key_name] not in (config.get("api-keys") or []):
        raise GatewayClientError(f"{key_name} in {client} is not an api-key of {directory / 'config.yaml'}")
    endpoint, model = str(policy["endpoint"]).rstrip("/"), str(policy["model"])
    if base_name in values and values[base_name].rstrip("/") != endpoint:
        raise GatewayClientError(f"{client} assigns {base_name} to another endpoint than the profile's")
    if model_name in values and values[model_name] != model:
        raise GatewayClientError(f"{client} assigns {model_name} to another model than the profile's")
    bound = {name: values[name] for name in (key_name, *proxy_names) if name in values}
    bound[base_name], bound[model_name] = endpoint, model
    host = str(config.get("host"))
    if not _proxy_exempts({**environ, **bound}, host):
        raise GatewayClientError(
            f"an HTTP proxy is set and {'/'.join(NO_PROXY_VARIABLES)} does not exempt the gateway host {host}")
    environ.update(bound)
    sources = {name: "client_file" for name in values}
    sources.setdefault(base_name, "profile_endpoint")
    sources.setdefault(model_name, "profile_model")
    return {
        "schema": RECEIPT_SCHEMA, "gateway_config": str(directory / "config.yaml"), "client_file": str(client),
        "bound_variables": {name: sources[name] for name in sorted(bound)},
        "api_key_accepted_by_gateway": True,
        "absent_from_client_file": sorted({base_name, model_name, *proxy_names} - set(values)),
    }
