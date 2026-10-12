"""Host-only credential ordering and nonserializable qualification handoff."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


SELECTION_SCHEMA = 'ora.native-credential-selection.v1'
CREDENTIAL_HTTP_STATUSES = frozenset({401, 402, 403, 429})


def credential_ring(
    primary: str, standby: Sequence[str], slot_id: Any = None,
    *, selected_index: int | None = None,
) -> tuple[list[str], int]:
    keys = list(dict.fromkeys(key.strip() for key in (primary, *standby) if key.strip()))
    if not keys:
        raise RuntimeError('native credential pool is empty')
    digits = ''.join(ch for ch in str(slot_id or '') if ch.isdigit())
    index = (int(digits) % len(keys) if digits else 0) if selected_index is None else selected_index
    if type(index) is not int or not 0 <= index < len(keys):
        raise RuntimeError('native credential selection is out of range')
    return keys[index:] + keys[:index], index


def redact_credentials(value: Any, keys: Sequence[str]) -> Any:
    if isinstance(value, str):
        for key in sorted(keys, key=len, reverse=True):
            if key:
                value = value.replace(key, '[REDACTED]')
        return value
    if isinstance(value, Mapping):
        return {name: redact_credentials(item, keys) for name, item in value.items()}
    if isinstance(value, list):
        return [redact_credentials(item, keys) for item in value]
    return value


@dataclass(frozen=True)
class _CredentialHandoff:
    keys: tuple[str, ...] = field(repr=False)
    api_key_env: str = field(repr=False)
    plan_id: str | None = field(repr=False)
    plan_sha256: str | None = field(repr=False)
    selected_index: int = field(repr=False)


class _CredentialCheck(dict):
    """JSON serializes only public check fields, never the private snapshot."""


def bind_credential_handoff(
    check: Mapping[str, Any], plan: Mapping[str, Any], keys: Sequence[str], index: int,
) -> dict[str, Any]:
    result = _CredentialCheck(check)
    result._credential_handoff = _CredentialHandoff(
        tuple(keys), str(plan['policy']['api_key_env']),
        plan.get('plan_id'), plan.get('plan_sha256'), index,
    )
    return result


def _provider_check(preflight: Mapping[str, Any]) -> Mapping[str, Any]:
    checks = [item for item in preflight.get('checks', []) if item.get('check') == 'provider_marker']
    if len(checks) != 1 or checks[0].get('passed') is not True:
        raise RuntimeError('one passing native provider qualification is required')
    return checks[0]


def qualified_credential_index(
    plan: Mapping[str, Any], preflight: Mapping[str, Any], pool_size: int,
) -> int:
    if (
        preflight.get('schema') != 'ora.preflight.v1'
        or preflight.get('passed') is not True
        or preflight.get('plan_id') != plan.get('plan_id')
        or preflight.get('plan_sha256') != plan.get('plan_sha256')
    ):
        raise RuntimeError('native credential qualification belongs to another plan')
    detail = _provider_check(preflight).get('detail')
    selected = detail.get('credential_selection') if isinstance(detail, Mapping) else None
    if not isinstance(selected, Mapping) or selected.get('schema') != SELECTION_SCHEMA:
        raise RuntimeError('native credential selection is missing')
    index = selected.get('selected_index')
    if (
        type(selected.get('pool_size')) is not int or selected['pool_size'] != pool_size
        or type(index) is not int or not 0 <= index < pool_size
    ):
        raise RuntimeError('native credential selection or pool size is invalid')
    return index


def launch_environment(
    plan: Mapping[str, Any], preflight: Mapping[str, Any], environ: Mapping[str, str],
) -> dict[str, str]:
    environment = dict(environ)
    policy = plan['policy']
    if policy['kind'] != 'native_api' or not policy.get('require_live_marker', True):
        return environment
    handoff = getattr(_provider_check(preflight), '_credential_handoff', None)
    if not isinstance(handoff, _CredentialHandoff):
        raise RuntimeError('native launch requires a fresh in-process credential qualification')
    index = qualified_credential_index(plan, preflight, len(handoff.keys))
    if (
        (handoff.plan_id, handoff.plan_sha256) != (plan.get('plan_id'), plan.get('plan_sha256'))
        or handoff.api_key_env != policy['api_key_env'] or index != handoff.selected_index
    ):
        raise RuntimeError('native credential handoff differs from qualification')
    environment[handoff.api_key_env] = handoff.keys[0]
    environment[f'{handoff.api_key_env}S'] = ','.join(handoff.keys[1:])
    environment[policy['base_url_env']] = policy['endpoint']
    return environment
