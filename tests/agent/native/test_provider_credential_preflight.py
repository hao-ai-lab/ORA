"""Bounded credential qualification and exact, secret-free launch handoff."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest

from ora.agent.contracts import PolicySessionError
from ora.agent.providers.credentials import (
    credential_ring, launch_environment, qualified_credential_index,
)
from ora.agent.providers.native import (
    AnthropicMessagesProvider, OpenAICompatibleProvider,
)
from ora.orchestrator.preflight import _provider_marker


KEYS = ('secret-primary-fixture', 'secret-second-fixture', 'secret-third-fixture')


def plan(*, slot='gpu0', continuity=True):
    return {
        'plan_id': 'test-plan', 'plan_sha256': 'a' * 64,
        'runtime': {'resource_binding': {'slot_id': slot}},
        'policy': {
            'kind': 'native_api', 'require_live_marker': True,
            'base_url_env': 'TEST_POOL_URL', 'api_key_env': 'TEST_POOL_KEY',
            'endpoint': 'https://provider.invalid', 'model': 'fixed-model',
            'provider': 'anthropic_messages', 'reasoning_effort': 'high',
            'stream': True, 'supports_images': True,
            **({'provider_qualification_schema': 'ora.native-provider-probe.v1'} if continuity else {}),
        },
    }


def probe_fixture(monkeypatch, failures=None, *, wrong_image=False, wrong_marker=False, responses=None):
    calls = []
    closed = []
    failures = failures or {}
    responses = responses or {}
    monkeypatch.setenv('TEST_POOL_URL', 'https://provider.invalid')
    monkeypatch.setenv('TEST_POOL_KEY', KEYS[0])
    monkeypatch.setenv('TEST_POOL_KEYS', ','.join([KEYS[1], KEYS[1], KEYS[0], KEYS[2]]))

    class Provider:
        def __init__(self, _protocol, **kwargs):
            self.key = kwargs['api_key']
            assert not kwargs.get('standby_api_keys')
            self.preflight_probe = False

        def close(self):
            closed.append(self.key)

        def complete(self, *, model_call_id, request):
            assert self.preflight_probe is True
            assert request['model'] == 'fixed-model'
            calls.append((self.key, model_call_id))
            failure = failures.get((self.key, model_call_id))
            if failure is not None:
                error = PolicySessionError(' '.join(KEYS))
                if isinstance(failure, int):
                    error.http_status = failure
                raise error
            if model_call_id in responses:
                return responses[model_call_id]
            if model_call_id == 'preflight-marker':
                content = 'wrong marker' if wrong_marker else request['messages'][0]['content'].rsplit(': ', 1)[-1]
            elif model_call_id == 'preflight-image':
                content = wrong_image if isinstance(wrong_image, str) else (KEYS[2] if wrong_image else 'red')
            elif model_call_id == 'preflight-continuity-tool':
                token = request['messages'][0]['content'].split('with token ', 1)[1].split('.', 1)[0]
                return {'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
                    'id': 'call-1', 'type': 'function',
                    'function': {'name': 'preflight_echo', 'arguments': json.dumps({'token': token})},
                }]}, 'finish_reason': 'tool_calls'}
            else:
                assert model_call_id == 'preflight-continuity-resume'
                content = request['messages'][-1]['content'].rsplit('exactly ', 1)[-1]
            return {'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}

    monkeypatch.setattr('ora.agent.providers.build_native_provider', Provider)
    return calls, closed


STAGES = ['preflight-marker', 'preflight-image', 'preflight-continuity-tool', 'preflight-continuity-resume']


@pytest.mark.parametrize('stage', STAGES)
@pytest.mark.parametrize('status', [401, 402, 403, 429])
def test_fallback_restarts_entire_suite_on_next_distinct_credential(monkeypatch, stage, status):
    calls, closed = probe_fixture(monkeypatch, {(KEYS[0], stage): status})
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is True
    selection = check['detail']['credential_selection']
    assert selection['selected_index'] == 1
    assert selection['pool_size'] == 3
    assert [name for key, name in calls if key == KEYS[1]] == STAGES
    assert not any(key == KEYS[2] for key, _ in calls)
    assert closed == [KEYS[0], KEYS[1]]
    assert all(key not in json.dumps(check) and key not in repr(check) for key in KEYS)


def test_every_credential_rejected_is_one_finite_pool_lap(monkeypatch):
    calls, closed = probe_fixture(monkeypatch, {(key, 'preflight-marker'): 403 for key in KEYS})
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is False
    assert calls == [(key, 'preflight-marker') for key in KEYS]
    assert closed == list(KEYS)
    assert check['detail']['credential_selection']['selected_index'] is None
    assert all(key not in json.dumps(check) for key in KEYS)


@pytest.mark.parametrize('status', [400, 404, 500, 503, 'transport'])
def test_noncredential_failures_do_not_try_another_key(monkeypatch, status):
    calls, _ = probe_fixture(monkeypatch, {(KEYS[0], 'preflight-marker'): status})
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is False
    assert calls == [(KEYS[0], 'preflight-marker')]
    assert all(key not in json.dumps(check) for key in KEYS)


def test_wrong_image_is_not_a_credential_failure_and_never_leaks_pool(monkeypatch):
    calls, _ = probe_fixture(monkeypatch, wrong_image=True)
    check = _provider_marker(plan(continuity=False), subprocess.run)
    assert check['passed'] is False
    assert {key for key, _ in calls} == {KEYS[0]}
    assert all(key not in json.dumps(check) for key in KEYS)


def test_slot_start_matches_existing_policy_ring(monkeypatch):
    calls, _ = probe_fixture(monkeypatch)
    check = _provider_marker(plan(slot='gpu2'), subprocess.run)
    assert calls[0][0] == KEYS[2]
    assert check['detail']['credential_selection']['selected_index'] == 2
    assert credential_ring(KEYS[0], [KEYS[1], KEYS[1], KEYS[2]], 'gpu2') == (
        [KEYS[2], KEYS[0], KEYS[1]], 2,
    )


def report_for(value, check):
    return {'schema': 'ora.preflight.v1', 'passed': True,
            'plan_id': value['plan_id'], 'plan_sha256': value['plan_sha256'], 'checks': [check]}


def test_launch_uses_exact_private_snapshot_without_global_env_mutation(monkeypatch):
    probe_fixture(monkeypatch, {(KEYS[0], 'preflight-marker'): 403})
    value = plan()
    check = _provider_marker(value, subprocess.run)
    report = report_for(value, check)
    changed_env = {'TEST_POOL_KEY': 'new-unqualified-key', 'TEST_POOL_KEYS': 'other-key', 'PATH': '/bin'}
    child = launch_environment(value, report, changed_env)
    assert child['TEST_POOL_KEY'] == KEYS[0]
    assert child['TEST_POOL_KEYS'] == ','.join(KEYS[1:])
    assert changed_env['TEST_POOL_KEY'] == 'new-unqualified-key'
    selected = qualified_credential_index(value, json.loads(json.dumps(report)), len(KEYS))
    assert selected == 1
    assert credential_ring(KEYS[0], list(KEYS[1:]), 'gpu2', selected_index=selected)[0][0] == KEYS[1]
    with pytest.raises(RuntimeError, match='in-process'):
        launch_environment(value, json.loads(json.dumps(report)), changed_env)


def test_tampered_or_wrong_plan_selection_is_rejected(monkeypatch):
    probe_fixture(monkeypatch)
    value = plan()
    check = _provider_marker(value, subprocess.run)
    report = report_for(value, check)
    with pytest.raises(RuntimeError):
        qualified_credential_index({**value, 'plan_sha256': 'b' * 64}, report, 3)
    with pytest.raises(RuntimeError):
        qualified_credential_index(value, report, 2)
    check['detail']['credential_selection']['selected_index'] = 2
    with pytest.raises(RuntimeError):
        launch_environment(value, report, {})


@pytest.mark.parametrize('provider_type', [AnthropicMessagesProvider, OpenAICompatibleProvider])
@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_probe_http_failures_are_structured_and_never_retried(monkeypatch, provider_type, status):
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={'error': {'message': 'fixture failure'}})
    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = provider_type(base_url='https://provider.invalid', api_key=KEYS[0], client=client)
    provider.preflight_probe = True
    monkeypatch.setattr('ora.agent.providers.native.time.sleep', lambda _: pytest.fail('probe retried'))
    with pytest.raises(PolicySessionError) as caught:
        provider.complete(model_call_id='probe', request={'model': 'model', 'messages': [{'role': 'user', 'content': 'OK'}], 'max_tokens': 16, 'stream': False})
    assert caught.value.http_status == status
    assert len(calls) == 1
    client.close()


def test_rotation_redaction_keeps_original_primary_in_protected_pool():
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    provider = AnthropicMessagesProvider(base_url='https://provider.invalid', api_key=KEYS[0], standby_api_keys=KEYS[1:], client=client)
    provider._rotate_credential(status_code=403, detail='quota')
    assert all(key not in provider._redact(' '.join(KEYS)) for key in KEYS)
    client.close()


@pytest.mark.parametrize('failed_stage', ['marker', 'image'])
def test_semantic_failure_stops_before_later_http_rejection(monkeypatch, failed_stage):
    following_stage = 'preflight-image' if failed_stage == 'marker' else 'preflight-continuity-tool'
    calls, _ = probe_fixture(
        monkeypatch, {(KEYS[0], following_stage): 403},
        wrong_marker=failed_stage == 'marker',
        wrong_image='white' if failed_stage == 'image' else False,
    )
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is False
    assert following_stage not in [stage for _, stage in calls]
    assert {key for key, _ in calls} == {KEYS[0]}


def test_long_mixed_case_standby_echo_is_rejected_before_truncation(monkeypatch):
    echoed_key = 'SecretFiXture-' * 12
    probe_fixture(monkeypatch, wrong_image=echoed_key)
    monkeypatch.setenv('TEST_POOL_KEYS', ','.join([KEYS[1], echoed_key]))
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is False
    assert echoed_key.lower()[:80] not in json.dumps(check).lower()


def test_formal_factory_requires_preflight_before_provider_construction(monkeypatch, tmp_path):
    from ora.orchestrator import policy_factory
    probe_fixture(monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail('provider construction preceded qualification validation')
    monkeypatch.setattr(policy_factory, 'build_native_provider', forbidden)
    with pytest.raises(RuntimeError, match='qualification'):
        policy_factory.build_policy_session(
            plan=plan(), run_root=tmp_path, run_id='test-run', workspace=tmp_path,
            control_root=tmp_path, events_path=tmp_path / 'events.jsonl',
            compiled_policy={}, tools=None, state_root=tmp_path,
            record_context_bundle=None,
        )


def test_factory_uses_qualified_key_instead_of_slot_selection(monkeypatch, tmp_path):
    from ora.orchestrator import policy_factory
    probe_fixture(monkeypatch, {(KEYS[0], 'preflight-marker'): 403})
    value = plan()
    report = report_for(value, _provider_marker(value, subprocess.run))
    (tmp_path / 'preflight.json').write_text(json.dumps(report), encoding='utf-8')
    child = launch_environment(value, report, {})
    for name, content in child.items():
        monkeypatch.setenv(name, content)
    observed = {}
    class Constructed(Exception):
        pass
    def construct(*args, **kwargs):
        observed.update(kwargs)
        raise Constructed
    monkeypatch.setattr(policy_factory, 'build_native_provider', construct)
    with pytest.raises(Constructed):
        policy_factory.build_policy_session(
            plan=value, run_root=tmp_path, run_id='test-run', workspace=tmp_path,
            control_root=tmp_path, events_path=tmp_path / 'events.jsonl',
            compiled_policy={}, tools=None, state_root=tmp_path,
            record_context_bundle=None,
        )
    assert observed['api_key'] == KEYS[1]
    assert observed['credential_start_index'] == 1
    assert observed['standby_api_keys'] == (KEYS[2], KEYS[0])


@pytest.mark.parametrize('slot', ['gpu7', 'gpu7b'])
def test_five_account_recovery_reaches_good_key_then_pins_it(monkeypatch, tmp_path, slot):
    """Failed launches exhausted indices 2/3/4 before reaching good 0/1."""
    from ora.orchestrator import policy_factory
    keys = tuple(f'fixture-key-{index}' for index in range(5))
    monkeypatch.setattr(sys.modules[__name__], 'KEYS', keys)
    calls, _ = probe_fixture(monkeypatch, {(key, 'preflight-marker'): 403 for key in keys[2:]})
    monkeypatch.setenv('TEST_POOL_KEYS', ','.join(keys[1:]))
    value = plan(slot=slot)
    report = report_for(value, _provider_marker(value, subprocess.run))
    assert report['passed'] is True
    assert [key for key, stage in calls if stage == 'preflight-marker'] == [keys[2], keys[3], keys[4], keys[0]]
    assert report['checks'][0]['detail']['credential_selection']['selected_index'] == 0
    (tmp_path / 'preflight.json').write_text(json.dumps(report))
    child = launch_environment(value, report, {})
    for name, content in child.items():
        monkeypatch.setenv(name, content)
    class Constructed(Exception):
        pass
    observed = {}
    def construct(*args, **kwargs):
        observed.update(kwargs)
        raise Constructed
    monkeypatch.setattr(policy_factory, 'build_native_provider', construct)
    with pytest.raises(Constructed):
        policy_factory.build_policy_session(
            plan=value, run_root=tmp_path, run_id='recovery-run', workspace=tmp_path,
            control_root=tmp_path, events_path=tmp_path / 'events.jsonl',
            compiled_policy={}, tools=None, state_root=tmp_path, record_context_bundle=None)
    assert observed['api_key'] == keys[0]
    assert observed['credential_start_index'] == 0


@pytest.mark.parametrize('serialized_only', [False, True])
def test_spawn_passes_snapshot_or_rejects_before_lease(monkeypatch, tmp_path, serialized_only):
    from ora.orchestrator import cli
    probe_fixture(monkeypatch, {(KEYS[0], 'preflight-marker'): 403})
    value = plan()
    value['source'] = {'commit': 'test-source'}
    report = report_for(value, _provider_marker(value, subprocess.run))
    if serialized_only:
        report = json.loads(json.dumps(report))
    monkeypatch.setenv('TEST_POOL_KEY', 'unqualified-new-primary')
    monkeypatch.setattr(cli, 'validate_run_plan', lambda *_: None)
    monkeypatch.setattr(cli, '_require_launchable', lambda *args, **kwargs: None)
    monkeypatch.setattr(cli.subprocess, 'run', lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, stdout='test-source' if 'rev-parse' in command else '',
    ))
    created = []
    def lease(*args):
        created.append(args)
        return {'claim_token': 'test-claim'}
    monkeypatch.setattr(cli, '_create_lease', lease)
    root = tmp_path / 'run'
    monkeypatch.setattr(cli.paths, 'new_run_root', lambda *args: root)
    monkeypatch.setattr(cli.paths, 'existing_run_location', lambda *args: None)
    monkeypatch.setattr(cli.paths, 'workspace_topology', lambda: {})
    monkeypatch.setattr(cli.paths, 'register_run', lambda *args: None)
    monkeypatch.setattr(cli, '_snapshot_launch_gate', lambda *args: None)
    monkeypatch.setattr(cli, '_claim_launch_admission', lambda *args: {})
    monkeypatch.setattr(cli, '_plan_resource_binding', lambda *args: {})
    monkeypatch.setattr(cli, '_assign_lease', lambda lease, pid: lease)
    observed = {}
    def popen(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(pid=12345)
    monkeypatch.setattr(cli.subprocess, 'Popen', popen)
    if serialized_only:
        with pytest.raises(RuntimeError, match='in-process'):
            cli._spawn(value, report, yes=True)
        assert not created and not observed and not root.exists()
    else:
        cli._spawn(value, report, yes=True)
        assert len(created) == 1
        assert observed['env']['TEST_POOL_KEY'] == KEYS[0]
        assert observed['env']['TEST_POOL_KEYS'] == ','.join(KEYS[1:])
        assert all(key not in (root / 'preflight.json').read_text() for key in KEYS)


@pytest.mark.parametrize('provider_type', [AnthropicMessagesProvider, OpenAICompatibleProvider])
def test_probe_transport_error_is_not_retried(monkeypatch, provider_type):
    calls = []
    def respond(request):
        calls.append(request)
        raise httpx.ReadTimeout('fixture transport error')
    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = provider_type(base_url='https://provider.invalid', api_key=KEYS[0], client=client)
    provider.preflight_probe = True
    monkeypatch.setattr('ora.agent.providers.native.time.sleep', lambda _: pytest.fail('probe retried'))
    with pytest.raises(httpx.ReadTimeout):
        provider.complete(model_call_id='probe', request={'model': 'm', 'messages': [], 'max_tokens': 16, 'stream': False})
    assert len(calls) == 1
    client.close()


def test_anthropic_probe_does_not_replay_incomplete_content_block(monkeypatch):
    events = [
        {'type': 'message_start', 'message': {'id': 'msg-partial'}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': 'partial'}},
        {'type': 'message_stop'},
    ]
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, text='\n\n'.join('data: ' + json.dumps(event) for event in events))
    client = httpx.Client(transport=httpx.MockTransport(respond))
    provider = AnthropicMessagesProvider(base_url='https://provider.invalid', api_key=KEYS[0], client=client)
    provider.preflight_probe = True
    monkeypatch.setattr('ora.agent.providers.native.time.sleep', lambda _: pytest.fail('probe retried'))
    with pytest.raises(PolicySessionError, match='precedes completed content blocks'):
        provider.complete(model_call_id='probe', request={'model': 'm', 'messages': [], 'max_tokens': 16, 'stream': True})
    assert len(requests) == 1
    client.close()


@pytest.mark.parametrize('stage', ['preflight-marker', 'preflight-image', 'preflight-continuity-tool'])
def test_cap_probe_keeps_finish_usage_and_failed_stage_without_changing_budget(monkeypatch, stage):
    calls, _ = probe_fixture(monkeypatch, responses={stage: {
        'message': {'role': 'assistant', 'content': '', 'reasoning_content': 'private reasoning'},
        'finish_reason': 'length', 'usage': {'input_tokens': 70, 'output_tokens': 256,
            'reasoning_output_tokens': 256, 'cached_input_tokens': 0, 'private': 'private usage'},
    }})
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is False
    probes = check['detail']['probes']
    assert list(probes) == STAGES[:STAGES.index(stage) + 1]
    assert all(p['semantic_passed'] is True for name, p in probes.items() if name != stage)
    assert probes[stage] == {'response_observed': True, 'semantic_passed': False,
        'finish_reason': 'length', 'content_observed': False, 'reasoning_observed': True,
        'tool_call_count': 0, 'usage': {'input_tokens': 70, 'output_tokens': 256,
            'reasoning_output_tokens': 256, 'cached_input_tokens': 0}, 'failure_category': 'output_cap'}
    assert all(key == KEYS[0] for key, _ in calls)
    assert 'private reasoning' not in json.dumps(check) and 'private usage' not in json.dumps(check)


def test_later_transport_failure_keeps_passed_probe_evidence(monkeypatch):
    probe_fixture(monkeypatch, failures={(KEYS[0], 'preflight-continuity-resume'): 'transport'})
    check = _provider_marker(plan(), subprocess.run)
    probes = check['detail']['probes']
    assert list(probes) == STAGES
    assert all(probes[stage]['semantic_passed'] is True for stage in STAGES[:-1])
    assert probes[STAGES[-1]] == {'response_observed': False, 'semantic_passed': None,
                                 'failure_category': 'provider_exception'}
    assert not any(key in json.dumps(check) for key in KEYS)


@pytest.mark.parametrize('response,category', [
    ([], 'malformed_response'),
    ({'message': {'role': 'assistant', 'content': 'blue'}, 'finish_reason': 'stop'}, 'semantic_mismatch'),
])
def test_invalid_image_probe_has_explicit_failure_category(monkeypatch, response, category):
    probe_fixture(monkeypatch, responses={'preflight-image': response})
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is False
    assert check['detail']['probes']['preflight-image']['failure_category'] == category


class OneShot(dict):
    """A scripted response served once; later calls get the healthy default."""

    def __getitem__(self, key):
        return self.pop(key)


def test_response_level_miss_reruns_whole_suite_once_on_same_credential(monkeypatch):
    capped = {'message': {'role': 'assistant', 'content': '', 'reasoning_content': 'pixels'},
              'finish_reason': 'length', 'usage': {'input_tokens': 225, 'output_tokens': 1024}}
    calls, _ = probe_fixture(monkeypatch, responses=OneShot({'preflight-image': capped}))
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is True
    assert calls == [(KEYS[0], stage) for stage in ['preflight-marker', 'preflight-image', *STAGES]]
    selection = check['detail']['credential_selection']
    assert selection['selected_index'] == 0
    assert [(a['passed'], a['response_retry'], a['stage']) for a in selection['attempts']] == [
        (False, False, 'preflight-image'), (True, True, 'preflight-continuity-resume'),
    ]


def test_repeated_response_level_miss_fails_after_one_rerun(monkeypatch):
    calls, _ = probe_fixture(monkeypatch, wrong_marker=True)
    check = _provider_marker(plan(), subprocess.run)
    assert check['passed'] is False
    assert calls == [(KEYS[0], 'preflight-marker')] * 2
    assert check['detail']['probes']['preflight-marker']['failure_category'] == 'semantic_mismatch'
    assert [a['response_retry'] for a in check['detail']['credential_selection']['attempts']] == [False, True]
