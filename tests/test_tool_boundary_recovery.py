"""Correctable inputs and public evidence survive the Native provider boundary."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
pytestmark = pytest.mark.usefixtures("ready_model_services")


from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.tool_surface import (
    PROFILE_CATALOG, ToolSurfaceError, validate_schema_instance,
)
from ora.agent.context.projection import _causal_projection_violations
from ora.agent.evidence.trajectory import ArtifactStore
from ora.agent.memory.artifacts import (
    ARTIFACT_READ_TOOL, ContextArtifactCatalog, RecoverableArtifactToolRuntime,
)
from ora.agent.native.gateway import _openai_parameters, openai_tool_definitions
from ora.agent.native.request_assembly import NativeRequestAssembler
from ora.agent.native.session import NativePlanner
from ora.agent.native.tool_loop import _tool_error
from ora.agent.providers.native import translate_anthropic_request
from ora.agent.tools.runtime import ManipulationToolRuntime
from ora.agent.tools.robot import ToolError


@pytest.mark.parametrize("keyword,bound,accepted,rejected", [
    ("minimum", -2, [-2, 0, 2], [-3]),
    ("maximum", 2, [-2, 0, 2], [3]),
    ("exclusiveMinimum", -2, [0, 2], [-3, -2]),
    ("exclusiveMaximum", 2, [-2, 0], [2, 3]),
])
def test_declared_numeric_bounds_accept_exact_closed_edges_and_reject_outside(
    keyword, bound, accepted, rejected,
):
    schema = {"type": "number", keyword: bound}
    for value in accepted:
        validate_schema_instance(value, schema)
    for value in rejected:
        with pytest.raises(ToolSurfaceError, match=rf"\$result.*{keyword} {bound}"):
            validate_schema_instance(value, schema)


@pytest.mark.parametrize("bad", [True, False, float("nan"), float("inf"), -float("inf")])
def test_numeric_schema_never_accepts_boolean_or_nonfinite_values(bad):
    with pytest.raises(ToolSurfaceError):
        validate_schema_instance(bad, {"type": "number", "minimum": -1, "maximum": 1})


def test_numeric_bound_error_names_nested_array_reference_and_rejected_branches():
    schema = {
        "$defs": {"count": {"oneOf": [
            {"type": "integer", "minimum": 0, "maximum": 3}, {"type": "null"},
        ]}},
        "type": "object", "additionalProperties": False,
        "properties": {"counts": {"type": "array", "items": {"$ref": "#/$defs/count"}}},
        "required": ["counts"],
    }
    validate_schema_instance({"counts": [0, 3, None]}, schema, root_path="$input")
    for bad, keyword in [(-1, "minimum"), (4, "maximum")]:
        with pytest.raises(ToolSurfaceError) as caught:
            validate_schema_instance({"counts": [0, bad]}, schema, root_path="$input")
        message = str(caught.value)
        assert "$input.counts[1]" in message
        assert "oneOf[0]" in message and "oneOf[1]" in message and keyword in message


def test_nullable_branch_cannot_skip_a_sibling_numeric_bound():
    schema = {"oneOf": [{"type": "number"}, {"type": "null"}], "minimum": 0}
    validate_schema_instance(None, schema)
    validate_schema_instance(0, schema)
    with pytest.raises(ToolSurfaceError, match="minimum"):
        validate_schema_instance(-1, schema)


#: The kept default surfaces.
LIBERO = "libero_placement_direct_overhead_v1"
ROBOTWIN = "robotwin_toolchain_b2_closed_noservo_v2"


def _planner(surface):
    planner = object.__new__(NativePlanner)
    planner.tools = SimpleNamespace(surface=surface)
    return planner


@pytest.fixture
def artifact_reader(tmp_path):
    surface = resolve_tool_surface(LIBERO)
    runtime = RecoverableArtifactToolRuntime(
        base=SimpleNamespace(surface=surface, definitions=lambda: []), run_root=tmp_path,
        event_path=tmp_path / "events.jsonl",
    )
    catalog = ContextArtifactCatalog(tmp_path, "recovery-test", ArtifactStore(tmp_path))
    reference = catalog.put_bytes(
        b'{"json_value":[{"reachable":false}],"status":"ok"}',
        kind="context_tool_result", suffix=".json", media_type="application/json",
        round_index=0, action_step=1,
    )
    runtime.bind_context_artifacts(catalog)
    return runtime, reference["ref"]


@pytest.mark.parametrize("pointer,extra", [
    ("/json_value", {"limit": 8192}),
    ("/results/0/reachable", {}),
    ("bad-pointer", {}),
])
@pytest.mark.parametrize("protocol", ["openai", "anthropic"])
def test_artifact_input_error_reaches_wire_with_recovery_fields(
    artifact_reader, pointer, extra, protocol,
):
    runtime, ref = artifact_reader
    with pytest.raises(ToolError) as caught:
        runtime._read_ref({"ref": ref, "json_pointer": pointer, **extra})
    result = _tool_error(caught.value)
    visible = _planner(runtime.surface)._causal_tool_result("artifact_read", result)
    error = visible["error"]
    assert error["code"] == "invalid_arguments" and error["retryable"] is True
    assert error["field"] == "json_pointer" and error["received"] is not None
    assert error["constraints"]["allowed"]
    if not extra:
        assert error["constraints"]["top_level_keys"] == ["json_value", "status"]
    provider = SimpleNamespace()
    if protocol == "anthropic":
        provider.prepare_request = translate_anthropic_request
    assembler = NativeRequestAssembler(
        model="offline-test", reasoning_effort="provider_default",
        max_output_tokens=128, stream=False, provider=provider,
    )
    request = assembler.prepare({
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "read-1", "type": "function", "function": {
                    "name": "artifact.read", "arguments": json.dumps({"ref": ref}),
                },
            }]},
            {"role": "tool", "name": "artifact.read", "tool_call_id": "read-1",
             "content": json.dumps(visible)},
        ],
        "tools": openai_tool_definitions(["artifact.read"], definitions=[ARTIFACT_READ_TOOL]),
    })
    wire = json.dumps(request.wire)
    assert "invalid_arguments" in wire and "constraints" in wire and "allowed" in wire
    assert not _causal_projection_violations(visible)
    assert runtime._read_ref({"ref": ref, "json_pointer": "/json_value/0/reachable"})["json_value"] is False


def test_artifact_schema_declares_pointer_and_range_modes(artifact_reader):
    _runtime, ref = artifact_reader
    schema = ARTIFACT_READ_TOOL["inputSchema"]
    for arguments in ({"ref": ref}, {"ref": ref, "offset": 2, "limit": 10},
                      {"ref": ref, "json_pointer": "/json_value"}):
        validate_schema_instance(arguments, schema, root_path="$input")
    with pytest.raises(ToolSurfaceError, match=r"\$input.*oneOf\[0\].*oneOf\[1\]"):
        validate_schema_instance(
            {"ref": ref, "json_pointer": "/json_value", "limit": 8192},
            schema, root_path="$input",
        )


def test_unexpected_key_error_is_still_a_nonretryable_internal_fault():
    result = _tool_error(KeyError("worker implementation defect"))
    assert result["error"]["code"] == "internal_tool_error"
    assert result["error"]["retryable"] is False


def test_missing_orientation_names_input_and_the_missing_field():
    runtime = object.__new__(ManipulationToolRuntime)
    runtime.surface = resolve_tool_surface(ROBOTWIN)
    with pytest.raises(ToolError) as caught:
        runtime._validate_public_arguments("move_pose", {
            "targets": {"left": {"position_mode": "absolute", "position_m": [0.0, 0.0, 0.9],
                                  "interaction": "free_space"}},
        })
    message = str(caught.value)
    assert "$input" in message and "$result" not in message
    assert "orientation_mode" in message
    assert caught.value.code == "invalid_arguments" and caught.value.retryable is True


@pytest.mark.parametrize("profile_id", sorted(PROFILE_CATALOG))
def test_registered_profile_wire_resolves_every_local_schema_reference(profile_id):
    surface = resolve_tool_surface(profile_id)

    def visit(node, root):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                target = root
                for token in ref[2:].split("/"):
                    target = target[token.replace("~1", "/").replace("~0", "~")]
            for value in node.values():
                visit(value, root)
        elif isinstance(node, list):
            for value in node:
                visit(value, root)

    for definition in surface.definitions:
        original = copy.deepcopy(definition["inputSchema"])
        projected = _openai_parameters(original)
        visit(projected, projected)
        assert original == definition["inputSchema"]


def test_observation_and_point_follow_their_declared_public_schema():
    planner = _planner(resolve_tool_surface(ROBOTWIN))
    facts = {"observation_id": "actual-3",
             "end_effectors": {"left": {"pose_xyz_xyzw": [0, 0, 0, 0, 0, 0, 1],
                                        "gripper_command_target": "open"}}}
    observed = {
        "success": True, "observation_id": "actual-3", "status": "actual",
        "public_robot_facts": facts, "views": ["head_camera"], "object_gt": "must remain private",
    }
    visible = planner._causal_tool_result("observe", observed)
    assert visible["success"] is True and visible["observation_id"] == "actual-3"
    assert visible["public_robot_facts"] == facts
    assert "object_gt" not in visible
    point = {"success": True, "status": "solved", "observation_id": "actual-3", "view": "head_camera",
             "mark_count": 1, "solved_count": 1, "failed_count": 0,
             "marks": [{"point_id": "point-1", "status": "solved", "xyz_m": [0.1, 0.2, 0.9],
                        "action_workspace": {"admissible": False}}]}
    assert planner._causal_tool_result("mark_point", point) == point
    assert not _causal_projection_violations(visible)


def test_unregistered_refusal_retains_planner_recovery_without_pose_pair():
    planner = _planner(resolve_tool_surface(LIBERO))
    effect = planner._minimal_action_effect({
        "schema": "future-generic-motion-receipt", "action": "recover_motion",
        "motion_status": "refused", "feedback": {
            "planning_failure_reason": "INVALID_START_STATE_JOINT_LIMITS",
        },
    })
    assert (effect["status"], effect["failure_stage"], effect["failure_code"]) == (
        "plan_rejected", "planner", "invalid_start_state",
    )


def test_declared_diagnostics_survives_the_operational_filter():
    """A schema-declared content property is promised evidence.

    ``grasp_plan`` builds its candidate ledger into ``diagnostics`` deliberately
    for the policy and the result schema declares the property, but the shared
    operational filter removed the whole block. A policy therefore could not see
    how many candidates the backend returned, how many survived compilation, or
    why the rest were rejected — and so could not cite an untried candidate.

    ``backend`` is the other half of that filter and is deliberately excluded:
    model id, version and config hash are provenance the policy cannot act on,
    and sixty schema sites declare it as an artefact of how those schemas were
    written rather than as a promise. Three older tests assert it stays out.
    When the v4 generation removes those declarations, this exclusion and the
    split in ``projection.py`` both go away.
    """

    planner = _planner(resolve_tool_surface(ROBOTWIN))
    result = {
        "success": True,
        "status": "candidates",
        "observation_id": "obs-7",
        "target_ref": "target-" + "a" * 24,
        "camera": "head_camera",
        "candidates": [],
        "backend": {"id": "contact_graspnet_pytorch", "version": "53e9966d"},
        "diagnostics": {
            "raw_candidate_limit": 32,
            "returned_candidate_count": 29,
            "compiled_candidate_count": 3,
            "rejected": [{"rank": 4, "reasons": ["unreachable"]}],
        },
    }
    visible = planner._causal_tool_result("grasp_plan", copy.deepcopy(result))
    assert visible["diagnostics"] == result["diagnostics"]
    assert "backend" not in visible
    assert not _causal_projection_violations([{"role": "tool", "content": json.dumps(visible)}])


def test_operational_keys_stay_stripped_below_the_declared_top_level():
    """``declared`` is not propagated: a nested ``backend`` is still provenance."""

    planner = _planner(resolve_tool_surface(ROBOTWIN))
    visible = planner._causal_tool_result("grasp_plan", {
        "success": True,
        "status": "candidates",
        "observation_id": "obs-8",
        "candidates": [{"candidate_id": "grasp-" + "b" * 24,
                        "backend": {"id": "leaked"},
                        "diagnostics": {"internal": 1}}],
        "diagnostics": {"raw_candidate_limit": 32},
    })
    assert visible["diagnostics"] == {"raw_candidate_limit": 32}
    candidate = visible["candidates"][0]
    assert "backend" not in candidate and "diagnostics" not in candidate


def test_privilege_filter_still_applies_to_declared_properties():
    """Relaxing the operational filter must not relax the privilege boundary."""

    planner = _planner(resolve_tool_surface(ROBOTWIN))
    visible = planner._causal_tool_result("grasp_plan", {
        "success": True,
        "status": "candidates",
        "observation_id": "obs-9",
        "candidates": [],
        "diagnostics": {"raw_candidate_limit": 32, "path": "/runs/x", "sha256": "d" * 64},
    })
    assert visible["diagnostics"] == {"raw_candidate_limit": 32}


def test_an_unparseable_tool_call_reports_one_name_to_both_halves():
    """A malformed call must still pair, or the run dies instead of retrying.

    A provider returned a tool call with no function name. The proposal event
    recorded that -- name null -- and the error result was recorded as
    `invalid_tool_call`, so the ActionStep inventory refused the pair with
    "tool result name differs from its call" and the run ended as an
    infrastructure fault after 56 model calls and 34 decisions. The error was
    marked retryable and the policy never saw it.

    The placeholder is fine; using it on one side only is not. Dispatch still
    sees the absent name, so the refusal keeps saying the call lacks a function
    name rather than that some tool is unknown.
    """

    import inspect

    from ora.agent.native import tool_loop

    source = inspect.getsource(tool_loop)
    # The proposal payload and the recorded result must agree on the name.
    assert 'payload={"name": name, "effect": effect, "batch_valid": False}' not in source, (
        "the batch-rejected proposal still records a name the result renames"
    )
    assert '"name": name, "arguments": args, "effect": effect' not in source, (
        "the accepted proposal still records a name the result may rename"
    )
    # One proposal site and one result site remain since the Agent-state
    # whole-batch refusal left with Task Memory.
    assert source.count("reported_name") >= 3, (
        "the proposal site and the result site should read one name"
    )


def test_every_stale_target_refusal_names_the_ref_it_refused():
    """A refusal that does not say which ref went stale cannot be acted on.

    Three tools refuse a target bound to an earlier observation. Two of them
    carried the value and one of those also names the way back; the third --
    the one `align_plan` reaches, and the most-called of the three -- passed
    only `field="target_ref"`, so thirteen refusals in one cohort told the
    policy that something was stale without saying what. The ref is the whole
    content of the refusal.
    """

    import ast
    import inspect
    import sys

    from ora.agent.tools.runtime import core as runtime_core

    # The runtime is runtime/core.py plus its tool-family mixin modules.
    modules = sorted({
        sys.modules[klass.__module__] for klass in runtime_core.ManipulationToolRuntime.__mro__
        if klass.__module__.startswith("ora.agent.tools.runtime.")
    }, key=lambda module: module.__name__)
    nodes = [
        node for module in modules
        for node in ast.walk(ast.parse(inspect.getsource(module)))
    ]
    missing = []
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        name = getattr(function, "id", None) or getattr(function, "attr", None)
        if name != "ToolError":
            continue
        keywords = {kw.arg for kw in node.keywords}
        code = next(
            (kw.value.value for kw in node.keywords
             if kw.arg == "code" and isinstance(kw.value, ast.Constant)),
            None,
        )
        # A refusal that names no input field is not about something the policy
        # passed -- "observation changed before action persistence" has no ref
        # to report. One that does name a field is refusing a value, and the
        # value is what makes the refusal actionable.
        if (
            code in {"stale_observation", "target_ref_unavailable",
                     "target_geometry_unavailable"}
            and "field" in keywords
            and "received" not in keywords
        ):
            missing.append((code, node.lineno))
    assert not missing, (
        "these target refusals name a field but not the value they refused: "
        f"{missing}"
    )


@pytest.mark.parametrize('function_name,signed', [('compute', False), (None, False), ('compute', True), (None, True), ('move_pose', False), ('move_pose', True)])
def test_bad_tool_json_reaches_the_next_lossless_provider_request(tmp_path, function_name, signed):
    named = function_name is not None
    from ora.agent.providers.native import FakeProvider
    from test_lossless_context_runtime import (
        _media_runtime, PROJECT_POLICY, EPISODE_TASK, HARNESS_SYSTEM, MANAGEMENT,
    )
    base, _, png, pointer = _media_runtime(tmp_path / 'world')
    # The Synthetic fixture publishes no public robot facts; they are not what
    # this recovery contract is about.
    import dataclasses
    base.surface = dataclasses.replace(base.surface, capabilities=tuple(
        name for name in base.surface.capabilities if name != 'current_public_robot_facts_v1'))
    run_root = tmp_path / 'world'
    from ora.agent.evidence.policy_state import PolicyStateStore
    observation_root = next((run_root / 'source').iterdir())
    observation = json.loads((observation_root / 'observation.json').read_text())
    state_store = PolicyStateStore(run_root, 'json-recovery', 'test-plan')
    state = state_store.publish(observation=observation, observation_root=observation_root,
        cause='reset', task_instruction='Inspect.', previous_public_action=None,
        remaining_budget={'robot_decisions': 4, 'model_calls': 2})
    pointer['policy_state'] = {**state_store.steps[-1], 'step': state['step'],
                               'state_sha256': state['state_sha256']}
    tools = RecoverableArtifactToolRuntime(base=base, run_root=run_root, event_path=run_root / 'unused.jsonl')
    function = {'name': function_name, 'arguments': '{"bindings": [], "expressions": "1"}}'} if named else {'arguments': '{}'}
    if function_name == 'move_pose':
        function['arguments'] = '{"targets":{"right":{"position_m":[0.3,0.1,0.9],"orientation_xyzw":[0,0,0,1]}}}}'
    provider = FakeProvider([
        {'message': {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': 'bad-json', 'type': 'function', 'function': function}]},
         'finish_reason': 'tool_calls'},
        {'message': {'role': 'assistant', 'content': 'I can correct the call.'}, 'finish_reason': 'stop'},
    ])
    requests = provider.requests
    client = None
    if signed:
        import httpx
        from ora.agent.providers.native import OpenAICompatibleProvider
        requests = []
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                message = {'role': 'assistant', 'content': '', 'reasoning_content': 'Compute.',
                    'tool_calls': [{'id': 'bad-json', 'type': 'function', 'function': function}]}
            else:
                previous = next(item for item in body['messages'] if item['role'] == 'assistant')
                assert previous['tool_calls'][0]['function']['arguments'] == function['arguments']
                assert previous['reasoning_content'] == 'Compute.'
                message = {'role': 'assistant', 'content': 'I can correct the call.'}
            return httpx.Response(200, json={'choices': [{'message': message,
                'finish_reason': 'tool_calls' if len(requests) == 1 else 'stop'}]})
        client = httpx.Client(transport=httpx.MockTransport(handler))
        provider = OpenAICompatibleProvider('https://provider.invalid', 'test', client=client)
    planner = NativePlanner(
        run_root=run_root, run_id='json-recovery', tools=tools, provider=provider,
        project_instructions_path=PROJECT_POLICY, episode_prefix_path=EPISODE_TASK,
        harness_system_path=HARNESS_SYSTEM, task_id='task-0', model='fake',
        context_profile='native_causal_context_v1', context_window_tokens=200_000,
        max_output_tokens=2048, context_composition='native_slots_v3',
        context_retention='action_step_lossless_v4', summary_policy='rollout_checkpoint_v1',
        context_management=MANAGEMENT, max_model_calls_per_observation=2, stream=False,
    )
    image = tmp_path / 'actual.png'
    image.write_bytes(png)
    try:
        planner.run_observation(task_instruction='Inspect.', context=pointer, image_path=image)
        assert len(requests) == 2
        messages = requests[1]['messages']
        reply = next(m for m in messages if m.get('tool_call_id') == 'bad-json')
        result = json.loads(reply['content'])
        assert result['error']['retryable'] is True
        assert reply['name'] == (function_name if named else 'invalid_tool_call')
        assert result['error']['code'] == ('invalid_arguments' if named else 'invalid_tool_call')
        prior = next(m for m in messages if m.get('role') == 'assistant' and m.get('tool_calls'))
        expected_function = function if named or not signed else {'name': '', **function}
        assert prior['tool_calls'][0]['function'] == expected_function
        events = [json.loads(line) for line in planner.recorder.path.read_text().splitlines()]
        returned = next(event for event in events if event['type'] == 'provider_response')
        assert returned['payload']['message']['tool_calls'][0]['function'] == expected_function
        if signed:
            from ora.agent.providers.continuation import (
                PROVIDER_CONTINUATION_MESSAGE_KEY, ProviderContinuationState, canonical_assistant_sha256,
            )
            original_assistant = returned['payload']['message']
            continuation = ProviderContinuationState.from_mapping(original_assistant[PROVIDER_CONTINUATION_MESSAGE_KEY])
            assert continuation.canonical_message_sha256 == canonical_assistant_sha256(original_assistant)
            raw_events = json.loads((run_root / returned['artifact_ref']['path']).read_text())
            assert raw_events[0]['choices'][0]['message']['tool_calls'][0]['function'] == function
        assert not (base.control_root / 'submitted-action.json').exists()
    finally:
        planner.close()
        if client is not None:
            client.close()


@pytest.mark.parametrize('profile,reader_name', [
    (LIBERO, 'artifact_read'),
    (ROBOTWIN, 'artifact_read'),
    ('robotwin_toolchain_b2_closed_noservo_v1', 'artifact_read'),
    ('robodojo_placement_closed_skills_v2', 'artifact_read'),
])
@pytest.mark.parametrize('protocol', ['openai', 'anthropic'])
def test_delivered_reader_matches_pointer_and_recovers_exact_evidence(tmp_path, profile, reader_name, protocol):
    from ora.agent.native.gateway import NativeToolGateway
    from ora.agent.memory.artifacts import (
        recoverable_artifact_envelope, recoverable_artifact_pointer,
        encode_recoverable_artifact_marker, decode_recoverable_artifact_marker,
    )
    surface = resolve_tool_surface(profile)
    from pathlib import Path
    base = ManipulationToolRuntime(control_root=tmp_path / 'control',
        manual_path=Path(__file__).resolve().parents[1] / 'ora/policy/embodiments/aloha-agilex/manual-v1.md',
        event_path=tmp_path / 'base-events.jsonl', workspace_bounds_m={
            'x': [-1, 1], 'y': [-1, 1], 'z': [0, 2]}, tool_surface_profile=profile,
        segmentation_backend=SimpleNamespace(), grounding_backend=SimpleNamespace(),
        grasp_backend=SimpleNamespace())
    runtime = RecoverableArtifactToolRuntime(base, tmp_path, tmp_path / 'events.jsonl')
    catalog = ContextArtifactCatalog(tmp_path, 'wire-reader', ArtifactStore(tmp_path))
    runtime.bind_context_artifacts(catalog, event_sink=lambda *args, **kwargs: None)
    gateway = NativeToolGateway(runtime)
    definitions = gateway.definitions()  # Actual allowlist filter that dropped the live reader.
    delivered = [item['function']['name'] for item in definitions]
    assert delivered == [item['name'] for item in surface.definitions] + [reader_name]
    raw = b'{"result":{"reachable":false},"note":"exact original evidence"}'
    reference = catalog.put_bytes(raw, kind='context_tool_result', suffix='.json',
        media_type='application/json', round_index=0, action_step=1)
    pointer = recoverable_artifact_pointer(reference)
    assert pointer['open'] == reader_name
    assert decode_recoverable_artifact_marker(encode_recoverable_artifact_marker(
        recoverable_artifact_envelope(reference))) == pointer
    result = gateway.execute(pointer['open'], {'ref': pointer['ref'], 'json_pointer': '/result/reachable'}, call_context={})
    assert result['json_value'] is False
    chunks, offset = [], 0
    while True:
        page = gateway.execute(pointer['open'], {'ref': pointer['ref'], 'offset': offset, 'limit': 11}, call_context={})
        chunks.append(page['content'])
        offset = page['next_offset']
        if page['eof']:
            break
    assert ''.join(chunks).encode() == raw
    with pytest.raises(ToolError) as caught:
        gateway.execute('observe', {'ref': pointer['ref']}, call_context={})
    error = _planner(runtime.surface)._causal_tool_result('observe', _tool_error(caught.value))
    assert error['error']['code'] == 'invalid_arguments'
    assert error['error']['constraints']['open']['tool'] == reader_name
    visible = _planner(runtime.surface)._causal_tool_result(reader_name, result)
    provider = SimpleNamespace()
    if protocol == 'anthropic':
        provider.prepare_request = translate_anthropic_request
    request = NativeRequestAssembler(model='offline', reasoning_effort='provider_default',
        max_output_tokens=128, stream=False, provider=provider).prepare({
        'tools': definitions, 'messages': [
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'read-1', 'type': 'function',
             'function': {'name': reader_name, 'arguments': json.dumps({'ref': pointer['ref'], 'json_pointer': '/result/reachable'})}}]},
            {'role': 'tool', 'name': reader_name, 'tool_call_id': 'read-1', 'content': json.dumps(visible)},
        ]})
    assert '"json_value": false' in json.dumps(request.wire).replace('\\"', '"')
    assert reference['ref'] in json.dumps(request.wire)


@pytest.mark.parametrize("result_name,content", [
    ("invalid_tool_call", {"ok": True}),
    ("invalid_tool_call", {"error": {"code": "invalid_arguments", "retryable": True}}),
    ("invalid_tool_call", {"error": {"code": "invalid_tool_call", "retryable": False}}),
    ("compute", {"error": {"code": "invalid_tool_call", "retryable": True}}),
    (None, {"error": {"code": "invalid_tool_call", "retryable": True}}),
    ("invalid_tool_call", {"success": True, "error": {"code": "invalid_tool_call", "retryable": True}}),
])
def test_nameless_history_requires_its_matching_failure_receipt(result_name, content):
    from ora.agent.contracts import PolicySessionError
    from ora.agent.context.projection import _action_step_inventory
    messages = [
        {"role": "user", "content": "Actual", "_observation": {"step": 0}},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "bad-call", "type": "function", "function": {"arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "bad-call", "name": result_name, "content": json.dumps(content)},
    ]
    original = copy.deepcopy(messages)
    with pytest.raises(PolicySessionError, match="nameless tool call lacks its invalid_tool_call refusal"):
        _action_step_inventory(messages, mutation_tools=["invalid_tool_call"])
    assert messages == original


def test_nameless_refusal_inventory_preserves_canonical_history_without_mutation():
    from ora.agent.context.projection import _action_step_inventory
    messages = [
        {"role": "user", "content": "Actual", "_observation": {"step": 0}},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "bad-call", "type": "function", "function": {"arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "bad-call", "name": "invalid_tool_call",
         "content": json.dumps({"ok": False, "error": {"code": "invalid_tool_call", "retryable": True}})},
    ]
    original = copy.deepcopy(messages)
    inventory = _action_step_inventory(messages, mutation_tools=["invalid_tool_call"])
    assert messages == original
    assert inventory[0]["mutation"] is None and inventory[0]["mutation_tool_call_id"] is None
    messages[-1]["tool_call_id"] = "different-call"
    from ora.agent.contracts import PolicySessionError
    with pytest.raises(PolicySessionError, match="does not pair every tool call and result"):
        _action_step_inventory(messages, mutation_tools=[])
