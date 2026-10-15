"""Reader arguments and integral floats, checked before dispatch.

EXP11B refusal census 6, 8, A2, A5 and E2 (0059 open-bugs triage rows 11, 21,
22, 23):

* ``artifact_read`` and ``observe(ref)`` argument errors are ``invalid_arguments``
  naming ref, offset, limit or json_pointer -- never ``internal_tool_error``,
  an infrastructure code a policy argument could use to unlock abort -- while a
  store fault behind a ref stays infrastructure;
* a selected JSON value is bounded like a byte range;
* an integral float at an ``integer`` position reaches the handler as an int
  through both gateways, and nothing else is rewritten.

The v2 abort refusal (row 11) is pinned in ``test_finish_outcomes.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ora.agent.composition import resolve_tool_surface
from ora.agent.composition.integral_arguments import coerce_integral_arguments
from ora.agent.evidence.trajectory import ArtifactStore
from ora.agent.memory.artifacts import (
    ARTIFACT_READ_LIMIT_BYTES, ContextArtifactCatalog, RecoverableArtifactToolRuntime,
)
from ora.agent.native.tool_arguments import _tool_error


@pytest.fixture
def reader(tmp_path):
    surface = resolve_tool_surface("libero_placement_direct_overhead_v1")
    runtime = RecoverableArtifactToolRuntime(
        base=SimpleNamespace(surface=surface, definitions=lambda: []), run_root=tmp_path,
        event_path=tmp_path / "events.jsonl",
    )
    catalog = ContextArtifactCatalog(tmp_path, "envelope-test", ArtifactStore(tmp_path))
    runtime.bind_context_artifacts(catalog, event_sink=lambda *args, **kwargs: None)

    def put(data: bytes, media_type: str = "application/json") -> str:
        suffix = {"application/json": ".json", "text/plain": ".txt", "image/png": ".png"}[media_type]
        return catalog.put_bytes(data, kind="context_tool_result", suffix=suffix, media_type=media_type,
                                 round_index=0, action_step=1)["ref"]

    return runtime, put


def _error(call, *args, **kwargs) -> dict:
    with pytest.raises(Exception) as caught:
        call(*args, **kwargs)
    return _tool_error(caught.value)["error"]


# --- artifact_read / observe(ref) (rows 21, 23) --------------------------------

@pytest.mark.parametrize("arguments,field", [
    ({"ref": "artifact://elsewhere/" + "0" * 64}, "ref"),
    ({"ref": "artifact://envelope-test/" + "0" * 64}, "ref"),
    ({"ref": "not-a-ref"}, "ref"),
    ({"ref": "", "limit": 10}, "ref"),
    ({"ref": "R", "limit": 0}, "limit"),
    ({"ref": "R", "limit": ARTIFACT_READ_LIMIT_BYTES + 1}, "limit"),
    ({"ref": "R", "offset": -1}, "offset"),
    ({"ref": "R", "offset": 999}, "offset"),
    ({"ref": "R", "bogus": 1}, "bogus"),
    ({"ref": "R", "json_pointer": 3}, "json_pointer"),
])
def test_reader_argument_errors_are_arguments_by_name(reader, arguments, field):
    runtime, put = reader
    ref = put(b'{"a": 1}')
    arguments = {key: (ref if value == "R" else value) for key, value in arguments.items()}
    error = _error(runtime.invoke, "artifact.read", arguments)
    assert (error["code"], error["field"]) == ("invalid_arguments", field)
    assert error["infrastructure"] is False and error["retry_requires_changed_arguments"] is True


def test_observe_ref_argument_errors_are_arguments(reader):
    runtime, put = reader
    ref = put(b'{"a": 1}')
    extra = _error(runtime.invoke, "observe", {"ref": ref, "views": ["agentview"]})
    assert (extra["code"], extra["field"]) == ("invalid_arguments", "views")
    unknown = _error(runtime.invoke, "observe", {"ref": "artifact://envelope-test/" + "1" * 64})
    assert (unknown["code"], unknown["field"]) == ("invalid_arguments", "ref")


def test_a_store_fault_behind_a_ref_is_infrastructure(reader, tmp_path):
    runtime, put = reader
    ref = put(b'{"a": 1}')
    entry, _data = runtime.catalog.resolve(ref)
    stored = tmp_path / entry["storage_ref"]["path"]
    stored.chmod(0o644)
    stored.write_bytes(b'{"a": 2}')
    error = _error(runtime.invoke, "artifact.read", {"ref": ref})
    assert error["code"] == "artifact_hash_mismatch" and error["infrastructure"] is True
    assert error["retryable"] is False


def test_an_oversized_pointer_value_is_refused_with_its_size_and_keys(reader):
    runtime, put = reader
    ref = put(json.dumps({"needed": "R" * 60_000, "small": 1}).encode())
    error = _error(runtime.invoke, "artifact.read", {"ref": ref, "json_pointer": ""})
    assert (error["code"], error["field"], error["received"]) == ("invalid_arguments", "json_pointer", "")
    assert error["constraints"]["maximum_bytes"] == ARTIFACT_READ_LIMIT_BYTES
    assert error["constraints"]["value_bytes"] > ARTIFACT_READ_LIMIT_BYTES
    assert error["constraints"]["keys"] == ["needed", "small"]
    assert runtime.invoke("artifact.read", {"ref": ref, "json_pointer": "/small"})["json_value"] == 1


def test_an_image_ref_is_refused_by_media_type(reader):
    runtime, put = reader
    ref = put(b"\x89PNG", media_type="image/png")
    error = _error(runtime.invoke, "artifact.read", {"ref": ref})
    assert (error["field"], error["constraints"]) == ("ref", {"media_type": "image/png"})


# --- integral floats (row 22) --------------------------------------------------

SCHEMA = {"name": "mark", "inputSchema": {
    "type": "object", "$defs": {"pixel": {"type": "array", "items": {"type": "integer"}}},
    "properties": {
        "points_uv": {"type": "array", "items": {"$ref": "#/$defs/pixel"}},
        "count": {"type": ["integer", "null"]},
        "radius_m": {"type": "number"},
        "either": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
    },
}}


def test_integral_floats_at_integer_positions_become_ints():
    arguments = {"points_uv": [[316.0, 2.0], [1, 4.0]], "count": 3.0, "radius_m": 2.0, "either": 7.0}
    coerced = coerce_integral_arguments([SCHEMA], "mark", arguments)
    assert coerced == {"points_uv": [[316, 2], [1, 4]], "count": 3, "radius_m": 2.0, "either": 7}
    assert all(type(value) is int for pixel in coerced["points_uv"] for value in pixel)
    assert type(coerced["radius_m"]) is float
    assert arguments["points_uv"][0][0] == 316.0 and type(arguments["count"]) is float


@pytest.mark.parametrize("value", [316.5, float("inf"), float("nan")])
def test_a_fractional_or_non_finite_float_is_left_for_the_validator(value):
    coerced = coerce_integral_arguments([SCHEMA], "mark", {"count": value})
    assert coerced["count"] is value


def test_unchanged_arguments_are_returned_as_they_came():
    arguments = {"points_uv": [[1, 2]], "radius_m": 0.5}
    assert coerce_integral_arguments([SCHEMA], "mark", arguments) is arguments
    assert coerce_integral_arguments([SCHEMA], "other", {"count": 1.0}) == {"count": 1.0}


def test_the_gateway_hands_the_handler_ints():
    from ora.agent.native.gateway import NativeToolGateway

    seen = []
    runtime = SimpleNamespace(
        surface=SimpleNamespace(capabilities=()), allowed_tools=("mark",),
        definitions=lambda: [SCHEMA],
        invoke=lambda name, arguments: seen.append(arguments) or {"ok": True},
    )
    NativeToolGateway(runtime).execute("mark", {"points_uv": [[316.0, 2.0]]})
    assert seen == [{"points_uv": [[316, 2]]}]
    assert all(type(value) is int for arguments in seen for value in arguments["points_uv"][0])

