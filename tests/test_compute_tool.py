"""Restricted geometry evaluator: acceptance, rejection, isolation and leak contracts.

Offline only. The evaluator is a pure function with no physical dependency, so
every contract here is proved against fixtures rather than a simulator; the one
process this suite starts is the evaluator's own warm worker.

Two of these tests exist because the failure they guard against invalidates an
entire run rather than one call: a block that binds a name matching
``FORBIDDEN_FRAGMENTS`` must not put that name on a key path, and no result may
carry a JSON-parseable string that the causal projection auditor would open.
Both are checked against the real scanners, not a local reimplementation.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from ora.agent.context.projection import _causal_projection_violations
from ora.agent.evidence.trajectory import _forbidden_path
from ora.agent.tools.compute import (
    CHILD_DATA_LIMIT_BYTES,
    MAX_OUTPUTS,
    MAX_POW_EXPONENT,
    MAX_SOURCE_BYTES,
    MAX_VALUE_ELEMENTS,
    RESERVED_NAMES,
    RestrictedComputeRuntime,
    validate_source,
)
from ora.agent.tools.robot import ToolError


# The measured teacher derivation this unit exists to replace: quaternion to
# approach axis and separation angle. Values are the ones a source trace shows
# the policy deriving by hand across several thousand reasoning characters.
MEASURED_BLOCK = """
q = np.asarray([0.0, 0.0, 0.7106, 0.7036])
q = q / np.linalg.norm(q)
R = Rotation.from_quat(q)
approach = R.apply([1.0, 0.0, 0.0])
angle_deg = np.degrees(np.arccos(np.clip(np.dot(approach, [0.0, 0.0, 1.0]), -1, 1)))
"""


@pytest.fixture(scope="module")
def runtime():
    instance = RestrictedComputeRuntime()
    try:
        yield instance
    finally:
        instance.close()


def _named(payload: dict[str, object]) -> dict[str, object]:
    return {entry["name"]: entry["value"] for entry in payload["outputs"]}


def _visible(payload: dict[str, object]) -> dict[str, object]:
    """The half that reaches the model, as `native/session.py` builds it.

    `session.py::_tool_message` pops `_runtime` before serialising the tool
    message, and the V3 branch projects only declared result properties. Auditing
    the raw payload instead would test a string that is never put on the wire --
    and would fail, because `schema` and `sha256` are forbidden keys in the
    causal projection and the provenance receipt carries both.
    """

    visible = json.loads(json.dumps(payload))
    visible.pop("_runtime", None)
    return visible


# ---------------------------------------------------------------------------
# Accepted blocks
# ---------------------------------------------------------------------------

def test_measured_derivation_reproduces_the_traced_hand_computation(runtime):
    values = _named(runtime.evaluate(MEASURED_BLOCK))
    assert values["approach"] == pytest.approx([-0.009899, 0.999951, 0.0], abs=1e-5)
    assert values["angle_deg"] == pytest.approx(90.0, abs=1e-6)


def test_bindings_accept_both_references_and_literals(runtime):
    payload = runtime.evaluate(
        "delta_deg = np.degrees(np.linalg.norm((Rotation.from_quat(a).inv() "
        "* Rotation.from_quat(b)).as_rotvec()))",
        {"a": [0.0, 0.0, 0.0, 1.0], "b": [0.0, 0.0, 0.7071068, 0.7071068]},
    )
    assert _named(payload)["delta_deg"] == pytest.approx(90.0, abs=1e-3)


def test_axis_extrapolation_returns_a_world_frame_point(runtime):
    payload = runtime.evaluate(
        "R = Rotation.from_quat(q)\n"
        "approach = R.apply([0.0, 0.0, 1.0])\n"
        "fingertip = np.asarray(tcp) + approach * 0.0092",
        {"q": [0.0, 0.0, 0.0, 1.0], "tcp": [0.02, -0.15, 0.72]},
    )
    values = _named(payload)
    assert values["fingertip"] == pytest.approx([0.02, -0.15, 0.7292], abs=1e-6)


def test_trailing_expression_is_returned_as_result(runtime):
    payload = runtime.evaluate("np.degrees(np.arctan2(0.5, 0.5))")
    assert _named(payload) == {"result": pytest.approx(45.0)}


def test_every_output_entry_declares_its_kind(runtime):
    payload = runtime.evaluate(MEASURED_BLOCK)
    kinds = {entry["name"]: entry["kind"] for entry in payload["outputs"]}
    assert kinds["q"] == "vector4"
    assert kinds["approach"] == "vector3"
    assert kinds["angle_deg"] == "scalar"


# ---------------------------------------------------------------------------
# Intermediates are named, never silently dropped
# ---------------------------------------------------------------------------

def test_non_numeric_intermediates_are_named_rather_than_dropped(runtime):
    payload = runtime.evaluate(MEASURED_BLOCK)
    assert "R" not in _named(payload)
    assert "R" in payload["intermediates_not_returned"]


def test_a_block_returning_only_an_intermediate_is_an_error(runtime):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("R = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])")
    assert caught.value.code == "invalid_arguments"
    assert "intermediate" in str(caught.value)


# ---------------------------------------------------------------------------
# No state survives a call
# ---------------------------------------------------------------------------

def test_bindings_do_not_survive_into_the_next_call(runtime):
    runtime.evaluate(MEASURED_BLOCK)
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("leaked = approach")
    assert caught.value.code == "invalid_arguments"
    assert "approach" in str(caught.value)


def test_repeated_identical_calls_are_independent(runtime):
    """A name bound by one call cannot be read, extended or accumulated by the next."""

    assert _named(runtime.evaluate("counter = 1.0")) == {"counter": 1.0}
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("counter = counter + 1.0")
    assert "counter" in str(caught.value)
    assert _named(runtime.evaluate("counter = 1.0")) == {"counter": 1.0}


# ---------------------------------------------------------------------------
# Rejections — every one must be policy-fixable
# ---------------------------------------------------------------------------

REJECTED = [
    ("syntax", "q = np.asarray([1, 2,"),
    ("import", "import os\nx = 1.0"),
    ("from_import", "from os import path\nx = 1.0"),
    ("lambda", "f = lambda x: x"),
    ("function_def", "def f(x):\n    return x"),
    ("class_def", "class C:\n    pass"),
    ("comprehension", "xs = [i for i in [1, 2, 3]]"),
    ("for_loop", "for i in [1]:\n    x = i"),
    ("while_loop", "while True:\n    x = 1"),
    ("with_block", "with open('x') as fh:\n    x = 1"),
    ("try_block", "try:\n    x = 1\nexcept Exception:\n    x = 2"),
    ("dict_literal", "d = {'a': 1}"),
    ("fstring", "s = f'{1}'"),
    ("walrus", "x = (y := 1.0)"),
    ("dunder_attribute", "x = ().__class__"),
    ("dunder_subclasses", "x = ().__class__.__bases__"),
    ("builtin_open", "x = open('/etc/passwd')"),
    ("builtin_eval", "x = eval('1')"),
    ("module_not_exposed", "x = np.load('/etc/passwd')"),
    ("nested_module_not_exposed", "x = np.linalg.matrix_power"),
    ("undefined_name", "x = undefined_quantity + 1.0"),
    ("unpack_into_attribute", "np.pi, b = 1.0, 2.0"),
    ("attribute_assignment", "np.pi = 1.0"),
]


@pytest.mark.parametrize("label,source", REJECTED, ids=[case[0] for case in REJECTED])
def test_rejected_blocks_are_policy_fixable(runtime, label, source):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate(source)
    error = caught.value.as_dict()
    assert error["code"] == "invalid_arguments", f"{label} reported {error['code']}"
    assert error["retryable"] is True
    assert error["message"], f"{label} carried no message"


def test_reserved_names_cannot_be_rebound(runtime):
    for name in sorted(RESERVED_NAMES):
        with pytest.raises(ToolError) as caught:
            runtime.evaluate(f"{name} = 1.0")
        assert caught.value.code == "invalid_arguments"
        assert name in str(caught.value.received or "")


# ---------------------------------------------------------------------------
# Attribute reachability — the tables are an allowlist, and so is this
# ---------------------------------------------------------------------------

#: Reached from a *value* rather than from the module tables, so the table
#: allowlist never sees them. Each breaks a contract this evaluator states.
ESCAPES_VIA_INSTANCE_ATTRIBUTE = [
    # File and memory access the layer-2 table deliberately withholds on `np`.
    ("ndarray_tofile", 'x = np.zeros(3)\ny = x.tofile("/tmp/pwn")\nz = 1.0', "tofile"),
    ("ndarray_dump", 'x = np.zeros(3)\ny = x.dump("/tmp/pwn")\nz = 1.0', "dump"),
    ("ndarray_ctypes", "x = np.zeros(3)\np = x.ctypes\nz = 1.0", "ctypes"),
    ("ndarray_tobytes", "x = np.zeros(3)\nb = x.tobytes()\nz = 1.0", "tobytes"),
    ("ndarray_view_retype", "x = np.zeros(3)\nv = x.view()\nz = 1.0", "view"),
    # Randomness, which the boundary withholds: a receipt that cannot be
    # recomputed offline is not a receipt.
    ("rotation_random", "R = Rotation.identity()\nq = R.random().as_quat()", "random"),
    ("rotation_create_group", 'R = Rotation.identity()\ng = R.create_group("C1")\nz = g.as_quat()', "create_group"),
]


@pytest.mark.parametrize(
    "label,source,attribute",
    ESCAPES_VIA_INSTANCE_ATTRIBUTE,
    ids=[case[0] for case in ESCAPES_VIA_INSTANCE_ATTRIBUTE],
)
def test_instance_attributes_outside_the_allowlist_are_refused(runtime, label, source, attribute):
    """Rejecting only dunder attributes leaves these reachable.

    ``np.load`` and ``np.fromfile`` are withheld from the table, but
    ``ndarray.tofile`` and ``ndarray.dump`` are the same capability reached from
    a value the block already produced, and ``Rotation.identity().random()``
    reaches the classmethod through the instance.

    Empty builtins alone is not a defence: numpy happens to fail these with
    ``KeyError: '__import__'`` from its own internal import, which is an
    accident of the dependency and reaches the policy as a message it cannot
    act on. The refusal must name the attribute.
    """

    with pytest.raises(ToolError) as caught:
        runtime.evaluate(source)
    error = caught.value.as_dict()
    assert error["code"] == "invalid_arguments", f"{label} reported {error['code']}"
    assert error["retryable"] is True
    assert error["received"] == attribute, (
        f"{label} must name the refused attribute, not report {error['received']!r}"
    )
    assert "__import__" not in error["message"], (
        f"{label} reported a dependency's internal failure rather than a decision"
    )


def test_randomness_is_unreachable_by_any_route(runtime):
    """The evaluator's receipt is re-executable offline only if it is a pure function."""

    for source in (
        "q = Rotation.random().as_quat()",
        "R = Rotation.identity()\nq = R.random().as_quat()",
        "x = np.random.rand(3)",
    ):
        with pytest.raises(ToolError) as caught:
            runtime.evaluate(source)
        assert caught.value.code == "invalid_arguments"


def test_the_same_block_returns_the_same_numbers(runtime):
    """Purity, asserted rather than assumed."""

    first = _named(runtime.evaluate(MEASURED_BLOCK))
    second = _named(runtime.evaluate(MEASURED_BLOCK))
    assert first == second


def test_every_table_entry_stays_reachable(runtime):
    """The allowlist is derived from the tables, so it cannot fall behind them.

    A restated list would let a table entry become unreachable the moment it is
    added -- the failure mode would be a tool that advertises a function the
    evaluator refuses.
    """

    from ora.agent.tools.compute import (
        _LINALG, _NUMPY, _ROTATION, _reachable_attributes,
    )

    reachable = _reachable_attributes()
    for table in (_NUMPY, _LINALG, _ROTATION):
        for entry in dir(table):
            assert entry in reachable, f"{entry} is advertised by a table but unreachable"


#: One evaluable call per published instance attribute, receiver included.
#:
#: Restating the list would let an entry drop out of coverage the moment it is
#: added, so the test walks ``_INSTANCE_ATTRIBUTES`` itself and fails on an
#: entry that has no case here.
_INSTANCE_CALLS = {
    "T": "{a}.T", "shape": "{a}.shape", "ndim": "{a}.ndim", "size": "{a}.size",
    "real": "{a}.real", "imag": "{a}.imag",
    "reshape": "{a}.reshape(4)", "ravel": "{a}.ravel()", "flatten": "{a}.flatten()",
    "transpose": "{a}.transpose()", "squeeze": "{a}.squeeze()", "copy": "{a}.copy()",
    "tolist": "{a}.tolist()", "item": "{a}.item(0)", "astype": "{a}.astype('f')",
    "diagonal": "{a}.diagonal()", "trace": "{a}.trace()",
    "sum": "{a}.sum()", "mean": "{a}.mean()", "min": "{a}.min()", "max": "{a}.max()",
    "argmin": "{a}.argmin()", "argmax": "{a}.argmax()", "prod": "{a}.prod()",
    "std": "{a}.std()", "var": "{a}.var()", "cumsum": "{a}.cumsum()",
    "dot": "{a}.dot({a})", "round": "{a}.round(2)", "clip": "{a}.clip(0.0, 1.0)",
    "conj": "{a}.conj()", "any": "{a}.any()", "all": "{a}.all()",
    "sort": "{a}.sort()", "argsort": "{a}.argsort()", "nonzero": "{a}.nonzero()",
    "take": "{a}.take([0])", "repeat": "{a}.repeat(2)",
    "apply": "{r}.apply([1.0, 0.0, 0.0])", "inv": "{r}.inv().as_quat()",
    "magnitude": "{r}.magnitude()", "approx_equal": "{r}.approx_equal({r})",
    "as_quat": "{r}.as_quat()", "as_matrix": "{r}.as_matrix()",
    "as_rotvec": "{r}.as_rotvec()", "as_euler": "{r}.as_euler('xyz')",
    "as_mrp": "{r}.as_mrp()",
    "as_davenport": "{r}.as_davenport(np.eye(3), 'e', True)",
}


def test_a_comparison_returns_its_booleans(runtime):
    """The dtype gate admits `b`, so the serialiser has to be able to return it.

    `Compare` is an allowed node and a margin test is the natural way to ask
    whether something fits, so the block a policy writes first produced
    `UFuncTypeError: Cannot cast ufunc 'multiply' output ... to dtype('bool')`
    from inside `np.round` -- a dependency's internal cast rule reaching the
    policy as the reason its comparison failed.
    """

    payload = runtime.evaluate(
        "clearance = np.asarray([0.02, 0.001])\n"
        "clear = clearance > 0.01\n"
        "ranking = clearance.argsort()"
    )
    named = _named(payload)
    assert named["clear"] == [True, False]
    # Integers come back exact rather than rounded through a float.
    assert named["ranking"] == [1, 0]
    assert named["clearance"] == [0.02, 0.001]


def test_every_published_instance_attribute_actually_evaluates(runtime):
    """Passing the allowlist is not the same claim as returning a number.

    `test_every_table_entry_stays_reachable` asserts that a name gets through
    the gate. It cannot see an attribute that the gate admits and numpy then
    refuses: ten of these dispatch through `numpy.core._methods`, which imports
    on first use, and under the child's empty builtins that import raised
    `KeyError: '__import__'` -- an advertised capability that no policy could
    use and no test could see, `clip` among them.

    So this one runs a block per entry and requires a value back.
    """

    from ora.agent.tools.compute import _INSTANCE_ATTRIBUTES

    uncovered = sorted(set(_INSTANCE_ATTRIBUTES) - set(_INSTANCE_CALLS))
    assert not uncovered, f"published but not exercised here: {uncovered}"

    array = "np.asarray([[1.0, 2.0], [3.0, 4.0]])"
    rotation = "Rotation.from_quat([0.0, 0.0, 0.0, 1.0])"
    for entry in sorted(_INSTANCE_ATTRIBUTES):
        call = _INSTANCE_CALLS[entry].format(a=array, r=rotation)
        # `sort` and `inv` return None or an object, so bind something numeric
        # beside them; the evaluator refuses a block that returns no measurement.
        block = f"value = {call}\nprobe = np.asarray([1.0]).tolist()"
        try:
            payload = runtime.evaluate(block)
        except Exception as error:  # noqa: BLE001 - the message is the evidence
            raise AssertionError(
                f"{entry} is published by _INSTANCE_ATTRIBUTES but raised: {error}"
            ) from error
        assert payload["outputs"], f"{entry} evaluated but returned nothing"


def test_documented_conversions_still_evaluate(runtime):
    """The allowlist must not cost the uses the tool exists for."""

    payload = runtime.evaluate(
        "R = Rotation.from_quat(q)\n"
        "m = R.as_matrix()\n"
        "axis = R.apply([0.0, 0.0, 1.0])\n"
        "back = R.inv().as_rotvec()\n"
        "deg = np.degrees(R.magnitude())\n"
        "flat = np.asarray(axis).round(6)",
        {"q": [0.0, 0.0, 0.7071068, 0.7071068]},
    )
    values = _named(payload)
    assert values["deg"] == pytest.approx(90.0, abs=1e-4)
    assert values["axis"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)


def test_unbounded_exponent_is_refused_before_execution():
    with pytest.raises(ToolError) as caught:
        validate_source("x = 10 ** 10 ** 10")
    assert caught.value.code == "invalid_arguments"

    with pytest.raises(ToolError) as caught:
        validate_source(f"x = 2.0 ** {MAX_POW_EXPONENT + 1}")
    assert caught.value.constraints["maximum_absolute_exponent"] == MAX_POW_EXPONENT


def test_oversized_source_is_refused_before_parsing():
    with pytest.raises(ToolError) as caught:
        validate_source("x = 1.0\n" * MAX_SOURCE_BYTES)
    assert caught.value.constraints["maximum_bytes"] == MAX_SOURCE_BYTES


def test_memory_bomb_is_bounded_by_the_child_limit(runtime):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("x = np.zeros((100000, 100000))")
    error = caught.value.as_dict()
    assert error["code"] == "invalid_arguments"
    assert error["constraints"]["maximum_bytes"] == CHILD_DATA_LIMIT_BYTES


def test_oversized_result_is_refused(runtime):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate(f"x = np.zeros({MAX_VALUE_ELEMENTS + 1})")
    assert caught.value.constraints["maximum_elements"] == MAX_VALUE_ELEMENTS


def test_too_many_outputs_is_refused(runtime):
    block = "\n".join(f"v{index} = 1.0" for index in range(MAX_OUTPUTS + 2))
    with pytest.raises(ToolError) as caught:
        runtime.evaluate(block)
    assert caught.value.constraints["maximum_outputs"] == MAX_OUTPUTS


def test_non_finite_measurements_are_refused(runtime):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("x = np.arccos(2.0)")
    assert caught.value.code == "invalid_arguments"


# ---------------------------------------------------------------------------
# Leak contracts — these guard run validity, not one call
# ---------------------------------------------------------------------------

FORBIDDEN_IDENTIFIERS = ["contact", "segmentation", "task_source", "object_gt", "actor_state"]


@pytest.mark.parametrize("identifier", FORBIDDEN_IDENTIFIERS)
def test_forbidden_identifier_never_reaches_a_key_path(runtime, identifier):
    """A policy may name a binding ``contact``; that must not invalidate the run.

    ``_forbidden_path`` scans key paths, so the result keeps fixed key names and
    carries the policy's identifier as a value.
    """

    payload = runtime.evaluate(f"{identifier} = np.asarray([0.0, 0.0, 1.0])")
    assert _named(payload) == {identifier: [0.0, 0.0, 1.0]}
    assert _forbidden_path(payload) is None
    assert _forbidden_path({"tool_result": payload}) is None


@pytest.mark.parametrize("identifier", FORBIDDEN_IDENTIFIERS)
def test_forbidden_identifier_survives_the_causal_projection_audit(runtime, identifier):
    payload = runtime.evaluate(f"{identifier} = np.asarray([0.0, 0.0, 1.0])")
    message = {"role": "tool", "content": json.dumps(_visible(payload))}
    assert _causal_projection_violations([message]) == []


def test_json_shaped_string_literal_is_not_echoed_into_the_result(runtime):
    """Both auditors open strings that begin with ``{`` or ``[``.

    The evaluator never echoes source, so a JSON-shaped literal cannot become a
    parsed object carrying a forbidden key.
    """

    source = 'label = "{\\"contact\\":1}"\nx = np.asarray([1.0, 2.0, 3.0])'
    payload = runtime.evaluate(source)
    serialised = json.dumps(_visible(payload))
    assert "\"contact\"" not in serialised
    assert "contact" not in serialised
    assert _forbidden_path(payload) is None
    assert _causal_projection_violations(
        [{"role": "tool", "content": serialised}]
    ) == []


def test_result_never_carries_the_source_block(runtime):
    payload = runtime.evaluate(MEASURED_BLOCK)
    serialised = json.dumps(payload)
    assert "Rotation.from_quat" not in serialised
    assert "np.linalg.norm" not in serialised
    assert MEASURED_BLOCK not in serialised


def test_result_keys_are_fixed(runtime):
    payload = runtime.evaluate(MEASURED_BLOCK)
    assert set(payload) <= {"outputs", "intermediates_not_returned", "_runtime"}
    for entry in payload["outputs"]:
        assert set(entry) == {"name", "value", "kind"}


# ---------------------------------------------------------------------------
# Provenance — a receipt that can be recomputed offline
# ---------------------------------------------------------------------------

def test_a_bare_binding_is_recorded_as_a_literal(runtime):
    """The policy may bring its own constants, and the receipt says so."""

    payload = runtime.evaluate("z = np.sum(a)", {"a": [1.0, 2.0, 3.0]})
    inputs = payload["_runtime"]["provenance"]["inputs"]
    assert inputs == [{
        "name": "a", "origin": "literal",
        "sha256": inputs[0]["sha256"],
    }]
    assert _named(payload)["z"] == pytest.approx(6.0)


def test_an_annotated_binding_records_the_result_it_came_from(runtime):
    """Naming the source is optional; when given it is kept verbatim."""

    payload = runtime.evaluate(
        "z = np.sum(a)",
        {"a": {"value": [1.0, 2.0, 3.0], "from": "mark_point.point_world"}},
    )
    entry = payload["_runtime"]["provenance"]["inputs"][0]
    assert entry["name"] == "a"
    assert entry["origin"] == "mark_point.point_world"
    assert _named(payload)["z"] == pytest.approx(6.0)


def test_both_binding_forms_may_appear_in_one_call(runtime):
    """A reference form that were mandatory would block a legitimate constant."""

    payload = runtime.evaluate(
        "z = np.linalg.norm(np.asarray(measured) - np.asarray(nominal))",
        {
            "measured": {"value": [0.1, 0.0, 0.0], "from": "mark_point.point_world"},
            "nominal": [0.0, 0.0, 0.0],
        },
    )
    origins = {
        entry["name"]: entry["origin"]
        for entry in payload["_runtime"]["provenance"]["inputs"]
    }
    assert origins == {"measured": "mark_point.point_world", "nominal": "literal"}


def test_the_receipt_binds_the_block_by_digest_not_by_echo(runtime):
    """The source is recorded in the tool call; the receipt only has to bind to it."""

    payload = runtime.evaluate(MEASURED_BLOCK)
    provenance = payload["_runtime"]["provenance"]
    assert provenance["source_sha256"] == hashlib.sha256(
        json.dumps(MEASURED_BLOCK, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert MEASURED_BLOCK not in json.dumps(payload)


def test_the_same_inputs_produce_the_same_digests(runtime):
    """Offline recomputation needs the receipt to be stable, not merely present."""

    first = runtime.evaluate("z = np.sum(a)", {"a": [1.0, 2.0]})
    second = runtime.evaluate("z = np.sum(a)", {"a": [1.0, 2.0]})
    assert first["_runtime"] == second["_runtime"]
    third = runtime.evaluate("z = np.sum(a)", {"a": [1.0, 3.0]})
    assert third["_runtime"] != first["_runtime"]


def test_a_malformed_origin_is_policy_fixable(runtime):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("z = np.sum(a)", {"a": {"value": [1.0], "from": 7}})
    assert caught.value.code == "invalid_arguments"


def test_provenance_never_reaches_a_key_path_or_the_projection_audit(runtime):
    """The receipt carries policy identifiers, so it obeys the same rule outputs do."""

    payload = runtime.evaluate(
        "z = np.sum(contact)",
        {"contact": {"value": [1.0], "from": "grasp.plan.contact_point"}},
    )
    assert _forbidden_path(payload) is None
    assert _forbidden_path({"tool_result": payload}) is None
    assert _causal_projection_violations(
        [{"role": "tool", "content": json.dumps(_visible(payload))}]
    ) == []
    # The receipt is evidence, not context: stripping `_runtime` must remove
    # every provenance key, because `schema` and `sha256` are forbidden there.
    assert "provenance" not in json.dumps(_visible(payload))


def test_the_shared_runtime_is_one_worker_and_survives_reuse():
    """No tool runtime has a teardown hook, so a per-session worker would leak."""

    from ora.agent.tools.compute import shared_runtime

    first = shared_runtime()
    assert shared_runtime() is first
    assert _named(first.evaluate("z = 2.0 + 2.0"))["z"] == pytest.approx(4.0)



# ---------------------------------------------------------------------------
# Binding names obey the rule the source side already obeys
# ---------------------------------------------------------------------------

SHADOWING_BINDINGS = [
    ("shadow_np", {"np": [1.0, 2.0, 3.0]}),
    ("shadow_rotation", {"Rotation": [1.0]}),
    ("shadow_pi", {"pi": 4.0}),
    ("replace_builtins", {"__builtins__": {"x": 1}}),
    ("dunder_name", {"__class__": [1.0]}),
    ("non_identifier", {"not an identifier": [1.0]}),
    ("empty_name", {"": [1.0]}),
]


@pytest.mark.parametrize(
    "label,bindings", SHADOWING_BINDINGS, ids=[case[0] for case in SHADOWING_BINDINGS]
)
def test_a_binding_cannot_shadow_the_evaluator_namespace(runtime, label, bindings):
    """`RESERVED_NAMES` was enforced on assignment targets but not on bindings.

    A binding is written into the namespace after it is built, so `np` and
    `Rotation` became whatever was passed and `__builtins__` stopped being the
    empty mapping. JSON cannot carry a callable, so this was never a route to
    execution -- but it voided the namespace contract silently, and the one case
    that did fail reported numpy's `KeyError: '__import__'` rather than a reason.
    """

    with pytest.raises(ToolError) as caught:
        runtime.evaluate("z = 1.0 + 1.0", bindings)
    error = caught.value.as_dict()
    assert error["code"] == "invalid_arguments", f"{label} reported {error['code']}"
    assert "__import__" not in error["message"], (
        f"{label} reported a dependency's internal failure rather than a decision"
    )


def test_a_legitimate_binding_name_is_still_accepted(runtime):
    payload = runtime.evaluate("z = np.sum(tcp_offset_m)", {"tcp_offset_m": [1.0, 2.0]})
    assert _named(payload)["z"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Tool-call entry point
# ---------------------------------------------------------------------------

def test_handle_evaluates_a_well_formed_tool_call():
    from ora.agent.tools.compute import handle

    payload = handle({
        "expressions": "gap_m = np.linalg.norm(np.asarray(a) - np.asarray(b))",
        "bindings": {"a": [0.0, 0.0, 0.0], "b": [0.0, 3.0, 4.0]},
    })
    assert _named(payload)["gap_m"] == pytest.approx(5.0)


def test_handle_accepts_a_call_with_no_bindings():
    from ora.agent.tools.compute import handle

    assert _named(handle({"expressions": "z = 2.0 * 3.0"}))["z"] == pytest.approx(6.0)


MALFORMED_CALLS = [
    ("not_an_object", []),
    ("unknown_field", {"expressions": "z = 1.0", "bindinsg": {}}),
    ("bindings_not_an_object", {"expressions": "z = 1.0", "bindings": [1, 2]}),
    ("missing_expressions", {"bindings": {}}),
]


@pytest.mark.parametrize(
    "label,arguments", MALFORMED_CALLS, ids=[case[0] for case in MALFORMED_CALLS]
)
def test_a_malformed_tool_call_is_policy_fixable(label, arguments):
    from ora.agent.tools.compute import handle

    with pytest.raises(ToolError) as caught:
        handle(arguments)
    assert caught.value.code == "invalid_arguments", label
    assert caught.value.retryable is True


# --------------------------------------------------------------------------
# Publication: the surface join the kernel merged without
# --------------------------------------------------------------------------

LIVE_PROFILES = (
    "libero_placement_direct_overhead_v1",
    "robotwin_toolchain_b2_closed_noservo_v2",
    "robodojo_placement_closed_skills_v2",
)


@pytest.mark.parametrize("profile", LIVE_PROFILES)
def test_the_live_surfaces_declare_compute_and_validate_what_they_publish(profile):
    """A published result schema that nothing checks is a guard that cannot fail.

    The tool is read-only by construction on ``native_atomic_causal``: absence
    from ``mutation_tools`` and ``lifecycle_tools`` is the machine-provable form
    of that claim, so it is asserted here rather than described in help text.
    """
    from ora.agent.composition import resolve_tool_surface, validate_schema_instance
    from ora.agent.tools.compute import handle

    surface = resolve_tool_surface(profile)
    assert "compute" in {item["name"] for item in surface.definitions}
    assert "compute" not in surface.mutation_tools
    assert "compute" not in surface.lifecycle_tools

    result = handle({
        "expressions": "gap = np.linalg.norm(np.array(b) - np.array(a))",
        "bindings": [{"name": "a", "value": [0.1, 0.2, 0.3]},
                     {"name": "b", "value": [0.4, 0.2, 0.3]}],
    })
    visible = {key: value for key, value in result.items() if key != "_runtime"}
    validate_schema_instance(visible, surface.result_schemas["compute"])


@pytest.mark.parametrize("profile", LIVE_PROFILES)
def test_the_published_input_schema_carries_no_union(profile):
    """The entry form exists to remove the bare-or-annotated union, not move it.

    ``bindings`` could not be a mapping keyed by the policy's own identifiers:
    ``policy_tool_help_v1`` closes every object node, and a name-per-entry list
    is the shape that closes.
    """
    from ora.agent.composition import resolve_tool_surface

    surface = resolve_tool_surface(profile)
    schema = next(i for i in surface.definitions if i["name"] == "compute")["inputSchema"]
    encoded = json.dumps(schema)
    for forbidden in ("oneOf", "anyOf", "allOf", "$ref", "$defs"):
        assert forbidden not in encoded, f"compute still delivers {forbidden}"
    assert schema["properties"]["bindings"]["items"]["type"] == "object"


def test_both_binding_spellings_reach_the_same_measurement(runtime):
    """The entry form is what a surface publishes; the mapping form still reads."""
    from ora.agent.tools.compute import handle

    block = "d = np.linalg.norm(np.array(p))"
    entries = handle({"expressions": block,
                      "bindings": [{"name": "p", "value": [3.0, 4.0, 0.0],
                                    "from": "mark_point A xyz_m"}]})
    mapping = handle({"expressions": block, "bindings": {"p": [3.0, 4.0, 0.0]}})
    assert _named(entries) == _named(mapping) == {"d": 5.0}
    origins = {item["name"]: item["origin"] for item in entries["_runtime"]["provenance"]["inputs"]}
    assert origins["p"] == "mark_point A xyz_m"


@pytest.mark.parametrize("call,field", [
    ({"expressions": "x = 1", "bindings": [{"value": 1}]}, "bindings[0].name"),
    ({"expressions": "x = 1", "bindings": [{"name": "p"}]}, "bindings[0].value"),
    ({"expressions": "x = 1", "bindings": [{"name": "p", "value": 1, "unit": "m"}]}, "bindings[0]"),
    ({"expressions": "x = 1", "bindings": [{"name": "p", "value": 1},
                                           {"name": "p", "value": 2}]}, "bindings[1].name"),
])
def test_a_binding_refusal_names_the_entry_the_policy_must_repair(call, field):
    """Naming only the block would reproduce the union error this shape removes."""
    from ora.agent.tools.compute import handle

    with pytest.raises(ToolError) as raised:
        handle(call)
    assert raised.value.field == field
    assert raised.value.code == "invalid_arguments"


def test_one_quantity_carries_one_precision_however_it_is_bound(runtime):
    """A result schema cannot state a precision that depends on how it was asked.

    The same subtraction reads 0.30000000000000004 alone and 0.3 as a vector
    element while only the array path rounds, which is float noise a policy has
    to reason about.
    """
    payload = runtime.evaluate(
        "alone = 0.4 - 0.1\ntogether = np.array([0.4, 0.4]) - np.array([0.1, 0.0])", {},
    )
    values = _named(payload)
    assert values["alone"] == values["together"][0] == 0.3


# ---------------------------------------------------------------------------
# Blocks a live policy actually wrote, and the refusals they met
# ---------------------------------------------------------------------------

def test_scaffolding_a_later_line_reads_is_not_a_returned_measurement(runtime):
    """Half of every compute call in one batch was refused, and this was why.

    Sixteen calls, eight refusals; four of them said the block returns more
    measurements than one result may carry. The blocks were not asking for
    twelve answers -- they were building one. Writing a rotation matrix by
    element binds r00 through r22 before anything uses them, and every one
    counted. The policy then rewrote the same computation with fewer names and
    it passed, so the cap cost a round trip and taught nothing.

    A name a later line reads is scaffolding. It is still reported, by name, in
    intermediates_not_returned -- silent omission is the defect this unit
    exists to remove -- but it is not one of the twelve answers.
    """

    block = "\n".join([
        *(f"r{i} = {float(i)} + qx" for i in range(14)),
        "total = " + " + ".join(f"r{i}" for i in range(14)),
    ])
    payload = runtime.evaluate(block, {"qx": 1.0})
    values = _named(payload)
    # The answer the block was written for survives, and the block is answered
    # rather than refused.
    assert values["total"] == pytest.approx(sum(range(14)) + 14.0)
    assert len(payload["outputs"]) <= 12
    # Demotion is not omission: what stepped aside is named, and only as much
    # as the ceiling required stepped aside.
    listed = payload["intermediates_not_returned"]
    listed = listed if isinstance(listed, str) else "; ".join(listed)
    assert "read by a later line" in listed
    assert listed.startswith("r0 ")
    assert "r13" not in listed, "only the overflow is demoted, not every reader"


def test_twelve_terminal_answers_is_still_the_ceiling(runtime):
    with pytest.raises(ToolError, match="more measurements than one result may carry"):
        runtime.evaluate("\n".join(f"a{i} = {float(i)}" for i in range(13)))


def test_a_quaternion_unpacks_the_way_it_is_written(runtime):
    """`x, y, z, w = q` is how a quaternion is read; it was refused as a target."""

    payload = runtime.evaluate(
        "x, y, z, w = q\nnorm = np.sqrt(x*x + y*y + z*z + w*w)",
        {"q": [0.0, 0.0, 0.7071068, 0.7071068]},
    )
    assert _named(payload)["norm"] == pytest.approx(1.0, abs=1e-6)


def test_the_namespace_carries_what_a_rotation_block_reaches_for(runtime):
    """Three refusals were names the evaluator could have bound and did not.

    `np.column_stack` builds a rotation from its columns, `float` appears in
    every hand-written quaternion conversion, and a block asked to `import
    array` for something `np.array` already provides. Each refusal was correct
    and each cost a turn; binding them is cheaper than teaching around them.
    """

    payload = runtime.evaluate(
        "R = np.column_stack([a, b, np.cross(a, b)])\n"
        "trace = float(R[0][0] + R[1][1] + R[2][2])",
        {"a": [1.0, 0.0, 0.0], "b": [0.0, 1.0, 0.0]},
    )
    assert _named(payload)["trace"] == pytest.approx(3.0)


def test_import_is_still_refused_and_says_what_to_use(runtime):
    with pytest.raises(ToolError, match="import"):
        runtime.evaluate("from numpy import array\nv = array([1.0, 2.0, 3.0])")


def test_a_surface_normal_is_computed_rather_than_refused(runtime):
    """The block that was refused for reaching `eigh`, run end to end.

    A plane normal from measured points is the smallest principal axis of
    their scatter. Two live blocks wrote exactly this and were told `eigh` is
    not reachable, so the derivation went back into the reply.
    """

    values = _named(runtime.evaluate(
        "c = pts.mean(axis=0)\n"
        "M = pts - c\n"
        "w, v = np.linalg.eigh(M.T @ M)\n"
        "n = geom.unit(v.T[0])\n"
        "normal = n * np.sign(n[2])",
        {"pts": [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                 [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]},
        outputs=["normal"]))
    assert values["normal"] == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


@pytest.mark.parametrize("source,expected", [
    ("value = abs(-0.045)", 0.045),
    ("value = max(0.01, 0.02)", 0.02),
    ("value = min(0.01, 0.02)", 0.01),
    ("value = round(0.0448964, 3)", 0.045),
])
def test_the_numeric_builtins_a_block_writes_by_reflex_are_bound(runtime, source, expected):
    """Five EXP10A refusals were nothing but `abs` and `max` being absent.

    They are conversions of numbers the block already holds, in the same
    class as `float` and `int`, which are bound for the same reason.
    """

    assert _named(runtime.evaluate(source, outputs=["value"]))["value"] == pytest.approx(expected)


@pytest.mark.parametrize("name", ["abs", "max", "min", "round"])
def test_binding_one_of_those_names_still_works(runtime, name):
    """They are bound but not reserved, so a block already using the word wins."""

    values = _named(runtime.evaluate(f"value = {name} * 2", {name: 1.5}, outputs=["value"]))
    assert values["value"] == pytest.approx(3.0)
    assert name not in RESERVED_NAMES


def test_a_conditional_expression_chooses_without_a_branch(runtime):
    """`a if c else b` binds one name to one number and takes no branch."""

    values = _named(runtime.evaluate(
        "normal = n if n[2] > 0 else -n",
        {"n": [0.0, 0.0, -1.0]}, outputs=["normal"]))
    assert values["normal"] == pytest.approx([0.0, 0.0, 1.0])


def test_the_if_statement_stays_refused_and_names_its_replacement(runtime):
    """Control flow is still out; the refusal now says what to write instead."""

    with pytest.raises(ToolError) as caught:
        runtime.evaluate("n = [0.0, 0.0, -1.0]\nif n[2] < 0:\n    n = [0.0, 0.0, 1.0]")
    error = caught.value
    assert "if statement" in str(error)
    assert "else" in error.constraints.get("recovery", "")


@pytest.mark.parametrize('magnitude', [1e-300, 1.0, 1e300])
def test_geometry_helpers_normalize_without_numeric_overflow(runtime, magnitude):
    values = _named(runtime.evaluate(
        'direction = geom.unit(a)\nangle = geom.angle_deg(a, b)',
        {'a': [magnitude, magnitude, 0.0], 'b': [-magnitude, magnitude, 0.0]},
        outputs=['direction', 'angle']))
    assert values['direction'] == pytest.approx([2 ** -.5, 2 ** -.5, 0.0])
    assert values['angle'] == pytest.approx(90.0)


def test_prebound_geometry_helpers_keep_quaternion_order_and_transform_direction(runtime):
    values = _named(runtime.evaluate(
        'rotated = geom.rotate(q, [1, 0, 0])\n'
        'point = geom.transform_point(t, q, [1, 0, 0])\n'
        'radians = math.radians(90)',
        {'q': [0.0, 0.0, 1.0, 1.0], 't': [1.0, 2.0, 3.0]},
        outputs=['rotated', 'point', 'radians']))
    assert values['rotated'] == pytest.approx([0, 1, 0])
    assert values['point'] == pytest.approx([1, 3, 3])
    assert values['radians'] == pytest.approx(1.570796327)


@pytest.mark.parametrize('source', [
    'geom.unit([0,0,0])', 'geom.angle_deg([0,0,0],[1,0,0])',
    'geom.rotate([0,0,0,0],[1,0,0])', 'geom.unit([1,2])',
    'math.__dict__', 'geom.__dict__', 'math.os', 'geom = 1', 'math = 1',
])
def test_helper_invalid_inputs_and_namespace_escapes_remain_retryable(runtime, source):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate(source)
    assert caught.value.code == 'invalid_arguments' and caught.value.retryable


def test_explicit_outputs_return_small_answer_without_serializing_large_working_array(runtime):
    result = runtime.evaluate('working = np.ones(100)\nanswer = np.mean(working)', outputs=['answer'])
    assert _named(result) == {'answer': 1.0}
    assert 'working (not selected for output)' in result['intermediates_not_returned']
    assert result['_runtime']['provenance']['selected_outputs'] == ['answer']


def test_explicit_outputs_choose_order_and_a_final_expression(runtime):
    assert _named(runtime.evaluate('x = 2\ny = 3\nx+y', outputs=['result'])) == {'result': 5.0}
    result = runtime.evaluate('x = 2\ny = 3', outputs=['y', 'x'])
    assert [entry['name'] for entry in result['outputs']] == ['y', 'x']


@pytest.mark.parametrize('outputs', [[], ['x', 'x'], ['missing'], ['np'], ['_x'], ['x'] * 13, 'x', [1]])
def test_bad_output_selection_names_the_repairable_field(runtime, outputs):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate('x = 1', outputs=outputs)
    assert caught.value.code == 'invalid_arguments' and caught.value.field == 'outputs'


# The two call-format examples every v2 compute description publishes; the test
# asserts each is present verbatim in the delivered description.
PUBLISHED_COMPUTE_EXAMPLES = (
    {"bindings": [{"name": "a", "value": [0.1, 0.2, 0.3]}, {"name": "b", "value": [0.4, 0.6, 0.3]}],
     "expressions": "distance_m = np.linalg.norm(b - a)", "outputs": ["distance_m"]},
    {"bindings": [{"name": "p", "value": [0.1, 0.2, 0.3]}, {"name": "q", "value": [0.0, 0.0, 0.0, 1.0]}],
     "expressions": "axis = geom.rotate(q, [1, 0, 0])\npoint = p + 0.03 * axis\nangle = geom.angle_deg(axis, [0, 0, 1])",
     "outputs": ["point", "angle"]},
)


def test_published_compute_examples_execute_through_every_applicable_surface():
    from types import SimpleNamespace
    from ora.agent.composition.tool_surface import PROFILE_CATALOG, resolve_tool_surface
    from ora.agent.composition import validate_schema_instance
    from ora.agent.tools.compute_arguments import invoke
    checked = []
    for name in PROFILE_CATALOG:
        surface = resolve_tool_surface(name)
        definition = next((d for d in surface.definitions if d['name'] == 'compute'), None)
        if definition is None or 'outputs' not in definition['inputSchema']['properties']:
            continue
        checked.append(name)
        for sample in PUBLISHED_COMPUTE_EXAMPLES:
            validate_schema_instance(sample, definition['inputSchema'])
            assert json.dumps(sample, separators=(',', ':')) in definition['description']
            result = invoke(SimpleNamespace(surface=surface), sample)
            expected = {'distance_m': .5} if len(sample['outputs']) == 1 else {'point': [.13,.2,.3], 'angle': 90.0}
            assert _named(result) == expected
            assert _causal_projection_violations([{'role': 'tool', 'content': json.dumps(_visible(result))}]) == []
    # The three defaults, their composed-tools successors and the RoboTwin
    # prepared-grasp v1 surface all execute the same published examples
    # through their real schema/handler.
    from ora.agent.composition.profile_catalog import COMPOSED_SUCCESSORS
    assert set(checked) == {
        "libero_placement_direct_overhead_v1", "robotwin_toolchain_b2_closed_noservo_v2",
        "robodojo_placement_closed_skills_v2", "robotwin_toolchain_b2_closed_noservo_v1",
        *COMPOSED_SUCCESSORS,
    }


# ---------------------------------------------------------------------------
# An escaped line break is read as the statement separator it stands for
# ---------------------------------------------------------------------------

# The EXP11A g003 shape: a double-escaped JSON string delivers a backslash and
# an `n` where a line break was meant, and the whole block was refused.
ESCAPED_BLOCK = "link_grasp = car - off*appr\\nmid = [-0.19, -0.02, 1.03]"


def test_an_escaped_line_break_separates_statements():
    from ora.agent.tools.compute_arguments import (
        LITERAL_NEWLINE_NORMALIZATION, handle,
    )

    result = handle({
        "expressions": ESCAPED_BLOCK,
        "bindings": [
            {"name": "car", "value": [0.0, 0.0, 1.0]},
            {"name": "off", "value": 0.5},
            {"name": "appr", "value": [0.0, 0.0, 1.0]},
        ],
        "outputs": ["link_grasp", "mid"],
    })
    assert _named(result) == {"link_grasp": [0.0, 0.0, 0.5], "mid": [-0.19, -0.02, 1.03]}
    record = result["_runtime"]["provenance"]
    assert record["source_normalization"] == {
        "kind": LITERAL_NEWLINE_NORMALIZATION,
        "submitted_source_sha256": record["source_normalization"]["submitted_source_sha256"],
    }
    assert record["source_normalization"]["submitted_source_sha256"] != record["source_sha256"]
    assert "normalization" not in json.dumps(_visible(result))


def test_an_escaped_crlf_and_a_real_line_break_are_both_accepted():
    from ora.agent.tools.compute_arguments import handle

    escaped = handle({"expressions": "a = 1.0\\r\\nb = a + 1.0"})
    real = handle({"expressions": "a = 1.0\nb = a + 1.0"})
    assert _named(escaped)["b"] == _named(real)["b"] == pytest.approx(2.0)
    assert "source_normalization" not in real["_runtime"]["provenance"]


@pytest.mark.parametrize("source", [
    # Parses as sent: an escape inside a short label is never rewritten.
    "r = Rotation.from_euler(x, 0.0)\nlabel = len(a\nb)",
])
def test_a_block_that_already_parses_is_never_rewritten(source):
    from ora.agent.tools.compute_arguments import normalize_block

    assert normalize_block(source) == (source, None)


@pytest.mark.parametrize("source, reason", [
    # The rewrite would put a line break inside a quoted label: the original
    # refusal stands rather than a different one.
    ("r = Rotation.from_euler(x\ny, [0.0, 0.0]) +", "not valid Python"),
    # A rewritten block still meets the unchanged AST allowlist.
    ("import os\\nz = 1.0", "import"),
    ("def f(x):\\n    return x\\nz = f(1.0)", "function definition"),
])
def test_an_escaped_line_break_never_widens_the_allowlist(source, reason):
    from ora.agent.tools.compute_arguments import handle

    with pytest.raises(ToolError) as caught:
        handle({"expressions": source})
    assert caught.value.code == "invalid_arguments"
    assert reason in str(caught.value)


# A raw-string tool wire (vLLM DeepSeek DSML) delivers the help's JSON-style
# "\n" as two characters. 29 of 136 cluster episodes lost their first compute
# call to exactly this before the evaluator read them as line breaks.
def test_literal_backslash_n_between_statements_reads_as_line_breaks(runtime):
    payload = runtime.evaluate("a = 1.5\\nb = a * 2\\nc = b + 1")
    assert _named(payload) == {"a": 1.5, "b": 3.0, "c": 4.0}


def test_unrepairable_literal_backslash_n_names_the_cause(runtime):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("a = (1.5\\nb = 2")
    error = caught.value.as_dict()
    assert error["code"] == "invalid_arguments"
    assert "literal backslash" in error["constraints"]["recovery"]


def test_runtime_name_errors_say_what_is_available(runtime):
    with pytest.raises(ToolError) as caught:
        runtime.evaluate("print(1.0)")
    assert "no print" in caught.value.as_dict()["constraints"]["allowed"]
