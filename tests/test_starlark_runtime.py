"""The Starlark boundary for composed tool bodies: dialect, limits and replay."""

from __future__ import annotations

import math
import time

import pytest

from ora.agent.tools.composed import (
    Completed,
    StarlarkLimits,
    StarlarkProgramError,
    Suspended,
    load_program,
)

HOST = {
    "sdk.observe": "query",
    "sdk.check.hold": "query",
    "sdk.gripper": "action",
    "sdk.move": "action",
}

BODY = """
def plan(inputs):
    seen = sdk.observe(role = "head")
    return {"target": seen["z"] + inputs["lift_m"], "arm": inputs["arm"]}

def execute(c):
    results = []
    for name in ["open", "approach"]:
        r = sdk.move(c["arm"], name, stage = name) if name != "open" else sdk.gripper(c["arm"], "open", stage = name)
        results.append(r["status"])
        if r["status"] != "reached":
            return {"status": "aborted", "at": name, "results": results}
    for attempt in range(3):
        r = sdk.gripper(c["arm"], "close", stage = "close")
        results.append(r["status"])
        if sdk.check.hold(c["arm"])["verdict"] == "supported":
            return {"status": "completed", "attempts": attempt + 1, "results": results}
    return {"status": "aborted", "at": "close", "results": results}
"""


def _program(source: str = BODY, **kwargs):
    return load_program(source, filename="test.star", host=HOST, **kwargs)


def _queries(hold=("contradicted", "supported")):
    verdicts = list(hold)
    log = []

    def observe(*, role):
        log.append(("observe", role))
        return {"z": 0.25}

    def check_hold(arm):
        log.append(("hold", arm))
        return {"verdict": verdicts.pop(0)}

    return {"sdk.observe": observe, "sdk.check.hold": check_hold}, log


def _drive(program, entry, args, queries, outcomes):
    """Run an execute entry to completion the way an executor would."""

    journal, runs, requests = (), 0, []
    while True:
        runs += 1
        result = program.run(entry, args=args, queries=queries, allow_actions=True, journal=journal)
        if isinstance(result, Completed):
            return result, runs, requests
        requests.append((result.request.symbol, result.request.args, result.request.kwargs))
        journal = result.journal + (result.request.resolve(outcomes.pop(0)),)


# ---- dialect and loading -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("def plan():\n    x = 0\n    while x < 3:\n        x += 1\n", "parse_error"),
        ('load("other.star", "f")\n', "load_forbidden"),
        ('def plan():\n    print("x")\n', "evaluation_error"),
        ("def plan():\n    return undeclared(1)\n", "evaluation_error"),
        ("def plan():\n    return __host_0()\n", "reserved_identifier"),
        ('X = sdk.observe(role = "head")\ndef plan():\n    return X\n', "host_call_at_load"),
        ("def other():\n    return 1\n", "missing_entrypoint"),
        ("plan = 3\n", "missing_entrypoint"),
    ],
)
def test_load_refuses_outside_the_dialect(source, code):
    with pytest.raises(StarlarkProgramError) as raised:
        _program(source, entrypoints=("plan",))
    assert raised.value.code == code


def test_host_symbols_are_nested_immutable_namespaces():
    program = _program('def plan():\n    sdk.check.hold = None\n    return 1\n', entrypoints=("plan",))
    queries, _ = _queries()
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("plan", queries=queries, allow_actions=False)
    assert raised.value.code == "evaluation_error"


def test_host_table_is_validated():
    with pytest.raises(ValueError):
        load_program("", filename="t.star", host={"sdk": "query", "sdk.move": "action"}, entrypoints=())
    with pytest.raises(ValueError):
        load_program("", filename="t.star", host={"sdk.Move": "action"}, entrypoints=())
    program = _program()
    with pytest.raises(ValueError, match="queries must bind exactly"):
        program.run("plan", args=[{}], queries={}, allow_actions=False)


def test_digest_binds_the_source_bytes():
    assert _program().digest == _program().digest
    assert _program().digest != _program(BODY + "\n# changed\n").digest


# ---- queries, actions and replay -----------------------------------------------------------------


def test_plan_journals_queries_and_replays_without_calling_them():
    program = _program()
    queries, log = _queries()
    first = program.run("plan", args=[{"lift_m": 0.1, "arm": "left"}], queries=queries, allow_actions=False)
    assert isinstance(first, Completed)
    assert first.value == {"target": pytest.approx(0.35), "arm": "left"}
    assert [(call.symbol, call.kwargs) for call in first.journal] == [("sdk.observe", {"role": "head"})]

    def refuse(**_):
        raise AssertionError("replay must not call the host")

    again = program.run(
        "plan",
        args=[{"lift_m": 0.1, "arm": "left"}],
        queries={"sdk.observe": refuse, "sdk.check.hold": refuse},
        allow_actions=False,
        journal=first.journal,
    )
    assert again.value == first.value and again.journal == first.journal
    assert log == [("observe", "head")]


def test_execute_suspends_at_each_action_and_branches_on_results():
    program = _program()
    queries, log = _queries(hold=("contradicted", "supported"))
    outcomes = [{"status": "reached"}] * 4
    result, runs, requests = _drive(program, "execute", [{"arm": "left"}], queries, outcomes)
    assert result.value == {"status": "completed", "attempts": 2, "results": ["reached"] * 4}
    assert requests == [
        ("sdk.gripper", ["left", "open"], {"stage": "open"}),
        ("sdk.move", ["left", "approach"], {"stage": "approach"}),
        ("sdk.gripper", ["left", "close"], {"stage": "close"}),
        ("sdk.gripper", ["left", "close"], {"stage": "close"}),
    ]
    assert runs == len(requests) + 1
    # Each live query ran exactly once although the body was interpreted five times.
    assert log == [("hold", "left"), ("hold", "left")]
    assert [call.symbol for call in result.journal] == [
        "sdk.gripper",
        "sdk.move",
        "sdk.gripper",
        "sdk.check.hold",
        "sdk.gripper",
        "sdk.check.hold",
    ]


def test_an_unreached_stage_ends_the_body_on_the_recorded_result():
    program = _program()
    queries, _ = _queries()
    result, runs, _ = _drive(program, "execute", [{"arm": "right"}], queries, [{"status": "reached"}, {"status": "not_reached"}])
    assert result.value == {"status": "aborted", "at": "approach", "results": ["reached", "not_reached"]}
    assert runs == 3


def test_same_inputs_and_journal_give_the_same_result():
    program = _program()
    queries, _ = _queries()
    result, _, _ = _drive(program, "execute", [{"arm": "left"}], queries, [{"status": "reached"}] * 4)
    replays = [
        program.run("execute", args=[{"arm": "left"}], queries=queries, allow_actions=True, journal=result.journal)
        for _ in range(3)
    ]
    assert all(r == replays[0] for r in replays)
    assert replays[0].value == result.value


def test_replay_divergence_is_refused():
    program = _program()
    queries, _ = _queries()
    result, _, _ = _drive(program, "execute", [{"arm": "left"}], queries, [{"status": "reached"}] * 4)
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("execute", args=[{"arm": "right"}], queries=queries, allow_actions=True, journal=result.journal)
    assert raised.value.code == "replay_divergence"

    extra = result.journal + (result.journal[0],)
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("execute", args=[{"arm": "left"}], queries=queries, allow_actions=True, journal=extra)
    assert raised.value.code == "replay_divergence"


def test_read_only_entry_cannot_reach_an_action():
    program = _program()
    queries, _ = _queries()
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("execute", args=[{"arm": "left"}], queries=queries, allow_actions=False)
    assert raised.value.code == "action_in_read_only_entry"


def test_suspension_returns_the_request_without_a_result():
    program = _program()
    queries, _ = _queries()
    first = program.run("execute", args=[{"arm": "left"}], queries=queries, allow_actions=True)
    assert isinstance(first, Suspended)
    assert first.journal == ()
    assert first.request.symbol == "sdk.gripper"


# ---- limits and failures -------------------------------------------------------------------------


def _single(source, host=None, **kwargs):
    return load_program(source, filename="t.star", host=host or {}, entrypoints=("f",), **kwargs)


def test_unbounded_range_exhausts_the_tick_budget():
    program = _single("def f():\n    s = 0\n    for i in range(1000000000):\n        s += i\n    return s\n")
    started = time.monotonic()
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", queries={}, allow_actions=False, limits=StarlarkLimits(max_ticks=100))
    assert raised.value.code == "tick_budget_exhausted"
    assert time.monotonic() - started < 5


def test_wall_deadline_includes_host_time():
    def slow():
        time.sleep(0.2)
        return 1

    program = _single(
        "def f():\n    s = slow()\n    for i in range(1000000000):\n        s += i\n    return s\n",
        host={"slow": "query"},
    )
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", queries={"slow": slow}, allow_actions=False, limits=StarlarkLimits(wall_s=0.1))
    assert raised.value.code == "deadline_exceeded"


def test_recursion_is_bounded_by_the_call_stack_limit():
    program = _single("def f(n):\n    return 0 if n == 0 else 1 + f(n - 1)\n", limits=StarlarkLimits(max_callstack=16))
    assert program.run("f", args=[5], queries={}, allow_actions=False).value == 5
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", args=[100], queries={}, allow_actions=False)
    assert raised.value.code == "callstack_exceeded"


def test_fail_is_a_contract_failure():
    program = _single('def f():\n    fail("stage table mismatch")\n')
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", queries={}, allow_actions=False)
    assert raised.value.code == "contract_fail"
    assert "stage table mismatch" in raised.value.detail


def test_host_failure_is_reported_by_symbol():
    def broken():
        raise RuntimeError("camera unavailable")

    program = _single("def f():\n    return broken()\n", host={"broken": "query"})
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", queries={"broken": broken}, allow_actions=False)
    assert raised.value.code == "host_error"
    assert raised.value.detail == "broken: RuntimeError: camera unavailable"


@pytest.mark.parametrize("value", [math.nan, math.inf, object(), {1: "non-string key"}])
def test_host_results_must_be_finite_plain_data(value):
    program = _single("def f():\n    return q()\n", host={"q": "query"})
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", queries={"q": lambda: value}, allow_actions=False)
    assert raised.value.code == "non_data_value"


def test_inputs_and_returns_must_be_plain_data():
    program = _single("def f(x):\n    return x\n")
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", args=[math.nan], queries={}, allow_actions=False)
    assert raised.value.code == "non_data_value"
    returns_function = _single("def f():\n    return f\n")
    with pytest.raises(StarlarkProgramError) as raised:
        returns_function.run("f", queries={}, allow_actions=False)
    assert raised.value.code == "non_data_value"


def test_body_non_finite_floats_leave_starlark_as_none():
    """Pins the binding's lossy edge documented in the module; an upgrade that changes it shows here."""

    program = _single('def f():\n    return [float("nan"), float("inf"), 1.5]\n')
    assert program.run("f", queries={}, allow_actions=False).value == [None, None, 1.5]


# ---- pure symbols, globals and host refusals -----------------------------------------------------


def _pure_program(source, pure=None, **kwargs):
    calls = []

    def double(x):
        calls.append(x)
        return {"kind": "doubled", "value": x * 2}

    host = {"lib.double": "pure", "sdk.observe": "query", "sdk.move": "action"}
    program = load_program(
        source, filename="t.star", host=host, pure=pure or {"lib.double": double}, **kwargs,
    )
    return program, calls


def test_pure_symbols_run_at_load_and_are_never_journaled():
    source = (
        "HEADER = lib.double(21)\n"
        "def plan():\n"
        "    return lib.double(sdk.observe()['z'])\n"
        "def execute():\n"
        "    r = sdk.move(lib.double(1)['value'], stage = 'a')\n"
        "    return lib.double(r['n'])\n"
    )
    program, calls = _pure_program(source, read_globals=("HEADER",))
    assert program.globals == {"HEADER": {"kind": "doubled", "value": 42}}
    planned = program.run("plan", queries={"sdk.observe": lambda: {"z": 4}}, allow_actions=False)
    assert planned.value == {"kind": "doubled", "value": 8}
    assert [call.symbol for call in planned.journal] == ["sdk.observe"]
    first = program.run("execute", queries={"sdk.observe": lambda: {}}, allow_actions=True)
    assert isinstance(first, Suspended) and first.request.args == [2]
    done = program.run(
        "execute", queries={"sdk.observe": lambda: {}}, allow_actions=True,
        journal=first.journal + (first.request.resolve({"n": 5}),),
    )
    assert done.value == {"kind": "doubled", "value": 10}
    assert [call.symbol for call in done.journal] == ["sdk.move"]
    # Recomputed on every interpretation: load, plan, two execute passes (one call each, two on the last).
    assert calls == [21, 4, 1, 1, 5]


def test_pure_binding_must_match_the_host_table():
    with pytest.raises(ValueError, match="pure must bind exactly"):
        load_program("", filename="t.star", host={"lib.double": "pure"}, entrypoints=())


def test_pure_failure_at_load_is_a_host_error_naming_the_symbol():
    def strict(x):
        raise ValueError("x must be positive")

    with pytest.raises(StarlarkProgramError) as raised:
        _pure_program("HEADER = lib.double(-1)\n", pure={"lib.double": strict}, entrypoints=())
    assert raised.value.code == "host_error"
    assert raised.value.detail == "lib.double: ValueError: x must be positive"


def test_missing_global_is_refused():
    with pytest.raises(StarlarkProgramError) as raised:
        _pure_program("def plan():\n    return 1\n", entrypoints=("plan",), read_globals=("TOOL",))
    assert raised.value.code == "missing_global"


def test_host_refusal_keeps_its_code():
    def spent():
        raise StarlarkProgramError("host_refusal", "budget_exhausted:queries")

    program = _single("def f():\n    return q()\n", host={"q": "query"})
    with pytest.raises(StarlarkProgramError) as raised:
        program.run("f", queries={"q": spent}, allow_actions=False)
    assert (raised.value.code, raised.value.detail) == ("host_refusal", "budget_exhausted:queries")
