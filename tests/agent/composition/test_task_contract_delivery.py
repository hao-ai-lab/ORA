"""A reviewed completion contract is delivered, or its absence is recorded.

The defect these tests exist to prevent was silent: fifty reviewed RoboTwin
contracts sat in a qualification directory while the episode-task template told
the policy to "use the exact task language and any separately advertised public
completion contract as the objective". Nothing advertised one, nothing said so,
and every check stayed green because the only assertion about contracts was
that the file existed.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from ora.agent.composition.task_contract import (
    REGISTRY_SCHEMA,
    REGISTRIES,
    resolve_task_contract_layer,
)
from ora.orchestrator import defaults
from ora.agent.composition.baseline import load_native_baseline
from tests.support.run_specs import native_spec

ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = ROOT / REGISTRIES["robotwin"]
REGISTRY = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
DEFAULT_REVISION = load_native_baseline(ROOT)["instances"]["robotwin"]["runtime"]["required_revision"]

# Numbers, comparison words and evaluator vocabulary are the leak this content
# must never carry: the contract states requirements, the evaluator states
# thresholds, and only the environment returns a verdict.
FORBIDDEN_NUMBER = re.compile(r"\d")
FORBIDDEN_WORDS = re.compile(
    r"\b(within|at least|at most|no more than|tolerance|threshold|epsilon"
    r"|check_success|eval_success|predicate|quaternion|qpos)\b",
    re.IGNORECASE,
)


def _robotwin_runtime(task: str, revision: str = DEFAULT_REVISION) -> dict:
    return {"kind": "robotwin", "task": task, "required_revision": revision}


def test_registry_entries_resolve_to_unchanged_reviewed_layers():
    assert REGISTRY["schema"] == REGISTRY_SCHEMA
    tasks = REGISTRY["tasks"]
    assert tasks, "an empty registry would make every delivery assertion vacuous"
    for task, entry in tasks.items():
        layer, disclosure = resolve_task_contract_layer(_robotwin_runtime(task), root=ROOT)
        assert layer is not None, f"{task} is registered but did not resolve"
        assert disclosure["delivered"] is True
        assert (ROOT / layer).is_file()
        assert layer == entry["path"]
        assert entry["check_success_sha256"]
        assert entry["clause_count"] >= 1


@pytest.mark.parametrize("task", sorted(REGISTRY["tasks"]))
def test_contract_text_states_requirements_without_evaluator_detail(task):
    text = (ROOT / REGISTRY["tasks"][task]["path"]).read_text(encoding="utf-8")
    assert text.strip(), f"{task} contract layer is empty"
    number = FORBIDDEN_NUMBER.search(text)
    assert number is None, (
        f"{task} contract leaks a numeric detail at {number.start()}: "
        "contracts carry semantic requirements, never coordinates or tolerances"
    )
    word = FORBIDDEN_WORDS.search(text)
    assert word is None, f"{task} contract leaks evaluator vocabulary: {word.group(0)!r}"


def test_a_registered_task_reaches_the_episode_task_layer():
    """The delivery invariant: registered plus pinned means actually sent."""
    task = sorted(REGISTRY["tasks"])[0]
    spec = native_spec("robotwin")
    spec["runtime"]["task"] = task          # the baseline supplies the revision
    resolved = defaults.expand_native_defaults(spec, root=ROOT)
    assert resolved["runtime"]["required_revision"] == DEFAULT_REVISION
    layers = resolved["policy_context"]["episode_task"]
    assert len(layers) == 2, "the reviewed contract must follow the task template"
    assert layers[1] == REGISTRY["tasks"][task]["path"]
    disclosure = resolved["policy"]["task_contract"]
    assert disclosure["delivered"] is True
    assert disclosure["task"] == task
    assert disclosure["check_success_sha256"] == REGISTRY["tasks"][task]["check_success_sha256"]
    assert DEFAULT_REVISION in disclosure["verified_revisions"]


@pytest.mark.parametrize("runtime,expected", [
    (_robotwin_runtime("place_dual_shoes", "0" * 40),
     "contract_verified_against_a_different_source_revision"),
    ({"kind": "robotwin", "required_revision": DEFAULT_REVISION},
     "runtime_task_identity_unavailable"),
    (_robotwin_runtime("a_task_that_has_no_reviewed_contract"),
     "task_absent_from_verified_contract_registry"),
    ({"kind": "libero_pro", "task": "libero_spatial-t00"},
     "backend_has_no_reviewed_task_contract_registry"),
])
def test_a_withheld_contract_is_always_disclosed_and_never_silent(runtime, expected):
    layer, disclosure = resolve_task_contract_layer(runtime, root=ROOT)
    assert layer is None
    assert disclosure["delivered"] is False
    assert disclosure["reason"] == expected, "every withheld state names its own reason"


def test_an_unregistered_task_leaves_the_template_alone_and_still_reports():
    """The withheld path must also reach the plan, not just the resolver."""
    spec = native_spec("robotwin")
    spec["runtime"]["task"] = "a_task_that_has_no_reviewed_contract"
    resolved = defaults.expand_native_defaults(spec, root=ROOT)
    assert len(resolved["policy_context"]["episode_task"]) == 1
    disclosure = resolved["policy"]["task_contract"]
    assert disclosure["delivered"] is False
    assert disclosure["reason"] == "task_absent_from_verified_contract_registry"
    assert disclosure["runtime_revision"] == DEFAULT_REVISION


def test_the_template_still_promises_what_the_registry_now_delivers():
    """The template's promise and the registry must not drift apart again."""
    template = (
        ROOT / "ora/policy/prompt_bundles/templates/episode-task-v2.md"
    ).read_text(encoding="utf-8")
    assert "completion" in template.lower() and "contract" in template.lower(), (
        "the episode-task template advertises a completion contract; if that "
        "sentence is removed, this delivery mechanism must be removed with it"
    )


def test_every_reviewed_contract_is_registered_or_deliberately_excluded():
    """A contract may be unverified, but it may not be forgotten."""
    from ora.environment.benchmarks.robotwin import TASKS

    # The fifty-task roster the reviewed contract sources covered.
    audited = sorted(TASKS)
    registered = set(REGISTRY["tasks"])
    excluded = [task for task in audited if task not in registered]
    note = REGISTRY.get("note", "")
    assert note.strip(), "the registry must say why an audited task can be absent"
    assert registered, "no reviewed contract is registered"
    assert set(registered) <= set(audited), (
        f"registered without a reviewed source: {sorted(registered - set(audited))}"
    )
    # Excluded tasks are expected (their predicate moved); the point is that the
    # exclusion is derivable from the registry rather than invisible.
    for task in excluded:
        layer, disclosure = resolve_task_contract_layer(_robotwin_runtime(task), root=ROOT)
        assert layer is None
        assert disclosure["reason"] == "task_absent_from_verified_contract_registry"


def test_the_disclosure_alone_identifies_the_text_that_was_sent():
    """The disclosure is the partition key for behavioral comparison.

    Delivering a contract changes the objective text the policy receives while
    leaving the ToolSurface identity byte-identical, so runs on either side of
    that boundary are different treatments. A reader partitioning results must
    not have to reach into policy_context to learn which text was sent.
    """
    task = sorted(REGISTRY["tasks"])[0]
    _, disclosure = resolve_task_contract_layer(_robotwin_runtime(task), root=ROOT)
    assert disclosure["contract_sha256"] == REGISTRY["tasks"][task]["sha256"]

    spec = native_spec("robotwin")
    spec["runtime"]["task"] = task
    resolved = defaults.expand_native_defaults(spec, root=ROOT)
    layer = resolved["policy_context"]["episode_task"][1]
    assert layer == REGISTRY["tasks"][task]["path"]


def _explicit_registered_spec():
    """Match a package that builds its own context and selects a surface."""
    spec = native_spec("robotwin")
    spec["runtime"]["task"] = sorted(REGISTRY["tasks"])[0]
    spec = defaults.expand_native_defaults(spec, root=ROOT)
    spec["policy"].pop("baseline")
    return spec


def test_explicit_surface_cannot_omit_its_deliverable_contract():
    from ora.orchestrator.spec import RunSpecError, resolve_run_spec

    spec = _explicit_registered_spec()
    spec["policy_context"]["episode_task"] = spec["policy_context"]["episode_task"][:1]
    spec["policy"].pop("task_contract")
    with pytest.raises(RunSpecError, match="episode_task does not carry"):
        resolve_run_spec(spec, root=ROOT)


def test_shared_resolver_supplies_disclosure_for_explicit_contract_delivery():
    from ora.orchestrator.spec import resolve_run_spec

    spec = _explicit_registered_spec()
    expected = spec["policy"].pop("task_contract")
    plan = resolve_run_spec(spec, root=ROOT)
    assert plan["policy"]["task_contract"] == expected
    assert "task_contract" not in spec["policy"]


def test_explicit_surface_cannot_forge_contract_disclosure():
    from ora.orchestrator.spec import RunSpecError, resolve_run_spec

    spec = _explicit_registered_spec()
    spec["policy"]["task_contract"]["delivered"] = False
    with pytest.raises(RunSpecError, match="policy.task_contract disagrees"):
        resolve_run_spec(spec, root=ROOT)


def test_unbound_task_does_not_require_a_registry_in_an_unrelated_root(tmp_path):
    from ora.agent.composition.task_contract import TaskContractError

    layer, disclosure = resolve_task_contract_layer({"kind": "robotwin"}, root=tmp_path)
    assert layer is None
    assert disclosure["reason"] == "runtime_task_identity_unavailable"
    with pytest.raises(TaskContractError, match="registry is missing"):
        resolve_task_contract_layer(_robotwin_runtime("place_dual_shoes"), root=tmp_path)
