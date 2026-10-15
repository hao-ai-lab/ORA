"""Tracked YAML declarations parse once per distinct content on hot resolve paths."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ora.orchestrator import resources, yaml_cache

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _cold_cache():
    """Content keys are shared process-wide; each test starts from no parse."""
    yaml_cache.clear()
    yield
    yaml_cache.clear()


def _count_parses(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every YAML parse; `yaml.safe_load` also resolves `load` here."""
    calls: list[object] = []
    original = yaml.load

    def counting(stream, *args, **kwargs):
        calls.append(stream)
        return original(stream, *args, **kwargs)

    monkeypatch.setattr(yaml, "load", counting)
    return calls


def _typed(value):
    """A comparison form that tells 1, 1.0 and True (and list and tuple) apart."""
    if isinstance(value, dict):
        return ("dict", tuple((_typed(key), _typed(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_typed(item) for item in value))
    if isinstance(value, (set, frozenset)):
        return (type(value).__name__, tuple(sorted(repr(_typed(item)) for item in value)))
    return (type(value).__name__, value)


def test_a_parse_is_reused_per_content_not_per_file_state(monkeypatch, tmp_path):
    calls = _count_parses(monkeypatch)
    path = tmp_path / "declaration.yaml"
    path.write_text("value: 1\n", encoding="utf-8")
    assert yaml_cache.load(path) == {"value": 1}
    assert yaml_cache.load(path) == {"value": 1}
    # Rewriting the same bytes (a new modification time) needs no new parse ...
    path.write_text("value: 1\n", encoding="utf-8")
    assert yaml_cache.load(path) == {"value": 1}
    # ... and another path with those bytes shares the parse.
    twin = tmp_path / "twin.yaml"
    twin.write_text("value: 1\n", encoding="utf-8")
    assert yaml_cache.load(twin) == {"value": 1}
    assert len(calls) == 1
    path.write_text("value: 2\n", encoding="utf-8")
    assert yaml_cache.load(path) == {"value": 2}
    assert len(calls) == 2


def test_a_same_size_in_place_rewrite_is_visible_immediately(tmp_path):
    # With an mtime/size/inode key this read the old value in 477 of 500
    # rounds: the kernel stamps mtime from a coarse clock.
    path = tmp_path / "declaration.yaml"
    for round_index in range(200):
        written = round_index % 2
        with path.open("w", encoding="utf-8") as stream:
            stream.write(f"value: {written}\n")
        assert yaml_cache.load(path) == {"value": written}, round_index
        assert yaml_cache.load_frozen(path)["value"] == written, round_index


def test_callers_get_independent_copies_and_the_shared_view_is_read_only(tmp_path):
    path = tmp_path / "declaration.yaml"
    path.write_text(
        "outer:\n  items: [1, 2]\n  tags: !!set {a: null}\n  pairs: !!pairs [{k: {x: 1}}]\n",
        encoding="utf-8",
    )
    owned = yaml_cache.load(path)
    assert type(owned) is dict and type(owned["outer"]["items"]) is list
    # Pair tuples come back as two-item lists (documented on `load`).
    assert owned["outer"]["pairs"] == [["k", {"x": 1}]]
    owned["outer"]["items"].append(3)
    owned["outer"]["tags"].add("b")
    owned["outer"]["pairs"][0][1]["x"] = 2
    owned["extra"] = True
    assert yaml_cache.load(path) == {
        "outer": {"items": [1, 2], "tags": {"a"}, "pairs": [["k", {"x": 1}]]},
    }
    shared = yaml_cache.load_frozen(path)
    assert shared["outer"]["items"] == (1, 2)
    assert shared["outer"]["tags"] == frozenset({"a"})
    with pytest.raises(TypeError):
        shared["extra"] = True
    with pytest.raises(TypeError):
        shared["outer"]["items"] = ()
    # A mapping nested in a pair tuple is frozen too.
    with pytest.raises(TypeError):
        shared["outer"]["pairs"][0][1]["x"] = 2


def test_an_unreadable_malformed_or_recursive_file_is_refused_and_not_cached(tmp_path):
    path = tmp_path / "declaration.yaml"
    with pytest.raises(OSError):
        yaml_cache.load(path)
    path.write_text("value: [unterminated\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        yaml_cache.load(path)
    for recursive in ("value: &x [*x]\n", "value: &x {inner: *x}\n"):
        path.write_text(recursive, encoding="utf-8")
        with pytest.raises(yaml_cache.RecursiveDocumentError):
            yaml_cache.load(path)
    path.write_text("value: [1]\n", encoding="utf-8")
    assert yaml_cache.load(path) == {"value": [1]}


def test_the_loader_prefers_libyaml_and_falls_back_to_the_pure_python_safe_loader():
    class Fast:
        pass

    class Pure:
        pass

    assert yaml_cache.select_safe_loader(SimpleNamespace(CSafeLoader=Fast, SafeLoader=Pure)) is Fast
    assert yaml_cache.select_safe_loader(SimpleNamespace(SafeLoader=Pure)) is Pure
    assert yaml_cache.SAFE_LOADER is getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def test_libyaml_parses_every_tracked_declaration_like_the_pure_python_loader():
    # A resource binding and a RunPlan hash values parsed from these files; the
    # loader swap must not change one value or type, or every hash moves.
    declarations = sorted({
        *resources.HOSTS_ROOT.glob("*.yaml"),
        *ROOT.glob("experiments/*/experiment.yaml"),
        *ROOT.glob("ora/policy/prompt_bundles/**/*.yaml"),
        ROOT / "ora/agent/providers/builtin-profiles.yaml",
    })
    assert ROOT / "experiments/runtime_v1/experiment.yaml" in declarations
    assert ROOT / "ora/policy/prompt_bundles/current_v4/robodojo-x5-native_api-v4.yaml" in declarations
    for path in declarations:
        text = path.read_text(encoding="utf-8")
        pure = yaml.load(text, Loader=yaml.SafeLoader)
        assert _typed(yaml.load(text, Loader=yaml_cache.SAFE_LOADER)) == _typed(pure), path
        assert _typed(yaml_cache.load(path)) == _typed(pure), path


def _host_profile(minimum: int) -> str:
    return yaml.safe_dump({
        "schema": resources.HOST_PROFILE_SCHEMA, "id": "probehost",
        "execution_resources": {"sim-slot": {
            "runtime_kind": "robodojo", "resource_class": "robodojo_sim", "slot_id": "gpu4",
            "worker_environment": {"CUDA_VISIBLE_DEVICES": "4"},
            "minimum_free_memory_mib": minimum,
        }},
    })


def _probe_hosts(monkeypatch, tmp_path: Path, minimum: int) -> Path:
    hosts = tmp_path / "ora/orchestrator/hosts"
    hosts.mkdir(parents=True)
    (hosts / "probehost.yaml").write_text(_host_profile(minimum), encoding="utf-8")
    monkeypatch.setattr(resources, "HOSTS_ROOT", hosts)
    return hosts / "probehost.yaml"


def _resolve() -> dict:
    return resources.resolve_execution_resource("sim-slot", runtime_kind="robodojo", required_host="probehost")


def test_resolving_many_resources_parses_the_host_profile_once(monkeypatch, tmp_path):
    _probe_hosts(monkeypatch, tmp_path, 4096)
    calls = _count_parses(monkeypatch)
    bindings = [_resolve() for _ in range(25)]
    assert len(calls) == 1
    assert all(binding == bindings[0] for binding in bindings)
    # A profile outside the project is recorded by its absolute path.
    assert bindings[0]["source"] == (tmp_path / "ora/orchestrator/hosts/probehost.yaml").resolve().as_posix()
    assert bindings[0]["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "4"}
    # A caller editing its binding cannot reach the shared parse.
    bindings[0]["worker_environment"]["CUDA_VISIBLE_DEVICES"] = "7"
    again = _resolve()
    assert again["worker_environment"] == {"CUDA_VISIBLE_DEVICES": "4"}
    assert again["profile_sha256"] == bindings[1]["profile_sha256"]


def test_an_edited_host_profile_is_read_again_at_once(monkeypatch, tmp_path):
    path = _probe_hosts(monkeypatch, tmp_path, 4096)
    first = _resolve()
    path.write_text(_host_profile(8192), encoding="utf-8")
    second = _resolve()
    assert (first["minimum_free_memory_mib"], second["minimum_free_memory_mib"]) == (4096, 8192)
    assert first["profile_sha256"] != second["profile_sha256"]


@pytest.mark.parametrize("content", [
    "schema: [unterminated\n",
    "schema: &x [*x]\n",
    b"schema: \xff\xfe\n",
])
def test_a_malformed_host_profile_still_fails_closed(monkeypatch, tmp_path, content):
    path = _probe_hosts(monkeypatch, tmp_path, 4096)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(resources.ExecutionResourceError, match="cannot load execution host profile"):
        _resolve()
