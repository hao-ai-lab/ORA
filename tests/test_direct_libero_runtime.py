"""The direct LIBERO driver and its RunSpec boundary."""
from __future__ import annotations

import pytest

from ora.orchestrator.libero_configuration import validate_libero_configuration
from ora.orchestrator.spec import RunSpecError, resolve_run_spec
from test_native_baseline import native_spec


def test_direct_libero_reset_official_termination_and_no_auto_reset():
    import numpy as np
    from ora.environment.backends.libero_pro.direct import DirectLiberoEnvironment
    class Simulator:
        def __init__(self):
            self.unwrapped = self
            self.commands = []
            self.done = False
            self.closed = False
        def seed(self, value): self.seed_value = value
        def reset(self): self.commands.clear()
        def set_init_state(self, value): self.state = value
        def step(self, action):
            self.commands.append(action.copy())
            return {"image": np.zeros((2, 2, 3))}, 0.0, self.done, {}
        def close(self): self.closed = True
    sim = Simulator()
    env = DirectLiberoEnvironment(factory=lambda: sim, init_states=["a", "b"],
        first_reset_id=10, reset_id=11, seed=7, task_language="task", max_sim_steps=2)
    env.reset()
    assert sim.state == "b" and sim.seed_value == 7
    assert len(sim.commands) == 15
    assert all(list(a) == [0, 0, 0, 0, 0, 0, -1] for a in sim.commands)
    assert env.get_env_attr("state") == ["b"]
    env.set_env_attr("state", "changed")
    assert sim.state == "changed"
    _, _, done, truncated, _ = env.step(np.zeros((1, 7)))
    assert not done[0] and not truncated[0]
    _, _, done, truncated, _ = env.step(np.zeros((1, 7)))
    assert not done[0] and truncated[0]
    sim.done = True
    assert env.step(np.zeros((1, 7)))[2][0]
    assert len(sim.commands) == 18  # no terminal auto-reset
    env.reset(reset_state_ids=np.array([10]))
    assert sim.state == "a" and env._elapsed_steps == 0
    with pytest.raises(ValueError, match="another task"):
        env.reset(reset_state_ids=np.array([12]))
    with pytest.raises(ValueError, match="finite"):
        env.step(np.full((1, 7), np.nan))
    env.close()
    assert sim.closed


@pytest.mark.parametrize("field,value", [("rlinf_root_env", "RLINF_ROOT"),
                                         ("required_rlinf_revision", "a" * 40)])
def test_direct_run_spec_rejects_external_driver_fields(field, value):
    spec = native_spec("libero_pro")
    assert resolve_run_spec(spec)["runtime"]["environment_driver"] == "direct_libero_v1"
    spec["runtime"][field] = value
    with pytest.raises(RunSpecError, match="binds no external driver checkout"):
        resolve_run_spec(spec)


def test_a_libero_runtime_without_a_driver_selects_the_direct_driver():
    runtime = dict(resolve_run_spec(native_spec("libero_pro"))["runtime"])
    runtime.pop("environment_driver")
    validate_libero_configuration(runtime)
    runtime["environment_driver"] = "rlinf_v1"
    with pytest.raises(RunSpecError, match="environment_driver is unsupported"):
        validate_libero_configuration(runtime)
