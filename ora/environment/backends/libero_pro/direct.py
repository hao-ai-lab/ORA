"""Single LIBERO simulator owned by the isolated Harness worker, without RLinf."""
from __future__ import annotations

from typing import Any, Callable

DRIVER = "direct_libero_v1"


class DirectLiberoEnvironment:
    """Adapt stock LIBERO reset/step to the existing trusted Session port.

    There is no vector scheduler or auto-reset. The outer Runtime already owns
    process isolation, rounds and action admission. The tiny attribute port is
    private to the trusted Session/controller wrapper, never a policy tool.
    """

    def __init__(self, *, factory: Callable[[], Any], init_states: Any,
                 first_reset_id: int, reset_id: int, seed: int,
                 task_language: str, max_sim_steps: int) -> None:
        self._factory = factory
        self._delegate = None
        self._init_states = init_states
        self._first_reset_id = first_reset_id
        self.specific_reset_id = reset_id
        self.seed = seed
        self.task_descriptions = [task_language]
        self.max_sim_steps = max_sim_steps
        self._elapsed_steps = 0
        self.current_raw_obs = None
        self.env = self

    def reset(self, *, reset_state_ids=None):
        import numpy as np
        if reset_state_ids is not None:
            ids = np.asarray(reset_state_ids)
            if ids.shape != (1,):
                raise ValueError("direct LIBERO reset requires exactly one state id")
            self.specific_reset_id = int(ids[0])
        trial = self.specific_reset_id - self._first_reset_id
        if not 0 <= trial < len(self._init_states):
            raise ValueError("direct LIBERO reset state belongs to another task")
        if self._delegate is None:
            self._delegate = self._factory()
        self._delegate.seed(self.seed)
        self._delegate.reset()
        self._delegate.set_init_state(self._init_states[trial])
        # Retain the source treatment's 15 settling steps with gripper open.
        # These are reset preparation, not policy actions or official outcomes.
        command = np.zeros(7, dtype=np.float32)
        command[-1] = -1
        for _ in range(15):
            raw, _reward, _done, _info = self._delegate.step(command)
        self.current_raw_obs = [dict(raw)]
        self._elapsed_steps = 0
        return self.current_raw_obs, {}

    def step(self, actions):
        import numpy as np
        command = np.asarray(actions)
        if command.shape != (1, 7) or not np.isfinite(command).all():
            raise ValueError("direct LIBERO action must be a finite 1x7 command")
        if self._delegate is None:
            raise RuntimeError("direct LIBERO environment must be reset before step")
        raw, reward, terminated, info = self._delegate.step(command[0])
        self.current_raw_obs = [dict(raw)]
        self._elapsed_steps += 1
        # Stock LIBERO's step returns its official success boundary. Keep the
        # independently owned time limit separate from that Boolean verdict.
        return (self.current_raw_obs, np.asarray([reward]), np.asarray([bool(terminated)]),
                np.asarray([self._elapsed_steps >= self.max_sim_steps]), dict(info))

    def get_env_attr(self, name: str):
        if self._delegate is None:
            raise RuntimeError("direct LIBERO worker has no environment")
        return [getattr(self._delegate.unwrapped, name)]

    def set_env_attr(self, name: str, value: Any):
        if self._delegate is None:
            raise RuntimeError("direct LIBERO worker has no environment")
        setattr(self._delegate.unwrapped, name, value)

    def close(self):
        if self._delegate is not None:
            delegate, self._delegate = self._delegate, None
            delegate.close()
