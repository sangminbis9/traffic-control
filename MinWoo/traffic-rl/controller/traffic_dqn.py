"""DQN with temporally correlated exploration, no inference-time rules."""
from __future__ import annotations

import numpy as np
from stable_baselines3 import DQN


class TrafficDQN(DQN):
    def __init__(self, *args, exploration_hold_min: int = 1, exploration_hold_max: int = 1, **kwargs):
        self.exploration_hold_min = exploration_hold_min
        self.exploration_hold_max = exploration_hold_max
        self._held_action = None
        self._hold_remaining = 0
        super().__init__(*args, **kwargs)

    def _sample_action(self, learning_starts: int, action_noise=None, n_envs: int = 1):
        if self.exploration_hold_max == 1:
            return super()._sample_action(learning_starts, action_noise, n_envs)
        if n_envs != 1:
            raise ValueError("Temporally correlated exploration currently requires one env per process")
        if self._hold_remaining > 0:
            self._hold_remaining -= 1
            action = self._held_action.copy()
        elif self.num_timesteps < learning_starts or np.random.random() < self.exploration_rate:
            action = np.array([self.action_space.sample()], dtype=np.int64)
            self._held_action = action.copy()
            self._hold_remaining = int(np.random.randint(self.exploration_hold_min, self.exploration_hold_max + 1)) - 1
        else:
            action, _ = self.predict(self._last_obs, deterministic=True)
        return action, action.copy()
