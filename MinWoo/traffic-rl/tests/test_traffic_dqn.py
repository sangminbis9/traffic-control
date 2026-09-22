import gymnasium as gym
import numpy as np

from controller.traffic_dqn import TrafficDQN


def test_correlated_exploration_and_saved_greedy_prediction(tmp_path):
    env = gym.make("CartPole-v1")
    model = TrafficDQN("MlpPolicy", env, seed=11, exploration_hold_min=5, exploration_hold_max=5,
                       buffer_size=100, device="cpu")
    first, stored = model._sample_action(100)
    np.testing.assert_array_equal(first, stored)
    assert model._hold_remaining == 4
    for _ in range(4):
        action, _ = model._sample_action(100)
        np.testing.assert_array_equal(action, first)
    assert model._hold_remaining == 0
    obs, _ = env.reset(seed=11)
    before, _ = model.predict(obs, deterministic=True)
    model.save(tmp_path / "model")
    from stable_baselines3 import DQN
    after, _ = DQN.load(tmp_path / "model").predict(obs, deterministic=True)
    np.testing.assert_array_equal(before, after)
    env.close()
