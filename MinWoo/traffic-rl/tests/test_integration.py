import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from build_network import validate_network
from controller.fixed_controller import FixedTimeController
from env.intersection_env import IntersectionEnv
from utils.config import Config, ROOT

pytestmark = pytest.mark.integration


def run_episode(env, seed=2001):
    observation, info = env.reset(seed=7, options={"traffic_seed": seed, "scenario": "balanced"})
    fixed = FixedTimeController(env.config.signal.fixed_green)
    trace = []
    while True:
        observation, reward, terminated, truncated, info = env.step(fixed.action(env.signal))
        assert env.observation_space.contains(observation)
        trace.append((observation.copy(), reward))
        if terminated or truncated:
            break
    assert not terminated and truncated
    return trace, info["episode_metrics"]


def test_network_mapping():
    validate_network(ROOT / "sumo/intersection.net.xml")


def test_check_env_and_paired_reproducibility(tmp_path):
    config = Config()
    config.simulation.episode_seconds = 90
    config.simulation.demand_seconds = 60
    env = IntersectionEnv(config, output_dir=tmp_path)
    try:
        check_env(env, warn=True)
        trace1, row1 = run_episode(env)
        trace2, row2 = run_episode(env)
        for (obs1, reward1), (obs2, reward2) in zip(trace1, trace2):
            np.testing.assert_array_equal(obs1, obs2)
            assert reward1 == reward2
        for key in row1:
            if key != "episode":
                assert row1[key] == row2[key]
        assert row1["collisions"] == row1["teleports"] == 0
        assert row1["throughput"] > 0
        assert row1["departed"] == row1["throughput"] + row1["unfinished"]
        assert row1["scheduled"] == row1["departed"] + row1["not_inserted"]
        process = env.backend.process
    finally:
        env.close()
        env.close()
    assert process.poll() is not None
    with pytest.raises(RuntimeError):
        env.step(0)


def test_independent_connections_and_error_cleanup(tmp_path):
    a = IntersectionEnv(output_dir=tmp_path / "a")
    b = IntersectionEnv(output_dir=tmp_path / "b")
    try:
        a.reset(seed=1)
        b.reset(seed=2)
        a.step(0)
        b.step(2)
        assert a.backend.connection is not b.backend.connection
        process = a.backend.process
        original = a.backend.step

        def fail():
            raise RuntimeError("injected failure")

        a.backend.step = fail
        with pytest.raises(RuntimeError, match="injected failure"):
            a.step(0)
        assert process.poll() is not None
        a.backend.step = original
    finally:
        a.close()
        b.close()
