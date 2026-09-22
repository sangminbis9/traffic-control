import numpy as np
import pytest

from controller.fixed_controller import FixedTimeController
from env.intersection_env import IntersectionEnv
from utils.config import Config


@pytest.mark.integration
@pytest.mark.parametrize("include_approaching", [False, True])
def test_libsumo_matches_traci(tmp_path, include_approaching):
    pytest.importorskip("libsumo")
    config = Config()
    config.observation.include_approaching = include_approaching
    config.simulation.episode_seconds = 90
    config.simulation.demand_seconds = 60
    traces, metrics = [], []
    for backend in ("traci", "libsumo"):
        env = IntersectionEnv(config, output_dir=tmp_path / backend, backend=backend)
        fixed = FixedTimeController(config.signal.fixed_green)
        trace = []
        try:
            env.reset(seed=3, options={"traffic_seed": 2001, "scenario": "balanced"})
            while True:
                observation, reward, _, truncated, info = env.step(fixed.action(env.signal))
                trace.append(np.append(observation, reward))
                if truncated:
                    metrics.append(info["episode_metrics"])
                    break
        finally:
            env.close()
        traces.append(np.array(trace))
    np.testing.assert_array_equal(*traces)
    assert metrics[0] == metrics[1]
    if include_approaching:
        assert traces[0][:, 26:34].max() > 0
