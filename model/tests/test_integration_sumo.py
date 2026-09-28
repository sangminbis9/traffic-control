"""Opt-in checks of the observation contract against a running SUMO process."""

from __future__ import annotations

import os

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from stable_baselines3.common.env_checker import check_env

from model.env.intersection_env import IntersectionEnv


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SUMO_INTEGRATION") != "1",
    reason="Set RUN_SUMO_INTEGRATION=1 to run actual SUMO contract checks.",
)


def test_real_sumo_gymnasium_and_sb3_contract() -> None:
    env = IntersectionEnv(scenario="uniform", seed=71, episode_seconds=60)
    try:
        assert env.action_space.n == 8
        assert env.observation_space.shape == (60,)
        gymnasium_check_env(env, skip_render_check=True)
        check_env(env, warn=True)
    finally:
        env.close()


def test_real_sumo_group_waiting_matches_lane_vehicles() -> None:
    env = IntersectionEnv(scenario="heavy", seed=73, episode_seconds=60)
    saw_waiting = saw_moving_with_wait = False
    phases: set[int] = set()
    try:
        env.reset(seed=73)
        for step in range(60):
            observation, _, _, _, _ = env.step((step // 6) % 8)
            if env.signal_controller.stage == "green":
                phases.add(env.signal_controller.current_phase)
            totals, maxima = [], []
            for approach in ("N", "S", "E", "W"):
                for lanes in ((2,), (0, 1)):
                    ids = [
                        vehicle
                        for lane in lanes
                        for vehicle in env.connection.lane.getLastStepVehicleIDs(f"{approach}_in_{lane}")
                    ]
                    waits = [env.connection.vehicle.getAccumulatedWaitingTime(vehicle) for vehicle in ids]
                    totals.append(sum(waits))
                    maxima.append(max(waits, default=0.0))
                    saw_waiting |= any(wait > 0 for wait in waits)
                    saw_moving_with_wait |= any(
                        wait > 0 and env.connection.vehicle.getSpeed(vehicle) >= 0.1
                        for vehicle, wait in zip(ids, waits, strict=True)
                    )
            snapshot = env._previous_snapshot
            assert snapshot is not None
            np.testing.assert_allclose(snapshot.total_waiting_by_group, totals)
            np.testing.assert_allclose(snapshot.max_waiting_by_group, maxima)
            np.testing.assert_allclose(observation[8:16], np.clip(np.asarray(totals) / 6000, 0, 1))
            np.testing.assert_allclose(observation[16:24], np.clip(np.asarray(maxima) / 120, 0, 1))
            assert snapshot.total_waiting_time == pytest.approx(sum(totals))
            assert snapshot.max_waiting_time == max(maxima)
            assert env.observation_space.contains(observation)
        assert phases == set(range(8))
        assert saw_waiting and saw_moving_with_wait
    finally:
        env.close()
