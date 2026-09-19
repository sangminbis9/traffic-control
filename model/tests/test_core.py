from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from model.controller.signal_controller import SignalController
from model.env.reward import calculate_reward
from model.env.state_provider import SnapshotStateProvider, TrafficSnapshot
from model.traffic.route_generator import TrafficDemand, generate_route_file
from model.utils.config import ProjectConfig, RewardConfig, SignalConfig


class FakeTrafficLight:
    def __init__(self) -> None:
        self.states: list[str] = []

    def setRedYellowGreenState(self, _tl_id: str, state: str) -> None:
        self.states.append(state)


class FakeConnection:
    def __init__(self) -> None:
        self.trafficlight = FakeTrafficLight()


class CoreTests(unittest.TestCase):
    def test_route_generation_is_seed_reproducible(self) -> None:
        directory = ProjectConfig().sumo_dir / "generated"
        first = generate_route_file(
            directory / "unit_first.rou.xml",
            TrafficDemand(duration=20, scenario="random", seed=11),
        )
        second = generate_route_file(
            directory / "unit_second.rou.xml",
            TrafficDemand(duration=20, scenario="random", seed=11),
        )
        self.assertEqual(first.read_text(), second.read_text())
        ET.parse(first)

    def test_signal_controller_enforces_yellow_and_all_red(self) -> None:
        connection = FakeConnection()
        controller = SignalController(
            connection,
            "center",
            SignalConfig(min_green=3.0, max_green=15.0, yellow=1.0, all_red=1.0),
        )
        controller.reset()
        self.assertTrue(controller.apply_action(2).ignored)
        controller.advance(3.0)
        self.assertTrue(controller.apply_action(2).switched)
        controller.advance(1.0)
        self.assertEqual(connection.trafficlight.states[-1], "r" * 16)
        controller.advance(1.0)
        self.assertEqual(controller.current_phase, 2)
        self.assertEqual(controller.phase_elapsed, 0.0)

    def test_observation_is_bounded_and_reward_improves_when_queue_falls(self) -> None:
        before = TrafficSnapshot((10, 5, 0, 0, 0, 0, 0, 0), 15, 30, 12, 4, 0)
        after = TrafficSnapshot((5, 3, 0, 0, 0, 0, 0, 0), 8, 20, 8, 3, 1)
        provider = SnapshotStateProvider(after)
        observation = provider.observation(phase=2, phase_elapsed=5, max_green=15)
        self.assertEqual(observation.shape, (12,))
        self.assertTrue(np.all(observation >= 0.0))
        self.assertTrue(np.all(observation <= 1.0))
        reward_after = calculate_reward(before, after, False, RewardConfig())
        reward_worse = calculate_reward(after, before, False, RewardConfig())
        self.assertLess(reward_after, 0.0)
        self.assertGreater(reward_after, reward_worse)


if __name__ == "__main__":
    unittest.main()
