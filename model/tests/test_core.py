from __future__ import annotations

import hashlib
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from model.controller.signal_controller import GREEN_LINKS, SignalController
from model.env.reward import calculate_reward
from model.env.state_provider import SnapshotStateProvider, TrafficSnapshot
from model.sumo.build_network import validate_network
from model.traffic.route_generator import TrafficDemand, _route_for, generate_route_file
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
    def test_sumo_configuration_uses_standard_extension(self) -> None:
        config_path = ProjectConfig().sumo_config_file
        self.assertEqual(config_path.name, "simulation.sumocfg")
        self.assertTrue(config_path.exists())

    def test_legacy_model_is_preserved_and_network_is_valid(self) -> None:
        config = ProjectConfig()
        model_path = config.results_dir / "dqn_intersection.zip"
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

        self.assertEqual(
            model_hash,
            "920fe8d6141ad7f7621ec8e28cf5957fe7e92237ed2dc9d6f35b4f3efa13b28d",
        )
        validate_network(config.network_file)

    def test_route_movements_use_correct_lanes_and_horizontal_turns(self) -> None:
        expected_destinations = {
            "N": {"left": "E_out", "straight": "S_out", "right": "W_out"},
            "S": {"left": "W_out", "straight": "N_out", "right": "E_out"},
            "E": {"left": "S_out", "straight": "W_out", "right": "N_out"},
            "W": {"left": "N_out", "straight": "E_out", "right": "S_out"},
        }
        expected_lanes = {"left": 2, "straight": 1, "right": 0}

        for approach, destinations in expected_destinations.items():
            for movement, destination in destinations.items():
                incoming, actual_destination, lane = _route_for(approach, movement)
                self.assertEqual(incoming, f"{approach}_in")
                self.assertEqual(actual_destination, destination)
                self.assertEqual(lane, expected_lanes[movement])

    def test_network_lane_connections_follow_driver_left_to_right_layout(self) -> None:
        root = ET.parse(ProjectConfig().sumo_dir / "connections.con.xml").getroot()
        connections = root.findall("connection")
        destinations = {
            "N": {"left": "E_out", "straight": "S_out", "right": "W_out"},
            "S": {"left": "W_out", "straight": "N_out", "right": "E_out"},
            "E": {"left": "S_out", "straight": "W_out", "right": "N_out"},
            "W": {"left": "N_out", "straight": "E_out", "right": "S_out"},
        }

        for approach, movement_targets in destinations.items():
            incoming = f"{approach}_in"
            by_target = {
                target: [
                    connection
                    for connection in connections
                    if connection.attrib["from"] == incoming and connection.attrib["to"] == target
                ]
                for target in movement_targets.values()
            }
            self.assertEqual(
                {connection.attrib["fromLane"] for connection in by_target[movement_targets["left"]]},
                {"2"},
            )
            self.assertEqual(
                {connection.attrib["fromLane"] for connection in by_target[movement_targets["straight"]]},
                {"0", "1"},
            )
            self.assertEqual(
                {connection.attrib["fromLane"] for connection in by_target[movement_targets["right"]]},
                {"0"},
            )

    def test_generated_network_signal_links_match_eight_logical_phases(self) -> None:
        root = ET.parse(ProjectConfig().network_file).getroot()
        controlled = {
            int(connection.attrib["linkIndex"]): (
                connection.attrib["from"][0],
                connection.attrib["dir"],
            )
            for connection in root.findall("connection")
            if connection.attrib.get("tl") == "J"
        }
        expected = {
            0: {(approach, movement) for approach in "NS" for movement in ("s", "r")},
            1: {(approach, "l") for approach in "NS"},
            2: {("N", movement) for movement in ("s", "r", "l")},
            3: {("S", movement) for movement in ("s", "r", "l")},
            4: {(approach, movement) for approach in "EW" for movement in ("s", "r")},
            5: {(approach, "l") for approach in "EW"},
            6: {("E", movement) for movement in ("s", "r", "l")},
            7: {("W", movement) for movement in ("s", "r", "l")},
        }

        for phase, link_indices in GREEN_LINKS.items():
            self.assertEqual({controlled[index] for index in link_indices}, expected[phase])

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
            "J",
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
        connection = FakeConnection()
        controller = SignalController(connection, "J", SignalConfig())
        controller.reset(2)
        controller.advance(5.0)
        observation = provider.observation(controller)
        self.assertEqual(observation.shape, (60,))
        self.assertTrue(np.all(observation >= 0.0))
        self.assertTrue(np.all(observation <= 1.0))
        reward_after = calculate_reward(before, after, False, RewardConfig())
        reward_worse = calculate_reward(after, before, False, RewardConfig())
        self.assertLess(reward_after, 0.0)
        self.assertGreater(reward_after, reward_worse)

    def test_reward_strongly_penalizes_starvation_and_switching(self) -> None:
        snapshot = TrafficSnapshot((1, 0, 0, 0, 0, 0, 0, 0), 1, 0, 90, 0, 0)
        config = RewardConfig()
        reward_held = calculate_reward(snapshot, snapshot, False, config)
        reward_switched = calculate_reward(snapshot, snapshot, True, config)

        self.assertAlmostEqual(config.max_waiting_weight, 0.5)
        self.assertAlmostEqual(config.max_waiting_scale, 120.0)
        self.assertAlmostEqual(reward_held, -0.3875)
        self.assertAlmostEqual(reward_switched - reward_held, -0.5)


if __name__ == "__main__":
    unittest.main()
