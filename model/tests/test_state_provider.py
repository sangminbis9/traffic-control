from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from model.controller.signal_controller import CONTROLLER_FEATURE_COUNT
from model.env.state_provider import (
    GROUP_COUNT,
    LANE_GROUPS,
    OBSERVATION_SIZE,
    SUMOTrafficStateProvider,
    SnapshotStateProvider,
    TrafficSnapshot,
)


class FakeVehicles:
    def __init__(self, records: dict[str, tuple[str, float, float, float]]) -> None:
        # lane id, accumulated waiting, speed, lane position
        self.records = records

    def getIDList(self) -> tuple[str, ...]:
        return tuple(self.records)

    def getLaneID(self, vehicle_id: str) -> str:
        return self.records[vehicle_id][0]

    def getAccumulatedWaitingTime(self, vehicle_id: str) -> float:
        return self.records[vehicle_id][1]

    def getSpeed(self, vehicle_id: str) -> float:
        return self.records[vehicle_id][2]

    def getLanePosition(self, vehicle_id: str) -> float:
        return self.records[vehicle_id][3]


def connection_for(records: dict[str, tuple[str, float, float, float]]) -> SimpleNamespace:
    return SimpleNamespace(
        vehicle=FakeVehicles(records),
        lane=SimpleNamespace(
            getLength=lambda _lane: 100.0,
            getLastStepHaltingNumber=lambda lane: sum(
                record[0] == lane and record[2] < 0.1 for record in records.values()
            ),
        ),
        simulation=SimpleNamespace(
            getDepartedIDList=lambda: ("departed",),
            getArrivedIDList=lambda: ("arrived",),
            getPendingVehicles=lambda: ("pending1", "pending2"),
            getCollidingVehiclesNumber=lambda: 1,
            getStartingTeleportNumber=lambda: 2,
        ),
    )


class StateProviderTests(unittest.TestCase):
    def test_waiting_uses_all_current_incoming_vehicles_by_movement(self) -> None:
        records = {
            "n_left_stopped": ("N_in_2", 12.0, 0.0, 95.0),
            "n_left_moving_far": ("N_in_2", 9.0, 7.0, 10.0),
            "n_lane0_moving": ("N_in_0", 8.0, 3.0, 90.0),
            "n_lane1_stopped": ("N_in_1", 5.0, 0.0, 95.0),
            "s_left_moving": ("S_in_2", 3.0, 5.0, 70.0),
            "s_lane0_stopped": ("S_in_0", 4.0, 0.0, 95.0),
            "s_lane1_slow": ("S_in_1", 6.0, 0.09, 99.0),
            "e_left_stopped": ("E_in_2", 15.0, 0.0, 95.0),
            "e_straight_moving": ("E_in_1", 18.0, 4.0, 98.0),
            "w_left_at_boundary": ("W_in_2", 7.0, 4.0, 50.0),
            "w_lane0_stopped_far": ("W_in_0", 10.0, 0.0, 10.0),
            "w_lane1_moving_far": ("W_in_1", 20.0, 5.0, 40.0),
            "outgoing": ("S_out_0", 4000.0, 0.0, 95.0),
            "internal": (":J_0_0", 7000.0, 0.0, 1.0),
        }
        snapshot = SUMOTrafficStateProvider(connection_for(records)).snapshot()

        self.assertEqual(
            LANE_GROUPS,
            (("N", "left"), ("N", "straight"), ("S", "left"), ("S", "straight"),
             ("E", "left"), ("E", "straight"), ("W", "left"), ("W", "straight")),
        )
        self.assertEqual(snapshot.total_waiting_by_group, (21, 13, 3, 10, 15, 18, 7, 30))
        self.assertEqual(snapshot.max_waiting_by_group, (12, 8, 3, 6, 15, 18, 7, 20))
        self.assertEqual(snapshot.queue_by_group, (1, 1, 0, 2, 1, 0, 0, 1))
        self.assertEqual(snapshot.approaching_by_group, (0, 1, 1, 0, 0, 1, 1, 0))
        self.assertEqual(snapshot.total_queue, 6.0)
        self.assertEqual(snapshot.total_waiting_time, 117.0)
        self.assertEqual(snapshot.max_waiting_time, 20.0)
        self.assertEqual(snapshot.vehicle_waiting, {key: value[1] for key, value in records.items()})
        self.assertEqual(snapshot.vehicle_count, 14)
        self.assertEqual(snapshot.arrived, 1)
        self.assertEqual(snapshot.departed_ids, ("departed",))
        self.assertEqual(snapshot.arrived_ids, ("arrived",))
        self.assertEqual((snapshot.pending, snapshot.collisions, snapshot.teleports), (2, 1, 2))

    def test_empty_groups_have_zero_waiting(self) -> None:
        records = {"moving": ("W_in_2", 17.0, 5.0, 20.0)}
        provider = SUMOTrafficStateProvider(connection_for(records))
        snapshot = provider.snapshot()
        self.assertEqual(snapshot.total_waiting_by_group, (0, 0, 0, 0, 0, 0, 17, 0))
        self.assertEqual(snapshot.max_waiting_by_group, (0, 0, 0, 0, 0, 0, 17, 0))
        self.assertEqual(snapshot.queue_by_group, (0.0,) * GROUP_COUNT)
        self.assertEqual(snapshot.approaching_by_group, (0.0,) * GROUP_COUNT)

        records.clear()
        empty = provider.snapshot()
        self.assertEqual(empty.total_waiting_by_group, (0.0,) * GROUP_COUNT)
        self.assertEqual(empty.max_waiting_by_group, (0.0,) * GROUP_COUNT)
        self.assertEqual((empty.total_waiting_time, empty.max_waiting_time), (0.0, 0.0))

    def test_observation_order_normalization_and_snapshot_adapter(self) -> None:
        snapshot = TrafficSnapshot(
            queue_by_group=(0, 1, 2, 5, 10, 15, 20, 25),
            total_queue=78,
            total_waiting_time=99_999,
            max_waiting_time=99_999,
            vehicle_count=0,
            arrived=0,
            approaching_by_group=(25, 20, 15, 10, 5, 2, 1, 0),
            total_waiting_by_group=(0, 10, 20, 50, 100, 150, 200, 250),
            max_waiting_by_group=(0, 2, 4, 10, 20, 30, 40, 50),
        )
        signal_features = np.linspace(0.0, 1.0, CONTROLLER_FEATURE_COUNT, dtype=np.float32)
        controller = SimpleNamespace(features=lambda: signal_features)
        provider = SnapshotStateProvider(snapshot, waiting_scale=100.0, max_waiting_scale=20.0)
        sumo_provider = SUMOTrafficStateProvider(
            connection_for({}), waiting_scale=100.0, max_waiting_scale=20.0
        )
        observation = provider.observation(controller)
        expected_groups = np.asarray((0, 0.1, 0.2, 0.5, 1, 1, 1, 1), dtype=np.float32)
        expected = np.concatenate(
            [expected_groups, expected_groups, expected_groups, signal_features, expected_groups[::-1]]
        )
        self.assertEqual(OBSERVATION_SIZE, 60)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_allclose(observation, expected)
        np.testing.assert_array_equal(sumo_provider.observation(controller, snapshot), observation)
        np.testing.assert_array_equal(
            provider.observation(controller, replace(snapshot, total_waiting_time=0, max_waiting_time=0)),
            observation,
        )

    def test_default_waiting_scales_apply_independently_to_all_groups(self) -> None:
        snapshot = TrafficSnapshot(
            (0.0,) * GROUP_COUNT, 0, 12_000, 240, 0, 0,
            total_waiting_by_group=(0, 600, 1200, 3000, 6000, 6600, 12_000, 60),
            max_waiting_by_group=(0, 12, 24, 60, 120, 132, 240, 1.2),
        )
        controller = SimpleNamespace(features=lambda: np.zeros(CONTROLLER_FEATURE_COUNT))
        observation = SnapshotStateProvider(snapshot).observation(controller)
        expected = (0, 0.1, 0.2, 0.5, 1, 1, 1, 0.01)
        np.testing.assert_allclose(observation[8:16], expected)
        np.testing.assert_allclose(observation[16:24], expected)

    def test_legacy_snapshot_constructor_does_not_invent_group_waiting(self) -> None:
        snapshot = TrafficSnapshot((0.0,) * GROUP_COUNT, 0, 1200, 90, 2, 0)
        controller = SimpleNamespace(features=lambda: np.zeros(CONTROLLER_FEATURE_COUNT))
        observation = SnapshotStateProvider(snapshot).observation(controller)
        self.assertEqual(snapshot.total_waiting_by_group, (0.0,) * GROUP_COUNT)
        self.assertEqual(snapshot.max_waiting_by_group, (0.0,) * GROUP_COUNT)
        self.assertEqual(observation.shape, (OBSERVATION_SIZE,))
        np.testing.assert_array_equal(observation, np.zeros(OBSERVATION_SIZE, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
