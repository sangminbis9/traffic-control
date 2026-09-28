"""Traffic state abstraction shared by SUMO and future camera providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

from model.controller.signal_controller import CONTROLLER_FEATURE_COUNT


LANE_GROUPS: tuple[tuple[str, str], ...] = (
    ("N", "left"),
    ("N", "straight"),
    ("S", "left"),
    ("S", "straight"),
    ("E", "left"),
    ("E", "straight"),
    ("W", "left"),
    ("W", "straight"),
)
INCOMING_EDGES = {approach: f"{approach}_in" for approach in ("N", "S", "E", "W")}
GROUP_COUNT = len(LANE_GROUPS)
# queues, total waiting, max waiting, controller state, approaching vehicles.
OBSERVATION_SIZE = GROUP_COUNT * 4 + CONTROLLER_FEATURE_COUNT


@dataclass(frozen=True)
class TrafficSnapshot:
    """Raw traffic measurements before observation normalization."""

    queue_by_group: tuple[float, ...]
    total_queue: float
    total_waiting_time: float
    max_waiting_time: float
    vehicle_count: int
    arrived: int
    approaching_by_group: tuple[float, ...] = (0.0,) * GROUP_COUNT
    vehicle_waiting: Mapping[str, float] = field(default_factory=dict)
    departed_ids: tuple[str, ...] = ()
    arrived_ids: tuple[str, ...] = ()
    pending: int = 0
    collisions: int = 0
    teleports: int = 0
    # Global waiting above remains available for reward and episode metrics.
    # Missing group measurements default to zero; global totals cannot be
    # distributed across movements without knowing the vehicles' lanes.
    total_waiting_by_group: tuple[float, ...] = (0.0,) * GROUP_COUNT
    max_waiting_by_group: tuple[float, ...] = (0.0,) * GROUP_COUNT


class TrafficStateProvider(Protocol):
    def snapshot(self) -> TrafficSnapshot:
        ...

    def observation(self, signal_controller: Any, snapshot: TrafficSnapshot | None = None) -> np.ndarray:
        ...


def _observation(
    snapshot: TrafficSnapshot,
    signal_features: np.ndarray,
    *,
    queue_scale: float,
    waiting_scale: float,
    max_waiting_scale: float,
    approaching_scale: float,
) -> np.ndarray:
    traffic = np.concatenate(
        [
            np.asarray(snapshot.queue_by_group, dtype=np.float32) / max(queue_scale, 1e-9),
            np.asarray(snapshot.total_waiting_by_group, dtype=np.float32)
            / max(waiting_scale, 1e-9),
            np.asarray(snapshot.max_waiting_by_group, dtype=np.float32)
            / max(max_waiting_scale, 1e-9),
        ]
    )
    approaching = (
        np.asarray(snapshot.approaching_by_group, dtype=np.float32)
        / max(approaching_scale, 1e-9)
    )
    return np.concatenate(
        [
            np.clip(traffic, 0.0, 1.0),
            np.asarray(signal_features, dtype=np.float32),
            np.clip(approaching, 0.0, 1.0),
        ]
    ).astype(np.float32)


class SUMOTrafficStateProvider:
    """Read movement-level queue, waiting, and approaching features from TraCI."""

    def __init__(
        self,
        connection: Any,
        queue_scale: float = 10.0,
        waiting_scale: float = 6_000.0,
        max_waiting_scale: float = 120.0,
        detection_distance: float = 50.0,
        approaching_scale: float = 10.0,
    ) -> None:
        self.connection = connection
        self.queue_scale = queue_scale
        self.waiting_scale = waiting_scale
        self.max_waiting_scale = max_waiting_scale
        self.detection_distance = detection_distance
        self.approaching_scale = approaching_scale

    def snapshot(self) -> TrafficSnapshot:
        queue_values: list[float] = []
        approaching_values: list[float] = []
        moving_by_lane: dict[str, int] = {}
        waiting_by_lane: dict[str, list[float]] = {}
        lane_lengths: dict[str, float] = {}
        for approach in ("N", "S", "E", "W"):
            for lane in range(3):
                lane_id = f"{approach}_in_{lane}"
                moving_by_lane[lane_id] = 0
                waiting_by_lane[lane_id] = []
                lane_lengths[lane_id] = float(self.connection.lane.getLength(lane_id))

        vehicle_ids = tuple(self.connection.vehicle.getIDList())
        vehicle_waiting: dict[str, float] = {}
        incoming_waiting: list[float] = []
        for vehicle_id in vehicle_ids:
            lane_id = self.connection.vehicle.getLaneID(vehicle_id)
            waiting = float(self.connection.vehicle.getAccumulatedWaitingTime(vehicle_id))
            vehicle_waiting[vehicle_id] = waiting
            if lane_id not in lane_lengths:
                continue
            incoming_waiting.append(waiting)
            # Include every vehicle on an incoming lane, even if moving or
            # outside the approaching-feature detection distance.
            waiting_by_lane[lane_id].append(waiting)
            speed = float(self.connection.vehicle.getSpeed(vehicle_id))
            lane_position = float(self.connection.vehicle.getLanePosition(vehicle_id))
            if speed >= 0.1 and lane_lengths[lane_id] - lane_position <= self.detection_distance:
                moving_by_lane[lane_id] += 1

        total_waiting_values: list[float] = []
        max_waiting_values: list[float] = []
        for approach, group in LANE_GROUPS:
            edge_id = INCOMING_EDGES[approach]
            lane_indices = (2,) if group == "left" else (0, 1)
            group_waiting = [
                waiting
                for lane in lane_indices
                for waiting in waiting_by_lane[f"{edge_id}_{lane}"]
            ]
            total_waiting_values.append(float(sum(group_waiting)))
            max_waiting_values.append(max(group_waiting, default=0.0))
            queue_values.append(
                float(
                    sum(
                        self.connection.lane.getLastStepHaltingNumber(f"{edge_id}_{lane}")
                        for lane in lane_indices
                    )
                )
            )
            approaching_values.append(
                float(sum(moving_by_lane[f"{edge_id}_{lane}"] for lane in lane_indices))
            )

        departed = tuple(self.connection.simulation.getDepartedIDList())
        arrived_ids = tuple(self.connection.simulation.getArrivedIDList())
        pending_getter = getattr(self.connection.simulation, "getPendingVehicles", None)
        pending = len(pending_getter()) if pending_getter is not None else 0
        collisions_getter = getattr(
            self.connection.simulation, "getCollidingVehiclesNumber", lambda: 0
        )
        teleports_getter = getattr(
            self.connection.simulation, "getStartingTeleportNumber", lambda: 0
        )
        return TrafficSnapshot(
            queue_by_group=tuple(queue_values),
            total_queue=float(sum(queue_values)),
            total_waiting_time=float(sum(incoming_waiting)),
            max_waiting_time=max(incoming_waiting, default=0.0),
            vehicle_count=len(vehicle_ids),
            arrived=len(arrived_ids),
            approaching_by_group=tuple(approaching_values),
            total_waiting_by_group=tuple(total_waiting_values),
            max_waiting_by_group=tuple(max_waiting_values),
            vehicle_waiting=vehicle_waiting,
            departed_ids=departed,
            arrived_ids=arrived_ids,
            pending=pending,
            collisions=int(collisions_getter()),
            teleports=int(teleports_getter()),
        )

    def observation(
        self, signal_controller: Any, snapshot: TrafficSnapshot | None = None
    ) -> np.ndarray:
        return _observation(
            snapshot or self.snapshot(),
            signal_controller.features(),
            queue_scale=self.queue_scale,
            waiting_scale=self.waiting_scale,
            max_waiting_scale=self.max_waiting_scale,
            approaching_scale=self.approaching_scale,
        )


class SnapshotStateProvider:
    """Adapter useful for unit tests and future camera/vision implementations."""

    def __init__(
        self,
        snapshot: TrafficSnapshot,
        queue_scale: float = 10.0,
        waiting_scale: float = 6_000.0,
        max_waiting_scale: float = 120.0,
        approaching_scale: float = 10.0,
    ) -> None:
        self._snapshot = snapshot
        self.queue_scale = queue_scale
        self.waiting_scale = waiting_scale
        self.max_waiting_scale = max_waiting_scale
        self.approaching_scale = approaching_scale

    def snapshot(self) -> TrafficSnapshot:
        return self._snapshot

    def observation(
        self, signal_controller: Any, snapshot: TrafficSnapshot | None = None
    ) -> np.ndarray:
        return _observation(
            snapshot or self._snapshot,
            signal_controller.features(),
            queue_scale=self.queue_scale,
            waiting_scale=self.waiting_scale,
            max_waiting_scale=self.max_waiting_scale,
            approaching_scale=self.approaching_scale,
        )
