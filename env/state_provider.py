"""Traffic state abstraction shared by SUMO and future camera providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


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


@dataclass(frozen=True)
class TrafficSnapshot:
    """Raw traffic measurements before observation normalization."""

    queue_by_group: tuple[float, ...]
    total_queue: float
    total_waiting_time: float
    max_waiting_time: float
    vehicle_count: int
    arrived: int


class TrafficStateProvider(Protocol):
    """Minimal interface required by the RL environment."""

    def snapshot(self) -> TrafficSnapshot:
        ...

    def observation(self, phase: int, phase_elapsed: float, max_green: float) -> np.ndarray:
        ...


class SUMOTrafficStateProvider:
    """Read lane queues and vehicle waiting data from a TraCI connection."""

    def __init__(self, connection: Any, queue_scale: float = 40.0, waiting_scale: float = 120.0) -> None:
        self.connection = connection
        self.queue_scale = queue_scale
        self.waiting_scale = waiting_scale

    def snapshot(self) -> TrafficSnapshot:
        queue_values: list[float] = []
        for approach, group in LANE_GROUPS:
            edge_id = INCOMING_EDGES[approach]
            lane_indices = (0,) if group == "left" else (1, 2)
            queue_values.append(
                float(
                    sum(
                        self.connection.lane.getLastStepHaltingNumber(f"{edge_id}_{lane}")
                        for lane in lane_indices
                    )
                )
            )

        vehicle_ids = self.connection.vehicle.getIDList()
        waiting_times = [
            float(self.connection.vehicle.getAccumulatedWaitingTime(vehicle_id))
            for vehicle_id in vehicle_ids
        ]
        return TrafficSnapshot(
            queue_by_group=tuple(queue_values),
            total_queue=float(sum(queue_values)),
            total_waiting_time=float(sum(waiting_times)),
            max_waiting_time=max(waiting_times, default=0.0),
            vehicle_count=len(vehicle_ids),
            arrived=int(self.connection.simulation.getArrivedNumber()),
        )

    def observation(self, phase: int, phase_elapsed: float, max_green: float) -> np.ndarray:
        snapshot = self.snapshot()
        queue = np.asarray(snapshot.queue_by_group, dtype=np.float32) / max(self.queue_scale, 1.0)
        total_wait = min(snapshot.total_waiting_time / max(self.waiting_scale, 1.0), 1.0)
        max_wait = min(snapshot.max_waiting_time / max(self.waiting_scale, 1.0), 1.0)
        phase_value = float(np.clip(phase / 3.0, 0.0, 1.0))
        elapsed_value = float(np.clip(phase_elapsed / max(max_green, 1.0), 0.0, 1.0))
        return np.concatenate(
            [queue, np.asarray([total_wait, max_wait, phase_value, elapsed_value], dtype=np.float32)]
        ).astype(np.float32)


class SnapshotStateProvider:
    """Adapter useful for unit tests and future camera/vision implementations."""

    def __init__(self, snapshot: TrafficSnapshot, queue_scale: float = 40.0, waiting_scale: float = 120.0) -> None:
        self._snapshot = snapshot
        self.queue_scale = queue_scale
        self.waiting_scale = waiting_scale

    def snapshot(self) -> TrafficSnapshot:
        return self._snapshot

    def observation(self, phase: int, phase_elapsed: float, max_green: float) -> np.ndarray:
        queue = np.asarray(self._snapshot.queue_by_group, dtype=np.float32) / max(self.queue_scale, 1.0)
        return np.concatenate(
            [
                queue,
                np.asarray(
                    [
                        min(self._snapshot.total_waiting_time / max(self.waiting_scale, 1.0), 1.0),
                        min(self._snapshot.max_waiting_time / max(self.waiting_scale, 1.0), 1.0),
                        np.clip(phase / 3.0, 0.0, 1.0),
                        np.clip(phase_elapsed / max(max_green, 1.0), 0.0, 1.0),
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)

