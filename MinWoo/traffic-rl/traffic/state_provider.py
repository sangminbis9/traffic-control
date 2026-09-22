"""Physical traffic features reusable by a future camera implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import numpy as np

from simulation.backend import Snapshot
from utils.config import DIRECTIONS, NormalizationConfig


@dataclass(frozen=True)
class TrafficState:
    queues: tuple[float, ...]
    total_waiting: float
    max_waiting: float
    approaching: tuple[float, ...] = (0.0,) * 8

    def normalized(self, scales: NormalizationConfig) -> np.ndarray:
        values = [q / scales.queue_per_group for q in self.queues]
        values += [self.total_waiting / scales.total_waiting, self.max_waiting / scales.max_waiting]
        return np.clip(values, 0, 1).astype(np.float32)


class TrafficStateProvider(Protocol):
    def get_state(self, snapshot: Snapshot) -> TrafficState: ...


class SUMOTrafficStateProvider:
    def __init__(self, detection_distance: float = 50.0) -> None:
        self.detection_distance = detection_distance

    def get_state(self, snapshot: Snapshot) -> TrafficState:
        queues = []
        approaching = []
        moving = {lane: 0 for lane in snapshot.lane_lengths}
        for vehicle in snapshot.vehicles.values():
            if vehicle.lane in moving and vehicle.speed >= 0.1 and snapshot.lane_lengths[vehicle.lane] - vehicle.lane_position <= self.detection_distance:
                moving[vehicle.lane] += 1
        for d in DIRECTIONS:
            queues += [snapshot.lane_queues.get(f"{d}_in_2", 0),
                       sum(snapshot.lane_queues.get(f"{d}_in_{lane}", 0) for lane in (0, 1))]
            approaching += [moving.get(f"{d}_in_2", 0), sum(moving.get(f"{d}_in_{lane}", 0) for lane in (0, 1))]
        waits = [v.accumulated_waiting for v in snapshot.vehicles.values() if "_in_" in v.lane]
        return TrafficState(tuple(queues), sum(waits), max(waits, default=0), tuple(approaching))
