"""Time-weighted metrics, including unfinished vehicles and insertion backlog."""
from __future__ import annotations

import csv
from pathlib import Path

from simulation.backend import Snapshot


class EpisodeMetrics:
    def __init__(self) -> None:
        self.waits: dict[str, float] = {}
        self.arrived: set[str] = set()
        self.queue_area = 0.0
        self.max_queue = 0
        self.duration = 0.0
        self.reward = 0.0
        self.collisions = 0
        self.teleports = 0
        self.pending = 0
        self.max_pending = 0
        self.term_totals = dict(queue=0.0, waiting=0.0, max_waiting=0.0, switching=0.0)

    def update(self, snapshot: Snapshot, dt: float) -> None:
        for vehicle in snapshot.departed:
            self.waits.setdefault(vehicle, 0.0)
        for vehicle, reading in snapshot.vehicles.items():
            self.waits[vehicle] = max(self.waits.get(vehicle, 0), reading.accumulated_waiting)
        self.arrived.update(snapshot.arrived)
        queue = sum(snapshot.lane_queues.values())
        self.queue_area += queue * dt
        self.max_queue = max(self.max_queue, queue)
        self.duration += dt
        self.collisions += snapshot.collisions
        self.teleports += snapshot.teleports
        self.pending = snapshot.pending
        self.max_pending = max(self.max_pending, snapshot.pending)

    def summary(self) -> dict:
        return {"avg_waiting_time": sum(self.waits.values()) / max(len(self.waits), 1),
                "max_waiting_time": max(self.waits.values(), default=0),
                "avg_queue": self.queue_area / max(self.duration, 1e-9), "max_queue": self.max_queue,
                "throughput": len(self.arrived), "episode_reward": self.reward, "departed": len(self.waits),
                "unfinished": len(self.waits) - len(self.arrived), "pending": self.pending,
                "max_pending": self.max_pending, "collisions": self.collisions, "teleports": self.teleports,
                "duration": self.duration, **{f"reward_{k}": v for k, v in self.term_totals.items()}}


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
