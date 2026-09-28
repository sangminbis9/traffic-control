"""Episode metric collection and plotting utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model.env.state_provider import TrafficSnapshot


@dataclass
class EpisodeMetrics:
    """Collect MinWoo-compatible time-weighted and per-vehicle metrics."""

    controller: str
    seed: int
    traffic_scenario: str
    waits: dict[str, float] = field(default_factory=dict)
    arrived_ids: set[str] = field(default_factory=set)
    queue_area: float = 0.0
    duration: float = 0.0
    max_queue: float = 0.0
    max_waiting: float = 0.0
    pending: int = 0
    max_pending: int = 0
    collisions: int = 0
    teleports: int = 0
    phase_changes: int = 0
    throughput: int = 0
    reward: float = 0.0

    forced_changes: int = 0

    def record(
        self,
        snapshot: TrafficSnapshot,
        reward: float = 0.0,
        phase_changed: bool = False,
        dt: float = 1.0,
    ) -> None:
        for vehicle_id in snapshot.departed_ids:
            self.waits.setdefault(vehicle_id, 0.0)
        for vehicle_id, waiting in snapshot.vehicle_waiting.items():
            self.waits[vehicle_id] = max(self.waits.get(vehicle_id, 0.0), float(waiting))
        self.arrived_ids.update(snapshot.arrived_ids)
        self.queue_area += snapshot.total_queue * dt
        self.duration += dt
        self.max_queue = max(self.max_queue, snapshot.total_queue)
        self.max_waiting = max(self.max_waiting, snapshot.max_waiting_time)
        self.pending = snapshot.pending
        self.max_pending = max(self.max_pending, snapshot.pending)
        self.collisions += snapshot.collisions
        self.teleports += snapshot.teleports
        self.throughput += snapshot.arrived
        self.reward += float(reward)
        if phase_changed:
            self.phase_changes += 1

    def summary(self, episode: int | None = None) -> dict[str, float | int | str]:
        row: dict[str, float | int | str] = {
            "episode": -1 if episode is None else episode,
            "controller": self.controller,
            "seed": self.seed,
            "traffic_scenario": self.traffic_scenario,
            "avg_waiting_time": float(sum(self.waits.values()) / max(len(self.waits), 1)),
            "max_waiting_time": float(max(self.waits.values(), default=self.max_waiting)),
            "avg_queue": float(self.queue_area / max(self.duration, 1e-9)),
            "max_queue": float(self.max_queue),
            "throughput": int(self.throughput),
            "phase_changes": int(self.phase_changes),
            "forced_changes": int(self.forced_changes),
            "episode_reward": float(self.reward),
            "departed": len(self.waits),
            "unfinished": max(len(self.waits) - len(self.arrived_ids), 0),
            "pending": int(self.pending),
            "max_pending": int(self.max_pending),
            "collisions": int(self.collisions),
            "teleports": int(self.teleports),
            "duration": float(self.duration),
        }
        return row


def save_metrics(rows: Sequence[dict[str, object]], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)
    return frame


def save_comparison_plots(frame: pd.DataFrame, output_dir: Path) -> None:
    """Save the requested baseline comparison and reward plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("avg_waiting_time", "Average Waiting Time", "seconds", "average_waiting_time.png"),
        ("avg_queue", "Average Queue Length", "vehicles", "average_queue.png"),
        ("throughput", "Throughput", "vehicles", "throughput.png"),
        ("phase_changes", "Phase Changes", "changes", "phase_changes.png"),
    ]
    for column, title, ylabel, filename in metrics:
        grouped = frame.groupby("controller")[column].agg(["mean", "std"])
        ax = grouped["mean"].plot(kind="bar", yerr=grouped["std"].fillna(0.0), capsize=4, color=["#4c78a8", "#f58518"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / filename, dpi=160)
        plt.close(ax.figure)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for controller, group in frame.groupby("controller"):
        reward = group.sort_values("episode")["episode_reward"].to_numpy()
        if len(reward) > 1:
            window = min(10, len(reward))
            smooth = pd.Series(reward).rolling(window, min_periods=1).mean()
            ax.plot(smooth, label=controller)
        else:
            ax.plot(reward, marker="o", label=controller)
    ax.set_title("Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "episode_reward.png", dpi=160)
    plt.close(fig)
