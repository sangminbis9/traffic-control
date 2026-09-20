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
    """Collect raw per-decision values and convert them to one CSV row."""

    controller: str
    seed: int
    traffic_scenario: str
    queue_samples: list[float] = field(default_factory=list)
    waiting_samples: list[float] = field(default_factory=list)
    max_waiting_samples: list[float] = field(default_factory=list)
    max_queue_samples: list[float] = field(default_factory=list)
    waiting_per_vehicle_samples: list[float] = field(default_factory=list)
    phase_changes: int = 0
    throughput: int = 0
    reward: float = 0.0

    def record(self, snapshot: TrafficSnapshot, reward: float = 0.0, phase_changed: bool = False) -> None:
        self.queue_samples.append(snapshot.total_queue)
        self.max_queue_samples.append(max(snapshot.queue_by_group, default=0.0))
        self.waiting_samples.append(snapshot.total_waiting_time)
        self.max_waiting_samples.append(snapshot.max_waiting_time)
        self.waiting_per_vehicle_samples.append(
            snapshot.total_waiting_time / max(snapshot.vehicle_count, 1)
        )
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
            "avg_waiting_time": float(np.mean(self.waiting_per_vehicle_samples)) if self.waiting_per_vehicle_samples else 0.0,
            "max_waiting_time": float(max(self.max_waiting_samples, default=0.0)),
            "avg_queue": float(np.mean(self.queue_samples)) if self.queue_samples else 0.0,
            "max_queue": float(max(self.max_queue_samples, default=0.0)),
            "throughput": int(self.throughput),
            "phase_changes": int(self.phase_changes),
            "episode_reward": float(self.reward),
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
