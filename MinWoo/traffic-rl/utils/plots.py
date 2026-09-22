from __future__ import annotations

from pathlib import Path
import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "traffic-rl-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for metric, label in [("avg_waiting_time", "Mean accumulated waiting per inserted vehicle (s)"),
                          ("avg_queue", "Time-average queued vehicles"), ("throughput", "Arrived vehicles"),
                          ("phase_changes", "Phase changes")]:
        grouped = frame.groupby("controller")[metric].agg(["mean", "std"]).reindex(["fixed", "dqn"])
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0), color=["#607d8b", "#2676d2"], capsize=5)
        ax.set_ylabel(label)
        ax.set_title("Fixed-Time vs DQN (mean ± episode SD)")
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(output / f"{metric}.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for controller, group in frame.groupby("controller"):
        ax.plot(range(1, len(group) + 1), group["episode_reward"], label=controller, alpha=0.8)
    ax.set(xlabel="Evaluation case (scenario, seed)", ylabel="Episode reward", title="Paired evaluation rewards")
    ax.legend()
    fig.savefig(output / "evaluation_rewards.png", dpi=160)
    plt.close(fig)


def plot_training(path: Path, output: Path) -> None:
    frame = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.plot(frame["episode"], frame["episode_reward"], alpha=0.4, label="Episode")
    ax.plot(frame["episode"], frame["episode_reward"].rolling(10, min_periods=1).mean(), label="10-episode mean")
    ax.set(xlabel="Training episode", ylabel="Episode reward", title="Training reward (randomized traffic)")
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
