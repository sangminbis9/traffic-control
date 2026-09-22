"""Produce transparent learning-budget plots and held-out paired statistics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from utils.plots import plt


def plot_selection(leaderboard: Path, output: Path) -> None:
    frame = pd.read_csv(leaderboard)
    output.mkdir(parents=True, exist_ok=True)
    for column, ylabel, filename in [
        ("waiting_improvement_pct", "Waiting reduction vs Fixed (%)", "learning_budget_waiting.png"),
        ("queue_improvement_pct", "Queue reduction vs Fixed (%)", "learning_budget_queue.png"),
        ("throughput_change_pct", "Throughput change vs Fixed (%)", "learning_budget_throughput.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
        for seed, group in frame.groupby("training_seed"):
            group = group.sort_values("steps")
            ax.plot(group.steps / 1000, group[column], marker="o", label=f"Training seed {seed}")
        ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Fixed-Time")
        ax.set(xlabel="Training decisions (thousands)", ylabel=ylabel, title="Validation across six traffic scenarios")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def heldout_statistics(fixed_csv: Path, dqn_csvs: list[Path], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    fixed = pd.read_csv(fixed_csv)
    frames = []
    labels = []
    for index, path in enumerate(dqn_csvs):
        frame = pd.read_csv(path)
        frame["model_index"] = index
        frame["model_source"] = str(path)
        frames.append(frame)
        match = re.search(r"_s(\d+)", str(path))
        labels.append(f"DQN seed {match.group(1)}" if match else f"DQN model {index + 1}")
    dqn = pd.concat(frames, ignore_index=True)
    merged = dqn.merge(fixed, on=["traffic_scenario", "seed"], suffixes=("_dqn", "_fixed"), validate="many_to_one")
    if len(merged) != len(dqn) or not (merged.route_sha256_dqn == merged.route_sha256_fixed).all():
        raise AssertionError("Unmatched held-out traffic")
    merged.to_csv(output / "paired_cases.csv", index=False)
    metrics = ["avg_waiting_time", "avg_queue", "throughput", "max_waiting_time", "phase_changes", "forced_changes", "not_inserted"]
    rng = np.random.default_rng(92187)
    summary = {}
    for metric in metrics:
        differences = merged[f"{metric}_dqn"] - merged[f"{metric}_fixed"]
        # Keep six fixed scenarios together when resampling a traffic seed.
        grouped = merged.assign(delta=differences).groupby(["model_index", "seed"]).delta.mean().unstack("seed")
        values = grouped.to_numpy()
        if np.isnan(values).any():
            raise ValueError("Every selected model must have the complete final traffic split")
        model_indices = rng.integers(values.shape[0], size=(10000, values.shape[0]))
        traffic_indices = rng.integers(values.shape[1], size=(10000, values.shape[1]))
        replicates = values[model_indices[:, :, None], traffic_indices[:, None, :]].mean(axis=(1, 2))
        summary[metric] = {"fixed_mean": float(fixed[metric].mean()), "dqn_mean": float(dqn[metric].mean()),
            "delta_mean": float(differences.mean()), "delta_ci95": np.quantile(replicates, [0.025, 0.975]).tolist()}
    summary["design"] = {"models": len(dqn_csvs), "traffic_seeds": int(fixed.seed.nunique()),
                         "scenarios": int(fixed.traffic_scenario.nunique()), "bootstrap": "paired, two-way model/traffic-seed resampling, 10000 draws"}
    (output / "statistics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dqn.groupby(["model_index", "traffic_scenario"])[metrics].mean().to_csv(output / "by_model_scenario.csv")
    fixed.groupby("traffic_scenario")[metrics].mean().to_csv(output / "fixed_by_scenario.csv")
    dqn.groupby("model_index")[metrics].mean().to_csv(output / "by_model.csv")
    for metric, label in [("avg_waiting_time", "Mean waiting (s)"), ("avg_queue", "Mean queue (vehicles)"), ("throughput", "Throughput (vehicles / 300s)")]:
        fig, ax = plt.subplots(figsize=(7, 4.6), layout="constrained")
        means = [summary[metric]["fixed_mean"]] + [float(frame[metric].mean()) for frame in frames]
        ax.bar(["Fixed"] + labels, means, color=["#6a7b8d"] + ["#2676d2"] * len(frames))
        ax.set(ylabel=label, title=f"Held-out test: {fixed.traffic_scenario.nunique()} scenarios, traffic seeds {fixed.seed.min()}–{fixed.seed.max()}")
        ax.grid(axis="y", alpha=0.2)
        fig.savefig(output / f"final_{metric}.png", dpi=160)
        plt.close(fig)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard", type=Path)
    parser.add_argument("--fixed", type=Path)
    parser.add_argument("--dqn", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.leaderboard:
        plot_selection(args.leaderboard, args.output)
    if args.fixed and args.dqn:
        print(json.dumps(heldout_statistics(args.fixed, args.dqn, args.output), indent=2))
