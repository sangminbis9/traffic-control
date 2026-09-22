"""Collect every evaluated checkpoint, including unsuccessful candidates."""
from pathlib import Path

import pandas as pd

from utils.config import ROOT
from analyze_experiments import plot_selection


if __name__ == "__main__":
    frames = []
    for path in sorted((ROOT / "results").glob("selection_*/leaderboard.csv")):
        frame = pd.read_csv(path)
        frame["family"] = path.parent.name.removeprefix("selection_")
        frames.append(frame)
        plot_selection(path, path.parent / "plots")
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(ROOT / "results/experiment_leaderboard.csv", index=False)
    grouped = combined.groupby(["family", "steps"])[["waiting_improvement_pct", "queue_improvement_pct", "throughput_change_pct"]].agg(["mean", "std", "count"])
    grouped.to_csv(ROOT / "results/experiment_budget_summary.csv")
    print(combined.sort_values("waiting_improvement_pct", ascending=False)[["family", "training_seed", "steps",
        "waiting_improvement_pct", "queue_improvement_pct", "throughput_change_pct", "all_primary_better"]].round(2).to_string(index=False))
