"""Apply the predeclared common-budget validation selection rule."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from build_network import network_fingerprint
from utils.config import ROOT


if __name__ == "__main__":
    output = ROOT / "experiments/frozen_selection.json"
    if output.exists():
        raise FileExistsError("Selection is already frozen")
    table = pd.read_csv(ROOT / "results/selection_approach/leaderboard.csv")
    if len(table) != 12:
        raise ValueError("Wait for all 3 seeds × 4 declared checkpoints")
    eligible = []
    for steps, group in table.groupby("steps"):
        if set(group.training_seed) == {11, 22, 33} and group.all_primary_better.all():
            eligible.append((group.avg_waiting_time_dqn.mean(), int(steps)))
    if not eligible:
        raise ValueError("No common budget meets the declared validation criteria")
    _, steps = min(eligible)
    chosen = table[table.steps == steps].sort_values("training_seed")
    demo_seed = int(chosen.loc[chosen.avg_waiting_time_dqn.idxmin(), "training_seed"])
    models = []
    destination = ROOT / "results/models"
    destination.mkdir(parents=True, exist_ok=True)
    for row in chosen.to_dict("records"):
        source = ROOT / "results/experiments_approach" / row["model"]
        target = destination / f"dqn_s{row['training_seed']}.zip"
        for suffix in (".zip", ".config.json", ".meta.json"):
            shutil.copy2(source.with_suffix(suffix), target.with_suffix(suffix))
        models.append({"training_seed": int(row["training_seed"]), "steps": steps,
            "model": target.relative_to(ROOT).as_posix(), "model_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "source_model": source.relative_to(ROOT).as_posix(),
            "validation_waiting_improvement_pct": row["waiting_improvement_pct"],
            "validation_queue_improvement_pct": row["queue_improvement_pct"],
            "validation_throughput_change_pct": row["throughput_change_pct"]})
    for suffix in (".zip", ".config.json", ".meta.json"):
        shutil.copy2(destination / f"dqn_s{demo_seed}{suffix}", destination / f"dqn{suffix}")
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "network_sha256": network_fingerprint(),
        "selection_rule": "Lowest across-seed mean validation waiting among common budgets where every seed improves waiting/queue and maintains throughput",
        "validation_seeds": [3001, 3002, 3003, 3004, 3005], "test_seed_start": 4001, "test_seed_count": 30,
        "default_model_training_seed": demo_seed, "models": models}
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
