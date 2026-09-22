"""Evaluate a frozen selection exactly once on a declared untouched split."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
from stable_baselines3 import DQN

from analyze_experiments import heldout_statistics
from build_network import network_fingerprint
from experiment_evaluate import compare, evaluate_cases
from utils.config import ROOT, load_config


def run_job(config_path: str, model_path: str | None, output: str, seeds: list[int]) -> str:
    torch.set_num_threads(1)
    config = load_config(config_path)
    model = DQN.load(model_path, device="cpu") if model_path else None
    evaluate_cases(config, Path(output), seeds, model)
    return str(Path(output) / "episodes.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (args.output / "manifest.json").exists():
        raise FileExistsError("This final evaluation was already started; use its existing results")
    if manifest["network_sha256"] != network_fingerprint():
        raise ValueError("Frozen network has changed")
    model_paths = [ROOT / run["model"] for run in manifest["models"]]
    reference = load_config(model_paths[0].with_suffix(".config.json"))
    seeds = list(range(manifest["test_seed_start"], manifest["test_seed_start"] + manifest["test_seed_count"]))
    for run, path in zip(manifest["models"], model_paths):
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["model_sha256"]:
            raise ValueError("Frozen model has changed")
        config = load_config(path.with_suffix(".config.json"))
        for section in ("simulation", "signal", "traffic", "reward", "normalization", "observation"):
            if asdict(getattr(config, section)) != asdict(getattr(reference, section)):
                raise ValueError(f"Final models have different {section} settings")
        if any(config.training.traffic_seed_min <= seed <= config.training.traffic_seed_max for seed in seeds):
            raise ValueError("Test split overlaps training")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=3) as pool:
        fixed_future = pool.submit(run_job, str(model_paths[0].with_suffix(".config.json")), None,
                                   str(args.output / "fixed"), seeds)
        futures = [pool.submit(run_job, str(path.with_suffix(".config.json")), str(path),
                               str(args.output / f"dqn_s{run['training_seed']}"), seeds)
                   for run, path in zip(manifest["models"], model_paths)]
        fixed_path = Path(fixed_future.result())
        dqn_paths = [Path(future.result()) for future in futures]
    statistics = heldout_statistics(fixed_path, dqn_paths, args.output / "analysis")
    import pandas as pd
    fixed = pd.read_csv(fixed_path)
    rows = [{"training_seed": run["training_seed"], "steps": run["steps"],
             **compare(fixed, pd.read_csv(path))} for run, path in zip(manifest["models"], dqn_paths)]
    pd.DataFrame(rows).to_csv(args.output / "model_comparison.csv", index=False)
    print(json.dumps(statistics, indent=2), flush=True)
    print(pd.DataFrame(rows)[["training_seed", "steps", "waiting_improvement_pct", "queue_improvement_pct", "throughput_change_pct", "all_primary_better"]].to_string(index=False))


if __name__ == "__main__":
    main()
