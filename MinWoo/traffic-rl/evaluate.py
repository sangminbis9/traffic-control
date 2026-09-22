from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import DQN

from build_network import network_fingerprint

from controller.fixed_controller import FixedTimeController
from env.intersection_env import IntersectionEnv
from utils.config import ROOT, SCENARIOS, load_config
from utils.plots import plot_evaluation, plot_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/dqn.zip")
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--episodes", type=int, default=30, help="Seed count per scenario")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS[:-1]))
    parser.add_argument("--output", type=Path, default=ROOT / "results/evaluation")
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--backend", choices=["traci", "libsumo"], default="traci")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")
    model_config = args.model.with_suffix(".config.json")
    config = load_config(args.config or (model_config if not args.fixed_only else None))
    seeds = range(args.seed_start, args.seed_start + args.episodes)
    if any(config.training.traffic_seed_min <= seed <= config.training.traffic_seed_max for seed in seeds):
        raise ValueError("Evaluation seeds overlap the training traffic seed range")
    model = None
    metadata = None
    torch.set_num_threads(config.training.torch_threads)
    if not args.fixed_only:
        metadata = json.loads(args.model.with_suffix(".meta.json").read_text(encoding="utf-8"))
        trained_config = load_config(model_config)
        for section in ("normalization", "signal", "simulation", "reward", "observation"):
            if asdict(getattr(trained_config, section)) != asdict(getattr(config, section)):
                raise ValueError(f"Evaluation {section} differs from the saved training configuration")
        lo, hi = metadata["training_seed_range"]
        if any(lo <= seed <= hi for seed in seeds):
            raise ValueError("Evaluation seeds overlap model training seeds")
        digest = network_fingerprint()
        if digest != metadata["network_sha256"]:
            raise ValueError("Network differs from the training network")
        model = DQN.load(args.model, device="cpu")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "episodes.csv").exists():
        raise FileExistsError("Use a fresh --output directory to avoid mixing evaluations")
    config.save(args.output / "config.json")
    rows = []
    fixed = FixedTimeController(config.signal.fixed_green)
    for scenario in args.scenarios:
        for seed in seeds:
            route_hash = None
            for controller in (["fixed"] if args.fixed_only else ["fixed", "dqn"]):
                env = IntersectionEnv(config, output_dir=args.output / "runs" / f"{scenario}_{seed}_{controller}", controller_name=controller, backend=args.backend)
                try:
                    observation, info = env.reset(seed=seed, options={"traffic_seed": seed, "scenario": scenario})
                    if route_hash is not None and route_hash != info["route_sha256"]:
                        raise AssertionError("Paired traffic XML does not match")
                    route_hash = info["route_sha256"]
                    while True:
                        action = fixed.action(env.signal) if controller == "fixed" else int(model.predict(observation, deterministic=True)[0])
                        observation, _, terminated, truncated, info = env.step(action)
                        if terminated or truncated:
                            break
                    rows.append(info["episode_metrics"])
                    pd.DataFrame(rows).to_csv(args.output / "episodes.csv", index=False)
                    print(f"{scenario} seed={seed} {controller}: waiting={rows[-1]['avg_waiting_time']:.2f}, throughput={rows[-1]['throughput']}", flush=True)
                finally:
                    env.close()
    frame = pd.DataFrame(rows)
    metrics = ["avg_waiting_time", "max_waiting_time", "avg_queue", "max_queue", "throughput", "phase_changes", "episode_reward", "unfinished", "not_inserted"]
    for keys, name in [(["controller"], "summary"), (["traffic_scenario", "controller"], "summary_by_scenario")]:
        summary = frame.groupby(keys)[metrics].agg(["mean", "std", "count"])
        summary.columns = ["_".join(column) for column in summary.columns]
        summary.to_csv(args.output / f"{name}.csv")
    if not args.fixed_only:
        paired = frame.pivot(index=["traffic_scenario", "seed"], columns="controller", values=metrics)
        delta = pd.DataFrame({metric: paired[metric]["dqn"] - paired[metric]["fixed"] for metric in metrics})
        delta.to_csv(args.output / "paired_deltas.csv")
        delta.agg(["mean", "std", "count"]).to_csv(args.output / "paired_summary.csv")
        plot_evaluation(frame, args.output / "plots")
        training_log = Path(metadata["training_output"]) / "simulation/episodes.csv"
        if training_log.exists():
            plot_training(training_log, args.output / "plots/training_rewards.png")
    print(frame.groupby("controller")[metrics[:5]].mean().to_string())


if __name__ == "__main__":
    main()
