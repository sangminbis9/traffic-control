"""Evaluate saved learning milestones on a fixed, declared validation split."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import pandas as pd
import torch
from stable_baselines3 import DQN

from build_network import network_fingerprint
from controller.fixed_controller import FixedTimeController
from env.intersection_env import IntersectionEnv
from utils.config import SCENARIOS, Config, load_config


def evaluate_cases(config: Config, output: Path, seeds: list[int], model: DQN | None, backend: str = "libsumo") -> pd.DataFrame:
    controller = "fixed" if model is None else getattr(model, "controller_name", "dqn")
    env = IntersectionEnv(config, output_dir=output / "simulation", controller_name=controller, backend=backend)
    fixed = FixedTimeController(config.signal.fixed_green)
    rows = []
    try:
        for scenario in SCENARIOS[:-1]:
            for seed in seeds:
                observation, _ = env.reset(seed=seed, options={"traffic_seed": seed, "scenario": scenario})
                while True:
                    action = fixed.action(env.signal) if model is None else int(model.predict(observation, deterministic=True)[0])
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        rows.append(info["episode_metrics"])
                        break
    finally:
        env.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "episodes.csv", index=False)
    return frame


def compare(fixed: pd.DataFrame, dqn: pd.DataFrame) -> dict:
    keys = ["traffic_scenario", "seed"]
    pairs = dqn.merge(fixed, on=keys, suffixes=("_dqn", "_fixed"), validate="one_to_one")
    if len(pairs) != len(fixed) or not (pairs.route_sha256_dqn == pairs.route_sha256_fixed).all():
        raise AssertionError("Missing or mismatched paired traffic cases")
    row = {"cases": len(pairs)}
    for metric in ("avg_waiting_time", "avg_queue", "throughput", "max_waiting_time", "phase_changes", "forced_changes", "not_inserted"):
        for controller in ("fixed", "dqn"):
            row[f"{metric}_{controller}"] = float(pairs[f"{metric}_{controller}"].mean())
        row[f"{metric}_delta"] = row[f"{metric}_dqn"] - row[f"{metric}_fixed"]
    row["waiting_improvement_pct"] = -100 * row["avg_waiting_time_delta"] / row["avg_waiting_time_fixed"]
    row["queue_improvement_pct"] = -100 * row["avg_queue_delta"] / row["avg_queue_fixed"]
    row["throughput_change_pct"] = 100 * row["throughput_delta"] / row["throughput_fixed"]
    row["waiting_case_win_rate"] = float((pairs.avg_waiting_time_dqn < pairs.avg_waiting_time_fixed).mean())
    row["all_primary_better"] = bool(row["avg_waiting_time_delta"] < 0 and row["avg_queue_delta"] < 0 and row["throughput_delta"] >= 0)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=3001)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--expected-models", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "leaderboard.csv"
    rows = pd.read_csv(summary_path).to_dict("records") if summary_path.exists() else []
    evaluated = {r["model"] for r in rows}
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    started = time.monotonic()
    fixed = None
    baseline_config = None
    while True:
        for meta_path in sorted(args.root.rglob("dqn_*.meta.json")):
            model_path = meta_path.with_name(meta_path.name.replace(".meta.json", ".zip"))
            identity = str(model_path.relative_to(args.root)).replace("\\", "/")
            if identity in evaluated:
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if metadata["network_sha256"] != network_fingerprint():
                raise ValueError("Network mismatch")
            config = load_config(model_path.with_suffix(".config.json"))
            lo, hi = metadata["training_seed_range"]
            if any(lo <= seed <= hi for seed in seeds):
                raise ValueError("Selection seeds overlap training")
            if fixed is None:
                saved = args.output / "fixed/episodes.csv"
                if saved.exists():
                    fixed = pd.read_csv(saved)
                    baseline_config = load_config(args.output / "fixed/config.json")
                else:
                    baseline_config = config
                    fixed = evaluate_cases(config, args.output / "fixed", seeds, None)
                    config.save(args.output / "fixed/config.json")
            for section in ("simulation", "signal", "traffic"):
                if asdict(getattr(config, section)) != asdict(getattr(baseline_config, section)):
                    raise ValueError(f"Cached baseline incompatible: {section}")
            model = DQN.load(model_path, device="cpu")
            run_output = args.output / identity.replace("/", "__").replace(".zip", "")
            dqn = evaluate_cases(config, run_output, seeds, model)
            row = {"model": identity, "training_seed": metadata["training_seed"], "steps": metadata["timesteps"],
                   "episodes_path": str((run_output / "episodes.csv").resolve()), **compare(fixed, dqn)}
            rows.append(row)
            evaluated.add(identity)
            pd.DataFrame(rows).sort_values("avg_waiting_time_dqn").to_csv(summary_path, index=False)
            print(json.dumps({k: row[k] for k in ("model", "waiting_improvement_pct", "queue_improvement_pct", "throughput_change_pct", "all_primary_better")}), flush=True)
        if not args.expected_models or len(evaluated) >= args.expected_models:
            break
        if time.monotonic() - started > args.timeout:
            raise TimeoutError(f"Only {len(evaluated)}/{args.expected_models} models became available")
        time.sleep(5)


if __name__ == "__main__":
    main()
