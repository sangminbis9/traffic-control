"""Evaluate Fixed-Time and DQN controllers on identical seeded demand files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

from model.env.intersection_env import IntersectionEnv
from model.utils.config import ProjectConfig, ensure_directories
from model.utils.metrics import save_comparison_plots, save_metrics


def evaluate_dqn(env: IntersectionEnv, model: DQN, episodes: int, seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode, seed in enumerate(seeds[:episodes]):
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(int(action))
        rows.append(env.episode_summary(episode))
    return rows


def evaluate_fixed(env: IntersectionEnv, episodes: int, seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode, seed in enumerate(seeds[:episodes]):
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _, terminated, truncated, _ = env.step(0)
        rows.append(env.episode_summary(episode))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=ProjectConfig().results_dir / "dqn_intersection.zip",
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--scenario", default="random")
    parser.add_argument("--episode-seconds", type=int, default=300)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    config = ProjectConfig()
    ensure_directories(config)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    dqn_model = DQN.load(str(args.model))

    dqn_env = IntersectionEnv(config, controller_name="DQN", scenario=args.scenario, use_gui=args.gui, episode_seconds=args.episode_seconds)
    fixed_env = IntersectionEnv(config, controller_name="Fixed-Time", scenario=args.scenario, use_gui=args.gui, episode_seconds=args.episode_seconds)
    try:
        rows = evaluate_fixed(fixed_env, args.episodes, seeds)
        rows.extend(evaluate_dqn(dqn_env, dqn_model, args.episodes, seeds))
    finally:
        fixed_env.close()
        dqn_env.close()

    output_csv = config.results_dir / "evaluation_metrics.csv"
    frame = save_metrics(rows, output_csv)
    save_comparison_plots(frame, config.results_dir)
    print(frame.groupby("controller")[
        ["avg_waiting_time", "avg_queue", "throughput", "phase_changes"]
    ].agg(["mean", "std"]).to_string())
    print(f"Saved metrics to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
