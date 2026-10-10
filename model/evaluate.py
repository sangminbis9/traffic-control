"""Evaluate selected controllers on identical seeded demand files.

For the frozen rule tuning/heldout protocol and a fixed policy with identical
safety constraints, use model.experiments.rule_based_comparison instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from model.controller.traffic_dqn import TrafficDQN
from model.controller.rule_based_controller import RuleBasedController, RulePolicyConfig
from model.env.intersection_env import IntersectionEnv
from model.utils.config import ProjectConfig, ensure_directories
from model.utils.metrics import save_comparison_plots, save_metrics


def evaluate_dqn(env: IntersectionEnv, model: TrafficDQN, episodes: int, seeds: list[int]) -> list[dict[str, object]]:
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


def evaluate_rule(env: IntersectionEnv, policy: RuleBasedController, episodes: int,
                  seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode, seed in enumerate(seeds[:episodes]):
        observation, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            observation, _, terminated, truncated, _ = env.step(policy.action(observation))
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
    parser.add_argument("--controllers", nargs="+", choices=("fixed", "rule", "dqn"),
                        default=["fixed", "dqn"])
    parser.add_argument("--rule-policy", type=Path,
                        help="Rule parameters JSON, or selection.json from the frozen comparison")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for evaluation CSV and plots")
    args = parser.parse_args()
    if args.episodes < 1 or args.episode_seconds < 1:
        parser.error("episodes and episode-seconds must be positive")

    config = ProjectConfig()
    ensure_directories(config)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    controllers = list(dict.fromkeys(args.controllers))
    dqn_model = TrafficDQN.load(str(args.model), device="cpu") if "dqn" in controllers else None
    if dqn_model is not None and (dqn_model.observation_space.shape != (60,) or dqn_model.action_space.n != 8):
        raise ValueError("Choose a current 60D/8-action checkpoint; the preserved legacy model is incompatible")
    rule_settings = RulePolicyConfig()
    if args.rule_policy:
        payload = json.loads(args.rule_policy.read_text(encoding="utf-8"))
        rule_settings = RulePolicyConfig.from_dict(payload.get("policy", payload))
    rows: list[dict[str, object]] = []
    for controller in controllers:
        label = {"fixed": "Fixed-Time", "rule": "Rule-Based", "dqn": "DQN"}[controller]
        env = IntersectionEnv(config, controller_name=label, scenario=args.scenario,
                              use_gui=args.gui, episode_seconds=args.episode_seconds)
        try:
            if controller == "fixed":
                rows.extend(evaluate_fixed(env, args.episodes, seeds))
            elif controller == "rule":
                rows.extend(evaluate_rule(env, RuleBasedController(config, rule_settings), args.episodes, seeds))
            else:
                rows.extend(evaluate_dqn(env, dqn_model, args.episodes, seeds))
        finally:
            env.close()

    output_dir = args.output_dir or config.results_dir
    output_csv = output_dir / "evaluation_metrics.csv"
    frame = save_metrics(rows, output_csv)
    save_comparison_plots(frame, output_dir)
    print(frame.groupby("controller")[
        ["avg_waiting_time", "avg_queue", "throughput", "phase_changes"]
    ].agg(["mean", "std"]).to_string())
    print(f"Saved metrics to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
