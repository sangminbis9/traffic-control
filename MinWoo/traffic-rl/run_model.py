"""Run a saved model through real TraCI, optionally displaying SUMO GUI."""
import argparse
from pathlib import Path
import time

import torch
from stable_baselines3 import DQN

from env.intersection_env import IntersectionEnv
from utils.config import ROOT, SCENARIOS, load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/dqn.zip")
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--scenario", choices=SCENARIOS, default="heavy")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--delay", type=float, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "results/model_demo")
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.model.with_suffix(".config.json"))
    model = DQN.load(args.model, device="cpu")
    env = IntersectionEnv(config, render_mode="human" if args.gui else None, output_dir=args.output)
    try:
        observation, info = env.reset(seed=args.seed, options={"traffic_seed": args.seed, "scenario": args.scenario})
        next_print = 0
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, _, truncated, info = env.step(int(action))
            if info["time"] >= next_print:
                print(f"t={info['time']:.0f} phase={info['phase']} stage={info['stage']} queue={sum(info['queues']):.0f} waiting={info['total_waiting']:.1f}", flush=True)
                next_print += 10
            if truncated:
                print(info["episode_metrics"])
                break
            if args.delay:
                time.sleep(args.delay)
    finally:
        env.close()
