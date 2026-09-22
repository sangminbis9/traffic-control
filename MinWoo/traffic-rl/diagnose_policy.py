"""Record actual policy requests, accepted transitions and raw traffic states."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from stable_baselines3 import DQN

from env.intersection_env import IntersectionEnv
from utils.config import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3001)
    parser.add_argument("--scenario", default="heavy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = load_config(args.model.with_suffix(".config.json"))
    model = DQN.load(args.model, device="cpu")
    env = IntersectionEnv(config, output_dir=args.output, backend="libsumo")
    rows = []
    try:
        observation, info = env.reset(seed=args.seed, options={"traffic_seed": args.seed, "scenario": args.scenario})
        while True:
            action, _ = model.predict(observation, deterministic=True)
            row = {"time": info["time"], "phase": info["phase"], "stage": info["stage"], "elapsed": info["phase_elapsed"],
                   "action": int(action), **{f"queue_{i}": q for i, q in enumerate(info["queues"])}}
            observation, reward, _, truncated, info = env.step(int(action))
            row.update(reward=reward, accepted=info["action_accepted"], switch_reason=info["switch_reason"])
            rows.append(row)
            if truncated:
                break
    finally:
        env.close()
    pd.DataFrame(rows).to_csv(args.output / "actions.csv", index=False)
