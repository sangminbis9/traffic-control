"""A reproducible learning trajectory with evaluable milestone artifacts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from build_network import network_fingerprint
from controller.traffic_dqn import TrafficDQN
from env.intersection_env import IntersectionEnv
from utils.config import Config, load_config


class Milestones(BaseCallback):
    def __init__(self, config: Config, output: Path, checkpoints: list[int]):
        super().__init__()
        self.config, self.output = config, output
        self.checkpoints = set(checkpoints)
        self.started = time.perf_counter()

    def save(self, name: str) -> None:
        path = self.output / "models" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        self.config.save(path.with_suffix(".config.json"))
        metadata = {"timesteps": self.num_timesteps, "gradient_updates": self.model._n_updates,
            "training_seed_range": [self.config.training.traffic_seed_min, self.config.training.traffic_seed_max],
            "training_seed": self.config.training.seed, "observation_size": self.model.observation_space.shape[0],
            "network_sha256": network_fingerprint(), "network_hash_format": "canonical_xml_v1",
            "training_output": str(self.output.resolve()), "backend": "libsumo",
            "elapsed_seconds": time.perf_counter() - self.started}
        path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _on_step(self) -> bool:
        # Publish the final milestone only after learn() completes its last update.
        if self.num_timesteps in self.checkpoints and self.num_timesteps < self.config.training.total_timesteps:
            self.save(f"dqn_{self.num_timesteps}")
        if self.num_timesteps % 10000 == 0:
            progress = {"seed": self.config.training.seed, "steps": self.num_timesteps,
                        "updates": self.model._n_updates, "elapsed": round(time.perf_counter() - self.started, 1)}
            (self.output / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
            print(json.dumps(progress), flush=True)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--milestones", nargs="+", type=int, default=[20000, 50000, 100000, 200000, 300000])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config.training.seed = args.seed
    config.training.total_timesteps = args.steps
    config.validate()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "progress.json").exists():
        raise FileExistsError("Use a fresh experiment output directory")
    config.save(args.output / "config.json")
    torch.set_num_threads(config.training.torch_threads)
    env = Monitor(IntersectionEnv(config, output_dir=args.output / "simulation", backend="libsumo"), str(args.output / "monitor.csv"))
    params = asdict(config.training)
    for key in ("traffic_seed_min", "traffic_seed_max", "total_timesteps", "net_arch", "torch_threads"):
        params.pop(key)
    model = TrafficDQN("MlpPolicy", env, **params, policy_kwargs={"net_arch": config.training.net_arch}, device="cpu", verbose=0)
    callback = Milestones(config, args.output, args.milestones)
    try:
        model.learn(args.steps, callback=callback)
        callback.save(f"dqn_{args.steps}")
        model.save_replay_buffer(args.output / "models/final.replay.pkl")
        print(f"COMPLETE seed={args.seed} steps={model.num_timesteps}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
