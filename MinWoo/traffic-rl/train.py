from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from build_network import network_fingerprint
from controller.traffic_dqn import TrafficDQN

from env.intersection_env import IntersectionEnv
from utils.config import ROOT, load_config


class ProgressCallback(BaseCallback):
    def _on_step(self) -> bool:
        if self.num_timesteps % 1000 == 0:
            print(f"Training steps={self.num_timesteps}, updates={self.model._n_updates}", flush=True)
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--learning-starts", type=int)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/dqn")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--backend", choices=["traci", "libsumo"], default="traci")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config_path = args.config or (args.resume.with_suffix(".config.json") if args.resume else None)
    config = load_config(config_path)
    if args.seed is not None:
        config.training.seed = args.seed
    if args.steps is not None:
        config.training.total_timesteps = args.steps
    if args.learning_starts is not None:
        config.training.learning_starts = args.learning_starts
    if config.training.total_timesteps <= 0:
        raise ValueError("steps must be positive")
    config.validate()
    if config.training.total_timesteps <= config.training.learning_starts and not args.resume:
        print("WARNING: this run ends before gradient updates. Use --learning-starts 500 for a short smoke test.")
    output = args.output or ROOT / "results/training" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    config.save(output / "config.json")
    torch.set_num_threads(config.training.torch_threads)
    raw_env = IntersectionEnv(config, "human" if args.gui else None, output / "simulation", backend=args.backend)
    env = Monitor(raw_env, str(output / "monitor.csv"))
    model = None
    interrupted = False
    try:
        check_env(raw_env, warn=True)
        if args.resume:
            model = TrafficDQN.load(args.resume, env=env, device="cpu")
            replay = args.resume.with_suffix(".replay.pkl")
            if replay.exists():
                model.load_replay_buffer(replay)
        else:
            params = asdict(config.training)
            for key in ("traffic_seed_min", "traffic_seed_max", "total_timesteps", "net_arch", "torch_threads"):
                params.pop(key)
            model = TrafficDQN("MlpPolicy", env, **params, policy_kwargs={"net_arch": config.training.net_arch},
                        verbose=1, device="cpu")
        checkpoints = CheckpointCallback(save_freq=10000, save_path=str(output / "checkpoints"), name_prefix="dqn")
        try:
            model.learn(total_timesteps=config.training.total_timesteps, callback=[checkpoints, ProgressCallback()],
                        reset_num_timesteps=not bool(args.resume), log_interval=5)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted: saving current model and replay buffer.")
        model.save(args.model)
        model.save_replay_buffer(args.model.with_suffix(".replay.pkl"))
        config.save(args.model.with_suffix(".config.json"))
        metadata = {"timesteps": model.num_timesteps, "gradient_updates": model._n_updates,
                    "interrupted": interrupted, "training_seed_range": [config.training.traffic_seed_min, config.training.traffic_seed_max],
                    "observation_size": raw_env.observation_space.shape[0], "network_sha256": network_fingerprint(),
                    "network_hash_format": "canonical_xml_v1",
                    "training_output": str(output.resolve())}
        args.model.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        restored = DQN.load(args.model, device="cpu")
        observation, _ = raw_env.reset(seed=123)
        original_action, _ = model.predict(observation, deterministic=True)
        restored_action, _ = restored.predict(observation, deterministic=True)
        if int(original_action) != int(restored_action):
            raise AssertionError("Saved model prediction differs after reload")
        print(json.dumps(metadata, indent=2))
        print("Model saved and reload prediction verified.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
