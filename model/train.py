"""Train a Stable-Baselines3 DQN on the SUMO intersection."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from model.env.intersection_env import IntersectionEnv
from model.utils.config import ProjectConfig, ensure_directories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scenario", default="random")
    parser.add_argument("--episode-seconds", type=int, default=300)
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()

    config = ProjectConfig()
    ensure_directories(config)
    env = IntersectionEnv(
        config,
        controller_name="DQN",
        scenario=args.scenario,
        seed=args.seed,
        use_gui=args.gui,
        episode_seconds=args.episode_seconds,
    )
    try:
        if args.check_env:
            check_env(env, warn=True)
            print("Gymnasium check_env: PASS")
        tensorboard_log = (
            str(config.results_dir / "tensorboard")
            if importlib.util.find_spec("tensorboard")
            else None
        )
        if tensorboard_log is None:
            print("TensorBoard is not installed; continuing without TensorBoard logging.")
        if args.resume is not None:
            print(f"Resuming DQN model from {args.resume}")
            model = DQN.load(str(args.resume), env=env, device="auto")
            model.learn(
                total_timesteps=args.timesteps,
                reset_num_timesteps=False,
                progress_bar=False,
            )
        else:
            model = DQN(
                config.dqn.policy,
                env,
                learning_rate=config.dqn.learning_rate,
                buffer_size=config.dqn.buffer_size,
                learning_starts=(
                    config.dqn.learning_starts
                    if args.learning_starts is None
                    else args.learning_starts
                ),
                batch_size=config.dqn.batch_size,
                gamma=config.dqn.gamma,
                train_freq=config.dqn.train_freq,
                gradient_steps=config.dqn.gradient_steps,
                target_update_interval=config.dqn.target_update_interval,
                exploration_fraction=config.dqn.exploration_fraction,
                exploration_final_eps=config.dqn.exploration_final_eps,
                seed=args.seed,
                verbose=1,
                tensorboard_log=tensorboard_log,
            )
            model.learn(total_timesteps=args.timesteps, progress_bar=False)
        output = config.results_dir / "dqn_intersection"
        model.save(str(output))
        print(f"Saved DQN model to {output}.zip")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
