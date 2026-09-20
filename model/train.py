"""Train a Stable-Baselines3 DQN on the SUMO intersection."""

from __future__ import annotations

import argparse
from pathlib import Path
import uuid

from stable_baselines3.common.env_checker import check_env

from model.env.intersection_env import IntersectionEnv
from model.training import TrainingOptions, TrainingRunner
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
    parser.add_argument("--validation-interval", type=int, default=10_000)
    parser.add_argument("--validation-episodes", type=int, default=5)
    args = parser.parse_args()

    config = ProjectConfig()
    ensure_directories(config)
    if args.check_env:
        env = IntersectionEnv(
            config,
            controller_name="DQN-Check",
            scenario=args.scenario,
            seed=args.seed,
            use_gui=args.gui,
            episode_seconds=args.episode_seconds,
        )
        try:
            check_env(env, warn=True)
            print("Gymnasium check_env: PASS")
        finally:
            env.close()

    session_id = f"cli_{uuid.uuid4().hex[:8]}"
    options = TrainingOptions(
        total_steps=args.timesteps,
        scenario=args.scenario,
        episode_seconds=args.episode_seconds,
        seed=args.seed,
        validation_interval=args.validation_interval,
        validation_episodes=args.validation_episodes,
        learning_starts=args.learning_starts,
        resume_additional_steps=args.resume is not None,
        use_gui=args.gui,
    )
    runner = TrainingRunner(session_id, options, resume_checkpoint=args.resume)
    result = runner.run()
    final_model = runner.output_dir / "final.zip"
    canonical_model = config.results_dir / "dqn_intersection.zip"
    canonical_model.write_bytes(final_model.read_bytes())
    print(f"Training status: {result['status']}")
    print(f"Saved DQN model to {canonical_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
