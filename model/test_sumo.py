"""Smoke test TraCI, signal switching, queues, and waiting-time reporting."""

from __future__ import annotations

import argparse
import math

import numpy as np

try:
    from model.env.intersection_env import IntersectionEnv
    from model.env.state_provider import LANE_GROUPS
    from model.controller.signal_controller import GREEN_LINKS, LINK_COUNT, PHASE_COUNT
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing Python dependency: {exc.name}. Run 'python -m pip install -r model/requirements.txt' "
        "inside the activated virtual environment."
    ) from exc
from model.utils.config import ProjectConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scenario", default="uniform")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    config = ProjectConfig()
    env = IntersectionEnv(
        config,
        controller_name="TraCI-Test",
        scenario=args.scenario,
        seed=args.seed,
        use_gui=args.gui,
        episode_seconds=args.seconds,
    )
    try:
        if args.check_env:
            from gymnasium.utils.env_checker import check_env as gymnasium_check_env
            from stable_baselines3.common.env_checker import check_env

            gymnasium_check_env(env, skip_render_check=True)
            check_env(env, warn=True)
            print("Gymnasium and Stable-Baselines3 check_env: PASS")
        observation, _ = env.reset(seed=args.seed)
        assert env.action_space.n == 8
        assert observation.shape == (60,) and env.observation_space.contains(observation)
        requested_actions: set[int] = set()
        green_phases = {env.signal_controller.current_phase}
        stages = {env.signal_controller.stage}
        # Leave enough time for clearance and minimum green in every phase.
        interval = config.signal.decision_interval
        hold_steps = math.ceil(
            (config.signal.min_green + config.signal.yellow + config.signal.all_red) / interval
        ) + 1
        for step in range(math.ceil(args.seconds / interval)):
            action = (step // hold_steps) % PHASE_COUNT
            requested_actions.add(action)
            observation, reward, terminated, truncated, info = env.step(action)
            assert observation.shape == (60,) and env.observation_space.contains(observation)
            assert np.isfinite(reward)
            controller = env.signal_controller
            stages.add(controller.stage)
            if controller.stage == "green":
                green_phases.add(controller.current_phase)
            expected = ["r"] * LINK_COUNT
            if controller.stage != "all_red":
                for index in GREEN_LINKS[controller.current_phase]:
                    expected[index] = "G" if controller.stage == "green" else "y"
            assert env.connection.trafficlight.getRedYellowGreenState(config.tl_id) == "".join(expected)
            if step % 5 == 0 or info["phase_changed"]:
                snapshot = env._previous_snapshot
                assert snapshot is not None
                queue_text = " ".join(
                    f"{approach}_{group}={snapshot.queue_by_group[index]:.0f}"
                    for index, (approach, group) in enumerate(LANE_GROUPS)
                )
                print(
                    f"t={env.connection.simulation.getTime():05.1f}s phase={info['phase']} "
                    f"{queue_text} total_queue={snapshot.total_queue:.0f} "
                    f"waiting={snapshot.total_waiting_time:.1f}s "
                    f"max_wait={snapshot.max_waiting_time:.1f}s "
                    f"reward={reward:.3f} changed={info['phase_changed']}"
                )
            if terminated or truncated:
                break
        if args.seconds >= PHASE_COUNT * hold_steps * interval:
            assert requested_actions == green_phases == set(range(PHASE_COUNT))
            # A one-second yellow can finish within a single env.step().
            # All source/target yellow transitions are also covered by unit tests.
            assert {"green", "all_red"} <= stages
        print(
            f"Actions requested={sorted(requested_actions)}, green phases={sorted(green_phases)}, "
            f"observation={observation.shape}, stages={sorted(stages)}"
        )
    finally:
        env.close()
    print("TraCI SUMO smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
