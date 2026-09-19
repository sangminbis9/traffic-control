"""Smoke test TraCI, signal switching, queues, and waiting-time reporting."""

from __future__ import annotations

import argparse

try:
    from model.env.intersection_env import IntersectionEnv
    from model.env.state_provider import LANE_GROUPS
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
        observation, _ = env.reset(seed=args.seed)
        for second in range(args.seconds):
            # Deliberately request a new phase periodically to exercise yellow/all-red.
            action = (second // 10) % 4
            observation, reward, terminated, truncated, info = env.step(action)
            if second % 5 == 0 or info["phase_changed"]:
                snapshot = env._previous_snapshot
                assert snapshot is not None
                queue_text = " ".join(
                    f"{approach}_{group}={snapshot.queue_by_group[index]:.0f}"
                    for index, (approach, group) in enumerate(LANE_GROUPS)
                )
                print(
                    f"t={second + 1:03d}s phase={info['phase']} "
                    f"{queue_text} total_queue={snapshot.total_queue:.0f} "
                    f"waiting={snapshot.total_waiting_time:.1f}s "
                    f"max_wait={snapshot.max_waiting_time:.1f}s "
                    f"reward={reward:.3f} changed={info['phase_changed']}"
                )
            if terminated or truncated:
                break
    finally:
        env.close()
    print("TraCI SUMO smoke test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
