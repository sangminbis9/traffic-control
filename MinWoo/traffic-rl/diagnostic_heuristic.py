"""Queue-only heuristic as a diagnostic upper reference, never labeled DQN."""
from pathlib import Path

import numpy as np

from experiment_evaluate import compare, evaluate_cases
from utils.config import load_config


class QueueHeuristic:
    controller_name = "queue_heuristic"

    def predict(self, observation, deterministic=True):
        q = observation[:8] * 40
        demand = np.array([q[1] + q[3], q[0] + q[2], q[5] + q[7], q[4] + q[6]])
        phase = int(np.argmax(observation[10:14]))
        elapsed = observation[14] * 15
        if observation[15] < 0.5 or elapsed < (6 if phase in (0, 2) else 3):
            return phase, None
        target = int(np.argmax(demand))
        if target != phase and (demand[phase] < 0.5 or demand[target] > demand[phase] * 1.5 + 3):
            return target, None
        return phase, None


if __name__ == "__main__":
    import pandas as pd
    config = load_config()
    result = evaluate_cases(config, Path("results/diagnostics/queue_heuristic"), list(range(3001, 3006)), QueueHeuristic())
    print(compare(pd.read_csv("results/selection_default/fixed/episodes.csv"), result))
