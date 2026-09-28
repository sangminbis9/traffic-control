"""Simple normalized reward model with explicit, tunable terms."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state_provider import TrafficSnapshot


def reward_terms(current: TrafficSnapshot, config: Any) -> dict[str, float]:
    """Return MinWoo's bounded queue/waiting reward components."""

    queues = np.clip(
        np.asarray(current.queue_by_group, dtype=np.float64) / max(config.queue_scale, 1e-9),
        0.0,
        1.0,
    )
    return {
        "queue": -config.queue_weight * float(queues.mean()),
        "waiting": -config.waiting_weight
        * float(np.clip(current.total_waiting_time / max(config.waiting_scale, 1e-9), 0.0, 1.0)),
        "max_waiting": -config.max_waiting_weight
        * float(np.clip(current.max_waiting_time / max(config.max_waiting_scale, 1e-9), 0.0, 1.0)),
    }


def switching_penalty(changes: int, config: Any) -> float:
    return -config.switch_penalty * int(changes)


def calculate_reward(
    previous: TrafficSnapshot,
    current: TrafficSnapshot,
    switched: bool,
    config: Any,
) -> float:
    """Compatibility wrapper for one-snapshot reward calculations."""

    del previous
    return float(sum(reward_terms(current, config).values()) + switching_penalty(switched, config))
