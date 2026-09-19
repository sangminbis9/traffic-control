"""Simple normalized reward model with explicit, tunable terms."""

from __future__ import annotations

from typing import Any

from .state_provider import TrafficSnapshot


def calculate_reward(previous: TrafficSnapshot, current: TrafficSnapshot, switched: bool, config: Any) -> float:
    """Penalize current congestion and discourage unnecessary switching.

    Absolute normalized penalties align the learning signal with the evaluation
    metrics. The previous snapshot remains part of the signature so alternative
    delta-based reward experiments can be added without changing the environment.
    """

    del previous
    queue_penalty = current.total_queue / max(config.queue_scale, 1.0)
    waiting_penalty = current.total_waiting_time / max(config.waiting_scale, 1.0)
    max_wait_penalty = current.max_waiting_time / max(config.max_waiting_scale, 1.0)
    switching_cost = 1.0 if switched else 0.0
    return float(
        -config.queue_weight * queue_penalty
        - config.waiting_weight * waiting_penalty
        - config.max_waiting_weight * max_wait_penalty
        - config.switch_penalty * switching_cost
    )
