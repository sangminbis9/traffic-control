from traffic.state_provider import TrafficState
from utils.config import NormalizationConfig, RewardConfig


def reward_terms(state: TrafficState, scales: NormalizationConfig, weights: RewardConfig) -> dict[str, float]:
    normalized = state.normalized(scales)
    return {"queue": -weights.queue * float(normalized[:8].mean()),
            "waiting": -weights.waiting * float(normalized[8]),
            "max_waiting": -weights.max_waiting * float(normalized[9])}


def switching_penalty(changes: int, weights: RewardConfig) -> float:
    return -weights.switching * changes
