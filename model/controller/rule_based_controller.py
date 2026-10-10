"""Demand- and delay-aware actuated policy using exactly the DQN observation.

This is a deterministic scheduling heuristic, not a learned model or a SUMO
lookahead. Policy coefficients are separate from the unchanged reward weights.
All requested actions still pass through the environment's SignalController.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from itertools import product
import math
from typing import Any, Mapping

import numpy as np

from model.env.state_provider import OBSERVATION_SIZE
from model.utils.config import ProjectConfig


# N_left, N_straight, S_left, S_straight, E_left, E_straight,
# W_left, W_straight. Straight groups include the right-turn shared lane.
PHASE_GROUPS = ((1, 3), (0, 2), (0, 1), (2, 3),
                (5, 7), (4, 6), (4, 5), (6, 7))
GROUP_LANES = np.asarray([1, 2] * 4, dtype=np.float64)


@dataclass(frozen=True)
class RulePolicyConfig:
    """Transparent heuristic settings; choose them using tuning traffic only.

    service_horizon=0 disables the approximate discharge-capacity adjustment.
    Otherwise horizon/headway are seconds and the margins are pressure units.
    Neither mode predicts individual trajectories or reads future arrivals.
    """

    waiting_weight: float = 4.0
    max_waiting_weight: float = 1.0
    approaching_weight: float = 0.5
    red_age_weight: float = 0.1
    service_horizon: float = 8.0
    discharge_headway: float = 2.0
    switch_margin: float = 0.1
    relative_margin: float = 0.1

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and nonnegative")
        if self.discharge_headway <= 0:
            raise ValueError("discharge_headway must be positive")

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> RulePolicyConfig:
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown rule settings: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in values.items()})


def candidate_policies() -> dict[str, RulePolicyConfig]:
    """Predeclared search including simpler policies, not just elaborate ones."""
    profiles = {
        "demand": (0.0, 0.0, 0.0),
        "balanced": (4.0, 1.0, 0.1),
        "delay": (12.0, 2.0, 0.25),
    }
    candidates: dict[str, RulePolicyConfig] = {}
    for (name, weights), horizon, approaching, margin in product(
        profiles.items(), (0.0, 8.0), (0.25, 0.75), (0.02, 0.1, 0.25)
    ):
        key = f"{name}_h{horizon:g}_a{approaching:g}_m{margin:g}"
        candidates[key] = RulePolicyConfig(
            waiting_weight=weights[0], max_waiting_weight=weights[1],
            red_age_weight=weights[2], service_horizon=horizon,
            approaching_weight=approaching, switch_margin=margin,
        )
    candidates["queue_only"] = replace(
        RulePolicyConfig(), waiting_weight=0, max_waiting_weight=0,
        red_age_weight=0, approaching_weight=0, service_horizon=0,
    )
    return candidates


class RuleBasedController:
    """Stateless observation-to-action policy, usable with SUMO or a camera.

    Queue, total/max accumulated waiting and approaching demand determine
    movement urgency. A bounded phase-age bonus favors neglected phases with
    observed demand. Capacity-aware variants discount service that cannot fit
    within a short window, including the lost yellow/all-red time. Hysteresis
    avoids changing phase for a small improvement. The shared signal controller
    is the authority for min/max green, max-red and all signal transitions.
    """

    def __init__(self, config: ProjectConfig | None = None,
                 policy: RulePolicyConfig | None = None) -> None:
        self.config = config or ProjectConfig()
        self.policy = policy or RulePolicyConfig()
        if (0 < self.policy.service_horizon
                <= self.config.signal.yellow + self.config.signal.all_red):
            raise ValueError("service_horizon must exceed yellow + all_red, or be zero")

    @staticmethod
    def _observation(observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"Expected observation shape ({OBSERVATION_SIZE},), got {obs.shape}")
        if not np.isfinite(obs).all() or np.any(obs < 0) or np.any(obs > 1):
            raise ValueError("Observation must contain finite values in [0, 1]")
        for name, values in (("phase", obs[24:32]), ("stage", obs[33:36])):
            if not np.all((values == 0) | (values == 1)) or values.sum() != 1:
                raise ValueError(f"{name} must be one-hot")
        target = obs[36:44]
        if not np.all((target == 0) | (target == 1)):
            raise ValueError("target must be binary")
        if target.sum() != (0 if obs[33] == 1 else 1):
            raise ValueError("target must be zero in green and one-hot during transition")
        return obs

    def explain(self, observation: np.ndarray) -> dict[str, Any]:
        """Return the action plus its scores/reason for reproducible diagnostics."""
        obs = self._observation(observation)
        current = int(np.argmax(obs[24:32]))
        stage = int(np.argmax(obs[33:36]))
        p, signal = self.policy, self.config.signal
        elapsed = float(obs[32]) * (signal.max_green, signal.yellow, signal.all_red)[stage]
        urgency = (obs[0:8] + p.waiting_weight * obs[8:16]
                   + p.max_waiting_weight * obs[16:24]
                   + p.approaching_weight * obs[52:60])
        # Waiting remains informative for moving/far incoming vehicles. Never
        # call a group empty solely because queue and approaching are both zero.
        present = ((obs[0:8] + obs[8:16] + obs[16:24] + obs[52:60]) > 0)
        scores = np.zeros(8, dtype=np.float64)
        for phase, groups in enumerate(PHASE_GROUPS):
            indices = list(groups)
            benefit = urgency[indices].copy()
            if p.service_horizon > 0:
                green_window = (
                    min(p.service_horizon, max(0.0, signal.max_green - elapsed))
                    if phase == current else
                    min(signal.max_green, p.service_horizon - signal.yellow - signal.all_red)
                )
                demand = (obs[0:8] * self.config.reward.queue_scale
                          + p.approaching_weight * obs[52:60] * self.config.approaching_scale)
                # A positive waiting history proves at least one incoming car
                # is present, even when its position/speed is not observable.
                demand = np.maximum(demand, present.astype(np.float64))
                capacity = GROUP_LANES[indices] * green_window / p.discharge_headway
                fraction = np.minimum(1.0, capacity / np.maximum(demand[indices], 1e-9))
                benefit *= fraction
            scores[phase] = float(benefit.sum())
            if present[indices].any():
                scores[phase] += p.red_age_weight * float(obs[44 + phase]) ** 2

        others = [phase for phase in range(8) if phase != current]
        # Age and stable phase id break equal competitor scores, not RNG or
        # hidden unclipped controller ages. Current phase wins marginal ties.
        best = max(others, key=lambda phase: (scores[phase], obs[44 + phase], -phase))
        threshold = (1 + p.relative_margin) * scores[current] + p.switch_margin
        if stage != 0:
            action, reason = int(np.argmax(obs[36:44])), "transition"
        elif elapsed + 1e-6 < signal.min_green:
            action, reason = current, "minimum_green"
        elif elapsed + signal.decision_interval + 1e-6 >= signal.max_green:
            # advance() would otherwise choose the oldest phase automatically
            # at max-green. Request a useful next phase while choice is allowed.
            action, reason = best, "maximum_green_soon"
        elif not present[list(PHASE_GROUPS[current])].any() and scores[best] > 0:
            action, reason = best, "empty_current"
        elif scores[best] > threshold + 1e-12:
            action, reason = best, "higher_priority"
        else:
            action, reason = current, "hold"
        return {"action": action, "reason": reason, "current_phase": current,
                "elapsed": elapsed, "scores": scores.tolist(),
                "movement_urgency": urgency.tolist(), "switch_threshold": float(threshold)}

    def action(self, observation: np.ndarray) -> int:
        return int(self.explain(observation)["action"])
