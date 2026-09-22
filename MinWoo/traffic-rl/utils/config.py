"""Validated configuration; paths are relative to the project, never the shell."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("N", "S", "E", "W")
SCENARIOS = ("balanced", "ns_heavy", "ew_heavy", "left_heavy", "heavy", "low", "random")


@dataclass
class SimulationConfig:
    step_length: float = 0.5
    episode_seconds: float = 300
    demand_seconds: float = 240
    decision_interval: float = 1


@dataclass
class SignalConfig:
    minimum_green: float = 3
    maximum_green: float = 15
    yellow: float = 1
    all_red: float = 1
    max_red: float = 60
    fixed_green: list[float] = field(default_factory=lambda: [10, 4, 10, 4])


@dataclass
class NormalizationConfig:
    queue_per_group: float = 40
    total_waiting: float = 6000
    max_waiting: float = 120


@dataclass
class RewardConfig:
    queue: float = 1
    waiting: float = 0.3
    max_waiting: float = 0.5
    switching: float = 0.2


@dataclass
class TrafficConfig:
    scenario: str = "random"
    left_direction: str = "N"
    turn_ratios: list[float] = field(default_factory=lambda: [0.25, 0.6, 0.15])
    random_rate_range: list[float] = field(default_factory=lambda: [150, 1500])
    rates: dict[str, float] | None = None


@dataclass
class TrainingConfig:
    seed: int = 42
    traffic_seed_min: int = 1
    traffic_seed_max: int = 1000
    total_timesteps: int = 100000
    learning_rate: float = 1e-4
    buffer_size: int = 100000
    learning_starts: int = 5000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    n_steps: int = 1
    exploration_hold_min: int = 1
    exploration_hold_max: int = 1
    target_update_interval: int = 1000
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05
    net_arch: list[int] = field(default_factory=lambda: [128, 128])
    torch_threads: int = 1


@dataclass
class ObservationConfig:
    include_approaching: bool = False
    detection_distance: float = 50.0
    count_scale: float = 10.0


@dataclass
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    def validate(self) -> None:
        s, t = self.simulation, self.signal
        if not 0 < s.step_length <= s.decision_interval:
            raise ValueError("step_length must be positive and <= decision_interval")
        for name, value in {"decision_interval": s.decision_interval, "episode_seconds": s.episode_seconds,
                            "minimum_green": t.minimum_green, "maximum_green": t.maximum_green,
                            "yellow": t.yellow, "all_red": t.all_red, "max_red": t.max_red}.items():
            if not math.isfinite(value) or value <= 0 or not math.isclose(value / s.step_length, round(value / s.step_length)):
                raise ValueError(f"{name} must be positive and a multiple of step_length")
        if not 0 < s.demand_seconds <= s.episode_seconds:
            raise ValueError("0 < demand_seconds <= episode_seconds required")
        if t.minimum_green > t.maximum_green or t.max_red <= t.maximum_green:
            raise ValueError("minimum_green <= maximum_green < max_red required")
        if len(t.fixed_green) != 4 or any(not t.minimum_green <= x <= t.maximum_green or not math.isclose(x / s.step_length, round(x / s.step_length)) for x in t.fixed_green):
            raise ValueError("fixed_green requires four aligned durations within min/max green")
        if self.traffic.scenario not in SCENARIOS or self.traffic.left_direction not in DIRECTIONS:
            raise ValueError("Unknown scenario/direction")
        ratios = self.traffic.turn_ratios
        if len(ratios) != 3 or any(not math.isfinite(x) or x < 0 for x in ratios) or not math.isclose(sum(ratios), 1):
            raise ValueError("turn_ratios must be nonnegative [left, straight, right] summing to 1")
        bounds = self.traffic.random_rate_range
        if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] or not all(map(math.isfinite, bounds)):
            raise ValueError("Invalid random_rate_range")
        if self.traffic.rates is not None and (set(self.traffic.rates) != set(DIRECTIONS) or any(not math.isfinite(x) or x < 0 for x in self.traffic.rates.values())):
            raise ValueError("rates must contain nonnegative vehicles/hour for N,S,E,W")
        if any(not math.isfinite(x) or x <= 0 for x in asdict(self.normalization).values()):
            raise ValueError("Normalization scales must be positive")
        if any(not math.isfinite(x) or x < 0 for x in asdict(self.reward).values()):
            raise ValueError("Reward weights must be nonnegative")
        if not 0 <= self.training.traffic_seed_min <= self.training.traffic_seed_max:
            raise ValueError("Invalid training traffic seed range")
        if any(type(x) is not int for x in (self.training.exploration_hold_min, self.training.exploration_hold_max)) or not 1 <= self.training.exploration_hold_min <= self.training.exploration_hold_max:
            raise ValueError("Invalid exploration hold interval")
        if type(self.training.n_steps) is not int or self.training.n_steps < 1:
            raise ValueError("n_steps must be positive")
        if any(not math.isfinite(x) or x <= 0 for x in (self.observation.detection_distance, self.observation.count_scale)):
            raise ValueError("Observation distance and count scale must be positive")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_config(path: str | Path | None = None) -> Config:
    data = json.loads(Path(path or ROOT / "config.json").read_text(encoding="utf-8-sig"))
    classes = {"simulation": SimulationConfig, "signal": SignalConfig, "normalization": NormalizationConfig,
               "reward": RewardConfig, "traffic": TrafficConfig, "training": TrainingConfig, "observation": ObservationConfig}
    if set(data) - set(classes):
        raise ValueError(f"Unknown configuration sections: {set(data) - set(classes)}")
    config = Config(**{key: classes[key](**value) for key, value in data.items()})
    config.validate()
    return config
