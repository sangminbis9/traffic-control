"""Pydantic contracts shared by REST and WebSocket payloads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ScenarioName = Literal[
    "uniform",
    "north_south_congested",
    "east_west_congested",
    "left_turn_congested",
    "heavy",
    "low",
    "random",
]


class TrainingCreate(BaseModel):
    mode: Literal["fixed_steps", "auto_convergence"] = "fixed_steps"
    total_steps: int = Field(20_000, ge=1)
    maximum_steps: int = Field(1_000_000, ge=1)
    minimum_steps: int = Field(100_000, ge=0)
    validation_interval: int = Field(10_000, ge=1)
    validation_episodes: int = Field(5, ge=1, le=100)
    no_improvement_patience: int = Field(8, ge=1)
    minimum_improvement: float = Field(0.05, ge=0)
    scenario: ScenarioName = "random"
    episode_seconds: int = Field(300, ge=10, le=86_400)
    seed: int = 1
    validation_seed_start: int = 10_001
    learning_starts: int | None = Field(None, ge=0)


class LanePlacementInput(BaseModel):
    left: int = Field(0, ge=0, le=5)
    straight: int = Field(0, ge=0, le=5)
    lane3_total: int = Field(0, ge=0, le=5)
    lane3_right: int = Field(0, ge=0, le=5)

    @model_validator(mode="after")
    def validate_lane3(self) -> "LanePlacementInput":
        if self.lane3_right > self.lane3_total:
            raise ValueError("lane3_right cannot exceed lane3_total")
        return self


class InitialTrafficInput(BaseModel):
    N: LanePlacementInput = Field(default_factory=LanePlacementInput)
    S: LanePlacementInput = Field(default_factory=LanePlacementInput)
    E: LanePlacementInput = Field(default_factory=LanePlacementInput)
    W: LanePlacementInput = Field(default_factory=LanePlacementInput)


class ContinuousTrafficInput(BaseModel):
    scenario: ScenarioName = "uniform"
    n_rate: float | None = Field(None, ge=0, le=1)
    s_rate: float | None = Field(None, ge=0, le=1)
    e_rate: float | None = Field(None, ge=0, le=1)
    w_rate: float | None = Field(None, ge=0, le=1)
    left_ratio: float = Field(0.2, ge=0, le=1)
    straight_ratio: float = Field(0.6, ge=0, le=1)
    right_ratio: float = Field(0.2, ge=0, le=1)

    @model_validator(mode="after")
    def validate_ratios_and_rates(self) -> "ContinuousTrafficInput":
        if abs(self.left_ratio + self.straight_ratio + self.right_ratio - 1.0) > 1e-6:
            raise ValueError("turn ratios must sum to 1")
        rates = (self.n_rate, self.s_rate, self.e_rate, self.w_rate)
        if any(rate is not None for rate in rates) and not all(rate is not None for rate in rates):
            raise ValueError("custom mode requires all four approach rates")
        return self


class ComparisonCreate(BaseModel):
    test_mode: Literal["initial", "continuous"] = "initial"
    run_mode: Literal["single", "batch"] = "single"
    initial: InitialTrafficInput = Field(default_factory=InitialTrafficInput)
    continuous: ContinuousTrafficInput = Field(default_factory=ContinuousTrafficInput)
    model_path: str | None = None
    seed: int = 20_001
    runs: int = Field(1, ge=1, le=100)
    duration: int = Field(300, ge=10, le=86_400)
    render_interval: int = Field(1, ge=1, le=30)
    speed: Literal["0.5", "1", "2", "4", "max"] = "1"


class SimulationSpeedUpdate(BaseModel):
    speed: Literal["0.5", "1", "2", "4", "max"]


class StatusResponse(BaseModel):
    id: str
    status: str
    detail: dict = Field(default_factory=dict)
