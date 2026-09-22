"""Safe, time-based signal state machine; independent of SUMO and Gymnasium."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from controller.phases import ALL_RED, GREEN_STATES, YELLOW_STATES
from utils.config import SignalConfig


class SignalSink(Protocol):
    def set_signal(self, state: str) -> None: ...


class SignalController:
    def __init__(self, sink: SignalSink, config: SignalConfig) -> None:
        self.sink, self.config = sink, config
        self.phase = 0
        self.target = 0
        self.stage = "green"
        self.elapsed = 0.0
        self.red_age = np.zeros(4, dtype=np.float64)
        self.phase_changes = 0
        self.forced_changes = 0
        self.last_reason = "initial"
        self.sink.set_signal(GREEN_STATES[0])

    def request(self, action: int) -> bool:
        if action not in range(4):
            raise ValueError("Action must be in [0, 3]")
        if self.stage != "green" or self.elapsed + 1e-8 < self.config.minimum_green:
            return False
        oldest = self._oldest_other()
        if self.red_age[oldest] >= self.config.max_red:
            self._switch(oldest, "max_red")
        elif self.elapsed + 1e-8 >= self.config.maximum_green:
            self._switch(action if action != self.phase else oldest, "max_green")
        elif action != self.phase:
            self._switch(action, "policy")
        else:
            return False
        return True

    def _oldest_other(self) -> int:
        return max((p for p in range(4) if p != self.phase), key=lambda p: (self.red_age[p], -p))

    def _switch(self, target: int, reason: str) -> None:
        self.target = target
        self.stage = "yellow"
        self.elapsed = 0.0
        self.phase_changes += 1
        self.forced_changes += int(reason != "policy")
        self.last_reason = reason
        self.sink.set_signal(YELLOW_STATES[self.phase])

    def tick(self, dt: float) -> None:
        """Call AFTER the simulator advances dt under the previously set signal."""
        self.red_age += dt
        if self.stage == "green":
            self.red_age[self.phase] = 0
        self.elapsed += dt
        if self.stage == "yellow" and self.elapsed + 1e-8 >= self.config.yellow:
            self.stage, self.elapsed = "all_red", 0.0
            self.sink.set_signal(ALL_RED)
        elif self.stage == "all_red" and self.elapsed + 1e-8 >= self.config.all_red:
            self.phase = self.target
            self.stage, self.elapsed = "green", 0.0
            self.red_age[self.phase] = 0
            self.sink.set_signal(GREEN_STATES[self.phase])
        elif self.stage == "green":
            # Safety bounds hold at simulation resolution, even between decisions.
            if self.elapsed + 1e-8 >= self.config.maximum_green:
                self._switch(self._oldest_other(), "max_green")
            elif self.elapsed + 1e-8 >= self.config.minimum_green and self.red_age[self._oldest_other()] >= self.config.max_red:
                self._switch(self._oldest_other(), "max_red")

    def features(self) -> list[float]:
        """16 controller features: current phase, elapsed, stage, target, red ages."""
        phase = [float(self.phase == p) for p in range(4)]
        stages = [float(self.stage == s) for s in ("green", "yellow", "all_red")]
        target = [float(self.target == p and self.stage != "green") for p in range(4)]
        scale = {"green": self.config.maximum_green, "yellow": self.config.yellow, "all_red": self.config.all_red}[self.stage]
        return phase + [min(self.elapsed / scale, 1)] + stages + target + np.clip(self.red_age / self.config.max_red, 0, 1).tolist()
