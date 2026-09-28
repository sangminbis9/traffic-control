"""Safe signal state machine compatible with MinWoo's trained policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Explicit link order: N 0-3, E 4-7, S 8-11, W 12-15.
# Each approach: right, straight lane 0, straight lane 1, left.
GREEN_LINKS: dict[int, tuple[int, ...]] = {
    0: (0, 1, 2, 8, 9, 10),
    1: (3, 11),
    2: (4, 5, 6, 12, 13, 14),
    3: (7, 15),
}
LINK_COUNT = 16


@dataclass(frozen=True)
class ActionResult:
    requested_action: int
    applied_action: int
    switched: bool
    ignored: bool
    forced: bool = False


class SignalController:
    """Translate four policy actions into green/yellow/all-red transitions."""

    def __init__(self, connection: Any, tl_id: str, config: Any) -> None:
        self.connection = connection
        self.tl_id = tl_id
        self.config = config
        self.current_phase = 0
        self.target_phase = 0
        self.phase_elapsed = 0.0
        self._transition_stage: str | None = None
        self.red_age = np.zeros(4, dtype=np.float64)
        self.phase_changes = 0
        self.forced_changes = 0
        self.last_reason = "initial"

    @property
    def in_transition(self) -> bool:
        return self._transition_stage is not None

    @property
    def signal_state(self) -> str:
        return self._transition_stage.upper() if self._transition_stage else "GREEN"

    @property
    def stage(self) -> str:
        return self._transition_stage or "green"

    def reset(self, initial_phase: int = 0) -> None:
        self.current_phase = int(initial_phase)
        self.target_phase = int(initial_phase)
        self.phase_elapsed = 0.0
        self._transition_stage = None
        self.red_age = np.zeros(4, dtype=np.float64)
        self.phase_changes = 0
        self.forced_changes = 0
        self.last_reason = "initial"
        self._set_green(self.current_phase)

    def _state_for_links(self, links: tuple[int, ...], value: str) -> str:
        chars = ["r"] * LINK_COUNT
        for index in links:
            chars[index] = value
        return "".join(chars)

    def _set_green(self, phase: int) -> None:
        self.connection.trafficlight.setRedYellowGreenState(
            self.tl_id, self._state_for_links(GREEN_LINKS[phase], "G")
        )

    def _set_yellow(self) -> None:
        self.connection.trafficlight.setRedYellowGreenState(
            self.tl_id, self._state_for_links(GREEN_LINKS[self.current_phase], "y")
        )

    def _set_all_red(self) -> None:
        self.connection.trafficlight.setRedYellowGreenState(self.tl_id, "r" * LINK_COUNT)

    def _oldest_other(self) -> int:
        return max(
            (phase for phase in range(4) if phase != self.current_phase),
            key=lambda phase: (self.red_age[phase], -phase),
        )

    def _switch(self, target: int, reason: str) -> None:
        self.target_phase = int(target)
        self._transition_stage = "yellow"
        self.phase_elapsed = 0.0
        self.phase_changes += 1
        self.forced_changes += int(reason not in {"policy", "fixed"})
        self.last_reason = reason
        self._set_yellow()

    def apply_action(self, action: int, force: bool = False) -> ActionResult:
        requested = int(action)
        if requested not in GREEN_LINKS:
            raise ValueError(f"Action must be one of 0, 1, 2, 3; got {action}")
        if self.in_transition:
            return ActionResult(requested, self.current_phase, False, True)
        if not force and self.phase_elapsed + 1e-8 < self.config.min_green:
            return ActionResult(requested, self.current_phase, False, True)

        reason = "fixed" if force else "policy"
        target = requested
        oldest = self._oldest_other()
        if not force and self.red_age[oldest] >= self.config.max_red:
            target, reason = oldest, "max_red"
        elif not force and self.phase_elapsed + 1e-8 >= self.config.max_green:
            target, reason = (requested if requested != self.current_phase else oldest), "max_green"
        elif requested == self.current_phase:
            return ActionResult(requested, self.current_phase, False, False)

        self._switch(target, reason)
        return ActionResult(requested, target, True, False, reason not in {"policy", "fixed"})

    def advance(self, seconds: float) -> None:
        """Advance timers after SUMO has moved under the previously set signal."""

        dt = max(float(seconds), 0.0)
        self.red_age += dt
        if not self.in_transition:
            self.red_age[self.current_phase] = 0.0
        self.phase_elapsed += dt

        if self._transition_stage == "yellow" and self.phase_elapsed + 1e-8 >= self.config.yellow:
            self._transition_stage = "all_red"
            self.phase_elapsed = 0.0
            self._set_all_red()
            return
        if self._transition_stage == "all_red" and self.phase_elapsed + 1e-8 >= self.config.all_red:
            self.current_phase = self.target_phase
            self._transition_stage = None
            self.phase_elapsed = 0.0
            self.red_age[self.current_phase] = 0.0
            self._set_green(self.current_phase)
            return
        if self._transition_stage is None:
            oldest = self._oldest_other()
            if self.phase_elapsed + 1e-8 >= self.config.max_green:
                self._switch(oldest, "max_green")
            elif (
                self.phase_elapsed + 1e-8 >= self.config.min_green
                and self.red_age[oldest] >= self.config.max_red
            ):
                self._switch(oldest, "max_red")

    def features(self) -> np.ndarray:
        """Return MinWoo's 16 normalized controller-state features."""

        phase = [float(self.current_phase == value) for value in range(4)]
        stages = [float(self.stage == value) for value in ("green", "yellow", "all_red")]
        target = [
            float(self.target_phase == value and self.stage != "green") for value in range(4)
        ]
        scale = {
            "green": self.config.max_green,
            "yellow": self.config.yellow,
            "all_red": self.config.all_red,
        }[self.stage]
        return np.asarray(
            phase
            + [min(self.phase_elapsed / max(scale, 1e-9), 1.0)]
            + stages
            + target
            + np.clip(self.red_age / self.config.max_red, 0.0, 1.0).tolist(),
            dtype=np.float32,
        )
