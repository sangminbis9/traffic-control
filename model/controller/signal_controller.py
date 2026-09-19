"""Safety-aware logical phase controller used by both RL and baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GREEN_LINKS: dict[int, tuple[int, ...]] = {
    # The order is SUMO's controlled-link order for the checked-in network.
    0: (1, 2, 3, 9, 10, 11),
    1: (0, 8),
    2: (5, 6, 7, 13, 14, 15),
    3: (4, 12),
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
    """Translate four RL actions into green/yellow/all-red signal states."""

    def __init__(self, connection: Any, tl_id: str, config: Any) -> None:
        self.connection = connection
        self.tl_id = tl_id
        self.config = config
        self.current_phase = 0
        self.phase_elapsed = 0.0
        self._transition_stage: str | None = None
        self._transition_remaining = 0.0
        self._pending_phase: int | None = None
        self.phase_changes = 0

    @property
    def in_transition(self) -> bool:
        return self._transition_stage is not None

    def reset(self, initial_phase: int = 0) -> None:
        self.current_phase = int(initial_phase)
        self.phase_elapsed = 0.0
        self._transition_stage = None
        self._transition_remaining = 0.0
        self._pending_phase = None
        self.phase_changes = 0
        self._set_green(self.current_phase)

    def _state_for_links(self, links: tuple[int, ...], value: str) -> str:
        chars = ["r"] * LINK_COUNT
        for index in links:
            chars[index] = value
        return "".join(chars)

    def _set_green(self, phase: int) -> None:
        self.connection.trafficlight.setRedYellowGreenState(
            self.tl_id, self._state_for_links(GREEN_LINKS[phase], "g")
        )

    def _set_yellow(self) -> None:
        self.connection.trafficlight.setRedYellowGreenState(
            self.tl_id, self._state_for_links(GREEN_LINKS[self.current_phase], "y")
        )

    def _set_all_red(self) -> None:
        self.connection.trafficlight.setRedYellowGreenState(self.tl_id, "r" * LINK_COUNT)

    def apply_action(self, action: int, force: bool = False) -> ActionResult:
        requested = int(action)
        if requested not in GREEN_LINKS:
            raise ValueError(f"Action must be one of 0, 1, 2, 3; got {action}")
        if self.in_transition:
            return ActionResult(requested, self.current_phase, False, True)
        if not force and self.phase_elapsed < self.config.min_green:
            return ActionResult(requested, self.current_phase, False, True)
        if requested == self.current_phase:
            if not force and self.phase_elapsed < self.config.max_green:
                return ActionResult(requested, self.current_phase, False, False)
            # Never let a learned policy hold a phase forever at the safety limit.
            requested = (self.current_phase + 1) % 4
            force = True
        self._pending_phase = requested
        self._transition_stage = "yellow"
        self._transition_remaining = max(float(self.config.yellow), 0.0)
        self._set_yellow()
        self.phase_changes += 1
        return ActionResult(int(action), requested, True, False, force)

    def advance(self, seconds: float) -> None:
        """Advance controller timers after one or more SUMO simulation steps."""

        remaining = max(float(seconds), 0.0)
        while remaining > 1e-9:
            if self._transition_stage is None:
                self.phase_elapsed += remaining
                return
            consumed = min(remaining, self._transition_remaining)
            self._transition_remaining -= consumed
            remaining -= consumed
            if self._transition_remaining > 1e-9:
                continue
            if self._transition_stage == "yellow":
                self._transition_stage = "all_red"
                self._transition_remaining = max(float(self.config.all_red), 0.0)
                self._set_all_red()
                if self._transition_remaining <= 1e-9:
                    self._finish_transition()
            else:
                self._finish_transition()

    def _finish_transition(self) -> None:
        if self._pending_phase is None:
            raise RuntimeError("Signal transition finished without a pending phase")
        self.current_phase = self._pending_phase
        self.phase_elapsed = 0.0
        self._pending_phase = None
        self._transition_stage = None
        self._transition_remaining = 0.0
        self._set_green(self.current_phase)
