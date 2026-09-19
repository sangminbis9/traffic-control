"""Fixed-time baseline built on the same safety-aware signal controller."""

from __future__ import annotations

from typing import Any, Mapping

from .signal_controller import ActionResult, SignalController


class FixedTimeController:
    """Cycle through phases using configurable green durations."""

    def __init__(self, signal_controller: SignalController, green_times: Mapping[int, float]) -> None:
        self.signal_controller = signal_controller
        self.green_times = {int(key): float(value) for key, value in green_times.items()}
        if set(self.green_times) != {0, 1, 2, 3}:
            raise ValueError("green_times must contain phase durations for actions 0, 1, 2, and 3")

    def reset(self) -> None:
        self.signal_controller.reset(0)

    def action(self) -> int:
        current = self.signal_controller.current_phase
        if self.signal_controller.in_transition:
            return current
        if self.signal_controller.phase_elapsed >= self.green_times[current]:
            return (current + 1) % 4
        return current

    def apply_if_due(self) -> ActionResult:
        current = self.signal_controller.current_phase
        if self.signal_controller.in_transition:
            return ActionResult(current, current, False, True)
        if self.signal_controller.phase_elapsed >= self.green_times[current]:
            return self.signal_controller.apply_action((current + 1) % 4, force=True)
        return ActionResult(current, current, False, False)

