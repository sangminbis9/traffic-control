from controller.signal_controller import SignalController


class FixedTimeController:
    def __init__(self, green_durations: list[float]) -> None:
        self.green_durations = green_durations

    def action(self, signal: SignalController) -> int:
        if signal.stage == "green" and signal.elapsed + 1e-8 >= self.green_durations[signal.phase]:
            return (signal.phase + 1) % 4
        return signal.phase
