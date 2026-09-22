import numpy as np
import pytest

from controller.phases import ALL_RED, GREEN_STATES, YELLOW_STATES
from controller.signal_controller import SignalController
from controller.fixed_controller import FixedTimeController
from utils.config import SignalConfig


class Sink:
    def __init__(self):
        self.states = []

    def set_signal(self, state):
        self.states.append(state)


def advance(signal, duration):
    for _ in range(round(duration / 0.5)):
        signal.tick(0.5)


def test_minimum_same_action_and_clearance():
    sink = Sink()
    signal = SignalController(sink, SignalConfig())
    assert not signal.request(2)
    advance(signal, 2.5)
    assert not signal.request(2)
    advance(signal, 0.5)
    assert not signal.request(0)
    assert signal.elapsed == 3
    assert signal.request(2)
    assert sink.states[-1] == YELLOW_STATES[0]
    assert not signal.request(1)  # Pending target cannot be overwritten mid-transition.
    advance(signal, 0.5)
    assert signal.stage == "yellow"
    advance(signal, 0.5)
    assert sink.states[-1] == ALL_RED
    advance(signal, 1)
    assert signal.phase == 2 and signal.stage == "green" and signal.elapsed == 0
    assert sink.states == [GREEN_STATES[0], YELLOW_STATES[0], ALL_RED, GREEN_STATES[2]]
    assert signal.phase_changes == 1


def test_maximum_green_forces_switch():
    signal = SignalController(Sink(), SignalConfig())
    advance(signal, 15)
    assert signal.stage == "yellow" and signal.target != 0
    assert signal.forced_changes == 1


def test_persistent_biased_actions_cannot_starve_a_phase():
    signal = SignalController(Sink(), SignalConfig())
    last_served = np.zeros(4)
    longest = np.zeros(4)
    for tick in range(1200):
        signal.request(0 if signal.phase == 1 else 1)
        signal.tick(0.5)
        now = (tick + 1) * 0.5
        if signal.stage == "green":
            longest[signal.phase] = max(longest[signal.phase], now - last_served[signal.phase])
            last_served[signal.phase] = now
    assert (last_served > 500).all()
    assert longest.max() <= 75  # Threshold + clearance and servicing another overdue phase.
    assert signal.forced_changes > 0


def test_fixed_cycle_durations():
    signal = SignalController(Sink(), SignalConfig())
    fixed = FixedTimeController([10, 4, 10, 4])
    transitions = []
    for tick in range(72):
        if signal.request(fixed.action(signal)):
            transitions.append((tick * 0.5, signal.phase, signal.target))
        signal.tick(0.5)
    assert transitions == [(10, 0, 1), (16, 1, 2), (28, 2, 3), (34, 3, 0)]
    assert signal.phase == 0 and signal.stage == "green"


@pytest.mark.parametrize("red", [0.5, 1.0])
def test_fractional_all_red(red):
    signal = SignalController(Sink(), SignalConfig(all_red=red))
    advance(signal, 3)
    signal.request(3)
    advance(signal, 1 + red)
    assert signal.phase == 3 and signal.stage == "green"
