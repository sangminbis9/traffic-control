from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from model.controller.fixed_controller import FixedTimeController
from model.controller.signal_controller import (
    CONTROLLER_FEATURE_COUNT,
    PHASE_COUNT,
    PHASE_NAMES,
    SignalController,
)
from model.utils.config import ProjectConfig, SignalConfig


EXPECTED_GREEN_STATES = (
    "GGGrrrrrGGGrrrrr",  # North/south through and right turns.
    "rrrGrrrrrrrGrrrr",  # North/south protected left turns.
    "GGGGrrrrrrrrrrrr",  # Only north has green.
    "rrrrrrrrGGGGrrrr",  # Only south has green.
    "rrrrGGGrrrrrGGGr",  # East/west through and right turns.
    "rrrrrrrGrrrrrrrG",  # East/west protected left turns.
    "rrrrGGGGrrrrrrrr",  # Only east has green.
    "rrrrrrrrrrrrGGGG",  # Only west has green.
)


class FakeTrafficLight:
    def __init__(self) -> None:
        self.states: list[str] = []

    def setRedYellowGreenState(self, _tl_id: str, state: str) -> None:
        self.states.append(state)


class FakeConnection:
    def __init__(self) -> None:
        self.trafficlight = FakeTrafficLight()


class SignalControllerTests(unittest.TestCase):
    def make_controller(self, config: SignalConfig | None = None) -> SignalController:
        return SignalController(FakeConnection(), "J", config or SignalConfig())

    def test_each_action_has_the_expected_protected_green_state(self) -> None:
        self.assertEqual(PHASE_COUNT, 8)
        self.assertEqual(len(PHASE_NAMES), 8)
        controller = self.make_controller()
        for phase, expected in enumerate(EXPECTED_GREEN_STATES):
            with self.subTest(phase=phase):
                controller.reset(phase)
                self.assertEqual(controller.current_phase, phase)
                self.assertEqual(controller.connection.trafficlight.states[-1], expected)

    def test_all_distinct_phase_changes_clear_old_links_before_new_green(self) -> None:
        for current in range(8):
            for target in range(8):
                if target == current:
                    continue
                with self.subTest(current=current, target=target):
                    controller = self.make_controller()
                    controller.reset(current)
                    controller.advance(3.0)
                    result = controller.apply_action(target)
                    self.assertTrue(result.switched)
                    self.assertEqual(result.applied_action, target)
                    self.assertEqual(controller.target_phase, target)
                    self.assertEqual(
                        controller.connection.trafficlight.states[-1],
                        EXPECTED_GREEN_STATES[current].replace("G", "y"),
                    )
                    self.assertTrue(controller.apply_action((target + 1) % 8).ignored)
                    controller.advance(0.5)
                    self.assertEqual(controller.stage, "yellow")
                    controller.advance(0.5)
                    self.assertEqual(controller.stage, "all_red")
                    self.assertEqual(controller.connection.trafficlight.states[-1], "r" * 16)
                    self.assertTrue(controller.apply_action((target + 1) % 8).ignored)
                    controller.advance(0.5)
                    self.assertEqual(controller.stage, "all_red")
                    controller.advance(0.5)
                    self.assertEqual(controller.stage, "green")
                    self.assertEqual(controller.current_phase, target)
                    self.assertEqual(controller.phase_elapsed, 0.0)
                    self.assertEqual(controller.red_age[target], 0.0)
                    self.assertEqual(
                        controller.connection.trafficlight.states[-1],
                        EXPECTED_GREEN_STATES[target],
                    )

    def test_minimum_green_holds_all_eight_phases(self) -> None:
        controller = self.make_controller()
        for phase in range(8):
            with self.subTest(phase=phase):
                controller.reset(phase)
                controller.advance(2.5)
                self.assertTrue(controller.apply_action((phase + 1) % 8).ignored)
                self.assertEqual(controller.current_phase, phase)
                controller.advance(0.5)
                self.assertTrue(controller.apply_action((phase + 1) % 8).switched)

    def test_maximum_green_cannot_be_held_by_repeated_same_action(self) -> None:
        for phase in range(8):
            with self.subTest(phase=phase):
                controller = self.make_controller()
                controller.reset(phase)
                controller.advance(14.5)
                result = controller.apply_action(phase)
                self.assertFalse(result.switched)
                controller.advance(0.5)
                self.assertTrue(controller.in_transition)
                self.assertNotEqual(controller.target_phase, phase)
                self.assertEqual(controller.last_reason, "max_green")
                self.assertEqual(controller.forced_changes, 1)

    def test_max_red_can_force_phase_seven_after_minimum_green(self) -> None:
        controller = self.make_controller(SignalConfig(max_green=100.0, max_red=10.0))
        controller.reset(0)
        controller.red_age[7] = 10.0
        controller.advance(2.5)
        self.assertFalse(controller.in_transition)
        self.assertTrue(controller.apply_action(1).ignored)
        controller.advance(0.5)
        self.assertEqual(controller.target_phase, 7)
        self.assertEqual(controller.last_reason, "max_red")
        self.assertEqual(controller.forced_changes, 1)

    def test_action_at_max_red_overrides_policy_with_oldest_of_eight(self) -> None:
        controller = self.make_controller()
        controller.reset(6)
        controller.advance(3.0)
        controller.red_age[7] = controller.config.max_red
        result = controller.apply_action(4)
        self.assertTrue(result.forced)
        self.assertEqual(result.requested_action, 4)
        self.assertEqual(result.applied_action, 7)
        self.assertEqual(controller.last_reason, "max_red")

    def test_controller_feature_layout_covers_phase_target_and_red_age(self) -> None:
        controller = self.make_controller()
        controller.reset(7)
        controller.advance(3.0)
        feature = controller.features()
        self.assertEqual(CONTROLLER_FEATURE_COUNT, 28)
        self.assertEqual(feature.shape, (28,))
        self.assertEqual(feature.dtype, np.float32)
        np.testing.assert_array_equal(feature[:8], [0, 0, 0, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(feature[8], 3.0 / 15.0)
        np.testing.assert_array_equal(feature[9:12], [1, 0, 0])
        np.testing.assert_array_equal(feature[12:20], np.zeros(8))
        np.testing.assert_allclose(feature[20:28], [0.05] * 7 + [0.0])

        controller.apply_action(6)
        controller.advance(0.5)
        feature = controller.features()
        np.testing.assert_array_equal(feature[:8], [0, 0, 0, 0, 0, 0, 0, 1])
        self.assertAlmostEqual(feature[8], 0.5)
        np.testing.assert_array_equal(feature[9:12], [0, 1, 0])
        np.testing.assert_array_equal(feature[12:20], [0, 0, 0, 0, 0, 0, 1, 0])
        self.assertGreater(feature[27], 0.0)
        controller.advance(0.5)
        np.testing.assert_array_equal(controller.features()[9:12], [0, 0, 1])
        controller.advance(1.0)
        np.testing.assert_array_equal(controller.features()[12:20], np.zeros(8))
        controller.red_age[:] = controller.config.max_red * 2.0
        np.testing.assert_array_equal(controller.features()[20:28], np.ones(8))

    def test_invalid_action_and_initial_phase_are_rejected(self) -> None:
        controller = self.make_controller()
        for invalid in (-1, 8):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    controller.reset(invalid)
                with self.assertRaises(ValueError):
                    controller.apply_action(invalid)

    def test_fixed_controller_cycles_all_eight_phases_and_wraps(self) -> None:
        config = ProjectConfig()
        controller = self.make_controller(config.signal)
        fixed = FixedTimeController(controller, config.fixed_green_times)
        fixed.reset()
        for phase in range(8):
            with self.subTest(phase=phase):
                self.assertEqual(controller.current_phase, phase)
                controller.advance(config.fixed_green_times[phase] - 0.5)
                self.assertFalse(fixed.apply_if_due().switched)
                self.assertEqual(fixed.action(), phase)
                controller.advance(0.5)
                self.assertEqual(fixed.action(), (phase + 1) % 8)
                self.assertTrue(fixed.apply_if_due().switched)
                controller.advance(config.signal.yellow)
                controller.advance(config.signal.all_red)
                self.assertEqual(controller.current_phase, (phase + 1) % 8)
        self.assertEqual(controller.phase_changes, 8)
        self.assertEqual(controller.forced_changes, 0)

    def test_fixed_controller_requires_every_phase_duration(self) -> None:
        with self.assertRaises(ValueError):
            FixedTimeController(self.make_controller(), {0: 10.0, 1: 4.0, 2: 10.0, 3: 4.0})

    def test_static_network_and_source_program_match_runtime_states(self) -> None:
        config = ProjectConfig()
        for path in (config.network_file, config.sumo_dir / "traffic_lights.add.xml"):
            with self.subTest(path=path.name):
                program = ET.parse(path).getroot().find("tlLogic[@id='J']")
                self.assertIsNotNone(program)
                phases = program.findall("phase")
                self.assertEqual(len(phases), 24)
                for phase, expected in enumerate(EXPECTED_GREEN_STATES):
                    green, yellow, all_red = phases[phase * 3:phase * 3 + 3]
                    self.assertEqual(green.attrib["state"], expected)
                    self.assertEqual(yellow.attrib["state"], expected.replace("G", "y"))
                    self.assertEqual(all_red.attrib["state"], "r" * 16)
                    self.assertEqual(float(green.attrib["duration"]), config.fixed_green_times[phase])
                    self.assertEqual(float(yellow.attrib["duration"]), config.signal.yellow)
                    self.assertEqual(float(all_red.attrib["duration"]), config.signal.all_red)


if __name__ == "__main__":
    unittest.main()
