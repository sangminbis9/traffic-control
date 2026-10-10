"""Behavioral contracts for the observation-only actuated comparison policy."""

from __future__ import annotations

from dataclasses import asdict, replace
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from model.controller.rule_based_controller import (
    GROUP_LANES,
    PHASE_GROUPS,
    RuleBasedController,
    RulePolicyConfig,
    candidate_policies,
)
from model.controller.signal_controller import GREEN_LINKS, SignalController
from model.env.state_provider import (
    LANE_GROUPS,
    SUMOTrafficStateProvider,
    SnapshotStateProvider,
    TrafficSnapshot,
)
from model.tests.test_signal_controller import FakeConnection
from model.tests.test_state_provider import connection_for
from model.utils.config import ProjectConfig


def make_signal(phase: int = 0, elapsed: float = 5.0) -> SignalController:
    signal = SignalController(FakeConnection(), "J", ProjectConfig().signal)
    signal.reset(phase)
    signal.advance(elapsed)
    return signal


def observe(signal: SignalController, **values: object) -> np.ndarray:
    snapshot = TrafficSnapshot(
        queue_by_group=(0.0,) * 8,
        total_queue=0.0,
        total_waiting_time=0.0,
        max_waiting_time=0.0,
        vehicle_count=0,
        arrived=0,
    )
    return SnapshotStateProvider(replace(snapshot, **values)).observation(signal)


def test_phase_service_groups_match_controlled_network_lanes() -> None:
    network = ET.parse(ProjectConfig().network_file).getroot()
    connections = {
        int(link.attrib["linkIndex"]): link.attrib
        for link in network.findall("connection") if link.get("tl") == "J"
    }
    actual_lanes: dict[int, set[str]] = {group: set() for group in range(8)}
    for phase, links in GREEN_LINKS.items():
        served_groups: set[int] = set()
        for link in links:
            connection = connections[link]
            approach = connection["from"].removesuffix("_in")
            movement = "left" if connection["fromLane"] == "2" else "straight"
            group = LANE_GROUPS.index((approach, movement))
            served_groups.add(group)
            actual_lanes[group].add(connection["fromLane"])
        assert served_groups == set(PHASE_GROUPS[phase])
    assert [len(actual_lanes[group]) for group in range(8)] == GROUP_LANES.tolist()


@pytest.mark.parametrize(
    ("target", "current", "groups"),
    [(0, 4, (1, 3)), (1, 4, (0, 2)), (2, 6, (0, 1)), (3, 6, (2, 3)),
     (4, 0, (5, 7)), (5, 0, (4, 6)), (6, 2, (4, 5)), (7, 2, (6, 7))],
)
def test_every_phase_can_be_selected_and_reached_safely(
    target: int, current: int, groups: tuple[int, int]
) -> None:
    signal = make_signal(current)
    queue = [0.0] * 8
    for group in groups:
        queue[group] = 2.0
    action = RuleBasedController().action(observe(signal, queue_by_group=tuple(queue)))
    assert action == target
    old_green = signal.connection.trafficlight.states[-1]
    result = signal.apply_action(action)
    assert result.switched and not result.forced
    assert signal.connection.trafficlight.states[-1] == old_green.replace("G", "y")
    signal.advance(signal.config.yellow)
    assert signal.connection.trafficlight.states[-1] == "r" * 16
    signal.advance(signal.config.all_red)
    assert signal.current_phase == target
    assert {index for index, color in enumerate(signal.connection.trafficlight.states[-1])
            if color == "G"} == set(GREEN_LINKS[target])


@pytest.mark.parametrize("competing_queue", [4.0, 4.2])
def test_equal_or_slightly_higher_pressure_holds_current_phase(competing_queue: float) -> None:
    signal = make_signal()
    obs = observe(signal, queue_by_group=(0, 4, 0, 4, 0, competing_queue, 0, competing_queue))
    rule = RuleBasedController(policy=replace(RulePolicyConfig(), red_age_weight=0))
    assert rule.explain(obs)["reason"] == "hold"
    assert rule.action(obs) == 0
    if competing_queue > 4:
        without_hysteresis = RuleBasedController(policy=replace(
            rule.policy, switch_margin=0, relative_margin=0))
        assert without_hysteresis.action(obs) == 4


def test_substantial_competing_demand_overcomes_hysteresis() -> None:
    signal = make_signal()
    obs = observe(signal, queue_by_group=(0, 1, 0, 1, 0, 5, 0, 5))
    choice = RuleBasedController().explain(obs)
    assert choice["action"] == 4
    assert choice["reason"] == "higher_priority"


def test_capacity_variant_recognizes_two_straight_lanes() -> None:
    obs = observe(make_signal(4), queue_by_group=(6, 5, 6, 5, 0, 0, 0, 0))
    policy = replace(RulePolicyConfig(), red_age_weight=0)
    assert RuleBasedController(policy=replace(policy, service_horizon=0)).action(obs) == 1
    assert RuleBasedController(policy=policy).action(obs) == 0


@pytest.mark.parametrize("horizon", [0.0, 8.0])
def test_moving_far_vehicle_waiting_is_not_treated_as_empty(horizon: float) -> None:
    signal = make_signal(4)
    connection = connection_for({
        "north_left": ("N_in_2", 40.0, 7.0, 10.0),
        "south_left": ("S_in_2", 40.0, 7.0, 10.0),
    })
    obs = SUMOTrafficStateProvider(connection).observation(signal)
    assert not obs[:8].any() and not obs[52:60].any()
    assert obs[8:24].any()
    rule = RuleBasedController(policy=replace(RulePolicyConfig(), service_horizon=horizon))
    choice = rule.explain(obs)
    assert choice["action"] == 1
    assert choice["reason"] == "empty_current"
    assert choice["scores"][1] > 0


def test_waiting_only_current_demand_does_not_trigger_empty_gap_out() -> None:
    signal = make_signal(0)
    obs = observe(signal, total_waiting_by_group=(0, 60, 0, 60, 0, 0, 0, 0),
                  max_waiting_by_group=(0, 60, 0, 60, 0, 0, 0, 0),
                  approaching_by_group=(0, 0, 0, 0, 0, 1, 0, 1))
    choice = RuleBasedController().explain(obs)
    assert choice["action"] == 0
    assert choice["reason"] == "hold"


def test_approaching_traffic_can_request_a_phase_before_queue_forms() -> None:
    obs = observe(make_signal(0), approaching_by_group=(0, 0, 0, 0, 0, 2, 0, 2))
    assert RuleBasedController().action(obs) == 4


@pytest.mark.parametrize("elapsed", [0.0, 2.5])
def test_minimum_green_holds_even_for_large_competing_demand(elapsed: float) -> None:
    signal = make_signal(7, elapsed)
    obs = observe(signal, queue_by_group=(0, 10, 0, 10, 0, 0, 0, 0))
    choice = RuleBasedController().explain(obs)
    assert choice["action"] == 7
    assert choice["reason"] == "minimum_green"


def test_transition_returns_committed_target_and_cannot_interrupt_clearance() -> None:
    signal = make_signal(0)
    signal.apply_action(7)
    rule = RuleBasedController()
    for expected_stage in ("yellow", "all_red"):
        assert signal.stage == expected_stage
        obs = observe(signal, queue_by_group=(10, 10, 0, 0, 0, 0, 0, 0))
        choice = rule.explain(obs)
        assert choice["action"] == 7
        assert choice["reason"] == "transition"
        assert signal.apply_action(choice["action"]).ignored
        signal.advance(1.0)
    assert signal.current_phase == 7


def test_fourteen_second_choice_preempts_automatic_oldest_phase() -> None:
    signal = make_signal(0, 14.0)
    obs = observe(signal, queue_by_group=(0, 0, 0, 0, 0, 4, 0, 4))
    choice = RuleBasedController().explain(obs)
    assert choice["action"] == 4
    assert choice["reason"] == "maximum_green_soon"
    assert signal.apply_action(choice["action"]).switched
    signal.advance(1.0)
    assert signal.target_phase == 4
    assert signal.last_reason == "policy"
    assert signal.forced_changes == 0
    without_request = make_signal(0, 14.0)
    without_request.advance(1.0)
    assert without_request.target_phase == 1
    assert without_request.last_reason == "max_green"


def test_shared_controller_still_overrides_request_at_maximum_red() -> None:
    signal = make_signal(0)
    signal.red_age[7] = signal.config.max_red
    obs = observe(signal, queue_by_group=(0, 0, 0, 0, 0, 4, 0, 4))
    requested = RuleBasedController().action(obs)
    assert requested == 4
    applied = signal.apply_action(requested)
    assert applied.requested_action == 4
    assert applied.applied_action == 7
    assert applied.forced
    assert signal.last_reason == "max_red"


def test_identical_clipped_observations_cannot_reveal_raw_state_or_history() -> None:
    first, second = make_signal(0), make_signal(0)
    first.red_age[7], second.red_age[7] = 60.0, 600.0
    obs1 = observe(first, queue_by_group=(0, 0, 0, 0, 0, 10, 0, 10),
                   total_waiting_by_group=(0, 0, 0, 0, 0, 6000, 0, 6000),
                   max_waiting_by_group=(0, 0, 0, 0, 0, 120, 0, 120),
                   total_queue=20, total_waiting_time=12000, vehicle_count=20)
    obs2 = observe(second, queue_by_group=(0, 0, 0, 0, 0, 100, 0, 100),
                   total_waiting_by_group=(0, 0, 0, 0, 0, 60000, 0, 60000),
                   max_waiting_by_group=(0, 0, 0, 0, 0, 1200, 0, 1200),
                   total_queue=200, total_waiting_time=120000, vehicle_count=200)
    np.testing.assert_array_equal(obs1, obs2)
    rule = RuleBasedController()
    expected = rule.explain(obs1)
    rule.action(observe(make_signal(4), queue_by_group=(1, 1, 0, 0, 0, 0, 0, 0)))
    assert rule.explain(obs2) == expected


@pytest.mark.parametrize("invalid", [np.zeros(59), np.zeros((1, 60)), np.zeros(61)])
def test_incorrect_observation_shape_is_rejected(invalid: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        RuleBasedController().action(invalid)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_nonfinite_and_out_of_range_features_are_rejected(value: float) -> None:
    obs = observe(make_signal())
    obs[0] = value
    with pytest.raises(ValueError, match="finite values"):
        RuleBasedController().action(obs)


@pytest.mark.parametrize("indices,values", [
    ([24], [0]), ([25], [1]), ([24, 25], [.5, .5]),
    ([33], [0]), ([34], [1]), ([33, 34], [.5, .5]),
    ([36], [1]), ([36], [.5]),
    ([33, 34], [0, 1]), ([33, 34, 36, 37], [0, 1, 1, 1]),
])
def test_malformed_phase_stage_and_target_contract_is_rejected(indices, values) -> None:
    obs = observe(make_signal())
    obs[indices] = values
    with pytest.raises(ValueError):
        RuleBasedController().action(obs)


def test_candidate_set_covers_simple_and_capacity_delay_aware_policies() -> None:
    candidates = candidate_policies()
    assert len(candidates) == 37
    assert len({tuple(asdict(policy).values()) for policy in candidates.values()}) == 37
    assert {policy.service_horizon for policy in candidates.values()} == {0.0, 8.0}
    assert {policy.waiting_weight for policy in candidates.values()} == {0.0, 4.0, 12.0}
    assert {policy.approaching_weight for policy in candidates.values()} >= {0.0, 0.25, 0.75}
    assert len({policy.switch_margin for policy in candidates.values()}) >= 3
    obs = observe(make_signal(), queue_by_group=(3, 2, 1, 4, 5, 2, 3, 4))
    before = obs.copy()
    for name, policy in candidates.items():
        assert RulePolicyConfig.from_dict(asdict(policy)) == policy, name
        action = RuleBasedController(policy=policy).action(obs)
        assert type(action) is int and 0 <= action < 8, name
    np.testing.assert_array_equal(obs, before)


@pytest.mark.parametrize("settings", [
    {"waiting_weight": -1}, {"approaching_weight": float("inf")},
    {"red_age_weight": float("nan")}, {"discharge_headway": 0}, {"typo": 1},
])
def test_invalid_policy_settings_are_rejected(settings: dict) -> None:
    with pytest.raises(ValueError):
        RulePolicyConfig.from_dict(settings)


@pytest.mark.parametrize("horizon", [1.0, 2.0])
def test_capacity_window_must_allow_green_after_clearance(horizon: float) -> None:
    with pytest.raises(ValueError, match="service_horizon"):
        RuleBasedController(policy=replace(RulePolicyConfig(), service_horizon=horizon))


@pytest.mark.skipif(os.environ.get("RUN_SUMO_INTEGRATION") != "1",
                    reason="Set RUN_SUMO_INTEGRATION=1 for the actual SUMO rule smoke test.")
def test_rule_completes_real_sumo_episode_with_shared_signal_safety(tmp_path) -> None:
    from model.env.intersection_env import IntersectionEnv

    env = IntersectionEnv(controller_name="Rule-Based", scenario="heavy", seed=19071,
                          episode_seconds=90, route_dir=tmp_path)
    rule = RuleBasedController(env.config)
    phases: set[int] = set()
    stages: set[str] = set()
    try:
        obs, _ = env.reset(seed=19071)
        for step in range(90):
            action = rule.action(obs)
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            assert not terminated
            assert truncated == (step == 89)
            assert info["reward_terms"]["switching"] <= 0
            if env.signal_controller.stage == "green":
                phases.add(env.signal_controller.current_phase)
                state = env.connection.trafficlight.getRedYellowGreenState("J")
                assert {index for index, color in enumerate(state) if color == "G"} == set(
                    GREEN_LINKS[env.signal_controller.current_phase])
            stages.add(env.signal_controller.stage)
        summary = env.episode_summary()
        assert len(phases) >= 2
        assert "all_red" in stages and "green" in stages
        assert summary["phase_changes"] > 0
        assert summary["collisions"] == summary["teleports"] == 0
        assert summary["duration"] == 90.0
        assert summary["throughput"] > 0
    finally:
        env.close()
