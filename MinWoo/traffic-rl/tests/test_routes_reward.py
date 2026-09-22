import numpy as np
import pytest
import xml.etree.ElementTree as ET

from traffic.route_generator import generate_routes
from traffic.state_provider import TrafficState
from utils.config import Config, SCENARIOS
from utils.reward import reward_terms, switching_penalty


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_reproducible_routes(tmp_path, scenario):
    config = Config()
    one = generate_routes(tmp_path / "one.xml", config, 2001, scenario)
    two = generate_routes(tmp_path / "two.xml", config, 2001, scenario)
    other = generate_routes(tmp_path / "other.xml", config, 2002, scenario)
    assert one == two
    assert one["sha256"] != other["sha256"]
    vehicles = ET.parse(tmp_path / "one.xml").getroot().findall("vehicle")
    assert len(vehicles) == one["scheduled"]
    departures = [float(v.attrib["depart"]) for v in vehicles]
    assert departures == sorted(departures)
    for v in vehicles:
        if v.attrib["route"].endswith("left"):
            assert v.attrib["departLane"] == "2"
        if v.attrib["route"].endswith("right"):
            assert v.attrib["departLane"] == "0"


def test_no_demand_and_invalid_rates(tmp_path):
    config = Config()
    config.traffic.rates = dict(N=0, S=0, E=0, W=0)
    assert generate_routes(tmp_path / "empty.xml", config, 1, "low")["scheduled"] == 0
    config.traffic.rates["N"] = -1
    with pytest.raises(ValueError):
        config.validate()


def test_reward_bounds_and_monotonicity():
    config = Config()
    empty = TrafficState((0,) * 8, 0, 0)
    busy = TrafficState((1,) * 8, 20, 10)
    saturated = TrafficState((100000,) * 8, 1e9, 1e9)
    assert sum(reward_terms(empty, config.normalization, config.reward).values()) == 0
    assert sum(reward_terms(busy, config.normalization, config.reward).values()) < 0
    terms = reward_terms(saturated, config.normalization, config.reward)
    assert terms == {"queue": -1.0, "waiting": -0.3, "max_waiting": -0.5}
    assert switching_penalty(2, config.reward) == -0.4
    assert np.isfinite(saturated.normalized(config.normalization)).all()


def test_configuration_rejects_unaligned_timing():
    config = Config()
    config.signal.all_red = 0.7
    with pytest.raises(ValueError):
        config.validate()
