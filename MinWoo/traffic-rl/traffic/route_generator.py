"""Seeded Poisson arrivals, materialized as identical vehicle XML for paired runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from build_network import DESTINATIONS, write_xml
from utils.config import Config, DIRECTIONS, ROOT, SCENARIOS, load_config


def generate_routes(path: Path, config: Config, seed: int, scenario: str | None = None) -> dict:
    config.validate()
    rng = np.random.default_rng(seed)
    selected = scenario or config.traffic.scenario
    if selected not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {selected}")
    realized = str(rng.choice(SCENARIOS[:-1])) if selected == "random" else selected
    base = {"balanced": [600] * 4, "ns_heavy": [1300, 1300, 350, 350],
            "ew_heavy": [350, 350, 1300, 1300], "left_heavy": [600] * 4,
            "heavy": [1600] * 4, "low": [150] * 4}[realized]
    if selected == "random":
        base = rng.uniform(*config.traffic.random_rate_range, size=4).tolist()
    rates = config.traffic.rates or dict(zip(DIRECTIONS, base))
    ratios = {d: list(config.traffic.turn_ratios) for d in DIRECTIONS}
    if realized == "left_heavy":
        ratios[config.traffic.left_direction] = [0.7, 0.2, 0.1]
        if config.traffic.rates is None:
            rates[config.traffic.left_direction] = 1300
    if selected == "random":
        ratios = {d: rng.dirichlet(np.array(ratios[d]) * 30 + 0.1).tolist() for d in DIRECTIONS}
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", accel="2.6", decel="4.5", sigma="0.5", length="5",
                  minGap="2.5", maxSpeed="11.11", tau="1.0")
    for direction in DIRECTIONS:
        for turn, target in zip(("left", "straight", "right"), DESTINATIONS[direction]):
            ET.SubElement(root, "route", id=f"{direction}_{turn}", edges=f"{direction}_in {target}_out")
    vehicles = []
    for direction in DIRECTIONS:
        rate = rates[direction]
        if rate == 0:
            continue
        depart = float(rng.exponential(3600 / rate))
        while depart < config.simulation.demand_seconds:
            turn = str(rng.choice(["left", "straight", "right"], p=ratios[direction]))
            lane = "2" if turn == "left" else "0" if turn == "right" else str(rng.integers(0, 2))
            vehicles.append((depart, direction, turn, lane))
            depart += float(rng.exponential(3600 / rate))
    for i, (depart, direction, turn, lane) in enumerate(sorted(vehicles)):
        ET.SubElement(root, "vehicle", id=f"veh_{i:06}", type="car", route=f"{direction}_{turn}",
                      depart=f"{depart:.3f}", departLane=lane, departSpeed="max")
    path.parent.mkdir(parents=True, exist_ok=True)
    write_xml(root, path)
    metadata = {"seed": seed, "scenario": selected, "realized_scenario": realized, "rates_vph": rates,
                "turn_ratios": ratios, "scheduled": len(vehicles), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--scenario", choices=SCENARIOS, default="balanced")
    parser.add_argument("--output", type=Path, default=ROOT / "sumo" / "routes.rou.xml")
    args = parser.parse_args()
    print(json.dumps(generate_routes(args.output, load_config(args.config), args.seed, args.scenario), indent=2))
