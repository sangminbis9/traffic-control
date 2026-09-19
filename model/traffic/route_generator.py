"""Deterministic SUMO route generation for repeatable controller comparisons."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from model.utils.config import SUMO_DIR


SCENARIOS = (
    "uniform",
    "north_south_congested",
    "east_west_congested",
    "left_turn_congested",
    "heavy",
    "low",
    "random",
)

APPROACHES = ("N", "S", "E", "W")
MOVEMENTS = ("left", "straight", "right")


@dataclass(frozen=True)
class TrafficDemand:
    """Demand and turn-ratio parameters for one generated episode."""

    duration: int = 300
    base_arrivals_per_second: float = 0.075
    scenario: str = "uniform"
    left_ratio: float = 0.20
    straight_ratio: float = 0.60
    right_ratio: float = 0.20
    seed: int = 1
    vehicle_type: str = "passenger"

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {self.scenario!r}; choose from {SCENARIOS}")
        ratio_total = self.left_ratio + self.straight_ratio + self.right_ratio
        if abs(ratio_total - 1.0) > 1e-6:
            raise ValueError("left_ratio + straight_ratio + right_ratio must equal 1")


def _scenario_multipliers(scenario: str, rng: random.Random) -> Dict[str, float]:
    if scenario == "uniform":
        return {approach: 1.0 for approach in APPROACHES}
    if scenario == "north_south_congested":
        return {"N": 2.2, "S": 2.2, "E": 0.45, "W": 0.45}
    if scenario == "east_west_congested":
        return {"N": 0.45, "S": 0.45, "E": 2.2, "W": 2.2}
    if scenario == "left_turn_congested":
        return {approach: 1.0 for approach in APPROACHES}
    if scenario == "heavy":
        return {approach: 2.7 for approach in APPROACHES}
    if scenario == "low":
        return {approach: 0.30 for approach in APPROACHES}
    # Randomized episode family: the same seed still reproduces the draw.
    return {approach: rng.uniform(0.45, 2.35) for approach in APPROACHES}


def _movement_multiplier(scenario: str, movement: str) -> float:
    if scenario == "left_turn_congested":
        return {"left": 2.9, "straight": 0.75, "right": 0.75}[movement]
    return 1.0


def _route_for(approach: str, movement: str) -> tuple[str, str, int]:
    """Return target edge and the fixed incoming lane for a movement."""

    destinations = {
        "N": {"left": "E_out", "straight": "S_out", "right": "W_out"},
        "S": {"left": "W_out", "straight": "N_out", "right": "E_out"},
        "E": {"left": "N_out", "straight": "W_out", "right": "S_out"},
        "W": {"left": "S_out", "straight": "E_out", "right": "N_out"},
    }
    lanes = {"left": 0, "straight": 1, "right": 2}
    return f"{approach}_in", destinations[approach][movement], lanes[movement]


def generate_route_file(path: Path, demand: TrafficDemand) -> Path:
    """Generate a route XML file and return its absolute path."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(demand.seed)
    multipliers = _scenario_multipliers(demand.scenario, rng)
    ratios = {"left": demand.left_ratio, "straight": demand.straight_ratio, "right": demand.right_ratio}

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes>',
        '    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" '
        'minGap="2.5" maxSpeed="13.89" guiShape="passenger" />',
    ]
    route_ids: dict[tuple[str, str], str] = {}
    for approach in APPROACHES:
        for movement in MOVEMENTS:
            incoming, outgoing, _ = _route_for(approach, movement)
            route_id = f"route_{approach}_{movement}"
            route_ids[(approach, movement)] = route_id
            lines.append(f'    <route id="{route_id}" edges="{incoming} {outgoing}" />')

    vehicle_index = 0
    for depart in range(demand.duration):
        for approach in APPROACHES:
            for movement in MOVEMENTS:
                probability = (
                    demand.base_arrivals_per_second
                    * multipliers[approach]
                    * _movement_multiplier(demand.scenario, movement)
                    * ratios[movement]
                )
                if rng.random() >= min(probability, 0.95):
                    continue
                lane = _route_for(approach, movement)[2]
                # Two straight lanes deliberately share demand to model a 3-lane approach.
                if movement == "straight":
                    lane = rng.choice((1, 2))
                vehicle_id = f"veh_{vehicle_index:07d}"
                vehicle_index += 1
                lines.append(
                    f'    <vehicle id="{vehicle_id}" type="{demand.vehicle_type}" '
                    f'route="{route_ids[(approach, movement)]}" depart="{depart}" '
                    f'departLane="{lane}" departSpeed="max" />'
                )
    lines.append("</routes>")
    # All generated values are controlled identifiers/numbers; keep XML tags intact.
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUMO_DIR / "routes.rou.xml")
    parser.add_argument("--scenario", choices=SCENARIOS, default="uniform")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", type=int, default=300)
    args = parser.parse_args()
    output = generate_route_file(
        args.output,
        TrafficDemand(duration=args.duration, scenario=args.scenario, seed=args.seed),
    )
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
