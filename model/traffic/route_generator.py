"""MinWoo-compatible seeded Poisson traffic generation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    duration: int = 300
    demand_seconds: int | None = None
    base_arrivals_per_second: float = 600.0 / 3_600.0
    scenario: str = "uniform"
    left_ratio: float = 0.25
    straight_ratio: float = 0.60
    right_ratio: float = 0.15
    seed: int = 1
    vehicle_type: str = "passenger"
    approach_arrival_rates: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {self.scenario!r}; choose from {SCENARIOS}")
        if abs(self.left_ratio + self.straight_ratio + self.right_ratio - 1.0) > 1e-6:
            raise ValueError("left_ratio + straight_ratio + right_ratio must equal 1")
        if self.approach_arrival_rates is not None:
            if len(self.approach_arrival_rates) != 4:
                raise ValueError("approach_arrival_rates must contain N, S, E, W values")
            if any(rate < 0.0 for rate in self.approach_arrival_rates):
                raise ValueError("arrival rates cannot be negative")


def _route_for(approach: str, movement: str) -> tuple[str, str, int]:
    destinations = {
        "N": {"left": "E_out", "straight": "S_out", "right": "W_out"},
        "S": {"left": "W_out", "straight": "N_out", "right": "E_out"},
        "E": {"left": "S_out", "straight": "W_out", "right": "N_out"},
        "W": {"left": "N_out", "straight": "E_out", "right": "S_out"},
    }
    lanes = {"left": 2, "straight": 1, "right": 0}
    return f"{approach}_in", destinations[approach][movement], lanes[movement]


def _scenario_rates(scenario: str, rng: np.random.Generator) -> dict[str, float]:
    if scenario == "uniform":
        values = [600.0] * 4
    elif scenario == "north_south_congested":
        values = [1_300.0, 1_300.0, 350.0, 350.0]
    elif scenario == "east_west_congested":
        values = [350.0, 350.0, 1_300.0, 1_300.0]
    elif scenario in {"left_turn_congested"}:
        values = [1_300.0, 600.0, 600.0, 600.0]
    elif scenario == "heavy":
        values = [1_600.0] * 4
    elif scenario == "low":
        values = [150.0] * 4
    else:
        values = rng.uniform(150.0, 1_500.0, size=4).tolist()
    return dict(zip(APPROACHES, values, strict=True))


def generate_route_file(path: Path, demand: TrafficDemand) -> Path:
    """Materialize a deterministic route file for paired Fixed/DQN runs."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(demand.seed)
    if demand.scenario == "random":
        # Preserve MinWoo's RNG stream: the original generator records a realized
        # scenario label before drawing independent random rates and turn ratios.
        rng.choice(SCENARIOS[:-1])
    rates = _scenario_rates(demand.scenario, rng)
    if demand.approach_arrival_rates is not None:
        # The existing web API expresses custom rates as vehicles/second.
        rates = {
            direction: float(rate) * 3_600.0
            for direction, rate in zip(APPROACHES, demand.approach_arrival_rates, strict=True)
        }

    base_ratios = np.asarray(
        [demand.left_ratio, demand.straight_ratio, demand.right_ratio], dtype=np.float64
    )
    ratios = {direction: base_ratios.copy() for direction in APPROACHES}
    if demand.scenario == "left_turn_congested":
        ratios["N"] = np.asarray([0.7, 0.2, 0.1], dtype=np.float64)
    elif demand.scenario == "random":
        ratios = {
            direction: rng.dirichlet(base_ratios * 30.0 + 0.1)
            for direction in APPROACHES
        }

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        '    <vType id="passenger" accel="2.6" decel="4.5" sigma="0.5" length="5" '
        'minGap="2.5" maxSpeed="11.11" tau="1.0" guiShape="passenger" />',
    ]
    for approach in APPROACHES:
        for movement in MOVEMENTS:
            incoming, outgoing, _ = _route_for(approach, movement)
            lines.append(
                f'    <route id="route_{approach}_{movement}" edges="{incoming} {outgoing}" />'
            )

    demand_seconds = min(demand.duration, demand.demand_seconds or demand.duration)
    vehicles: list[tuple[float, str, str, int]] = []
    for approach in APPROACHES:
        rate = rates[approach]
        if rate <= 0.0:
            continue
        depart = float(rng.exponential(3_600.0 / rate))
        while depart < demand_seconds:
            movement = str(rng.choice(MOVEMENTS, p=ratios[approach]))
            lane = 2 if movement == "left" else 0 if movement == "right" else int(rng.integers(0, 2))
            vehicles.append((depart, approach, movement, lane))
            depart += float(rng.exponential(3_600.0 / rate))

    for index, (depart, approach, movement, lane) in enumerate(sorted(vehicles)):
        lines.append(
            f'    <vehicle id="veh_{index:06d}" type="{demand.vehicle_type}" '
            f'route="route_{approach}_{movement}" depart="{depart:.3f}" '
            f'departLane="{lane}" departSpeed="max" />'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUMO_DIR / "routes.rou.xml")
    parser.add_argument("--scenario", choices=SCENARIOS, default="uniform")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--demand-seconds", type=int, default=240)
    args = parser.parse_args()
    output = generate_route_file(
        args.output,
        TrafficDemand(
            duration=args.duration,
            demand_seconds=min(args.duration, args.demand_seconds),
            scenario=args.scenario,
            seed=args.seed,
        ),
    )
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
