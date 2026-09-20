"""Deterministic 12-lane initial congestion route generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from model.utils.config import ProjectConfig


APPROACHES = ("N", "S", "E", "W")
DESTINATIONS = {
    "N": {"left": "E_out", "straight": "S_out", "right": "W_out"},
    "S": {"left": "W_out", "straight": "N_out", "right": "E_out"},
    "E": {"left": "N_out", "straight": "W_out", "right": "S_out"},
    "W": {"left": "S_out", "straight": "E_out", "right": "N_out"},
}


@dataclass(frozen=True)
class LanePlacement:
    left: int = 0
    straight: int = 0
    lane3_total: int = 0
    lane3_right: int = 0

    def validate(self) -> None:
        values = (self.left, self.straight, self.lane3_total)
        if any(value < 0 or value > 5 for value in values):
            raise ValueError("Each lane count must be between 0 and 5")
        if self.lane3_right < 0 or self.lane3_right > self.lane3_total:
            raise ValueError("lane3_right must be between 0 and lane3_total")

    @property
    def total(self) -> int:
        return self.left + self.straight + self.lane3_total


def _lane_lengths(network_file: Path) -> dict[str, float]:
    root = ET.parse(network_file).getroot()
    lengths: dict[str, float] = {}
    for lane in root.findall(".//lane"):
        lane_id = lane.get("id")
        if lane_id and lane_id[:1] in APPROACHES and "_in_" in lane_id:
            lengths[lane_id] = float(lane.get("length", "0"))
    return lengths


def generate_initial_route_file(
    path: Path,
    placements: dict[str, LanePlacement],
    config: ProjectConfig | None = None,
) -> tuple[Path, int]:
    """Create explicitly positioned vehicles and return the expected count."""

    cfg = config or ProjectConfig()
    if set(placements) != set(APPROACHES):
        raise ValueError("Placements must define N, S, E, and W")
    for placement in placements.values():
        placement.validate()

    lengths = _lane_lengths(cfg.network_file)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        '    <vType id="passenger" accel="2.6" decel="4.5" sigma="0" length="5" '
        'minGap="2.5" maxSpeed="13.89" guiShape="passenger" />',
    ]
    for approach in APPROACHES:
        for movement, outgoing in DESTINATIONS[approach].items():
            lines.append(
                f'    <route id="route_{approach}_{movement}" '
                f'edges="{approach}_in {outgoing}" />'
            )

    vehicles: list[tuple[str, str, int, float]] = []
    for approach in APPROACHES:
        placement = placements[approach]
        lane_movements = {
            0: ["left"] * placement.left,
            1: ["straight"] * placement.straight,
            2: ["right"] * placement.lane3_right
            + ["straight"] * (placement.lane3_total - placement.lane3_right),
        }
        for lane_index, movements in lane_movements.items():
            lane_id = f"{approach}_in_{lane_index}"
            lane_length = lengths.get(lane_id)
            if lane_length is None:
                raise ValueError(f"Lane not found in network: {lane_id}")
            for queue_index, movement in enumerate(movements):
                position = lane_length - 10.0 - queue_index * 8.0
                if position < 5.0:
                    raise ValueError(f"Too many vehicles for lane length: {lane_id}")
                vehicle_id = f"initial_{approach}_{lane_index}_{queue_index}"
                vehicles.append((vehicle_id, f"route_{approach}_{movement}", lane_index, position))

    for vehicle_id, route_id, lane_index, position in vehicles:
        lines.append(
            f'    <vehicle id="{vehicle_id}" type="passenger" route="{route_id}" '
            f'depart="0" departLane="{lane_index}" departPos="{position:.2f}" '
            'departSpeed="0" />'
        )
    lines.append("</routes>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path, len(vehicles)
