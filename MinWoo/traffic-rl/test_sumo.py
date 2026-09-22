"""Independent TraCI smoke test, before importing Gymnasium or DQN."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from build_network import build_network
from controller.fixed_controller import FixedTimeController
from controller.phases import PHASE_NAMES
from controller.signal_controller import SignalController
from simulation.backend import SUMOBackend
from traffic.route_generator import generate_routes
from traffic.state_provider import SUMOTrafficStateProvider
from utils.config import DIRECTIONS, ROOT, SCENARIOS, load_config
from utils.metrics import EpisodeMetrics, append_csv
from utils.reward import reward_terms, switching_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--capture", action="store_true", help="Save an actual SUMO GUI screenshot")
    parser.add_argument("--delay", type=float, default=0, help="Wall seconds per simulation step")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--scenario", choices=SCENARIOS, default="balanced")
    parser.add_argument("--print-every", type=float, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "results/sumo_test")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seconds:
        config.simulation.episode_seconds = args.seconds
        config.simulation.demand_seconds = min(config.simulation.demand_seconds, args.seconds * 0.8)
    config.validate()
    if not (ROOT / "sumo/intersection.net.xml").exists():
        build_network(config)
    metadata = generate_routes(args.output / "routes.rou.xml", config, args.seed, args.scenario)
    backend, metrics, provider = SUMOBackend(), EpisodeMetrics(), SUMOTrafficStateProvider()
    fixed = FixedTimeController(config.signal.fixed_green)
    next_print = 0.0
    try:
        backend.start(config, args.output / "routes.rou.xml", args.seed, args.output, args.gui)
        signal = SignalController(backend, config.signal)
        if args.gui:
            backend.connection.gui.setZoom("View #0", 160)
        nsteps = round(config.simulation.episode_seconds / config.simulation.step_length)
        for step in range(nsteps):
            previous_changes = signal.phase_changes
            signal.request(fixed.action(signal))
            snapshot = backend.step()
            signal.tick(config.simulation.step_length)
            metrics.update(snapshot, config.simulation.step_length)
            state = provider.get_state(snapshot)
            terms = {key: value * config.simulation.step_length / config.simulation.decision_interval
                     for key, value in reward_terms(state, config.normalization, config.reward).items()}
            terms["switching"] = switching_penalty(signal.phase_changes - previous_changes, config.reward)
            metrics.reward += sum(terms.values())
            for key, value in terms.items():
                metrics.term_totals[key] += value
            if snapshot.collisions or snapshot.teleports:
                raise RuntimeError("Collision/teleport detected; inspect sumo.log")
            if args.capture and args.gui and step == nsteps - 3:
                backend.connection.gui.screenshot("View #0", os.path.relpath((args.output / "sumo_gui.png").resolve()), 1200, 900)
            if snapshot.time >= next_print:
                for i, d in enumerate(DIRECTIONS):
                    print(f"{d} Left: {state.queues[2*i]:.0f} | Straight+Right: {state.queues[2*i+1]:.0f}")
                print(f"t={snapshot.time:.1f} Total Queue={sum(state.queues):.0f} Total Waiting={state.total_waiting:.1f}s "
                      f"Max Waiting={state.max_waiting:.1f}s Current Phase={signal.phase} ({PHASE_NAMES[signal.phase]}) "
                      f"stage={signal.stage} SUMO state={backend.connection.trafficlight.getRedYellowGreenState('J')}")
                next_print += args.print_every
            if args.delay:
                time.sleep(args.delay)
        row = {"episode": 1, "controller": "fixed", "seed": args.seed, "traffic_scenario": args.scenario,
               **metrics.summary(), "phase_changes": signal.phase_changes, "route_sha256": metadata["sha256"]}
        append_csv(args.output / "metrics.csv", row)
        print(json.dumps(row, indent=2))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
