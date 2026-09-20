"""Synchronized Fixed-Time versus DQN SUMO execution and live frame capture."""

from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any, Callable

import numpy as np
from stable_baselines3 import DQN

from api.app.schemas.models import ComparisonCreate
from model.env.intersection_env import IntersectionEnv
from model.traffic.initial_placement import LanePlacement, generate_initial_route_file
from model.traffic.route_generator import TrafficDemand, generate_route_file
from model.utils.config import ProjectConfig
from model.utils.reproducibility import collect_reproducibility_metadata


PHASE_NAMES = ("NS Straight", "NS Left", "EW Straight", "EW Left")


class SynchronizedComparisonRunner:
    def __init__(
        self,
        experiment_id: str,
        request: ComparisonCreate,
        model_path: Path,
        artifact_dir: Path,
        event_handler: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.request = request
        self.model_path = model_path
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.event_handler = event_handler
        self.pause_requested = threading.Event()
        self.stop_requested = threading.Event()
        self.status = "PENDING"
        self.fixed_env: IntersectionEnv | None = None
        self.dqn_env: IntersectionEnv | None = None
        self.timeseries: list[dict[str, Any]] = []
        self.waiting_history: dict[str, dict[str, float]] = {"fixed": {}, "dqn": {}}
        self.group_waiting_history: dict[str, dict[str, dict[str, float]]] = {
            controller: {
                f"{approach}_{movement}": {}
                for approach in ("N", "S", "E", "W")
                for movement in ("left", "straight")
            }
            for controller in ("fixed", "dqn")
        }
        self.phase_timeline: dict[str, list[dict[str, Any]]] = {"fixed": [], "dqn": []}

    def pause(self) -> None:
        self.pause_requested.set()

    def resume(self) -> None:
        self.pause_requested.clear()

    def stop(self) -> None:
        self.stop_requested.set()
        self.pause_requested.clear()

    def set_speed(self, speed: str) -> None:
        if speed not in {"0.5", "1", "2", "4", "max"}:
            raise ValueError(f"Unsupported simulation speed: {speed}")
        self.request.speed = speed

    def _emit(self, payload: dict[str, Any]) -> None:
        event = {"experiment_id": self.experiment_id, **payload}
        if self.event_handler:
            self.event_handler(event)

    def _route_file(self) -> tuple[Path, int | None]:
        config = ProjectConfig()
        route_file = self.artifact_dir / f"traffic_{self.request.seed}.rou.xml"
        if self.request.test_mode == "initial":
            placements = {
                approach: LanePlacement(**getattr(self.request.initial, approach).model_dump())
                for approach in ("N", "S", "E", "W")
            }
            return generate_initial_route_file(route_file, placements, config)
        custom = self.request.continuous
        rates = None
        if custom.n_rate is not None:
            rates = (custom.n_rate, custom.s_rate, custom.e_rate, custom.w_rate)
        generate_route_file(
            route_file,
            TrafficDemand(
                duration=self.request.duration,
                scenario=custom.scenario,
                left_ratio=custom.left_ratio,
                straight_ratio=custom.straight_ratio,
                right_ratio=custom.right_ratio,
                seed=self.request.seed,
                approach_arrival_rates=rates,
            ),
        )
        return route_file, None

    def run(self) -> dict[str, Any]:
        self.status = "RUNNING"
        route_file, requested_initial = self._route_file()
        config = ProjectConfig()
        self.fixed_env = IntersectionEnv(
            config,
            controller_name="Fixed-Time",
            scenario=self.request.continuous.scenario,
            seed=self.request.seed,
            episode_seconds=self.request.duration,
            route_file=route_file,
        )
        self.dqn_env = IntersectionEnv(
            config,
            controller_name="DQN",
            scenario=self.request.continuous.scenario,
            seed=self.request.seed,
            episode_seconds=self.request.duration,
            route_file=route_file,
        )
        model = DQN.load(str(self.model_path), device="cpu")
        started = time.monotonic()
        try:
            fixed_observation, _ = self.fixed_env.reset(seed=self.request.seed)
            dqn_observation, _ = self.dqn_env.reset(seed=self.request.seed)
            if requested_initial is not None:
                fixed_observation = self._prime_and_validate(self.fixed_env, requested_initial)
                dqn_observation = self._prime_and_validate(self.dqn_env, requested_initial)
            initial_count = requested_initial or 0
            milestone_times: dict[str, dict[str, float | None]] = {
                "fixed": {"50": None, "90": None, "100": None},
                "dqn": {"50": None, "90": None, "100": None},
            }
            step = 0
            fixed_done = dqn_done = False
            while not self.stop_requested.is_set():
                while self.pause_requested.is_set() and not self.stop_requested.is_set():
                    self.status = "PAUSED"
                    time.sleep(0.05)
                self.status = "RUNNING"
                fixed_observation, _, _, fixed_done, _ = self.fixed_env.step(0)
                action, _ = model.predict(dqn_observation, deterministic=True)
                dqn_observation, _, _, dqn_done, _ = self.dqn_env.step(int(action))
                step += 1
                self._assert_synchronized()
                simulation_time = step * config.signal.decision_interval
                fixed_side = self._capture_side("fixed", self.fixed_env, simulation_time, initial_count)
                dqn_side = self._capture_side("dqn", self.dqn_env, simulation_time, initial_count)
                self._update_milestones(milestone_times["fixed"], fixed_side, simulation_time, initial_count)
                self._update_milestones(milestone_times["dqn"], dqn_side, simulation_time, initial_count)
                row = self._timeseries_row(simulation_time, fixed_side, dqn_side)
                self.timeseries.append(row)
                if step % self.request.render_interval == 0:
                    self._emit(
                        {
                            "type": "simulation_frame",
                            "status": self.status,
                            "simulation_time": simulation_time,
                            "network_bounds": self._network_bounds(self.fixed_env),
                            "fixed": fixed_side,
                            "dqn": dqn_side,
                            "delta": self._delta(fixed_side["metrics"], dqn_side["metrics"]),
                        }
                    )
                if self.request.test_mode == "initial":
                    if fixed_side["metrics"]["vehicles_remaining"] == 0 and dqn_side["metrics"]["vehicles_remaining"] == 0:
                        break
                    if fixed_done and dqn_done:
                        break
                elif fixed_done and dqn_done:
                    break
                self._throttle()

            self.status = "STOPPED" if self.stop_requested.is_set() else "COMPLETED"
            result = self._build_result(milestone_times, requested_initial, time.monotonic() - started)
            self._persist(result)
            self._emit({"type": "comparison_complete", "status": self.status, "result": result})
            return result
        except Exception as exc:
            self.status = "FAILED"
            self._emit({"type": "comparison_error", "status": self.status, "error": str(exc)})
            raise
        finally:
            if self.fixed_env is not None:
                self.fixed_env.close()
            if self.dqn_env is not None:
                self.dqn_env.close()

    def _prime_and_validate(self, env: IntersectionEnv, expected: int) -> np.ndarray:
        env.connection.simulationStep()
        env.signal_controller.advance(env.config.simulation.step_length)
        provider = env._state_provider
        if provider is None:
            raise RuntimeError("SUMO state provider is not initialized")
        snapshot = provider.snapshot()
        env._previous_snapshot = snapshot
        actual = snapshot.vehicle_count
        if actual != expected:
            raise RuntimeError(
                f"Initial placement mismatch: requested {expected} vehicles, actual {actual}"
            )
        requested_lanes = self._requested_lane_counts()
        actual_lanes = {lane: 0 for lane in requested_lanes}
        for vehicle_id in env.connection.vehicle.getIDList():
            lane_id = env.connection.vehicle.getLaneID(vehicle_id)
            if lane_id in actual_lanes:
                actual_lanes[lane_id] += 1
        if actual_lanes != requested_lanes:
            raise RuntimeError(
                f"Initial lane placement mismatch: requested {requested_lanes}, actual {actual_lanes}"
            )
        return provider.observation(
            env.signal_controller.current_phase,
            env.signal_controller.phase_elapsed,
            env.config.signal.max_green,
        )

    def _requested_lane_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for approach in ("N", "S", "E", "W"):
            lane = getattr(self.request.initial, approach)
            counts[f"{approach}_in_0"] = lane.left
            counts[f"{approach}_in_1"] = lane.straight
            counts[f"{approach}_in_2"] = lane.lane3_total
        return counts

    def _assert_synchronized(self) -> None:
        assert self.fixed_env is not None and self.dqn_env is not None
        fixed_time = float(self.fixed_env.connection.simulation.getTime())
        dqn_time = float(self.dqn_env.connection.simulation.getTime())
        if abs(fixed_time - dqn_time) > 1e-9:
            raise RuntimeError(f"SUMO clock mismatch: fixed={fixed_time}, dqn={dqn_time}")

    def _capture_side(
        self, key: str, env: IntersectionEnv, simulation_time: float, initial_count: int
    ) -> dict[str, Any]:
        snapshot = env._previous_snapshot
        if snapshot is None:
            raise RuntimeError("Missing traffic snapshot")
        vehicles = []
        for vehicle_id in env.connection.vehicle.getIDList():
            x, y = env.connection.vehicle.getPosition(vehicle_id)
            waiting = float(env.connection.vehicle.getAccumulatedWaitingTime(vehicle_id))
            self.waiting_history[key][vehicle_id] = max(self.waiting_history[key].get(vehicle_id, 0.0), waiting)
            lane_id = env.connection.vehicle.getLaneID(vehicle_id)
            if "_in_" in lane_id and lane_id[:1] in {"N", "S", "E", "W"}:
                movement = "left" if lane_id.endswith("_0") else "straight"
                group = f"{lane_id[0]}_{movement}"
                histories = self.group_waiting_history[key][group]
                histories[vehicle_id] = max(histories.get(vehicle_id, 0.0), waiting)
            vehicles.append(
                {
                    "id": vehicle_id,
                    "x": float(x),
                    "y": float(y),
                    "angle": float(env.connection.vehicle.getAngle(vehicle_id)),
                    "speed": float(env.connection.vehicle.getSpeed(vehicle_id)),
                    "waiting_time": waiting,
                    "lane": lane_id,
                    "route": list(env.connection.vehicle.getRoute(vehicle_id)),
                }
            )
        remaining = int(env.connection.simulation.getMinExpectedNumber())
        throughput = int(env._metrics.throughput if env._metrics else 0)
        clearance = 0.0 if initial_count <= 0 else min(100.0, throughput / initial_count * 100.0)
        queues = dict(zip(("N_left", "N_straight", "S_left", "S_straight", "E_left", "E_straight", "W_left", "W_straight"), snapshot.queue_by_group, strict=True))
        metrics = {
            "vehicles_remaining": remaining,
            "current_queue": snapshot.total_queue,
            "average_waiting": snapshot.total_waiting_time / max(snapshot.vehicle_count, 1),
            "maximum_waiting": snapshot.max_waiting_time,
            "throughput": throughput,
            "phase_changes": env.signal_controller.phase_changes,
            "clearance_percent": clearance,
            "approach_queues": queues,
            "approach_waiting": {
                group: {
                    "average": float(np.mean(list(values.values()))) if values else 0.0,
                    "maximum": max(values.values(), default=0.0),
                }
                for group, values in self.group_waiting_history[key].items()
            },
        }
        self.phase_timeline[key].append(
            {"time": simulation_time, "phase": env.signal_controller.current_phase, "state": env.signal_controller.signal_state}
        )
        return {
            "phase": env.signal_controller.current_phase,
            "phase_name": PHASE_NAMES[env.signal_controller.current_phase],
            "phase_elapsed": env.signal_controller.phase_elapsed,
            "signal_state": env.signal_controller.signal_state,
            "vehicles": vehicles,
            "metrics": metrics,
        }

    @staticmethod
    def _network_bounds(env: IntersectionEnv) -> list[list[float]]:
        lower, upper = env.connection.simulation.getNetBoundary()
        return [[float(lower[0]), float(lower[1])], [float(upper[0]), float(upper[1])]]

    @staticmethod
    def _delta(fixed: dict[str, Any], dqn: dict[str, Any]) -> dict[str, float | None]:
        output: dict[str, float | None] = {}
        for name in ("vehicles_remaining", "current_queue", "average_waiting", "maximum_waiting", "throughput", "phase_changes"):
            baseline = float(fixed[name])
            output[name] = None if baseline == 0 else (float(dqn[name]) - baseline) / baseline * 100.0
        return output

    @staticmethod
    def _update_milestones(
        milestones: dict[str, float | None], side: dict[str, Any], simulation_time: float, initial_count: int
    ) -> None:
        if initial_count <= 0:
            return
        cleared = 100.0 - side["metrics"]["vehicles_remaining"] / initial_count * 100.0
        for threshold in (50, 90, 100):
            if milestones[str(threshold)] is None and cleared >= threshold:
                milestones[str(threshold)] = simulation_time

    @staticmethod
    def _timeseries_row(time_value: float, fixed: dict[str, Any], dqn: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {"simulation_time": time_value}
        for key, side in (("fixed", fixed), ("dqn", dqn)):
            for metric, value in side["metrics"].items():
                if metric != "approach_queues":
                    row[f"{key}_{metric}"] = value
            row[f"{key}_phase"] = side["phase"]
            row[f"{key}_signal_state"] = side["signal_state"]
        return row

    def _build_result(
        self,
        milestone_times: dict[str, dict[str, float | None]],
        initial_count: int | None,
        wall_seconds: float,
    ) -> dict[str, Any]:
        assert self.fixed_env is not None and self.dqn_env is not None
        fixed = self.fixed_env.episode_summary(0)
        dqn = self.dqn_env.episode_summary(0)
        for key, summary in (("fixed", fixed), ("dqn", dqn)):
            summary["clearance_time"] = milestone_times[key]["100"]
            summary["time_to_50_clearance"] = milestone_times[key]["50"]
            summary["time_to_90_clearance"] = milestone_times[key]["90"]
            summary["queue_auc"] = float(sum(row[f"{key}_current_queue"] for row in self.timeseries))
            waits = list(self.waiting_history[key].values()) or [0.0]
            summary["waiting_p50"] = float(np.percentile(waits, 50))
            summary["waiting_p90"] = float(np.percentile(waits, 90))
            summary["waiting_p95"] = float(np.percentile(waits, 95))
            summary["group_waiting"] = {
                group: {
                    "average": float(np.mean(list(values.values()))) if values else 0.0,
                    "maximum": max(values.values(), default=0.0),
                    "vehicles": len(values),
                }
                for group, values in self.group_waiting_history[key].items()
            }
            summary["fairness_max_group_wait"] = max(
                (value["maximum"] for value in summary["group_waiting"].values()),
                default=0.0,
            )
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "test_mode": self.request.test_mode,
            "seed": self.request.seed,
            "requested_initial_vehicles": initial_count,
            "actual_initial_vehicles": initial_count,
            "model_path": str(self.model_path),
            "fixed": fixed,
            "dqn": dqn,
            "difference": self._summary_difference(fixed, dqn),
            "phase_timeline": self.phase_timeline,
            "wall_seconds": wall_seconds,
            "reproducibility": collect_reproducibility_metadata(self.model_path),
        }

    @staticmethod
    def _summary_difference(fixed: dict[str, Any], dqn: dict[str, Any]) -> dict[str, Any]:
        difference: dict[str, Any] = {}
        for metric in set(fixed) & set(dqn):
            if isinstance(fixed[metric], (int, float)) and isinstance(dqn[metric], (int, float)):
                raw = float(dqn[metric]) - float(fixed[metric])
                percent = None if float(fixed[metric]) == 0 else raw / float(fixed[metric]) * 100.0
                difference[metric] = {"raw": raw, "percent": percent}
        return difference

    def _persist(self, result: dict[str, Any]) -> None:
        (self.artifact_dir / "experiment_config.json").write_text(
            self.request.model_dump_json(indent=2), encoding="utf-8"
        )
        (self.artifact_dir / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        if self.timeseries:
            with (self.artifact_dir / "raw_timeseries.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.timeseries[0]))
                writer.writeheader()
                writer.writerows(self.timeseries)

    def _throttle(self) -> None:
        if self.request.speed == "max":
            return
        factor = float(self.request.speed)
        time.sleep(max(0.0, 1.0 / factor))
