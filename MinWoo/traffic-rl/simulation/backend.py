"""TraCI lifetime and sensing boundary. The policy never imports TraCI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import subprocess

import traci
import traci.constants as tc
from sumolib.miscutils import getFreeSocketPort

from simulation.binaries import find_binary, sumo_environment
from utils.config import Config, DIRECTIONS, ROOT


@dataclass(frozen=True)
class VehicleReading:
    lane: str
    speed: float
    waiting: float
    accumulated_waiting: float
    lane_position: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    time: float
    lane_counts: dict[str, int]
    lane_queues: dict[str, int]
    vehicles: dict[str, VehicleReading]
    departed: tuple[str, ...]
    arrived: tuple[str, ...]
    pending: int
    collisions: int
    teleports: int
    lane_lengths: dict[str, float] = field(default_factory=dict)


class SUMOBackend:
    def __init__(self) -> None:
        self.connection = None
        self.process: subprocess.Popen | None = None
        self._log = None
        self.lane_lengths: dict[str, float] = {}
        self.vehicle_variables = [tc.VAR_LANE_ID, tc.VAR_SPEED, tc.VAR_WAITING_TIME, tc.VAR_ACCUMULATED_WAITING_TIME]

    def configure_sensors(self, config: Config) -> None:
        self.vehicle_variables = [tc.VAR_LANE_ID, tc.VAR_SPEED, tc.VAR_WAITING_TIME, tc.VAR_ACCUMULATED_WAITING_TIME]
        if config.observation.include_approaching:
            self.vehicle_variables.append(tc.VAR_LANEPOSITION)
        self.lane_lengths = {}
        for direction in DIRECTIONS:
            for lane in range(3):
                lane_id = f"{direction}_in_{lane}"
                self.connection.lane.subscribe(lane_id, [tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
                if config.observation.include_approaching:
                    self.lane_lengths[lane_id] = self.connection.lane.getLength(lane_id)

    def start(self, config: Config, route_file: Path, seed: int, output_dir: Path, gui: bool = False) -> None:
        self.close()
        output_dir.mkdir(parents=True, exist_ok=True)
        port = getFreeSocketPort()
        command = [find_binary("sumo-gui" if gui else "sumo"), "-c", str(ROOT / "sumo/simulation.sumocfg"),
                   "--route-files", str(route_file), "--step-length", str(config.simulation.step_length),
                   "--end", str(config.simulation.episode_seconds), "--seed", str(seed),
                   "--waiting-time-memory", str(config.simulation.episode_seconds + 1),
                   "--tripinfo-output", str(output_dir / "tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
                   "--no-step-log", "true", "--duration-log.disable", "true", "--remote-port", str(port)]
        if gui:
            command += ["--start", "--quit-on-end"]
        try:
            self._log = (output_dir / "sumo.log").open("w", encoding="utf-8")
            self.process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT, env=sumo_environment(command[0]))
            (output_dir / "process.json").write_text(json.dumps({"pid": self.process.pid, "command": command}, indent=2), encoding="utf-8")
            self.connection = traci.connect(port=port, numRetries=20, proc=self.process)
            # TraCI 1.27 has no public timeout setter. Keep this version-specific
            # boundary here so an unresponsive GUI cannot hang cleanup forever.
            self.connection._socket.settimeout(20)
            self.configure_sensors(config)
        except BaseException:
            self.close()
            raise

    def set_signal(self, state: str) -> None:
        self.connection.trafficlight.setRedYellowGreenState("J", state)

    def step(self) -> Snapshot:
        self.connection.simulationStep()
        departed = tuple(self.connection.simulation.getDepartedIDList())
        for vehicle in departed:
            self.connection.vehicle.subscribe(vehicle, self.vehicle_variables)
        return self.snapshot(departed)

    def snapshot(self, departed: tuple[str, ...] = ()) -> Snapshot:
        conn = self.connection
        lane_data = conn.lane.getAllSubscriptionResults()
        vehicles = {vehicle: VehicleReading(values[tc.VAR_LANE_ID], values[tc.VAR_SPEED],
                    values[tc.VAR_WAITING_TIME], values[tc.VAR_ACCUMULATED_WAITING_TIME], values.get(tc.VAR_LANEPOSITION, 0.0))
                    for vehicle, values in conn.vehicle.getAllSubscriptionResults().items()}
        return Snapshot(conn.simulation.getTime(),
            {lane: values[tc.LAST_STEP_VEHICLE_NUMBER] for lane, values in lane_data.items()},
            {lane: values[tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for lane, values in lane_data.items()},
            vehicles, departed, tuple(conn.simulation.getArrivedIDList()),
            len(conn.simulation.getPendingVehicles()), conn.simulation.getCollidingVehiclesNumber(),
            conn.simulation.getStartingTeleportNumber(), self.lane_lengths)

    def close(self) -> None:
        conn, self.connection = self.connection, None
        try:
            if conn is not None:
                conn.close(wait=False)
        except (OSError, traci.exceptions.TraCIException, traci.exceptions.FatalTraCIError):
            pass
        finally:
            process, self.process = self.process, None
            if process is not None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            if self._log is not None:
                self._log.close()
                self._log = None
