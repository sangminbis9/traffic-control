"""Gymnasium environment backed by a single SUMO signalized intersection."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from model.controller.signal_controller import SignalController
from model.controller.fixed_controller import FixedTimeController
from model.traffic.route_generator import SCENARIOS, TrafficDemand, generate_route_file
from model.utils.config import ProjectConfig, ensure_directories, resolve_sumo_binary
from model.utils.metrics import EpisodeMetrics

from .reward import calculate_reward
from .state_provider import SUMOTrafficStateProvider, TrafficSnapshot


class IntersectionEnv(gym.Env[np.ndarray, int]):
    """Discrete four-action signal control environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: ProjectConfig | None = None,
        *,
        controller_name: str = "DQN",
        scenario: str | None = None,
        seed: int = 1,
        use_gui: bool | None = None,
        episode_seconds: int | None = None,
        route_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ProjectConfig()
        ensure_directories(self.config)
        self.controller_name = controller_name
        self.scenario = scenario or self.config.simulation.scenario
        self.base_seed = int(seed)
        self.use_gui = self.config.simulation.use_gui if use_gui is None else use_gui
        self.episode_seconds = episode_seconds or self.config.simulation.episode_seconds
        self.route_dir = route_dir or (self.config.sumo_dir / "generated")
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(12,), dtype=np.float32)
        self._connection: Any | None = None
        self._signal_controller: SignalController | None = None
        self._fixed_controller: FixedTimeController | None = None
        self._state_provider: SUMOTrafficStateProvider | None = None
        self._metrics: EpisodeMetrics | None = None
        self._previous_snapshot: TrafficSnapshot | None = None
        self._episode_seed = self.base_seed
        self._episode_step = 0
        self._episode_reward = 0.0
        self._last_route_file: Path | None = None

    @property
    def signal_controller(self) -> SignalController:
        if self._signal_controller is None:
            raise RuntimeError("Environment has not been reset")
        return self._signal_controller

    @property
    def connection(self) -> Any:
        if self._connection is None:
            raise RuntimeError("Environment has not been reset")
        return self._connection

    def _start_sumo(self, route_file: Path, seed: int) -> None:
        try:
            import traci
        except ImportError as exc:
            raise RuntimeError("TraCI is missing. Install requirements.txt first.") from exc

        binary = resolve_sumo_binary(self.use_gui)
        command = [
            binary,
            "-c",
            str(self.config.sumo_config_file),
            "--net-file",
            str(self.config.network_file),
            "--route-files",
            str(route_file),
            "--seed",
            str(seed),
            "--step-length",
            str(self.config.simulation.step_length),
            "--end",
            str(self.episode_seconds),
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--time-to-teleport",
            "-1",
        ]
        command.extend(self.config.simulation.extra_sumo_args)
        label = f"traffic_rl_{uuid.uuid4().hex}"
        try:
            # TraCI's start() returns the SUMO version tuple; the live API object
            # is retrieved from its labelled connection pool.
            traci.start(command, label=label)
            self._connection = traci.getConnection(label)
        except Exception as exc:
            self._connection = None
            raise RuntimeError(f"Failed to start SUMO with command: {' '.join(command)}") from exc

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = int(seed)
        else:
            # Training resets without an explicit seed must see varied demand.
            # Evaluation can still reproduce an episode by passing seed=... .
            self._episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        options = options or {}
        scenario = str(options.get("scenario", self.scenario))
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}")
        self.close()

        self._last_route_file = self.route_dir / f"routes_{self.controller_name.lower()}_{self._episode_seed}_{scenario}.rou.xml"
        generate_route_file(
            self._last_route_file,
            TrafficDemand(duration=self.episode_seconds, scenario=scenario, seed=self._episode_seed),
        )
        self._start_sumo(self._last_route_file, self._episode_seed)
        self._signal_controller = SignalController(self.connection, self.config.tl_id, self.config.signal)
        self._signal_controller.reset(0)
        self._fixed_controller = (
            FixedTimeController(self._signal_controller, self.config.fixed_green_times)
            if self.controller_name.lower().startswith("fixed")
            else None
        )
        self._state_provider = SUMOTrafficStateProvider(
            self.connection,
            queue_scale=self.config.reward.queue_scale,
            waiting_scale=self.config.reward.waiting_scale,
        )
        self._episode_step = 0
        self._episode_reward = 0.0
        self._previous_snapshot = self._state_provider.snapshot()
        self._metrics = EpisodeMetrics(self.controller_name, self._episode_seed, scenario)
        observation = self._state_provider.observation(0, 0.0, self.config.signal.max_green)
        info = {"seed": self._episode_seed, "scenario": scenario, "route_file": str(self._last_route_file)}
        return observation, info

    def step(self, action: int):
        if self._state_provider is None or self._previous_snapshot is None:
            raise RuntimeError("Call reset() before step()")
        if self._fixed_controller is not None:
            result = self._fixed_controller.apply_if_due()
        else:
            result = self.signal_controller.apply_action(int(action))
        for _ in range(max(1, int(round(self.config.signal.decision_interval / self.config.simulation.step_length)))):
            self.connection.simulationStep()
            self.signal_controller.advance(self.config.simulation.step_length)
        current_snapshot = self._state_provider.snapshot()
        reward = calculate_reward(
            self._previous_snapshot,
            current_snapshot,
            switched=result.switched,
            config=self.config.reward,
        )
        self._episode_reward += reward
        self._episode_step += 1
        terminated = False
        truncated = self._episode_step * self.config.signal.decision_interval >= self.episode_seconds
        observation = self._state_provider.observation(
            self.signal_controller.current_phase,
            self.signal_controller.phase_elapsed,
            self.config.signal.max_green,
        )
        if self._metrics is not None:
            self._metrics.record(current_snapshot, reward=reward, phase_changed=result.switched)
        self._previous_snapshot = current_snapshot
        info = {
            "phase": self.signal_controller.current_phase,
            "phase_elapsed": self.signal_controller.phase_elapsed,
            "total_queue": current_snapshot.total_queue,
            "total_waiting_time": current_snapshot.total_waiting_time,
            "max_waiting_time": current_snapshot.max_waiting_time,
            "throughput": current_snapshot.arrived,
            "phase_changed": result.switched,
            "ignored_action": result.ignored,
        }
        return observation, reward, terminated, truncated, info

    def episode_summary(self, episode: int | None = None) -> dict[str, Any]:
        if self._metrics is None:
            raise RuntimeError("No episode has been started")
        return self._metrics.summary(episode)

    def close(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:
                pass
            finally:
                self._connection = None
                self._signal_controller = None
                self._fixed_controller = None
                self._state_provider = None
                self._previous_snapshot = None
