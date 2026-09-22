"""Gymnasium wrapper with fixed one-second decision steps and safe transitions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import uuid

import gymnasium as gym
import numpy as np

from build_network import validate_network
from controller.signal_controller import SignalController
from simulation.backend import SUMOBackend
from traffic.route_generator import generate_routes
from traffic.state_provider import SUMOTrafficStateProvider
from utils.config import Config, ROOT, load_config
from utils.metrics import EpisodeMetrics, append_csv
from utils.reward import reward_terms, switching_penalty


class IntersectionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, config: Config | None = None, render_mode: str | None = None,
                 output_dir: Path | None = None, controller_name: str = "dqn", backend: str = "traci") -> None:
        super().__init__()
        self.config = deepcopy(config or load_config())
        self.config.validate()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or human")
        validate_network(ROOT / "sumo/intersection.net.xml")
        self.render_mode = render_mode
        self.output_dir = Path(output_dir or ROOT / "results/runs" / uuid.uuid4().hex[:12])
        self.controller_name = controller_name
        self.action_space = gym.spaces.Discrete(4)
        # 10 traffic features + 16 fully observable controller features.
        size = 34 if self.config.observation.include_approaching else 26
        self.observation_space = gym.spaces.Box(0, 1, shape=(size,), dtype=np.float32)
        if backend == "libsumo":
            from simulation.libsumo_backend import LibSUMOBackend
            self.backend = LibSUMOBackend()
        elif backend == "traci":
            self.backend = SUMOBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")
        self.provider = SUMOTrafficStateProvider(self.config.observation.detection_distance)
        self.episode = 0
        self.signal: SignalController | None = None
        self._finished = True

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.close()
        options = options or {}
        self.episode += 1
        training = self.config.training
        # Gym seed controls a stream of episode seeds; options pins a paired test.
        self.traffic_seed = int(options.get("traffic_seed", self.np_random.integers(training.traffic_seed_min, training.traffic_seed_max + 1)))
        self.scenario = options.get("scenario", self.config.traffic.scenario)
        episode_dir = self.output_dir / f"episode_{self.episode:05d}"
        self.route_metadata = generate_routes(episode_dir / "routes.rou.xml", self.config, self.traffic_seed, self.scenario)
        self.metrics = EpisodeMetrics()
        try:
            self.backend.start(self.config, episode_dir / "routes.rou.xml", self.traffic_seed, episode_dir, self.render_mode == "human")
            self.signal = SignalController(self.backend, self.config.signal)
            self.snapshot = self.backend.snapshot()
            self.state = self.provider.get_state(self.snapshot)
            self._finished = False
            return self._observation(), self._info()
        except BaseException:
            self.close()
            raise

    def _observation(self) -> np.ndarray:
        parts = [self.state.normalized(self.config.normalization), self.signal.features()]
        if self.config.observation.include_approaching:
            parts.append(np.clip(np.array(self.state.approaching) / self.config.observation.count_scale, 0, 1))
        return np.concatenate(parts).astype(np.float32)

    def _info(self) -> dict:
        return {"time": self.snapshot.time, "traffic_seed": self.traffic_seed, "traffic_scenario": self.scenario,
                "route_sha256": self.route_metadata["sha256"], "phase": self.signal.phase,
                "stage": self.signal.stage, "phase_elapsed": self.signal.elapsed,
                "queues": self.state.queues, "total_waiting": self.state.total_waiting,
                "max_waiting": self.state.max_waiting, "pending": self.snapshot.pending,
                "approaching": self.state.approaching}

    def step(self, action: int):
        if self._finished:
            raise RuntimeError("Call reset() before stepping a new/finished episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        try:
            before = self.signal.phase_changes
            accepted = self.signal.request(int(action))
            terms = {"queue": 0.0, "waiting": 0.0, "max_waiting": 0.0}
            sim = self.config.simulation
            remaining = round((sim.episode_seconds - self.snapshot.time) / sim.step_length)
            ticks = min(round(sim.decision_interval / sim.step_length), remaining)
            for _ in range(ticks):
                self.snapshot = self.backend.step()
                self.signal.tick(sim.step_length)
                self.state = self.provider.get_state(self.snapshot)
                self.metrics.update(self.snapshot, sim.step_length)
                if self.snapshot.collisions or self.snapshot.teleports:
                    raise RuntimeError(f"Collision/teleport at t={self.snapshot.time}; see {self.output_dir}")
                for key, value in reward_terms(self.state, self.config.normalization, self.config.reward).items():
                    terms[key] += value * sim.step_length / sim.decision_interval
            terms["switching"] = switching_penalty(self.signal.phase_changes - before, self.config.reward)
            reward = float(sum(terms.values()))
            self.metrics.reward += reward
            for key, value in terms.items():
                self.metrics.term_totals[key] += value
            truncated = self.snapshot.time + 1e-8 >= sim.episode_seconds
            info = {**self._info(), "action_accepted": accepted, "reward_terms": terms,
                    "switch_reason": self.signal.last_reason}
            if truncated:
                row = {"episode": self.episode, "controller": self.controller_name, "seed": self.traffic_seed,
                       "traffic_scenario": self.scenario, "realized_scenario": self.route_metadata["realized_scenario"],
                       **self.metrics.summary(), "phase_changes": self.signal.phase_changes,
                       "forced_changes": self.signal.forced_changes, "scheduled": self.route_metadata["scheduled"],
                       "not_inserted": self.route_metadata["scheduled"] - len(self.metrics.waits),
                       "route_sha256": self.route_metadata["sha256"]}
                info["episode_metrics"] = row
                append_csv(self.output_dir / "episodes.csv", row)
                self._finished = True
            # Time-limit truncation allows SB3 to bootstrap; no artificial early empty termination.
            return self._observation(), reward, False, bool(truncated), info
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.backend.close()
        self._finished = True
