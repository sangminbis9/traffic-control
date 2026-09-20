"""Reusable DQN training runner with safe checkpoint-based pause and resume."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from model.env.intersection_env import IntersectionEnv
from model.utils.config import ProjectConfig, ensure_directories
from model.utils.reproducibility import collect_reproducibility_metadata


ProgressHandler = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class TrainingOptions:
    mode: str = "fixed_steps"
    total_steps: int = 20_000
    maximum_steps: int = 1_000_000
    minimum_steps: int = 100_000
    validation_interval: int = 10_000
    validation_episodes: int = 5
    no_improvement_patience: int = 8
    minimum_improvement: float = 0.05
    checkpoint_interval: int = 10_000
    progress_interval: int = 250
    visualization_interval: int = 10
    scenario: str = "random"
    episode_seconds: int = 300
    seed: int = 1
    validation_seed_start: int = 10_001
    max_waiting_limit: float = 120.0
    throughput_retention: float = 0.90
    waiting_improvement_target: float = 10.0
    queue_improvement_target: float = 5.0
    fixed_win_throughput_retention: float = 0.95
    learning_starts: int | None = None
    resume_additional_steps: bool = False
    use_gui: bool = False

    @property
    def target_steps(self) -> int:
        return self.maximum_steps if self.mode == "auto_convergence" else self.total_steps


@dataclass(frozen=True)
class ValidationResult:
    timestep: int
    avg_waiting_time: float
    max_waiting_time: float
    avg_queue: float
    max_queue: float
    throughput: float
    phase_changes: float
    validation_reward: float
    fixed_avg_waiting_time: float
    fixed_max_waiting_time: float
    fixed_avg_queue: float
    fixed_max_queue: float
    fixed_throughput: float
    fixed_phase_changes: float
    waiting_improvement_pct: float
    max_waiting_improvement_pct: float
    queue_improvement_pct: float
    throughput_change_pct: float
    throughput_retention_pct: float
    waiting_target_met: bool
    max_waiting_target_met: bool
    queue_target_met: bool
    throughput_target_met: bool
    beats_fixed: bool
    score: float
    eligible: bool


def _mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate episode summaries into controller-level validation metrics."""
    keys = (
        "avg_waiting_time",
        "max_waiting_time",
        "avg_queue",
        "max_queue",
        "throughput",
        "phase_changes",
        "episode_reward",
    )
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def _lower_is_better_improvement(candidate: float, baseline: float) -> float:
    if baseline <= 1e-9:
        return 0.0 if candidate <= 1e-9 else -100.0
    return (baseline - candidate) / baseline * 100.0


def _higher_is_better_change(candidate: float, baseline: float) -> float:
    if baseline <= 1e-9:
        return 0.0 if candidate <= 1e-9 else 100.0
    return (candidate - baseline) / baseline * 100.0


class TrainingControlCallback(BaseCallback):
    def __init__(self, runner: "TrainingRunner") -> None:
        super().__init__(verbose=0)
        self.runner = runner
        self.episode_reward = 0.0
        self.episodes = 0
        self.last_validation_step = 0
        self.last_checkpoint_step = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is not None:
            self.episode_reward += float(np.asarray(rewards).mean())
        if dones is not None and bool(np.asarray(dones).any()):
            self.episodes += 1
            self.runner.rolling_rewards.append(self.episode_reward)
            self.runner.rolling_rewards[:] = self.runner.rolling_rewards[-100:]
            self.episode_reward = 0.0

        step = int(self.model.num_timesteps)
        if step % max(self.runner.options.progress_interval, 1) == 0:
            self.runner._record_training_progress(step, self.episodes)
        if step % max(self.runner.options.visualization_interval, 1) == 0:
            self.runner._emit_training_frame(step, self.episodes)
        if step - self.last_checkpoint_step >= self.runner.options.checkpoint_interval:
            self.runner._save_checkpoint(step)
            self.last_checkpoint_step = step
        if step - self.last_validation_step >= self.runner.options.validation_interval:
            result = self.runner._validate(step)
            self.runner._handle_validation(result)
            self.last_validation_step = step
        return not (self.runner.pause_requested.is_set() or self.runner.stop_requested.is_set())


class TrainingRunner:
    """Own one training lifecycle and persist everything needed to resume it."""

    def __init__(
        self,
        session_id: str,
        options: TrainingOptions,
        *,
        output_dir: Path | None = None,
        progress_handler: ProgressHandler | None = None,
        resume_checkpoint: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.options = options
        self.config = ProjectConfig()
        ensure_directories(self.config)
        self.output_dir = output_dir or self.config.results_dir / "training" / session_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_handler = progress_handler
        self.resume_checkpoint = resume_checkpoint
        self.pause_requested = threading.Event()
        self.stop_requested = threading.Event()
        self.status = "PENDING"
        self.started_at = 0.0
        self.model: DQN | None = None
        self.rolling_rewards: list[float] = []
        self.training_rows: list[dict[str, Any]] = []
        self.validation_rows: list[dict[str, Any]] = []
        self.checkpoint_rows: list[dict[str, Any]] = []
        self.best_validation: ValidationResult | None = None
        self.latest_validation: ValidationResult | None = None
        self.fixed_baseline: dict[str, float] | None = None
        self.training_env: IntersectionEnv | None = None
        self.no_improvement_count = 0
        self.stop_reason: str | None = None
        self._runtime_target = options.target_steps

    def request_pause(self) -> None:
        self.pause_requested.set()

    def request_stop(self) -> None:
        self.stop_requested.set()

    def _emit(self, payload: dict[str, Any]) -> None:
        message = {"type": "training_progress", "session_id": self.session_id, **payload}
        if self.progress_handler is not None:
            self.progress_handler(message)

    def _create_model(self, env: IntersectionEnv) -> DQN:
        if self.resume_checkpoint is not None:
            model = DQN.load(str(self.resume_checkpoint), env=env, device="auto")
            replay_path = self.resume_checkpoint.with_suffix(".replay.pkl")
            if replay_path.exists():
                model.load_replay_buffer(str(replay_path))
            return model
        return DQN(
            self.config.dqn.policy,
            env,
            learning_rate=self.config.dqn.learning_rate,
            buffer_size=self.config.dqn.buffer_size,
            learning_starts=(
                self.config.dqn.learning_starts
                if self.options.learning_starts is None
                else self.options.learning_starts
            ),
            batch_size=self.config.dqn.batch_size,
            gamma=self.config.dqn.gamma,
            train_freq=self.config.dqn.train_freq,
            gradient_steps=self.config.dqn.gradient_steps,
            target_update_interval=self.config.dqn.target_update_interval,
            exploration_fraction=self.config.dqn.exploration_fraction,
            exploration_final_eps=self.config.dqn.exploration_final_eps,
            seed=self.options.seed,
            policy_kwargs={"net_arch": list(self.config.dqn.network_architecture)},
            verbose=0,
        )

    def run(self) -> dict[str, Any]:
        self.started_at = time.monotonic()
        self.status = "TRAINING"
        self._write_json("training_config.json", asdict(self.options))
        env = IntersectionEnv(
            self.config,
            controller_name="DQN",
            scenario=self.options.scenario,
            seed=self.options.seed,
            episode_seconds=self.options.episode_seconds,
            use_gui=self.options.use_gui,
        )
        self.training_env = env
        try:
            self.model = self._create_model(env)
            current = int(self.model.num_timesteps)
            self._runtime_target = (
                current + self.options.total_steps
                if self.resume_checkpoint is not None and self.options.resume_additional_steps
                else self.options.target_steps
            )
            remaining = max(self._runtime_target - current, 0)
            if remaining:
                self.model.learn(
                    total_timesteps=remaining,
                    callback=TrainingControlCallback(self),
                    reset_num_timesteps=current == 0,
                    progress_bar=False,
                )
            final_step = int(self.model.num_timesteps)
            checkpoint = self._save_checkpoint(final_step, name="final")
            if self.pause_requested.is_set():
                self.status = "PAUSED"
                self.stop_reason = "pause_requested"
            elif self.stop_reason == "converged":
                self.status = "COMPLETED"
            elif self.stop_requested.is_set():
                self.status = "STOPPED"
                self.stop_reason = self.stop_reason or "stop_requested"
            else:
                self.status = "COMPLETED"
                self.stop_reason = self.stop_reason or "target_reached"
            self._flush_csv_files()
            self._write_metadata(checkpoint)
            self._emit(self.snapshot())
            return self.snapshot()
        except Exception as exc:
            self.status = "FAILED"
            self.stop_reason = str(exc)
            self._emit(self.snapshot())
            raise
        finally:
            env.close()
            self.training_env = None

    def snapshot(self) -> dict[str, Any]:
        step = int(self.model.num_timesteps) if self.model is not None else 0
        best = asdict(self.best_validation) if self.best_validation else None
        latest = asdict(self.latest_validation) if self.latest_validation else None
        return {
            "status": self.status,
            "timesteps": step,
            "total_timesteps": self._runtime_target,
            "episodes": len(self.rolling_rewards),
            "epsilon": float(getattr(self.model, "exploration_rate", 1.0)) if self.model else 1.0,
            "replay_buffer_size": self._replay_size(),
            "rolling_episode_reward": float(np.mean(self.rolling_rewards[-20:])) if self.rolling_rewards else 0.0,
            "elapsed_seconds": max(time.monotonic() - self.started_at, 0.0) if self.started_at else 0.0,
            "best_validation": best,
            "validation": latest,
            "fixed_baseline": self.fixed_baseline,
            "stop_reason": self.stop_reason,
            "current_checkpoint": self.checkpoint_rows[-1]["model_path"] if self.checkpoint_rows else None,
            "checkpoints": list(self.checkpoint_rows),
        }

    def _replay_size(self) -> int:
        replay = getattr(self.model, "replay_buffer", None)
        return int(replay.size()) if replay is not None else 0

    def _record_training_progress(self, step: int, episodes: int) -> None:
        logger_values = getattr(getattr(self.model, "logger", None), "name_to_value", {})
        row = {
            "timestep": step,
            "episode": episodes,
            "epsilon": float(getattr(self.model, "exploration_rate", 1.0)),
            "replay_buffer_size": self._replay_size(),
            "rolling_episode_reward": float(np.mean(self.rolling_rewards[-20:])) if self.rolling_rewards else 0.0,
            "loss": float(logger_values.get("train/loss", 0.0)),
            "elapsed_seconds": time.monotonic() - self.started_at,
        }
        self.training_rows.append(row)
        self._emit({**self.snapshot(), **row})

    def _emit_training_frame(self, step: int, episodes: int) -> None:
        if self.training_env is None:
            return
        try:
            frame = self.training_env.visualization_state()
        except RuntimeError:
            return
        self._emit(
            {
                "type": "training_frame",
                "status": self.status,
                "timestep": step,
                "episode": episodes,
                "scenario": self.options.scenario,
                **frame,
            }
        )

    def _evaluate_validation_controller(self, controller_name: str) -> dict[str, float]:
        rows: list[dict[str, Any]] = []
        for offset in range(self.options.validation_episodes):
            env = IntersectionEnv(
                self.config,
                controller_name=controller_name,
                scenario=self.options.scenario,
                seed=self.options.validation_seed_start + offset,
                episode_seconds=self.options.episode_seconds,
            )
            try:
                observation, _ = env.reset(seed=self.options.validation_seed_start + offset)
                terminated = truncated = False
                while not (terminated or truncated):
                    if controller_name.lower().startswith("fixed"):
                        action = 0
                    else:
                        if self.model is None:
                            raise RuntimeError("Training model is not initialized")
                        action, _ = self.model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, _ = env.step(int(action))
                rows.append(env.episode_summary(offset))
            finally:
                env.close()
        return _mean_metrics(rows)

    def _validate(self, step: int) -> ValidationResult:
        if self.model is None:
            raise RuntimeError("Training model is not initialized")
        if self.fixed_baseline is None:
            self.fixed_baseline = self._evaluate_validation_controller("Fixed-Time")
        dqn = self._evaluate_validation_controller("Validation-DQN")
        fixed = self.fixed_baseline

        waiting_improvement = _lower_is_better_improvement(
            dqn["avg_waiting_time"], fixed["avg_waiting_time"]
        )
        max_waiting_improvement = _lower_is_better_improvement(
            dqn["max_waiting_time"], fixed["max_waiting_time"]
        )
        queue_improvement = _lower_is_better_improvement(dqn["avg_queue"], fixed["avg_queue"])
        throughput_change = _higher_is_better_change(dqn["throughput"], fixed["throughput"])
        throughput_retention = (
            dqn["throughput"] / fixed["throughput"]
            if fixed["throughput"] > 1e-9
            else 1.0
        )

        waiting_target_met = waiting_improvement >= self.options.waiting_improvement_target
        max_waiting_target_met = (
            dqn["max_waiting_time"] <= fixed["max_waiting_time"]
            and dqn["max_waiting_time"] <= self.options.max_waiting_limit
        )
        queue_target_met = queue_improvement >= self.options.queue_improvement_target
        throughput_target_met = throughput_retention >= self.options.fixed_win_throughput_retention
        beats_fixed = all(
            (waiting_target_met, max_waiting_target_met, queue_target_met, throughput_target_met)
        )

        eligible = (
            dqn["max_waiting_time"] <= self.options.max_waiting_limit
            and throughput_retention >= self.options.throughput_retention
        )
        score = dqn["avg_waiting_time"] + 0.25 * dqn["max_waiting_time"] + (0.0 if eligible else 1_000.0)
        return ValidationResult(
            timestep=step,
            avg_waiting_time=dqn["avg_waiting_time"],
            max_waiting_time=dqn["max_waiting_time"],
            avg_queue=dqn["avg_queue"],
            max_queue=dqn["max_queue"],
            throughput=dqn["throughput"],
            phase_changes=dqn["phase_changes"],
            validation_reward=dqn["episode_reward"],
            fixed_avg_waiting_time=fixed["avg_waiting_time"],
            fixed_max_waiting_time=fixed["max_waiting_time"],
            fixed_avg_queue=fixed["avg_queue"],
            fixed_max_queue=fixed["max_queue"],
            fixed_throughput=fixed["throughput"],
            fixed_phase_changes=fixed["phase_changes"],
            waiting_improvement_pct=waiting_improvement,
            max_waiting_improvement_pct=max_waiting_improvement,
            queue_improvement_pct=queue_improvement,
            throughput_change_pct=throughput_change,
            throughput_retention_pct=throughput_retention * 100.0,
            waiting_target_met=waiting_target_met,
            max_waiting_target_met=max_waiting_target_met,
            queue_target_met=queue_target_met,
            throughput_target_met=throughput_target_met,
            beats_fixed=beats_fixed,
            score=score,
            eligible=eligible,
        )

    def _handle_validation(self, result: ValidationResult) -> None:
        self.latest_validation = result
        row = asdict(result)
        self.validation_rows.append(row)
        improved = (
            result.eligible
            and (
                self.best_validation is None
                or self.best_validation.score - result.score >= self.options.minimum_improvement
            )
        )
        if improved:
            self.best_validation = result
            self.no_improvement_count = 0
            self._save_checkpoint(result.timestep, name="best")
        else:
            self.no_improvement_count += 1
        self._emit({**self.snapshot(), "validation": row})
        if (
            self.options.mode == "auto_convergence"
            and result.timestep >= self.options.minimum_steps
            and self.no_improvement_count >= self.options.no_improvement_patience
        ):
            self.stop_reason = "converged"
            self.stop_requested.set()

    def _save_checkpoint(self, step: int, name: str | None = None) -> Path:
        if self.model is None:
            raise RuntimeError("Training model is not initialized")
        stem = name or f"checkpoint_{step}"
        path = self.output_dir / stem
        self.model.save(str(path))
        zip_path = path.with_suffix(".zip")
        replay_path = path.with_suffix(".replay.pkl")
        self.model.save_replay_buffer(str(replay_path))
        self.checkpoint_rows.append(
            {"timestep": step, "name": stem, "model_path": str(zip_path), "replay_buffer_path": str(replay_path)}
        )
        return zip_path

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        (self.output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )

    def _write_metadata(self, checkpoint: Path) -> None:
        self._write_json(
            "training_metadata.json",
            {
                **self.snapshot(),
                "session_id": self.session_id,
                "checkpoint": str(checkpoint),
                "reproducibility": collect_reproducibility_metadata(checkpoint),
            },
        )

    def _flush_csv_files(self) -> None:
        self._write_rows("training_metrics.csv", self.training_rows)
        self._write_rows("validation_metrics.csv", self.validation_rows)
        self._write_rows("checkpoint_metrics.csv", self.checkpoint_rows)

    def _write_rows(self, filename: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            (self.output_dir / filename).write_text("", encoding="utf-8")
            return
        with (self.output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
