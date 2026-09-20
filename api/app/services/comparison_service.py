"""Comparison session orchestration, batching, persistence, and live events."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import statistics
import threading
import uuid
from typing import Any

from api.app.database import ARTIFACTS_DIR, get_row, upsert_experiment
from api.app.schemas.models import ComparisonCreate
from api.app.services.events import EventHub
from api.app.services.model_registry import ModelRegistry
from api.app.services.simulation_runner import SynchronizedComparisonRunner


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ComparisonService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.sessions: dict[str, dict[str, Any]] = {}
        self.events = EventHub()
        self._lock = threading.Lock()

    def start(self, request: ComparisonCreate) -> dict[str, Any]:
        experiment_id = uuid.uuid4().hex
        model_path = self.registry.resolve(request.model_path)
        artifact_dir = ARTIFACTS_DIR / "experiments" / experiment_id
        created = _now()
        record = {
            "id": experiment_id,
            "kind": f"{request.run_mode}_{request.test_mode}",
            "status": "PENDING",
            "created_at": created,
            "updated_at": created,
            "config": request.model_dump(),
            "result": {},
            "artifact_dir": str(artifact_dir),
            "runner": None,
            "thread": None,
            "model_path": model_path,
        }
        with self._lock:
            self.sessions[experiment_id] = record
        upsert_experiment(self._serializable(record))
        thread = threading.Thread(target=self._run, args=(record, request), daemon=True)
        record["thread"] = thread
        thread.start()
        return self.get(experiment_id)

    def _run(self, record: dict[str, Any], request: ComparisonCreate) -> None:
        self._update(record, "RUNNING", {})
        try:
            if request.run_mode == "batch":
                result = self._run_batch(record, request)
            else:
                runner = SynchronizedComparisonRunner(
                    record["id"], request, record["model_path"], Path(record["artifact_dir"]),
                    event_handler=lambda event: self.events.publish(record["id"], event),
                )
                record["runner"] = runner
                result = runner.run()
            self._update(record, result.get("status", "COMPLETED"), result)
        except Exception as exc:
            result = {"error": str(exc)}
            self._update(record, "FAILED", result)
            self.events.publish(record["id"], {"type": "comparison_error", "status": "FAILED", **result})

    def _run_batch(self, record: dict[str, Any], request: ComparisonCreate) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        run_results: list[dict[str, Any]] = []
        for index in range(request.runs):
            if record.get("stop_requested"):
                break
            run_request = request.model_copy(
                update={"seed": request.seed + index, "run_mode": "single", "speed": "max", "render_interval": 30}
            )
            run_dir = Path(record["artifact_dir"]) / f"run_{index + 1:03d}"
            runner = SynchronizedComparisonRunner(
                f"{record['id']}_{index + 1}", run_request, record["model_path"], run_dir
            )
            record["runner"] = runner
            result = runner.run()
            run_results.append(result)
            for controller in ("fixed", "dqn"):
                rows.append({"run": index + 1, "seed": run_request.seed, "controller": controller, **result[controller]})
            self.events.publish(
                record["id"],
                {"type": "batch_progress", "completed_runs": index + 1, "total_runs": request.runs, "latest": result},
            )
        statistics_result = self._batch_statistics(run_results)
        artifact_dir = Path(record["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        if rows:
            with (artifact_dir / "per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return {
            "experiment_id": record["id"],
            "status": "STOPPED" if record.get("stop_requested") else "COMPLETED",
            "runs_completed": len(run_results),
            "runs_requested": request.runs,
            "statistics": statistics_result,
            "run_results": run_results,
        }

    @staticmethod
    def _describe(values: list[float]) -> dict[str, float]:
        n = len(values)
        mean = statistics.fmean(values) if values else 0.0
        std = statistics.stdev(values) if n > 1 else 0.0
        margin = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
        return {
            "n": n,
            "mean": mean,
            "std": std,
            "median": statistics.median(values) if values else 0.0,
            "min": min(values, default=0.0),
            "max": max(values, default=0.0),
            "ci95_low": mean - margin,
            "ci95_high": mean + margin,
        }

    def _batch_statistics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = (
            "avg_waiting_time", "max_waiting_time", "avg_queue", "max_queue",
            "throughput", "phase_changes", "clearance_time", "queue_auc",
        )
        output: dict[str, Any] = {}
        for metric in metrics:
            fixed_values = [float(item["fixed"][metric]) for item in results if item["fixed"].get(metric) is not None]
            dqn_values = [float(item["dqn"][metric]) for item in results if item["dqn"].get(metric) is not None]
            paired = [dqn - fixed for fixed, dqn in zip(fixed_values, dqn_values, strict=False)]
            fixed_stats = self._describe(fixed_values)
            dqn_stats = self._describe(dqn_values)
            paired_stats = self._describe(paired)
            improvement = None if fixed_stats["mean"] == 0 else -paired_stats["mean"] / fixed_stats["mean"] * 100.0
            output[metric] = {"fixed": fixed_stats, "dqn": dqn_stats, "paired_difference": paired_stats, "improvement_percent": improvement}
        return output

    def _update(self, record: dict[str, Any], status: str, result: dict[str, Any]) -> None:
        with self._lock:
            record["status"] = status
            record["result"] = result
            record["updated_at"] = _now()
        upsert_experiment(self._serializable(record))

    @staticmethod
    def _serializable(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key not in {"runner", "thread", "model_path", "stop_requested"}}

    def get(self, experiment_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.sessions.get(experiment_id)
            if record:
                return {
                    "id": experiment_id,
                    "status": record["status"],
                    "detail": record["result"],
                    "config": record["config"],
                    "artifact_dir": record["artifact_dir"],
                }
        persisted = get_row("experiments", experiment_id)
        if persisted is None:
            raise KeyError(experiment_id)
        return persisted

    def pause(self, experiment_id: str) -> dict[str, Any]:
        record = self.sessions.get(experiment_id)
        if record is None or record.get("runner") is None:
            raise KeyError(experiment_id)
        record["runner"].pause()
        self._update(record, "PAUSED", record["result"])
        return self.get(experiment_id)

    def resume(self, experiment_id: str) -> dict[str, Any]:
        record = self.sessions.get(experiment_id)
        if record is None or record.get("runner") is None:
            raise KeyError(experiment_id)
        record["runner"].resume()
        self._update(record, "RUNNING", record["result"])
        return self.get(experiment_id)

    def stop(self, experiment_id: str) -> dict[str, Any]:
        record = self.sessions.get(experiment_id)
        if record is None:
            raise KeyError(experiment_id)
        record["stop_requested"] = True
        if record.get("runner") is not None:
            record["runner"].stop()
        return self.get(experiment_id)

    def set_speed(self, experiment_id: str, speed: str) -> dict[str, Any]:
        record = self.sessions.get(experiment_id)
        if record is None or record.get("runner") is None:
            raise KeyError(experiment_id)
        record["runner"].set_speed(speed)
        record["config"]["speed"] = speed
        record["updated_at"] = _now()
        upsert_experiment(self._serializable(record))
        return self.get(experiment_id)
