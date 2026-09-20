"""Non-blocking training lifecycle used by REST and WebSocket routes."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import threading
import uuid
from typing import Any

from api.app.database import ARTIFACTS_DIR, get_row, list_rows, upsert_training
from api.app.schemas.models import TrainingCreate
from api.app.services.events import EventHub
from model.training import TrainingOptions, TrainingRunner


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingService:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, Any]] = {}
        self.events = EventHub()
        self._lock = threading.Lock()

    def start(self, request: TrainingCreate) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        created = _now()
        output_dir = ARTIFACTS_DIR / "training" / session_id
        options = TrainingOptions(**request.model_dump())
        record = {
            "id": session_id,
            "status": "PENDING",
            "created_at": created,
            "updated_at": created,
            "config": request.model_dump(),
            "state": {},
            "history": [],
            "latest_frame": None,
            "artifact_dir": str(output_dir),
            "runner": None,
            "thread": None,
            "options": options,
        }
        with self._lock:
            self.sessions[session_id] = record
        upsert_training({
            key: value for key, value in record.items()
            if key not in {"runner", "thread", "options", "history", "latest_frame"}
        })
        self._launch(record, None)
        return self.get(session_id)

    def _launch(self, record: dict[str, Any], checkpoint: Path | None) -> None:
        runner = TrainingRunner(
            record["id"],
            record["options"],
            output_dir=Path(record["artifact_dir"]),
            progress_handler=lambda event: self._on_event(record["id"], event),
            resume_checkpoint=checkpoint,
        )
        record["runner"] = runner
        thread = threading.Thread(target=self._run, args=(record, runner), daemon=True)
        record["thread"] = thread
        thread.start()

    def _run(self, record: dict[str, Any], runner: TrainingRunner) -> None:
        try:
            state = runner.run()
        except Exception as exc:
            state = {**runner.snapshot(), "error": str(exc)}
        self._on_event(record["id"], {"type": "training_progress", **state})

    def _on_event(self, session_id: str, event: dict[str, Any]) -> None:
        if event.get("type") == "training_frame":
            with self._lock:
                record = self.sessions[session_id]
                record["latest_frame"] = event
            self.events.publish(session_id, event)
            return

        with self._lock:
            record = self.sessions[session_id]
            record["state"] = event
            if isinstance(event.get("timesteps"), int):
                record["history"].append(event)
                record["history"] = record["history"][-300:]
            record["status"] = event.get("status", record["status"])
            record["updated_at"] = _now()
            serializable = {
                key: value for key, value in record.items()
                if key not in {"runner", "thread", "options", "history", "latest_frame"}
            }
        upsert_training(serializable)
        self.events.publish(session_id, event)

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            record = self.sessions.get(session_id)
            if record:
                return {
                    "id": session_id,
                    "status": record["status"],
                    "detail": record["state"],
                    "history": list(record["history"]),
                    "frame": record["latest_frame"],
                    "config": record["config"],
                    "artifact_dir": record["artifact_dir"],
                }
        persisted = get_row("training_sessions", session_id)
        if persisted is None:
            raise KeyError(session_id)
        return self._persisted_response(persisted)

    def list(self) -> list[dict[str, Any]]:
        rows = list_rows("training_sessions")
        with self._lock:
            active_ids = set(self.sessions)
        return [
            self._persisted_response(row, interrupted=row["id"] not in active_ids)
            for row in rows
        ]

    @staticmethod
    def _persisted_response(row: dict[str, Any], *, interrupted: bool = True) -> dict[str, Any]:
        status = row["status"]
        detail = json.loads(row["state_json"])
        if interrupted and status in {"PENDING", "TRAINING", "RESUMING", "PAUSED"}:
            status = "INTERRUPTED"
            detail = {**detail, "status": status, "stop_reason": "server_restarted"}
        return {
            "id": row["id"],
            "status": status,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "config": json.loads(row["config_json"]),
            "detail": detail,
            "history": [],
            "frame": None,
            "artifact_dir": row["artifact_dir"],
        }

    def pause(self, session_id: str) -> dict[str, Any]:
        record = self.sessions.get(session_id)
        if record is None or record["runner"] is None:
            raise KeyError(session_id)
        record["runner"].request_pause()
        return self.get(session_id)

    def stop(self, session_id: str) -> dict[str, Any]:
        record = self.sessions.get(session_id)
        if record is None or record["runner"] is None:
            raise KeyError(session_id)
        record["runner"].request_stop()
        return self.get(session_id)

    def resume(self, session_id: str) -> dict[str, Any]:
        record = self.sessions.get(session_id)
        if record is None:
            raise KeyError(session_id)
        thread = record.get("thread")
        if thread is not None and thread.is_alive():
            raise RuntimeError("Training session is still running")
        checkpoint = Path(record["artifact_dir"]) / "final.zip"
        if not checkpoint.exists():
            raise FileNotFoundError("Pause checkpoint is missing")
        record["status"] = "RESUMING"
        self._launch(record, checkpoint)
        return self.get(session_id)
