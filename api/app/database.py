"""Small SQLite metadata store; binary artifacts remain on the filesystem."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "api" / "data"
ARTIFACTS_DIR = REPOSITORY_ROOT / "api" / "artifacts"
DATABASE_PATH = DATA_DIR / "traffic_control.db"


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS training_sessions (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                artifact_dir TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                artifact_dir TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_metadata (
                sha256 TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                artifact_dir TEXT NOT NULL,
                zip_path TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            """
        )


def upsert_training(row: dict[str, Any]) -> None:
    with connect() as connection:
        connection.execute(
            """INSERT INTO training_sessions VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status,
            updated_at=excluded.updated_at, state_json=excluded.state_json""",
            (
                row["id"], row["status"], row["created_at"], row["updated_at"],
                json.dumps(row["config"], ensure_ascii=False),
                json.dumps(row.get("state", {}), ensure_ascii=False, default=str),
                row["artifact_dir"],
            ),
        )


def upsert_experiment(row: dict[str, Any]) -> None:
    with connect() as connection:
        connection.execute(
            """INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status,
            updated_at=excluded.updated_at, result_json=excluded.result_json""",
            (
                row["id"], row["kind"], row["status"], row["created_at"], row["updated_at"],
                json.dumps(row["config"], ensure_ascii=False),
                json.dumps(row.get("result", {}), ensure_ascii=False, default=str),
                row["artifact_dir"],
            ),
        )


def upsert_model(metadata: dict[str, Any]) -> None:
    with connect() as connection:
        connection.execute(
            """INSERT INTO model_metadata VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(sha256) DO UPDATE SET path=excluded.path,
            name=excluded.name, created_at=excluded.created_at,
            metadata_json=excluded.metadata_json""",
            (
                metadata["sha256"], metadata["path"], metadata["name"],
                metadata["created_at"], json.dumps(metadata, ensure_ascii=False, default=str),
            ),
        )


def list_rows(table: str) -> list[dict[str, Any]]:
    if table not in {"training_sessions", "experiments", "reports"}:
        raise ValueError("Unsupported table")
    with connect() as connection:
        rows = connection.execute(f"SELECT * FROM {table} ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def get_row(table: str, row_id: str) -> dict[str, Any] | None:
    if table not in {"training_sessions", "experiments", "reports"}:
        raise ValueError("Unsupported table")
    with connect() as connection:
        row = connection.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)).fetchone()
    return dict(row) if row else None
