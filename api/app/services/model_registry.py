"""Filesystem model registry with reproducibility hashes and compatibility checks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN

from api.app.database import ARTIFACTS_DIR, REPOSITORY_ROOT, initialize_database, upsert_model
from model.utils.config import ProjectConfig


class ModelRegistry:
    def __init__(self) -> None:
        self.config = ProjectConfig()
        self.allowed_roots = [
            self.config.results_dir.resolve(),
            (ARTIFACTS_DIR / "training").resolve(),
        ]

    def _allowed(self, path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == root or root in resolved.parents for root in self.allowed_roots)

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def inspect(self, path: Path) -> dict[str, Any]:
        path = path.resolve()
        if not self._allowed(path) or not path.is_file() or path.suffix.lower() != ".zip":
            raise ValueError("Model path is not part of the project model registry")
        compatible = False
        error: str | None = None
        observation_shape: list[int] | None = None
        action_count: int | None = None
        timesteps = 0
        try:
            model = DQN.load(str(path), device="cpu")
            observation_shape = list(model.observation_space.shape or ())
            action_count = int(getattr(model.action_space, "n", 0))
            timesteps = int(model.num_timesteps)
            compatible = observation_shape == [12] and action_count == 4
            if not compatible:
                error = "Model/environment observation mismatch"
        except Exception as exc:
            error = str(exc)
        stat = path.stat()
        training_metadata: dict[str, Any] = {}
        for metadata_path in (path.parent / "training_metadata.json", path.with_suffix(".metadata.json")):
            if metadata_path.exists():
                try:
                    import json
                    training_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    training_metadata = {}
                break
        training_config: dict[str, Any] = {}
        config_path = path.parent / "training_config.json"
        if config_path.exists():
            try:
                import json
                training_config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                training_config = {}
        return {
            "name": path.stem,
            "path": str(path),
            "relative_path": str(path.relative_to(REPOSITORY_ROOT)),
            "sha256": self.sha256(path),
            "size_bytes": stat.st_size,
            "created_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "timesteps": timesteps,
            "observation_shape": observation_shape,
            "action_count": action_count,
            "compatible": compatible,
            "error": error,
            "training_scenario": training_config.get("scenario") or training_metadata.get("training_scenario"),
            "last_validation_score": (training_metadata.get("best_validation") or {}).get("score"),
        }

    def list_models(self) -> list[dict[str, Any]]:
        paths = set(self.config.results_dir.glob("*.zip"))
        training_root = ARTIFACTS_DIR / "training"
        if training_root.exists():
            paths.update(training_root.rglob("*.zip"))
        models = [self.inspect(path) for path in sorted(paths, key=lambda item: item.stat().st_mtime, reverse=True)]
        initialize_database()
        for metadata in models:
            upsert_model(metadata)
        return models

    def resolve(self, requested: str | None) -> Path:
        if requested is None:
            path = self.config.results_dir / "dqn_intersection.zip"
        else:
            path = Path(requested)
            if not path.is_absolute():
                path = REPOSITORY_ROOT / path
        metadata = self.inspect(path)
        if not metadata["compatible"]:
            raise ValueError(metadata["error"] or "Model is incompatible")
        return path.resolve()
