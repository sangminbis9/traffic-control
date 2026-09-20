"""Collect lightweight runtime metadata for reproducible experiments."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
from typing import Any

from model.utils.config import ProjectConfig, resolve_sumo_binary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=5, check=False
        )
        output = (completed.stdout or completed.stderr).strip().splitlines()
        return output[0] if output else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect_reproducibility_metadata(model_path: Path | None = None) -> dict[str, Any]:
    config = ProjectConfig()
    try:
        sumo_binary = resolve_sumo_binary(False)
        sumo_version = _command_output([sumo_binary, '--version'])
    except FileNotFoundError:
        sumo_binary = None
        sumo_version = None
    git_sha = _command_output(['git', 'rev-parse', 'HEAD'])
    model = model_path.resolve() if model_path is not None else None
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit_sha': git_sha,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'sumo_version': sumo_version,
        'sumo_binary': sumo_binary,
        'model_path': str(model) if model else None,
        'model_sha256': file_sha256(model) if model and model.exists() else None,
        'signal_config': asdict(config.signal),
        'fixed_green_times': config.fixed_green_times,
        'reward_config': asdict(config.reward),
        'dqn_config': asdict(config.dqn),
        'simulation_config': asdict(config.simulation),
    }
