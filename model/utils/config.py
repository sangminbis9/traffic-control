"""Project configuration with conservative, reproducible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMO_DIR = PROJECT_ROOT / "sumo"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class SignalConfig:
    """Timing values in simulation seconds."""

    decision_interval: float = 1.0
    min_green: float = 3.0
    max_green: float = 15.0
    yellow: float = 1.0
    all_red: float = 1.0


@dataclass(frozen=True)
class RewardConfig:
    """Weights for normalized reward terms."""

    queue_weight: float = 1.0
    waiting_weight: float = 0.3
    max_waiting_weight: float = 1.0
    switch_penalty: float = 0.4
    queue_scale: float = 40.0
    waiting_scale: float = 120.0
    max_waiting_scale: float = 90.0


@dataclass(frozen=True)
class DQNConfig:
    """Stable-Baselines3 DQN defaults, exposed for short smoke tests."""

    learning_rate: float = 1e-4
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.25
    exploration_final_eps: float = 0.05
    policy: str = "MlpPolicy"
    network_architecture: Tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime settings shared by tests, training, and evaluation."""

    step_length: float = 1.0
    episode_seconds: int = 300
    warmup_seconds: int = 0
    use_gui: bool = False
    seed: int = 1
    scenario: str = "random"
    route_depart_interval: int = 1
    extra_sumo_args: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level immutable configuration."""

    project_root: Path = PROJECT_ROOT
    sumo_dir: Path = SUMO_DIR
    results_dir: Path = RESULTS_DIR
    signal: SignalConfig = field(default_factory=SignalConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    fixed_green_times: Dict[int, float] = field(
        default_factory=lambda: {0: 10.0, 1: 4.0, 2: 10.0, 3: 4.0}
    )

    @property
    def network_file(self) -> Path:
        return self.sumo_dir / "intersection.net.xml"

    @property
    def sumo_config_file(self) -> Path:
        return self.sumo_dir / "simulation.sumo.cfg"

    @property
    def default_route_file(self) -> Path:
        return self.sumo_dir / "routes.rou.xml"

    @property
    def tl_id(self) -> str:
        return "center"


def ensure_directories(config: ProjectConfig | None = None) -> None:
    """Create output directories without touching user data outside the project."""

    cfg = config or ProjectConfig()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.sumo_dir / "generated").mkdir(parents=True, exist_ok=True)


def resolve_sumo_binary(use_gui: bool = False) -> str:
    """Resolve SUMO from PATH or SUMO_HOME with a useful error message."""

    import shutil

    executable = "sumo-gui" if use_gui else "sumo"
    candidate = shutil.which(executable)
    if candidate:
        return candidate
    candidate_roots = []
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate_roots.append(Path(sumo_home))
    candidate_roots.extend(
        [
            Path(r"C:\Program Files\Eclipse SUMO"),
            Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
            Path(r"C:\Program Files (x86)\Eclipse SUMO"),
        ]
    )
    for root in candidate_roots:
        candidate_path = root / "bin" / f"{executable}.exe"
        if candidate_path.exists():
            # Ensure the child SUMO process can also locate its built-in XML/type data.
            os.environ["SUMO_HOME"] = str(root)
            return str(candidate_path)
    # The official ``eclipse-sumo`` wheel exposes its bundled installation this way.
    try:
        import sumo as sumo_package

        package_home = getattr(sumo_package, "SUMO_HOME", None)
        if package_home:
            package_binary = Path(package_home) / "bin" / f"{executable}.exe"
            if package_binary.exists():
                return str(package_binary)
    except ImportError:
        pass
    raise FileNotFoundError(
        f"Could not find {executable}. Install SUMO and add its bin directory to PATH "
        "or set SUMO_HOME."
    )
