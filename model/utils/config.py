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
    max_red: float = 60.0


@dataclass(frozen=True)
class RewardConfig:
    """Fixed reward baseline; selection evidence is documented in model/README.md."""

    queue_weight: float = 1.0
    waiting_weight: float = 0.3
    max_waiting_weight: float = 0.5
    switch_penalty: float = 0.5
    queue_scale: float = 10.0
    waiting_scale: float = 6_000.0
    max_waiting_scale: float = 120.0


@dataclass(frozen=True)
class DQNConfig:
    """Stable-Baselines3 DQN defaults, exposed for short smoke tests."""

    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    learning_starts: int = 5_000
    batch_size: int = 64
    gamma: float = 0.95
    train_freq: int = 4
    gradient_steps: int = 1
    n_steps: int = 5
    exploration_hold_min: int = 5
    exploration_hold_max: int = 17
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.20
    exploration_final_eps: float = 0.01
    policy: str = "MlpPolicy"
    network_architecture: Tuple[int, ...] = (128, 128)
    torch_threads: int = 1


@dataclass(frozen=True)
class SimulationConfig:
    """Runtime settings shared by tests, training, and evaluation."""

    step_length: float = 0.5
    episode_seconds: int = 300
    demand_seconds: int = 240
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
        default_factory=lambda: {
            0: 10.0, 1: 4.0, 2: 4.0, 3: 4.0,
            4: 10.0, 5: 4.0, 6: 4.0, 7: 4.0,
        }
    )
    approaching_distance: float = 50.0
    approaching_scale: float = 10.0

    @property
    def network_file(self) -> Path:
        return self.sumo_dir / "intersection.net.xml"

    @property
    def sumo_config_file(self) -> Path:
        return self.sumo_dir / "simulation.sumocfg"

    @property
    def default_route_file(self) -> Path:
        return self.sumo_dir / "routes.rou.xml"

    @property
    def tl_id(self) -> str:
        return "J"


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
