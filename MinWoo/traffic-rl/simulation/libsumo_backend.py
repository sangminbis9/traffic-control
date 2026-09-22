"""Optional in-process SUMO backend. One active environment per OS process."""
from __future__ import annotations

import os
from pathlib import Path

from simulation.backend import SUMOBackend
from simulation.binaries import find_binary, sumo_environment
from utils.config import Config, ROOT


class LibSUMOBackend(SUMOBackend):
    _owner: LibSUMOBackend | None = None

    def start(self, config: Config, route_file: Path, seed: int, output_dir: Path, gui: bool = False) -> None:
        if gui:
            raise ValueError("Use the traci backend for GUI on Windows")
        if self._owner is not None and self._owner is not self:
            raise RuntimeError("libsumo allows one active environment per process; use separate processes")
        self.close()
        output_dir.mkdir(parents=True, exist_ok=True)
        child_env = sumo_environment(find_binary("sumo"))
        keys = ("SUMO_HOME", "LANG", "LC_ALL", "LC_CTYPE")
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                if key in child_env:
                    os.environ[key] = child_env[key]
                else:
                    os.environ.pop(key, None)
            import libsumo

            self.connection = libsumo
            LibSUMOBackend._owner = self
            libsumo.start(["sumo", "-c", str(ROOT / "sumo/simulation.sumocfg"),
                "--route-files", str(route_file.resolve()), "--step-length", str(config.simulation.step_length),
                "--end", str(config.simulation.episode_seconds), "--seed", str(seed),
                "--waiting-time-memory", str(config.simulation.episode_seconds + 1),
                "--tripinfo-output", str((output_dir / "tripinfo.xml").resolve()),
                "--tripinfo-output.write-unfinished", "true", "--no-step-log", "true",
                "--duration-log.disable", "true", "--log", str((output_dir / "sumo.log").resolve())])
            self.configure_sensors(config)
        except BaseException:
            self.close()
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def close(self) -> None:
        if LibSUMOBackend._owner is self:
            try:
                if self.connection is not None:
                    self.connection.close()
            finally:
                self.connection = None
                LibSUMOBackend._owner = None
