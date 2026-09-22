"""Resolve a pip SUMO installation, SUMO_HOME, or system PATH."""
import importlib.util
import os
from pathlib import Path
import shutil
import sys


def find_binary(name: str) -> str:
    filename = name + (".exe" if os.name == "nt" else "")
    candidates = [Path(sys.executable).parent / filename]
    if os.environ.get("SUMO_HOME"):
        candidates.insert(0, Path(os.environ["SUMO_HOME"]) / "bin" / filename)
    spec = importlib.util.find_spec("sumo")
    if spec and spec.origin:
        candidates.insert(0, Path(spec.origin).parent / "bin" / filename)
    for path in candidates:
        if path.is_file():
            return str(path)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found. Install requirements.txt or set SUMO_HOME to your SUMO installation.")


def sumo_environment(binary: str) -> dict[str, str]:
    env = os.environ.copy()
    env["SUMO_HOME"] = str(Path(binary).parent.parent)
    if os.name == "nt":
        # GNU C.UTF-8 from agent shells is not a supported Windows SUMO locale.
        env.update(LANG="C", LC_ALL="C", LC_CTYPE="C")
        if not env["SUMO_HOME"].isascii():
            # A relative ASCII path avoids SUMO 1.27 Windows gettext path errors.
            relative_home = os.path.relpath(env["SUMO_HOME"])
            if relative_home.isascii():
                env["SUMO_HOME"] = relative_home
            else:
                env.pop("SUMO_HOME")
    return env
