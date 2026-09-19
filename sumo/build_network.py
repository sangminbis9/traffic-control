"""Build the binary SUMO network from the checked-in XML source files."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


SUMO_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUMO_DIR / "intersection.net.xml")
    args = parser.parse_args()

    netconvert = shutil.which("netconvert")
    candidate_roots = []
    if os.environ.get("SUMO_HOME"):
        candidate_roots.append(Path(os.environ["SUMO_HOME"]))
    candidate_roots.extend(
        [
            Path(r"C:\Program Files\Eclipse SUMO"),
            Path(r"C:\Program Files (x86)\Eclipse\Sumo"),
            Path(r"C:\Program Files (x86)\Eclipse SUMO"),
        ]
    )
    if not netconvert:
        for root in candidate_roots:
            candidate = root / "bin" / "netconvert.exe"
            if candidate.exists():
                netconvert = str(candidate)
                os.environ["SUMO_HOME"] = str(root)
                break
    if not netconvert:
        try:
            import sumo as sumo_package

            package_home = getattr(sumo_package, "SUMO_HOME", None)
            if package_home:
                candidate = Path(package_home) / "bin" / "netconvert.exe"
                if candidate.exists():
                    netconvert = str(candidate)
        except ImportError:
            pass
    if not netconvert:
        raise SystemExit(
            "netconvert was not found. Install SUMO and add SUMO/bin to PATH, "
            "then run: python sumo/build_network.py"
        )

    command = [
        netconvert,
        "--node-files",
        str(SUMO_DIR / "nodes.nod.xml"),
        "--edge-files",
        str(SUMO_DIR / "edges.edg.xml"),
        "--connection-files",
        str(SUMO_DIR / "connections.con.xml"),
        "--tllogic-files",
        str(SUMO_DIR / "traffic_lights.add.xml"),
        "--output-file",
        str(args.output),
        "--no-turnarounds",
        "--junctions.corner-detail",
        "5",
        "--default.junctions.radius",
        "20",
        "--tls.default-type",
        "static",
    ]
    subprocess.run(command, check=True)
    print(f"Created {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
