"""Build and validate the four-way SUMO network with eight protected phases."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


SUMO_DIR = Path(__file__).resolve().parent


def _find_netconvert() -> str:
    candidate = shutil.which("netconvert")
    if candidate:
        return candidate
    roots = [
        Path(value)
        for value in (
            os.environ.get("SUMO_HOME"),
            r"C:\Program Files\Eclipse SUMO",
            r"C:\Program Files (x86)\Eclipse\Sumo",
            r"C:\Program Files (x86)\Eclipse SUMO",
        )
        if value
    ]
    for root in roots:
        executable = root / "bin" / "netconvert.exe"
        if executable.exists():
            os.environ["SUMO_HOME"] = str(root)
            return str(executable)
    try:
        import sumo as sumo_package

        package_home = getattr(sumo_package, "SUMO_HOME", None)
        executable = Path(package_home) / "bin" / "netconvert.exe" if package_home else None
        if executable and executable.exists():
            return str(executable)
    except ImportError:
        pass
    raise FileNotFoundError("netconvert was not found; install SUMO or set SUMO_HOME")


def validate_network(path: Path) -> None:
    root = ET.parse(path).getroot()
    links = [
        connection
        for connection in root.findall("connection")
        if connection.attrib.get("tl") == "J"
    ]
    if len(links) != 16 or {int(link.attrib["linkIndex"]) for link in links} != set(range(16)):
        raise ValueError("Network must contain 16 explicitly indexed signal links")
    expected = [("0", "r"), ("0", "s"), ("1", "s"), ("2", "l")]
    for offset, direction in enumerate(("N", "E", "S", "W")):
        direction_links = sorted(
            (link for link in links if link.attrib["from"] == f"{direction}_in"),
            key=lambda link: int(link.attrib["linkIndex"]),
        )
        actual = [(link.attrib["fromLane"], link.attrib["dir"]) for link in direction_links]
        if actual != expected:
            raise ValueError(f"Incorrect lane/link mapping for {direction}: {actual}")
        if [int(link.attrib["linkIndex"]) for link in direction_links] != list(
            range(offset * 4, offset * 4 + 4)
        ):
            raise ValueError(f"Incorrect signal-link order for {direction}")


def network_fingerprint(path: Path = SUMO_DIR / "intersection.net.xml") -> str:
    """Hash semantic XML content while ignoring netconvert comments and paths."""

    xml = ET.tostring(ET.parse(path).getroot(), encoding="unicode")
    canonical = ET.canonicalize(xml, strip_text=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_network(output: Path = SUMO_DIR / "intersection.net.xml") -> Path:
    output = output.resolve()
    command = [
        _find_netconvert(),
        "--node-files",
        str(SUMO_DIR / "nodes.nod.xml"),
        "--edge-files",
        str(SUMO_DIR / "edges.edg.xml"),
        "--connection-files",
        str(SUMO_DIR / "connections.con.xml"),
        "--tllogic-files",
        str(SUMO_DIR / "traffic_lights.add.xml"),
        "--no-turnarounds",
        "true",
        "--output-file",
        str(output),
    ]
    subprocess.run(command, check=True)
    validate_network(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SUMO_DIR / "intersection.net.xml")
    args = parser.parse_args()
    print(f"Created {build_network(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
