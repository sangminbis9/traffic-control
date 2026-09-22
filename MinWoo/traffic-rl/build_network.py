"""Build all plain XML and a validated SUMO network without NetEdit."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from controller.phases import ALL_RED, GREEN_STATES, LINK_DIRECTIONS, YELLOW_STATES
from simulation.binaries import find_binary, sumo_environment
from utils.config import DIRECTIONS, ROOT, Config, load_config

DESTINATIONS = {"N": ("E", "S", "W"), "S": ("W", "N", "E"),
                "E": ("S", "W", "N"), "W": ("N", "E", "S")}


def write_xml(root: ET.Element, path: Path) -> None:
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build_network(config: Config, folder: Path = ROOT / "sumo") -> Path:
    config.validate()
    folder.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="J", x="0", y="0", type="traffic_light")
    coordinates = {"N": (0, 250), "S": (0, -250), "E": (250, 0), "W": (-250, 0)}
    edges, connections = ET.Element("edges"), ET.Element("connections")
    for index, direction in enumerate(LINK_DIRECTIONS):
        x, y = coordinates[direction]
        ET.SubElement(nodes, "node", id=direction, x=str(x), y=str(y), type="priority")
        for suffix, origin, destination in [("in", direction, "J"), ("out", "J", direction)]:
            ET.SubElement(edges, "edge", {"id": f"{direction}_{suffix}", "from": origin, "to": destination,
                                         "numLanes": "3", "speed": "11.11", "priority": "1"})
        left, straight, right = DESTINATIONS[direction]
        for offset, (lane, target, to_lane) in enumerate([(0, right, 0), (0, straight, 0), (1, straight, 1), (2, left, 2)]):
            ET.SubElement(connections, "connection", {"from": f"{direction}_in", "to": f"{target}_out",
                "fromLane": str(lane), "toLane": str(to_lane), "tl": "J", "linkIndex": str(index * 4 + offset)})
    logic = ET.Element("additional")
    tls = ET.SubElement(logic, "tlLogic", id="J", type="static", programID="fixed", offset="0")
    for phase in range(4):
        for duration, state in [(config.signal.fixed_green[phase], GREEN_STATES[phase]),
                                (config.signal.yellow, YELLOW_STATES[phase]), (config.signal.all_red, ALL_RED)]:
            ET.SubElement(tls, "phase", duration=str(duration), state=state)
    for tree, filename in [(nodes, "intersection.nod.xml"), (edges, "intersection.edg.xml"),
                           (connections, "intersection.con.xml"), (logic, "intersection.tll.xml")]:
        write_xml(tree, folder / filename)
    net = folder / "intersection.net.xml"
    binary = find_binary("netconvert")
    result = subprocess.run([binary, "--node-files", str(folder / "intersection.nod.xml"),
                    "--edge-files", str(folder / "intersection.edg.xml"), "--connection-files", str(folder / "intersection.con.xml"),
                    "--tllogic-files", str(folder / "intersection.tll.xml"), "--no-turnarounds", "true",
                    "--output-file", str(net)], capture_output=True, text=True, env=sumo_environment(binary))
    if result.returncode:
        raise RuntimeError(f"netconvert failed: {result.stdout}\n{result.stderr}")
    validate_network(net)
    return net


def validate_network(net: Path) -> None:
    root = ET.parse(net).getroot()
    links = [c for c in root.findall("connection") if c.get("tl") == "J"]
    if len(links) != 16 or {int(c.attrib["linkIndex"]) for c in links} != set(range(16)):
        raise ValueError("Network must have 16 explicitly indexed traffic light links")
    for index, direction in enumerate(LINK_DIRECTIONS):
        edge = root.find(f"edge[@id='{direction}_in']")
        if edge is None or len(edge.findall("lane")) != 3:
            raise ValueError(f"Expected three incoming lanes for {direction}")
        for c in (c for c in links if c.get("from") == f"{direction}_in"):
            offset = int(c.attrib["linkIndex"]) - index * 4
            expected = [("0", "r"), ("0", "s"), ("1", "s"), ("2", "l")][offset]
            if (c.attrib["fromLane"], c.attrib["dir"]) != expected:
                raise ValueError(f"Wrong link/lane mapping: {c.attrib}")


def network_fingerprint(net: Path = ROOT / "sumo/intersection.net.xml") -> str:
    """Ignore netconvert timestamp/path comments; hash actual network content."""
    xml = ET.tostring(ET.parse(net).getroot(), encoding="unicode")
    return hashlib.sha256(ET.canonicalize(xml, strip_text=True).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    print(build_network(load_config(args.config)))
