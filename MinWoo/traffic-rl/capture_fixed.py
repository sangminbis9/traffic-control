"""Record the first 60 simulation seconds of Fixed-Time control at 1x speed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from PIL import Image

from controller.fixed_controller import FixedTimeController
from env.intersection_env import IntersectionEnv
from simulation.backend import SUMOBackend
from utils.config import ROOT, load_config


class RecordingBackend(SUMOBackend):
    def __init__(self, output: Path, step_length: float):
        super().__init__()
        self.output = output
        self.step_length = step_length
        self.frames = []

    def step(self):
        capture_time = self.connection.simulation.getTime() + self.step_length
        path = None
        if capture_time <= 60 + 1e-8:
            path = self.output / "frames" / f"frame_{len(self.frames):04d}.png"
            self.connection.gui.screenshot("View #0", os.path.relpath(path), 960, 720)
        snapshot = super().step()
        if path is not None:
            deadline = time.monotonic() + 5
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not path.exists():
                raise RuntimeError(f"SUMO did not render {path.name}")
            time.sleep(0.08)
            self.frames.append({"file": str(path.relative_to(self.output)), "time_seconds": snapshot.time,
                                "signal": self.connection.trafficlight.getRedYellowGreenState("J")})
        return snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory")
    (output / "frames").mkdir(parents=True)
    # Same configuration and demand as the previous DQN recording; only control changes.
    config = load_config(ROOT / "results/models/dqn.config.json")
    config.save(output / "config.json")
    env = IntersectionEnv(config, render_mode="human", output_dir=output / "simulation", controller_name="fixed")
    backend = RecordingBackend(output, config.simulation.step_length)
    env.backend = backend
    fixed = FixedTimeController(config.signal.fixed_green)
    try:
        _, info = env.reset(seed=4001, options={"traffic_seed": 4001, "scenario": "ns_heavy"})
        gui = backend.connection.gui
        gui.setZoom("View #0", 320)
        gui.setOffset("View #0", *backend.connection.junction.getPosition("J"))
        while True:
            _, _, terminated, truncated, info = env.step(fixed.action(env.signal))
            if info["time"] % 30 == 0:
                print(f"t={info['time']:.0f}s, captured={len(backend.frames)}", flush=True)
            if terminated or truncated:
                metrics = info["episode_metrics"]
                break
    finally:
        env.close()
    images = []
    duration_ms = round(config.simulation.step_length * 1000)
    try:
        for record in backend.frames:
            with Image.open(output / record["file"]) as image:
                images.append(image.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        images[0].save(output / "fixed_sumo_1x_60s.gif", save_all=True, append_images=images[1:],
                       duration=duration_ms, loop=0, optimize=False, disposal=2)
    finally:
        for image in images:
            image.close()
    manifest = {"source": "actual SUMO GUI frames", "controller": "fixed", "speedup": 1,
                "clip_start_seconds": 0, "clip_end_seconds": 60, "playback_seconds": 60,
                "frame_duration_ms": duration_ms, "scenario": "ns_heavy", "traffic_seed": 4001,
                "fixed_green_seconds": config.signal.fixed_green,
                "yellow_seconds": config.signal.yellow, "all_red_seconds": config.signal.all_red,
                "note": "Full original 300s episode was run to retain identical demand; GIF covers only the first 60s. Metrics describe the full episode.",
                "metrics": metrics, "frames": backend.frames}
    (output / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(str(output / "fixed_sumo_1x_60s.gif"), flush=True)


if __name__ == "__main__":
    main()
