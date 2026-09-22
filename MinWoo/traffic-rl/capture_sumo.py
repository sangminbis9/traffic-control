"""Capture genuine SUMO GUI frames while the saved DQN controls an episode."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from PIL import Image
import torch
from stable_baselines3 import DQN

from env.intersection_env import IntersectionEnv
from utils.config import ROOT, SCENARIOS, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "results/models/dqn.zip")
    parser.add_argument("--scenario", choices=SCENARIOS, default="ns_heavy")
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zoom", type=float, default=320)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh output directory to preserve earlier recordings")
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    config = load_config(args.model.with_suffix(".config.json"))
    config.save(output / "config.json")
    model = DQN.load(args.model, device="cpu")
    env = IntersectionEnv(config, render_mode="human", output_dir=output / "simulation")
    records = []
    try:
        obs, info = env.reset(seed=args.seed, options={"traffic_seed": args.seed, "scenario": args.scenario})
        gui = env.backend.connection.gui
        gui.setZoom("View #0", args.zoom)
        x, y = env.backend.connection.junction.getPosition("J")
        gui.setOffset("View #0", x, y)
        while True:
            index = len(records)
            frame = frames_dir / f"frame_{index:04d}.png"
            # SUMO renders the requested screenshot on the next simulationStep.
            capture_time = info["time"] + config.simulation.step_length
            gui.screenshot("View #0", os.path.relpath(frame), 960, 720)
            action = int(model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, info = env.step(action)
            # Allow the GUI render thread to flush before scheduling another image.
            deadline = time.monotonic() + 5
            while not frame.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            if not frame.exists():
                raise RuntimeError(f"SUMO did not render {frame.name}")
            time.sleep(0.08)
            records.append({"file": str(frame.relative_to(output)), "capture_time_seconds": capture_time,
                            "decision_action": action, "following_observation_time": info["time"],
                            "phase": info["phase"], "stage": info["stage"], "queue": sum(info["queues"])})
            if len(records) % 30 == 0:
                print(f"Captured {len(records)} frames, simulation t={info['time']:.0f}s", flush=True)
            if terminated or truncated:
                metrics = info["episode_metrics"]
                break
    finally:
        env.close()
    images = []
    try:
        for record in records:
            with Image.open(output / record["file"]) as source:
                images.append(source.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        frame_duration = round(config.simulation.decision_interval * 1000 / 5)
        images[0].save(output / "dqn_sumo_5x.gif", save_all=True, append_images=images[1:],
                       duration=frame_duration, loop=0, optimize=False, disposal=2)
    finally:
        for image in images:
            image.close()
    for second in (60, 120, 180):
        record = min(records, key=lambda row: abs(row["capture_time_seconds"] - second))
        shutil.copy2(output / record["file"], output / f"screenshot_{second}s.png")
    manifest = {"source": "actual sumo-gui screenshot API, unmodified PNG frames",
                "model": str(args.model.resolve()), "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
                "scenario": args.scenario, "traffic_seed": args.seed, "speedup": 5,
                "resolution": [960, 720], "frame_duration_ms": frame_duration,
                "simulation_step": config.simulation.step_length,
                "note": "PNG capture occurs on the first simulation substep; phase/stage fields describe the following full decision observation.",
                "metrics": metrics, "frames": records}
    (output / "capture_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"gif": str(output / "dqn_sumo_5x.gif"), "frames": len(records), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
