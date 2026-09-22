"""Audit the completed frozen experiment and TraCI reproduction artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from build_network import network_fingerprint
from utils.config import ROOT


def main():
    output = ROOT / "results/heldout_approach"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == json.loads((ROOT / "experiments/frozen_selection.json").read_text(encoding="utf-8"))
    assert network_fingerprint() == manifest["network_sha256"]
    fixed = pd.read_csv(output / "fixed/episodes.csv")
    models = {}
    warning_rows = []
    keys = ["traffic_scenario", "seed"]
    for name in ["fixed"] + [f"dqn_s{run['training_seed']}" for run in manifest["models"]]:
        frame = pd.read_csv(output / name / "episodes.csv")
        assert len(frame) == 180 and not frame.duplicated(keys).any()
        assert set(frame.seed) == set(range(4001, 4031))
        assert (frame[["collisions", "teleports"]] == 0).all().all()
        pairs = frame.merge(fixed, on=keys, suffixes=("_dqn", "_fixed"), validate="one_to_one")
        assert (pairs.route_sha256_dqn == pairs.route_sha256_fixed).all()
        models[name] = frame
        counts = {"controller": name, "episodes": len(frame), "emergency_braking": 0, "emergency_stop": 0}
        logs = list((output / name / "simulation").rglob("sumo.log"))
        assert len(logs) == 180
        for log in logs:
            content = log.read_text(encoding="utf-8", errors="replace")
            counts["emergency_braking"] += content.count("performs emergency braking")
            counts["emergency_stop"] += content.count("performs emergency stop")
        warning_rows.append(counts)
    for run in manifest["models"]:
        assert hashlib.sha256((ROOT / run["model"]).read_bytes()).hexdigest() == run["model_sha256"]
    default = next(run for run in manifest["models"] if run["training_seed"] == manifest["default_model_training_seed"])
    assert hashlib.sha256((ROOT / "results/models/dqn.zip").read_bytes()).hexdigest() == default["model_sha256"]
    traci = pd.read_csv(ROOT / "results/traci_final_check/episodes.csv")
    compared = []
    for name, source in [("fixed", models["fixed"]), ("dqn", models["dqn_s22"])]:
        reference = source.loc[source.seed == 4001].sort_values(keys).reset_index(drop=True)
        actual = traci.loc[traci.controller == name].sort_values(keys).reset_index(drop=True)
        columns = [col for col in reference.select_dtypes(include="number").columns if col != "episode"]
        pd.testing.assert_frame_equal(actual[columns], reference[columns], check_exact=True)
        assert actual.route_sha256.equals(reference.route_sha256)
        compared.append({"controller": name, "cases": len(actual), "numeric_columns": columns, "exact_match": True})
    pd.DataFrame(warning_rows).to_csv(output / "analysis/warnings.csv", index=False)
    metrics = ["avg_waiting_time", "avg_queue", "throughput", "max_waiting_time", "not_inserted"]
    scenario_rows = []
    dqn = pd.concat([frame for name, frame in models.items() if name != "fixed"])
    for scenario, group in fixed.groupby("traffic_scenario"):
        agent = dqn[dqn.traffic_scenario == scenario]
        row = {"scenario": scenario}
        for metric in metrics:
            row[f"{metric}_fixed"] = float(group[metric].mean())
            row[f"{metric}_dqn"] = float(agent[metric].mean())
            row[f"{metric}_change_pct"] = 100 * (row[f"{metric}_dqn"] / row[f"{metric}_fixed"] - 1) if row[f"{metric}_fixed"] else None
        scenario_rows.append(row)
    pd.DataFrame(scenario_rows).to_csv(output / "analysis/scenario_comparison.csv", index=False)
    audit = {"episodes": 720, "model_hashes_match": True, "paired_routes_match": True,
             "collisions": 0, "teleports": 0, "traci_parity": compared, "warnings": warning_rows}
    (output / "analysis/audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    print(pd.DataFrame(scenario_rows).to_string(index=False))


if __name__ == "__main__":
    main()
