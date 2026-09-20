"""Generate reproducible presentation packs from completed experiments."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid
import zipfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from api.app.database import ARTIFACTS_DIR, connect, get_row
from api.app.services.model_registry import ModelRegistry
from model.utils.config import ProjectConfig


class ReportService:
    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def generate(self, experiment_id: str) -> dict[str, Any]:
        experiment = get_row("experiments", experiment_id)
        if experiment is None:
            raise KeyError(experiment_id)
        config = json.loads(experiment["config_json"])
        result = json.loads(experiment["result_json"])
        source_dir = Path(experiment["artifact_dir"])
        report_id = uuid.uuid4().hex
        output_dir = ARTIFACTS_DIR / "reports" / f"presentation_pack_{experiment_id}_{report_id[:8]}"
        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        (output_dir / "experiment_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        model_path = config.get("model_path") or result.get("model_path")
        try:
            model_metadata = self.registry.inspect(Path(model_path)) if model_path else self.registry.list_models()[0]
        except Exception as exc:
            model_metadata = {"path": model_path, "error": str(exc)}
        (output_dir / "model_metadata.json").write_text(
            json.dumps(model_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        for filename in ("raw_timeseries.csv", "per_run_metrics.csv"):
            source = source_dir / filename
            if source.exists():
                shutil.copy2(source, output_dir / filename)
            else:
                (output_dir / filename).write_text("", encoding="utf-8")

        summary_rows = self._summary_rows(result)
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "fixed", "dqn", "difference", "improvement_percent", "sample_size"])
            writer.writeheader()
            writer.writerows(summary_rows)
        self._charts(result, source_dir, charts_dir)
        (output_dir / "report.html").write_text(
            self._html_report(experiment_id, config, summary_rows), encoding="utf-8"
        )
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in output_dir.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(output_dir))
        created = datetime.now(timezone.utc).isoformat()
        metadata = {"files": [str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()]}
        with connect() as connection:
            connection.execute(
                "INSERT INTO reports VALUES (?, ?, ?, ?, ?, ?)",
                (report_id, experiment_id, created, str(output_dir), str(zip_path), json.dumps(metadata)),
            )
        return {"id": report_id, "experiment_id": experiment_id, "artifact_dir": str(output_dir), "zip_path": str(zip_path), **metadata}

    @staticmethod
    def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if "statistics" in result:
            for metric, values in result["statistics"].items():
                fixed = values["fixed"]["mean"]
                dqn = values["dqn"]["mean"]
                rows.append({"metric": metric, "fixed": fixed, "dqn": dqn, "difference": dqn - fixed, "improvement_percent": values["improvement_percent"], "sample_size": values["fixed"]["n"]})
            return rows
        fixed = result.get("fixed", {})
        dqn = result.get("dqn", {})
        for metric in sorted(set(fixed) & set(dqn)):
            if isinstance(fixed[metric], (int, float)) and isinstance(dqn[metric], (int, float)):
                difference = float(dqn[metric]) - float(fixed[metric])
                improvement = None if float(fixed[metric]) == 0 else -difference / float(fixed[metric]) * 100.0
                rows.append({"metric": metric, "fixed": fixed[metric], "dqn": dqn[metric], "difference": difference, "improvement_percent": improvement, "sample_size": 1})
        return rows

    def _charts(self, result: dict[str, Any], source_dir: Path, charts_dir: Path) -> None:
        rows = {row["metric"]: row for row in self._summary_rows(result)}
        chart_specs = (
            ("avg_waiting_time", "Average Waiting Time", "seconds", "avg_waiting"),
            ("max_waiting_time", "Maximum Waiting Time", "seconds", "max_waiting"),
            ("avg_queue", "Average Queue", "vehicles", "avg_queue"),
            ("throughput", "Throughput", "vehicles", "throughput"),
        )
        for metric, title, unit, filename in chart_specs:
            if metric in rows:
                self._bar_chart(rows[metric], title, unit, charts_dir / filename)
        timeseries_path = source_dir / "raw_timeseries.csv"
        if timeseries_path.exists() and timeseries_path.stat().st_size:
            frame = pd.read_csv(timeseries_path)
            self._line_chart(frame, "fixed_vehicles_remaining", "dqn_vehicles_remaining", "Remaining Vehicles", "vehicles", charts_dir / "clearance_curve")
            self._phase_chart(frame, charts_dir / "phase_timeline")
        else:
            self._placeholder_chart("Clearance Curve", charts_dir / "clearance_curve")
            self._placeholder_chart("Phase Timeline", charts_dir / "phase_timeline")
        self._percentile_chart(result, charts_dir / "waiting_distribution")
        self._multi_metric_chart(rows, charts_dir / "scenario_comparison")

    @staticmethod
    def _save_figure(fig: plt.Figure, output: Path) -> None:
        fig.set_size_inches(16, 9)
        fig.tight_layout()
        fig.savefig(output.with_suffix(".png"), dpi=120)
        fig.savefig(output.with_suffix(".svg"))
        plt.close(fig)

    def _bar_chart(self, row: dict[str, Any], title: str, unit: str, output: Path) -> None:
        fig, ax = plt.subplots()
        ax.bar(["Fixed-Time", "DQN"], [row["fixed"], row["dqn"]], color=["#f5b942", "#48d597"])
        ax.set_title(f"{title} (n={row.get('sample_size', 1)})")
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        self._save_figure(fig, output)

    def _line_chart(self, frame: pd.DataFrame, fixed_col: str, dqn_col: str, title: str, unit: str, output: Path) -> None:
        fig, ax = plt.subplots()
        ax.plot(frame["simulation_time"], frame[fixed_col], label="Fixed-Time", color="#f5b942")
        ax.plot(frame["simulation_time"], frame[dqn_col], label="DQN", color="#48d597")
        ax.set_title(title)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
        ax.legend()
        self._save_figure(fig, output)

    def _phase_chart(self, frame: pd.DataFrame, output: Path) -> None:
        fig, axes = plt.subplots(2, 1, sharex=True)
        for ax, prefix, label in ((axes[0], "fixed", "Fixed-Time"), (axes[1], "dqn", "DQN")):
            ax.step(frame["simulation_time"], frame[f"{prefix}_phase"], where="post")
            ax.set_ylabel(label)
            ax.set_yticks([0, 1, 2, 3])
        axes[1].set_xlabel("Simulation time (s)")
        fig.suptitle("Phase Timeline")
        self._save_figure(fig, output)

    def _percentile_chart(self, result: dict[str, Any], output: Path) -> None:
        if "fixed" not in result:
            self._placeholder_chart("Waiting Distribution", output)
            return
        labels = ["P50", "P90", "P95", "Max"]
        fixed = [result["fixed"].get("waiting_p50", 0), result["fixed"].get("waiting_p90", 0), result["fixed"].get("waiting_p95", 0), result["fixed"].get("max_waiting_time", 0)]
        dqn = [result["dqn"].get("waiting_p50", 0), result["dqn"].get("waiting_p90", 0), result["dqn"].get("waiting_p95", 0), result["dqn"].get("max_waiting_time", 0)]
        x = range(len(labels))
        fig, ax = plt.subplots()
        ax.bar([index - 0.2 for index in x], fixed, width=0.4, label="Fixed-Time", color="#f5b942")
        ax.bar([index + 0.2 for index in x], dqn, width=0.4, label="DQN", color="#48d597")
        ax.set_xticks(list(x), labels)
        ax.set_ylabel("seconds")
        ax.set_title("Vehicle Waiting Percentiles")
        ax.legend()
        self._save_figure(fig, output)

    def _multi_metric_chart(self, rows: dict[str, dict[str, Any]], output: Path) -> None:
        selected = [metric for metric in ("avg_waiting_time", "avg_queue", "throughput") if metric in rows]
        if not selected:
            self._placeholder_chart("Scenario Comparison", output)
            return
        fig, ax = plt.subplots()
        x = range(len(selected))
        ax.bar([index - 0.2 for index in x], [rows[m]["fixed"] for m in selected], 0.4, label="Fixed-Time", color="#f5b942")
        ax.bar([index + 0.2 for index in x], [rows[m]["dqn"] for m in selected], 0.4, label="DQN", color="#48d597")
        ax.set_xticks(list(x), selected)
        ax.set_title("Experiment Metric Comparison")
        ax.set_ylim(bottom=0)
        ax.legend()
        self._save_figure(fig, output)

    def _placeholder_chart(self, title: str, output: Path) -> None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not available for this experiment mode", ha="center", va="center")
        ax.set_title(title)
        ax.axis("off")
        self._save_figure(fig, output)

    @staticmethod
    def _html_report(experiment_id: str, config: dict[str, Any], rows: list[dict[str, Any]]) -> str:
        project = ProjectConfig()
        result_rows = "".join(
            f"<tr><td>{row['metric']}</td><td>{row['fixed']}</td><td>{row['dqn']}</td><td>{row['difference']}</td></tr>"
            for row in rows
        )
        return f"""<!doctype html><html><head><meta charset='utf-8'><title>Traffic Control Experiment</title>
<style>body{{font-family:Arial;margin:40px;color:#152238}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border:1px solid #ccd4df}}img{{max-width:100%}}</style></head><body>
<h1>Traffic Control Experiment {experiment_id}</h1>
<h2>Table 1 — Environment</h2><table><tr><th>Min Green</th><th>Max Green</th><th>Yellow</th><th>All Red</th><th>Decision Interval</th></tr>
<tr><td>{project.signal.min_green}</td><td>{project.signal.max_green}</td><td>{project.signal.yellow}</td><td>{project.signal.all_red}</td><td>{project.signal.decision_interval}</td></tr></table>
<p>Episode duration: {config.get('duration', project.simulation.episode_seconds)} seconds</p>
<h2>Table 2 — DQN Configuration</h2><pre>{json.dumps(asdict(project.dqn), indent=2)}</pre>
<h2>Table 3 — Test Conditions</h2><pre>{json.dumps(config, indent=2)}</pre>
<h2>Table 4 — Results</h2><table><tr><th>Metric</th><th>Fixed</th><th>DQN</th><th>DQN - Fixed</th></tr>{result_rows}</table>
<h2>Charts</h2><img src='charts/avg_waiting.png'><img src='charts/clearance_curve.png'><img src='charts/phase_timeline.png'>
</body></html>"""
