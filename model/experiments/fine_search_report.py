"""Render completed fine-search evidence as a Korean report and seven figures.

This module never trains, selects, or changes a recommendation. Statistical
estimates come from ``fine_search_analysis`` output; CSV rows provide the
per-training-seed points and an independent check on the reported means.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


STAGES = {
    "fine": ("fine_search_results.csv", "fine_search_analysis.json"),
    "multiseed": ("multiseed_results.csv", "multiseed_analysis.json"),
    "holdout": ("final_holdout_results.csv", "final_holdout_analysis.json"),
    "robustness": ("robustness_results.csv", "robustness_analysis.json"),
}
METRICS = (
    "avg_waiting_time", "max_waiting_time", "avg_queue", "phase_changes", "throughput",
)
LABELS = {
    "avg_waiting_time": ("Mean accumulated waiting", "seconds", "평균 누적 대기시간(초)"),
    "max_waiting_time": ("Mean episode maximum waiting", "seconds", "회차 최대 대기시간의 평균(초)"),
    "avg_queue": ("Mean total queue", "vehicles", "평균 총 대기열(대)"),
    "phase_changes": ("Actual phase changes", "changes / episode", "실제 신호 전환(회/회차)"),
    "throughput": ("Throughput", "arrivals / episode", "처리량(대/회차)"),
}
SCENARIO_LABELS = {
    "random": "Random\n(final holdout)",
    "uniform": "Uniform",
    "low": "Low",
    "north_south_congested": "North / south\ncongested",
    "east_west_congested": "East / west\ncongested",
    "left_turn_congested": "Left-turn\ncongested",
    "heavy": "Heavy",
}
FIGURE_NAMES = (
    "01_final_five_metrics", "02_fine_search_waiting", "03_max_waiting_weight",
    "04_switch_weight", "05_waiting_weight", "06_training_seed_distribution",
    "07_scenario_robustness",
)
INCUMBENT_COLOR = "#334155"
CANDIDATE_COLOR = "#008477"
OTHER_COLOR = "#a3b1c2"
REGRESSION_COLOR = "#bb3947"
INTEGRITY_AUDIT_PATH = Path("audit/retraining_integrity/audit_result.json")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _id(value: Any) -> str:
    return str(value)


def _id_sort(value: Any) -> tuple[int, int | str]:
    candidate = _id(value)
    return (0, int(candidate)) if candidate.isdecimal() else (1, candidate)


def _number(row: dict, key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite report input: {key}={value}")
    return value


def _weights(row: dict) -> tuple[float, float, float]:
    return tuple(_number(row, key) for key in ("alpha", "beta", "switch_gamma"))


def _weight_text(row: dict) -> str:
    return "(" + ", ".join(f"{value:g}" for value in _weights(row)) + ")"


def _key(row: dict) -> tuple[str, int, str]:
    return _id(row["candidate_id"]), int(row["training_steps"]), str(row["scenario"])


@dataclass(frozen=True)
class ReportInputs:
    root: Path
    config: dict
    shortlist: dict
    selection: dict
    recommendation: dict
    raw: dict[str, list[dict]]
    analysis: dict[str, dict]
    source_files: tuple[Path, ...]
    integrity_audit: dict | None = None

    @property
    def incumbent_id(self) -> str:
        return _id(self.config["incumbent_id"])

    @property
    def selected_id(self) -> str:
        return _id(self.selection["selected_candidate_id"])


def _matching_rows(rows: list[dict], candidate_id: str, scenario: str = "random") -> list[dict]:
    return [row for row in rows
            if _id(row["candidate_id"]) == candidate_id and row["scenario"] == scenario]


def _one(rows: list[dict], candidate_id: str, scenario: str = "random") -> dict:
    matches = _matching_rows(rows, candidate_id, scenario)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {candidate_id}/{scenario} summary, found {len(matches)}")
    return matches[0]


def _check_analysis_means(rows: list[dict], analysis: dict, filename: str) -> None:
    """Refuse stale summaries without independently reproducing inference."""
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault(_key(row), []).append(row)
    aggregates = analysis["aggregates"]
    if {_key(row) for row in aggregates} != set(grouped) or len(aggregates) != len(grouped):
        raise ValueError(f"Raw CSV and analysis groups differ: {filename}")
    for aggregate in aggregates:
        episodes = grouped[_key(aggregate)]
        if int(aggregate["episode_count"]) != len(episodes):
            raise ValueError(f"Raw CSV and analysis row counts differ: {filename}")
        for metric in METRICS:
            actual_mean = float(np.mean([_number(row, metric) for row in episodes]))
            if not math.isclose(actual_mean, _number(aggregate, metric), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"Raw CSV and analysis means differ: {filename}/{metric}")
            low = _number(aggregate, metric + "_ci95_low")
            high = _number(aggregate, metric + "_ci95_high")
            if low > high:
                raise ValueError(f"Reversed confidence interval: {filename}/{metric}")


def load_inputs(root: Path) -> ReportInputs:
    """Load complete evidence, preserving the runner's locked decision."""
    root = root.resolve()
    fixed_files = tuple(root / name for name in (
        "status.json", "search_config.json", "shortlist_lock.json",
        "selection_lock.json", "final_recommendation.json",
    ))
    stage_files = tuple(root / name for names in STAGES.values() for name in names)
    missing = [path.name for path in (*fixed_files, *stage_files) if not path.is_file()]
    if missing:
        raise ValueError("Final report requires completed evidence; missing: " + ", ".join(missing))
    status, config, shortlist, selection, recommendation = map(_read_json, fixed_files)
    if status.get("status") != "COMPLETED":
        raise ValueError("Final report is withheld until status.json records COMPLETED")
    incumbent_id = _id(config["incumbent_id"])
    selected_id = _id(selection["selected_candidate_id"])
    if (selection.get("locked_before_holdout") is not True
            or selection.get("locked_before_robustness") is not True
            or selection.get("holdout_reselection_permitted") is not False):
        raise ValueError("Selection lock must precede both tests and prohibit reselection")
    if _id(recommendation["selected_candidate_id"]) != selected_id:
        raise ValueError("Final recommendation changed the holdout-locked candidate")
    confirmed = recommendation["confirmed"]
    if not isinstance(confirmed, bool):
        raise ValueError("Recommendation confirmed must be a JSON boolean")
    expected_recommendation = selected_id if confirmed else incumbent_id
    if _id(recommendation["recommended_candidate_id"]) != expected_recommendation:
        raise ValueError("Failed confirmation must retain the incumbent without reselection")

    raw, analysis = {}, {}
    for stage, (csv_name, json_name) in STAGES.items():
        raw[stage] = _read_csv(root / csv_name)
        analysis[stage] = _read_json(root / json_name)
        expected_split = "validation" if stage in ("fine", "multiseed") else stage
        if analysis[stage].get("split") != expected_split:
            raise ValueError(f"Unexpected analysis split: {json_name}")
        if _id(analysis[stage]["incumbent_candidate_id"]) != incumbent_id:
            raise ValueError(f"Unexpected analysis incumbent: {json_name}")
        if (analysis[stage]["inference"].get("ci_method")
                != "two_way_crossed_cluster_percentile_bootstrap"):
            raise ValueError(f"Report requires the agreed crossed bootstrap analysis: {json_name}")
        _check_analysis_means(raw[stage], analysis[stage], csv_name)
    for stage in ("fine", "multiseed", "holdout"):
        _one(analysis[stage]["aggregates"], incumbent_id)
        _one(analysis[stage]["aggregates"], selected_id)
    for stage in ("holdout", "robustness"):
        if analysis[stage].get("selection"):
            raise ValueError(f"Test analysis must not select a replacement candidate: {stage}")
    audit_file = root / INTEGRITY_AUDIT_PATH
    integrity_audit = _read_json(audit_file) if audit_file.is_file() else None
    if integrity_audit is not None and not isinstance(integrity_audit, dict):
        raise ValueError("Optional integrity audit must be a JSON object")
    optional_files = (audit_file,) if audit_file.is_file() else ()
    return ReportInputs(root, config, shortlist, selection, recommendation,
                        raw, analysis, (*fixed_files, *stage_files, *optional_files), integrity_audit)


def _configure_matplotlib(output: Path) -> None:
    # Do not require a writable user home in desktop/sandbox environments.
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _interval(ax: Any, positions: list[float], lows: list[float], highs: list[float],
              *, color: str = INCUMBENT_COLOR, horizontal: bool = False) -> None:
    # Draw endpoints directly: percentile CIs need not contain the point estimate.
    if horizontal:
        ax.hlines(positions, lows, highs, color=color, linewidth=1.3, zorder=4)
        ax.plot(lows, positions, "|", color=color, markersize=6, zorder=4)
        ax.plot(highs, positions, "|", color=color, markersize=6, zorder=4)
    else:
        ax.vlines(positions, lows, highs, color=color, linewidth=1.3, zorder=4)
        ax.plot(positions, lows, "_", color=color, markersize=7, zorder=4)
        ax.plot(positions, highs, "_", color=color, markersize=7, zorder=4)


def _save(fig: Any, output: Path, name: str) -> list[Path]:
    import matplotlib.pyplot as plt
    paths = [output / (name + suffix) for suffix in (".png", ".pdf")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def _ci_note(analysis: dict) -> str:
    counts = {int(row["training_seed_count"]) for row in analysis["aggregates"]}
    if counts == {1}:
        return "95% bootstrap CI over traffic seeds; one training seed"
    return "95% paired two-way cluster bootstrap CI; training and traffic seeds resampled"


def _plot_final(data: ReportInputs, output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    incumbent = _one(data.analysis["holdout"]["aggregates"], data.incumbent_id)
    selected = _one(data.analysis["holdout"]["aggregates"], data.selected_id)
    rows = [incumbent, selected] if data.incumbent_id != data.selected_id else [incumbent]
    colors = [INCUMBENT_COLOR, CANDIDATE_COLOR][:len(rows)]
    labels = ["Incumbent", "Locked candidate"][:len(rows)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.3), layout="constrained")
    for ax, metric in zip(axes.flat, METRICS):
        positions = list(range(len(rows)))
        means = [_number(row, metric) for row in rows]
        upper = [_number(row, metric + "_ci95_high") for row in rows]
        ax.bar(positions, means, color=colors, width=0.55)
        _interval(ax, positions, [_number(row, metric + "_ci95_low") for row in rows],
                  upper)
        ax.set_xticks(positions, labels)
        ax.set_title(LABELS[metric][0])
        ax.set_ylabel(LABELS[metric][1])
        top = max([*upper, *means, 1.0])
        ax.set_ylim(0, top * 1.16)
        for position, mean, high in zip(positions, means, upper):
            ax.text(position, max(mean, high) + top * 0.025, f"{mean:.2f}",
                    ha="center", va="bottom", fontsize=9)
    axes.flat[-1].axis("off")
    incumbent_retained = data.selected_id == data.incumbent_id
    verdict = ("No new improving candidate;\nincumbent retained" if incumbent_retained else
               "Confirmation passed" if data.recommendation["confirmed"] else
               "Confirmation failed: retain incumbent")
    candidate_description = ("Validation retained the incumbent." if incumbent_retained else
                             f"Locked C{data.selected_id}: {_weight_text(selected)}")
    axes.flat[-1].text(0.0, 0.85,
        f"Incumbent C{data.incumbent_id}: {_weight_text(incumbent)}\n"
        f"{candidate_description}\n\n"
        f"{verdict}\n\n"
        f"{incumbent['training_seed_count']} training seeds x "
        f"{incumbent['evaluation_seed_count']} paired traffic seeds\n"
        "Final random-traffic holdout\nWeights: (waiting, maximum waiting, switching)\n"
        "Error bars: 95% cluster bootstrap CI\n"
        "Individual bar overlap is not a paired test.", va="top", fontsize=10, linespacing=1.7)
    fig.suptitle("Final incumbent reference | no new candidate selected" if incumbent_retained else
                 "Final locked-candidate comparison | five operational metrics", fontsize=15)
    return _save(fig, output, FIGURE_NAMES[0])


def _plot_fine(data: ReportInputs, output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    rows = sorted(data.analysis["fine"]["aggregates"],
                  key=lambda row: (_number(row, "avg_waiting_time"), _id_sort(row["candidate_id"])))
    fig, ax = plt.subplots(figsize=(12.2, max(8, len(rows) * 0.33)), layout="constrained")
    positions = list(range(len(rows)))
    colors = [INCUMBENT_COLOR if _id(row["candidate_id"]) == data.incumbent_id else
              CANDIDATE_COLOR if _id(row["candidate_id"]) == data.selected_id else OTHER_COLOR
              for row in rows]
    ax.barh(positions, [_number(row, "avg_waiting_time") for row in rows], color=colors, height=0.68)
    _interval(ax, positions, [_number(row, "avg_waiting_time_ci95_low") for row in rows],
              [_number(row, "avg_waiting_time_ci95_high") for row in rows], horizontal=True)
    def label(row: dict) -> str:
        role = (" [incumbent]" if _id(row["candidate_id"]) == data.incumbent_id else
                " [locked]" if _id(row["candidate_id"]) == data.selected_id else "")
        return f"C{row['candidate_id']}  {_weight_text(row)}{role}"
    ax.set_yticks(positions, [label(row) for row in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel("Mean accumulated waiting (seconds); lower is better")
    ax.set_title("Fine search: all completed candidates\n"
                 "Labels: (waiting, maximum waiting, switching) | " + _ci_note(data.analysis["fine"]), fontsize=12)
    ax.grid(axis="y", visible=False)
    return _save(fig, output, FIGURE_NAMES[1])


def _plot_slice(data: ReportInputs, output: Path, axis_key: str, name: str) -> list[Path]:
    import matplotlib.pyplot as plt
    incumbent = _one(data.analysis["fine"]["aggregates"], data.incumbent_id)
    axes_keys = ("alpha", "beta", "switch_gamma")
    fixed = [key for key in axes_keys if key != axis_key]
    rows = [row for row in data.analysis["fine"]["aggregates"]
            if all(math.isclose(_number(row, key), _number(incumbent, key), abs_tol=1e-12)
                   for key in fixed)]
    rows.sort(key=lambda row: _number(row, axis_key))
    if len(rows) < 2:
        raise ValueError(f"Conditional slice has fewer than two observed weights: {axis_key}")
    x = [_number(row, axis_key) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    weight_labels = {"alpha": "Waiting weight (alpha)", "beta": "Maximum-waiting weight (beta)",
                     "switch_gamma": "Switching penalty weight"}
    for ax, metric in zip(axes, ("avg_waiting_time", "max_waiting_time", "throughput")):
        ax.plot(x, [_number(row, metric) for row in rows], "o-", color=CANDIDATE_COLOR, linewidth=1.6)
        _interval(ax, x, [_number(row, metric + "_ci95_low") for row in rows],
                  [_number(row, metric + "_ci95_high") for row in rows], color=CANDIDATE_COLOR)
        ax.axvline(_number(incumbent, axis_key), color=INCUMBENT_COLOR, linestyle="--", linewidth=1,
                   label="Incumbent weight")
        ax.set_xticks(x)
        ax.set_xlabel(weight_labels[axis_key])
        ax.set_ylabel(LABELS[metric][1])
        ax.set_title(LABELS[metric][0])
        ax.margins(x=0.15, y=0.15)
    axes[0].legend(frameon=False, fontsize=9)
    fixed_text = ", ".join(f"{key}={_number(incumbent, key):g}" for key in fixed)
    fig.suptitle(f"Conditional weight slice | fixed {fixed_text}\n"
                 "Other-weight combinations are excluded; this is not an averaged independent effect.\n"
                 + _ci_note(data.analysis["fine"]), fontsize=11)
    return _save(fig, output, name)


def _plot_training_seeds(data: ReportInputs, output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    aggregates = sorted(data.analysis["multiseed"]["aggregates"], key=lambda row: _id_sort(row["candidate_id"]))
    seeds = sorted({int(row["training_seed"]) for row in data.raw["multiseed"]})
    markers = ("o", "s", "^", "D", "v", "P", "X")
    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.9), layout="constrained")
    positions = list(range(len(aggregates)))
    offsets = np.linspace(-0.16, 0.16, len(seeds))
    for ax, metric in zip(axes.flat, METRICS):
        means = [_number(row, metric) for row in aggregates]
        deviations = [_number(row, metric + "_training_seed_std") for row in aggregates]
        _interval(ax, positions, [mean - sd for mean, sd in zip(means, deviations)],
                  [mean + sd for mean, sd in zip(means, deviations)])
        ax.plot(positions, means, "_", color="black", markersize=13, markeredgewidth=2,
                label="Across-training-seed mean")
        for index, seed in enumerate(seeds):
            values = []
            for aggregate in aggregates:
                rows = [row for row in data.raw["multiseed"]
                        if _key(row) == _key(aggregate) and int(row["training_seed"]) == seed]
                if not rows:
                    raise ValueError(f"Missing training seed {seed} for C{aggregate['candidate_id']}")
                values.append(float(np.mean([_number(row, metric) for row in rows])))
            ax.scatter(np.asarray(positions) + offsets[index], values, s=45,
                       marker=markers[index % len(markers)], label=f"Training seed {seed}", zorder=5)
        ax.set_xticks(positions, [f"C{row['candidate_id']}" +
                                 ("\nincumbent" if _id(row["candidate_id"]) == data.incumbent_id else "")
                                 for row in aggregates])
        ax.set_title(LABELS[metric][0])
        ax.set_ylabel(LABELS[metric][1])
        ax.margins(x=0.15, y=0.17)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="upper left", frameon=False)
    axes.flat[-1].text(0.0, 0.4,
        "Each point is a training-seed mean\nover its paired validation episodes.\n\n"
        "Error bars: mean +/- sample SD of\ntraining-seed means (not a 95% CI).\n\n"
        "Only three seeds: uncertainty about\ntraining variability remains substantial.", va="top", linespacing=1.5)
    fig.suptitle("150k validation | variation between independent training runs", fontsize=15)
    return _save(fig, output, FIGURE_NAMES[5])


def _paired(data: ReportInputs, stage: str, scenario: str) -> dict:
    return _one(data.analysis[stage]["paired_comparisons"], data.selected_id, scenario)


def _plot_incumbent_scenarios(data: ReportInputs, scenarios: list[str], output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    summaries = [_one(data.analysis["holdout"]["aggregates"], data.incumbent_id, "random")]
    summaries += [_one(data.analysis["robustness"]["aggregates"], data.incumbent_id, scenario)
                  for scenario in scenarios]
    labels = [SCENARIO_LABELS[scenario] for scenario in ("random", *scenarios)]
    positions = list(range(len(summaries)))
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.1), layout="constrained", sharex=True)
    for ax, metric in zip(axes, ("avg_waiting_time", "max_waiting_time", "throughput")):
        estimates = [_number(row, metric) for row in summaries]
        lows = [_number(row, metric + "_ci95_low") for row in summaries]
        highs = [_number(row, metric + "_ci95_high") for row in summaries]
        _interval(ax, positions, lows, highs, color=INCUMBENT_COLOR)
        ax.scatter(positions, estimates, color=INCUMBENT_COLOR, s=50, zorder=5)
        for position, estimate, high in zip(positions, estimates, highs):
            ax.annotate(f"{estimate:.2f}", (position, max(estimate, high)),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0.5, color=OTHER_COLOR, linestyle=":")
        ax.set_ylabel(LABELS[metric][1])
        ax.set_title(LABELS[metric][0], loc="left")
        ax.margins(x=0.06, y=0.23)
        ax.set_ylim(bottom=0)
    axes[-1].set_xticks(positions, labels, fontsize=9)
    fig.suptitle(f"Incumbent C{data.incumbent_id} scenario performance | reference only\n"
                 "Absolute means and 95% bootstrap CI | random holdout and six separate robustness scenarios\n"
                 "No new candidate selected; these values do not establish comparative superiority.", fontsize=13)
    return _save(fig, output, FIGURE_NAMES[6])


def _plot_scenarios(data: ReportInputs, output: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    scenarios = [scenario for scenario in SCENARIO_LABELS if scenario != "random"]
    observed = {row["scenario"] for row in data.analysis["robustness"]["aggregates"]}
    if set(scenarios) != observed:
        raise ValueError("Robustness evidence must cover exactly all six prespecified scenarios")
    if data.selected_id == data.incumbent_id:
        return _plot_incumbent_scenarios(data, scenarios, output)
    comparisons = [_paired(data, "holdout", "random")]
    comparisons += [_paired(data, "robustness", scenario) for scenario in scenarios]
    labels = [SCENARIO_LABELS[scenario] for scenario in ("random", *scenarios)]
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 10.1), layout="constrained", sharex=True)
    fields = (
        ("avg_waiting_time", "_difference", "Mean waiting difference (seconds); below zero is better"),
        ("max_waiting_time", "_difference", "Mean episode-maximum waiting difference (seconds)"),
        ("throughput", "_change_pct", "Throughput change (%); above zero is better"),
    )
    for ax, (metric, suffix, title) in zip(axes, fields):
        positions = list(range(len(comparisons)))
        estimates = [_number(row, metric + suffix) for row in comparisons]
        lows = [_number(row, metric + suffix + "_ci95_low") for row in comparisons]
        highs = [_number(row, metric + suffix + "_ci95_high") for row in comparisons]
        for position, estimate, low, high in zip(positions, estimates, lows, highs):
            better = estimate >= 0 if metric == "throughput" else estimate <= 0
            color = CANDIDATE_COLOR if better else REGRESSION_COLOR
            _interval(ax, [position], [low], [high], color=color)
            ax.scatter([position], [estimate], color=color, s=50, zorder=5)
        ax.axhline(0, color=INCUMBENT_COLOR, linewidth=1)
        ax.axvline(0.5, color=OTHER_COLOR, linestyle=":")
        if metric == "throughput":
            ax.axhline(-5, color=REGRESSION_COLOR, linestyle="--", linewidth=0.8,
                       label="95% retention: descriptive reference")
            ax.legend(frameon=False, fontsize=9)
        ax.set_ylabel("%" if metric == "throughput" else "seconds")
        ax.set_title(title, loc="left")
        ax.margins(x=0.05, y=0.18)
    axes[-1].set_xticks(list(range(len(labels))), labels, fontsize=9)
    fig.suptitle(f"Locked C{data.selected_id} minus incumbent C{data.incumbent_id}\n"
                 "Random holdout and six separate robustness scenarios | paired 95% bootstrap CI\n"
                 "These scenario contrasts are not multiplicity-adjusted; no pooled across-scenario claim.", fontsize=13)
    return _save(fig, output, FIGURE_NAMES[6])


def render_figures(data: ReportInputs, output: Path) -> list[Path]:
    _configure_matplotlib(output)
    paths = _plot_final(data, output) + _plot_fine(data, output)
    for key, name in (("beta", FIGURE_NAMES[2]), ("switch_gamma", FIGURE_NAMES[3]),
                      ("alpha", FIGURE_NAMES[4])):
        paths.extend(_plot_slice(data, output, key, name))
    paths.extend(_plot_training_seeds(data, output))
    paths.extend(_plot_scenarios(data, output))
    return paths


def _format_interval(row: dict, key: str, *, signed: bool = False) -> str:
    spec = "+.3f" if signed else ".3f"
    if row.get(key) is None:
        return "산출 불가(기준 분모가 0)"
    if row.get(key + "_ci95_low") is None or row.get(key + "_ci95_high") is None:
        return f"{format(_number(row, key), spec)} [CI 산출 불가]"
    return (f"{format(_number(row, key), spec)} "
            f"[95% CI {format(_number(row, key + '_ci95_low'), spec)}, "
            f"{format(_number(row, key + '_ci95_high'), spec)}]")


def _integrity_audit_lines(data: ReportInputs) -> list[str]:
    audit = data.integrity_audit
    lines = ["", "실행 복구 및 추가 재학습 무결성 감사:"]
    if audit is None:
        return lines + ["무결성 감사 자료가 제공되지 않았습니다. 추가 재학습 검증 통과를 주장하지 않습니다.",
                        f"선택 입력 경로: {INTEGRITY_AUDIT_PATH.as_posix()}"]

    comparisons = audit.get("comparisons", [])
    completed = audit.get("integrity_completed_models")
    passed = (audit.get("status") == "PASS" and audit.get("all_equal") is True
              and completed == 5 and isinstance(comparisons, list)
              and len(comparisons) == 5
              and all(isinstance(row, dict) and row.get("all_equal") is True
                      for row in comparisons)
              and len({(str(row.get("candidate_id")), str(row.get("training_seed")))
                       for row in comparisons}) == 5)
    comparison_models = sum(len({(_id(row["candidate_id"]), int(row["training_seed"]),
                                  int(row["training_steps"])) for row in data.raw[stage]})
                            for stage in ("fine", "multiseed"))
    lines += [f"감사 원본: {INTEGRITY_AUDIT_PATH.as_posix()}",
              f"감사 기록 상태: {audit.get('status', '미기록')}; all_equal={audit.get('all_equal', '미기록')}",
              ("독립 재학습 무결성 감사 통과: 명시한 5개 모델의 재현 비교가 모두 일치했습니다."
               if passed else "무결성 감사 통과를 확인하지 못했습니다. 미완료 또는 불일치 기록을 확인해야 합니다."),
              f"성과 비교에 사용한 완료 모델: {comparison_models}개; 감사 계획의 비교 모델 수: "
              f"{audit.get('expected_comparison_models', '미기록')}개.",
              f"추가 재학습 및 검증 평가를 완료한 감사 모델: {completed if completed is not None else '미기록'}개."]
    if isinstance(completed, int) and not isinstance(completed, bool) and completed >= 0:
        lines.append(f"성공적으로 완료한 학습 모델의 총 연산 수: 비교 {comparison_models}개 + "
                     f"독립 재학습 검증 {completed}개 = {comparison_models + completed}개.")
    failed = audit.get("failed_attempt_count")
    if isinstance(failed, int) and not isinstance(failed, bool) and failed >= 0:
        lines.append(f"실패 후 폐기한 미완료 시도: {failed}건 "
                     f"(본 실험 {audit.get('main_failed_attempt_count', '미기록')}건, "
                     f"감사 {audit.get('integrity_failed_attempt_count', '미기록')}건). "
                     "이는 성공 모델 수에 포함하지 않으며 추가 연산은 발생했습니다.")
    else:
        lines.append("실패·폐기 시도 수가 기록되지 않았습니다.")
    lines += ["감사 모델은 원본 모델을 대체하지 않으며 후보 선택·성능 집계·신뢰구간의 표본에 추가하지 않았습니다.",
              "동일 seed로 반복한 재현 검증을 새 학습 seed 또는 추가 독립 통계 표본으로 세지 않습니다.",
              "비교 범위: Q/Q-target 네트워크, optimizer와 학습 상태, 중간 checkpoint, "
              "학습·평가 수요, 평가 원시 지표와 행, 학습 진행 step.",
              "감사 사유: " + json.dumps(audit.get("rationale", "미기록"), ensure_ascii=False),
              "복구 및 감사 참고사항: " + json.dumps(audit.get("notes", []), ensure_ascii=False)]
    if isinstance(comparisons, list):
        for row in comparisons:
            if not isinstance(row, dict):
                continue
            checkpoints = [checkpoint for checkpoint in row.get("checkpoints", [])
                           if isinstance(checkpoint, dict)]
            matching = sum(checkpoint.get("all_equal") is True for checkpoint in checkpoints)
            lines.append(f"  감사 C{row.get('candidate_id', '?')}/seed {row.get('training_seed', '?')}: "
                         f"전체 일치={row.get('all_equal', '미기록')}, checkpoint {matching}/{len(checkpoints)} 일치.")
            component_counts = []
            for component in ("q_net", "q_net_target", "optimizer", "training_state"):
                values = [checkpoint.get("components", {}).get(component) for checkpoint in checkpoints]
                verified = sum((value.get("all_equal") if isinstance(value, dict) else value) is True
                               for value in values)
                component_counts.append(f"{component} {verified}/{len(checkpoints)}")
            lines.append("    상태 일치: " + ", ".join(component_counts) + ".")
            lines.append("    수요 일치(학습/평가)="
                         f"{row.get('training_demand_equal', '미기록')}/{row.get('evaluation_demand_equal', '미기록')}; "
                         "평가 일치(원시 지표/행)="
                         f"{row.get('raw_evaluation_metrics_equal', '미기록')}/{row.get('evaluation_rows_equal', '미기록')}; "
                         f"학습 step 일치={row.get('training_progress_steps_equal', '미기록')}.")
        lines.append("개별 checkpoint 이름, 텐서별 대조와 파일 해시는 감사 원본 JSON에 보존했습니다.")
    return lines


def build_korean_text(data: ReportInputs) -> str:
    incumbent = _one(data.analysis["holdout"]["aggregates"], data.incumbent_id)
    candidate = _one(data.analysis["holdout"]["aggregates"], data.selected_id)
    confirmed = data.recommendation["confirmed"]
    incumbent_retained = data.selected_id == data.incumbent_id
    lines = [
        "Incumbent 중심 보상 가중치 정밀 탐색 — 실제 완료 결과 보고서", "",
        (f"새 개선 후보 없음: 검증 단계에서 incumbent C{data.incumbent_id} 자체를 유지하기로 고정했습니다."
         if incumbent_retained else f"고정한 최종후보 C{data.selected_id}가 채택 조건을 통과했습니다."
         if confirmed else f"최종후보의 채택 조건을 충족하지 못하여 incumbent C{data.incumbent_id}를 유지합니다."),
        f"Incumbent C{data.incumbent_id}: {_weight_text(incumbent)}",
        f"미사용 평가 전 고정된 후보 C{data.selected_id}: {_weight_text(candidate)}",
        f"최종 추천 ID: {data.recommendation['recommended_candidate_id']}",
        f"판정 사유: {data.recommendation.get('reason', 'final_recommendation.json 참조')}",
        f"실패 조건: {json.dumps(data.recommendation.get('failed_checks', []), ensure_ascii=False)}",
        "confirmed는 새로운 후보의 평균 개선과 사전 고정한 성능·안전 조건을 모두 충족했는지를 뜻하며, 통계적 유의성 선언과 다릅니다.",
        "가중치 순서는 (waiting, maximum waiting, switching)입니다.",
        "Queue 가중치 1.0, 정규화 queue/10·total_waiting/6000·max_waiting/120, 60입력/8행동과 기존 DQN 학습 방식을 유지했습니다.",
        "학습은 random 교통·300초 회차·240초 수요이며, 50k도 원래 150k 탐색률 일정을 사용합니다.",
        "추천은 이 탐색 범위와 학습량에 한정하며, 전역 최적값이나 임의 교통 상황에서의 우월성을 뜻하지 않습니다.",
        "원래 보상 기본값, 원래 모델 및 보존된 baseline 결과는 이 보고서가 변경하지 않습니다.", "",
        "실제 사용한 보상 계산식:",
        "c(x) = min(max(x, 0), 1), q_i,h = 0.5초 tick h의 이동 그룹 i 정지 차량 수(8개 그룹).",
        "W_h와 M_h는 현재 모든 진입 차로 차량의 accumulated_waiting 합계와 최대값입니다. 움직이는 차량도 포함합니다.",
        "r_h = -(1/8) * sum_i c(q_i,h / 10) - alpha * c(W_h / 6000) - beta * c(M_h / 120).",
        "1초 결정 구간의 reward = sum_h (0.5 / 1.0) * r_h - switch_gamma * 실제 phase_changes 증가량.",
        "즉 정상적인 1초 구간에서는 두 tick의 교통 벌점을 평균하고 실제 신호 전환 횟수만큼 전환 벌점을 차감합니다.",
        "alpha는 waiting, beta는 max_waiting, switch_gamma는 전환 벌점 계수입니다. DQN 할인율 gamma=0.95와는 별개입니다.",
        "관측값은 그룹별 대기를 사용하지만 보상의 W_h와 M_h는 기존 전역 진입 차량 집계를 유지합니다.",
        "구현 근거: model/env/reward.py의 reward_terms·switching_penalty와 model/env/intersection_env.py의 step.", "",
        "1. 단계별 실제 자료", "",
    ]
    for key in ("fine_model_count", "multiseed_model_count", "total_trained_models"):
        if key in data.recommendation:
            lines.append(f"{key}: {data.recommendation[key]}")
    for stage, (csv_name, _) in STAGES.items():
        rows = data.raw[stage]
        candidates = sorted({_id(row["candidate_id"]) for row in rows}, key=_id_sort)
        training_seeds = sorted({int(row["training_seed"]) for row in rows})
        evaluation_seeds = sorted({int(row["evaluation_seed"]) for row in rows})
        steps = sorted({int(row["training_steps"]) for row in rows})
        scenarios = sorted({row["scenario"] for row in rows})
        lines.append(f"{stage}: 후보 {len(candidates)}개, 학습량 {steps}, 학습 seed {training_seeds}, "
                     f"평가 교통 seed {len(evaluation_seeds)}개, 실제 평가 행 {len(rows)}개.")
        lines.append(f"  시나리오 {scenarios}; 근거 {csv_name}")
    lines += _integrity_audit_lines(data)
    lines += ["", "실제로 평가한 전체 교통 seed:",
              "validation: " + str(sorted({int(row["evaluation_seed"]) for row in data.raw["fine"]})),
              "holdout: " + str(sorted({int(row["evaluation_seed"]) for row in data.raw["holdout"]})),
              "robustness: " + str(sorted({int(row["evaluation_seed"]) for row in data.raw["robustness"]})),
              "", "50k 후보 비교(검증 평균 대기순; 순위만으로 채택하지 않음):",
              "후보 | (waiting, max_waiting, switching) | 평균 대기 | 최대 대기 평균 | 평균 대기열 | 전환수 | 처리량 | 제약 통과"]
    for row in sorted(data.analysis["fine"]["aggregates"], key=lambda item: _number(item, "avg_waiting_time")):
        values = " | ".join(f"{_number(row, metric):.3f}" for metric in METRICS)
        lines.append(f"C{row['candidate_id']} | {_weight_text(row)} | {values} | {row.get('eligible', '미기록')}")
    lines += ["", "150k 검증: 평균 ± 학습 seed별 평균의 표본 표준편차:",
              "후보 | 가중치 | 평균 대기 | 최대 대기 평균 | 평균 대기열 | 전환수 | 처리량"]
    for row in data.analysis["multiseed"]["aggregates"]:
        values = " | ".join(f"{_number(row, metric):.3f} ± "
                            f"{_number(row, metric + '_training_seed_std'):.3f}" for metric in METRICS)
        lines.append(f"C{row['candidate_id']} | {_weight_text(row)} | {values}")
    lines += ["", "Pareto 후보(검증의 다섯 지표 기준; 최종 채택 기준과 구분):"]
    for stage in ("fine", "multiseed"):
        lines.append(f"{stage}: " + json.dumps(data.analysis[stage].get("pareto_fronts", []), ensure_ascii=False))
    lines += ["", "경계 확장 및 선택 고정 기록:",
              json.dumps(data.recommendation.get("boundary", {}), ensure_ascii=False),
              "경계에서 여전히 개선되는 경우 탐색 범위가 제한적이며 국소 최적값으로 단정하지 않습니다.",
              "", "2. 최종 미사용 random 평가 — 평균과 95% bootstrap 신뢰구간", "",
              "지표 | Incumbent | 사전 고정 후보"]
    for metric in METRICS:
        lines.append(f"{LABELS[metric][2]} | {_format_interval(incumbent, metric)} | "
                     f"{_format_interval(candidate, metric)}")
    comparison = _paired(data, "holdout", "random")
    waiting_low = comparison.get("avg_waiting_time_difference_ci95_low")
    waiting_high = comparison.get("avg_waiting_time_difference_ci95_high")
    if incumbent_retained:
        lines.append("최종 평가에는 incumbent만 있습니다. 아래 후보−incumbent 차이는 동일 후보의 자기 비교이므로 0이며, 새로운 후보의 개선을 시험한 결과가 아닙니다.")
    elif waiting_high is not None and waiting_high < 0:
        lines.append("고정 후보의 random holdout 평균 대기 차이 CI는 0보다 작습니다. 이 사전 고정 비교에 한정된 개선 방향의 근거이며 학습 seed가 3개라는 한계가 있습니다.")
    elif waiting_low is not None and waiting_low > 0:
        lines.append("고정 후보의 random holdout 평균 대기 차이 CI는 0보다 큽니다. 이 비교에서는 악화 방향입니다.")
    elif waiting_low is not None and waiting_high is not None:
        lines.append("고정 후보의 random holdout 평균 대기 차이 CI는 0을 포함합니다. 경험적 채택 판정과 별개로 통계적 개선은 확정하지 않습니다.")
    else:
        lines.append("고정 후보의 random holdout 평균 대기 차이 CI를 추정할 수 없어 통계적 개선은 판단하지 않습니다.")
    lines += ["", "같은 학습 seed와 교통 seed끼리 짝지은 후보−incumbent 차이:"]
    for metric in METRICS:
        lines.append(f"{LABELS[metric][2]}: {_format_interval(comparison, metric + '_difference', signed=True)}; "
                     f"평균 비율 변화 {_format_interval(comparison, metric + '_change_pct', signed=True)}%")
    lines += ["", "학습 seed별 최종 holdout 평균 대기 차이(후보−incumbent):"]
    for row in data.analysis["holdout"]["training_seed_comparisons"]:
        if _id(row["candidate_id"]) == data.selected_id:
            lines.append(f"학습 seed {row['training_seed']}: "
                         f"{_number(row, 'avg_waiting_time_difference'):+.3f}초; "
                         f"악화 없음={row['avg_waiting_time_not_worse']}")
    lines.append("교통 에피소드 단위 평균 대기 개선/동률/악화: "
                 + "/".join(str(comparison["avg_waiting_time_" + suffix]) for suffix in ("wins", "ties", "losses"))
                 + " (독립 재학습 횟수가 아님)")
    lines += ["", "3. 시나리오별 확인 — random holdout과 섞어 평균내지 않음", ""]
    if incumbent_retained:
        lines += ["그림 7과 아래 표는 incumbent의 시나리오별 절대 평균 및 95% CI입니다. "
                  "새 후보를 선택하지 않았으므로 후보 간 개선이나 비교 우월성을 보여주는 자료가 아닙니다.",
                  "시나리오 | 평균 대기(초) | 회차 최대 대기 평균(초) | 처리량(대/회차)"]
        for scenario in SCENARIO_LABELS:
            stage = "holdout" if scenario == "random" else "robustness"
            reference = _one(data.analysis[stage]["aggregates"], data.incumbent_id, scenario)
            lines.append(f"{scenario} | {_format_interval(reference, 'avg_waiting_time')} | "
                         f"{_format_interval(reference, 'max_waiting_time')} | "
                         f"{_format_interval(reference, 'throughput')}")
    else:
        lines.append("시나리오 | 평균 대기 차이(초) | 처리량 변화(%) | 회차 최대 대기 차이(초)")
        for scenario in SCENARIO_LABELS:
            if scenario == "random":
                continue
            paired = _paired(data, "robustness", scenario)
            lines.append(f"{scenario} | {_format_interval(paired, 'avg_waiting_time_difference', signed=True)} | "
                         f"{_format_interval(paired, 'throughput_change_pct', signed=True)} | "
                         f"{_format_interval(paired, 'max_waiting_time_difference', signed=True)}")
    lines += ["", "4. 제외·진단 자료", ""]
    for stage in ("holdout", "robustness"):
        for row in data.analysis[stage]["aggregates"]:
            if _id(row["candidate_id"]) not in {data.incumbent_id, data.selected_id}:
                continue
            diagnostic_keys = ("collisions_total", "teleports_total", "unfinished", "pending", "max_pending",
                               "maximum_waiting_worst_episode", "maximum_pending_worst_episode")
            diagnostics = {key: row[key] for key in diagnostic_keys if key in row}
            episodes = [episode for episode in data.raw[stage] if _key(episode) == _key(row)]
            for metric in ("unfinished", "pending", "max_pending"):
                if metric in row:
                    diagnostics[metric + "_mean_ci95"] = _format_interval(row, metric)
                    diagnostics[metric + "_worst_episode"] = row.get(metric + "_worst_episode")
                elif episodes and all(episode.get(metric) not in (None, "") for episode in episodes):
                    values = [_number(episode, metric) for episode in episodes]
                    diagnostics[metric + "_mean"] = float(np.mean(values))
                    diagnostics[metric + "_worst_episode"] = max(values)
            lines.append(f"{stage}/{row['scenario']}/C{row['candidate_id']}: "
                         + json.dumps(diagnostics, ensure_ascii=False))
    lines += ["", "5. 선택 고정과 통계 해석", "",
              "검증 제약: 충돌·텔레포트 0, incumbent 처리량 95% 이상, 회차 최대 대기의 평균 비증가.",
              "통과 후보를 평균 대기 → 최대 대기 평균 → 평균 대기열 → 실제 전환수 → 처리량 순으로 비교합니다.",
              "150k 검증과 최종 random 확인에서는 세 학습 seed 중 어느 하나에서도 평균 대기 방향이 악화되지 않는지 봅니다.",
              "최종 random에서는 평균 대기가 개선되어야 하며, 각 강건성 시나리오의 평균 대기는 5% 악화 상한도 확인합니다.",
              "shortlist_lock.json은 50k 검증 뒤 150k 대상 후보를 고정하고, selection_lock.json은 최종 시험 전에 후보를 고정합니다.",
              "최종 holdout과 강건성 자료를 보고 다른 후보로 바꿔 선택하지 않습니다. 실패 시 incumbent를 유지합니다.",
              "시험자료의 선택 목록이 비어 있는지와 최종 추천이 selection lock의 후보 또는 incumbent인지 검사했습니다.",
              "경험적 채택 조건을 통과하더라도 CI가 0을 포함할 수 있습니다. 이 경우 조건 통과와 통계적 개선 미확정을 구분합니다.",
              "그림 1~5와 7의 오차막대는 분석 모듈에서 산출한 95% bootstrap 신뢰구간입니다.",
              "단일 학습 seed인 50k 단계의 구간은 교통 seed 변동만 반영하며 학습 재현성을 검증하지 않습니다.",
              "150k 단계는 학습 seed와 교통 seed를 각각 재표집하는 two-way cluster bootstrap을 사용합니다.",
              "동일 모델의 여러 교통 에피소드를 서로 독립적인 재학습 결과로 세지 않습니다.",
              "학습 seed 3개만으로 얻은 구간은 제한적이며, 더 넓은 학습 무작위성의 충분한 검증은 아닙니다.",
              "그림 6의 오차막대는 학습 seed별 평균의 표본 표준편차입니다. 신뢰구간과 구분해야 합니다.",
              "대기 차이의 CI 전체가 0 미만이면 해당 사전 고정 비교에서 개선 방향의 근거가 됩니다. 0을 포함하면 개선을 확정하지 않습니다.",
              "개별 평균의 오차막대 중첩 여부로 paired 차이의 유의성을 판단하지 않습니다.",
              "검증 후보 및 여러 시나리오의 구간은 다중비교 보정되지 않았으므로 전 후보·전 상황의 유의한 우월성을 주장하지 않습니다.",
              "가중치별 그림은 나머지 두 가중치를 incumbent 값으로 고정한 조건부 절편입니다. 모든 조합을 평균낸 독립 효과가 아닙니다.",
              "평균 대기는 300초 cutoff까지 실제 진입한 departed 차량 전부의 관측 누적 대기 평균입니다. 미완료 차량은 포함하고 진입 전 pending 차량은 제외합니다.",
              "평균 대기열은 8개 진입 이동 그룹의 정지 차량 수를 합한 값의 시간 평균입니다.",
              "회차 최대 대기의 평균과 전체 최악 회차의 최대 대기는 서로 다른 지표입니다. pending·미완료·처리량을 함께 해석해야 합니다.",
              "충돌·텔레포트가 0이어도 급제동 등 모든 안전성이 검증된 것은 아닙니다. SUMO emergency braking/stop 경고는 이 두 지표에 포함되지 않으므로 개별 실행 로그도 확인해야 합니다.",
              "계수별 reward 합계는 서로 다른 목적함수이므로 계수 선택의 공통 성능 척도로 비교하지 않습니다.", "",
              "분석 모듈의 실제 inference 설정:"]
    for stage in STAGES:
        lines.append(f"{stage}: " + json.dumps(data.analysis[stage].get("inference", {}), ensure_ascii=False))
    lines += ["", "6. 재현 명령과 원본", "",
              f'python -m model.experiments.fine_search_report --input "{data.root}"',
              "학습 실행 및 단계별 재현 명령: model/experiments/FINE_SEARCH.md의 실행 절.",
              "실행 조건과 seed 목록: search_config.json",
              "후보 고정 기록: shortlist_lock.json, selection_lock.json",
              "채택 판단의 원본: final_recommendation.json",
              "수치 원본: *_results.csv 및 *_analysis.json",
              "학습량·가중치·runtime·수요·모델 해시는 각 실행 폴더와 runner의 재현 기록을 참조하십시오.",
              "기존 baseline 폴더의 파일별 SHA-256은 search_config.json의 prior_results_sha256에 저장되며 runner가 완료 전 보존 여부를 검증합니다.",
              "최종 추천에 연결된 모델 기록:",
              json.dumps(data.recommendation.get("model_results", []), ensure_ascii=False),
              "이 보고서는 저장된 결과를 읽어 생성했으며, 후보 재선정·모델 재학습·기본 설정 변경을 수행하지 않았습니다.", ""]
    return "\n".join(lines)


def generate_report(root: Path, output_dir: Path | None = None) -> dict:
    data = load_inputs(root)
    output = (output_dir or data.root / "reports").resolve()
    # A completed experiment is the only default destination; archived sources
    # are read-only inputs. Explicit output paths must not replace input files.
    if output in data.source_files:
        raise ValueError("Report output must be a directory, not an input artifact")
    output.mkdir(parents=True, exist_ok=True)
    figures = render_figures(data, output)
    report = output / "final-report-ko.txt"
    report.write_text(build_korean_text(data), encoding="utf-8")
    manifest = {
        "input_root": str(data.root), "report": str(report),
        "incumbent_id": data.incumbent_id, "locked_candidate_id": data.selected_id,
        "recommended_candidate_id": data.recommendation["recommended_candidate_id"],
        "confirmed": data.recommendation["confirmed"],
        "figures": [str(path) for path in figures],
        "figure_count": len(FIGURE_NAMES),
        "statistics_source": "model.experiments.fine_search_analysis",
        "report_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {path.relative_to(data.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in data.source_files},
        "selection_performed_by_report": False,
        "defaults_or_original_baselines_changed": False,
    }
    (output / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed fine-search result directory")
    parser.add_argument("--output", type=Path, help="Report directory; defaults to INPUT/reports")
    args = parser.parse_args()
    try:
        result = generate_report(args.input, args.output)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
