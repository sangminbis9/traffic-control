import pandas as pd
import pytest

from experiment_evaluate import compare


def frame(wait: float, queue: float, throughput: int, route: str = "same"):
    return pd.DataFrame([dict(traffic_scenario="balanced", seed=3001, route_sha256=route,
        avg_waiting_time=wait, avg_queue=queue, throughput=throughput, max_waiting_time=20,
        phase_changes=30, forced_changes=0, not_inserted=0)])


def test_comparison_requires_matching_routes():
    with pytest.raises(AssertionError, match="mismatched"):
        compare(frame(10, 5, 100), frame(8, 4, 105, "different"))


def test_comparison_requires_all_primary_metrics():
    result = compare(frame(10, 5, 100), frame(8, 4, 105))
    assert result["waiting_improvement_pct"] == 20
    assert result["all_primary_better"]
    assert not compare(frame(10, 5, 100), frame(8, 4, 90))["all_primary_better"]
