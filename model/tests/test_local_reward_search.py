"""Local experiment selection and isolation checks, with synthetic data only."""

from dataclasses import replace

import pytest

from model.env.reward import reward_terms, switching_penalty
from model.env.state_provider import TrafficSnapshot
from model.experiments.local_reward_search import (
    CANDIDATES, METRICS, summarize_local, verify_baseline, write_csv, write_json,
)
from model.utils.config import RewardConfig


def test_runtime_weights_affect_real_reward_without_changing_defaults():
    original = RewardConfig()
    changed = replace(original, waiting_weight=.35, max_waiting_weight=.35, switch_penalty=.60)
    snapshot = TrafficSnapshot((5.,) * 8, 40., 3000., 60., 1, 0)
    terms = reward_terms(snapshot, changed)
    assert terms == pytest.approx({'queue': -.5, 'waiting': -.175, 'max_waiting': -.175})
    assert switching_penalty(1, changed) == -.60
    assert (original.waiting_weight, original.max_waiting_weight, original.switch_penalty) == (.1, .1, .2)


def synthetic_results(folder):
    # candidate 1 fails throughput; 4 fails safety; 3 and 5 tie on average waiting.
    values = {1: (17, 65, 14, 40, 180), 2: (20, 70, 16, 41, 194.1),
              3: (19, 66, 15, 40, 190), 4: (18, 62, 14, 40, 195),
              5: (19, 60, 15, 40, 193), 6: (22, 75, 18, 43, 194)}
    for candidate, weights in CANDIDATES.items():
        run = folder / f'experiment_{candidate}'
        run.mkdir(parents=True)
        rows = []
        for seed in range(2001, 2031):
            rows.append({'seed': seed, **dict(zip(METRICS, values[candidate])),
                         'collisions': int(candidate == 4 and seed == 2001), 'teleports': 0})
        write_csv(run / 'evaluation_metrics.csv', rows)
        summary = {'experiment': candidate, 'alpha': weights[0], 'beta': weights[1],
                   'switch_gamma': weights[2], 'training_steps': 50000,
                   'training_seed': 22, 'scenario': 'random', 'evaluation_episodes': 30,
                   **dict(zip(METRICS, values[candidate]))}
        summary.update({m + '_std': 0.0 for m in METRICS})
        summary['maximum_waiting_worst_episode'] = summary['max_waiting_time']
        write_json(run / 'summary.json', summary)
        for name in ('training_routes.json', 'evaluation_routes.json'):
            write_json(run / name, [{'seed': 22, 'route_sha256': 'synthetic-only'}])


def test_safety_and_throughput_exclusions_then_ordered_tie_break(tmp_path):
    synthetic_results(tmp_path)
    result = summarize_local(tmp_path)
    assert result['recommended']['experiment'] == 5
    assert not result['eligibility'][1]['eligible']
    assert not result['eligibility'][4]['eligible']
    assert result['statistical_significance_test_performed'] is False
    with (tmp_path / 'changes_vs_baseline.csv').open(encoding='utf-8-sig') as stream:
        import csv
        rows = list(csv.DictReader(stream))
    baseline = next(row for row in rows if row['candidate'] == '2')
    assert all(float(baseline[m + '_change_pct']) == 0 for m in METRICS)
    winner = next(row for row in rows if row['candidate'] == '5')
    assert float(winner['avg_waiting_time_change_pct']) == pytest.approx(-5)


def test_demand_mismatch_prevents_successful_comparison(tmp_path):
    synthetic_results(tmp_path)
    write_json(tmp_path / 'experiment_3' / 'training_routes.json', [{'seed': 99, 'route_sha256': 'wrong'}])
    with pytest.raises(RuntimeError, match='Paired demand verification failed'):
        summarize_local(tmp_path)
    assert not (tmp_path / 'recommendation.json').exists()


def test_baseline_reward_mismatch_rejected_before_model_or_buffer_use(tmp_path):
    write_json(tmp_path / 'experiment_config.json', {'reward_config': {'waiting_weight': .4}})
    write_json(tmp_path / 'summary.json', {})
    with pytest.raises(ValueError, match='reward_config'):
        verify_baseline(tmp_path)
