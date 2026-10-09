"""Guard the comparable cohort and aggregation without launching SUMO or DQN."""

from copy import deepcopy
from math import sqrt
from pathlib import Path

import pytest

from model.experiments import paired_reward_comparison as comparison


def episodes():
    rows, manifest = [], []
    for index, seed in enumerate(range(2001, 2031)):
        rows.append({
            'seed': seed, 'traffic_scenario': 'random', 'duration': 300,
            'avg_waiting_time': index + 1, 'max_waiting_time': 100 + index,
            'avg_queue': 10 + index / 2, 'phase_changes': 40,
            'throughput': 190 + index, 'departed': 200 + index,
            'unfinished': 10, 'pending': 0, 'max_pending': 2,
            'max_queue': 40, 'forced_changes': 2, 'episode_reward': -50,
            'collisions': 0, 'teleports': 0,
        })
        manifest.append({'seed': seed, 'route_sha256': f'demand-{seed}'})
    return rows, manifest


def test_complete_paired_cohort_accepts_numeric_csv_fields():
    rows, manifest = episodes()
    csv_rows = [{key: str(value) for key, value in row.items()} for row in rows]
    comparison.validate_episodes(csv_rows, manifest)


@pytest.mark.parametrize('corruption', ['wrong_seed', 'duplicate_seed', 'missing_row', 'reordered'])
def test_evaluation_rows_must_cover_the_exact_ordered_cohort(corruption):
    rows, manifest = episodes()
    if corruption == 'wrong_seed':
        rows[-1]['seed'] = 8001
    elif corruption == 'duplicate_seed':
        rows[-1]['seed'] = rows[-2]['seed']
    elif corruption == 'missing_row':
        rows.pop()
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match='row count or seed order'):
        comparison.validate_episodes(rows, manifest)


@pytest.mark.parametrize('corruption', ['wrong_seed', 'duplicate_seed', 'missing_route'])
def test_route_manifest_must_match_the_evaluation_cohort(corruption):
    rows, manifest = episodes()
    if corruption == 'wrong_seed':
        manifest[-1]['seed'] = 8001
    elif corruption == 'duplicate_seed':
        manifest[-1]['seed'] = manifest[-2]['seed']
    else:
        manifest.pop()
    with pytest.raises(ValueError, match='manifest mismatch'):
        comparison.validate_episodes(rows, manifest)


@pytest.mark.parametrize(('key', 'value'), [('duration', 240), ('traffic_scenario', 'heavy')])
def test_cohort_rejects_other_horizons_and_traffic_scenarios(key, value):
    rows, manifest = episodes()
    rows[5][key] = value
    with pytest.raises(ValueError, match='scenario or duration'):
        comparison.validate_episodes(rows, manifest)


def test_missing_duration_cannot_silently_enter_the_table():
    rows, manifest = episodes()
    del rows[5]['duration']
    with pytest.raises((KeyError, ValueError), match='duration'):
        comparison.validate_episodes(rows, manifest)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_performance_metrics_are_rejected(value):
    rows, manifest = episodes()
    rows[5]['avg_waiting_time'] = value
    with pytest.raises(ValueError, match='Nonfinite metric'):
        comparison.validate_episodes(rows, manifest)


def test_unfinished_vehicles_remain_in_departed_vehicle_accounting():
    rows, manifest = episodes()
    rows[5]['departed'] = rows[5]['throughput']
    with pytest.raises(ValueError, match='Vehicle accounting'):
        comparison.validate_episodes(rows, manifest)


@pytest.fixture
def comparison_evidence(tmp_path, monkeypatch):
    output = tmp_path / 'results'
    output.mkdir()
    rows, manifest = episodes()
    folders = {}
    for candidate in range(1, 8):
        folder = tmp_path / f'candidate-{candidate}'
        folder.mkdir()
        comparison.write_json(folder / 'training_routes.json', [{'seed': 22, 'route_sha256': 'training'}])
        comparison.write_json(folder / 'evaluation_routes.json', manifest)
        comparison.write_csv(folder / 'evaluation_metrics.csv', rows)
        folders[candidate] = folder
    states = {candidate: {'num_timesteps': 50_000, 'n_updates': 12_375,
                           'exploration_rate': .05, 'observation_shape': [60],
                           'action_count': 8} for candidate in folders}

    def validated(_folder, candidate):
        return {'model_sha256': f'model-{candidate}'}, deepcopy(states[candidate])

    monkeypatch.setattr(comparison, 'validate_training', validated)
    monkeypatch.setattr(comparison, 'core_hashes', lambda: {'core': 'unchanged'})
    monkeypatch.setattr(comparison, 'current_runtime', lambda: {'runtime': 'unchanged'})
    config = {'core_sha256': {'core': 'unchanged'}, 'runtime': {'runtime': 'unchanged'},
              'sumo_config_sha256': comparison.sha256(comparison.ProjectConfig().sumo_config_file),
              'preserved_source_sha256': {}, 'script_sha256': comparison.sha256(Path(comparison.__file__))}
    return output, config, folders, states


def test_aggregate_uses_episode_sample_sd_and_keeps_each_demand_model_pair(comparison_evidence):
    output, config, folders, _ = comparison_evidence
    comparison.aggregate(output, config, folders)
    summaries = comparison.read_json(output / 'comparison.json')['results']
    assert len(summaries) == 7
    for row in summaries:
        assert row['evaluation_episodes'] == 30
        assert row['avg_waiting_time'] == 15.5
        # For 1..30, sample variance is 77.5; population variance would be 899/12.
        assert row['avg_waiting_time_std'] == pytest.approx(sqrt(77.5))
        assert row['max_waiting_time'] == 114.5
        assert row['maximum_waiting_worst_episode'] == 129
    combined = comparison.read_csv(output / 'evaluation_metrics.csv')
    pairs = {(int(row['experiment']), int(row['seed'])) for row in combined}
    assert pairs == {(candidate, seed) for candidate in range(1, 8) for seed in range(2001, 2031)}
    assert all(row['demand_sha256'] == f"demand-{row['seed']}" for row in combined)
    assert all(row['model_sha256'] == f"model-{row['experiment']}" for row in combined)
    assert comparison.read_json(output / 'verification.json')['status'] == 'PASS'
    assert '15.50 ± 8.80' in (output / 'comparison-ko.txt').read_text(encoding='utf-8')


@pytest.mark.parametrize('mismatch', ['training_routes', 'evaluation_routes', 'training_state'])
def test_aggregate_refuses_unpaired_demand_or_update_schedule(comparison_evidence, mismatch):
    output, config, folders, states = comparison_evidence
    if mismatch == 'training_state':
        states[7]['n_updates'] += 1
    else:
        path = folders[7] / f'{mismatch}.json'
        routes = comparison.read_json(path)
        routes[-1]['route_sha256'] = 'different-demand'
        comparison.write_json(path, routes)
    with pytest.raises(ValueError, match='not paired'):
        comparison.aggregate(output, config, folders)
    assert not (output / 'comparison.csv').exists()
    assert not (output / 'verification.json').exists()


def test_aggregate_does_not_publish_a_partial_candidate_table(comparison_evidence):
    output, config, folders, _ = comparison_evidence
    del folders[7]
    with pytest.raises(ValueError, match='All seven candidates'):
        comparison.aggregate(output, config, folders)
    assert not (output / 'comparison.csv').exists()
