"""Fair-cohort and statistical-unit safeguards without running SUMO."""
from copy import deepcopy

import numpy as np
import pytest

from model.experiments import rule_based_comparison as benchmark
from model.utils.config import ProjectConfig


def cases():
    return [{'scenario': 'random', 'seed': seed, 'route_sha256': f'demand-{seed}',
             'demanded': 100} for seed in (41001, 41002)]


def episodes(rewards=(-100., -200.), waits=(10., 20.)):
    result = []
    for index, case in enumerate(cases()):
        row = {metric: 0 for metric in benchmark.METRICS}
        row.update(case)
        row.update({'traffic_scenario': case['scenario'], 'episode_reward': rewards[index],
                    'avg_waiting_time': waits[index], 'duration': 300,
                    'departed': 90, 'throughput': 80, 'unfinished': 10,
                    'pending': 10, 'max_pending': 10, 'max_waiting_time': 50})
        result.append(row)
    return result


def test_default_cohorts_are_independent_and_include_all_scenarios():
    tuning, heldout = benchmark.cohort('tune'), benchmark.cohort('evaluate')
    assert len(tuning) == 20
    assert len(heldout) == 60
    assert sum(case['scenario'] == 'random' for case in heldout) == 30
    assert {case['scenario'] for case in heldout} == set(benchmark.SCENARIOS)
    benchmark.validate_disjoint(tuning, heldout, range(1, 1001))
    with pytest.raises(ValueError, match='disjoint'):
        benchmark.validate_disjoint(tuning, heldout, [31001])
    with pytest.raises(ValueError, match='disjoint'):
        benchmark.validate_disjoint(tuning, [tuning[0]])
    with pytest.raises(ValueError, match='Duplicate'):
        benchmark.validate_disjoint(tuning + [tuning[0]], heldout)


def test_complete_episode_accounts_for_unentered_and_unfinished_vehicles():
    benchmark.validate_rows(episodes(), cases())
    missing_pending = episodes()
    missing_pending[0]['pending'] = 0
    with pytest.raises(ValueError, match='Demand/pending'):
        benchmark.validate_rows(missing_pending, cases())
    missing_unfinished = episodes()
    missing_unfinished[0]['unfinished'] = 0
    with pytest.raises(ValueError, match='Departed vehicle'):
        benchmark.validate_rows(missing_unfinished, cases())


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'route', 'horizon', 'nan', 'collision'])
def test_incomplete_unpaired_or_failed_episodes_cannot_enter_ranking(corruption):
    rows = episodes()
    if corruption == 'missing':
        rows.pop()
    elif corruption == 'duplicate':
        rows[-1] = deepcopy(rows[0])
    elif corruption == 'route':
        rows[0]['route_sha256'] = 'different-demand'
    elif corruption == 'horizon':
        rows[0]['duration'] = 299
    elif corruption == 'nan':
        rows[0]['episode_reward'] = float('nan')
    else:
        rows[0]['collisions'] = 1
    with pytest.raises(ValueError):
        benchmark.select_candidate({'only': rows}, cases(), {'only': {}})


def test_rule_selection_uses_reward_then_waiting_then_stable_id():
    result = {'a': episodes(rewards=(-100., -100.), waits=(1., 1.)),
              'b': episodes(rewards=(-80., -80.), waits=(40., 40.)),
              'c': episodes(rewards=(-80., -80.), waits=(30., 30.)),
              'd': episodes(rewards=(-80., -80.), waits=(30., 30.))}
    winner, ranking = benchmark.select_candidate(result, cases(), dict.fromkeys(result))
    assert winner == 'c'
    assert [row['candidate'] for row in ranking] == ['c', 'd', 'b', 'a']
    with pytest.raises(ValueError, match='Every frozen candidate'):
        benchmark.select_candidate({'a': result['a']}, cases(), dict.fromkeys(result))


def test_dqn_average_keeps_one_statistical_unit_per_traffic_case():
    models = {'seed1': episodes(waits=(10., 20.)),
              'seed2': episodes(waits=(20., 30.)),
              'seed3': episodes(waits=(30., 40.))}
    average = benchmark.average_models(models, list(models), cases())
    assert len(average) == 2  # Six model/case episodes do not become six independent traffic cases.
    assert [row['avg_waiting_time'] for row in average] == [20., 30.]
    comparison = benchmark.paired_differences(average, episodes(waits=(19., 29.)), samples=500)
    waiting = next(row for row in comparison if row['metric'] == 'avg_waiting_time')
    assert waiting == {'metric': 'avg_waiting_time', 'paired_cases': 2,
                       'mean_difference': 1., 'ci95_low': 1., 'ci95_high': 1.}
    models['seed3'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        benchmark.average_models(models, list(models), cases())


def test_paired_differences_reject_route_changes_and_reordering():
    original = episodes()
    with pytest.raises(ValueError, match='ordered traffic cases'):
        benchmark.paired_differences(original, list(reversed(original)))
    changed = deepcopy(original)
    changed[0]['route_sha256'] = 'not-the-same-traffic'
    with pytest.raises(ValueError, match='route mismatch'):
        benchmark.paired_differences(original, changed)


def test_stratified_bootstrap_does_not_change_the_predeclared_scenario_mix():
    first, second = episodes(waits=(1., 11.)), episodes(waits=(0., 0.))
    for rows in (first, second):
        rows[1]['scenario'] = rows[1]['traffic_scenario'] = 'heavy'
    compared = benchmark.paired_differences(first, second, samples=500)
    waiting = next(row for row in compared if row['metric'] == 'avg_waiting_time')
    # Each singleton scenario remains present in every bootstrap sample.
    assert waiting['mean_difference'] == waiting['ci95_low'] == waiting['ci95_high'] == 6.


def test_cycle_policy_ignores_demand_and_waits_for_configured_green():
    policy = benchmark.SafeFixedPolicy()
    obs = np.zeros(60, dtype=np.float32)
    obs[24], obs[33] = 1, 1
    obs[32] = 9 / ProjectConfig().signal.max_green
    assert policy.action(obs) == 0
    obs[:24], obs[44:] = 1, 1
    assert policy.action(obs) == 0
    obs[32] = 10 / ProjectConfig().signal.max_green
    assert policy.action(obs) == 1
    # During clearance it respects the committed target, including a safety override.
    obs[33], obs[34], obs[36 + 6] = 0, 1, 1
    assert policy.action(obs) == 6


def test_selection_cannot_be_replaced_after_tuning_evidence_is_frozen(tmp_path):
    candidates = {'a': {}, 'b': {}}
    protocol = {'rule_candidates': candidates, 'routes': {'tune': cases()}}
    benchmark.write_json(tmp_path / 'protocol.json', protocol)
    results = {'a': episodes(), 'b': episodes(rewards=(-300., -300.))}
    hashes = {}
    for name, rows in results.items():
        folder = tmp_path / 'tune' / name
        folder.mkdir(parents=True)
        benchmark.write_json(folder / 'episodes.json', rows)
        hashes[name] = benchmark.sha256(folder / 'episodes.json')
    winner, ranking = benchmark.select_candidate(results, cases(), candidates)
    selection = {'winner': winner, 'policy': {}, 'ranking': ranking,
                 'protocol_sha256': benchmark.sha256(tmp_path / 'protocol.json'),
                 'tuning_result_sha256': hashes}
    benchmark.write_json(tmp_path / 'selection.json', selection)
    assert benchmark.load_selection(tmp_path, protocol)['winner'] == 'a'
    selection['winner'] = 'b'
    benchmark.write_json(tmp_path / 'selection.json', selection)
    with pytest.raises(ValueError, match='does not match frozen'):
        benchmark.load_selection(tmp_path, protocol)
    selection['winner'] = 'a'
    benchmark.write_json(tmp_path / 'selection.json', selection)
    results['a'][0]['episode_reward'] += 1
    benchmark.write_json(tmp_path / 'tune/a/episodes.json', results['a'])
    with pytest.raises(ValueError, match='changed after selection'):
        benchmark.load_selection(tmp_path, protocol)


def test_report_rejects_missing_controller_before_publishing(tmp_path):
    with pytest.raises(ValueError, match='All heldout policies'):
        benchmark.aggregate(tmp_path, {}, {}, {'rule': episodes()})
    assert not (tmp_path / 'comparison.json').exists()
