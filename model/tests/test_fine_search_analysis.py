"""Analysis regressions using synthetic, deliberately dependent paired episodes."""

import json

import numpy as np
import pytest

from model.experiments.fine_search_analysis import analyze_episodes, pareto_front


def episode(candidate, train=1, traffic=101, waiting=20, maximum=60,
            throughput=100, collisions=0, teleports=0):
    return {
        'candidate_id': candidate, 'training_seed': train,
        'training_steps': 50_000, 'evaluation_seed': traffic, 'scenario': 'random',
        'waiting_weight': .3, 'max_waiting_weight': .3, 'switch_penalty': .5,
        'avg_waiting_time': waiting, 'max_waiting_time': maximum, 'avg_queue': 10,
        'phase_changes': 20, 'throughput': throughput,
        'collisions': collisions, 'teleports': teleports,
    }


def analyze(rows, split='validation'):
    return analyze_episodes(rows, 'incumbent', split=split, bootstrap_samples=1000, bootstrap_seed=7)


def candidate_row(result, key, candidate='challenger'):
    return next(row for row in result[key] if row['candidate_id'] == candidate)


def test_pairing_uses_keys_and_percent_of_means_is_not_mean_of_percents():
    rows = [episode('incumbent', traffic=102, waiting=100),
            episode('challenger', traffic=101, waiting=2),
            episode('incumbent', traffic=101, waiting=1),
            episode('challenger', traffic=102, waiting=90)]
    result = analyze(rows)
    comparison = candidate_row(result, 'paired_comparisons')
    assert comparison['avg_waiting_time_difference'] == -4.5
    assert comparison['avg_waiting_time_change_pct'] == pytest.approx(-9 / 101 * 100)
    assert comparison['avg_waiting_time_mean_paired_change_pct'] == 45
    assert comparison['avg_waiting_time_wins'] == 1
    assert comparison['avg_waiting_time_losses'] == 1
    assert comparison['avg_waiting_time_ties'] == 0
    pairs = [row for row in result['paired_rows'] if row['candidate_id'] == 'challenger']
    assert [(row['evaluation_seed'], row['avg_waiting_time_difference']) for row in pairs] == [(101, 1), (102, -10)]
    assert result == analyze(list(reversed(rows)))


def test_paired_bootstrap_preserves_shared_traffic_variation():
    rows = []
    for train in (1, 2, 3):
        for traffic in (101, 102, 103):
            waiting = train * 10 + (traffic - 100) * 100
            rows.extend([episode('incumbent', train, traffic, waiting=waiting, maximum=500),
                         episode('challenger', train, traffic, waiting=waiting - 5, maximum=500)])
    result = analyze(rows)
    comparison = candidate_row(result, 'paired_comparisons')
    assert comparison['avg_waiting_time_difference_ci95_low'] == pytest.approx(-5)
    assert comparison['avg_waiting_time_difference_ci95_high'] == pytest.approx(-5)
    assert comparison['paired_waiting_ci_below_zero']
    assert not result['inference']['statistical_significance_established']
    seeds = [row for row in result['training_seed_comparisons'] if row['candidate_id'] == 'challenger']
    assert len(seeds) == 3
    assert all(row['avg_waiting_time_improved'] for row in seeds)
    assert all(row['avg_waiting_time_difference'] == -5 for row in seeds)


def test_repeated_traffic_does_not_fake_independent_training_replicates():
    rows = []
    for train, difference in ((1, -10), (2, 10)):
        for traffic in range(101, 151):
            rows.extend([episode('incumbent', train, traffic, waiting=20),
                         episode('challenger', train, traffic, waiting=20 + difference)])
    result = analyze(rows)
    comparison = candidate_row(result, 'paired_comparisons')
    assert comparison['pair_count'] == 100
    assert comparison['training_seed_count'] == 2
    # IID resampling of 100 rows would spuriously shrink this interval.
    assert comparison['avg_waiting_time_difference_ci95_low'] == -10
    assert comparison['avg_waiting_time_difference_ci95_high'] == 10
    aggregate = candidate_row(result, 'aggregates')
    assert aggregate['avg_waiting_time_training_seed_std'] == pytest.approx(np.sqrt(200))
    assert aggregate['avg_waiting_time_std'] == pytest.approx(np.std([10] * 50 + [30] * 50, ddof=1))


def test_validation_constraints_precede_lexicographic_ranking():
    rows = [episode('incumbent'), episode('unsafe', waiting=1, collisions=1),
            episode('teleported', waiting=2, teleports=1),
            episode('low-throughput', waiting=3, throughput=94),
            episode('high-maximum', waiting=4, maximum=61),
            episode('challenger', waiting=18, maximum=59, throughput=95)]
    result = analyze(rows)
    assert result['selection'][0]['winner_candidate_id'] == 'challenger'
    for candidate in ('unsafe', 'teleported', 'low-throughput', 'high-maximum'):
        assert not candidate_row(result, 'aggregates', candidate)['eligible']
    assert candidate_row(result, 'aggregates')['throughput_retention'] == .95
    assert result['selection'][0]['selection_rules']['weighted_score_used'] is False


def test_numeric_candidate_ids_break_equal_metric_ties_in_numeric_order():
    rows = [episode('incumbent', waiting=30), episode(10), episode(2)]
    result = analyze(rows)
    assert result['selection'][0]['winner_candidate_id'] == '2'
    assert result['selection'][0]['ranked_candidate_ids'] == ['2', '10', 'incumbent']
    assert result['pareto_fronts'][0]['candidate_ids'] == ['2', '10']


def test_diagnostics_keep_crossed_ci_and_do_not_change_selection_or_pareto():
    rows = []
    for candidate in ('incumbent', 'challenger'):
        for train in (1, 2):
            for traffic in (101, 102):
                row = episode(candidate, train, traffic, waiting=19 if candidate == 'challenger' else 20)
                value = 100 * train if candidate == 'challenger' else 0
                row.update({metric: value for metric in ('unfinished', 'pending', 'max_pending',
                                                        'forced_changes', 'max_queue')})
                rows.append(row)
    result = analyze(rows)
    aggregate = candidate_row(result, 'aggregates')
    for metric in ('unfinished', 'pending', 'max_pending', 'forced_changes', 'max_queue'):
        assert aggregate[metric] == 150
        assert aggregate[metric + '_std'] == pytest.approx(np.std([100, 100, 200, 200], ddof=1))
        assert aggregate[metric + '_training_seed_std'] == pytest.approx(np.sqrt(5000))
        assert aggregate[metric + '_ci95_low'] == 100
        assert aggregate[metric + '_ci95_high'] == 200
        assert aggregate[metric + '_worst_episode'] == 200
    assert result['selection'][0]['winner_candidate_id'] == 'challenger'
    assert result['pareto_fronts'][0]['candidate_ids'] == ['challenger']


def test_partial_diagnostics_are_rejected_instead_of_zero_filled():
    rows = [episode('incumbent'), episode('challenger')]
    rows[0]['pending'] = 5
    with pytest.raises(ValueError, match='Incomplete diagnostic metric: pending'):
        analyze(rows)


@pytest.mark.parametrize('split', ['holdout', 'robustness'])
def test_nonvalidation_data_never_selects_a_candidate(split):
    result = analyze([episode('incumbent'), episode('challenger', waiting=1)], split)
    assert result['selection'] == []
    assert result['inference']['holdout_reselection_permitted'] is False


def test_pareto_uses_all_five_objectives_and_keeps_equal_vectors():
    context = {'training_steps': 50_000, 'scenario': 'random', 'split': 'validation'}
    base = {**context, 'avg_waiting_time': 10, 'max_waiting_time': 30,
            'avg_queue': 5, 'phase_changes': 10, 'throughput': 100}
    rows = [{**base, 'candidate_id': 'base'}, {**base, 'candidate_id': 'equal'},
            {**base, 'candidate_id': 'dominated', 'max_waiting_time': 31},
            {**base, 'candidate_id': 'phase_tradeoff', 'avg_waiting_time': 9, 'phase_changes': 11},
            {**base, 'candidate_id': 'throughput_tradeoff', 'avg_waiting_time': 11, 'throughput': 101}]
    assert pareto_front(rows) == ['base', 'equal', 'phase_tradeoff', 'throughput_tradeoff']
    rows[-1]['scenario'] = 'heavy'
    with pytest.raises(ValueError, match='one budget, scenario and split'):
        pareto_front(rows)


def test_zero_reference_and_singletons_are_json_safe_without_false_ci():
    rows = [episode('incumbent', waiting=0, maximum=0, throughput=0),
            episode('challenger', waiting=1, maximum=0, throughput=0)]
    result = analyze(rows)
    comparison = candidate_row(result, 'paired_comparisons')
    assert comparison['avg_waiting_time_change_pct'] is None
    assert comparison['avg_waiting_time_difference_ci95_low'] is None
    assert comparison['avg_waiting_time_change_pct_ci95_high'] is None
    assert comparison['ci_scope'] == 'no_sampling_replication'
    json.dumps(result, allow_nan=False)


def test_single_training_seed_ci_is_explicitly_conditional():
    rows = [episode(candidate, traffic=traffic)
            for candidate in ('incumbent', 'challenger') for traffic in (101, 102)]
    result = analyze(rows)
    assert all(row['ci_scope'] == 'conditional_on_one_trained_policy' for row in result['aggregates'])


def test_alias_fields_and_nested_weights():
    rows = [episode('incumbent'), episode('challenger')]
    for row in rows:
        row['experiment'] = row.pop('candidate_id')
        row['train_seed'] = row.pop('training_seed')
        row['seed'] = row.pop('evaluation_seed')
        row['traffic_scenario'] = row.pop('scenario')
        row['weights'] = tuple(row.pop(key) for key in ('waiting_weight', 'max_waiting_weight', 'switch_penalty'))
    assert len(analyze(rows)['aggregates']) == 2


@pytest.mark.parametrize('problem', ['duplicate', 'missing_cell', 'unmatched_seed', 'missing_incumbent',
                                      'mixed_split', 'nonfinite', 'inconsistent_weights'])
def test_invalid_design_is_rejected_instead_of_silently_dropping_pairs(problem):
    rows = [episode(candidate, train, traffic)
            for candidate in ('incumbent', 'challenger') for train in (1, 2) for traffic in (101, 102)]
    if problem == 'duplicate':
        rows.append(dict(rows[-1]))
    elif problem == 'missing_cell':
        rows.pop()
    elif problem == 'unmatched_seed':
        rows[-1]['evaluation_seed'] = 999
    elif problem == 'missing_incumbent':
        rows = [row for row in rows if row['candidate_id'] != 'incumbent']
    elif problem == 'mixed_split':
        rows[-1]['split'] = 'holdout'
    elif problem == 'nonfinite':
        rows[-1]['avg_waiting_time'] = float('nan')
    elif problem == 'inconsistent_weights':
        rows[-1]['waiting_weight'] = .4
    with pytest.raises(ValueError):
        analyze(rows)
