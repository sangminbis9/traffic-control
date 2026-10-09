"""Pure analysis of paired, crossed training-seed/traffic-seed experiments.

Each (training_steps, scenario) group must contain the same complete rectangular
training-seed x evaluation-seed design for every candidate. An episode is not an
independent replicate of training: episodes share a learned policy, and policies
also share the evaluation traffic seeds. CIs therefore independently resample the
two seed axes, using the SAME draws for candidate and incumbent. This is a
crossed-cluster (product/pigeonhole) percentile bootstrap, not an IID episode
bootstrap. It is approximate, can be conservative for interaction noise, and is
unstable with very few training seeds. It does not establish a global optimum.

Scenarios are fixed conditions analyzed separately, not resampled populations.
With one training seed, CIs are conditional on that learned policy; with one
evaluation seed, CIs are conditional on that traffic realization. With one of
each, no sampling CI is reported. Episode SD is descriptive only. Percent change
of aggregate means differs from the mean of episode-level percent changes; both
are named explicitly. Zero reference denominators produce None, never infinity.

Public entry point: analyze_episodes(rows, incumbent_id, split='validation').
Outputs are JSON-safe lists of flat rows. Only validation can select a candidate;
holdout and robustness are descriptive and cannot replace a locked winner.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np


METRICS = ('avg_waiting_time', 'max_waiting_time', 'avg_queue',
           'phase_changes', 'throughput')
DIAGNOSTICS = ('unfinished', 'pending', 'max_pending', 'forced_changes', 'max_queue')
SPLITS = frozenset(('validation', 'holdout', 'robustness'))
THROUGHPUT_RETENTION = 0.95
TIE_ABSOLUTE_TOLERANCE = 1e-9


def _get(row: Mapping[str, Any], name: str, *aliases: str) -> Any:
    for key in (name, *aliases):
        if key in row and row[key] is not None and row[key] != '':
            return row[key]
    raise ValueError(f'Missing required episode field: {name}')


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0:
        raise ValueError(f'{field} must be a finite nonnegative number')
    return number


def _integer(value: Any, field: str) -> int:
    number = _finite(value, field)
    if number != int(number):
        raise ValueError(f'{field} must be an integer')
    return int(number)


def _normalize(row: Mapping[str, Any], split: str) -> dict:
    if row.get('split', split) != split:
        raise ValueError('Mixed evaluation splits are not permitted')
    candidate = str(_get(row, 'candidate_id', 'experiment'))
    scenario = str(_get(row, 'scenario', 'traffic_scenario'))
    weights = row.get('weights', {})
    if isinstance(weights, (list, tuple)):
        if len(weights) != 3:
            raise ValueError('weights must contain exactly three coefficients')
        weights = dict(zip(('alpha', 'beta', 'switch_gamma'), weights, strict=True))
    if not isinstance(weights, Mapping):
        raise ValueError('weights must be a mapping or three-element sequence')
    merged = {**weights, **row}
    result = {
        'candidate_id': candidate,
        'alpha': _finite(_get(merged, 'alpha', 'waiting_weight'), 'alpha'),
        'beta': _finite(_get(merged, 'beta', 'max_waiting_weight'), 'beta'),
        'switch_gamma': _finite(_get(merged, 'switch_gamma', 'switch_penalty'), 'switch_gamma'),
        'training_seed': _integer(_get(row, 'training_seed', 'train_seed'), 'training_seed'),
        'training_steps': _integer(_get(row, 'training_steps'), 'training_steps'),
        'evaluation_seed': _integer(_get(row, 'evaluation_seed', 'seed'), 'evaluation_seed'),
        'scenario': scenario, 'split': split,
        'collisions': _integer(_get(row, 'collisions'), 'collisions'),
        'teleports': _integer(_get(row, 'teleports'), 'teleports'),
    }
    for metric in METRICS:
        result[metric] = _finite(_get(row, metric), metric)
    for metric in DIAGNOSTICS:
        if metric in row:
            result[metric] = _finite(_get(row, metric), metric)
    return result


def _candidate_key(candidate: str) -> tuple[int, int | str, str]:
    """Keep numeric IDs in numeric order and named IDs in lexical order."""
    text = str(candidate)
    try:
        return (0, int(text), text)
    except ValueError:
        return (1, text, text)


def _std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _percent(candidate: float, incumbent: float) -> float | None:
    return float(100.0 * (candidate - incumbent) / incumbent) if incumbent != 0 else None


def _ci(samples: np.ndarray, estimable: bool) -> tuple[float | None, float | None]:
    if not estimable or not np.isfinite(samples).all():
        return None, None
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _orientation(metric: str) -> int:
    return 1 if metric == 'throughput' else -1


def pareto_front(aggregates: Iterable[Mapping[str, Any]]) -> list[str]:
    """Nondominated IDs for ONE scenario and budget, using all five means.

    Waiting, maximum waiting, queue and phase changes are minimized; throughput
    is maximized. Feasibility is intentionally separate from Pareto dominance.
    Equal five-metric vectors both remain on the frontier.
    """
    rows = list(aggregates)
    groups = {(row['training_steps'], row['scenario'], row['split']) for row in rows}
    if len(groups) > 1:
        raise ValueError('Pareto comparison requires one budget, scenario and split')
    frontier = []
    for candidate in rows:
        value = np.asarray([float(candidate[metric]) * -_orientation(metric) for metric in METRICS])
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            alternative = np.asarray([float(other[metric]) * -_orientation(metric) for metric in METRICS])
            if np.all(alternative <= value) and np.any(alternative < value):
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate['candidate_id']))
    return sorted(frontier, key=_candidate_key)


def analyze_episodes(
    rows: Iterable[Mapping[str, Any]], incumbent_id: str | int, *, split: str,
    bootstrap_samples: int = 2000, bootstrap_seed: int = 1729,
) -> dict:
    """Analyze complete paired episode rows without model or filesystem access.

    Required metadata: candidate_id, training_seed, training_steps,
    evaluation_seed, scenario, alpha/beta/switch_gamma (or weights), the five
    METRICS and collisions/teleports. train_seed, experiment, seed,
    traffic_scenario and the RewardConfig weight names are accepted aliases.
    Optional DIAGNOSTICS must be present in every row if supplied in any row.
    They receive aggregate mean/SD/CI and worst-episode diagnostics, but never
    affect selection or the five-objective Pareto calculation.

    `aggregates` stores each mean under its metric name; `_std` is episode SD,
    `_training_seed_std` is SD of policy means, and `_ci95_low/high` is the
    crossed-bootstrap mean CI. `paired_comparisons` uses candidate minus
    incumbent for `_difference`; negative is better except for throughput.
    `_change_pct` is percent change of aggregate means, with a paired ratio CI.
    Win/tie/loss counts describe matched seed cells, not independent trials.
    """
    if split not in SPLITS:
        raise ValueError(f'split must be one of {sorted(SPLITS)}')
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 2:
        raise ValueError('bootstrap_samples must be an integer >= 2')
    normalized = [_normalize(row, split) for row in rows]
    if not normalized:
        raise ValueError('At least one episode is required')
    diagnostics = tuple(metric for metric in DIAGNOSTICS if any(metric in row for row in normalized))
    for metric in diagnostics:
        if any(metric not in row for row in normalized):
            raise ValueError(f'Incomplete diagnostic metric: {metric}')
    aggregate_metrics = (*METRICS, *diagnostics)
    incumbent_id = str(incumbent_id)
    grouped: dict[tuple[int, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    identities = set()
    candidate_weights = {}
    for row in normalized:
        identity = (row['candidate_id'], row['training_steps'], row['scenario'],
                    row['training_seed'], row['evaluation_seed'])
        if identity in identities:
            raise ValueError(f'Duplicate episode pairing key: {identity}')
        identities.add(identity)
        weights = (row['alpha'], row['beta'], row['switch_gamma'])
        if row['candidate_id'] in candidate_weights and candidate_weights[row['candidate_id']] != weights:
            raise ValueError(f'Inconsistent weights for candidate {row["candidate_id"]}')
        candidate_weights[row['candidate_id']] = weights
        grouped[(row['training_steps'], row['scenario'])][row['candidate_id']].append(row)

    result: dict[str, Any] = {
        'schema_version': 1, 'split': split, 'incumbent_candidate_id': incumbent_id,
        'aggregates': [], 'paired_rows': [], 'paired_comparisons': [],
        'training_seed_comparisons': [], 'pareto_fronts': [], 'selection': [],
        'inference': {
            'ci_method': 'two_way_crossed_cluster_percentile_bootstrap',
            'confidence_level': 0.95, 'bootstrap_samples': bootstrap_samples,
            'bootstrap_seed': bootstrap_seed,
            'resampling_units': ['training_seed', 'evaluation_seed'],
            'candidate_incumbent_resampling_is_paired': True,
            'assumptions': [
                'Complete balanced crossed seed design; equal weighting of both seed axes.',
                'Training seeds are exchangeable independent runs; traffic seeds are exchangeable draws.',
                'The two seed axes are independent; shared-policy and shared-traffic dependence is preserved.',
                'Scenarios and training budgets are fixed conditions, analyzed separately.',
                'A singleton axis is conditioned on; two singleton axes yield no sampling CI.',
                'Very few training seeds limit precision; product bootstrap may be conservative for interactions.',
            ],
            'episode_std_is_descriptive_not_standard_error': True,
            'sample_std_ddof': 1,
            'singleton_std_zero_is_display_convention_not_estimated_variation': True,
            'diagnostic_metrics': list(diagnostics),
            'win_tie_loss_unit': 'matched training_seed x evaluation_seed cell; counts are dependent',
            'tie_absolute_tolerance': TIE_ABSOLUTE_TOLERANCE,
            'aggregate_percent_change': '100 * (candidate_mean - incumbent_mean) / incumbent_mean',
            'zero_reference_percent_change': None,
            'multiplicity_adjustment': None,
            'statistical_significance_established': False,
            'holdout_reselection_permitted': False,
        },
    }
    rng = np.random.default_rng(bootstrap_seed)
    for (steps, scenario), candidates in sorted(grouped.items()):
        if incumbent_id not in candidates:
            raise ValueError(f'Missing incumbent for budget={steps}, scenario={scenario}')
        reference_rows = candidates[incumbent_id]
        training_seeds = sorted({row['training_seed'] for row in reference_rows})
        evaluation_seeds = sorted({row['evaluation_seed'] for row in reference_rows})
        expected = {(train, traffic) for train in training_seeds for traffic in evaluation_seeds}
        by_candidate = {}
        for candidate, candidate_rows in candidates.items():
            indexed = {(row['training_seed'], row['evaluation_seed']): row for row in candidate_rows}
            if set(indexed) != expected:
                raise ValueError(f'Incomplete or unmatched crossed seed design for candidate {candidate}')
            by_candidate[candidate] = indexed
        nt, ne = len(training_seeds), len(evaluation_seeds)
        estimable = nt > 1 or ne > 1
        train_draws = rng.integers(nt, size=(bootstrap_samples, nt))
        traffic_draws = rng.integers(ne, size=(bootstrap_samples, ne))
        scope = ('training_and_traffic_seed_variation' if nt > 1 and ne > 1 else
                 'conditional_on_one_trained_policy' if nt == 1 and ne > 1 else
                 'conditional_on_one_traffic_seed' if nt > 1 else 'no_sampling_replication')
        arrays, means, boots = {}, {}, {}
        aggregates = []
        context = {'training_steps': steps, 'scenario': scenario, 'split': split}
        for candidate in sorted(candidates, key=_candidate_key):
            indexed = by_candidate[candidate]
            values = np.asarray([[[indexed[(train, traffic)][metric] for metric in aggregate_metrics]
                                  for traffic in evaluation_seeds] for train in training_seeds])
            arrays[candidate] = values
            means[candidate] = values.mean(axis=(0, 1))
            boots[candidate] = values[train_draws[:, :, None], traffic_draws[:, None, :]].mean(axis=(1, 2))
            aggregate = {
                'candidate_id': candidate, **context,
                **dict(zip(('alpha', 'beta', 'switch_gamma'), candidate_weights[candidate], strict=True)),
                'episode_count': nt * ne, 'training_seed_count': nt, 'evaluation_seed_count': ne,
                'ci_scope': scope,
                'collisions_total': sum(row['collisions'] for row in indexed.values()),
                'teleports_total': sum(row['teleports'] for row in indexed.values()),
                'maximum_waiting_worst_episode': float(values[:, :, 1].max()),
            }
            for index, metric in enumerate(aggregate_metrics):
                low, high = _ci(boots[candidate][:, index], estimable)
                aggregate.update({metric: float(means[candidate][index]),
                                  metric + '_std': _std(values[:, :, index].ravel()),
                                  metric + '_training_seed_std': _std(values[:, :, index].mean(axis=1)),
                                  metric + '_ci95_low': low, metric + '_ci95_high': high})
                if metric in diagnostics:
                    aggregate[metric + '_worst_episode'] = float(values[:, :, index].max())
            aggregates.append(aggregate)

        reference = arrays[incumbent_id]
        for candidate in sorted(candidates, key=_candidate_key):
            differences = arrays[candidate] - reference
            comparison = {'candidate_id': candidate, 'incumbent_candidate_id': incumbent_id,
                          **context, 'pair_count': nt * ne, 'training_seed_count': nt,
                          'evaluation_seed_count': ne, 'ci_scope': scope}
            for index, metric in enumerate(METRICS):
                delta_samples = boots[candidate][:, index] - boots[incumbent_id][:, index]
                low, high = _ci(delta_samples, estimable)
                reference_boot = boots[incumbent_id][:, index]
                pct_samples = np.divide(100 * delta_samples, reference_boot,
                                        out=np.full_like(delta_samples, np.nan), where=reference_boot != 0)
                pct_low, pct_high = _ci(pct_samples, estimable)
                reference_values = reference[:, :, index]
                cell_pct = np.divide(100 * differences[:, :, index], reference_values,
                                     out=np.full_like(reference_values, np.nan), where=reference_values != 0)
                oriented = differences[:, :, index] * _orientation(metric)
                comparison.update({
                    metric + '_difference': float(differences[:, :, index].mean()),
                    metric + '_difference_ci95_low': low, metric + '_difference_ci95_high': high,
                    metric + '_change_pct': _percent(means[candidate][index], means[incumbent_id][index]),
                    metric + '_change_pct_ci95_low': pct_low, metric + '_change_pct_ci95_high': pct_high,
                    metric + '_mean_paired_change_pct': float(cell_pct.mean()) if np.isfinite(cell_pct).all() else None,
                    metric + '_wins': int(np.sum(oriented > TIE_ABSOLUTE_TOLERANCE)),
                    metric + '_ties': int(np.sum(np.abs(oriented) <= TIE_ABSOLUTE_TOLERANCE)),
                    metric + '_losses': int(np.sum(oriented < -TIE_ABSOLUTE_TOLERANCE)),
                })
            comparison['paired_waiting_ci_below_zero'] = (
                comparison['avg_waiting_time_difference_ci95_high'] is not None
                and comparison['avg_waiting_time_difference_ci95_high'] < 0)
            result['paired_comparisons'].append(comparison)
            for ti, train in enumerate(training_seeds):
                seed_comparison = {'candidate_id': candidate, 'incumbent_candidate_id': incumbent_id,
                                   **context, 'training_seed': train, 'evaluation_seed_count': ne}
                for mi, metric in enumerate(METRICS):
                    delta = float(differences[ti, :, mi].mean())
                    oriented = delta * _orientation(metric)
                    seed_comparison.update({metric + '_difference': delta,
                                            metric + '_change_pct': _percent(arrays[candidate][ti, :, mi].mean(), reference[ti, :, mi].mean()),
                                            metric + '_improved': oriented > TIE_ABSOLUTE_TOLERANCE,
                                            metric + '_not_worse': oriented >= -TIE_ABSOLUTE_TOLERANCE})
                result['training_seed_comparisons'].append(seed_comparison)
                for ei, traffic in enumerate(evaluation_seeds):
                    pair = {'candidate_id': candidate, 'incumbent_candidate_id': incumbent_id,
                            **context, 'training_seed': train, 'evaluation_seed': traffic}
                    for mi, metric in enumerate(METRICS):
                        pair[metric + '_difference'] = float(differences[ti, ei, mi])
                        pair[metric + '_change_pct'] = _percent(arrays[candidate][ti, ei, mi], reference[ti, ei, mi])
                    result['paired_rows'].append(pair)

        reference_summary = next(row for row in aggregates if row['candidate_id'] == incumbent_id)
        for row in aggregates:
            row['safety_passed'] = row['collisions_total'] == 0 and row['teleports_total'] == 0
            row['throughput_retention'] = (row['throughput'] / reference_summary['throughput']
                                           if reference_summary['throughput'] else None)
            row['throughput_constraint_passed'] = row['throughput'] >= THROUGHPUT_RETENTION * reference_summary['throughput']
            row['max_waiting_constraint_passed'] = row['max_waiting_time'] <= reference_summary['max_waiting_time']
            row['eligible'] = row['safety_passed'] and row['throughput_constraint_passed'] and row['max_waiting_constraint_passed']
        frontier = pareto_front(aggregates)
        eligible_frontier = pareto_front(row for row in aggregates if row['eligible'])
        for row in aggregates:
            row['pareto_nondominated'] = row['candidate_id'] in frontier
        result['aggregates'].extend(aggregates)
        result['pareto_fronts'].append({**context, 'candidate_ids': frontier,
                                        'eligible_candidate_ids': eligible_frontier})
        if split == 'validation':
            ranking = sorted((row for row in aggregates if row['eligible']), key=lambda row: (
                row['avg_waiting_time'], row['max_waiting_time'], row['avg_queue'],
                row['phase_changes'], -row['throughput'], _candidate_key(row['candidate_id'])))
            result['selection'].append({
                **context, 'incumbent_candidate_id': incumbent_id,
                'winner_candidate_id': ranking[0]['candidate_id'] if ranking else None,
                'ranked_candidate_ids': [row['candidate_id'] for row in ranking],
                'selection_locked_before_holdout': True,
                'selection_rules': {
                    'reject_any_collisions_or_teleports': True,
                    'minimum_throughput_retention': THROUGHPUT_RETENTION,
                    'mean_max_waiting_not_worse_than_incumbent': True,
                    'lexicographic_metric_order': [*METRICS[:4], '-throughput', 'candidate_id'],
                    'candidate_id_tie_order': 'numeric for integer IDs; lexical for named IDs',
                    'weighted_score_used': False, 'holdout_reselection_permitted': False,
                },
            })
    return result
