"""Bounded reward search with validation-locked selection and unseen holdout demand.

Run with a fresh folder, for example::

    python -m model.experiments.quick_reward_search --output model/results/reward_sensitivity_local/2026-10-09-quick

Every candidate starts from scratch. The holdout confirms the validation choice;
it never chooses a different candidate after seeing test results.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from model.controller.traffic_dqn import TrafficDQN
from model.experiments.reward_sensitivity import (
    AuditedEnv, METRICS, aggregate, sha256, worker, write_csv, write_json,
)
from model.utils.config import ProjectConfig, RewardConfig


CANDIDATES = {
    1: (0.10, 0.10, 0.20),
    2: (0.30, 0.30, 0.50),
    3: (0.50, 0.30, 0.50),
    4: (0.30, 0.50, 0.50),
    5: (0.30, 0.30, 0.20),
    6: (0.30, 0.30, 0.80),
}
BASELINE = 1
THROUGHPUT_RETENTION = 0.95
DIAGNOSTIC_METRICS = (
    'max_queue', 'departed', 'unfinished', 'pending', 'max_pending',
    'forced_changes', 'duration', 'episode_reward',
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def summarize_episodes(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError('An evaluation must contain at least one episode')
    summary = {'evaluation_episodes': len(rows)}
    for metric in (*METRICS, *DIAGNOSTIC_METRICS):
        values = np.asarray([float(row[metric]) for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite evaluation metric: {metric}')
        summary[metric] = float(values.mean())
        summary[metric + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    summary['maximum_waiting_worst_episode'] = max(float(row['max_waiting_time']) for row in rows)
    summary['maximum_pending_worst_episode'] = max(int(row['max_pending']) for row in rows)
    summary['collisions_total'] = sum(int(row['collisions']) for row in rows)
    summary['teleports_total'] = sum(int(row['teleports']) for row in rows)
    return summary


def eligibility(summary: dict, baseline: dict, rows: list[dict]) -> dict:
    collisions = sum(int(row['collisions']) for row in rows)
    teleports = sum(int(row['teleports']) for row in rows)
    throughput = float(summary['throughput'])
    reference = float(baseline['throughput'])
    finite = all(math.isfinite(float(summary[metric])) for metric in METRICS)
    retained = finite and throughput >= reference * THROUGHPUT_RETENTION
    return {
        'collisions': collisions,
        'teleports': teleports,
        'throughput_retention': throughput / reference if reference > 0 else None,
        'throughput_constraint_passed': retained,
        'eligible': finite and collisions == 0 and teleports == 0 and retained,
    }


def select_validation(summaries: list[dict], episode_rows: dict[int, list[dict]]) -> dict:
    """Pure selection, used only on validation data before holdout is opened."""
    by_candidate = {int(row['experiment']): row for row in summaries}
    if BASELINE not in by_candidate:
        raise ValueError('A completed fresh baseline is required')
    baseline = by_candidate[BASELINE]
    eligible = {
        candidate: eligibility(row, baseline, episode_rows[candidate])
        for candidate, row in by_candidate.items()
    }
    ranked = sorted(
        (row for candidate, row in by_candidate.items() if eligible[candidate]['eligible']),
        key=lambda row: (
            row['avg_waiting_time'], row['max_waiting_time'], row['avg_queue'],
            row['phase_changes'], -row['throughput'], row['experiment'],
        ),
    )
    shortlist = [int(row['experiment']) for row in ranked[:2]]
    return {
        'validation_winner_candidate_id': shortlist[0] if shortlist else None,
        'shortlist_candidate_ids': shortlist,
        'baseline_candidate_id': BASELINE,
        'holdout_candidate_ids': sorted(set([BASELINE, *shortlist])),
        'eligibility': eligible,
        'selection_locked_before_holdout': True,
        'selection_rules': {
            'reject_any_collisions_or_teleports': True,
            'minimum_throughput_retention': THROUGHPUT_RETENTION,
            'lexicographic_metric_order': [*METRICS[:4], '-throughput', 'candidate_id'],
            'shortlist_size': 2,
            'baseline_always_evaluated_separately': True,
            'holdout_reselection_permitted': False,
        },
    }


def confirm_locked_winner(selection: dict, summaries: list[dict],
                          episode_rows: dict[int, list[dict]]) -> dict:
    """Confirm or reject the locked winner without replacing it with a runner-up."""
    winner = selection['validation_winner_candidate_id']
    by_candidate = {int(row['experiment']): row for row in summaries}
    if winner is None:
        return {'candidate_id': None, 'confirmed': False, 'reason': 'no_eligible_validation_candidate'}
    if winner not in by_candidate or BASELINE not in by_candidate:
        return {'candidate_id': winner, 'confirmed': False, 'reason': 'holdout_evaluation_failed'}
    baseline, candidate = by_candidate[BASELINE], by_candidate[winner]
    checks = eligibility(candidate, baseline, episode_rows[winner])
    waiting_passed = candidate['avg_waiting_time'] <= baseline['avg_waiting_time']
    confirmed = checks['eligible'] and waiting_passed
    differences = [
        float(c['avg_waiting_time']) - float(b['avg_waiting_time'])
        for c, b in zip(episode_rows[winner], episode_rows[BASELINE], strict=True)
    ]
    return {
        'candidate_id': winner,
        'confirmed': confirmed,
        'reason': ('baseline_remains_selected' if winner == BASELINE else
                   'holdout_constraints_and_mean_waiting_passed' if confirmed else
                   'holdout_constraints_or_mean_waiting_failed'),
        'baseline_is_winner': winner == BASELINE,
        'mean_waiting_not_worse_than_baseline': waiting_passed,
        'mean_paired_waiting_difference_seconds': float(np.mean(differences)),
        'mean_waiting_change_pct': (
            (candidate['avg_waiting_time'] / baseline['avg_waiting_time'] - 1) * 100
            if baseline['avg_waiting_time'] > 0 else None
        ),
        **checks,
        'statistical_significance_established': False,
        'holdout_reselection_performed': False,
    }


def evaluate_holdout(output: Path, candidate: int | None, seeds: list[int]) -> dict:
    """Evaluate a saved checkpoint or Fixed-Time using isolated SUMO files."""
    label = 'fixed_time' if candidate is None else f'experiment_{candidate}'
    folder = output / 'holdout' / label
    folder.mkdir(parents=True, exist_ok=False)
    source = output / f'experiment_{BASELINE if candidate is None else candidate}'
    metadata = read_json(source / 'experiment_config.json')
    sumo_folder = folder / 'sumo'
    sumo_folder.mkdir()
    for filename in ('intersection.net.xml', 'simulation.sumocfg'):
        shutil.copy2(source / 'sumo' / filename, sumo_folder / filename)
    config = replace(
        ProjectConfig(), sumo_dir=sumo_folder, results_dir=folder,
        reward=RewardConfig(**metadata['reward_config']),
    )
    if sha256(config.network_file) != metadata['network_sha256']:
        raise ValueError('Holdout network differs from training network')
    model = None
    if candidate is not None:
        model_path = source / 'final.zip'
        if sha256(model_path) != metadata['model_sha256']:
            raise ValueError('Holdout checkpoint differs from saved training checkpoint')
        model = TrafficDQN.load(str(model_path), device='cpu')
        if model.observation_space.shape != (60,) or model.action_space.n != 8:
            raise ValueError('Holdout checkpoint does not use current 60x8 structure')
    write_json(folder / 'evaluation_config.json', {
        'candidate_id': candidate, 'controller': 'Fixed-Time' if model is None else 'DQN',
        'source_checkpoint': str((source / 'final.zip').resolve()) if model is not None else None,
        'source_model_sha256': metadata['model_sha256'] if model is not None else None,
        'network_sha256': metadata['network_sha256'], 'reward_config': metadata['reward_config'],
        'seeds': seeds, 'scenario': 'random', 'episode_seconds': 300,
        'deterministic_actions': True, 'used_for_candidate_selection': False,
    })
    env = AuditedEnv(
        config, controller_name='Fixed-Time' if model is None else 'Holdout-DQN',
        scenario='random', episode_seconds=300, cache=True,
        manifest=folder / 'evaluation_routes.json',
    )
    rows = []
    try:
        for episode, seed in enumerate(seeds):
            observation, _ = env.reset(seed=seed)
            done = False
            while not done:
                action = 0 if model is None else int(model.predict(observation, deterministic=True)[0])
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            row = env.episode_summary(episode)
            row['experiment'] = 'fixed_time' if candidate is None else candidate
            rows.append(row)
            write_csv(folder / 'evaluation_metrics.csv', rows)
            write_json(folder / 'progress.json', {
                'status': 'EVALUATING', 'completed_episodes': len(rows), 'total_episodes': len(seeds),
            })
        summary = {'experiment': 'fixed_time' if candidate is None else candidate,
                   **summarize_episodes(rows)}
        write_json(folder / 'summary.json', summary)
        write_json(folder / 'progress.json', {'status': 'COMPLETED', **summary})
        return summary
    except BaseException as exc:
        write_json(folder / 'progress.json', {'status': 'FAILED', 'error': repr(exc)})
        raise
    finally:
        env.close()


def verify_holdout_pairing(output: Path, summaries: list[dict]) -> None:
    manifests = []
    for summary in summaries:
        candidate = summary['experiment']
        label = 'fixed_time' if candidate == 'fixed_time' else f'experiment_{candidate}'
        manifests.append(read_json(output / 'holdout' / label / 'evaluation_routes.json'))
    if any(manifest != manifests[0] for manifest in manifests[1:]):
        raise RuntimeError('Paired holdout demand manifests differ')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timesteps', type=int, default=50_000)
    parser.add_argument('--schedule-timesteps', type=int, default=150_000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--eval-episodes', type=int, default=8)
    parser.add_argument('--eval-seed-start', type=int, default=4_001)
    parser.add_argument('--holdout-episodes', type=int, default=10)
    parser.add_argument('--holdout-seed-start', type=int, default=5_001)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--only', type=int, nargs='+', choices=list(CANDIDATES))
    args = parser.parse_args()
    for key in ('timesteps', 'schedule_timesteps', 'workers', 'eval_episodes', 'holdout_episodes'):
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.schedule_timesteps < args.timesteps:
        parser.error('--schedule-timesteps must be at least --timesteps')
    validation_seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
    holdout_seeds = list(range(args.holdout_seed_start, args.holdout_seed_start + args.holdout_episodes))
    # Random training resets draw from 1..1000; the first explicit seed is also used.
    training_seeds = set(range(1, 1_001)) | {args.seed}
    if set(validation_seeds) & set(holdout_seeds):
        parser.error('Validation and holdout seed ranges must be disjoint')
    if training_seeds & (set(validation_seeds) | set(holdout_seeds)):
        parser.error('Evaluation seeds must not overlap training seeds (1..1000 and --seed)')
    output = args.output.resolve()
    if output.exists():
        parser.error('--output must name a fresh, nonexistent directory')
    output.mkdir(parents=True, exist_ok=False)
    chosen = sorted(set(args.only or CANDIDATES) | {BASELINE})
    settings = {
        'output': str(output), 'timesteps': args.timesteps,
        'schedule_timesteps': args.schedule_timesteps, 'seed': args.seed,
        'scenario': 'random', 'episode_seconds': 300,
        'eval_seed_start': args.eval_seed_start, 'eval_episodes': args.eval_episodes,
        'cache': True, 'verify_cache': False,
    }
    write_json(output / 'search_config.json', {
        **settings, 'candidate_weights': {i: CANDIDATES[i] for i in chosen},
        'baseline_candidate_id': BASELINE, 'workers': args.workers,
        'validation_seeds': validation_seeds, 'holdout_seeds': holdout_seeds,
        'queue_weight': 1.0, 'fresh_training_per_candidate': True,
        'source_sha256': sha256(Path(__file__)),
        'selection_metric': 'validation average waiting with safety and throughput constraints',
        'holdout_confirmation_rule': 'locked winner has no safety event, >=95% baseline throughput, and mean waiting <= baseline',
        'holdout_used_for_reselection': False,
        'training_seed_count': 1, 'global_optimum_established': False,
        'statistical_significance_established': False,
    })
    started = time.monotonic()
    failures = []
    completed = []
    try:
        write_json(output / 'status.json', {'status': 'TRAINING', 'candidate_ids': chosen})
        with ProcessPoolExecutor(max_workers=min(args.workers, len(chosen))) as pool:
            futures = {
                pool.submit(worker, candidate, {**settings, 'weights': CANDIDATES[candidate]}): candidate
                for candidate in chosen
            }
            for future in as_completed(futures):
                candidate = futures[future]
                try:
                    future.result()
                    completed.append(candidate)
                    print(json.dumps({'completed_candidate': candidate}), flush=True)
                except Exception as exc:
                    failures.append({'stage': 'training_or_validation', 'candidate': candidate, 'error': repr(exc)})
                    write_json(output / 'failures.json', failures)
                    print(json.dumps({'failed_candidate': candidate, 'error': repr(exc)}), flush=True)
                write_json(output / 'status.json', {
                    'status': 'TRAINING', 'completed_candidate_ids': sorted(completed),
                    'failed_candidate_ids': [row['candidate'] for row in failures],
                    'elapsed_seconds': time.monotonic() - started,
                })
        if BASELINE not in completed:
            raise RuntimeError('Fresh baseline failed; no valid reward comparison is available')
        summaries = aggregate(output, sorted(completed))
        validation_rows = {
            candidate: read_csv(output / f'experiment_{candidate}' / 'evaluation_metrics.csv')
            for candidate in completed
        }
        selection = select_validation(summaries, validation_rows)
        selection['failed_trials'] = list(failures)
        write_json(output / 'locked_selection.json', selection)
        # Include pending and unfinished diagnostics in a validation table without
        # changing the original worker's established comparison schema.
        validation_diagnostics = [
            {'experiment': candidate, **summarize_episodes(validation_rows[candidate])}
            for candidate in sorted(completed)
        ]
        write_csv(output / 'validation_diagnostics.csv', validation_diagnostics)
        write_json(output / 'status.json', {
            'status': 'HOLDOUT', 'locked_winner_candidate_id': selection['validation_winner_candidate_id'],
            'holdout_candidate_ids': selection['holdout_candidate_ids'],
        })
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        holdout_summaries = []
        for candidate in [*selection['holdout_candidate_ids'], None]:
            try:
                result = evaluate_holdout(output, candidate, holdout_seeds)
                holdout_summaries.append(result)
                print(json.dumps({'completed_holdout': result['experiment']}), flush=True)
            except Exception as exc:
                failures.append({'stage': 'holdout', 'candidate': candidate, 'error': repr(exc)})
                write_json(output / 'failures.json', failures)
                print(json.dumps({'failed_holdout': candidate, 'error': repr(exc)}), flush=True)
        if holdout_summaries:
            verify_holdout_pairing(output, holdout_summaries)
            write_csv(output / 'holdout_comparison.csv', holdout_summaries)
            write_json(output / 'holdout_comparison.json', {
                'paired_route_hashes_verified': True, 'results': holdout_summaries,
            })
        candidate_summaries = [row for row in holdout_summaries if row['experiment'] != 'fixed_time']
        holdout_rows = {
            int(row['experiment']): read_csv(
                output / 'holdout' / f"experiment_{row['experiment']}" / 'evaluation_metrics.csv')
            for row in candidate_summaries
        }
        confirmation = confirm_locked_winner(selection, candidate_summaries, holdout_rows)
        winner = selection['validation_winner_candidate_id']
        recommendation = {
            'validation_winner_candidate_id': winner,
            'validation_winner_weights': CANDIDATES[winner] if winner is not None else None,
            'holdout_confirmation': confirmation,
            'recommended_candidate_id': winner if confirmation['confirmed'] else None,
            'recommended_weights': CANDIDATES[winner] if confirmation['confirmed'] else None,
            'default_configuration_changed': False,
            'model_path': str(output / f'experiment_{winner}' / 'final.zip') if winner is not None else None,
            'training_steps_per_candidate': args.timesteps,
            'exploration_schedule_steps': args.schedule_timesteps,
            'learning_started': args.timesteps > ProjectConfig().dqn.learning_starts,
            'training_seeds': [args.seed], 'validation_seeds': validation_seeds,
            'holdout_seeds': holdout_seeds, 'scenario': 'random',
            'global_optimum_established': False,
            'statistical_significance_established': False,
            'scope': 'single-training-seed bounded candidate search and random-traffic holdout',
            'failed_trials': failures,
        }
        write_json(output / 'recommendation.json', recommendation)
        write_json(output / 'status.json', {
            'status': 'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED',
            'elapsed_seconds': time.monotonic() - started,
            'completed_candidate_ids': sorted(completed),
            'holdout_confirmed': confirmation['confirmed'], 'failures': failures,
        })
        print(json.dumps(recommendation, ensure_ascii=False, indent=2), flush=True)
        return 1 if failures else 0
    except BaseException as exc:
        write_json(output / 'status.json', {
            'status': 'FAILED', 'error': repr(exc),
            'elapsed_seconds': time.monotonic() - started, 'failures': failures,
        })
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
