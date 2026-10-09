"""Paired local reward search using the repository's unchanged training worker.

python -m model.experiments.fine_reward_search --output model/results/reward_fine_search_2026-10-09 --workers 6
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
_mpl_cache = ROOT / '.tmp' / 'fine_search_matplotlib'
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_mpl_cache))

import numpy as np
import torch

from model.controller.traffic_dqn import TrafficDQN
from model.experiments.quick_reward_search import read_csv, read_json
from model.experiments.reward_sensitivity import AuditedEnv, METRICS, sha256, worker, write_csv, write_json
from model.utils.config import ProjectConfig, RewardConfig

INCUMBENT = (0.3, 0.5, 0.5)
GRID = {i: tuple(weights) for i, weights in enumerate(itertools.product(
    (0.2, 0.3, 0.4), (0.4, 0.5, 0.6), (0.4, 0.5, 0.6)), 1)}
INCUMBENT_ID = next(i for i, w in GRID.items() if w == INCUMBENT)
SCENARIOS = ('uniform', 'north_south_congested', 'east_west_congested',
             'left_turn_congested', 'heavy', 'low')
CORE_FILES = ('model/utils/config.py', 'model/env/reward.py', 'model/env/intersection_env.py',
              'model/env/state_provider.py', 'model/controller/signal_controller.py',
              'model/controller/traffic_dqn.py', 'model/training.py',
              'model/traffic/route_generator.py', 'model/utils/metrics.py',
              'model/experiments/reward_sensitivity.py',
              'model/experiments/subscription_connection.py', 'model/sumo/intersection.net.xml')


def now():
    return datetime.now(timezone.utc).isoformat()


def core_hashes():
    return {p: sha256(ROOT / p) for p in CORE_FILES if (ROOT / p).is_file()}


def current_runtime():
    return {'python': platform.python_version(), **{p: importlib.metadata.version(p)
            for p in ('numpy', 'torch', 'stable-baselines3', 'gymnasium', 'eclipse-sumo')}}


def initialize(output: Path, workers: int, resume: bool):
    path = output / 'search_config.json'
    if path.exists():
        if not resume:
            raise ValueError('Existing experiment requires --resume; results will not be overwritten')
        config = read_json(path)
        if config['core_sha256'] != core_hashes():
            raise ValueError('Immutable source files changed since this experiment began')
        if config['runtime'] != current_runtime():
            raise ValueError('Runtime changed since this experiment began; cannot mix trained models')
        return config
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output folder must be fresh')
    output.mkdir(parents=True, exist_ok=True)
    base = ProjectConfig()
    config = {
        'created_at': now(), 'incumbent_id': INCUMBENT_ID,
        'incumbent_weights': INCUMBENT, 'candidate_weights': GRID,
        'grid_axes': [[0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.4, 0.5, 0.6]],
        'fine_training_steps': 50000, 'fine_training_seed': 22,
        'multiseed_training_steps': 150000, 'training_seeds': [22, 42, 62],
        'schedule_timesteps': 150000, 'validation_seeds': list(range(8001, 8021)),
        'holdout_seeds': list(range(9001, 9031)),
        'robustness_seeds': list(range(10001, 10011)), 'robustness_scenarios': SCENARIOS,
        'previously_used_seeds_excluded': ['4001-4008', '5001-5010', '6001-6002', '7001-7020'],
        'episode_seconds': 300, 'demand_seconds': 240, 'queue_weight': 1.0,
        'workers': workers, 'core_sha256': core_hashes(),
        'project_config': asdict(base), 'fresh_training_per_candidate_and_seed': True,
        'observation_shape': [60], 'action_count': 8,
        'selection_rules': {
            'baseline': 'fresh incumbent trained at the same steps and training seed',
            'safety_events_allowed': 0, 'minimum_throughput_retention': 0.95,
            'maximum_waiting_mean_must_not_increase': True,
            'lexicographic_order': [*METRICS[:4], '-throughput', 'candidate_id'],
            'shortlist': 'top three eligible candidates plus incumbent, deduplicated',
            'multiseed_direction': 'mean waiting must not worsen for any training seed',
            'holdout_reselection_permitted': False,
            'robustness_used_for_selection': False,
            'robustness_maximum_waiting_increase_pct': 0.0,
            'robustness_mean_waiting_increase_limit_pct': 5.0,
            'robustness_failed_confirmation': 'retain incumbent; never promote runner-up',
        },
        'boundary_protocol': {
            'step': 0.1, 'leaders': 3, 'max_rounds': 2,
            'method': 'extend outward one axis at a time at each top-three boundary point',
            'unresolved_boundary': 'report limited search; do not claim a local or global optimum',
        },
        'bootstrap_samples': 5000, 'bootstrap_seed': 1729,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'runtime': current_runtime(),
        'prior_results_preserved': 'model/results/reward_search_2026-10-09',
        'prior_models_not_used_as_comparison_baselines': True,
        'execution_notes': [
            'The unchanged existing Progress callback stops at the requested sample count.',
            'Its terminal callback interrupts the last rollout before its final update; the same convention is used for every model.',
            'Checkpoint resume skips only completed audited runs. Interrupted training restarts fresh in a new attempt folder.',
            'Holdout and robustness are confirmation-only after a validation selection lock.',
        ],
    }
    # Keep an immutable byte snapshot of the prior result ledger for later verification.
    config['prior_results_sha256'] = {str(p.relative_to(ROOT)): sha256(p)
        for p in sorted((ROOT / config['prior_results_preserved']).rglob('*')) if p.is_file()}
    write_json(path, config)
    return read_json(path)


def write_status(output, stage, **extra):
    write_json(output / 'status.json', {'status': stage, 'updated_at': now(), **extra})


def checkpoint_fingerprints(results):
    fingerprints = []
    for result in results:
        job, folder = result['job'], Path(result['folder'])
        metadata = read_json(folder / 'experiment_config.json')
        actual = sha256(folder / 'final.zip')
        if actual != metadata['model_sha256']:
            raise ValueError('Model changed before selection was locked')
        fingerprints.append({'candidate_id': job['candidate_id'], 'training_seed': job['training_seed'],
            'training_steps': job['training_steps'], 'weights': job['weights'], 'model_sha256': actual,
            'training_demand_sha256': sha256(folder / 'training_routes.json'),
            'evaluation_demand_sha256': sha256(folder / 'evaluation_routes.json'),
            'evaluation_metrics_sha256': sha256(folder / 'evaluation_metrics.csv')})
    return sorted(fingerprints, key=lambda r: (r['candidate_id'], r['training_seed']))


def lock_payload(path, payload, identity_fields):
    """An existing decision must match all defining evidence, not just its ID."""
    identity = {key: payload[key] for key in identity_fields}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()
    if path.exists():
        previous = read_json(path)
        previous_identity = {key: previous[key] for key in identity_fields}
        if previous_identity != identity:
            raise ValueError(f'Locked decision or its evidence changed: {path.name}')
        return previous
    payload = {**payload, 'protocol_fingerprint': digest}
    write_json(path, payload)
    return payload


def freeze_protocol(output):
    files = ('model/experiments/fine_reward_search.py', 'model/experiments/fine_search_analysis.py')
    payload = {'source_sha256': {p: sha256(ROOT / p) for p in files},
               'rules_sha256': sha256(output / 'search_config.json')}
    lock_payload(output / 'protocol_code_lock.json', payload, ('source_sha256', 'rules_sha256'))


def verify_immutable(config):
    if core_hashes() != config['core_sha256']:
        raise RuntimeError('Core experimental conditions changed during execution')
    if config['runtime'] != current_runtime():
        raise RuntimeError('Runtime changed during execution')
    for path, expected in config['prior_results_sha256'].items():
        if not (ROOT / path).is_file() or sha256(ROOT / path) != expected:
            raise RuntimeError(f'Preserved prior result changed: {path}')


def preflight(output, config):
    stamp = output / 'preflight.json'
    if stamp.exists() and read_json(stamp).get('passed'):
        return
    from stable_baselines3.common.env_checker import check_env
    folder = output / 'preflight'
    folder.mkdir(exist_ok=True)
    sumo_folder = folder / 'sumo'
    sumo_folder.mkdir(exist_ok=True)
    for name in ('intersection.net.xml', 'simulation.sumocfg'):
        shutil.copy2(ProjectConfig().sumo_dir / name, sumo_folder / name)
    cfg = replace(ProjectConfig(), sumo_dir=sumo_folder, results_dir=folder)
    env = AuditedEnv(cfg, manifest=folder / 'routes.json', cache=True, verify_cache=True,
                     controller_name='Fine-Preflight', scenario='random', episode_seconds=300)
    try:
        check_env(env, warn=True)
        obs, _ = env.reset(seed=11001)
        for t in range(300):
            obs, reward, terminated, truncated, info = env.step((t // 10) % 8)
            assert obs.shape == (60,) and np.isfinite(obs).all()
            assert env.action_space.n == 8
            if terminated or truncated:
                break
        summary = env.episode_summary(0)
        assert summary['collisions'] == 0 and summary['teleports'] == 0
        write_json(stamp, {'passed': True, 'check_env': True, 'cache_equivalence_checked': True,
                           'all_eight_actions_requested': True, 'episode': summary, 'updated_at': now()})
    finally:
        env.close()


def run_training_job(job):
    """Use exactly the pre-existing worker; every attempt creates a new model."""
    candidate = job['candidate_id']
    parent = Path(job['parent'])
    attempts = sorted(parent.glob('attempt_*')) if parent.exists() else []
    for attempt in reversed(attempts):
        folder = attempt / f'experiment_{candidate}'
        if (folder / 'summary.json').exists() and (attempt / 'job.json').exists():
            recorded = read_json(attempt / 'job.json')
            if recorded != job:
                raise ValueError('Completed job protocol mismatch')
            metadata = read_json(folder / 'experiment_config.json')
            if metadata.get('completed_training_steps') != job['training_steps']:
                raise ValueError('Completed checkpoint step mismatch')
            if sha256(folder / 'final.zip') != metadata['model_sha256']:
                raise ValueError('Completed checkpoint hash mismatch')
            return {'job': job, 'folder': str(folder), 'reused_completed_run': True}
    attempt = parent / f'attempt_{len(attempts) + 1:03d}'
    attempt.mkdir(parents=True, exist_ok=False)
    write_json(attempt / 'job.json', job)
    settings = {
        'output': str(attempt), 'timesteps': job['training_steps'],
        'schedule_timesteps': 150000, 'seed': job['training_seed'],
        'scenario': 'random', 'episode_seconds': 300, 'weights': job['weights'],
        'eval_seed_start': 8001, 'eval_episodes': 20, 'cache': True, 'verify_cache': False,
    }
    worker(candidate, settings)
    return {'job': job, 'folder': str(attempt / f'experiment_{candidate}'), 'reused_completed_run': False}


def jobs_for(output, stage, ids, weights, training_seeds, steps):
    return [{'stage': stage, 'candidate_id': i, 'weights': list(weights[str(i)]),
             'training_seed': seed, 'training_steps': steps,
             'parent': str((output / 'runs' / stage / f'seed_{seed}' / f'candidate_{i}').resolve())}
            for seed in training_seeds for i in ids]


def run_jobs(output, stage, jobs, workers):
    completed, failures = [], []
    write_status(output, stage.upper(), total_jobs=len(jobs), completed_jobs=0)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_training_job, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                completed.append(result)
                print(json.dumps({'completed': stage, 'candidate': job['candidate_id'],
                                  'training_seed': job['training_seed'], 'folder': result['folder']}), flush=True)
            except Exception as exc:
                failures.append({'job': job, 'error': repr(exc), 'traceback': traceback.format_exc()})
                print(json.dumps({'failed': stage, 'candidate': job['candidate_id'], 'error': repr(exc)}), flush=True)
            write_json(output / f'{stage}_jobs.json', {'completed': completed, 'failures': failures})
            write_status(output, stage.upper(), total_jobs=len(jobs), completed_jobs=len(completed),
                         failed_jobs=len(failures))
    if failures:
        raise RuntimeError(f'{len(failures)} {stage} jobs failed; completed runs remain resumable')
    return completed


def normalize_rows(result, split='validation'):
    job, folder = result['job'], Path(result['folder'])
    manifest = read_json(folder / 'evaluation_routes.json')
    raw = read_csv(folder / 'evaluation_metrics.csv')
    if len(raw) != 20 or [int(r['seed']) for r in raw] != list(range(8001, 8021)):
        raise ValueError('Validation demand sequence mismatch')
    by_seed = {r['seed']: r['route_sha256'] for r in manifest}
    rows = []
    for r in raw:
        alpha, beta, switch = job['weights']
        rows.append({**r, 'candidate_id': job['candidate_id'], 'waiting_weight': alpha,
            'max_waiting_weight': beta, 'switch_penalty': switch,
            'alpha': alpha, 'beta': beta, 'switch_gamma': switch,
            'training_seed': job['training_seed'], 'train_seed': job['training_seed'],
            'training_steps': job['training_steps'], 'evaluation_seed': int(r['seed']),
            'scenario': 'random', 'split': split, 'is_incumbent': job['candidate_id'] == INCUMBENT_ID,
            'demand_sha256': by_seed[int(r['seed'])], 'model_path': str(folder / 'final.zip')})
    return rows


def verify_pairing(results):
    by_training_seed = {}
    for result in results:
        seed = result['job']['training_seed']
        pair = tuple(read_json(Path(result['folder']) / filename)
                     for filename in ('training_routes.json', 'evaluation_routes.json'))
        if seed in by_training_seed and by_training_seed[seed] != pair:
            raise RuntimeError(f'Paired training/evaluation demand hash mismatch for training seed {seed}')
        by_training_seed[seed] = pair


def verify_evaluation_hashes(rows):
    seen = {}
    for row in rows:
        key = (row['scenario'], int(row['evaluation_seed']))
        value = row['demand_sha256']
        if key in seen and seen[key] != value:
            raise RuntimeError(f'Evaluation demand differs across compared runs: {key}')
        seen[key] = value
    return {'verified': True, 'unique_demands': len(seen), 'evaluation_rows': len(rows)}


def basic_ranking(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row['candidate_id']), []).append(row)
    summaries = {i: {m: float(np.mean([float(r[m]) for r in group])) for m in METRICS}
                 for i, group in grouped.items()}
    reference = summaries[INCUMBENT_ID]
    eligible = []
    for i, stats in summaries.items():
        group = grouped[i]
        safe = all(int(r['collisions']) == 0 and int(r['teleports']) == 0 for r in group)
        if safe and stats['throughput'] >= .95 * reference['throughput'] and stats['max_waiting_time'] <= reference['max_waiting_time'] + 1e-9:
            eligible.append(i)
    return sorted(eligible, key=lambda i: tuple(summaries[i][m] * (-1 if m == 'throughput' else 1)
                                               for m in METRICS) + (i,))


def analyze_and_save(output, rows, name, split, config):
    from model.experiments.fine_search_analysis import analyze_episodes
    hashes = verify_evaluation_hashes(rows)
    analysis = analyze_episodes(rows, INCUMBENT_ID, split=split,
                               bootstrap_samples=config['bootstrap_samples'], bootstrap_seed=config['bootstrap_seed'])
    analysis['demand_hash_verification'] = hashes
    write_json(output / f'{name}_analysis.json', analysis)
    write_csv(output / f'{name}_summary.csv', analysis['aggregates'])
    if name == 'fine_search':
        frontier = [r for r in analysis['aggregates'] if r['pareto_nondominated']]
        write_csv(output / 'pareto_front.csv', frontier)
    # Explicit percentages in each episode row, always paired on both random seeds.
    references = {(int(r['training_seed']), int(r['training_steps']), r['scenario'], int(r['evaluation_seed'])): r
                  for r in rows if int(r['candidate_id']) == INCUMBENT_ID}
    for r in rows:
        ref = references[(int(r['training_seed']), int(r['training_steps']), r['scenario'], int(r['evaluation_seed']))]
        for m in METRICS:
            base, value = float(ref[m]), float(r[m])
            r[m + '_change_pct'] = 100 * (value / base - 1) if base else (0.0 if value == 0 else None)
            r[m + '_difference'] = value - base
    write_csv(output / f'{name}_results.csv', rows)
    return analysis


def boundary_extensions(ranked, weights, axes):
    additions = []
    existing = {tuple(w) for w in weights.values()}
    for candidate in ranked[:3]:
        values = list(weights[str(candidate)])
        for axis in range(3):
            direction = -1 if values[axis] == min(axes[axis]) else 1 if values[axis] == max(axes[axis]) else 0
            if direction:
                new = values.copy()
                new[axis] = round(new[axis] + .1 * direction, 10)
                if new[axis] >= 0 and tuple(new) not in existing:
                    additions.append(tuple(new))
                    existing.add(tuple(new))
    return additions


def run_fine(output, config, workers):
    weights = dict(config['candidate_weights'])
    boundary_file = output / 'boundary_state.json'
    if boundary_file.exists():
        state = read_json(boundary_file)
        weights = state['candidate_weights']
    else:
        state = {'candidate_weights': weights, 'rounds': [], 'axes': config['grid_axes']}
    results = run_jobs(output, 'fine', jobs_for(output, 'fine', sorted(map(int, weights)), weights, [22], 50000), workers)
    while True:
        verify_pairing(results)
        rows = [row for result in results for row in normalize_rows(result)]
        analysis = analyze_and_save(output, rows, 'fine_search', 'validation', config)
        ranked = basic_ranking(rows)
        additions = boundary_extensions(ranked, weights, state['axes'])
        if not additions or len(state['rounds']) >= config['boundary_protocol']['max_rounds']:
            state['boundary_unresolved'] = bool(additions)
            state['pending_outward_points'] = additions
            write_json(boundary_file, state)
            break
        ids = []
        for point in additions:
            candidate = max(map(int, weights)) + 1
            weights[str(candidate)] = list(point)
            ids.append(candidate)
        state['rounds'].append({'candidate_ids': ids, 'points': additions, 'based_on_ranking': ranked[:3]})
        state['candidate_weights'] = weights
        state['axes'] = [sorted({w[axis] for w in weights.values()}) for axis in range(3)]
        write_json(boundary_file, state)
        results += run_jobs(output, f'boundary_{len(state["rounds"])}',
                            jobs_for(output, 'fine', ids, weights, [22], 50000), workers)
    lock = {'locked_at': now(), 'incumbent_id': INCUMBENT_ID, 'ranked_candidate_ids': ranked,
            'shortlist_candidate_ids': sorted(set(ranked[:3] + [INCUMBENT_ID])),
            'candidate_weights': weights, 'boundary_unresolved': state['boundary_unresolved'],
            'validation_seeds': config['validation_seeds'], 'holdout_opened': False,
            'fine_model_results': results, 'model_fingerprints': checkpoint_fingerprints(results)}
    locked_path = output / 'shortlist_lock.json'
    if locked_path.exists() and 'model_fingerprints' not in read_json(locked_path):
        # Preserve the first process's existing lock bytes, including downstream hashes.
        lock_payload(locked_path, lock, ('shortlist_candidate_ids', 'candidate_weights', 'validation_seeds'))
        lock_payload(output / 'shortlist_evidence_lock.json', {
            'model_fingerprints': lock['model_fingerprints'], 'shortlist_sha256': sha256(locked_path)},
            ('model_fingerprints', 'shortlist_sha256'))
    else:
        lock_payload(locked_path, lock, ('shortlist_candidate_ids', 'candidate_weights',
                                        'validation_seeds', 'model_fingerprints'))
    write_json(output / 'fine_model_registry.json', results)
    write_status(output, 'FINE_COMPLETED', completed_models=len(results), shortlist=lock['shortlist_candidate_ids'])
    return lock


def run_multiseed(output, config, workers):
    lock = read_json(output / 'shortlist_lock.json')
    actual_fine_fingerprints = checkpoint_fingerprints(lock['fine_model_results'])
    if 'model_fingerprints' in lock and lock['model_fingerprints'] != actual_fine_fingerprints:
        raise ValueError('Shortlist models or validation evidence changed')
    # Separate evidence seal also supports the first running fine-stage process.
    lock_payload(output / 'shortlist_evidence_lock.json', {'model_fingerprints': actual_fine_fingerprints,
        'shortlist_sha256': sha256(output / 'shortlist_lock.json')}, ('model_fingerprints', 'shortlist_sha256'))
    results = run_jobs(output, 'multiseed', jobs_for(output, 'multiseed', lock['shortlist_candidate_ids'],
                       lock['candidate_weights'], config['training_seeds'], 150000), workers)
    verify_pairing(results)
    rows = [row for result in results for row in normalize_rows(result)]
    analyze_and_save(output, rows, 'multiseed', 'validation', config)
    ranked = basic_ranking(rows)
    by_seed = {}
    for candidate in ranked:
        differences = []
        for seed in config['training_seeds']:
            base = np.mean([float(r['avg_waiting_time']) for r in rows
                            if int(r['candidate_id']) == INCUMBENT_ID and int(r['training_seed']) == seed])
            value = np.mean([float(r['avg_waiting_time']) for r in rows
                             if int(r['candidate_id']) == candidate and int(r['training_seed']) == seed])
            differences.append(float(value - base))
        by_seed[str(candidate)] = differences
    stable = [i for i in ranked if all(v <= 1e-9 for v in by_seed[str(i)])]
    selected = stable[0] if stable else INCUMBENT_ID
    selection = {'locked_at': now(), 'selected_candidate_id': selected, 'incumbent_id': INCUMBENT_ID,
        'selected_weights': lock['candidate_weights'][str(selected)], 'ranked_candidate_ids': ranked,
        'eligible_stable_candidate_ids': stable, 'per_training_seed_waiting_differences': by_seed,
        'training_seeds': config['training_seeds'], 'training_steps': 150000,
        'validation_seeds': config['validation_seeds'], 'locked_before_holdout': True,
        'locked_before_robustness': True, 'holdout_reselection_permitted': False,
        'candidate_weights': lock['candidate_weights'], 'model_results': results,
        'model_fingerprints': checkpoint_fingerprints(results)}
    path = output / 'selection_lock.json'
    lock_payload(path, selection, ('selected_candidate_id', 'selected_weights', 'training_seeds',
        'training_steps', 'validation_seeds', 'candidate_weights', 'model_fingerprints'))
    write_json(output / 'multiseed_model_registry.json', results)
    write_status(output, 'MULTISEED_COMPLETED', completed_models=len(results), selected_candidate_id=selected)
    return selection


def evaluate_job(job):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder, source = Path(job['folder']), Path(job['source'])
    metadata = read_json(source / 'experiment_config.json')
    checkpoint = source / 'final.zip'
    if sha256(checkpoint) != metadata['model_sha256']:
        raise ValueError('Evaluation checkpoint hash mismatch')
    if metadata['model_sha256'] != job['model_sha256']:
        raise ValueError('Evaluation model differs from locked model')
    result_path = folder / 'results.json'
    if result_path.exists():
        if read_json(folder / 'evaluation_config.json') != job:
            raise ValueError('Evaluation resume protocol mismatch')
        return read_json(result_path)
    folder.mkdir(parents=True, exist_ok=True)
    sumo_folder = folder / 'sumo'
    sumo_folder.mkdir(exist_ok=True)
    for name in ('intersection.net.xml', 'simulation.sumocfg'):
        shutil.copy2(source / 'sumo' / name, sumo_folder / name)
    cfg = replace(ProjectConfig(), sumo_dir=sumo_folder, results_dir=folder,
                  reward=RewardConfig(**metadata['reward_config']))
    if sha256(cfg.network_file) != metadata['network_sha256']:
        raise ValueError('Evaluation network mismatch')
    write_json(folder / 'evaluation_config.json', job)
    model = TrafficDQN.load(str(checkpoint), device='cpu')
    assert model.observation_space.shape == (60,) and model.action_space.n == 8
    env = AuditedEnv(cfg, controller_name='Fine-Test-DQN', scenario=job['scenario'],
                     episode_seconds=300, cache=True, manifest=folder / 'evaluation_routes.json')
    rows = []
    try:
        for episode, seed in enumerate(job['seeds']):
            obs, info = env.reset(seed=seed)
            done = False
            while not done:
                action = int(model.predict(obs, deterministic=True)[0])
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            row = env.episode_summary(episode)
            weights = job['weights']
            row.update({'candidate_id': job['candidate_id'], 'waiting_weight': weights[0],
                'max_waiting_weight': weights[1], 'switch_penalty': weights[2],
                'alpha': weights[0], 'beta': weights[1], 'switch_gamma': weights[2],
                'training_seed': job['training_seed'], 'train_seed': job['training_seed'],
                'training_steps': 150000, 'evaluation_seed': seed, 'scenario': job['scenario'],
                'split': job['split'], 'is_incumbent': job['candidate_id'] == INCUMBENT_ID,
                'demand_sha256': sha256(Path(info['route_file'])), 'model_path': str(checkpoint)})
            rows.append(row)
            write_csv(folder / 'evaluation_metrics.csv', rows)
        write_json(result_path, rows)
        return rows
    finally:
        env.close()


def run_final(output, config, workers):
    selection = read_json(output / 'selection_lock.json')
    if checkpoint_fingerprints(selection['model_results']) != selection['model_fingerprints']:
        raise ValueError('Locked models or validation evidence changed before final evaluation')
    selected = selection['selected_candidate_id']
    candidate_ids = {INCUMBENT_ID, selected}
    jobs = []
    for result in selection['model_results']:
        train = result['job']
        candidate = train['candidate_id']
        if candidate not in candidate_ids:
            continue
        for split, scenario, seeds in [('holdout', 'random', config['holdout_seeds'])] + [
                ('robustness', s, config['robustness_seeds']) for s in SCENARIOS]:
            jobs.append({'candidate_id': candidate, 'training_seed': train['training_seed'],
                'weights': train['weights'], 'source': result['folder'], 'scenario': scenario,
                'model_sha256': sha256(Path(result['folder']) / 'final.zip'),
                'seeds': seeds, 'split': split, 'locked_selection_sha256': sha256(output / 'selection_lock.json'),
                'folder': str(output / 'evaluations' / split / scenario /
                              f'seed_{train["training_seed"]}' / f'candidate_{candidate}')})
    rows = []
    failures = []
    write_status(output, 'FINAL_EVALUATION', total_jobs=len(jobs), completed_jobs=0)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_job, job): job for job in jobs}
        for done_count, future in enumerate(as_completed(futures), 1):
            try:
                rows += future.result()
            except Exception as exc:
                failures.append({'job': futures[future], 'error': repr(exc)})
            write_status(output, 'FINAL_EVALUATION', total_jobs=len(jobs), completed_jobs=done_count,
                         failures=failures)
    if failures:
        write_json(output / 'final_evaluation_failures.json', failures)
        raise RuntimeError('Final evaluation incomplete; resume without changing the locked candidate')
    holdout = [r for r in rows if r['split'] == 'holdout']
    robustness = [r for r in rows if r['split'] == 'robustness']
    analyze_and_save(output, holdout, 'final_holdout', 'holdout', config)
    analyze_and_save(output, robustness, 'robustness', 'robustness', config)
    failed_checks = []
    changes = {}
    for scenario in ('random', *SCENARIOS):
        group = [r for r in rows if r['scenario'] == scenario]
        base = [r for r in group if r['candidate_id'] == INCUMBENT_ID]
        candidate = [r for r in group if r['candidate_id'] == selected]
        b = {m: float(np.mean([r[m] for r in base])) for m in METRICS}
        c = {m: float(np.mean([r[m] for r in candidate])) for m in METRICS}
        changes[scenario] = {m: 100 * (c[m] / b[m] - 1) if b[m] else None for m in METRICS}
        if any(r['collisions'] or r['teleports'] for r in group):
            failed_checks.append(f'{scenario}: safety event')
        if c['throughput'] < .95 * b['throughput']:
            failed_checks.append(f'{scenario}: throughput retention below 95%')
        if c['max_waiting_time'] > b['max_waiting_time'] + 1e-9:
            failed_checks.append(f'{scenario}: mean maximum waiting increased')
        limit = 1.0 if scenario == 'random' else 1.05
        if c['avg_waiting_time'] > limit * b['avg_waiting_time'] + 1e-9:
            failed_checks.append(f'{scenario}: average waiting confirmation failed')
        if scenario == 'random':
            for seed in config['training_seeds']:
                bv = np.mean([r['avg_waiting_time'] for r in base if r['training_seed'] == seed])
                cv = np.mean([r['avg_waiting_time'] for r in candidate if r['training_seed'] == seed])
                if cv > bv + 1e-9:
                    failed_checks.append(f'random: waiting worsened for training seed {seed}')
    confirmed = selected != INCUMBENT_ID and not failed_checks and changes['random']['avg_waiting_time'] < 0
    fine_registry = read_json(output / 'fine_model_registry.json')
    multi_registry = read_json(output / 'multiseed_model_registry.json')
    recommendation = {
        'completed_at': now(), 'incumbent_id': INCUMBENT_ID, 'incumbent_weights': INCUMBENT,
        'selected_candidate_id': selected, 'selected_weights': selection['selected_weights'],
        'recommended_candidate_id': selected if confirmed else INCUMBENT_ID,
        'recommended_weights': selection['selected_weights'] if confirmed else INCUMBENT,
        'confirmed': confirmed, 'failed_checks': failed_checks,
        'reason': 'locked_candidate_confirmed' if confirmed else 'incumbent_retained_no_stable_confirmed_improvement',
        'changes_vs_incumbent_pct': changes, 'total_trained_models': len(fine_registry) + len(multi_registry),
        'fine_model_count': len(fine_registry), 'multiseed_model_count': len(multi_registry),
        'training_seeds': config['training_seeds'], 'validation_seeds': config['validation_seeds'],
        'holdout_seeds': config['holdout_seeds'], 'robustness_seeds': config['robustness_seeds'],
        'training_steps_by_stage': {'fine': 50000, 'multiseed': 150000},
        'global_optimum_established': False, 'holdout_reselection_performed': False,
        'default_configuration_changed': False, 'prior_results_preserved': True,
        'boundary': read_json(output / 'boundary_state.json'),
        'statistical_significance_established': False,
        'statistical_note': 'Read paired bootstrap intervals; only three training seeds, no global optimality claim.',
        'scope': '실험한 탐색 범위와 학습 조건 내에서의 비교; 전역 최적값을 주장하지 않음',
        'model_results': [r for r in multi_registry if r['job']['candidate_id'] == (selected if confirmed else INCUMBENT_ID)],
    }
    verify_immutable(config)
    write_json(output / 'final_recommendation.json', recommendation)
    write_status(output, 'COMPLETED', total_trained_models=recommendation['total_trained_models'],
                 confirmed=confirmed, recommended_candidate_id=recommendation['recommended_candidate_id'])
    from model.experiments.fine_search_report import generate_report
    generate_report(output)
    return recommendation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--stage', choices=('preflight', 'fine', 'multiseed', 'final', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    config = initialize(output, args.workers, args.resume)
    try:
        if args.stage != 'preflight':
            freeze_protocol(output)
        preflight(output, config)
        if args.stage in ('fine', 'all'):
            run_fine(output, config, args.workers)
        if args.stage in ('multiseed', 'all'):
            run_multiseed(output, config, args.workers)
        if args.stage in ('final', 'all'):
            run_final(output, config, args.workers)
        verify_immutable(config)
        return 0
    except BaseException as exc:
        write_status(output, 'FAILED', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    raise SystemExit(main())
