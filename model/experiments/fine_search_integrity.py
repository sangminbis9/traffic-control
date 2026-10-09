"""Repeat five prespecified jobs to investigate a parallel SUMO startup failure.

This audit never changes the frozen experiment, replaces an original model, or
feeds replicas into candidate selection. Prepare its immutable plan first:

python -m model.experiments.fine_search_integrity --root RESULT_DIR --plan-only
python -m model.experiments.fine_search_integrity --root RESULT_DIR --workers 5
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from model.controller.traffic_dqn import TrafficDQN
from model.experiments import fine_reward_search as search
from model.experiments.reward_sensitivity import sha256, write_json


SOURCE_JOBS = ((22, 11), (22, 13), (22, 14), (22, 15), (42, 11))
EXPECTED_COMPARISON_MODELS = 42
EVIDENCE_FILES = ('job.json', 'experiment_config.json', 'final.zip',
                  'training_routes.json', 'evaluation_routes.json', 'evaluation_metrics.csv')
CHECKPOINT_NAMES = tuple(f'checkpoint_{step}.zip' for step in range(25_000, 150_001, 25_000)) + ('final.zip',)
RATIONALE = (
    'Technical investigation of candidate13/training_seed42/attempt001 failing during '
    'SUMO startup with a malformed getVersion response and adjacent Address already in use. '
    'The five explicit jobs below were active peers at that event. They are repeated '
    'independently of observed candidate rankings. Replicas are integrity evidence only, '
    'not additional training-seed samples or substitute selection models.'
)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_protocol(root, config):
    search.verify_immutable(config)
    protocol = search.read_json(root / 'protocol_code_lock.json')
    for relative, expected in protocol['source_sha256'].items():
        if sha256(search.ROOT / relative) != expected:
            raise ValueError(f'Frozen experiment source changed: {relative}')
    if sha256(root / 'search_config.json') != protocol['rules_sha256']:
        raise ValueError('Frozen experiment rules changed')


def source_evidence(root, seed, candidate):
    parent = root / 'runs' / 'multiseed' / f'seed_{seed}' / f'candidate_{candidate}'
    config = search.read_json(root / 'shortlist_lock.json')
    for attempt in sorted(parent.glob('attempt_*'), reverse=True):
        folder = attempt / f'experiment_{candidate}'
        if not (folder / 'summary.json').is_file():
            continue
        job = search.read_json(attempt / 'job.json')
        metadata = search.read_json(folder / 'experiment_config.json')
        expected = {
            'stage': 'multiseed', 'candidate_id': candidate,
            'weights': config['candidate_weights'][str(candidate)],
            'training_seed': seed, 'training_steps': 150_000, 'parent': str(parent),
        }
        if job != expected or metadata['completed_training_steps'] != 150_000:
            raise ValueError(f'Original job protocol mismatch: seed={seed}, candidate={candidate}')
        if metadata['model_sha256'] != sha256(folder / 'final.zip'):
            raise ValueError('Original final checkpoint hash mismatch')
        paths = {'job.json': attempt / 'job.json'}
        paths.update({name: folder / name for name in EVIDENCE_FILES if name != 'job.json'})
        paths.update({name: folder / name for name in CHECKPOINT_NAMES})
        return {'candidate_id': candidate, 'training_seed': seed, 'job': job,
                'source_folder': str(folder),
                'source_sha256': {name: sha256(path) for name, path in paths.items()}}
    raise ValueError(f'No completed original job: seed={seed}, candidate={candidate}')


def prepare_plan(root, workers):
    config = search.read_json(root / 'search_config.json')
    verify_protocol(root, config)
    output = root / 'audit' / 'retraining_integrity'
    output.mkdir(parents=True, exist_ok=True)
    sources = [source_evidence(root, seed, candidate) for seed, candidate in SOURCE_JOBS]
    identity = {
        'audit_script_sha256': sha256(Path(__file__)),
        'search_config_sha256': sha256(root / 'search_config.json'),
        'frozen_protocol_sha256': sha256(root / 'protocol_code_lock.json'),
        'runtime': search.current_runtime(), 'sources': sources,
        'expected_comparison_models': EXPECTED_COMPARISON_MODELS,
        'expected_integrity_models': len(SOURCE_JOBS),
        'comparison_checkpoint_names': list(CHECKPOINT_NAMES),
        'selection_sample_size_increased': False,
    }
    payload = {
        'schema_version': 1, 'planned_at_utc': search.now(), 'rationale': RATIONALE,
        'planned_workers': workers, 'identity': identity,
        'retry_policy': 'Reuse only audited completed replicas; interrupted replicas restart fresh in a new attempt.',
        'mismatch_policy': 'Report FAIL; preserve both models and all evidence; never replace an original or reselect.',
        'training_progress_comparison': 'Compare status, timesteps and total_timesteps; wall time and steps/second are excluded.',
        'notes': [
            'This new audit script is outside the frozen selection protocol and does not modify it.',
            'Five jobs were specified from the startup incident, not selected by performance.',
            'Exact tensor equality is checked at all saved 25k checkpoints and final checkpoint.',
            'ZIP byte hashes are recorded but may differ due to serialization metadata; tensor/state equality is decisive.',
            'Matching saved states is strong reproducibility evidence, not a recording of every unsaved simulator tick.',
        ],
    }
    plan = search.lock_payload(output / 'audit_plan.json', payload, ('identity', 'rationale'))
    return output, plan


def compare_nested(reference, replica):
    """Exact recursive comparison, including every optimizer tensor and scalar."""
    result = {'all_equal': True, 'tensor_count': 0, 'tensor_elements': 0,
              'leaf_count': 0, 'mismatch_count': 0, 'mismatches': []}

    def mismatch(path, reason):
        result['all_equal'] = False
        result['mismatch_count'] += 1
        if len(result['mismatches']) < 30:
            result['mismatches'].append({'path': path, 'reason': reason})

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            result['tensor_count'] += 1
            result['tensor_elements'] += left.numel()
            if not isinstance(right, torch.Tensor):
                mismatch(path, 'tensor type differs')
            elif left.dtype != right.dtype or left.shape != right.shape:
                mismatch(path, 'tensor dtype or shape differs')
            elif not torch.equal(left.cpu(), right.cpu()):
                mismatch(path, 'tensor values differ')
        elif isinstance(left, np.ndarray):
            result['leaf_count'] += 1
            if not isinstance(right, np.ndarray) or left.dtype != right.dtype or not np.array_equal(left, right):
                mismatch(path, 'array values, dtype or shape differ')
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                mismatch(path, 'mapping keys differ')
                return
            for key in left:
                visit(left[key], right[key], f'{path}.{key}')
        elif isinstance(left, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                mismatch(path, 'sequence type or length differs')
                return
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, f'{path}[{index}]')
        else:
            result['leaf_count'] += 1
            if type(left) is not type(right) or left != right:
                mismatch(path, 'scalar value or type differs')

    visit(reference, replica, 'state')
    return result


def compare_checkpoint(source, replica):
    left = TrafficDQN.load(str(source), device='cpu')
    right = TrafficDQN.load(str(replica), device='cpu')
    state_names = ('num_timesteps', '_total_timesteps', '_n_updates', '_n_calls',
                   'exploration_rate', '_current_progress_remaining', '_hold_remaining',
                   '_held_action', 'gamma', 'learning_starts', 'n_steps', 'target_update_interval')
    components = {
        'q_net': compare_nested(left.q_net.state_dict(), right.q_net.state_dict()),
        'q_net_target': compare_nested(left.q_net_target.state_dict(), right.q_net_target.state_dict()),
        'optimizer': compare_nested(left.policy.optimizer.state_dict(), right.policy.optimizer.state_dict()),
        'training_state': compare_nested({key: getattr(left, key) for key in state_names},
                                         {key: getattr(right, key) for key in state_names}),
    }
    contract = (left.observation_space.shape == right.observation_space.shape == (60,)
                and left.action_space.n == right.action_space.n == 8)
    return {'name': source.name, 'all_equal': contract and all(c['all_equal'] for c in components.values()),
            'observation_action_contract': contract, 'source_sha256': sha256(source),
            'verification_sha256': sha256(replica), 'source_steps': left.num_timesteps,
            'verification_steps': right.num_timesteps, 'source_updates': left._n_updates,
            'verification_updates': right._n_updates, 'components': components}


def compare_run(source, result):
    original, replica = Path(source['source_folder']), Path(result['folder'])
    checkpoints = [compare_checkpoint(original / name, replica / name) for name in CHECKPOINT_NAMES]
    evidence_names = ('training_routes.json', 'evaluation_routes.json', 'evaluation_metrics.csv',
                      'training_progress.csv', 'final.zip')
    hashes = {side: {name: sha256(folder / name) for name in evidence_names}
              for side, folder in (('source', original), ('verification', replica))}
    source_rows, replica_rows = csv_rows(original / 'evaluation_metrics.csv'), csv_rows(replica / 'evaluation_metrics.csv')
    required_seeds = list(range(8001, 8021))
    evaluation_complete = ([int(r['seed']) for r in source_rows] == required_seeds
                           and [int(r['seed']) for r in replica_rows] == required_seeds)
    progress_columns = ('status', 'timesteps', 'total_timesteps')
    progress = lambda folder: [{key: row[key] for key in progress_columns}
                               for row in csv_rows(folder / 'training_progress.csv')]
    checks = {
        'training_demand_equal': search.read_json(original / 'training_routes.json') == search.read_json(replica / 'training_routes.json'),
        'evaluation_demand_equal': search.read_json(original / 'evaluation_routes.json') == search.read_json(replica / 'evaluation_routes.json'),
        'raw_evaluation_metrics_equal': hashes['source']['evaluation_metrics.csv'] == hashes['verification']['evaluation_metrics.csv'],
        'evaluation_rows_equal': source_rows == replica_rows,
        'evaluation_complete': evaluation_complete,
        'training_progress_steps_equal': progress(original) == progress(replica),
    }
    return {'candidate_id': source['candidate_id'], 'training_seed': source['training_seed'],
            'source_folder': str(original), 'verification_folder': str(replica),
            'all_equal': all(checks.values()) and all(c['all_equal'] for c in checkpoints),
            **checks, 'evaluation_episode_count': len(replica_rows), 'checkpoints': checkpoints,
            'file_sha256': hashes, 'reused_completed_replica': result['reused_completed_run']}


def failed_attempts(path):
    failures = []
    for progress in path.rglob('progress.json'):
        value = search.read_json(progress)
        if value.get('status') == 'FAILED':
            failures.append({'path': str(progress), 'error': value.get('error')})
    return failures


def run_audit(root, output, plan, workers):
    torch.set_num_threads(1)
    started = time.monotonic()
    sources = plan['identity']['sources']
    jobs = [{**source['job'], 'parent': str(output / 'runs' /
             f'seed_{source["training_seed"]}' / f'candidate_{source["candidate_id"]}')}
            for source in sources]
    source_by_key = {(s['training_seed'], s['candidate_id']): s for s in sources}
    completed, failures, comparisons = [], [], []
    history = output / 'execution_history'
    history.mkdir(exist_ok=True)
    run_number = len(list(history.glob('run_*.json'))) + 1

    def publish(status):
        main_failures = failed_attempts(root / 'runs' / 'multiseed')
        audit_failures = failed_attempts(output / 'runs')
        equal = len(comparisons) == len(SOURCE_JOBS) and all(c['all_equal'] for c in comparisons)
        data = {
            'schema_version': 1, 'status': status, 'updated_at_utc': search.now(),
            'expected_comparison_models': EXPECTED_COMPARISON_MODELS,
            'expected_integrity_models': len(SOURCE_JOBS), 'integrity_completed_models': len(completed),
            'successful_models_including_integrity': EXPECTED_COMPARISON_MODELS + len(completed),
            'main_failed_attempt_count': len(main_failures), 'integrity_failed_attempt_count': len(audit_failures),
            'failed_attempt_count': len(main_failures) + len(audit_failures),
            'failed_attempts': main_failures + audit_failures,
            'comparisons': comparisons, 'all_equal': equal, 'failures': failures,
            'rationale': RATIONALE, 'selection_sample_size_increased': False,
            'plan_sha256': sha256(output / 'audit_plan.json'), 'audit_script_sha256': sha256(Path(__file__)),
            'workers_this_execution': workers, 'elapsed_seconds': time.monotonic() - started,
            'notes': plan['notes'],
        }
        write_json(history / f'run_{run_number:03d}.json', data)
        write_json(output / 'audit_result.json', data)
        return data

    publish('RUNNING')
    try:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = {pool.submit(search.run_training_job, task): task for task in jobs}
            for future in as_completed(futures):
                task = futures[future]
                key = (task['training_seed'], task['candidate_id'])
                try:
                    result = future.result()
                    completed.append(result)
                    comparisons.append(compare_run(source_by_key[key], result))
                    comparisons.sort(key=lambda row: (row['training_seed'], row['candidate_id']))
                    print(f'Integrity comparison seed={key[0]} candidate={key[1]}: '
                          f'{"equal" if next(c for c in comparisons if (c["training_seed"], c["candidate_id"]) == key)["all_equal"] else "MISMATCH"}', flush=True)
                except Exception as exc:
                    failures.append({'job': task, 'error': repr(exc), 'traceback': traceback.format_exc()})
                publish('RUNNING')
        # Revalidate original hashes and the frozen protocol after all replicas finish.
        prepare_plan(root, workers)
        passed = (not failures and len(comparisons) == len(SOURCE_JOBS)
                  and all(c['all_equal'] for c in comparisons))
        return publish('PASS' if passed else 'FAIL')
    except BaseException as exc:
        failures.append({'error': repr(exc), 'traceback': traceback.format_exc()})
        publish('INCOMPLETE')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 5:
        parser.error('--workers must be between 1 and 5')
    root = args.root.resolve()
    output, plan = prepare_plan(root, args.workers)
    if args.plan_only:
        print(f'Integrity plan saved: {output / "audit_plan.json"}', flush=True)
        return 0
    result = run_audit(root, output, plan, args.workers)
    print(f'Integrity audit: {result["status"]}', flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
