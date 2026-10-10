"""Tune an observation-only rule policy, then compare it with frozen DQNs.

python -m model.experiments.rule_based_comparison --output model/results/rule_comparison
No models are trained or changed. The tuning and test cohorts, rule grid, code,
runtime, routes and three existing checkpoints are frozen before evaluation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import torch

from model.controller.rule_based_controller import RuleBasedController, RulePolicyConfig, candidate_policies
from model.controller.traffic_dqn import TrafficDQN
from model.experiments.reward_sensitivity import AuditedEnv, sha256, write_csv, write_json
from model.traffic.route_generator import SCENARIOS, TrafficDemand, generate_route_file
from model.utils.config import ProjectConfig, resolve_sumo_binary

ROOT = Path(__file__).resolve().parents[2]
MODEL_SOURCE = ROOT / 'model/results/reward_fine_search_2026-10-09/runs/multiseed'
TRAINING_SEEDS = (22, 42, 62)
METRICS = ('avg_waiting_time', 'max_waiting_time', 'avg_queue', 'max_queue',
           'episode_reward', 'throughput', 'phase_changes', 'forced_changes',
           'demanded', 'departed', 'unfinished', 'pending', 'max_pending',
           'collisions', 'teleports', 'duration')
CODE_FILES = ('model/experiments/rule_based_comparison.py',
              'model/controller/rule_based_controller.py',
              'model/controller/signal_controller.py', 'model/controller/traffic_dqn.py',
              'model/env/intersection_env.py', 'model/env/state_provider.py',
              'model/env/reward.py', 'model/traffic/route_generator.py',
              'model/utils/config.py', 'model/utils/metrics.py',
              'model/experiments/reward_sensitivity.py',
              'model/experiments/subscription_connection.py')


def normalized(value):
    return json.loads(json.dumps(value, default=str))


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def case_key(row):
    return str(row.get('scenario', row.get('traffic_scenario'))), int(row['seed'])


def cohort(stage: str) -> list[dict]:
    """Predeclared traffic mix: random plus each of the six explicit scenarios."""
    if stage == 'tune':
        random_seeds, explicit_seeds = range(31001, 31009), range(31101, 31103)
    elif stage == 'evaluate':
        random_seeds, explicit_seeds = range(41001, 41031), range(41101, 41106)
    else:
        raise ValueError('Unknown cohort')
    return ([{'scenario': 'random', 'seed': seed} for seed in random_seeds]
            + [{'scenario': scenario, 'seed': seed} for scenario in SCENARIOS
               if scenario != 'random' for seed in explicit_seeds])


def validate_disjoint(tuning, evaluation, training_seeds=()):
    for cases in (tuning, evaluation):
        if len({case_key(case) for case in cases}) != len(cases):
            raise ValueError('Duplicate traffic case')
    tuning_seeds = {int(c['seed']) for c in tuning}
    evaluation_seeds = {int(c['seed']) for c in evaluation}
    if tuning_seeds & evaluation_seeds or (tuning_seeds | evaluation_seeds) & set(training_seeds):
        raise ValueError('Tuning, evaluation and training seeds must be disjoint')


class SafeFixedPolicy:
    """Demand-blind cyclic requests through the same safety controller as DQN.

    Unlike the legacy FixedTimeController, this class never passes force=True.
    Safety overrides can shorten or redirect the nominal schedule for all agents.
    """
    def __init__(self, config: ProjectConfig | None = None):
        self.config = config or ProjectConfig()

    def action(self, observation):
        phase = int(np.argmax(observation[24:32]))
        if observation[33] < .5:
            return int(np.argmax(observation[36:44]))
        elapsed = float(observation[32]) * self.config.signal.max_green
        if elapsed + 1e-6 >= self.config.fixed_green_times[phase]:
            return (phase + 1) % 8
        return phase


def validate_checkpoint(folder: Path, training_seed: int) -> dict:
    config = ProjectConfig()
    metadata = read_json(folder / 'experiment_config.json')
    expected = {'reward_config': asdict(config.reward), 'signal_config': asdict(config.signal),
                'simulation_config': asdict(config.simulation), 'dqn_config': asdict(config.dqn),
                'network_sha256': sha256(config.network_file),
                'completed_training_steps': 150000}
    for key, value in expected.items():
        if metadata.get(key) != normalized(value):
            raise ValueError(f'Checkpoint {training_seed}: incompatible {key}')
    options = metadata['training_options']
    if options['seed'] != training_seed or options['scenario'] != 'random' or options['episode_seconds'] != 300:
        raise ValueError('Checkpoint training protocol differs')
    digest = sha256(folder / 'final.zip')
    if digest != metadata['model_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    model = TrafficDQN.load(str(folder / 'final.zip'), device='cpu')
    if (model.num_timesteps != 150000 or model.seed != training_seed
            or model.observation_space.shape != (60,) or model.action_space.n != 8):
        raise ValueError('Checkpoint must be the 150k/60D/8-action trained model')
    if any(not torch.isfinite(value).all() for network in (model.q_net, model.q_net_target)
           for value in network.state_dict().values()):
        raise ValueError('Nonfinite checkpoint weights')
    routes = read_json(folder / 'training_routes.json')
    return {'path': str(folder / 'final.zip'), 'sha256': digest,
            'metadata_path': str(folder / 'experiment_config.json'),
            'metadata_sha256': sha256(folder / 'experiment_config.json'),
            'training_routes_path': str(folder / 'training_routes.json'),
            'training_routes_sha256': sha256(folder / 'training_routes.json'),
            'training_traffic_seeds': sorted({int(row['seed']) for row in routes}),
            'training_seed': training_seed, 'training_steps': 150000,
            'observation_shape': [60], 'action_count': 8}


def runtime():
    binary = Path(resolve_sumo_binary())
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {name: version(name) for name in
                         ('numpy', 'torch', 'stable-baselines3', 'gymnasium', 'traci')},
            'sumo_binary': str(binary), 'sumo_sha256': sha256(binary)}


def prepare(output: Path) -> dict:
    config = ProjectConfig()
    torch.set_num_threads(1)
    models = {str(seed): validate_checkpoint(
        MODEL_SOURCE / f'seed_{seed}/candidate_14/attempt_001/experiment_14', seed)
        for seed in TRAINING_SEEDS}
    tuning, evaluation = cohort('tune'), cohort('evaluate')
    training_seeds = {seed for model in models.values() for seed in model['training_traffic_seeds']}
    validate_disjoint(tuning, evaluation, training_seeds)
    protocol = {'created_at': now(), 'schema_version': 1,
                'project_config': asdict(config), 'models': models,
                'rule_candidates': {key: asdict(value) for key, value in candidate_policies().items()},
                'tuning_cases': tuning, 'evaluation_cases': evaluation,
                'episode_seconds': 300, 'demand_seconds': 240,
                'simulation_budget': {'tuning_policies': len(candidate_policies()),
                                      'tuning_episodes': len(candidate_policies()) * len(tuning),
                                      'tuning_decisions': len(candidate_policies()) * len(tuning) * 300,
                                      'heldout_policies': 5, 'heldout_episodes': 5 * len(evaluation),
                                      'heldout_decisions': 5 * len(evaluation) * 300},
                'selection': 'highest mean episode_reward, then lowest mean avg_waiting_time, then candidate id',
                'tuning_aggregation': 'equal weight per tuning case; random 8/20; explicit scenarios 2/20 each',
                'test_aggregation': 'equal weight per declared traffic case; random 30/60; explicit scenarios 5/60 each',
                'bootstrap': {'samples': 5000, 'seed': 271828, 'confidence': .95,
                              'unit': 'paired scenario/traffic seed, stratified by scenario',
                              'ci_scope': 'traffic variation conditional on the three frozen checkpoints; not retraining uncertainty',
                              'dqn': 'average three independent model runs per traffic case before resampling; not an executable ensemble'},
                'same_safety_path': True, 'deterministic_dqn': True,
                'runtime': runtime(),
                'code_sha256': {name: sha256(ROOT / name) for name in CODE_FILES},
                'network_sha256': sha256(config.network_file),
                'sumo_config_sha256': sha256(config.sumo_config_file)}
    path = output / 'protocol.json'
    if path.exists():
        previous = read_json(path)
        protocol['created_at'] = previous['created_at']
        # Routes are generated only for a fresh output, then verified on every resume.
        protocol['routes'] = previous['routes']
        if previous != normalized(protocol):
            raise ValueError('Existing output protocol/code/model/config differs; use a new output')
        verify_frozen(output, previous)
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be empty or have a matching frozen protocol')
    output.mkdir(parents=True, exist_ok=True)
    routes = {}
    for stage, cases in (('tune', tuning), ('evaluate', evaluation)):
        routes[stage] = []
        for case in cases:
            relative = f"routes/{stage}/{case['scenario']}_{case['seed']}.rou.xml"
            route = generate_route_file(output / relative, TrafficDemand(
                duration=300, demand_seconds=240, **case))
            routes[stage].append({**case, 'path': relative, 'route_sha256': sha256(route),
                                  'demanded': len(ET.parse(route).getroot().findall('vehicle'))})
    protocol['routes'] = routes
    write_json(path, protocol)
    return read_json(path)


def verify_frozen(output: Path, protocol: dict):
    for name, digest in protocol['code_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    config = ProjectConfig()
    if (sha256(config.network_file) != protocol['network_sha256']
            or sha256(config.sumo_config_file) != protocol['sumo_config_sha256']
            or runtime() != protocol['runtime']):
        raise ValueError('Frozen runtime/network changed')
    for model in protocol['models'].values():
        for path_key, hash_key in (('path', 'sha256'), ('metadata_path', 'metadata_sha256'),
                                   ('training_routes_path', 'training_routes_sha256')):
            if sha256(Path(model[path_key])) != model[hash_key]:
                raise ValueError('Frozen checkpoint evidence changed')
    for cases in protocol['routes'].values():
        for route in cases:
            if sha256(output / route['path']) != route['route_sha256']:
                raise ValueError('Frozen traffic route changed')


def validate_rows(rows, cases, *, complete=True):
    if len(rows) > len(cases) or (complete and len(rows) != len(cases)):
        raise ValueError('Incomplete or excess evaluation cases')
    for row, case in zip(rows, cases):
        if case_key(row) != case_key(case) or row['route_sha256'] != case['route_sha256']:
            raise ValueError('Unpaired, duplicate or reordered traffic cases')
        if any(not np.isfinite(float(row[metric])) for metric in METRICS):
            raise ValueError('Nonfinite evaluation metric')
        if float(row['duration']) != 300 or float(row['demanded']) != case['demanded']:
            raise ValueError('Incomplete horizon or changed demand')
        if int(row['departed']) != int(row['throughput']) + int(row['unfinished']):
            raise ValueError('Departed vehicle accounting mismatch')
        if int(row['demanded']) != int(row['departed']) + int(row['pending']):
            raise ValueError('Demand/pending accounting mismatch')
        if float(row['collisions']) or float(row['teleports']):
            raise ValueError('Unsafe episode cannot enter comparison')


def run_job(output_string: str, stage: str, policy_id: str, kind: str, settings: dict):
    started, started_at = time.monotonic(), now()
    output = Path(output_string)
    protocol = read_json(output / 'protocol.json')
    folder = output / stage / policy_id
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'stage': stage, 'policy_id': policy_id, 'kind': kind, 'settings': settings,
                'protocol_sha256': sha256(output / 'protocol.json')}
    identity_path = folder / 'job.json'
    if identity_path.exists() and read_json(identity_path) != normalized(identity):
        raise ValueError('Existing job identity differs')
    write_json(identity_path, identity)
    cases = protocol['routes'][stage]
    rows_path = folder / 'episodes.json'
    rows = read_json(rows_path) if rows_path.exists() else []
    validate_rows(rows, cases, complete=False)
    if len(rows) == len(cases):
        return rows
    prior_completed = len(rows)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    base = ProjectConfig()
    sumo_dir = folder / 'sumo'
    sumo_dir.mkdir(exist_ok=True)
    for name in ('intersection.net.xml', 'simulation.sumocfg'):
        shutil.copy2(base.sumo_dir / name, sumo_dir / name)
    config = replace(base, sumo_dir=sumo_dir, results_dir=folder)
    env = None
    try:
        if kind == 'dqn':
            model = TrafficDQN.load(settings['path'], device='cpu')
            action = lambda obs: int(model.predict(obs, deterministic=True)[0])
        else:
            policy = (RuleBasedController(config, RulePolicyConfig(**settings))
                      if kind == 'rule' else SafeFixedPolicy(config))
            action = policy.action
        for index in range(len(rows), len(cases)):
            case = cases[index]
            route_path = output / case['path']
            if sha256(route_path) != case['route_sha256']:
                raise ValueError('Route hash changed before episode')
            # This label must never start with "fixed": that activates a legacy force=True path.
            env = AuditedEnv(config, controller_name='Comparison-' + policy_id,
                             scenario=case['scenario'], episode_seconds=300, route_file=route_path,
                             route_dir=folder / 'generated', cache=True,
                             manifest=folder / f"route_{index:03d}.json")
            obs, _ = env.reset(seed=case['seed'])
            if env._fixed_controller is not None:
                raise RuntimeError('External policy bypassed shared safety path')
            while True:
                if obs.shape != (60,) or not np.isfinite(obs).all():
                    raise ValueError('Observation contract changed')
                obs, _, terminated, truncated, _ = env.step(action(obs))
                if terminated or truncated:
                    break
            row = {**env.episode_summary(index), 'policy_id': policy_id,
                   'route_sha256': case['route_sha256'], 'demanded': case['demanded']}
            env.close()
            env = None
            validate_rows([row], [case])
            rows.append(row)
            write_json(rows_path, rows)
            write_csv(folder / 'episodes.csv', rows)
            write_json(folder / 'progress.json', {'status': 'COMPLETED' if len(rows) == len(cases) else 'RUNNING',
                                                 'completed': len(rows), 'total': len(cases), 'updated_at': now()})
        return rows
    except BaseException:
        # Retain failures even if the same frozen job is resumed successfully later.
        failures_path = folder / 'failures.json'
        failures = read_json(failures_path) if failures_path.exists() else []
        failures.append({'at': now(), 'completed_cases': len(rows),
                         'next_case': cases[len(rows)] if len(rows) < len(cases) else None,
                         'traceback': traceback.format_exc()})
        write_json(failures_path, failures)
        raise
    finally:
        if env is not None:
            env.close()
        segments_path = folder / 'runtime_segments.json'
        segments = read_json(segments_path) if segments_path.exists() else []
        segments.append({'started_at': started_at, 'ended_at': now(),
                         'wall_seconds': time.monotonic() - started,
                         'previous_completed_cases': prior_completed,
                         'new_completed_cases': len(rows) - prior_completed,
                         'scope': 'this worker invocation including model load/SUMO startups; concurrent workers overlap'})
        write_json(segments_path, segments)


def execute_jobs(output, stage, jobs, workers):
    results, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_job, str(output), stage, name, kind, settings): name
                   for name, (kind, settings) in jobs.items()}
        for future in as_completed(pending):
            name = pending[future]
            try:
                results[name] = future.result()
                print(f'{stage}: {name} completed {len(results[name])} cases', flush=True)
            except Exception:
                failures.append({'policy_id': name, 'traceback': traceback.format_exc()})
            write_json(output / f'{stage}_progress.json',
                       {'completed_policies': sorted(results), 'failures': failures,
                        'total_policies': len(jobs), 'updated_at': now()})
    if failures:
        raise RuntimeError(f'{len(failures)} failed policies; inspect {stage}_progress.json; no partial ranking published')
    return results


def summaries(rows):
    result = {'episodes': len(rows)}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows])
        result[metric] = float(values.mean())
        result[metric + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return result


def select_candidate(results: dict, cases: list, candidates: dict) -> tuple[str, list]:
    if set(results) != set(candidates):
        raise ValueError('Every frozen candidate is required for selection')
    ranking = []
    for name, rows in results.items():
        validate_rows(rows, cases)
        ranking.append({'candidate': name, **summaries(rows)})
    ranking.sort(key=lambda row: (-row['episode_reward'], row['avg_waiting_time'], row['candidate']))
    return ranking[0]['candidate'], ranking


def tune(output, protocol, workers):
    results = execute_jobs(output, 'tune',
                           {name: ('rule', settings) for name, settings in protocol['rule_candidates'].items()}, workers)
    verify_frozen(output, protocol)
    winner, ranking = select_candidate(results, protocol['routes']['tune'], protocol['rule_candidates'])
    selection = {'winner': winner, 'policy': protocol['rule_candidates'][winner],
                 'protocol_sha256': sha256(output / 'protocol.json'),
                 'tuning_result_sha256': {name: sha256(output / 'tune' / name / 'episodes.json')
                                          for name in sorted(results)}, 'ranking': ranking}
    path = output / 'selection.json'
    if path.exists() and read_json(path) != normalized(selection):
        raise ValueError('Frozen selection changed; heldout cannot be used for reselection')
    write_json(output / 'tuning_results.json', results)
    write_csv(output / 'tuning_ranking.csv', ranking)
    write_json(path, selection)
    return selection


def load_selection(output, protocol):
    selection = read_json(output / 'selection.json')
    if selection['protocol_sha256'] != sha256(output / 'protocol.json'):
        raise ValueError('Selection protocol mismatch')
    results = {}
    for name, digest in selection['tuning_result_sha256'].items():
        path = output / 'tune' / name / 'episodes.json'
        if sha256(path) != digest:
            raise ValueError('Tuning results changed after selection')
        results[name] = read_json(path)
    winner, ranking = select_candidate(results, protocol['routes']['tune'], protocol['rule_candidates'])
    if (winner != selection['winner'] or ranking != selection['ranking']
            or selection['policy'] != protocol['rule_candidates'][winner]):
        raise ValueError('Selection does not match frozen tuning evidence')
    return selection


def average_models(results, model_names, cases):
    """One statistical unit per traffic case, not three duplicated traffic seeds."""
    if not model_names or len(set(model_names)) != len(model_names):
        raise ValueError('Distinct frozen DQN models required')
    for name in model_names:
        validate_rows(results[name], cases)
    return [{**case, 'traffic_scenario': case['scenario'],
             **{metric: float(np.mean([results[name][i][metric] for name in model_names]))
                for metric in METRICS}, 'policy_id': 'dqn_mean'}
            for i, case in enumerate(cases)]


def paired_differences(first, second, *, samples=5000, seed=271828):
    """First minus second, with paired traffic-case bootstrap within scenarios."""
    if not first or [case_key(row) for row in first] != [case_key(row) for row in second]:
        raise ValueError('Pairwise comparison requires identical ordered traffic cases')
    if any(a['route_sha256'] != b['route_sha256'] for a, b in zip(first, second)):
        raise ValueError('Pairwise route mismatch')
    rng = np.random.default_rng(seed)
    scenarios = sorted({case_key(row)[0] for row in first})
    strata = [np.array([i for i, row in enumerate(first) if case_key(row)[0] == scenario])
              for scenario in scenarios]
    indices = np.concatenate([rng.choice(group, size=(samples, len(group)), replace=True)
                              for group in strata], axis=1)
    result = []
    for metric in METRICS:
        differences = np.asarray([float(a[metric]) - float(b[metric]) for a, b in zip(first, second)])
        low, high = np.quantile(differences[indices].mean(axis=1), [.025, .975])
        result.append({'metric': metric, 'paired_cases': len(first), 'mean_difference': float(differences.mean()),
                       'ci95_low': float(low), 'ci95_high': float(high)})
    return result


def aggregate(output, protocol, selection, results):
    expected = {'rule', 'cycle_safe', *(f'dqn_seed_{seed}' for seed in TRAINING_SEEDS)}
    if set(results) != expected:
        raise ValueError('All heldout policies are required; partial comparison rejected')
    cases = protocol['routes']['evaluate']
    for rows in results.values():
        validate_rows(rows, cases)
    model_names = [f'dqn_seed_{seed}' for seed in TRAINING_SEEDS]
    dqn_mean = average_models(results, model_names, cases)
    policies = {**results, 'dqn_mean': dqn_mean}
    summary_rows, comparisons = [], []
    for scenario in ['all', *SCENARIOS]:
        subsets = {name: [row for row in rows if scenario == 'all' or case_key(row)[0] == scenario]
                   for name, rows in policies.items()}
        for name, rows in subsets.items():
            summary_rows.append({'policy_id': name, 'scenario': scenario, **summaries(rows)})
        for first, second in (('rule', 'dqn_mean'), ('rule', 'cycle_safe'), ('dqn_mean', 'cycle_safe')):
            comparisons.extend({'first': first, 'second': second, 'scenario': scenario, **row}
                               for row in paired_differences(subsets[first], subsets[second]))
    # Variation across trained models is separate from traffic-seed uncertainty.
    variation = []
    for metric in METRICS:
        model_means = [summaries(results[name])[metric] for name in model_names]
        variation.append({'metric': metric, 'models': len(model_names),
                          'mean': float(np.mean(model_means)),
                          'training_seed_std': float(np.std(model_means, ddof=1)),
                          'training_seed_min': float(min(model_means)),
                          'training_seed_max': float(max(model_means))})
    verify_frozen(output, protocol)
    if load_selection(output, protocol) != selection:
        raise ValueError('Selected policy changed during heldout evaluation')
    write_csv(output / 'heldout_episodes.csv', [row for rows in results.values() for row in rows])
    write_csv(output / 'summary.csv', summary_rows)
    write_csv(output / 'paired_differences.csv', comparisons)
    write_csv(output / 'dqn_training_seed_variation.csv', variation)
    write_json(output / 'comparison.json', {'selection': selection, 'summary': summary_rows,
                                          'paired_differences': comparisons, 'dqn_training_seed_variation': variation})
    write_json(output / 'verification.json', {'status': 'PASS', 'verified_at': now(),
        'observed_heldout_episodes': sum(len(rows) for rows in results.values()),
        'distinct_heldout_traffic_cases': len(cases), 'matched_routes': True,
        'frozen_code_models_runtime': True, 'selection_precedes_heldout': True,
        'complete_horizons': True, 'same_safety_path': True,
        'protocol_sha256': sha256(output / 'protocol.json'),
        'selection_sha256': sha256(output / 'selection.json')})


def evaluate(output, protocol, workers):
    selection = load_selection(output, protocol)
    # Persist exactly which selection opened heldout, and refuse later replacement.
    marker = {'selection_sha256': sha256(output / 'selection.json'),
              'protocol_sha256': sha256(output / 'protocol.json')}
    path = output / 'heldout_opened.json'
    if path.exists() and read_json(path) != marker:
        raise ValueError('Heldout already opened under a different selection')
    write_json(path, marker)
    jobs = {'rule': ('rule', selection['policy']), 'cycle_safe': ('cycle', {})}
    jobs.update({f'dqn_seed_{seed}': ('dqn', protocol['models'][str(seed)]) for seed in TRAINING_SEEDS})
    results = execute_jobs(output, 'evaluate', jobs, workers)
    aggregate(output, protocol, selection, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stage', choices=('tune', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.resolve()
    protocol = prepare(output)
    started, started_at = time.monotonic(), now()
    try:
        if args.stage in ('tune', 'all'):
            tune(output, protocol, args.workers)
        if args.stage in ('evaluate', 'all'):
            evaluate(output, protocol, args.workers)
        write_json(output / 'status.json', {'status': 'TUNED' if args.stage == 'tune' else 'COMPLETED',
                                           'updated_at': now()})
    except BaseException:
        write_json(output / 'status.json', {'status': 'FAILED', 'updated_at': now(),
                                           'traceback': traceback.format_exc()})
        raise
    finally:
        path = output / 'runtime_segments.json'
        segments = read_json(path) if path.exists() else []
        segments.append({'started_at': started_at, 'ended_at': now(), 'stage': args.stage,
                         'wall_seconds': time.monotonic() - started, 'workers': args.workers,
                         'scope': 'this invocation after protocol preparation; resumed invocations are separate segments'})
        write_json(path, segments)


if __name__ == '__main__':
    main()
