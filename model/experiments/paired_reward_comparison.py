"""Rebuild the seven-row reward table on one audited 50k/30-episode protocol.

The historical table's raw artifacts are unavailable. Its numbers are not reused.
Four compatible fine-search checkpoints are reused; three missing weights train
fresh through the unchanged sensitivity worker. All seven use traffic 2001..2030.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch

from model.controller.traffic_dqn import TrafficDQN
from model.experiments.fine_reward_search import (
    ROOT, core_hashes, current_runtime, now, preflight,
)
from model.experiments.quick_reward_search import read_csv, read_json, summarize_episodes
from model.experiments.reward_sensitivity import AuditedEnv, METRICS, sha256, worker, write_csv, write_json
from model.utils.config import ProjectConfig, RewardConfig

CANDIDATES = {1: (.05, .05, .1), 2: (.1, .1, .2), 3: (.2, .2, .3),
              4: (.3, .3, .5), 5: (.3, .5, .4), 6: (.3, .5, .5), 7: (.3, .5, .6)}
REUSE = {4: 30, 5: 13, 6: 14, 7: 15}
SEEDS = list(range(2001, 2031))
STEPS = 50000


def normalized(value):
    return json.loads(json.dumps(value, default=str))


def validate_training(folder: Path, candidate: int):
    metadata = read_json(folder / 'experiment_config.json')
    base = ProjectConfig()
    expected_reward = asdict(replace(base.reward, waiting_weight=CANDIDATES[candidate][0],
                                    max_waiting_weight=CANDIDATES[candidate][1],
                                    switch_penalty=CANDIDATES[candidate][2]))
    expected = {'reward_config': expected_reward, 'signal_config': asdict(base.signal),
                'dqn_config': asdict(base.dqn), 'simulation_config': asdict(base.simulation),
                'fixed_green_times': base.fixed_green_times,
                'completed_training_steps': STEPS, 'learning_schedule_steps': 150000,
                'deterministic_algorithms': True, 'traci_subscriptions': True,
                'reproduced_from_scratch': True,
                'python_version': current_runtime()['python'],
                'network_sha256': sha256(base.network_file),
                'experiment_source_sha256': sha256(ROOT / 'model/experiments/reward_sensitivity.py'),
                'subscription_source_sha256': sha256(ROOT / 'model/experiments/subscription_connection.py')}
    for key, value in expected.items():
        if metadata.get(key) != normalized(value):
            raise ValueError(f'Incompatible training {candidate}: {key}')
    options = metadata['training_options']
    for key, value in {'total_steps': STEPS, 'seed': 22, 'scenario': 'random', 'episode_seconds': 300}.items():
        if options[key] != value:
            raise ValueError(f'Incompatible training option {candidate}: {key}')
    if sha256(folder / 'final.zip') != metadata['model_sha256']:
        raise ValueError(f'Checkpoint hash mismatch: {candidate}')
    for name in ('intersection.net.xml', 'simulation.sumocfg'):
        if sha256(folder / 'sumo' / name) != sha256(base.sumo_dir / name):
            raise ValueError(f'Training SUMO file differs: {candidate}/{name}')
    model = TrafficDQN.load(str(folder / 'final.zip'), device='cpu')
    if model.num_timesteps != STEPS or model.observation_space.shape != (60,) or model.action_space.n != 8:
        raise ValueError(f'Checkpoint contract mismatch: {candidate}')
    if model._total_timesteps != 150000 or model.seed != 22 or model.n_steps != base.dqn.n_steps:
        raise ValueError(f'Checkpoint training schedule/seed mismatch: {candidate}')
    for network in (model.q_net, model.q_net_target):
        if any(not torch.isfinite(tensor).all() for tensor in network.state_dict().values()):
            raise ValueError(f'Nonfinite checkpoint: {candidate}')
    return metadata, {'num_timesteps': model.num_timesteps, 'n_updates': model._n_updates,
                      'exploration_rate': model.exploration_rate,
                      'learning_schedule_steps': model._total_timesteps, 'training_seed': model.seed,
                      'n_steps': model.n_steps, 'observation_shape': [60], 'action_count': 8}


def prepare(output, source):
    prior = read_json(source / 'search_config.json')
    if prior['core_sha256'] != core_hashes() or prior['runtime'] != current_runtime():
        raise ValueError('Fine-search runtime or core changed; saved models cannot be mixed')
    registry = {r['job']['candidate_id']: r for r in read_json(source / 'fine_model_registry.json')}
    preserved, reuse = {}, {}
    for candidate, source_id in REUSE.items():
        entry = registry[source_id]
        if entry['job']['weights'] != list(CANDIDATES[candidate]):
            raise ValueError('Reuse registry weights mismatch')
        folder = Path(entry['folder'])
        metadata, state = validate_training(folder, candidate)
        reuse[str(candidate)] = {'source_candidate_id': source_id, 'source_folder': str(folder),
                                'model_sha256': metadata['model_sha256'], 'model_state': state}
        for name in ('final.zip', 'experiment_config.json', 'training_routes.json', 'summary.json',
                     'evaluation_metrics.csv', 'evaluation_routes.json',
                     'sumo/intersection.net.xml', 'sumo/simulation.sumocfg'):
            preserved[str(folder / name)] = sha256(folder / name)
    # Freeze definitions and selection before evaluating the comparison cohort.
    config = {'created_at': now(), 'candidates': CANDIDATES, 'queue_weight': 1.0,
              'training_steps': STEPS, 'schedule_timesteps': 150000, 'training_seed': 22,
              'evaluation_seeds': SEEDS, 'scenario': 'random', 'episode_seconds': 300,
              'demand_seconds': 240, 'deterministic_evaluation': True,
              'aggregation': 'evaluation episode mean and sample SD (ddof=1)',
              'historical_numeric_rows_reused': False, 'historical_raw_artifacts_recovered': False,
              'purpose': 'uniform replacement comparison; not a reproduction claim or new holdout',
              'reused_models': reuse, 'fresh_training_candidates': [1, 2, 3],
              'source_search': str(source), 'runtime': current_runtime(), 'core_sha256': core_hashes(),
              'sumo_config_sha256': sha256(ProjectConfig().sumo_config_file),
              'script_sha256': sha256(Path(__file__)), 'preserved_source_sha256': preserved,
              'source_search_config_sha256': sha256(source / 'search_config.json'),
              'source_preflight_sha256': sha256(source / 'preflight.json')}
    path = output / 'comparison_config.json'
    if path.exists():
        previous = read_json(path)
        config['created_at'] = previous['created_at']
        if previous != normalized(config):
            raise ValueError('Existing comparison protocol differs')
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be fresh')
    output.mkdir(parents=True, exist_ok=True)
    write_json(path, config)
    return read_json(path)


def train_missing(output: str, candidate: int):
    parent = Path(output) / 'training' / f'candidate_{candidate}'
    attempts = sorted(parent.glob('attempt_*'))
    for attempt in reversed(attempts):
        folder = attempt / f'experiment_{candidate}'
        if (folder / 'summary.json').exists():
            validate_training(folder, candidate)
            return str(folder)
    attempt = parent / f'attempt_{len(attempts) + 1:03d}'
    attempt.mkdir(parents=True, exist_ok=False)
    settings = {'output': str(attempt), 'timesteps': STEPS, 'schedule_timesteps': 150000,
                'seed': 22, 'scenario': 'random', 'episode_seconds': 300,
                'weights': CANDIDATES[candidate], 'eval_seed_start': 2001,
                'eval_episodes': 30, 'cache': True, 'verify_cache': False}
    write_json(attempt / 'job.json', settings)
    worker(candidate, settings)
    return str(attempt / f'experiment_{candidate}')


def evaluate_saved(output: Path, candidate: int, source: Path):
    metadata, _ = validate_training(source, candidate)
    folder = output / 'evaluations' / f'candidate_{candidate}'
    identity = {'candidate': candidate, 'source': str(source), 'weights': CANDIDATES[candidate],
                'model_sha256': metadata['model_sha256'], 'training_steps': STEPS,
                'seeds': SEEDS, 'scenario': 'random', 'episode_seconds': 300,
                'deterministic_actions': True, 'runtime': current_runtime()}
    if (folder / 'summary.json').exists():
        if read_json(folder / 'evaluation_config.json') != normalized(identity):
            raise ValueError('Evaluation resume protocol mismatch')
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / 'evaluation_config.json', identity)
    # Copies make this comparison self-contained without mutating the source run.
    for name in ('final.zip', 'experiment_config.json', 'training_routes.json'):
        shutil.copy2(source / name, folder / name)
    sumo_dir = folder / 'sumo'
    sumo_dir.mkdir(exist_ok=True)
    for name in ('intersection.net.xml', 'simulation.sumocfg'):
        shutil.copy2(source / 'sumo' / name, sumo_dir / name)
    cfg = replace(ProjectConfig(), sumo_dir=sumo_dir, results_dir=folder,
                  reward=RewardConfig(**metadata['reward_config']))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = TrafficDQN.load(str(folder / 'final.zip'), device='cpu')
    env = AuditedEnv(cfg, controller_name='Sensitivity-DQN', scenario='random',
                     episode_seconds=300, cache=True, manifest=folder / 'evaluation_routes.json')
    rows = []
    try:
        for episode, seed in enumerate(SEEDS):
            obs, _ = env.reset(seed=seed)
            while True:
                action = int(model.predict(obs, deterministic=True)[0])
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            row = env.episode_summary(episode)
            row.update({'experiment': candidate, 'alpha': CANDIDATES[candidate][0],
                        'beta': CANDIDATES[candidate][1], 'switch_gamma': CANDIDATES[candidate][2]})
            rows.append(row)
            write_csv(folder / 'evaluation_metrics.csv', rows)
            write_json(folder / 'progress.json', {'status': 'EVALUATING', 'completed': len(rows), 'total': 30})
    finally:
        env.close()
    write_json(folder / 'summary.json', summarize_episodes(rows))
    write_json(folder / 'progress.json', {'status': 'COMPLETED', 'completed': 30, 'total': 30})
    return folder


def validate_episodes(rows, manifest):
    if len(rows) != 30 or [int(r['seed']) for r in rows] != SEEDS:
        raise ValueError('Evaluation row count or seed order mismatch')
    if len(manifest) != 30 or [r['seed'] for r in manifest] != SEEDS:
        raise ValueError('Evaluation manifest mismatch')
    for row in rows:
        if float(row['duration']) != 300 or row['traffic_scenario'] != 'random':
            raise ValueError('Evaluation scenario or duration mismatch')
        if int(row['departed']) != int(row['throughput']) + int(row['unfinished']):
            raise ValueError('Vehicle accounting mismatch')
        if not all(np.isfinite(float(row[m])) for m in METRICS):
            raise ValueError('Nonfinite metric')


def aggregate(output: Path, config: dict, folders: dict[int, Path]):
    if set(folders) != set(CANDIDATES):
        raise ValueError('All seven candidates are required')
    summaries, all_rows, records = [], [], []
    reference_train = reference_eval = reference_state = None
    for candidate, folder in sorted(folders.items()):
        metadata, state = validate_training(folder, candidate)
        train = read_json(folder / 'training_routes.json')
        manifest = read_json(folder / 'evaluation_routes.json')
        rows = read_csv(folder / 'evaluation_metrics.csv')
        validate_episodes(rows, manifest)
        if reference_train is not None and (train != reference_train or manifest != reference_eval or state != reference_state):
            raise ValueError('Training/evaluation demand or training update schedule is not paired')
        reference_train, reference_eval, reference_state = train, manifest, state
        summary = {'experiment': candidate, 'alpha': CANDIDATES[candidate][0],
                   'beta': CANDIDATES[candidate][1], 'switch_gamma': CANDIDATES[candidate][2],
                   'training_steps': STEPS, 'training_seed': 22, **summarize_episodes(rows)}
        summaries.append(summary)
        for row, demand in zip(rows, manifest, strict=True):
            all_rows.append({**row, 'experiment': candidate, 'training_steps': STEPS,
                             'training_seed': 22, 'demand_sha256': demand['route_sha256'],
                             'model_sha256': metadata['model_sha256']})
        records.append({'experiment': candidate, 'folder': str(folder), 'model_state': state,
                        'model_sha256': metadata['model_sha256'],
                        'training_routes_sha256': sha256(folder / 'training_routes.json'),
                        'evaluation_routes_sha256': sha256(folder / 'evaluation_routes.json'),
                        'evaluation_metrics_sha256': sha256(folder / 'evaluation_metrics.csv')})
    if core_hashes() != config['core_sha256'] or current_runtime() != config['runtime']:
        raise ValueError('Runtime/core changed during comparison')
    if sha256(ProjectConfig().sumo_config_file) != config['sumo_config_sha256']:
        raise ValueError('SUMO configuration changed during comparison')
    for path, digest in config['preserved_source_sha256'].items():
        if sha256(Path(path)) != digest:
            raise ValueError('Preserved source changed')
    if sha256(Path(__file__)) != config['script_sha256']:
        raise ValueError('Comparison implementation changed during execution')
    write_csv(output / 'comparison.csv', summaries)
    write_csv(output / 'evaluation_metrics.csv', all_rows)
    audit = {'verified_at': now(), 'status': 'PASS', 'evaluation_rows': len(all_rows),
             'paired_training_routes': True, 'paired_evaluation_routes': True,
             'matched_training_state': reference_state, 'source_models_preserved': True,
             'runtime_core_unchanged': True, 'records': records}
    write_json(output / 'comparison.json', {'config': config, 'results': summaries})
    write_json(output / 'verification.json', audit)
    headers = ['실험', 'α', 'β', 'γ', '평균 대기시간 ↓', '최대 대기시간 ↓',
               '평균 대기열 ↓', '신호 전환 횟수 ↓', '통과 차량 수 ↑']
    table = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * 9) + ' |']
    for row in summaries:
        label = str(row['experiment']) + (' (초기값)' if row['experiment'] == 2 else
                                         ' (추가)' if row['experiment'] >= 5 else '')
        fields = [label, str(row['alpha']), str(row['beta']), str(row['switch_gamma'])]
        fields += [f"{row[m]:.2f} ± {row[m + '_std']:.2f}" for m in METRICS]
        table.append('| ' + ' | '.join(fields) + ' |')
    text = '\n'.join([
        '# 동일 조건의 7개 보상 가중치 비교', '',
        '이 표의 1–4행은 이번에 동일 조건으로 산출한 값입니다. 과거 표의 숫자를 복사하지 않았습니다.',
        '과거 원본 설정/결과를 확보하지 못했으므로 과거 결과의 재현 성공을 주장하지 않습니다.', '',
        'Queue=1, 60차원/8행동, 학습 seed=22, 50,000 step, 원래 150,000-step 탐색률 일정.',
        '평가: random, seed 2001–2030의 동일 교통 30회, 회차 300초/수요 240초, deterministic 행동.',
        '값은 평가 30회 평균 ± 표본 표준편차(ddof=1). 학습 seed 간 편차나 95% CI가 아닙니다.', '',
        *table, '',
        '1–3은 새 독립 학습, 4–7은 동일 코드·runtime·학습 조건이 확인된 기존 50k 최종 모델을 재사용했습니다.',
        '모든 조합의 학습 수요와 평가 수요 SHA-256, 학습 step/update 수, 모델 해시를 검증했습니다.',
        '5–7 조합은 이번 평가를 열기 전에 고정했습니다. 과거 평가에 사용된 교통이므로 새로운 holdout으로 보지 않습니다.',
        '한 학습 seed의 기술 통계이며, 통계적 우월성·전역 최적성·150k 결과의 대체를 주장하지 않습니다.',
        '평균 대기는 진입한 차량의 관측 누적 대기(미완료 포함, pending 제외), 최대 대기는 회차별 최대의 평균입니다.',
        '평균 대기열은 8개 이동 그룹 정지 차량 합의 시간 평균이며, 전환 횟수는 강제 전환을 포함합니다.', '',
        '원자료: evaluation_metrics.csv. 설정/모델 출처: comparison_config.json. 검증: verification.json.',
    ])
    (output / 'comparison-ko.txt').write_text(text + '\n', encoding='utf-8')
    write_json(output / 'status.json', {'status': 'COMPLETED', 'updated_at': now(), 'evaluation_rows': 210})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('model/results/reward_fine_search_2026-10-09'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 3:
        parser.error('--workers must be between 1 and 3')
    output, source = args.output.resolve(), args.source.resolve()
    config = prepare(output, source)
    try:
        write_json(output / 'status.json', {'status': 'PREFLIGHT', 'updated_at': now()})
        preflight(output, config)
        folders, failures = {}, []
        write_json(output / 'status.json', {'status': 'TRAINING', 'updated_at': now(), 'candidates': [1, 2, 3]})
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(train_missing, str(output), i): i for i in (1, 2, 3)}
            for future in as_completed(jobs):
                candidate = jobs[future]
                try:
                    folders[candidate] = Path(future.result())
                    print(f'Training and 30 evaluations completed: {candidate}', flush=True)
                except Exception as exc:
                    failures.append({'candidate': candidate, 'error': repr(exc)})
                write_json(output / 'training_jobs.json', {'completed': folders, 'failures': failures})
        if failures:
            raise RuntimeError(f'Training failures: {failures}; resume uses fresh attempts')
        for candidate in REUSE:
            write_json(output / 'status.json', {'status': 'EVALUATING', 'candidate': candidate,
                                                'completed_candidates': sorted(folders), 'updated_at': now()})
            folders[candidate] = evaluate_saved(output, candidate, Path(config['reused_models'][str(candidate)]['source_folder']))
            print(f'Saved model evaluated on 30 seeds: {candidate}', flush=True)
        aggregate(output, config, folders)
        print('Completed and verified all 210 evaluation episodes.', flush=True)
    except Exception as exc:
        write_json(output / 'status.json', {'status': 'FAILED', 'error': repr(exc), 'updated_at': now()})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
