"""Six-candidate local search, reusing the verified 50k baseline only as a result.

Run: python -m model.experiments.local_reward_search --workers 5
Each new candidate starts with an independent model and empty replay buffer.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from model.experiments.reward_sensitivity import (
    METRICS, aggregate, sha256, worker, write_csv, write_json,
)
from model.utils.config import ProjectConfig
from model.utils.reproducibility import collect_reproducibility_metadata


CANDIDATES = {1: (0.25, 0.25, 0.40), 2: (0.30, 0.30, 0.50),
              3: (0.35, 0.35, 0.60), 4: (0.40, 0.40, 0.70),
              5: (0.30, 0.30, 0.60), 6: (0.35, 0.35, 0.50)}
BASELINE = 2
THROUGHPUT_RETENTION = 0.95
TIE_ORDER = (*METRICS[:4], '-throughput', 'candidate_id')


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_baseline(source: Path) -> dict:
    """Refuse results from a different model, schedule, demand or timing."""
    metadata = read_json(source / 'experiment_config.json')
    base = ProjectConfig()
    summary = read_json(source / 'summary.json')
    expected_reward = {**asdict(base.reward), 'waiting_weight': 0.30,
                       'max_waiting_weight': 0.30, 'switch_penalty': 0.50}
    expected = {'reward_config': expected_reward, 'dqn_config': asdict(base.dqn),
                'signal_config': asdict(base.signal),
                'simulation_config': asdict(base.simulation),
                'network_sha256': sha256(base.network_file),
                'subscription_source_sha256': sha256(Path(__file__).with_name('subscription_connection.py')),
                'learning_schedule_steps': 150000,
                'completed_training_steps': 50000,
                'evaluation_seed_start': 2001, 'evaluation_episodes': 30,
                'traci_subscriptions': True, 'deterministic_algorithms': True}
    # Normalize dataclass tuples to JSON arrays before comparing.
    expected = json.loads(json.dumps(expected))
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Baseline protocol mismatch: {key}')
    if metadata['training_options']['seed'] != 22 or metadata['training_options']['scenario'] != 'random':
        raise ValueError('Baseline training seed/scenario mismatch')
    if metadata['training_options']['episode_seconds'] != 300:
        raise ValueError('Baseline episode length mismatch')
    runtime = collect_reproducibility_metadata()
    for key in ('git_commit_sha', 'python_version', 'sumo_version'):
        if metadata[key] != runtime[key]:
            raise ValueError(f'Baseline runtime mismatch: {key}')
    if sha256(source / 'final.zip') != metadata['model_sha256']:
        raise ValueError('Baseline model SHA256 mismatch')
    with zipfile.ZipFile(source / 'final.zip') as archive:
        model_data = json.loads(archive.read('data'))
    if model_data['num_timesteps'] != 50000 or model_data['_total_timesteps'] != 150000:
        raise ValueError('Baseline model step/schedule mismatch')
    routes = read_json(source / 'evaluation_routes.json')
    if [row['seed'] for row in routes] != list(range(2001, 2031)):
        raise ValueError('Baseline evaluation route seeds mismatch')
    rows = read_csv(source / 'evaluation_metrics.csv')
    if len(rows) != 30 or [int(r['seed']) for r in rows] != list(range(2001, 2031)):
        raise ValueError('Baseline evaluation rows mismatch')
    for metric in METRICS:
        values = np.array([float(r[metric]) for r in rows])
        if not np.isclose(values.mean(), summary[metric], rtol=0, atol=1e-10):
            raise ValueError(f'Baseline summary mismatch: {metric}')
    return metadata


def import_baseline(source: Path, output: Path, metadata: dict) -> None:
    folder = output / f'experiment_{BASELINE}'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('final.zip', 'training_routes.json', 'evaluation_routes.json'):
        shutil.copy2(source / name, folder / name)
    rows = read_csv(source / 'evaluation_metrics.csv')
    for row in rows:
        row['experiment'] = BASELINE
    write_csv(folder / 'evaluation_metrics.csv', rows)
    summary = read_json(source / 'summary.json')
    summary.update({'experiment': BASELINE, 'reused_existing_result': True})
    # Keep summary columns consistent for the common aggregator.
    summary.pop('reused_existing_result')
    write_json(folder / 'summary.json', summary)
    write_json(folder / 'experiment_config.json', {
        **metadata, 'experiment': BASELINE, 'reused_existing_result': True,
        'source_experiment': metadata['experiment'], 'source_folder': str(source.resolve()),
        'replay_buffer_used_for_new_training': False,
    })
    write_json(folder / 'progress.json', {'status': 'REUSED', **summary})


def summarize_local(output: Path) -> dict:
    """Compare actual traffic metrics only, with rules fixed before execution."""
    summaries = aggregate(output, list(CANDIDATES))
    reference = next(row for row in summaries if row['experiment'] == BASELINE)
    changes, safety = [], {}
    for row in summaries:
        candidate = row['experiment']
        rows = read_csv(output / f'experiment_{candidate}' / 'evaluation_metrics.csv')
        collisions = sum(int(r['collisions']) for r in rows)
        teleports = sum(int(r['teleports']) for r in rows)
        eligible = collisions == 0 and teleports == 0 and row['throughput'] >= reference['throughput'] * THROUGHPUT_RETENTION
        safety[candidate] = {'collisions': collisions, 'teleports': teleports,
                             'throughput_retention': row['throughput'] / reference['throughput'],
                             'eligible': eligible}
        change = {'candidate': candidate, 'alpha': row['alpha'], 'beta': row['beta'],
                  'switch_gamma': row['switch_gamma'], **safety[candidate]}
        for metric in METRICS:
            change[metric + '_change_pct'] = (row[metric] / reference[metric] - 1) * 100
        changes.append(change)
    write_csv(output / 'changes_vs_baseline.csv', changes)
    valid = [r for r in summaries if safety[r['experiment']]['eligible']]
    best = min(valid, key=lambda r: (r['avg_waiting_time'], r['max_waiting_time'],
                                   r['avg_queue'], r['phase_changes'], -r['throughput'], r['experiment'])) if valid else None
    result = {'baseline_candidate': BASELINE, 'recommended': best,
              'eligibility': safety, 'selection_rules': {
                  'reject_any_collisions_or_teleports': True,
                  'minimum_throughput_retention': THROUGHPUT_RETENTION,
                  'lexicographic_metric_order': TIE_ORDER},
              'new_training_runs': 5, 'new_training_timesteps': 250000,
              'training_seeds': [22], 'evaluation_seeds': list(range(2001, 2031)),
              'statistical_significance_test_performed': False,
              'global_optimum_established': False}
    write_json(output / 'recommendation.json', result)
    lines = ['# 발표용 국소 Reward 가중치 탐색', '',
             '신규 5개 후보를 각 50,000 step씩 독립 학습했습니다. 후보 2의 기존 결과는 재학습 없이 재사용했습니다.',
             '학습 seed 22, random 교통, 회차 300초, 수요 생성 240초, 평가 seed 2001~2030으로 동일합니다.',
             '기존 150,000-step 탐색률 일정을 유지했고 모든 평가는 deterministic=True입니다.',
             '학습 및 평가 수요 파일의 seed 순서와 SHA-256이 여섯 후보 모두 일치했습니다.', '',
             '## 1. 전체 후보 비교표', '',
             '수치는 평가 30회의 평균 ± 표본 표준편차입니다. 이는 교통 seed 간 변동입니다.', '',
             '| 후보 | (α, β, γ) | Average Waiting ↓ (초) | Maximum Waiting ↓ (초) | Average Queue ↓ (대) | Phase Changes ↓ (회) | Throughput ↑ (대) |',
             '|---|---|---|---|---|---|---|']
    for row in summaries:
        fields = [str(row['experiment']) + (' (기존)' if row['experiment'] == BASELINE else ''),
                  str(CANDIDATES[row['experiment']])]
        fields += [f"{row[m]:.2f} ± {row[m + '_std']:.2f}" for m in METRICS]
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', 'Maximum Waiting은 회차별 최대 대기시간의 평균이며 30회 전체의 최댓값이 아닙니다.',
              'Average Waiting은 실제 출발한 차량을 대상으로 하며, 종료 시 미완료 차량을 포함하고 진입 전 pending 차량은 제외합니다.', '',
              '## 2. 기존 (0.30, 0.30, 0.50) 대비 변화율', '',
              '음수는 감소, 양수는 증가입니다. 처리량은 증가가, 나머지 네 지표는 감소가 좋습니다.', '',
              '| 후보 | Average Waiting | Maximum Waiting | Average Queue | Phase Changes | Throughput | 선정 가능 |',
              '|---|---|---|---|---|---|---|']
    for change in changes:
        fields = [str(change['candidate'])] + [f"{change[m + '_change_pct']:+.2f}%" for m in METRICS]
        fields += ['예' if change['eligible'] else '아니오']
        lines.append('| ' + ' | '.join(fields) + ' |')
    diagonal = [next(r for r in summaries if r['experiment'] == i) for i in (1, 2, 3, 4)]
    monotonic = all(b['avg_waiting_time'] <= a['avg_waiting_time'] for a, b in zip(diagonal, diagonal[1:]))
    all_monotonic = all(all((b[m] >= a[m] if m == 'throughput' else b[m] <= a[m])
                            for m in METRICS) for a, b in zip(diagonal, diagonal[1:]))
    lines += ['', '## 3. 높은 가중치에서 계속 좋아지는지', '',
              '세 계수를 함께 높인 후보 1→2→3→4에서 평균 대기시간은 ' + ('계속 감소했습니다.' if monotonic else '계속 감소하지 않았습니다.'),
              '다섯 지표가 모두 단조 개선되는지는 ' + ('이번 비교에서 확인되었습니다.' if all_monotonic else '이번 비교에서 확인되지 않았습니다.'),
              '후보 5·6은 전환 계수와 대기 계수의 영향을 일부 분리해 비교하는 보조 조합입니다.', '',
              '## 4. 현재 실험 범위 내 추천 후보', '',
              '사전 고정 규칙: 충돌·텔레포트 발생 또는 기준 처리량의 95% 미만이면 제외합니다.',
              '남은 후보에서 Average Waiting → Maximum Waiting → Average Queue → Phase Changes 순으로 비교합니다.']
    if best:
        winner = best['experiment']
        improvement = next(c for c in changes if c['candidate'] == winner)
        lines += [f'추천 후보는 **{winner}: {CANDIDATES[winner]}**입니다.',
                  f"기준 대비 평균 대기 변화 {improvement['avg_waiting_time_change_pct']:+.2f}%, 최대 대기 변화 {improvement['max_waiting_time_change_pct']:+.2f}%, 처리량 변화 {improvement['throughput_change_pct']:+.2f}%입니다."]
    else:
        lines += ['제외 기준을 통과한 후보가 없어 추천 후보를 결정하지 않았습니다.']
    for candidate, status in safety.items():
        if not status['eligible']:
            lines.append(f"후보 {candidate} 제외 사유: collisions={status['collisions']}, teleports={status['teleports']}, 처리량 유지율={status['throughput_retention']:.4f}.")
    lines += ['', '## 5. 전역 최적값이라고 할 수 없는 이유', '',
              '학습 seed는 22 하나이며 여섯 개 국소 조합만 평가했습니다. 모든 계수 조합을 탐색하지 않았고,',
              '50,000-step 이후 순위 변화도 검증하지 않았습니다. 통계적 유의성 검증과 다중 학습 seed 실험은 수행하지 않았습니다.',
              '이번 평가 seed로 후보를 선택했으므로 별도의 미사용 교통 seed에 대한 일반화도 검증하지 않았습니다.', '',
              '## 6. 발표용 3문장 결론', '']
    if best:
        lines += [f'기존 최고 조합 주변의 여섯 후보를 동일한 50,000-step 학습 조건과 교통 시나리오 30개로 비교했습니다.',
                  f"충돌·텔레포트와 처리량 유지 조건을 적용한 뒤 평균 대기시간을 우선한 결과, {CANDIDATES[best['experiment']]} 조합이 이번 범위에서 가장 유망했습니다.",
                  '이는 단일 학습 seed를 사용한 국소 탐색 결과이며, 전역 최적값을 입증하는 결과는 아닙니다.']
    else:
        lines += ['여섯 후보를 동일한 학습·평가 조건에서 비교했습니다.',
                  '이번 비교에서는 안전·처리량 조건을 만족하는 추천 후보를 결정하지 못했습니다.',
                  '단일 학습 seed의 국소 탐색이므로 전역 최적값을 주장하지 않습니다.']
    lines += ['', '원본 수치: comparison.csv, changes_vs_baseline.csv, 각 experiment 폴더의 evaluation_metrics.csv.',
              '신규 학습 5회, 총 250,000 step, 신규 평가 150회 + 기존 평가 30회 재사용.', '']
    (output / 'report-ko.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=Path('output/reward-sensitivity-50000-2026-10-04/experiment_4'))
    parser.add_argument('--output', type=Path, default=Path('model/results/reward_sensitivity_local'))
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        print(json.dumps(summarize_local(args.output), ensure_ascii=False, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f'experiment_{i}' / 'summary.json').exists() for i in CANDIDATES if i != BASELINE):
        parser.error('Output already contains new-candidate results; choose a fresh folder or --summarize-only')
    metadata = verify_baseline(args.baseline)
    settings = {'output': str(args.output.resolve()), 'timesteps': 50000,
                'schedule_timesteps': 150000, 'seed': 22, 'scenario': 'random',
                'episode_seconds': 300, 'eval_seed_start': 2001, 'eval_episodes': 30,
                'cache': True, 'verify_cache': False}
    write_json(args.output / 'search_config.json', {
        **settings, 'candidates': CANDIDATES, 'baseline_source': str(args.baseline.resolve()),
        'selection_rules': {'minimum_throughput_retention': THROUGHPUT_RETENTION,
                            'reject_any_collisions_or_teleports': True, 'tie_order': TIE_ORDER},
        'local_search_source_sha256': sha256(Path(__file__)),
        'statistics': 'means and sample SD over evaluation traffic seeds; no significance test',
    })
    import_baseline(args.baseline, args.output, metadata)
    started = time.monotonic()
    failures = []
    with ProcessPoolExecutor(max_workers=min(max(1, args.workers), 5)) as pool:
        futures = {pool.submit(worker, i, {**settings, 'weights': weights}): i
                   for i, weights in CANDIDATES.items() if i != BASELINE}
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                summary = future.result()
                print(json.dumps({'completed_candidate': candidate, 'summary': summary}), flush=True)
            except Exception as exc:
                failures.append({'candidate': candidate, 'error': repr(exc)})
                write_json(args.output / 'failures.json', failures)
                print(json.dumps({'failed_candidate': candidate, 'error': repr(exc)}), flush=True)
    if failures:
        write_json(args.output / 'status.json', {'status': 'FAILED', 'failures': failures})
        return 1
    result = summarize_local(args.output)
    write_json(args.output / 'status.json', {'status': 'COMPLETED',
                                           'elapsed_seconds': time.monotonic() - started})
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
