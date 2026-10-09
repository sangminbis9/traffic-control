"""Render the completed experiment's original evidence without reselecting."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def pct(value, reference):
    return (float(value) / float(reference) - 1) * 100 if float(reference) else None


def main():
    folder = Path(sys.argv[1]).resolve()
    status = read_json(folder / 'status.json')
    if not status['status'].startswith('COMPLETED'):
        raise RuntimeError('Search has not completed')
    config = read_json(folder / 'search_config.json')
    recommendation = read_json(folder / 'recommendation.json')
    validation = read_json(folder / 'comparison.json')['results']
    holdout = read_json(folder / 'holdout_comparison.json')['results']
    by_id = {row['experiment']: row for row in holdout}
    winner = recommendation['validation_winner_candidate_id']
    selected = by_id.get(winner)
    baseline = by_id[1]
    comparisons = []
    if selected:
        for label, reference in [('default_reward', baseline), ('fixed_time', by_id['fixed_time'])]:
            change = {'candidate': winner, 'reference': label}
            for metric in ['avg_waiting_time', 'max_waiting_time', 'avg_queue', 'phase_changes', 'throughput', 'unfinished', 'pending']:
                change[metric + '_difference'] = selected[metric] - reference[metric]
                change[metric + '_change_pct'] = pct(selected[metric], reference[metric])
            comparisons.append(change)
        paired = read_csv(folder / 'holdout' / f'experiment_{winner}' / 'evaluation_metrics.csv')
        paired_baseline = read_csv(folder / 'holdout' / 'experiment_1' / 'evaluation_metrics.csv')
        assert [row['seed'] for row in paired] == [row['seed'] for row in paired_baseline]
        differences = [float(a['avg_waiting_time']) - float(b['avg_waiting_time']) for a, b in zip(paired, paired_baseline, strict=True)]
        paired_stats = {'candidate': winner, 'reference': 1, 'seeds': [int(row['seed']) for row in paired],
                        'mean_waiting_differences_seconds': differences,
                        'better_episodes': sum(value < 0 for value in differences),
                        'equal_episodes': sum(value == 0 for value in differences),
                        'worse_episodes': sum(value > 0 for value in differences),
                        'mean_difference_seconds': float(np.mean(differences)),
                        'sample_sd_difference_seconds': float(np.std(differences, ddof=1)),
                        'statistical_significance_test_performed': False}
        (folder / 'paired_holdout_differences.json').write_text(json.dumps(paired_stats, indent=2), encoding='utf-8')
    (folder / 'holdout_changes.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
    if recommendation['recommended_candidate_id'] is not None:
        metadata = read_json(folder / f'experiment_{winner}' / 'experiment_config.json')
        (folder / 'recommended_reward_config.json').write_text(json.dumps({
            'reward_config': metadata['reward_config'],
            'model_path': recommendation['model_path'],
            'model_sha256': metadata['model_sha256'],
            'scope': recommendation['scope'],
            'training_steps': recommendation['training_steps_per_candidate'],
            'default_configuration_changed': False,
        }, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout='constrained')
    metrics = [('avg_waiting_time', 'Mean waiting (s)', 'lower is better'),
               ('max_waiting_time', 'Mean episode maximum wait (s)', 'lower is better'),
               ('throughput', 'Arrivals per episode', 'higher is better')]
    labels = ['Fixed-Time' if row['experiment'] == 'fixed_time' else
              ('Default reward' if row['experiment'] == 1 else f"Candidate {row['experiment']}") for row in holdout]
    colors = ['#94a3b8' if row['experiment'] in (1, 'fixed_time') else
              ('#0d9488' if row['experiment'] == winner else '#60a5fa') for row in holdout]
    for ax, (metric, title, direction) in zip(axes, metrics):
        bars = ax.bar(labels, [row[metric] for row in holdout], color=colors,
                      yerr=[row[metric + '_std'] for row in holdout], capsize=4)
        ax.bar_label(bars, fmt='%.1f', padding=5, fontsize=10)
        ax.set_title(f'{title}\n{direction}', fontsize=11)
        ax.tick_params(axis='x', rotation=20)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(row[metric] + row[metric + '_std'] for row in holdout) * 1.2)
    fig.suptitle('Reward search: unseen traffic holdout | mean +/- sample SD\n'
                 f"{len(config['holdout_seeds'])} paired traffic seeds, one training seed, {config['timesteps']:,} training steps", fontsize=14)
    fig.savefig(folder / 'holdout_comparison.png', dpi=160)
    plt.close(fig)
    lines = [
        '보상 파라미터 빠른 탐색 결과 (2026-10-09)',
        f"상태: {status['status']}, 본 실험 소요: {status['elapsed_seconds'] / 60:.1f}분 (환경 준비/벤치마크 제외)",
        f"검증에서 선택한 후보: {winner}, 가중치(waiting/max_waiting/switch): {recommendation['validation_winner_weights']}",
        f"별도 holdout 확인: {recommendation['holdout_confirmation']['confirmed']}",
        'queue_weight=1.0; 정규화, 신호시간, DQN 구조와 학습 설정은 모든 후보에서 동일.',
        f"후보 {len(validation)}개 × {config['timesteps']:,} step, 학습 seed {config['seed']}, 탐색률 일정 {config['schedule_timesteps']:,} step.",
        f"선정 seed {config['validation_seeds']}; 미사용 holdout seed {config['holdout_seeds']}.",
        '모든 교통은 random, 300초 평가/240초 수요. 각 후보 신규 학습, 동일 수요 파일 해시 확인.',
        '충돌/텔레포트 없음과 기본 처리량 95% 이상을 조건으로 평균 대기시간 최소화.',
        '상위 후보는 holdout 전에 확정했으며 holdout 결과로 후보를 다시 고르지 않았음.',
        '', '선정용 검증 결과:',
        '후보 | waiting/max_waiting/switch | 평균대기(s) | 회차최대대기 평균(s) | 평균대기열 | 처리량',
    ]
    for row in validation:
        lines.append(f"{row['experiment']} | {row['alpha']}/{row['beta']}/{row['switch_gamma']} | {row['avg_waiting_time']:.3f} | {row['max_waiting_time']:.3f} | {row['avg_queue']:.3f} | {row['throughput']:.3f}")
    lines += ['', '별도 holdout 결과:', '후보 | 평균대기(s) | 회차최대대기 평균(s) | 최악대기(s) | 평균대기열 | 처리량 | 미완료 | 진입대기 | 최대진입대기 | 전환수']
    for row in holdout:
        keys = ['avg_waiting_time', 'max_waiting_time', 'maximum_waiting_worst_episode', 'avg_queue', 'throughput', 'unfinished', 'pending', 'maximum_pending_worst_episode', 'phase_changes']
        lines.append(str(row['experiment']) + ' | ' + ' | '.join(f'{row[key]:.3f}' for key in keys))
    if selected:
        lines += ['', '검증에서 미리 선택한 후보의 holdout 변화:']
        for change in comparisons:
            lines.append(f"{change['reference']} 대비 평균대기 {change['avg_waiting_time_change_pct']:+.2f}%, 최대대기 평균 {change['max_waiting_time_change_pct']:+.2f}%, 처리량 {change['throughput_change_pct']:+.2f}%")
        lines.append(f"기본 설정 대비 대기시간 개선/동률/악화 회차: {paired_stats['better_episodes']}/{paired_stats['equal_episodes']}/{paired_stats['worse_episodes']}")
    lines += ['', '해석 범위:',
              '이번 여섯 후보와 단일 학습 seed/짧은 학습/표본 교통 안에서만 판단. 전역 최적값 또는 통계적 유의성을 입증하지 않음.',
              '평균 대기는 실제 진입 차량 기준이며, 진입 전 pending 차량을 제외. 미완료 차량은 포함.',
              '최대대기 평균은 회차 최대값의 평균이며, 최악 회차값은 별도 제시.',
              '서로 다른 보상 가중치의 episode_reward 합계를 직접 비교하여 후보를 선정하지 않음.',
              '여러 학습 seed에 대한 검증은 이번 비교에 포함하지 않음.',
              '학습 중 적색 신호 앞 급제동/급정지 SUMO 경고가 관찰됨. 충돌/텔레포트 집계만으로 모든 안전성을 입증하지 않으며 급제동 횟수는 후보별 평가 지표에 포함하지 않음.',
              '기본 설정 및 기존 기본 모델은 변경하지 않음.',
              '재현 자료: search_config.json, runtime_packages.json, 각 experiment의 모델/설정/수요 해시/평가 CSV.', '']
    stress_file = folder / 'stress' / 'comparison.json'
    if stress_file.exists():
        stress = read_json(stress_file)
        lines += ['고정 시나리오 추가 진단 (각 시나리오 새 교통 seed 2개, 추천값 재선정에 사용하지 않음):',
                  '시나리오 | 제어기 | 평균대기(s) | 회차최대대기 평균(s) | 처리량 | 미완료 | 진입대기']
        for row in stress['results']:
            values = [f"{row[key]:.3f}" for key in ['avg_waiting_time', 'max_waiting_time', 'throughput', 'unfinished', 'pending']]
            lines.append(f"{row['scenario']} | {row['controller']} | " + ' | '.join(values))
        lines += ['각 상황 2회 진단은 대규모 강건성 검증이 아님. 시나리오별 악화가 있으면 추천 적용 범위를 제한해야 함.', '']
    else:
        lines += ['고정 교통 시나리오별 강건성은 확인하지 않음.', '']
    (folder / 'report-ko.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'report': str(folder / 'report-ko.txt'), 'changes': comparisons, 'holdout': holdout}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
