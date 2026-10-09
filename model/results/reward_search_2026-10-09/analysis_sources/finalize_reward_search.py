"""Combine transparent exploratory stages and untouched final-test results."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_quick_reward_search import read_json, read_csv, pct


root = Path(sys.argv[1]).resolve()
stage = root / 'followup_confirmation'
status = read_json(stage / 'status.json')
if status['status'] != 'COMPLETED':
    raise RuntimeError('Follow-up confirmation incomplete')
rec = read_json(stage / 'recommendation.json')
evidence = read_json(stage / 'comparison.json')
assert evidence['paired_route_hashes_verified']
rows = {row['experiment']: row for row in evidence['results']}
candidate, baseline, fixed = rows[4], rows[1], rows['fixed_time']
model_meta = read_json(root / 'experiment_4' / 'experiment_config.json')
metrics = ['avg_waiting_time', 'max_waiting_time', 'avg_queue', 'throughput', 'phase_changes', 'unfinished', 'pending']
changes = {label: {key: pct(candidate[key], reference[key]) for key in metrics}
           for label, reference in [('default_reward', baseline), ('fixed_time', fixed)]}
trial_rows = read_csv(stage / 'holdout' / 'experiment_4' / 'evaluation_metrics.csv')
base_rows = read_csv(stage / 'holdout' / 'experiment_1' / 'evaluation_metrics.csv')
assert [r['seed'] for r in trial_rows] == [r['seed'] for r in base_rows] == [str(s) for s in range(7001, 7021)]
diffs = [float(a['avg_waiting_time']) - float(b['avg_waiting_time']) for a, b in zip(trial_rows, base_rows, strict=True)]
paired = {'better': sum(d < 0 for d in diffs), 'tied': sum(d == 0 for d in diffs),
          'worse': sum(d > 0 for d in diffs), 'differences_seconds': diffs,
          'mean_difference_seconds': float(np.mean(diffs))}
stress = read_json(root / 'stress_candidate4' / 'comparison.json')
assert stress['paired_route_hashes_verified']
stress_changes = []
for scenario in sorted({r['scenario'] for r in stress['results']}):
    group = {r['candidate_id']: r for r in stress['results'] if r['scenario'] == scenario}
    stress_changes.append({'scenario': scenario,
        'waiting_vs_default_pct': pct(group[4]['avg_waiting_time'], group[1]['avg_waiting_time']),
        'waiting_vs_fixed_pct': pct(group[4]['avg_waiting_time'], group[None]['avg_waiting_time']),
        'throughput_vs_default_pct': pct(group[4]['throughput'], group[1]['throughput']),
        'throughput_vs_fixed_pct': pct(group[4]['throughput'], group[None]['throughput'])})
final = {**rec, 'reward_config': model_meta['reward_config'],
         'model_sha256': model_meta['model_sha256'], 'final_test_changes_pct': changes,
         'final_test_paired_waiting': paired, 'stress_diagnostics': stress_changes,
         'scope': 'six candidate weights, 50000 training steps, one training seed; exploratory followup with 20 new paired random traffic seeds; six fixed scenarios with two seeds each as diagnostics',
         'source_recommendation': str(stage / 'recommendation.json')}
(root / 'final_recommendation.json').write_text(json.dumps(final, indent=2), encoding='utf-8')
if rec['confirmed']:
    (root / 'recommended_reward_config.json').write_text(json.dumps({
        'reward_config': model_meta['reward_config'], 'model_path': rec['model_path'],
        'model_sha256': model_meta['model_sha256'], 'scope': final['scope'],
        'default_configuration_changed': False}, indent=2), encoding='utf-8')
lines = ['보상 파라미터 탐색 최종 보고서 — 2026-10-09', '',
         ('현재 탐색 범위의 추천 후보: 4' if rec['confirmed'] else '새 보상 설정 추천 확인 실패'),
         'queue_weight=1.0, waiting_weight=0.3, max_waiting_weight=0.5, switch_penalty=0.5',
         '정규화 queue_scale=10, waiting_scale=6000, max_waiting_scale=120; DQN 할인율 gamma=0.95 유지.',
         '추천은 전역 최적값의 증명이 아니며, 이 학습량과 평가 범위에서 유망한 설정이다.', '',
         '최종 미사용 교통 평가: seed 7001~7020, random 교통 20회, 회차 300초/수요 240초.',
         '후보 4를 이 평가 전에 고정했다. 새 모델 학습 없이 저장된 50,000-step 모델을 평가했다.',
         '후보 | 평균대기(s) | 회차최대대기 평균(s) | 최악대기(s) | 평균대기열 | 처리량 | 미완료 | 진입대기 | 전환수']
for label, row in [('기본 보상', baseline), ('추천 후보 4', candidate), ('고정 신호', fixed)]:
    keys = ['avg_waiting_time', 'max_waiting_time', 'maximum_waiting_worst_episode', 'avg_queue', 'throughput', 'unfinished', 'pending', 'phase_changes']
    lines.append(label + ' | ' + ' | '.join(f'{row[k]:.3f}' for k in keys))
for label, values in changes.items():
    lines.append(f"{label} 대비 평균 대기 {values['avg_waiting_time']:+.2f}%, 회차최대대기 평균 {values['max_waiting_time']:+.2f}%, 처리량 {values['throughput']:+.2f}%")
lines += [f"기본 보상 대비 대기 개선/동률/악화: {paired['better']}/{paired['tied']}/{paired['worse']}회.",
          f"충돌/텔레포트 합계: 후보4 {candidate['collisions_total']}/{candidate['teleports_total']}, 기본값 {baseline['collisions_total']}/{baseline['teleports_total']}.",
          '', '실험 설계 및 탐색 경과:',
          '6개 보상 조합을 새로 학습했다. 동일 구조 60입력/8행동, 학습 seed22, 후보별50,000 step, 총300,000 step.',
          '학습 탐색률 일정150,000 step을 보존했다. queue가중치/정규화/신호시간/학습 하이퍼파라미터는 동일하다.',
          'seed4001~4008의 첫 검증에서는 후보6(0.3,0.3,0.8)이1위였다. seed5001~5010의 독립 검증에서는 기본값보다 평균대기가1.58%나빠져 탈락했다.',
          '첫 독립 검증에서 후보4가 유망해 새 탐색 단계의 후보로 고정했다. 따라서5001~5010은 후보4의 최종 성능 증거가 아니라 선택에 사용된 자료다.',
          '이후 전혀 쓰지 않은7001~7020에서 후보4를 한번 확인했다. 이 결과를 본 추가 후보 변경은 하지 않았다.',
          '원래 후보6의 실패 기록과 recommendation.json은 보존했다. 최종 기록은final_recommendation.json이다.', '',
          '고정 교통 상황별 추가 진단 — 각 2회, 후보 선택에는 사용하지 않음:',
          '시나리오 | 평균대기 변화 vs기본(%) | 처리량 변화 vs기본(%) | 평균대기 변화 vs고정신호(%) | 처리량 변화 vs고정신호(%)']
for row in stress_changes:
    lines.append(row['scenario'] + ' | ' + ' | '.join(f'{row[k]:+.2f}' for k in ['waiting_vs_default_pct', 'throughput_vs_default_pct', 'waiting_vs_fixed_pct', 'throughput_vs_fixed_pct']))
lines += ['', '해석 한계:',
          '한 개 학습 seed와 제한된 후보/학습량이다. 통계적 유의성, 여러 학습 seed에서의 재현성,150,000-step 학습 후 우열을 입증하지 않았다.',
          '보상 기본값 비교는 동일50,000-step 신규 학습 모델 기준이며, 과거 완전 학습 체크포인트 대비 비교가 아니다.',
          '평균대기는 실제 진입 차량 기준으로 미완료 차량을 포함하지만 진입 전 pending차량은 제외한다. 회차최대대기 평균과 전체최악대기는 서로 다른 지표다.',
          '고정 시나리오별2회 결과는 경향을 보는 추가 진단이며 충분한 강건성 검증은 아니다.',
          '학습/평가에서 적색신호 급제동 경고가 관찰됐다. 충돌/텔레포트0을 모든 안전성의 증거로 해석하면 안 된다.',
          '기존 기본 모델 및 기본 보상설정은 변경하지 않았다. 추천 설정과 모델을 별도 저장했다.', '',
          '자료 위치:', str(root / 'recommended_reward_config.json'), rec['model_path'],
          str(stage / 'comparison.csv'), str(root / 'stress_candidate4' / 'comparison.csv'),
          str(root / 'runtime_packages.json'), '']
(root / 'final-report-ko.txt').write_text('\n'.join(lines), encoding='utf-8')
fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout='constrained')
ordered = [baseline, candidate, fixed]
labels = ['Default reward', 'Candidate 4', 'Fixed-Time']
for ax, metric, title in zip(axes, ['avg_waiting_time', 'max_waiting_time', 'throughput'],
                            ['Mean waiting (s)', 'Mean episode max wait (s)', 'Arrivals per episode']):
    bars = ax.bar(labels, [r[metric] for r in ordered], yerr=[r[metric + '_std'] for r in ordered],
                  color=['#94a3b8', '#0d9488', '#cbd5e1'], capsize=4)
    ax.bar_label(bars, fmt='%.1f', padding=4)
    ax.set_title(title)
    ax.set_ylim(0, max(r[metric] + r[metric + '_std'] for r in ordered) * 1.2)
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Final fresh test: 20 paired traffic seeds | mean +/- sample SD\n'
             '50,000 training steps; one training seed; exploratory follow-up candidate fixed before test', fontsize=12)
fig.savefig(root / 'final_comparison.png', dpi=160)
plt.close(fig)
print(json.dumps({'confirmed': rec['confirmed'], 'changes_pct': changes, 'paired': paired,
                  'stress': stress_changes, 'report': str(root / 'final-report-ko.txt')}, indent=2))
