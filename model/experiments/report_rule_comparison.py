"""Render the completed, frozen rule/DQN comparison without changing selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


POLICIES = ('cycle_safe', 'rule', 'dqn_seed_22', 'dqn_seed_42', 'dqn_seed_62', 'dqn_mean')
LABELS = {'cycle_safe': '고정 순환(동일 안전 제약)', 'rule': '튜닝된 규칙 기반',
          'dqn_seed_22': 'DQN seed 22', 'dqn_seed_42': 'DQN seed 42',
          'dqn_seed_62': 'DQN seed 62', 'dqn_mean': 'DQN 3회 독립 운용 평균'}
METRICS = (('avg_waiting_time', '평균 대기시간 ↓'), ('max_waiting_time', '최대 대기시간 ↓'),
           ('avg_queue', '평균 대기열 ↓'), ('throughput', '통과 차량 ↑'),
           ('phase_changes', '전환 횟수 ↓'), ('episode_reward', '누적 보상 ↑'),
           ('unfinished', '미완료 차량 ↓'), ('pending', '미진입 차량 ↓'),
           ('forced_changes', '강제 전환 횟수'))
SCENARIOS = ('uniform', 'north_south_congested', 'east_west_congested',
             'left_turn_congested', 'heavy', 'low')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
            *['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows], '']


def load_verified(output):
    verification = read(output / 'verification.json')
    if verification.get('status') != 'PASS':
        raise ValueError('A completed verification PASS is required before reporting')
    for key in ('matched_routes', 'frozen_code_models_runtime', 'selection_precedes_heldout',
                'complete_horizons', 'same_safety_path'):
        if verification.get(key) is not True:
            raise ValueError(f'Comparison verification did not confirm {key}')
    for name in ('protocol', 'selection'):
        if digest(output / f'{name}.json') != verification[f'{name}_sha256']:
            raise ValueError(f'Verified {name} changed')
    protocol, comparison = read(output / 'protocol.json'), read(output / 'comparison.json')
    if comparison['selection'] != read(output / 'selection.json'):
        raise ValueError('Comparison selection differs from verified selection')
    if (verification['observed_heldout_episodes'] != 300
            or verification['distinct_heldout_traffic_cases'] != 60):
        raise ValueError('Expected complete 5-policy/60-case heldout comparison')
    summaries = {(row['scenario'], row['policy_id']): row for row in comparison['summary']}
    if len(summaries) != len(comparison['summary']):
        raise ValueError('Duplicate policy/scenario summary')
    for scenario, count in (('all', 60), ('random', 30), *((name, 5) for name in SCENARIOS)):
        for policy in POLICIES:
            if summaries[scenario, policy]['episodes'] != count:
                raise ValueError('Summary cohort size mismatch')
    return protocol, comparison, verification, summaries


def render_chart(output, summaries):
    names = ('cycle_safe', 'rule', 'dqn_mean')
    labels = ('Fixed cycle\n(shared safety)', 'Tuned rule', 'DQN mean\n(3 separate runs)')
    panels = (('avg_waiting_time', 'Mean waiting time — lower is better', 'Seconds'),
              ('max_waiting_time', 'Episode maximum waiting — lower is better', 'Seconds'),
              ('throughput', 'Throughput — higher is better', 'Vehicles'),
              ('episode_reward', 'Episode reward — higher is better', 'Reward'))
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, (metric, title, unit) in zip(axes.flat, panels):
            values = [summaries['all', name][metric] for name in names]
            bars = ax.bar(labels, values, color=('#929DA8', '#E58D32', '#3F78AD'), width=.62)
            ax.bar_label(bars, labels=[f'{value:.2f}' for value in values], padding=5, fontsize=12)
            ax.set_title(title, fontsize=12, pad=14)
            ax.set_ylabel(unit)
            ax.axhline(0, linewidth=.7, color='#666666')
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
            ax.margins(y=.20)
        fig.suptitle('All 60 held-out traffic cases: predeclared evaluation', fontsize=19, y=.97)
        fig.text(.5, .025, 'Equal case weights: 30 random cases + 6 explicit scenarios with 5 cases each.\n'
                 'Bars: episode means. DQN mean averages 3 independent pretrained runs; it is not an ensemble.\n'
                 'Same 60-D observation, 8 actions, safety controller, 300-second horizon and paired traffic routes.',
                 ha='center', fontsize=10, color='#444444')
        fig.tight_layout(rect=(0, .09, 1, .93), h_pad=3, w_pad=3)
        fig.savefig(output / 'comparison.png', dpi=160, facecolor='white')
        plt.close(fig)


def render_text(output, protocol, comparison, verification, summaries):
    selection = comparison['selection']
    budget = protocol['simulation_budget']
    signal, reward = protocol['project_config']['signal'], protocol['project_config']['reward']
    lines = ['# 동일 조건의 규칙 기반 · 고정 순환 · DQN 비교', '',
        '판정 범위: 사전에 고정한 교차로·수요 생성기·시간 제약에서의 결과입니다. 모든 현실 교통에서의 우열이나 전역 최적성을 뜻하지 않습니다.',
        '규칙 기반과 DQN은 동일한 정규화·클리핑된 60차원 관측과 8개 행동을 사용합니다. 고정 순환은 교통 수요를 읽지 않습니다.',
        '모든 정책은 같은 SignalController의 황색·전체 적색·min/max green·max-red 제약을 통과합니다. 고정 순환도 force=True를 사용하지 않습니다.',
        '안전 제약의 강제 전환이 있으므로 고정 순환의 실제 신호는 명목 주기에서 변경될 수 있습니다.', '',
        '## 사전 고정한 비교 방법', '',
        f"- 선택된 규칙: {selection['winner']}",
        '- 대기열만 사용하는 후보를 포함한 37개 설정을 평가 전에 고정했습니다. 규칙의 계수는 Reward 가중치와 별개입니다.',
        '- 튜닝: random 8회 + 명시적 시나리오 6종 각 2회 = 20개 교통 사례. 회차 평균 누적 보상 최대, 동률 시 평균 대기시간 최소, 이후 후보 ID 순으로 선택했습니다.',
        '- 테스트: random 30회 + 명시적 시나리오 6종 각 5회 = 60개 교통 사례. 테스트를 열기 전에 규칙을 확정했으며 테스트 결과로 재선택하지 않았습니다.',
        '- 튜닝은 random/명시적 시나리오 비중 40/60, 전체 테스트는 50/50입니다. 각 단계 내부에서는 사례별 동일 가중치입니다.',
        '- random 테스트 seed: 41001–41030. 명시적 시나리오별 seed: 41101–41105. 튜닝 및 기존 학습 교통 seed와 분리했습니다.',
        '- 각 회차 300초, 차량 유입 생성 240초. 동일 scenario/seed끼리 차량 경로 파일 SHA-256이 일치합니다.',
        f"- 판단 간격 {signal['decision_interval']:g}초; 최소/최대 녹색 {signal['min_green']:g}/{signal['max_green']:g}초; 황색/전체 적색 {signal['yellow']:g}/{signal['all_red']:g}초; max-red {signal['max_red']:g}초.",
        f"- Reward 가중치(queue / waiting / max-waiting / switching): {reward['queue_weight']:g} / {reward['waiting_weight']:g} / {reward['max_waiting_weight']:g} / {reward['switch_penalty']:g}.",
        '- DQN은 기존 C14의 150,000-step 최종 모델 3개(학습 seed 22/42/62)를 모두 사용합니다. 이번 비교에서 DQN을 새로 학습하거나 우수 모델만 선택하지 않았습니다.', '',
        '선택된 규칙의 전체 파라미터:', '```json', json.dumps(selection['policy'], ensure_ascii=False, indent=2), '```', '',
        '## 지표 해석', '',
        '표는 회차 평균 ± 회차 간 표본 표준편차입니다. 표준편차는 95% 신뢰구간이 아닙니다.',
        '대기시간 단위는 초, 대기열·통과·미완료·미진입 단위는 차량 수입니다. 최대 대기시간은 회차별 최대값의 평균입니다.',
        '평균 대기시간은 진입한 차량의 관측 누적 대기 이력 기준이며 미완료 차량은 포함하고 미진입 차량은 제외합니다.',
        '통과 차량 ↑, 누적 보상 ↑가 유리합니다. 대기·대기열·미완료·미진입은 ↓가 유리합니다. 전환 감소는 독립 목표로 해석하지 말고 통과·대기와 함께 봐야 합니다.',
        '강제 전환은 안전 제약 개입의 횟수이며 값 자체로 정책의 안전성이나 우열을 단정하지 않습니다.',
        'DQN 평균 행은 같은 교통에서 모델 3개를 각각 운용한 결과를 평균한 통계입니다. 실제 실행 가능한 앙상블 정책이 아닙니다.', '',]
    for scenario, title in (('all', '전체 60회: 사전 고정한 동일 사례 가중치 평가'),
                            ('random', 'random 30회: 학습과 같은 수요 생성 분포의 하위집단')):
        lines += [f'## {title}', '']
        if scenario == 'random':
            lines += ['random은 전체 평가에 포함된 하위집단입니다. 사전 프로토콜의 주평가 범위로 지정된 것은 아니며, 전체 60회 결과를 대체하지 않습니다.', '']
        for selected in (METRICS[:6], METRICS[6:]):
            lines += table(['제어', *[label for _, label in selected]],
                [[LABELS[name], *[f"{summaries[scenario, name][metric]:.2f} ± {summaries[scenario, name][metric + '_std']:.2f}"
                                  for metric, _ in selected]] for name in POLICIES])
    lines += ['## 개별 DQN 모델과 규칙의 차이', '',
              '아래 값은 규칙 − 해당 DQN 모델의 평균입니다. 대기시간은 음수, 통과·보상은 양수가 규칙에 유리합니다.', '']
    lines += table(['범위', 'DQN 학습 seed', '평균 대기 차이', '최대 대기 차이', '통과 차량 차이', '보상 차이'],
        [[scenario, seed, *[f"{summaries[scenario, 'rule'][metric] - summaries[scenario, f'dqn_seed_{seed}'][metric]:+.2f}"
                           for metric in ('avg_waiting_time', 'max_waiting_time', 'throughput', 'episode_reward')]]
         for scenario in ('all', 'random') for seed in (22, 42, 62)])
    lines += ['## 짝지은 차이와 95% bootstrap 구간', '',
        '모든 차이는 앞 제어 − 뒤 제어입니다. 동일 교통 사례끼리 차이를 계산합니다.',
        'DQN 관련 구간은 고정된 3개 모델을 조건으로 한 교통 변동성입니다. 재학습 시의 불확실성이나 새로운 모델 모집단의 성능을 추정하지 않습니다.',
        '사전 프로토콜의 평가는 전체 60회를 사례별 동일 가중치로 집계합니다. random 30회는 학습과 같은 수요 생성 분포의 하위집단 분석이며 사전에 주평가로 지정하지 않았습니다.',
        '전체 60회 구간은 시나리오별 층화 resampling을 사용하지만, 명시적 6종 사이에 seed 5개를 재사용한 의존성을 보존하지 않으므로 기술적 참고로만 봅니다. 이 구간으로 강한 통계적 우월성을 주장하지 않습니다.',
        '명시적 시나리오별 표본은 5회로 작습니다. 여러 지표를 비교한 구간에는 다중비교 보정이 없습니다.', '']
    for scenario in ('all', 'random'):
        lines += [f'범위: {scenario}' + (' (전체 평가의 기술적 참고 구간)' if scenario == 'all' else ' (하위집단)'), '']
        lines += table(['앞 제어 − 뒤 제어', '지표', '평균 차이', '95% 구간'],
            [[f"{LABELS[row['first']]} − {LABELS[row['second']]}", dict(METRICS)[row['metric']],
              f"{row['mean_difference']:+.2f}", f"[{row['ci95_low']:+.2f}, {row['ci95_high']:+.2f}]"]
             for row in comparison['paired_differences'] if row['scenario'] == scenario
             and row['metric'] in dict(METRICS)])
    lines += ['## 명시적 스트레스 시나리오별 평균(각 5회)', '',
              '전체 우열을 단정하지 않고 어떤 수요에서 차이가 발생하는지 확인하는 표입니다.', '']
    lines += table(['시나리오', '제어', '평균 대기', '최대 대기', '통과 차량', '보상', '미완료', '미진입'],
        [[scenario, LABELS[name], *[f'{summaries[scenario, name][metric]:.2f}' for metric in
                                  ('avg_waiting_time', 'max_waiting_time', 'throughput', 'episode_reward', 'unfinished', 'pending')]]
         for scenario in SCENARIOS for name in ('cycle_safe', 'rule', 'dqn_mean')])
    lines += ['## 비용 및 감사 정보', '',
        f"규칙 튜닝: {budget['tuning_policies']}개 후보 × 20회 = {budget['tuning_episodes']:,}회 / {budget['tuning_decisions']:,} decision step.",
        f"동결 후 테스트: 5개 제어 × 60회 = {budget['heldout_episodes']:,}회 / {budget['heldout_decisions']:,} decision step.",
        '기존 DQN 학습량: 3개 모델 × 150,000 = 450,000 step. 규칙 튜닝과 DQN 학습의 step은 계산 작업이 달라 동일한 시간 비용을 의미하지 않습니다.',
        f"검증: {verification['status']} ({verification['verified_at']}). 관측된 테스트 회차 {verification['observed_heldout_episodes']}개 모두 완료; 동일 안전 경로·교통 해시·코드/모델/런타임 고정 확인."]
    timing_path = output / 'runtime_segments.json'
    if timing_path.exists():
        segments = read(timing_path)
        lines += [f"비교 실행 벽시계 시간: 호출 {len(segments)}개의 기록 합계 {sum(row['wall_seconds'] for row in segments):.1f}초. 프로토콜 준비 시간은 제외하며, 재개 호출은 별도로 기록됩니다. 병렬 worker 시간은 합산하지 않았습니다."]
    failures = sorted(output.glob('**/failures.json'))
    lines += [f"기록된 실패 시도: {sum(len(read(path)) for path in failures)}회. 재개 후 성공 여부와 별도로 원본 실패 기록을 보존합니다.", '',
              '모델별 학습 seed 변동은 dqn_training_seed_variation.csv에 별도로 기록했습니다(전체 60회 기준, 모델 3개).',
              '원자료: heldout_episodes.csv; summary.csv; paired_differences.csv.',
              '설정·출처: protocol.json; selection.json; verification.json; 각 작업의 runtime_segments.json 및 episodes.json.',
              '시각화: comparison.png는 사전 프로토콜에 따라 전체 60회를 동일 가중치로 집계한 평균입니다. 그림에는 신뢰구간을 표시하지 않습니다.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    protocol, comparison, verification, summaries = load_verified(output)
    text = render_text(output, protocol, comparison, verification, summaries)
    render_chart(output, summaries)
    (output / 'comparison-ko.txt').write_text(text, encoding='utf-8')
    inputs = ('comparison.json', 'protocol.json', 'verification.json', 'selection.json',
              'heldout_episodes.csv', 'summary.csv', 'paired_differences.csv',
              'dqn_training_seed_variation.csv', 'runtime_segments.json')
    metadata = {'created_at': datetime.now(timezone.utc).isoformat(),
                'generator_sha256': digest(Path(__file__)),
                'input_sha256': {name: digest(output / name) for name in inputs if (output / name).exists()},
                'output_sha256': {name: digest(output / name) for name in ('comparison-ko.txt', 'comparison.png')},
                'evaluation_scope': 'all 60 predeclared cases, equal weight per case',
                'subgroup_scope': 'random 30 cases use the training traffic-generation distribution; not a preregistered primary scope',
                'dqn_ci_scope': 'conditional on the three frozen DQN models, not retraining uncertainty',
                'pooled_ci_scope': 'descriptive only; shared explicit-scenario seeds not preserved as cross-scenario clusters',
                'selection_changed': False}
    (output / 'report_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Created {output / 'comparison-ko.txt'} and comparison.png")


if __name__ == '__main__':
    main()
