"""Write the Korean 50k report and paired changes from the verified 25k run."""

import argparse
import json
from pathlib import Path

from model.experiments.reward_sensitivity import METRICS, write_csv


def summarize(previous: Path, current: Path) -> None:
    old = json.loads((previous / 'comparison.json').read_text(encoding='utf-8'))
    new = json.loads((current / 'comparison.json').read_text(encoding='utf-8'))
    assert old['paired_route_hashes_verified'] and new['paired_route_hashes_verified']
    before = {r['experiment']: r for r in old['results']}
    after = sorted(new['results'], key=lambda r: r['experiment'])
    changes = []
    for row in after:
        folder = current / f"experiment_{row['experiment']}"
        old_folder = previous / folder.name
        verification = json.loads((folder / 'prefix_verification.json').read_text(encoding='utf-8'))
        assert verification['verified']
        assert row['training_steps'] == 50000 and before[row['experiment']]['training_steps'] == 25000
        assert (folder / 'evaluation_routes.json').read_bytes() == (old_folder / 'evaluation_routes.json').read_bytes()
        change = {'experiment': row['experiment']}
        for metric in METRICS:
            earlier = before[row['experiment']][metric]
            change[f'{metric}_25000'] = earlier
            change[f'{metric}_50000'] = row[metric]
            change[f'{metric}_change_pct'] = (row[metric] / earlier - 1) * 100
        changes.append(change)
    write_csv(current / 'changes-from-25000.csv', changes)
    best = {m: (max if m == 'throughput' else min)(r[m] for r in after) for m in METRICS}
    lines = [
        '# 리워드 가중치 조합: 50,000-step 비교', '',
        '네 모델 모두 총 50,000 step을 실제 학습한 뒤 같은 교통 시나리오 30개로 평가했습니다.',
        '학습 seed 22, random 교통, 평가 seed 2001~2030, 회차당 300초, 수요 생성 240초입니다.',
        '기존 150,000-step 탐색률 일정을 그대로 유지했으며, replay buffer가 없는 이전 모델을',
        '단순 재개하는 대신 처음부터 재현했습니다. 25,000 step에서 이전 모델과 두 Q 네트워크,',
        '업데이트 횟수, 탐색률, 유지 중인 탐색 행동 및 교통 수요 파일이 정확히 일치했습니다.',
        '50,000-step 모델 간 학습·평가 수요 파일도 일치했고, 이전 평가와 같은 수요를 사용했습니다.', '',
        '평가 행동은 deterministic=True입니다. 수치는 평균 ± 표본 표준편차이며 굵은 값은 가장 좋은 평균입니다.', '',
        '| 실험 | Waiting α | Max Waiting β | Switch γ | Average Waiting ↓ (초) | Maximum Waiting ↓ (초) | Average Queue ↓ (대) | Phase Changes ↓ (회) | Throughput ↑ (대/회차) |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for row in after:
        fields = [str(row['experiment']) + (' (초기값)' if row['experiment'] == 2 else ''),
                  str(row['alpha']), str(row['beta']), str(row['switch_gamma'])]
        for metric in METRICS:
            mean = f"{row[metric]:.2f}"
            if row[metric] == best[metric]:
                mean = f'**{mean}**'
            fields.append(f"{mean} ± {row[metric + '_std']:.2f}")
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', '## 25,000 step 대비 변화', '',
              '음수는 감소, 양수는 증가입니다. 대기·대기열·전환은 감소가, 처리량은 증가가 좋습니다.', '',
              '| 실험 | Average Waiting | Maximum Waiting | Average Queue | Phase Changes | Throughput |',
              '|---|---|---|---|---|---|']
    for change in changes:
        values = [str(change['experiment'])] + [f"{change[m + '_change_pct']:+.2f}%" for m in METRICS]
        lines.append('| ' + ' | '.join(values) + ' |')
    lines += ['', '## 지표와 해석 범위', '',
              '- Average Waiting: 시뮬레이션에 실제 출발한 차량의 관측 누적 대기시간 평균. 종료 시 미완료 차량을 포함하며 진입 전 pending 차량은 제외합니다.',
              '- Maximum Waiting: 회차별 최대 대기시간의 30회 평균입니다.',
              '- Average Queue: 8개 진입 이동 그룹 전체 정지 차량 수의 시간 평균입니다.',
              '- Phase Changes: 강제 전환을 포함한 실제 녹색 phase 전환 횟수입니다.',
              '- Throughput: 300초 내 목적지에 도착한 차량 수입니다.',
              '- 학습 seed는 1개입니다. 표준편차는 교통 시나리오 간 변동이며 학습 seed 간 변동이 아닙니다.',
              '- 세 가중치를 동시에 바꾼 조합 비교입니다. 개별 계수의 독립적 효과나 최적성·수렴·통계적 유의성을 입증하지 않습니다.', '',
              '원본: comparison.csv, changes-from-25000.csv 및 각 experiment 폴더의 evaluation_metrics.csv.',
              '재현 확인: 각 experiment 폴더의 prefix_verification.json 및 experiment_config.json.',
              '30회 전체에서 관측한 최대 대기시간(실험 1~4): ' + ', '.join(f"{r['maximum_waiting_worst_episode']:.1f}초" for r in after) + '.', '']
    (current / 'comparison-ko.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'results': after, 'changes': changes}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=Path('output/reward-sensitivity-25000-2026-10-03'))
    parser.add_argument('--current', type=Path, default=Path('output/reward-sensitivity-50000-2026-10-04'))
    args = parser.parse_args()
    summarize(args.previous, args.current)
