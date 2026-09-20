# Evaluation results

이 폴더의 CSV와 PNG는 학습에 사용하지 않은 seed로 Fixed-Time과 DQN을
비교한 결과입니다. 모델을 채택할 때는 평균 대기시간 하나만 보지 말고 최대
대기시간, Queue, Throughput, 신호 전환 횟수와 긴 simulation horizon을 함께
확인해야 합니다.

## 300k-step experiment — rejected

- Training session: `48c30cca6cb74a28a8fbf69f3dd0fcc4`
- Training: random traffic, seed 1, 300,000 steps, 120-second episodes
- Selected checkpoint: best checkpoint at 297,171 steps
- Evaluation seeds: 2001–2030, 30 paired runs

### 120-second evaluation

| Metric | Fixed-Time | DQN | Result |
|---|---:|---:|---:|
| Average waiting time | 2.683 s | 2.256 s | 15.9% better |
| Maximum waiting time | 35.067 s | 56.400 s | 60.8% worse |
| Average queue | 2.796 | 2.645 | 5.4% better |
| Maximum queue | 4.500 | 3.667 | 18.5% better |
| Throughput | 9.700 | 10.067 | 3.8% better |
| Phase changes | 13.000 | 22.867 | 75.9% more |

### 300-second stress evaluation

| Metric | Fixed-Time | DQN | Result |
|---|---:|---:|---:|
| Average waiting time | 6.610 s | 16.571 s | 150.7% worse |
| Maximum waiting time | 65.433 s | 100.000 s | 52.8% worse |
| Average queue | 5.352 | 16.710 | 212.2% worse |
| Throughput | 81.767 | 54.167 | 33.8% worse |
| Phase changes | 33.000 | 46.300 | 40.3% more |

짧은 episode의 평균값은 개선됐지만 starvation과 잦은 신호 전환이 발생했고,
긴 horizon에서 성능이 붕괴했습니다. 따라서 이 checkpoint는 최종 모델로
채택하거나 Git에 배포하지 않습니다. 현재 `evaluation_metrics.csv`와 PNG는
120초 평가의 per-episode 결과와 시각화입니다.

다음 학습은 300초 이상 episode, 더 강한 maximum-wait/switch penalty, 다양한
traffic horizon을 사용한 validation, 30개 이상의 고정 hold-out seed를 사용해야
합니다.

다음 실험부터 기본 reward 설정은 starvation과 과도한 전환을 줄이기 위해
`max_waiting_weight=1.0`, `max_waiting_scale=90.0`,
`switch_penalty=0.4`로 강화했습니다.
