# Default DQN model

현재 기본 모델 dqn_intersection.zip은 GitHub의
MinWoo/traffic-rl/results/models/dqn.zip을 그대로 사용합니다.

- 학습 seed: 22
- 학습 decisions: 150,000
- observation: 34차원
- model SHA-256: 920fe8d6141ad7f7621ec8e28cf5957fe7e92237ed2dc9d6f35b4f3efa13b28d
- DQN: 5-step return, MLP [128, 128], gamma 0.95
- 탐색 action 유지: 5~17 decisions
- 학습 수요: 방향별 150~1500대/시간 무작위

원본 프로젝트의 미사용 교통 seed 4001~4030, 6개 시나리오, 학습 seed
11·22·33 평가에서 세 모델 평균은 Fixed-Time보다 다음과 같이 개선됐습니다.

| Metric | Fixed-Time | DQN 평균 | 변화 |
|---|---:|---:|---:|
| Average waiting time | 20.006초 | 13.775초 | 31.15% 감소 |
| Average queue | 16.464대 | 12.834대 | 22.04% 감소 |
| Throughput | 173.983대 | 185.567대 | 6.66% 증가 |
| Maximum waiting time | 95.322초 | 89.541초 | 6.07% 감소 |

heavy 시나리오에서는 평균 대기와 최대 대기가 Fixed-Time보다 나빴고,
low에서는 처리량이 소폭 감소했습니다. 모든 교통 조건에서 우세하다는
의미는 아닙니다.

이 저장소에서는 MinWoo 모델과 동일하게 다음 계약을 사용합니다.

- 250m 접근로, 제한속도 11.11m/s, SUMO step 0.5초
- 300초 episode와 240초 수요
- 8개 queue + 총/최대 대기 + 16개 신호 제어 상태 + 8개 접근 차량
- max_red=60초 공정성 제한
- Queue/대기/최대 대기/switch 보상 가중치 1.0/0.1/0.1/0.2

기존 12차원 300,000-step 모델은 300초 평가에서 Fixed-Time보다 평균
대기시간과 Queue가 악화되어 기본 모델에서 제외했습니다. 로컬 복구용 사본은
api/artifacts/legacy/dqn_intersection_pre_minwoo.zip에 저장됩니다.

이 폴더의 기존 CSV와 PNG는 교체 전 12차원 모델에서 생성된 과거 결과입니다.
새 비교 결과는 웹 비교 연구실 또는 `python -m model.evaluate`로 다시 생성해야
합니다.
