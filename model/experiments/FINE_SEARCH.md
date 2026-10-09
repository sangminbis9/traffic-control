# Incumbent 중심 보상 가중치 정밀 탐색

이 문서는 실행 절차와 결과 해석 기준입니다. 실행 전 성능 수치나 승자를
가정하지 않습니다. 이전 결과인 `model/results/reward_search_2026-10-09/`,
기본 모델, 기본 보상 설정은 보존합니다. 비교 기준 incumbent의 가중치는
`(waiting, max_waiting, switching) = (0.3, 0.5, 0.5)`이며,
각 단계에서 같은 학습량과 학습 seed로 **새로 학습한 incumbent**를 비교합니다.
기존 50k 모델을 신규 150k 모델의 성능 기준으로 섞지 않습니다.

## 실행

저장소 루트에서 새 결과 폴더를 지정합니다. 시스템에 필요한 Python/SUMO
의존성이 준비되어 있어야 합니다. Windows 프로젝트 가상환경에서는
`python` 대신 `.\.venv\Scripts\python.exe`를 사용할 수 있습니다.

```powershell
python -m model.experiments.fine_reward_search --stage all --output model/results/reward_fine_search_2026-10-09 --workers 6
```

단계를 나누어 실행할 수도 있습니다. 같은 결과 폴더를 계속 사용할 때는
`--resume`을 지정합니다.

```powershell
python -m model.experiments.fine_reward_search --stage preflight --output model/results/reward_fine_search_2026-10-09 --workers 6
python -m model.experiments.fine_reward_search --stage fine --output model/results/reward_fine_search_2026-10-09 --workers 6 --resume
python -m model.experiments.fine_reward_search --stage multiseed --output model/results/reward_fine_search_2026-10-09 --workers 6 --resume
python -m model.experiments.fine_reward_search --stage final --output model/results/reward_fine_search_2026-10-09 --workers 6 --resume
```

`preflight`는 현재 환경의 `check_env`, 60차원 Observation, 8개 Action,
TraCI 구독 캐시와 원래 getter의 동일성, SUMO 실행을 확인합니다.
실험 worker 수만 병렬성을 바꾸며 각 worker는 독립 모델과 환경을 사용합니다.
결과 폴더는 기존 결과를 덮어쓰는 용도로 사용하지 않습니다.

`--resume`은 검증된 완료 작업을 재사용합니다. 중단된 학습의 replay buffer를
이어 쓰는 기능으로 해석하면 안 됩니다. 미완료 작업은 새로운 attempt 폴더에서
처음부터 다시 학습하고, 기존 시도와 그 실패 기록은 남깁니다.

## 고정 조건과 단계

현재 60입력/8행동 DQN, queue 가중치 1.0, 정규화 `/10`, `/6000`, `/120`,
신호 시간, reward 식, DQN 알고리즘과 하이퍼파라미터를 유지합니다.
학습은 random 교통, 회차 300초와 수요 생성 240초를 사용하며, 50k 실행에서도
원래 150k 탐색률 일정을 유지합니다. 평가 행동은 `deterministic=True`입니다.
학습 환경과 수요 생성 코드는 기존 worker를 그대로 사용합니다.

| 단계 | 대상 | 독립 학습량 | 학습 seed | 평가 교통 seed |
|---|---|---:|---|---|
| 정밀 탐색 | 27개 격자 조합과 필요한 경계 확장 | 조합당 50,000 | 22 | 8001–8020 |
| 다중 학습 seed | 검증 상위 3개 + incumbent, 중복 제거 | 조합·seed당 150,000 | 22, 42, 62 | 8001–8020 |
| 최종 random holdout | 미리 고정한 후보와 incumbent의 저장 모델 | 추가 학습 없음 | 위 3개 모델 각각 | 9001–9030 |
| 강건성 확인 | 같은 두 후보, 6개 고정 시나리오 | 추가 학습 없음 | 위 3개 모델 각각 | 시나리오마다 10001–10010 |

27개 격자는 waiting `{0.2, 0.3, 0.4}` × max_waiting `{0.4, 0.5, 0.6}` ×
switching `{0.4, 0.5, 0.6}`입니다. 가운데 incumbent의 후보 ID는 14입니다.
switching 계수와 DQN 할인율 `gamma=0.95`는 다른 값입니다.

검증 상위 3개가 탐색 경계에 있으면 해당 지점에서 경계 방향의 한 계수를
0.1씩 확장합니다. 미리 정한 최대 2회의 확장 이후에도 경계가 미해결이면
`boundary_state.json`에 기록하고 탐색 범위의 한계로 보고합니다.
경계 확장이 있으므로 실제 학습 모델 수가 최초 27개보다 늘 수 있습니다.

강건성 시나리오는 `uniform`, `north_south_congested`,
`east_west_congested`, `left_turn_congested`, `heavy`, `low`입니다.
random과 각 고정 시나리오는 서로 다른 조건이므로 한 평균으로 합치지 않습니다.
기존 탐색에 사용한 4001–4008, 5001–5010, 6001–6002, 7001–7020은
이번 최종 holdout에 사용하지 않습니다. 새 seed 목록은 실행 시작 시 설정에 고정합니다.

## 선택 고정과 실패 처리

선정에는 동일 학습 seed·평가 교통 seed의 실제 지표를 비교합니다.
학습 및 평가 수요의 SHA-256도 비교합니다. `episode_reward`는 계수별
목적함수가 달라 공통 성능 점수로 사용하지 않습니다.

검증 후보는 충돌·텔레포트가 없고 incumbent 처리량의 95% 이상을 유지하며,
회차별 최대 대기시간의 평균이 incumbent보다 증가하지 않아야 합니다.
통과한 후보를 평균 대기 → 최대 대기 평균 → 평균 대기열 → 실제 전환수 →
처리량 순으로 비교하고 마지막 동률은 후보 ID로 결정합니다.

`shortlist_lock.json`은 50k 결과에서 150k 대상 후보를 고정합니다.
150k에서는 세 학습 seed 중 어느 하나에서도 평균 대기가 악화되지 않는
후보를 우선합니다. `selection_lock.json`은 미사용 holdout과 강건성 평가를
열기 전에 최종후보와 사용 모델을 고정합니다.

최종 확인에서 random 평균 대기가 개선되고 각 학습 seed 방향도 일치해야 합니다.
random 및 각 강건성 시나리오에서 처리량 95% 유지, 충돌·텔레포트 0,
최대 대기 평균 비증가를 확인합니다. 강건성 평균 대기는 사전 설정한
5% 악화 상한도 확인합니다. 정확한 적용 조건은 실행 폴더의
`search_config.json`과 `final_recommendation.json`에 기록합니다.

조건을 통과하지 못하면 **incumbent를 유지합니다.** 최종 시험 결과를 본 뒤
2위 후보를 대신 채택하지 않습니다. 실행 중 오류·누락이 있으면 성공 또는
성능 개선으로 처리하지 않고, 동일한 selection lock으로 재개할 수 있게 기록합니다.
모든 단계가 완료되어 `status.json`이 `COMPLETED`인 경우에만 최종 보고서와
7개 그림을 생성합니다. `confirmed=false`인 완료 실험도 실패 조건을 명시한
보고서를 생성합니다. 이는 실행 실패와 구분됩니다.

`confirmed`는 사전 고정한 경험적 성능·안전 제약의 통과 여부입니다.
통계적 유의성을 필수 채택 조건으로 추가하지 않습니다. 따라서 조건을 통과했어도
paired 대기 차이 CI가 0을 포함하면 통계적 개선은 미확정이라고 따로 명시합니다.

## 통계 해석

`fine_search_analysis.py`의 분석을 보고서에서도 그대로 사용합니다.
후보−incumbent 차이는 같은 학습 seed와 교통 seed끼리 짝지어 계산합니다.
완전한 학습 seed × 교통 seed 표를 확인하고 두 seed 축을 독립적으로
재표집하는 paired two-way cluster percentile bootstrap을 사용합니다.
기본 bootstrap 반복 수는 5,000, 재표집 seed는 1729입니다.

- 개별 에피소드의 표준편차는 교통과 모델 변동을 기술하는 값이며 표준오차가 아닙니다.
- 학습 seed별 평균의 표준편차는 독립 학습 결과의 변동입니다.
- 95% CI는 두 seed 축을 재표집한 추정 불확실성입니다. 같은 모델의 에피소드
  30개를 독립 모델 30개처럼 세지 않습니다.
- 50k 단계에는 학습 seed가 하나이므로 CI가 학습 재현성을 검증하지 않습니다.
- 150k도 학습 seed가 3개뿐이어서 구간은 근사적이고 학습 변동 추정에 한계가 있습니다.
- 대기 차이 CI가 0을 포함하면 개선이 확정됐다고 표현하지 않습니다. 개별 평균의
  오차막대 중첩 여부만으로 paired 비교의 결론을 내리지 않습니다.
- 후보·시나리오 다중비교 보정은 하지 않습니다. 27개 중 가장 좋은 검증값이나
  일부 유리한 시나리오만으로 전역 최적성 또는 모든 상황의 우월성을 주장하지 않습니다.
- 평균들의 비율 변화와 에피소드별 변화율의 평균은 다릅니다. 보고서 본문의
  변화율은 평균들의 비율 기준이며 원본 분석에는 두 종류가 구분되어 있습니다.
- 기준 분모가 0이면 변화율은 산출 불가로 표시합니다.

평균 대기는 실제 진입 차량의 관측 누적 대기로, 종료 시 미완료 차량은 포함하지만
진입 전 pending 차량은 제외합니다. 회차 최대 대기시간의 평균과 전체 최악 회차의
최대 대기는 별개입니다. 처리량, 미완료, pending 진단도 함께 확인합니다.
정확히는 300초 cutoff까지 실제 진입한 departed 차량 전부가 평균 대기의 분모이며,
평균 대기열은 8개 진입 이동 그룹의 정지 차량 합계를 시간 평균한 값입니다.
충돌·텔레포트 0만으로 급제동 등 모든 안전성이 검증된 것은 아닙니다.
SUMO의 emergency braking/stop 경고는 별도의 실행 로그도 확인해야 합니다.
원래 baseline 결과 폴더는 파일별 SHA-256으로 실행 전 기록하고 완료 전에 대조합니다.

## 결과 위치와 보고서

결과 루트에는 다음 증거가 저장됩니다.

| 파일 | 내용 |
|---|---|
| `search_config.json` | 후보 격자, 고정 조건, seed, 선정 규칙, runtime 및 원본 해시 |
| `fine_search_results.csv`, `fine_search_analysis.json` | 50k 개별 결과와 분석 |
| `multiseed_results.csv`, `multiseed_analysis.json` | 150k 다중 학습 seed 검증 |
| `final_holdout_results.csv`, `final_holdout_analysis.json` | 사전 고정 후보의 미사용 random 평가 |
| `robustness_results.csv`, `robustness_analysis.json` | 6개 고정 시나리오 평가 |
| `*_summary.csv`, `pareto_front.csv` | 집계 비교와 다섯 지표의 비지배 후보 |
| `shortlist_lock.json`, `selection_lock.json` | 단계별 선택 고정 기록 |
| `boundary_state.json` | 경계 확장과 미해결 경계 |
| `final_recommendation.json` | 최종 확인 결과·실패 사유·incumbent 유지 여부 |
| `runs/`, `evaluations/` | 모델, 독립 실행 설정, 개별 평가와 수요 해시 |
| `audit/retraining_integrity/audit_result.json` | 선택 입력: 실행 복구, 추가 재학습 수 및 원본과의 재현 비교 결과 |
| `reports/final-report-ko.txt` | 실제 완료 결과만 포함한 한국어 보고서 |
| `reports/report_manifest.json` | 보고서가 읽은 입력 파일의 SHA-256 및 생성물 목록 |

최종 보고서만 재생성하려면:

```powershell
python -m model.experiments.fine_search_report --input model/results/reward_fine_search_2026-10-09
```

`--output`으로 보고서 전용 폴더를 지정할 수 있습니다. 보고서 생성기는 모델을
학습하거나 후보를 다시 선택하지 않습니다. 원본 CSV와 분석 JSON의 평균·행 수를
대조하고, 선택 고정 기록과 최종 추천이 일치하지 않으면 생성하지 않습니다.

재학습 무결성 감사 JSON이 있으면 보고서는 해당 파일도 SHA-256 입력 목록에
포함하고, 성과 비교 모델과 추가 감사 모델의 연산 수 및 폐기된 미완료 시도를
구분합니다. 감사 모델은 후보 선택·성능 집계·신뢰구간의 표본에 추가하지 않습니다.
`PASS`, `all_equal=true`, 5개 완료 모델과 5개 고유 비교의 전부 일치가 확인되어야
감사 통과라고 설명합니다. 감사 자료가 없거나 통과하지 않은 경우에는 그 상태를
명시하며, 일반적인 보고서 생성을 위해 감사 파일을 필수로 요구하지 않습니다.

7개 그림은 각각 PNG 300dpi와 PDF로 저장합니다. 한글 보고서와 별도로 그림에는
영문 제목·단위를 사용해 다른 컴퓨터에서의 한글 폰트 누락을 피합니다.

1. `01_final_five_metrics`: incumbent와 고정한 최종후보의 다섯 지표, 95% CI.
2. `02_fine_search_waiting`: 27개 및 경계 확장 후보의 평균 대기, 95% CI.
3. `03_max_waiting_weight`: 다른 두 계수는 incumbent 값으로 고정한 β 조건부 절편.
4. `04_switch_weight`: 같은 방식의 switching 조건부 절편.
5. `05_waiting_weight`: 같은 방식의 α 조건부 절편.
6. `06_training_seed_distribution`: 150k의 학습 seed별 평균과 학습 seed 평균의 표본 SD.
7. `07_scenario_robustness`: random holdout 및 6개 시나리오의 paired 대기·최대대기·처리량 변화, 95% CI.
   최종 선택이 incumbent 자체이면 같은 세 지표의 절대 평균과 95% CI를 표시합니다.
   이 경우 새 후보와의 개선 비교가 아니라 incumbent의 시나리오별 참조 성능입니다.

계수별 그림 3–5는 특정 다른 계수값에 조건부인 비교입니다. 서로 다른 나머지
계수 조합을 섞은 평균을 한 계수의 독립 효과라고 설명하지 않습니다.
그림은 보고서의 실패 조건, 선택 과정, 통계 한계와 함께 해석해야 합니다.
