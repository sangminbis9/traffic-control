# Reward fine search — 완료 결과

**추천은 incumbent `(waiting, max_waiting, switching) = (0.3, 0.5, 0.5)` 유지입니다.**
이번 탐색에서 더 우수하고 안정적인 새 조합은 확인하지 못했습니다. 이는 incumbent의 전역 최적성이나 통계적 우월성을 입증한 결론은 아닙니다.

실험 시작 기준 Git commit은 `b0530d711eb9b5ec291b99284794e6fac8e1dba2`입니다. 2026-10-09에 시작해 2026-10-10(KST)에 완료했습니다. 기존 `../reward_search_2026-10-09/` 결과는 보존했으며, 과거 모델을 신규 모델의 직접 비교 기준으로 섞지 않았습니다.

## 핵심 결과

50k·학습 seed 22의 **제약 통과 후보 상위 5개**는 다음과 같습니다. 순위 기준은 실제 교통 성능이며 학습 reward 값이 아닙니다.

| 후보 | (waiting, max_waiting, switching) | 평균 대기(초) | incumbent 대비 |
|---|---|---:|---:|
| C15 | (0.3, 0.5, 0.6) | 21.603 | −2.61% |
| C13 | (0.3, 0.5, 0.4) | 21.938 | −1.10% |
| C11 | (0.3, 0.4, 0.5) | 21.996 | −0.84% |
| C14 incumbent | (0.3, 0.5, 0.5) | 22.183 | 0.00% |
| C1 | (0.2, 0.4, 0.4) | 22.197 | +0.07% |

평균 대기만 정렬한 전체 상위 5개는 C15·C12·C13·C20·C11입니다. C12와 C20은 최대 대기 제약에서 제외됐습니다. 전체 후보·제외 여부는 보고서와 CSV에 보존했습니다.

상위 3개와 incumbent를 150k로 새로 학습한 결과는 다음과 같습니다. 각 값은 학습 seed 3개 × 동일 validation 교통 seed 20개의 평균입니다.

| 후보 | 평균 대기(초) | 최대 대기 평균(초) | 평균 대기열(대) | 실제 전환(회) | 처리량(대) | 평균 대기 변화 |
|---|---:|---:|---:|---:|---:|---:|
| C14 incumbent | 21.002 | 75.025 | 18.700 | 41.467 | 202.833 | 0.00% |
| C13 | 21.428 | 78.300 | 19.090 | 42.000 | 201.467 | +2.03% |
| C15 | 21.499 | 74.750 | 19.038 | 41.050 | 202.600 | +2.37% |
| C11 | 22.188 | 79.592 | 19.674 | 41.383 | 199.183 | +5.65% |

C15의 평균 대기 차이는 +0.497초, paired 95% bootstrap CI는 [−0.392, +1.301]초입니다. 학습 seed 22·42·62의 차이는 각각 +0.619·+0.939·−0.067초여서 개선 방향이 유지되지 않았습니다. C11과 C13은 최대 대기 평균도 악화됐습니다. 150k Pareto 집합은 C14·C15이며, 사전 안정성 조건까지 충족한 후보는 C14뿐입니다.

## 탐색과 실제 실행량

- 최초 격자: waiting `{0.2, 0.3, 0.4}` × max_waiting `{0.4, 0.5, 0.6}` × switching `{0.4, 0.5, 0.6}` = 27개.
- 경계 확장 3개: `(0.3, 0.5, 0.7)`, `(0.3, 0.5, 0.3)`, `(0.3, 0.3, 0.5)`. 확장된 모든 축의 전체 곱집합을 탐색한 것은 아닙니다.
- Fine: 30개 × 50,000 step, 학습 seed 22, 기존 150,000-step exploration schedule 유지.
- Multiseed: 4개 × 학습 seed `[22, 42, 62]` × 150,000 step, 모두 fresh training.
- 비교용 학습 완료 42개 + 별도 재현 학습 5회 = 총 47회 학습 완료. 실패한 미완료 시도 1회는 제외했습니다.
- Validation: 8001–8020. Holdout: 9001–9030. Robustness: 각 시나리오 10001–10010.
- 비교용 평가 1,110회: fine 600 + multiseed 240 + holdout 90 + robustness 180. 재현 검사 평가 100회는 별도이며 선택·통계 표본에 합치지 않았습니다.

## 최종 holdout과 heavy

Validation에서 새 후보가 통과하지 못해 incumbent를 고정했습니다. 최종 평가에는 incumbent의 세 학습 seed 모델만 있습니다. CSV의 incumbent 대비 0% 및 0승·90무·0패는 **같은 정책의 자기 비교**이며 새 후보의 우월성 증거가 아닙니다.

| 조건 | 평균 대기(초), 95% CI | 최대 대기 평균(초) | 평균 대기열(대) | 처리량(대/300초) | 미완료 차량 평균 | 미출발 차량 평균 |
|---|---|---:|---:|---:|---:|---:|
| Random holdout, 90회 | 16.098 [14.145, 18.215] | 59.778 | 12.151 | 186.744 | 24.189 | 0.000 |
| Heavy, 30회 | 43.340 [41.088, 45.787] | 127.650 | 61.693 | 250.900 | 175.033 | 1.267 |

Uniform·남북 혼잡·동서 혼잡·좌회전 혼잡·heavy·low를 각각 30회 평가했습니다. Heavy의 미완료 차량이 많으므로 heavy 문제가 해결됐다고 결론 내릴 수 없습니다. 과거 기본 reward 대비 heavy 약점이 해소됐는지도 이번 incumbent 단독 최종 평가로 입증하지 않았습니다. Validation과 holdout은 교통 seed가 달라 두 평균의 차이를 정책 개선율로 해석하지 않습니다.

비교용 전체 1,110회에서 collisions와 teleports는 각각 0입니다. SUMO 급제동·적색 신호 긴급 정지 경고는 별도로 관측됐으며, 사고 두 지표가 0이라는 사실이 모든 안전성을 보장하지 않습니다.

## Reward와 고정 조건

`clip(x) = min(1, max(0, x))`일 때 0.5초 SUMO tick의 보상은 다음과 같습니다.

```text
tick_reward = -1.0 * mean_i(clip(queue_i / 10))
              -waiting_weight * clip(global_total_accumulated_waiting / 6000)
              -max_waiting_weight * clip(global_max_accumulated_waiting / 120)
decision_reward = mean(tick_reward during decision interval)
                  -switch_penalty * actual_phase_changes
```

8개 queue 그룹을 평균하며, reward의 waiting 항은 기존 global 값을 사용합니다. 60차원 observation의 그룹별 waiting 구조와 구분해야 합니다. 1초 의사결정, 최소 녹색 3초·최대 녹색 15초·황색 1초·전적색 1초·max-red 60초, 8개 행동, DQN·replay·target·traffic 생성·300초 episode·240초 demand를 유지했습니다. 학습/보상/신호 core 파일과 기존 결과는 SHA-256으로 보존 여부를 확인합니다.

## 실행 복구와 검증

병렬 학습 중 seed 42/C13의 SUMO 시작이 포트 충돌과 일치하는 오류로 중단됐습니다. 실패 시도를 보존하고 해당 모델을 단독으로 처음부터 재학습했습니다. 두 시도의 공통 100k 체크포인트는 정확히 일치했습니다.

당시 함께 실행되던 5개 모델을 별도로 다시 학습해 모델당 7개, 총 35개 체크포인트의 Q·target·optimizer·학습 상태와 수요·평가 원자료를 대조했고 모두 정확히 일치했습니다. 이 재현 모델은 선택에 사용하지 않았습니다. 상세 근거는 `infrastructure_failures/`와 `audit/retraining_integrity/`에 있습니다.

기존 모델/API 및 신규 실험/감사 테스트: **92 passed, 5 skipped**. Skip 5개는 별도 활성화가 필요한 SUMO 통합 테스트입니다. 실제 `check_env`, 300초 SUMO smoke, 60차원/8행동, cache/direct TraCI 일치 검사는 별도 preflight에서 통과했습니다.

CI는 학습 seed와 교통 seed를 각각 재표집하는 paired two-way bootstrap 5,000회입니다. 학습 seed가 3개뿐이고 validation은 후보 선택에 사용됐으므로 통계적 우월성·전역 최적성·실물 모형으로의 전이를 주장할 수 없습니다. 50k 후보 30개 중 150k 검증은 shortlist 4개에만 수행했으므로 모든 후보의 장기 학습 순위를 확인한 것도 아닙니다.

## 실행 명령과 파일

저장소 루트에서 실행한 주요 명령입니다. 이미 완료된 이 폴더는 `--resume`으로 검증된 완료 작업을 재사용합니다.

```powershell
.\.venv\Scripts\python.exe -m model.experiments.fine_reward_search --output model/results/reward_fine_search_2026-10-09 --stage preflight --workers 6
.\.venv\Scripts\python.exe -m model.experiments.fine_reward_search --output model/results/reward_fine_search_2026-10-09 --stage fine --workers 6 --resume
.\.venv\Scripts\python.exe -m model.experiments.fine_reward_search --output model/results/reward_fine_search_2026-10-09 --stage multiseed --workers 6 --resume
# 최초 multiseed 실행의 실패 작업만 fresh 재시도; 완료된 11개는 재사용
.\.venv\Scripts\python.exe -m model.experiments.fine_reward_search --output model/results/reward_fine_search_2026-10-09 --stage multiseed --workers 1 --resume
.\.venv\Scripts\python.exe -m model.experiments.fine_search_integrity --root model/results/reward_fine_search_2026-10-09 --plan-only
.\.venv\Scripts\python.exe -m model.experiments.fine_search_integrity --root model/results/reward_fine_search_2026-10-09 --workers 5
.\.venv\Scripts\python.exe -m model.experiments.fine_reward_search --output model/results/reward_fine_search_2026-10-09 --stage final --workers 1 --resume
.\.venv\Scripts\python.exe -m pytest model/tests api/tests -q
```

- 전체 한국어 보고서: [final-report-ko.txt](final-report-ko.txt)
- 최종 판정: [final_recommendation.json](final_recommendation.json)
- 원자료: [fine_search_results.csv](fine_search_results.csv), [multiseed_results.csv](multiseed_results.csv), [final_holdout_results.csv](final_holdout_results.csv), [robustness_results.csv](robustness_results.csv)
- 통계·paired CI·승무패: 단계별 `*_analysis.json`과 `*_summary.csv`
- Pareto: [pareto_front.csv](pareto_front.csv)
- 고정 프로토콜: `search_config.json`, `protocol_code_lock.json`, `shortlist_lock.json`, `shortlist_evidence_lock.json`, `selection_lock.json`
- 모델: `runs/{fine,multiseed}/seed_*/candidate_*/attempt_*/experiment_*/final.zip`
- 발표 그림: `reports/01_final_five_metrics`부터 `reports/07_scenario_robustness`까지 PNG 7개·PDF 7개. 그림 6의 오차막대는 학습 seed 평균의 표준편차, 나머지는 bootstrap 95% CI입니다.
- 보고서 입력 해시: `reports/report_manifest.json`; 독립 감사와 테스트 로그: `audit/`

기본 reward 설정과 GitHub에는 이 실험이 자동 적용되지 않습니다. 새 가중치로 기본값을 바꾸거나 commit/push하는 작업은 수행하지 않았습니다.
