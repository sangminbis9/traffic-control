# 실행 검증 보고서

> 이 문서는 초기 6,000스텝 검증 기록입니다. 이후 다중 시드·학습량 실험과 선정 모델의 최종 결과는 [EXPERIMENTS.md](EXPERIMENTS.md)를 확인하세요.

검증일: 2026-09-18, Windows, Python 3.12.14, SUMO/TraCI 1.27.1, Gymnasium 1.3.0, Stable-Baselines3 2.9.0, PyTorch 2.14.0 CPU. 전체 버전은 `requirements-lock.txt`에 있습니다.

## 요청한 첫 성공 기준 9개

| 항목 | 실제 확인 결과 |
|---|---|
| 4방향 × 3차로 사거리 | netconvert 생성, 16개 신호 연결·차로·회전 매핑 검사 통과 |
| Python Phase 변경 | 0/1/2/3 및 황색·전적색 상태를 실제 TraCI로 확인 |
| 방향별 Queue 출력 | N/S/E/W 좌회전·직진+우회전 출력 확인 |
| Waiting Time | 연속/누적 대기시간 수집, tripinfo 대조 확인 |
| Fixed-Time | [10,4,10,4]초 + 전환으로 36초 주기 검사 통과 |
| Gymnasium `check_env` | Stable-Baselines3 검사 통과 |
| DQN 수천 Step | 6,000 steps, 20개 완전 학습 episode, 1,375 gradient updates 완료 |
| 모델 저장·재로딩 | dqn_smoke.zip 저장, 재로딩 deterministic action 일치 |
| 동일 교통량 비교 | 6시나리오 × 5시드 × 2제어기 = 60 episodes, 30쌍 route hash 일치 |

## 테스트와 GUI

`python -m pytest -q`: **19 passed**. 최소 녹색 제한, 동일 액션 유지, 전환 순서와 정확한 시간, 최대 녹색 강제 전환, 반복 편향 액션에서 공정성, 0.5초 전적색, 고정 주기, 7종 시나리오 재현, reward 범위, 무수요/설정 검증, 실제 SUMO 환경 검사, 모든 관측/보상의 반복 실행 일치, 독립 연결, 오류 시 프로세스 정리를 검증했습니다.

`python -m pip check`: **No broken requirements found**.

GUI 120초: 평균 대기 8.58초, 평균 Queue 4.15대, 도착 32대, 전환 13회, 충돌 0, 텔레포트 0. 같은 120초 headless 테스트와 교통 지표가 일치했습니다. [실제 GUI 캡처](results/gui_capture/sumo_gui.png)를 확인했습니다.

평가 balanced/2001/Fixed의 `tripinfo.xml` 141대 평균 대기시간 **8.304964539초**가 CSV 값과 일치했습니다. 완료 차량과 아직 남은 차량 모두 포함합니다.

최종 프로세스 조회에서 실행 중인 `sumo.exe` / `sumo-gui.exe`가 없음을 확인했습니다. 네트워크 재생성 전후 canonical hash 일치도 검사했습니다. 최종 코드로 balanced/2001의 두 제어기를 다시 실행한 결과가 기존 평가와 같았으며 `results/final_verification/`에 보존했습니다. Python compileall 검사도 통과했습니다.

## 학습 및 비교 결과

학습 명령:

```powershell
python train.py --steps 6000 --learning-starts 500 --model results/models/dqn_smoke --output results/training/smoke_6000
```

약 57초 학습, 약 104 decision steps/s. seed pool 1~1000, 매 episode 300초, 수요 240초, 무작위 방향별 수요 및 회전 비율.

평가 명령:

```powershell
python evaluate.py --model results/models/dqn_smoke.zip --episodes 5 --output results/evaluation_smoke
```

평가 seed 2001~2005, balanced/ns_heavy/ew_heavy/left_heavy/heavy/low 각 5회, 제어기별 30회. 아래 값은 전체 episode의 **평균 ± 표본 표준편차**입니다.

| 지표 | Fixed-Time | DQN 6,000 |
|---|---:|---:|
| 차량 평균 누적 대기시간 (초) | 19.10 ± 8.65 | 27.18 ± 8.53 |
| 시간 평균 Queue (대) | 15.79 ± 14.91 | 21.22 ± 17.27 |
| Throughput (대/300초) | 169.87 ± 81.72 | 165.90 ± 80.48 |
| Phase 변경 (회) | 33.00 ± 0.00 | 21.60 ± 2.28 |
| Episode Reward | -80.03 ± 47.12 | -95.77 ± 51.11 |

**현재 DQN이 고정 신호보다 좋다고 볼 수 없습니다.** 전환 횟수는 줄었지만 평균 대기와 Queue가 증가했습니다. 짧은 학습의 실행 가능성을 검증한 결과이며 최적화 성공이나 수렴을 의미하지 않습니다. 모든 평가와 학습에서 탐지된 충돌·텔레포트는 0건입니다.

CSV와 6개 PNG는 `results/evaluation_smoke/`에 있습니다. 평가에 사용한 각 route XML과 tripinfo도 보존했습니다.

## Reward scale 확인

episode당 각 항의 평균 기여량:

| 항 | Fixed | DQN |
|---|---:|---:|
| Queue | -14.81 | -19.89 |
| Total waiting | -8.50 | -11.94 |
| Maximum waiting | -50.13 | -59.62 |
| Switching | -6.60 | -4.32 |

원시 초 단위가 무제한으로 reward를 압도하지 않도록 각 항에 정규화·clip을 적용했습니다. 실제로 최대 대기 항이 전체 절댓값의 약 62%를 차지해 가장 큽니다. Starvation 억제 의도가 반영되지만 장기 학습에서는 별도 검증 seed로 weight/scale을 비교해야 합니다. 이미 확인한 평가 seed에 맞춰 hyperparameter를 조정해 최종 성능을 주장하지 마세요.

## 발견해 수정한 문제

1. 기본 `python`은 실행 불가능한 Windows Store 별칭이었습니다. 사용 가능한 Python 3.12로 로컬 `.venv`를 만들었습니다.
2. netconvert가 linkIndex를 N/E/S/W 순으로 생성했습니다. 그 순서에 맞춰 신호 문자열을 수정하고, 결과 네트워크를 검사하도록 했습니다. 관측값의 방향 순서는 요청대로 N/S/E/W입니다.
3. Windows SUMO가 한글 절대 `SUMO_HOME`에서 `Quitting (on unknown error)`로 종료됐습니다. ASCII 상대 경로와 Windows locale 처리로 해결했습니다.
4. GUI PNG 저장에 한글 절대 경로를 전달하면 멈추는 현상이 있었습니다. 상대 경로로 저장한 후 GUI 실행이 정상 종료됨을 확인했습니다.
5. Matplotlib의 기본 캐시 위치에 쓰기 권한이 없어 임시 캐시 경로를 지정했습니다.
6. netconvert 생성 시각 주석 때문에 모델의 네트워크 해시가 달라질 수 있어, 실제 XML 내용만 canonical hash로 검사하도록 했습니다.

## 남은 문제와 다음 실험

- **성능 개선 미달:** 100,000~300,000스텝 학습과 여러 학습 seed 실험이 필요합니다. 현재 DQN은 평균 21.6회 전환 중 15.1회가 max-green/max-red 규칙의 개입이라, 상황별 유지시간을 충분히 학습했다고 보기 어렵습니다.
- **급제동 경고:** 평가 로그에 급제동 5건(4개 episode), 학습 로그에 1건이 있습니다. 실행 설정은 요청한 황색 1초와 접근속도 11.11m/s였으며, 짧은 황색 시간과 차량 감속 특성의 관계를 우선 점검해야 합니다. 개별 경고의 원인을 분리하는 실험까지 수행하지는 않았습니다. 충돌은 없었지만 물리적 안전성이 입증된 설정은 아닙니다. 후속 실험에서 황색 시간·접근속도·전적색을 함께 검토해야 합니다.
- **대기 상한 없음:** max-red는 Phase 서비스 지연 억제 규칙입니다. 초과 수요에서는 차량 누적 대기가 길어지며, 이번 평가에서 최대 216초까지 관측됐습니다. 모든 차량의 60초 내 통과를 보장하지 않습니다.
- **Episode 끝의 미처리 수요:** 아직 주행·대기 중인 차량과 삽입하지 못한 차량이 남습니다. 단순 평균 대기만 보지 말고 unfinished/not_inserted/throughput을 같이 비교해야 합니다. 평균 미삽입 차량은 Fixed 1.7대, DQN 0.6대였습니다.
- **정규화 포화:** 최대 대기 120초 이상은 관측값/reward에서 같은 값으로 clip됩니다. 더 긴 혼잡 실험은 scale 조정 및 재학습이 필요합니다.
- **모형 전이 미검증:** CameraTrafficStateProvider, GPIO, YOLO, Pi는 의도적으로 구현하지 않았습니다. 교체 인터페이스만 준비했습니다.

100,000/300,000스텝 장기 학습과 기본 2001~2030 전체 평가는 실행 명령을 준비했으며, 이번에는 6,000스텝과 2001~2005의 초기 검증까지 수행했습니다.
