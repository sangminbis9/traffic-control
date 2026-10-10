# SUMO + DQN Traffic Signal Control

4방향 3차로 교차로를 SUMO로 시뮬레이션하고, TraCI 상태를 Gymnasium 환경으로 제공한 뒤 Stable-Baselines3 DQN으로 신호를 제어하는 캡스톤디자인용 기준 프로젝트입니다. 카메라/YOLO/Raspberry Pi 코드는 포함하지 않으며, 상태 제공자 인터페이스는 이후 카메라 기반 구현으로 교체할 수 있게 분리했습니다.

## Installation

Windows에서 다음 순서로 설치합니다.

1. Python 3.11 또는 3.12 가상환경을 권장합니다. Stable-Baselines3와 PyTorch의 Windows/Python 호환성을 먼저 확인하세요.
2. [SUMO 공식 배포판](https://sumo.dlr.de/docs/Installing/index.html)을 설치합니다.
3. SUMO의 `bin` 폴더를 PATH에 추가하거나 `SUMO_HOME`을 설정합니다. 설치 방식에 따라 `C:\Program Files\Eclipse SUMO\bin` 또는 `C:\Program Files (x86)\Eclipse\Sumo\bin`일 수 있습니다. 프로젝트는 이 두 표준 경로도 자동 탐색합니다.
4. PowerShell에서 저장소 루트로 이동한 뒤 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r model\requirements.txt
python -m model.sumo.build_network
python -m model.traffic.route_generator --scenario uniform --seed 1
```

`build_network.py`는 `nodes.nod.xml`, `edges.edg.xml`, `connections.con.xml`, `traffic_lights.add.xml`을 이용해 `model/sumo/intersection.net.xml`을 생성합니다. NetEdit 없이 재생성할 수 있습니다.

각 진입로는 운전자 진행방향 기준으로 왼쪽부터 `좌회전 전용 · 직진 · 직진`입니다.
SUMO는 오른쪽부터 차로 번호를 부여하므로 내부 인덱스는 `2=좌회전`,
`1=직진`, `0=직진/우회전 공유`입니다. 우회전은 별도 신호 phase 없이
오른쪽 직진 차로를 공유합니다.

## Run SUMO test

먼저 TraCI 연결, 신호 전환, Queue, Waiting Time을 확인합니다.

```powershell
python -m model.test_sumo --seconds 60 --check-env
```

GUI로 보려면 다음처럼 실행합니다.

```powershell
python -m model.test_sumo --seconds 120 --gui
```

SUMO GUI에서 차량이 나타나지 않으면 `model/sumo/generated`의 route 파일과 `model/sumo/intersection.net.xml`의 생성 여부를 먼저 확인하세요. GUI 없이 직접 설정 파일을 열려면 다음을 사용할 수 있습니다.

```powershell
sumo-gui -c model\sumo\simulation.sumocfg --route-files model\sumo\routes.rou.xml
```

## Train

짧은 smoke 학습:

```powershell
python -m model.train --timesteps 1000 --episode-seconds 120 --check-env
```

기존 DQN 학습 설정으로 새 60차원/8-action 모델 학습:

```powershell
python -m model.train --timesteps 150000 --scenario random
```

학습 모델은 `model/results/dqn_intersection.zip`에 저장됩니다. 학습 시에는 기본적으로 `sumo` headless 바이너리를 사용하며, 디버깅할 때만 `--gui`를 추가합니다.

## Evaluate

학습 모델이 있는 상태에서, 같은 seed로 Fixed-Time과 DQN을 각각 실행합니다.

```powershell
python -m model.evaluate --model model\results\dqn_intersection.zip --episodes 30 --seed-start 2001
```

평가 결과는 `model/results/evaluation_metrics.csv`와 다음 PNG에 저장됩니다.

- `average_waiting_time.png`
- `average_queue.png`
- `throughput.png`
- `phase_changes.png`
- `episode_reward.png`

각 평가 episode는 controller별로 같은 seed와 같은 수요 생성 규칙을 사용합니다. `routes_DQN_2001_random.rou.xml`과 `routes_fixed-time_2001_random.rou.xml`의 내용은 seed가 같으면 동일합니다.

### Rule-based controller and fair comparison

`controller/rule_based_controller.py`의 `RuleBasedController`는 DQN과 **동일한
clipped 60차원 observation**만 받는 결정론적 감응 제어기입니다. SUMO 내부 차량
위치·미래 route·정규화 이전 값·DQN Q값을 읽지 않습니다. Reward, DQN 학습,
8개 Action 및 기존 신호 안전 제약은 변경하지 않았습니다.

- 이동 그룹 우선순위는 `queue + w_W * total_waiting + w_M * max_waiting + w_A * approaching`입니다.
  여기서 모든 입력은 observation의 정규화 값이며, **규칙 계수는 Reward 가중치와 별개**입니다.
- 각 Phase가 서비스하는 두 그룹을 합산하고, 수요가 관측된 Phase에는
  `w_R * normalized_red_age²`를 더합니다. Red-age는 이동 그룹이 아닌 Phase별 값입니다.
- 용량 추정 변형은 직진 그룹 2차로·좌회전 그룹 1차로, 추정 차두시간과
  녹색시간을 이용해 해당 구간에 처리 가능한 비율을 적용합니다. 다른 Phase는
  황색·전체 적색 시간을 제외하고, 현재 Phase는 최대 녹색까지 남은 시간을 고려합니다.
  차두시간 2초와 8초 관측 창은 **검증된 물리 모델이 아닌 규칙의 근사 설정**입니다.
- 경쟁 Phase가 현재 점수의 `(1 + relative_margin)`배와 `switch_margin`을 넘을 때
  전환합니다. 수요가 없는 현재 Phase에서는 기다리지 않습니다. Waiting만 관측되는
  움직이거나 먼 차량도 빈 그룹으로 취급하지 않습니다.
- 최소 녹색 및 전환 중에는 유지하며, 최대 녹색 자동 전환 직전에는 수요에 맞는
  다음 Phase를 요청합니다. 실제 적용과 max-red override는 공통 `SignalController`가 수행합니다.

규칙 단독 실행에는 모델 파일이 필요하지 않습니다.

```powershell
python -m model.evaluate --controllers rule --episodes 5 --scenario random --output-dir .tmp/rule_preview
```

기존 `model.evaluate`의 기본 Fixed-Time/DQN 동작과 웹/API의 과거 평가 경로는 유지합니다.
기존 Fixed-Time은 `force=True` 경로를 사용하므로, **공정한 규칙/DQN/고정 신호 비교에는
아래 전용 실험 명령을 사용**합니다. 전용 고정 신호는 동일한 주기 요청을 일반
`env.step()`으로 전달하며 다른 제어기와 똑같은 안전 제약을 거칩니다.

```powershell
python -m model.experiments.rule_based_comparison --output model/results/rule_based_comparison_2026-10-10 --workers 4
```

이 실험은 다음 절차를 고정합니다.

1. 대기 우선순위, 접근 차량 가중치, 전환 억제 및 용량 추정 유무를 조합한 37개
   규칙 후보를 20개 조정용 교통에서 비교합니다. 대기열만 사용하는 단순 후보도 포함합니다.
   **평균 episode reward 최대**를 선정 기준으로 사용하며 동점이면 평균 대기시간,
   후보 이름 순으로 결정합니다. 조정 비용은 740 episode / 222,000 결정 step입니다.
2. 선택을 저장한 뒤 별도 60개 교통에서 규칙 1개, 고정 신호 1개, 기존 DQN 3개를
   평가합니다. DQN은 Reward `1 / 0.3 / 0.5 / 0.5`, 학습 seed 22/42/62,
   각 150,000 step의 고정 checkpoint입니다. 재학습하거나 모델을 덮어쓰지 않습니다.
3. 조정은 random 8개와 나머지 6개 시나리오 각 2개, 평가는 random 30개와 나머지
   각 5개입니다. 두 집합의 seed는 분리하며 episode 300초, 수요 발생 240초입니다.
   random 비중은 조정 40%, 평가 50%로 다르므로 시나리오별 결과도 함께 기록합니다.
4. 코드·모델·설정·route SHA를 고정하고, 모든 정책이 같은 route와 SUMO seed를
   사용했는지 확인합니다. 대기·대기열·통과량·전환 외에 미완료 차량, 진입 대기 차량,
   강제 전환과 충돌/teleport를 기록하여 300초 종료 시 남은 수요를 숨기지 않습니다.

`selection.json`은 선택된 규칙, `summary.csv`는 시나리오별 결과,
`heldout_episodes.csv`는 평가 원자료, `paired_differences.csv`는 paired 차이와
bootstrap 95% 구간, `verification.json`은 실험 무결성 결과입니다.
`dqn_mean`은 세 모델을 각각 실행한 결과의 평균이며 **앙상블 제어기가 아닙니다**.
신뢰구간은 이 세 checkpoint를 고정했을 때의 교통 변동을 나타내며, 새로운 학습
seed에 대한 불확실성까지 포함하지 않습니다. 학습 seed 간 편차는 별도로 제공합니다.
평가 결과를 확인한 뒤 같은 평가 집합으로 규칙을 다시 조정하지 않습니다.

선택된 규칙을 재사용하려면 다음과 같이 실행합니다.

```powershell
python -m model.evaluate --controllers rule --rule-policy model/results/rule_based_comparison_2026-10-10/selection.json --episodes 5 --output-dir .tmp/rule_selected
```

별도 설정을 지정하지 않은 규칙의 기본값은 시작점이며, 비교에서 선정된 설정과
다를 수 있습니다. 입력 구조와 규칙 계수를 유지하면 같은 정책을 다른 state provider에도
연결할 수 있으나, 카메라 및 실제 모형에서의 성능은 별도 검증이 필요합니다.

2026-10-10 완료 결과는 [비교 보고서](results/rule_based_comparison_2026-10-10/comparison-ko.txt),
[그래프](results/rule_based_comparison_2026-10-10/comparison.png),
[검증 기록](results/rule_based_comparison_2026-10-10/checks.json)에 저장했습니다.
선정된 규칙은 `balanced_h8_a0.75_m0.25`이며, 아래는 사전에 고정한 **전체 60개 교통 사례**의 평균입니다.

| 제어 방식 | 평균 대기시간(초) ↓ | 회차별 최대 대기시간의 평균(초) ↓ | 통과 차량 수 ↑ | 신호 전환 횟수 ↓ |
| --- | ---: | ---: | ---: | ---: |
| 기존 고정 순환 + 공통 안전 제약 | 23.81 | 95.12 | 185.27 | 40.00 |
| 조정한 규칙 기반 | 19.88 | 81.75 | 183.52 | 47.07 |
| DQN 3개 모델의 평가 평균 | 19.51 | 68.58 | 187.47 | 41.32 |

전체 평균 대기시간 차이(rule − DQN)는 0.37초이며 bootstrap 구간이 0을 포함하므로,
평균 대기시간의 확실한 우위를 주장하지 않습니다. 전체 구간은 명시적 시나리오 사이의
seed 재사용에 따른 의존성을 보존하지 않아 참고용으로 해석합니다.
random 30개 하위 집합에서는 규칙 21.93초, DQN 20.19초였지만,
저수요 5개에서는 규칙 2.27초, DQN 8.82초로 규칙이 더 좋았습니다.
heavy 5개에서는 기존 고정 순환의 평균 대기시간과 통과량이 두 방식보다 좋았으며,
세 방식 모두 종료 시 상당한 미완료 차량이 남았습니다.
고정 순환은 이번 실험에서 최적화하지 않은 기존 시간표이므로,
이 비교만으로 강화학습의 필수성이나 모든 수요에서의 우월성을 주장할 수 없습니다.

## Project structure

```text
model/
  sumo/
  *.nod.xml, *.edg.xml, *.con.xml, traffic_lights.add.xml  # 네트워크 원본
  build_network.py                                         # netconvert 실행
  intersection.net.xml                                     # 생성 네트워크
  simulation.sumocfg
  generated/                                                # episode route 파일
  env/
  intersection_env.py                                      # Gymnasium API
  state_provider.py                                        # SUMO/카메라 교체 경계
  reward.py
  controller/
  signal_controller.py                                     # green/yellow/all-red 안전 전환
  fixed_controller.py
  traffic/route_generator.py                               # seed 재현 수요 생성
  utils/config.py, metrics.py
  train.py, evaluate.py, test_sumo.py
  results/
```

## State

Observation은 `Box(0, 1, shape=(60,), dtype=float32)`입니다. 모든 이동 그룹의 순서는
`N_left`, `N_straight`, `S_left`, `S_straight`, `E_left`, `E_straight`, `W_left`, `W_straight`입니다.

| 인덱스 (Python slice) | Feature | 정규화 |
|---|---|---|
| `0:8` | 그룹별 queue 8개 | `/10` |
| `8:16` | 그룹별 total accumulated waiting 8개 | `/6000` |
| `16:24` | 그룹별 max accumulated waiting 8개 | `/120` |
| `24:32` | 현재 Phase one-hot 8개 | 0 또는 1 |
| `32:33` | 현재 단계 경과시간 | 단계별 max green / yellow / all-red 시간으로 나눔 |
| `33:36` | green/yellow/all-red one-hot 3개 | 0 또는 1 |
| `36:44` | 전환 목표 Phase one-hot 8개 | green일 때 모두 0 |
| `44:52` | Phase별 red-age 8개 | `/max_red` |
| `52:60` | 정지선 50m 이내에서 움직이는 approaching 8개 | `/10` |

정규화 값은 `[0, 1]`로 clip합니다. Waiting은 그룹의 **현재 진입 차로에 있는 모든 차량**의
`getAccumulatedWaitingTime()`을 합산하거나 최댓값을 취합니다. 정지 여부나 정지선까지의
거리에 제한을 두지 않습니다. 좌회전 그룹은 차로 2, 직진 그룹은 차로 0·1(우회전 공유)을
사용하며, 빈 그룹은 0입니다. 전역 total/max waiting은 Observation에서 제외했지만
기존 reward와 지표 계산에는 그대로 사용합니다.

현재 SUMO 구현은 `SUMOTrafficStateProvider`를 사용합니다. 향후 YOLO/Tracking 결과를 같은 `TrafficSnapshot` 구조로 만들면 DQN과 환경의 나머지 부분은 유지할 수 있습니다.

## Action and signal safety

`Discrete(8)`의 의미는 다음과 같습니다.

- `0`: 남북 직진
- `1`: 남북 좌회전
- `2`: 북쪽 직진+좌회전
- `3`: 남쪽 직진+좌회전
- `4`: 동서 직진
- `5`: 동서 좌회전
- `6`: 동쪽 직진+좌회전
- `7`: 서쪽 직진+좌회전

직진+좌회전 Phase는 해당 한 방향만 녹색이며 나머지 세 방향은 적색입니다.
우회전 공유 차로는 기존처럼 해당 방향 직진 신호와 함께 열립니다.

DQN은 yellow/all-red를 직접 선택하지 않습니다. 다른 Phase를 선택하면 `SignalController`가 현재 green → yellow → all-red → 새 green으로 전환합니다. `Minimum Green=3s`, `Maximum Green=15s`, `Yellow=1s`, `All Red=1s`, `Max Red=60s`는 `utils/config.py`에서 수정할 수 있습니다. Maximum Green 또는 Max Red에 도달하면 가장 오래 서비스하지 않은 Phase를 우선해 starvation을 제한합니다.

## Reward

현재 보상은 **정규화된 절대 혼잡 비용**과 실제 신호 전환 비용에 음수를 부여합니다.
가중치와 정규화의 기준은 `utils/config.py`의 `RewardConfig`, 계산은
`env/reward.py`와 `env/intersection_env.py`에 정의돼 있습니다.

```text
tick_reward = -1.0 * mean(clip(each_group_queue / 10))
              -0.3 * clip(total_accumulated_waiting / 6000)
              -0.5 * clip(max_accumulated_waiting / 120)
reward = sum(tick_reward * step_length / decision_interval)
         -0.5 * actual_switch_count
```

`clip`은 `[0, 1]` 범위로 제한합니다. `sum`은 한 의사결정 구간의 SUMO tick에
대해 계산합니다. 현재 0.5초 tick·1초 의사결정에서는 두 tick 보상의 평균입니다.
전환 비용은 강제 전환까지 포함한 실제 Phase 변경 횟수에 부과합니다.
Queue는 8개 이동 그룹의 정규화된 정지 차량 수를 평균합니다.
대기 항의 α는 **평균 대기시간이 아니라 현재 진입 차량의 accumulated waiting 합계**에
곱하는 가중치입니다. 정지하지 않은 차량도 포함합니다. 그룹별 waiting Observation과
달리 reward는 기존 전역 total/max waiting을 사용합니다.

### Fixed baseline decision

2026-10-10부터 **Queue=1.0, α=0.3, β=0.5, 전환 계수=0.5**를 고정 기준으로
채택했습니다. 전환 계수는 DQN 할인율 `gamma=0.95`와 별개입니다.
`RewardConfig`를 사용하는 CLI·웹/API의 새 학습과 환경이 이 설정을 공유합니다.
보상식의 구조, 정규화, DQN 알고리즘과 신호 시간은 유지했습니다.

선정 근거는 짧은 학습에서의 최고 점수뿐 아니라 **더 긴 학습과 여러 학습 seed에서
기준 조합을 교체할 만큼 일관된 개선이 확인되는지**입니다.

- 50k 정밀 탐색은 학습 seed 22, 평가 교통 seed 8001–8020을 사용했습니다.
  `(0.3, 0.5, 0.6)`의 평균 대기는 21.603초로 기준 조합의 22.183초보다 낮았습니다.
- 동일 평가 교통에서 150k·학습 seed 22/42/62로 검증한 결과, 기준 조합의
  평균 대기는 **21.002초**, 전환 계수 0.6 조합은 **21.499초**였습니다.
  0.6 조합의 평균 대기 개선 방향도 세 학습 seed에 걸쳐 유지되지 않았습니다.
  0.6은 최대 대기·전환 횟수에서 소폭 유리했지만, 기준을 교체할 안정적인 개선은
  확인하지 못했습니다.
- 별도의 50k·교통 seed 2001–2030 비교에서도 0.6 조합은 19.31초,
  기준 조합은 19.45초로 약 0.13초 차이였습니다. 이 단일 학습 seed 결과만으로
  고정 기준을 바꾸지는 않습니다. 두 평가 교통 집합의 평균을 직접 비교하지 않습니다.

이 결정은 **현재 증거에 근거한 프로젝트 기준의 채택**입니다. 전역 최적값 또는
통계적 우월성의 입증은 아닙니다. 150k에서 `0.6 조합 − 기준 조합`의 평균 대기
차이는 +0.497초, paired 95% CI는 `[-0.392, +1.301]`초로 0을 포함했습니다.
학습 seed가 3개뿐이라는 한계도 있습니다.

근거 자료:

- [정밀 탐색·150k 다중 seed 결과](results/reward_fine_search_2026-10-09/README.md)
- [150k 평가 원자료](results/reward_fine_search_2026-10-09/multiseed_results.csv)
- [동일 조건의 7개 조합 비교표](results/reward_paired_comparison_2026-10-10/comparison-ko.txt)
- [7개 조합 독립 검증](results/reward_paired_comparison_2026-10-10/independent_audit.json)

이후 실험은 가중치와 정규화·클리핑을 함께 고정하고, 학습량·학습 seed·교통 구성의
효과를 구분해 비교합니다. 성능은 같은 평가 교통의 평균/최대 대기, 대기열, 통과량,
실제 전환 횟수로 판단합니다. 제어 목표나 입력 단위·교통 규모가 바뀌거나 특정 방향의
장기 대기처럼 목표와 행동의 불일치가 반복될 때 보상 기준을 다시 검토합니다.

### Checkpoint resume and historical evidence

재개할 checkpoint는 기록된 보상 가중치 4개와 정규화 scale 3개가 현재 설정과
일치해야 합니다. 새 checkpoint는 모델 SHA-256과 실제 보상 설정을
`*.metadata.json`에 함께 저장합니다. 기존 `experiment_config.json` 및
`training_metadata.json`의 보상 기록도 지원합니다. 설정이 다르거나 기록이 없으면
학습을 중단하고 새 학습을 요구하여, 이전 보상의 Q값·replay reward를 섞지 않습니다.
동일 보상이라도 replay buffer가 없는 모델은 학습 상태 전체를 충실히 이어가는
체크포인트로 볼 수 없습니다. CLI의 `dqn_intersection.zip`에는 보상 메타데이터를
함께 내보내지만 replay는 포함하지 않습니다. 저장된 replay까지 이어 학습하려면
세션 폴더의 `final.zip`과 같은 위치의 `final.replay.pkl`을 사용합니다.
Git에 보관한 실험 모델은 평가용 최종 모델이며,
대용량 replay buffer와 중간 checkpoint는 로컬에만 보존했습니다.

실험 코드와 결과는 기본값 변경 전 커밋 `4b52089d662a6dd5e019040b6fd9a84f51053097`에
먼저 보관했습니다. 이 커밋의 핵심 환경·학습 코드는 원래 실험 기준 `b0530d71`과
같습니다. 과거 결과의 설정·해시·당시 기본값을 새 기준으로 덮어쓰지 않았습니다.
현재 코드에서 해당 실행을 `--resume`하면 보존된 핵심 코드 해시와 달라 거부되는 것이
정상입니다. 역사적 실험의 재실행은 보관 커밋과 기록된 runtime을 기준으로 하고,
새 실험은 별도 결과 폴더를 사용합니다. 원자료에 기록된 로컬 절대 경로는 다른 PC에서
해당 checkout 위치에 맞게 해석해야 합니다.

## Fixed-Time Controller

Fixed-Time은 `0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 0` 순서로 순환합니다.
기본 고정 녹색시간은 `0·4: 10s`, 나머지는 `4s`이며 `ProjectConfig.fixed_green_times`에서 변경합니다. Fixed-Time도 동일한 `SignalController`를 통과하므로 yellow/all-red 처리와 phase-change metric이 DQN과 일관됩니다.

## Reproducibility and known limitations

- Train seed와 evaluation seed를 분리할 수 있습니다. 기본 평가 seed는 `2001`부터입니다.
- Route generator의 seed가 같으면 controller 이름이 달라도 동일한 route XML이 생성됩니다.
- 학습은 5-step return, `[128, 128]` MLP와 5~17 decision 동안 유지되는 탐색 action을 사용합니다.
- 웹 학습실은 실시간 시각화를 위해 TraCI를 사용하며 60차원/8-action 관측·행동 계약을 따릅니다. 기존 보상식과 DQN 알고리즘·학습 설정은 유지합니다.
- Windows에서 네트워크 생성, 60초 TraCI 테스트, 단위 테스트, Gymnasium `check_env()`, DQN 학습 및 모델 재로딩을 실제 실행해 검증했습니다.
- 보존된 기본 모델은 MinWoo의 seed 22, 150,000-step 모델이며 34차원/4-action입니다. 현재 환경과 호환되지 않으므로 새 학습이 필요하며 기존 checkpoint를 그대로 resume할 수 없습니다. 과거 모델의 성능 수치를 새 8-phase 환경의 성능으로 해석하면 안 됩니다.
