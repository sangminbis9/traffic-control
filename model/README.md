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
python -m model.test_sumo --seconds 60
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

민우의 검증 설정을 사용하는 기본 학습:

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

Observation은 민우 모델과 동일한 `Box(0, 1, shape=(34,))`입니다.

1. `N_left`, `N_straight`, `S_left`, `S_straight`, `E_left`, `E_straight`, `W_left`, `W_straight` queue 8개
2. 전체 누적 대기시간
3. 현재 차량 중 최대 누적 대기시간
4. 현재 Phase one-hot 4개
5. 현재 green/yellow/all-red 단계 경과시간 1개와 단계 one-hot 3개
6. 전환 목표 Phase one-hot 4개
7. Phase별 미서비스 시간 4개
8. 정지선 50m 이내에서 움직이는 접근 차량 8개

현재 SUMO 구현은 `SUMOTrafficStateProvider`를 사용합니다. 향후 YOLO/Tracking 결과를 같은 `TrafficSnapshot` 구조로 만들면 DQN과 환경의 나머지 부분은 유지할 수 있습니다.

## Action and signal safety

`Discrete(4)`의 의미는 다음과 같습니다.

- `0`: 남북 직진
- `1`: 남북 좌회전
- `2`: 동서 직진
- `3`: 동서 좌회전

DQN은 yellow/all-red를 직접 선택하지 않습니다. 다른 Phase를 선택하면 `SignalController`가 현재 green → yellow → all-red → 새 green으로 전환합니다. `Minimum Green=3s`, `Maximum Green=15s`, `Yellow=1s`, `All Red=1s`, `Max Red=60s`는 `utils/config.py`에서 수정할 수 있습니다. Maximum Green 또는 Max Red에 도달하면 가장 오래 서비스하지 않은 Phase를 우선해 starvation을 제한합니다.

## Reward

현재 보상은 **현재 step의 정규화된 절대 혼잡 비용**과 signal switching 비용에 음수를 부여합니다. README보다 `model/env/reward.py`가 source of truth입니다.

```text
reward = -1.0 * mean(clip(each_group_queue / 10))
       - 0.1 * clip(total_waiting / 6000)
       - 0.1 * clip(max_waiting / 120)
       - 0.2 * actual_switch_count
```

각 항목은 `utils/config.py`의 scale로 정규화합니다. 최대 대기시간 항이 starvation을 억제하고, 전환 penalty와 yellow/all-red 처리 손실이 잦은 전환을 억제합니다. 이전 snapshot은 향후 delta reward 실험을 위해 함수 signature에 남아 있지만 현재 계산에는 사용하지 않습니다.

## Fixed-Time Controller

기본 고정 녹색시간은 `0: 10s`, `1: 4s`, `2: 10s`, `3: 4s`이며 `ProjectConfig.fixed_green_times`에서 변경합니다. Fixed-Time도 동일한 `SignalController`를 통과하므로 yellow/all-red 처리와 phase-change metric이 DQN과 일관됩니다.

## Reproducibility and known limitations

- Train seed와 evaluation seed를 분리할 수 있습니다. 기본 평가 seed는 `2001`부터입니다.
- Route generator의 seed가 같으면 controller 이름이 달라도 동일한 route XML이 생성됩니다.
- 학습은 5-step return, `[128, 128]` MLP와 5~17 decision 동안 유지되는 탐색 action을 사용합니다.
- 웹 학습실은 실시간 시각화를 유지하기 위해 TraCI를 사용하며 MinWoo 원본과 동일한 관측·보상 계약을 따릅니다.
- Windows에서 네트워크 생성, 60초 TraCI 테스트, 단위 테스트, Gymnasium `check_env()`, DQN 학습 및 모델 재로딩을 실제 실행해 검증했습니다.
- 기본 모델은 MinWoo의 seed 22, 150,000-step 모델입니다. 원본 6개 시나리오 × 30개 hold-out seed 평가에서 전체 평균 대기시간이 Fixed-Time보다 32.38% 낮았지만 `heavy`에서는 악화됐으므로 과포화 조건은 별도로 개선해야 합니다.
