# SUMO + DQN Traffic Signal Control

4방향 3차로 교차로를 SUMO로 시뮬레이션하고, TraCI 상태를 Gymnasium 환경으로 제공한 뒤 Stable-Baselines3 DQN으로 신호를 제어하는 캡스톤디자인용 기준 프로젝트입니다. 카메라/YOLO/Raspberry Pi 코드는 포함하지 않으며, 상태 제공자 인터페이스는 이후 카메라 기반 구현으로 교체할 수 있게 분리했습니다.

## Installation

Windows에서 다음 순서로 설치합니다.

1. Python 3.11 또는 3.12 가상환경을 권장합니다. Stable-Baselines3와 PyTorch의 Windows/Python 호환성을 먼저 확인하세요.
2. [SUMO 공식 배포판](https://sumo.dlr.de/docs/Installing/index.html)을 설치합니다.
3. SUMO의 `bin` 폴더를 PATH에 추가하거나 `SUMO_HOME`을 설정합니다. 설치 방식에 따라 `C:\Program Files\Eclipse SUMO\bin` 또는 `C:\Program Files (x86)\Eclipse\Sumo\bin`일 수 있습니다. 프로젝트는 이 두 표준 경로도 자동 탐색합니다.
4. PowerShell에서 프로젝트 루트로 이동한 뒤 설치합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python sumo\build_network.py
python -m traffic.route_generator --output sumo\routes.rou.xml --scenario uniform --seed 1
```

`build_network.py`는 `nodes.nod.xml`, `edges.edg.xml`, `connections.con.xml`, `traffic_lights.add.xml`을 이용해 `sumo/intersection.net.xml`을 생성합니다. NetEdit 없이 재생성할 수 있습니다.

## Run SUMO test

먼저 TraCI 연결, 신호 전환, Queue, Waiting Time을 확인합니다.

```powershell
python test_sumo.py --seconds 60
```

GUI로 보려면 다음처럼 실행합니다.

```powershell
python test_sumo.py --seconds 120 --gui
```

SUMO GUI에서 차량이 나타나지 않으면 `sumo/generated`의 route 파일과 `sumo/intersection.net.xml`의 생성 여부를 먼저 확인하세요. GUI 없이 직접 설정 파일을 열려면 다음을 사용할 수 있습니다.

```powershell
sumo-gui -c sumo\simulation.sumo.cfg --route-files sumo\routes.rou.xml
```

## Train

짧은 smoke 학습:

```powershell
python train.py --timesteps 1000 --episode-seconds 120 --check-env
```

일반적인 시작점:

```powershell
python train.py --timesteps 20000 --scenario random
python train.py --timesteps 100000 --scenario random
```

학습 모델은 `results/dqn_intersection.zip`에 저장됩니다. 학습 시에는 기본적으로 `sumo` headless 바이너리를 사용하며, 디버깅할 때만 `--gui`를 추가합니다.

## Evaluate

학습 모델이 있는 상태에서, 같은 seed로 Fixed-Time과 DQN을 각각 실행합니다.

```powershell
python evaluate.py --model results\dqn_intersection.zip --episodes 30 --seed-start 2001
```

평가 결과는 `results/evaluation_metrics.csv`와 다음 PNG에 저장됩니다.

- `average_waiting_time.png`
- `average_queue.png`
- `throughput.png`
- `phase_changes.png`
- `episode_reward.png`

각 평가 episode는 controller별로 같은 seed와 같은 수요 생성 규칙을 사용합니다. `routes_DQN_2001_random.rou.xml`과 `routes_fixed-time_2001_random.rou.xml`의 내용은 seed가 같으면 동일합니다.

## Project structure

```text
sumo/
  *.nod.xml, *.edg.xml, *.con.xml, traffic_lights.add.xml  # 네트워크 원본
  build_network.py                                         # netconvert 실행
  intersection.net.xml                                     # 생성 네트워크
  simulation.sumo.cfg
  generated/                                                # episode route 파일
env/
  intersection_env.py                                      # Gymnasium API
  state_provider.py                                        # SUMO/카메라 교체 경계
  reward.py
controller/
  signal_controller.py                                     # green/yellow/all-red 안전 전환
  fixed_controller.py
traffic/route_generator.py                                 # seed 재현 수요 생성
utils/config.py, metrics.py
train.py, evaluate.py, test_sumo.py
results/
```

## State

Observation은 `Box(0, 1, shape=(12,))`입니다.

1. `N_left`, `N_straight`, `S_left`, `S_straight`, `E_left`, `E_straight`, `W_left`, `W_straight` queue 8개
2. 전체 누적 대기시간
3. 현재 차량 중 최대 누적 대기시간
4. 현재 Phase를 0~1로 정규화한 값
5. 현재 Phase 경과시간을 `Maximum Green` 기준으로 정규화한 값

현재 SUMO 구현은 `SUMOTrafficStateProvider`를 사용합니다. 향후 YOLO/Tracking 결과를 같은 `TrafficSnapshot` 구조로 만들면 DQN과 환경의 나머지 부분은 유지할 수 있습니다.

## Action and signal safety

`Discrete(4)`의 의미는 다음과 같습니다.

- `0`: 남북 직진
- `1`: 남북 좌회전
- `2`: 동서 직진
- `3`: 동서 좌회전

DQN은 yellow/all-red를 직접 선택하지 않습니다. 다른 Phase를 선택하면 `SignalController`가 현재 green → yellow → all-red → 새 green으로 전환합니다. `Minimum Green=3s`, `Maximum Green=15s`, `Yellow=1s`, `All Red=1s`는 `utils/config.py`에서 수정할 수 있습니다. Minimum Green 전에 들어온 변경 요청은 무시되고, Maximum Green에서 현재 Phase를 계속 선택하면 다음 Phase로 강제 전환됩니다.

## Reward

현재 보상은 이전 step 대비 queue, total waiting, maximum waiting이 줄어든 정도에서 signal switching penalty를 빼는 형태입니다.

```text
reward = 1.0 * queue_improvement
       + 0.3 * waiting_improvement
       + 0.5 * max_waiting_improvement
       - 0.2 * switched
```

각 항목은 `utils/config.py`의 scale로 먼저 정규화합니다. 최대 대기시간 항이 starvation을 억제하고, 전환 penalty와 yellow/all-red 손실이 잦은 전환을 억제합니다. 실제 성능에 따라 weight와 scale을 조정해야 합니다.

## Fixed-Time Controller

기본 고정 녹색시간은 `0: 10s`, `1: 4s`, `2: 10s`, `3: 4s`이며 `ProjectConfig.fixed_green_times`에서 변경합니다. Fixed-Time도 동일한 `SignalController`를 통과하므로 yellow/all-red 처리와 phase-change metric이 DQN과 일관됩니다.

## Reproducibility and known limitations

- Train seed와 evaluation seed를 분리할 수 있습니다. 기본 평가 seed는 `2001`부터입니다.
- Route generator의 seed가 같으면 controller 이름이 달라도 동일한 route XML이 생성됩니다.
- 현재 환경은 한 Python 프로세스에서 SUMO를 episode마다 재시작합니다. 장시간 학습에서는 `libsumo` 전환을 추가하면 속도를 개선할 수 있습니다.
- Windows에서 네트워크 생성, 60초 TraCI 테스트, 단위 테스트, Gymnasium `check_env()`, DQN 학습 및 모델 재로딩을 실제 실행해 검증했습니다.
- `results/evaluation_metrics.csv`는 평가 seed 2001~2030의 동일한 남북 혼잡 시나리오에서 Fixed-Time과 DQN을 비교한 결과입니다. 현재 모델은 평균 대기시간과 평균 queue에서 Fixed-Time보다 소폭 개선됐으며, 다른 교통 시나리오에 대한 추가 학습·평가는 계속 필요합니다.
