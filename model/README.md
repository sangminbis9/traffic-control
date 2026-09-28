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

현재 보상은 **현재 step의 정규화된 절대 혼잡 비용**과 signal switching 비용에 음수를 부여합니다. README보다 `model/env/reward.py`가 source of truth입니다.

```text
reward = -1.0 * mean(clip(each_group_queue / 10))
       - 0.1 * clip(total_waiting / 6000)
       - 0.1 * clip(max_waiting / 120)
       - 0.2 * actual_switch_count
```

각 항목은 `utils/config.py`의 scale로 정규화합니다. 최대 대기시간 항이 starvation을 억제하고, 전환 penalty와 yellow/all-red 처리 손실이 잦은 전환을 억제합니다. 이전 snapshot은 향후 delta reward 실험을 위해 함수 signature에 남아 있지만 현재 계산에는 사용하지 않습니다.

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
