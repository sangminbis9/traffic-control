# SUMO + DQN 교통 신호 제어

Windows PC에서 4방향 × 진입 3차로 교차로를 생성하고, TraCI 제어, Gymnasium 환경, DQN 학습, 고정 신호와의 동일 교통량 비교를 실행하는 프로젝트입니다. 실제 카메라·하드웨어 구현은 포함하지 않습니다.

**현재 상태:** 학습 시드 11·22·33의 15만 스텝 DQN을 확정하고, 별도 교통 시드 30개 × 6시나리오에서 Fixed와 비교했습니다. 최종 수치와 실패한 실험까지 [EXPERIMENTS.md](EXPERIMENTS.md)에 공개합니다. 기본 모델은 검증 데이터로 미리 선정한 seed 22입니다. 초기 6,000스텝 기록은 [VALIDATION.md](VALIDATION.md)에 보존했습니다.

## Installation

Python **3.12 64-bit**를 권장합니다. Python 설치 시 PATH 등록을 선택하세요. Windows의 `python`이 Microsoft Store 별칭으로 연결된다면 `py -3.12` 또는 실제 `python.exe` 경로를 사용합니다.

저장소 최상위 `Capstone`에서 PowerShell로 실행합니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\traffic-rl\requirements-fast-lock.txt
cd traffic-rl
..\.venv\Scripts\python.exe build_network.py
..\.venv\Scripts\python.exe -m traffic.route_generator --scenario balanced --seed 2001
```

`requirements-fast-lock.txt`는 이번 실험에서 사용한 libsumo 포함 전체 버전을 고정하며, `requirements-lock.txt`는 초기 TraCI 환경입니다. `requirements.txt`는 지원 범위를 지정합니다. `eclipse-sumo` 설치에 SUMO, sumo-gui, netconvert 바이너리가 포함되어 별도 NetEdit 작업이나 관리자 설치가 필요 없습니다. PyTorch를 포함하므로 다운로드와 설치에 시간이 걸립니다.

이미 SUMO를 설치했다면 그 설치 경로를 `SUMO_HOME`에 지정하거나 `bin`을 PATH에 추가할 수 있습니다. Python 모듈 `traci`도 설치해야 합니다. 프로젝트는 같은 가상환경의 SUMO 바이너리를 우선 사용합니다. 전역 환경변수 수정은 필요하지 않습니다.

이 작업 PC에는 기본 Python 명령이 실행되지 않아 Codex 제공 Python 3.12로 저장소 루트의 `.venv`를 만들었습니다. **현재 PC에서는 바로 `..\.venv\Scripts\python.exe`를 사용할 수 있습니다.** 다른 PC에서 `.venv` 폴더를 복사하지 말고 새로 생성하세요.

아래 예제의 `python`은 가상환경을 활성화한 경우입니다. 활성화하지 않았다면 모두 `..\.venv\Scripts\python.exe`로 바꾸면 됩니다. PowerShell 실행 정책을 바꿀 필요는 없습니다.

```powershell
# 선택 사항
..\.venv\Scripts\Activate.ps1
```

## Run SUMO test

```powershell
python test_sumo.py
python test_sumo.py --scenario heavy --seconds 300 --seed 2001
python check_env.py
python -m pytest -q
```

`test_sumo.py`는 Gymnasium이나 DQN을 거치지 않고 SUMO 시작, `simulationStep`, 방향별 Queue, 누적 Waiting Time, 현재 Phase, SUMO 실제 신호 문자열을 출력합니다. 고정 신호가 신호 제어기를 호출하며 CSV와 tripinfo XML을 저장합니다. 차로별 차량 수·정지 수, 차량별 현재 연속 대기시간·누적 대기시간은 `SUMOBackend.snapshot()`에서도 읽을 수 있습니다.

종료·예외 발생 시 `finally`에서 SUMO를 닫습니다. 연결은 환경마다 독립적입니다. 통신 무응답에는 20초 timeout을 적용하고 종료되지 않는 **해당 자식 프로세스만** 정리합니다.

## GUI debug 방법

학습된 기본 DQN의 실제 SUMO 화면을 PNG 300장과 5배속 GIF로 저장할 수 있습니다. 기존 녹화 폴더는 덮어쓰지 않습니다.

```powershell
..\.venv\Scripts\python.exe -E capture_sumo.py --scenario ns_heavy --seed 4001 --output results/capture_new
```

`-E`는 외부 `PYTHONPATH` 설정을 무시해 이 가상환경의 TraCI를 사용합니다. `dqn_sumo_5x.gif`, `screenshot_60s.png`, `screenshot_120s.png`, `screenshot_180s.png`와 실제 모델·경로·지표를 기록한 `capture_manifest.json`이 생성됩니다. PNG는 SUMO 자체 GUI 캡처이며, 스크린샷 요청 다음 0.5초 simulation step에 저장됩니다. [2026-09-22 촬영 GIF](results/capture_20260922_dqn/dqn_sumo_5x.gif)를 확인할 수 있습니다.

```powershell
python test_sumo.py --gui --delay 0.03 --seconds 120
python test_sumo.py --gui --capture --seconds 120 --output results/gui_demo
```

`--capture`는 실제 SUMO GUI 화면을 `sumo_gui.png`로 저장합니다. 검증된 화면은 [results/gui_capture/sumo_gui.png](results/gui_capture/sumo_gui.png)에 있습니다. 네 방향 진입 차로는 각 3개이며 출구도 3개입니다.

직접 GUI로 열 수도 있습니다.

```powershell
sumo-gui -c sumo/simulation.sumocfg
# 가상환경을 활성화하지 않았다면
..\.venv\Scripts\sumo-gui.exe -c sumo/simulation.sumocfg
```

GUI의 재생 버튼으로 시작합니다. 이 경우 XML에 포함된 고정 신호 프로그램이 실행됩니다. `config.json`의 고정 시간을 변경했다면 `build_network.py`를 다시 실행해야 정적 GUI 프로그램에도 반영됩니다. Python 제어는 매 실행 시 config 값을 읽습니다.

**Windows 한글 경로:** SUMO 1.27.1의 `SUMO_HOME` 처리와 PNG 저장에 한글 절대경로 문제가 확인되었습니다. Python 실행 경로에서는 자식 프로세스에 ASCII 상대 `SUMO_HOME`, `C` locale을 적용하고 화면도 상대경로로 저장합니다. 직접 `sumo-gui` 명령에서 `Quitting (on unknown error)`가 나오면 위의 `python test_sumo.py --gui` 경로를 사용하세요. 프로젝트를 다른 드라이브에서 실행할 때도 먼저 `traffic-rl`로 이동하는 것을 권장합니다.

## Train

현재 `config.json`은 최종 선정한 34차원 관측·5-step DQN 설정입니다. 기존 설정은 `experiments/default_reference.json`에 보존했습니다.

```powershell
# 선정한 설정으로 새 학습. 검증된 기본 모델을 보존하도록 다른 경로에 저장
python train.py --backend libsumo --seed 22 --steps 150000 --model results/models/dqn_new

# 여러 학습량을 한 학습 궤적에서 저장 (11을 22, 33으로 바꿔 반복)
python experiment_worker.py --config experiments/approaching_temporal.json --seed 11 --steps 150000 --milestones 20000 50000 100000 150000 --output results/reproduction/approach_s11

# 위 저장 모델들을 검증용 교통 시드에서 비교
python experiment_evaluate.py --root results/reproduction --output results/reproduction_validation

# 추가 학습: 별도 출력 모델 사용
python train.py --backend libsumo --resume results/models/dqn.zip --steps 50000 --model results/models/dqn_extended

# 실제 TraCI로 학습하려면 --backend traci (기본값)
```

재현 실험의 출력 폴더는 새 경로를 쓰세요. `train.py`는 시작 시 환경 검사 episode를 수행하므로 원래 실험과 같은 RNG 진행까지 재현하려면 `experiment_worker.py` 명령을 사용합니다. GPU 없이 CPU 스레드 1개로 학습했으며, libsumo는 프로세스마다 환경 하나를 실행합니다. 다른 학습 시드는 별도 프로세스로 병렬 실행할 수 있습니다. GUI에는 TraCI를 사용합니다.

학습률 3e-4, buffer 100,000, batch 64, gamma 0.95, n_steps 5, MLP [128,128], learning_starts 5,000, 총 150,000 decisions입니다. 처음 30,000스텝 동안 epsilon이 감소하고 최종값은 0.01입니다. 학습 중 무작위 탐색 액션을 5~17스텝 유지합니다. epsilon은 탐색 블록을 시작할 확률이므로 실제 탐색에 소비하는 시간 비율과 같지 않습니다. 평가에서는 무작위 탐색 없이 DQN의 deterministic action을 사용합니다. 휴리스틱·전문가 시연은 학습과 추론에 사용하지 않았습니다.

모델 `.zip`에는 설정 `.config.json`과 정보 `.meta.json`이 동반됩니다. `train.py`는 `.replay.pkl`도 저장합니다. 기본 `dqn.zip`에는 재개용 buffer를 함께 배치했습니다. 각 실험 worker의 최종 buffer는 해당 `models/final.replay.pkl`입니다. 중간 checkpoint에는 buffer가 없으므로 추론·평가용으로 사용하세요. 5-step return은 Stable-Baselines3 2.9 이상이 필요합니다.

## Evaluate

선정 모델로 같은 최종 조건을 재현하거나 GUI에서 실행합니다.

```powershell
python evaluate.py --model results/models/dqn.zip --seed-start 4001 --episodes 30 --backend libsumo --output results/evaluation_repeat
python run_model.py --gui --delay 0.03 --scenario heavy --seed 4001
# 실제 TraCI로 6시나리오 빠른 확인
python evaluate.py --model results/models/dqn.zip --seed-start 4001 --episodes 1 --output results/traci_repeat
# DQN 없이 고정 신호만 독립 실행
python evaluate.py --fixed-only --episodes 5 --output results/fixed_baseline
```

원래 최종 실험은 `experiments/frozen_selection.json`에 모델 해시·선정 규칙을 확정한 뒤 `final_evaluate.py`로 세 모델을 비교했습니다. 이미 확인한 4001~4030을 재실행하는 것은 재현 검사이며 새로운 독립 검증은 아닙니다. 이후 튜닝한다면 새 최종 교통 시드 범위를 사전에 정하세요.

기본 평가: **시나리오마다 30개 시드** × 6가지 × 2개 제어기 = 360 episodes입니다. 학습 seed pool은 1~1000, 평가는 기본 2001~2030입니다. 겹치는 평가 시드는 거부합니다. 각 Fixed/DQN 쌍은 차량 출발 시간·경로·차로를 포함한 route XML SHA-256이 일치해야 하며, SUMO seed도 같습니다. 제어 때문에 발생하는 실제 진입 지연과 도착 시각은 달라질 수 있습니다.

모델의 설정 파일을 자동 사용하고, 관측값 의미를 바꾸는 설정이나 네트워크가 다르면 비교를 거부합니다. 네트워크 해시는 생성 시각 주석을 제외하므로 같은 네트워크를 재생성해도 유효합니다. 기존 평가 CSV가 있는 출력 디렉터리는 실험 혼합을 막기 위해 덮어쓰지 않습니다.

## 교통 시나리오와 차로

| 시나리오 | 방향별 발생률 N/S/E/W (대/시간) |
|---|---|
| `balanced` | 600 / 600 / 600 / 600 |
| `ns_heavy` | 1300 / 1300 / 350 / 350 |
| `ew_heavy` | 350 / 350 / 1300 / 1300 |
| `left_heavy` | 기본 600, 지정 방향은 1300 및 좌회전 70% |
| `heavy` | 모두 1600 |
| `low` | 모두 150 |
| `random` | 매 episode 방향별 발생률 및 회전비 무작위 |

방향 이름은 차량이 **출발하는 접근 방향**입니다. SUMO 차로 0은 운전자 기준 오른쪽입니다. 차로 2=좌회전 전용, 차로 1=직진, 차로 0=직진+우회전입니다. 우회전은 해당 방향 직진 신호와 함께 움직이며 별도의 액션이나 적색 우회전은 없습니다. U-turn은 없습니다.

`traffic.rates`에 `{"N":900,"S":700,"E":400,"W":400}`처럼 지정하면 발생률을 직접 설정합니다. `turn_ratios`는 [좌회전, 직진, 우회전] 합계 1입니다. `left_heavy`는 지정 방향의 비율을 [0.7,0.2,0.1]로 덮어쓰며, `random`은 기본 비율 주변의 Dirichlet 표본으로 회전비도 바꿉니다. 정확히 지정한 비율을 쓰려면 `balanced` 등의 고정 시나리오를 선택합니다. 차량 발생은 지수 도착 간격을 사용하는 Poisson 과정입니다.

기본 episode는 300초, 수요 발생은 240초입니다. 마지막 60초에도 남은 차량이 있으면 그대로 기록합니다. 이 설정은 `simulation`에서 바꿉니다. 발생 종료 시각과 episode 종료를 함께 늘려 장기 혼잡 실험도 할 수 있습니다.

## State 설명

현재 기본은 `Box(0,1,(34,),float32)`입니다. `observation.include_approaching=false`인 초기 설정·모델은 26차원을 유지합니다.

| 인덱스 | 의미 |
|---|---|
| 0~7 | N_left, N_straight, S_left, S_straight, E_left, E_straight, W_left, W_straight |
| 8 | 진입 차로에 있는 차량들의 누적 정지 대기시간 합 |
| 9 | 진입 차로 차량의 최대 누적 정지 대기시간 |
| 10~13 | 현재 Phase one-hot |
| 14 | 현재 green/yellow/all_red 단계 경과시간 / 해당 단계 제한시간 |
| 15~17 | green / yellow / all_red one-hot |
| 18~21 | 전환 목표 Phase one-hot; green에서는 모두 0 |
| 22~25 | 각 Phase가 녹색을 제공하지 않은 시간 / max_red |
| 26~33 | 정지선까지 50m 이내에서 움직이는 차량 수, Queue와 같은 8그룹, 10으로 나눔 |

직진 계열 Queue는 0·1차로를 합쳐 우회전 차량도 포함합니다. 정지 기준은 SUMO의 0.1m/s 미만입니다. 차량별 `getWaitingTime`은 다시 움직이면 초기화되므로 관측값에는 episode 전체 누적값인 `getAccumulatedWaitingTime`을 사용합니다. 누적 기억 기간은 episode보다 길게 설정합니다.

Queue는 그룹당 10대(초기 설정은 40대), 총 대기는 6000초, 최대 대기는 120초로 나누고 [0,1]로 clip합니다. 원시 지표는 clip하지 않습니다. 포화된 관측은 더 심한 혼잡을 구별하지 못하므로 장기 실험에서는 scale을 검토하고 모델을 새로 학습하세요.

요청의 약 12개 기본 feature를 확장한 이유는 동일 Queue·현재 Phase에서도 황색/전적색 여부와 목표 Phase에 따라 다음 행동 결과가 달라지기 때문입니다. 제어기 내부 상태를 함께 노출해 이 모호함을 줄였습니다. 미래 수요·차량 ID·원시 좌표는 모델에 주지 않습니다. SUMO의 차로 위치는 정지선 50m 이내 접근 차량 집계에만 사용합니다. 속도 0.1m/s 이상인 차량만 추가 집계해 정지 Queue와 중복되지 않습니다.

## Action / Signal Controller 설명

`Discrete(4)`: 0=남북 직진(+우회전), 1=남북 좌회전, 2=동서 직진(+우회전), 3=동서 좌회전.

같은 Phase 요청은 유지, 다른 Phase 요청은 **황색 → 전적색 → 새 녹색**으로 전환합니다. 전환 중 새 요청은 무시하고 이미 선택한 목표를 유지합니다. Minimum Green 전의 요청도 무시합니다. DQN은 황색·전적색을 직접 선택하지 않습니다.

기본: decision 1초, SUMO step 0.5초, minimum green 3초, maximum green 15초, yellow 1초, all-red 1초. 0.5초 all-red도 지원합니다. 모든 시간은 simulation step의 배수여야 합니다.

최대 녹색에 도달하면 가장 오래 서비스하지 않은 다른 Phase로 강제 전환합니다. 추가로 `max_red=60`초 이상 서비스하지 않은 Phase가 생기면 minimum green 이후 이를 우선합니다. 이는 학습 초기 특정 Phase가 영원히 배제되지 않게 하는 보조 규칙이며, 60초가 차량별 대기의 절대 상한이라는 뜻은 아닙니다. 전환 시간·다른 밀린 Phase 처리·수요 초과로 더 오래 기다릴 수 있습니다. 강제 전환 횟수는 별도 기록합니다.

## Reward 설명

한 decision 동안 0.5초마다 관측한 아래 비용을 시간 평균합니다.

```text
q = mean(clip(each_group_queue / 10, 0, 1))
w = clip(total_waiting / 6000, 0, 1)
m = clip(max_waiting / 120, 0, 1)
reward = average(-1.0*q - 0.1*w - 0.1*m) - 0.2*actual_switch_count
```

무시된 전환 요청에는 벌점을 주지 않습니다. 실제 전환 시작 횟수에만 적용하며 강제 전환도 포함합니다. 전환 중 발생하는 Queue와 Waiting 비용도 모두 반영합니다. `utils/reward.py`에서 각 항을 반환하며 CSV의 `reward_queue`, `reward_waiting`, `reward_max_waiting`, `reward_switching`으로 scale을 확인할 수 있습니다.

## Fixed-Time Controller 설명

0 → 1 → 2 → 3 순환, 녹색 [10,4,10,4]초, 각 전환에 같은 황색·전적색을 사용합니다. 기본 주기는 36초입니다. DQN과 동일한 SignalController·네트워크·평가 metric을 사용합니다. 기본 시간에서는 정확한 주기가 검증됐습니다. 사용자 정의 시간이 decision interval보다 세밀하면 Gym 환경의 고정 제어 요청도 다음 decision 시각으로 양자화됩니다.

## 평가 지표 정의

- `avg_waiting_time`: episode 중 실제 삽입된 **모든** 차량(도착+미도착)의 누적 정지 대기시간 평균. 완료 차량만 평균내는 선택 편향을 피합니다.
- `max_waiting_time`: 위 차량의 최대 누적 정지 대기시간. 연속 대기시간과 다릅니다.
- `avg_queue`, `max_queue`: 12개 진입 차로 정지 차량 합계의 시간 평균/최댓값.
- `throughput`: episode 안에 출구 끝까지 도착한 차량 수. 교차로 진입 횟수와 다릅니다.
- `phase_changes`: 황색 전환 시작 횟수. `forced_changes`는 안전·공정성 규칙의 개입 횟수.
- `episode_reward`: 모든 decision reward 합.
- `scheduled`, `departed`, `unfinished`, `not_inserted`, `pending`: 계획 수요, 실제 삽입, 삽입 후 미도착, 아직 삽입 못한 차량, 마지막 시점 삽입 대기 수.
- `collisions`, `teleports`: 매 simulation step에서 탐지한 사건 수. 발견 시 실행을 실패시킵니다.

삽입 전 지연은 차량 대기 metric에 들어가지 않으므로 `not_inserted`, throughput, unfinished를 **함께** 비교해야 합니다. 원시 `tripinfo.xml`에는 차량별 `waitingTime`, `departDelay`, 도착 여부가 있어 추가 분석할 수 있습니다. 시간이 끝나도 무리하게 모든 차량을 강제 배출하지 않습니다.

## 프로젝트 구조 / 확장 지점

```text
sumo/                  원본 노드·차로 연결 XML, 생성 net.xml, 기본 routes, sumocfg
simulation/backend.py  SUMO 프로세스 수명·TraCI 센싱·신호 출력
traffic/               seed 기반 route 생성, TrafficState / StateProvider
controller/            Phase 정의, 시간 기반 전환기, 고정 제어 정책
env/                   Gymnasium reset / step / close
utils/                 설정 검증, reward, metrics, PNG 생성
tests/                 전환 시간·공정성·시드·실제 SUMO 통합 검사
build_network.py       NetEdit 없이 네트워크 생성·검사
test_sumo.py           강화학습 없이 TraCI / GUI 직접 검사
check_env.py           SB3 환경 검사
train.py               DQN 학습·저장·재로딩
evaluate.py            동일 수요 쌍별 비교·CSV·그래프
```

흐름은 SUMOBackend → Snapshot → SUMOTrafficStateProvider → TrafficState → 정규화 + SignalController 상태 → DQN → action → SignalController → SignalSink입니다. 미래 카메라에서는 동일한 단위의 `TrafficState`를 만들고, 실제 신호 출력은 `SignalSink.set_signal()`의 어댑터로 연결하면 됩니다. 카메라의 누적 대기 측정과 시뮬레이션의 정의를 맞춰야 하며 기존 모델의 성능이 자동 보장되지는 않습니다. `simulation/libsumo_backend.py`는 같은 센싱·출력 인터페이스의 고속 학습 backend이며, 실제 TraCI와 관측·보상·최종 지표 일치를 검사했습니다. 카메라 구현 시 정지선 50m 영역의 움직이는 차량 수까지 같은 정의로 측정해야 합니다.

## 결과 파일 위치

- [EXPERIMENTS.md](EXPERIMENTS.md): 다중 시드·학습량 실험과 최종 결과.
- `results/experiment_leaderboard.csv`: 실패를 포함한 전체 52개 검증 checkpoint.
- `results/heldout_approach/analysis/`: 최종 paired CSV·신뢰구간·시나리오별 표·그래프.
- `experiments/PROTOCOL.md`, `experiments/frozen_selection.json`: 실험 변경 이력과 최종 모델 확정 기록.

- `results/models/`: 모델, 설정, metadata, replay buffer.
- `results/training/<run>/simulation/episodes.csv`: 학습 episode별 reward와 metric.
- `results/sumo_test/`: TraCI 단독 테스트 CSV·로그·route·tripinfo.
- `results/evaluation_smoke/episodes.csv`: 실제 60회 실행 결과.
- `summary.csv`, `summary_by_scenario.csv`: 평균·표준편차(ddof=1)·episode 수.
- `paired_deltas.csv`, `paired_summary.csv`: 동일 scenario/seed의 DQN−Fixed 차이.
- `plots/`: 평균 대기·Queue·throughput·전환 횟수·평가 reward·학습 reward PNG 6개.
- 평가 `runs/`: 각 제어기의 실제 route XML, SUMO 로그, 차량별 tripinfo.

서로 다른 시나리오의 단순 평균은 시나리오별 동일 가중치의 기술 통계입니다. pooled 표준편차에는 시나리오 차이도 포함되므로 `summary_by_scenario.csv`와 paired 결과도 확인하세요. 현재 결과는 3개 학습 seed와 분리된 검증/최종 평가에 근거합니다. 더 긴 episode와 실제 교차로·다른 수요 분포로의 일반화는 추가 검증이 필요합니다.

## 참고 문서

- [SUMO Python 배포 및 설치](https://sumo.dlr.de/docs/Downloads.php#python_packages_virtual_environments)
- [Plain XML과 신호 연결 정의](https://sumo.dlr.de/docs/Networks/PlainXML.html)
- [TraCI 차량 대기시간 정의](https://sumo.dlr.de/docs/TraCI/Vehicle_Value_Retrieval.html)
- [Gymnasium custom environment](https://gymnasium.farama.org/introduction/create_custom_env/)
- [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
