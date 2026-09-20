# Traffic Control Lab

SUMO 기반 4방향·3차로 교차로, Stable-Baselines3 DQN, Fixed-Time 기준 제어기, FastAPI 실험 서버, React 연구 대시보드를 한 저장소에서 운영하는 캡스톤 프로젝트입니다. 웹의 차량 위치와 지표는 임의 애니메이션이 아니라 TraCI가 읽은 실제 SUMO 상태입니다.

## Architecture

```text
traffic-control/
├── model/   # SUMO, TraCI, Gymnasium, DQN 학습·평가, 안전 신호 제어
├── api/     # FastAPI, WebSocket, SQLite metadata, 실험/보고서 서비스
├── web/     # React + TypeScript + Vite 반응형 대시보드
└── README.md
```

실행 흐름은 `React → REST/WebSocket → FastAPI service → model.IntersectionEnv → TraCI → SUMO`입니다. 모델 바이너리와 CSV/PNG/SVG는 파일시스템에 저장하고 SQLite에는 세션·실험·모델·보고서 metadata만 저장합니다.

## Windows quick start

Python 3.11 이상, Node.js, [Eclipse SUMO](https://sumo.dlr.de/docs/Installing/index.html)가 필요합니다. 프로젝트는 PATH, `SUMO_HOME`, Windows 표준 설치 경로를 순서대로 탐색합니다. 현재 검증 환경은 Python 3.14.4, Node.js 24.19.0, SUMO 1.27.1입니다.

다른 Windows PC에서 저장소를 clone한 뒤 최초 한 번
`setup_traffic_control.bat`을 더블클릭하면 Python 가상환경, Python/웹 의존성,
SUMO 네트워크와 테스트를 준비합니다. 설치가 끝나면
`start_traffic_control.bat`으로 서버와 웹을 한 번에 실행합니다.
Python은 `requirements-windows.lock.txt`, 웹은 `web/package-lock.json`에 고정된
검증 버전을 설치하므로 두 PC의 실행 환경 차이를 줄일 수 있습니다.

```powershell
git clone https://github.com/sangminbis9/traffic-control.git
cd traffic-control
.\setup_traffic_control.bat
.\start_traffic_control.bat
```

수동 설치가 필요하면 다음 명령을 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r model\requirements.txt
python -m pip install -r api\requirements.txt
python -m model.sumo.build_network
```

터미널 1 — API:

```powershell
python -m api.app.main
```

터미널 2 — Web:

```powershell
cd web
npm install
npm run dev
```

브라우저에서 `http://127.0.0.1:5173`을 엽니다. API 문서는 `http://127.0.0.1:8000/docs`입니다.

### One-click launcher

최초 설치를 `setup_traffic_control.bat`으로 마친 뒤에는 프로젝트 루트의
`start_traffic_control.bat`을 더블클릭하면 됩니다. 누락된 SUMO 네트워크를
자동 생성하고 API·웹 서버를 백그라운드에서 시작한 다음 학습실을 기본
브라우저로 엽니다.

종료할 때는 `stop_traffic_control.bat`을 실행합니다. 실행 파일이 직접 시작한
프로세스만 PID와 시작 시간을 함께 확인한 뒤 종료합니다. 서버 로그는
`api/artifacts/logs/`에 저장됩니다.

## Existing model CLI

기존 CLI는 그대로 유지됩니다.

```powershell
python -m model.test_sumo --seconds 60
python -m model.train --timesteps 1000 --episode-seconds 120 --check-env
python -m model.evaluate --episodes 30 --seed-start 2001
```

세부 모델 설계는 [model/README.md](model/README.md), API lifecycle과 endpoint는 [api/README.md](api/README.md), 웹 화면과 개발 명령은 [web/README.md](web/README.md)를 참고하세요.

## Main capabilities

- Fixed Steps / Auto Convergence DQN 학습, 별도 validation seed, 교통 성능 기반 best checkpoint
- Callback 기반 안전 Pause/Stop과 model + replay buffer 저장, checkpoint Resume
- 12개 진입 차로별 0~5대 deterministic 초기 배치와 실제 차량 수 검증
- 동일 route·seed·network·simulation step을 사용하는 Fixed-Time/DQN paired comparison
- 실제 차량 좌표, 신호 전환, queue/wait/throughput을 WebSocket으로 전송
- 단일·batch 실험, paired difference, mean/std/median/min/max/95% CI
- CSV/JSON/HTML 및 16:9 PNG/SVG 차트를 포함한 Presentation Pack ZIP
- seed, Git SHA, model SHA-256, Python/SUMO 버전, reward/DQN/signal 설정 저장

## Tests

```powershell
python -m pytest model\tests api\tests -q
cd web
npm test
npm run build
```

실제 SUMO 프로세스 두 개와 학습 checkpoint 복원까지 검사하는 통합 테스트:

```powershell
$env:RUN_SUMO_INTEGRATION = "1"
python -m pytest api\tests\test_integration_sumo.py -q
```

생성되는 runtime 파일은 `api/data/`, `api/artifacts/`, `model/sumo/generated/`에 있으며 Git에서 제외됩니다.

## Model artifact policy

학습 중간 checkpoint와 replay buffer는 저장소 용량을 빠르게 증가시키므로
Git에서 제외합니다. 미사용 seed 30회 이상의 paired 평가에서 평균 대기시간,
Queue, Throughput과 최대 대기시간 기준을 모두 통과한 모델만 최종 artifact로
추가합니다. 300,000-step 실험의 채택/기각 근거는
[model/results/README.md](model/results/README.md)에 기록되어 있습니다.
