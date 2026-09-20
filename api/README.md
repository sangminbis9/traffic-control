# API

FastAPI 서버는 장시간 학습과 SUMO 비교를 worker thread에서 실행하므로 event loop를 막지 않습니다. 진행 정보는 WebSocket으로 전송하고, 브라우저 연결이 끊기면 comparison runner에 stop을 요청해 두 TraCI 연결을 `finally`에서 닫습니다.

## Run

```powershell
python -m pip install -r api\requirements.txt
python -m api.app.main
```

기본 주소는 `http://127.0.0.1:8000`, OpenAPI UI는 `/docs`입니다.

## REST and WebSocket

```text
GET  /api/health
GET  /api/models
GET|POST /api/training
GET  /api/training/{id}
POST /api/training/{id}/pause|resume|stop
WS   /ws/training/{id}

POST /api/comparisons
GET  /api/comparisons/{id}
POST /api/comparisons/{id}/pause|resume|stop|speed
WS   /ws/comparisons/{id}

GET  /api/experiments
GET  /api/experiments/{id}
POST /api/reports/{experiment_id}
GET  /api/reports/{report_id}/download
```

`simulation_frame`은 simulation time, network bounds, 두 controller의 phase/state/vehicles/metrics, delta를 담습니다. `training_progress`는 timestep, episode, epsilon, replay-buffer size, rolling reward, validation 및 checkpoint를 담습니다.

## Training lifecycle

Pause/Stop은 process suspend가 아니라 SB3 callback 종료 신호입니다. 종료 시 `final.zip`과 `final.replay.pkl`을 함께 저장합니다. Resume은 두 파일과 `num_timesteps`를 복원합니다. Validation은 별도 환경·seed를 사용하고 평균/최대 대기, queue, throughput, phase change를 평가합니다. Best model은 평균 대기 중심 score, 최대 대기 제한, 기존 best 대비 throughput 90% 유지 조건으로 선택합니다.

## Comparison lifecycle

Initial mode는 network XML의 실제 lane length, `departPos`, 8 m 간격을 사용합니다. 첫 SUMO step 뒤 12개 lane별 requested/actual count가 다르면 중단합니다. Fixed/DQN은 같은 route·seed·network·step·safety timing을 사용하며 매 decision 뒤 clock equality를 검사합니다.

## SQLite schema

- `training_sessions`: status, config/state JSON, artifact directory
- `experiments`: kind, status, config/result JSON, artifact directory
- `model_metadata`: SHA-256, path, name, compatibility metadata
- `reports`: experiment id, output/ZIP path, file metadata

모델 binary는 DB에 저장하지 않습니다.

## Presentation Pack

`api/artifacts/reports/`에 summary/raw/per-run CSV, experiment/model JSON, HTML, PNG/SVG charts를 만들고 ZIP으로 묶습니다. 단일 실험은 n=1, batch는 실제 n과 paired statistics를 표시합니다.

## Test

```powershell
python -m pytest api\tests -q
$env:RUN_SUMO_INTEGRATION = "1"
python -m pytest api\tests\test_integration_sumo.py -q
```
