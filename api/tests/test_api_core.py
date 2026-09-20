from __future__ import annotations

import asyncio
from pathlib import Path
from queue import Empty
import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from api.app import database
from api.app.main import _stream, app
from api.app.schemas.models import ComparisonCreate, ContinuousTrafficInput, LanePlacementInput, TrainingCreate
from api.app.services.comparison_service import ComparisonService
from api.app.services.events import EventHub
from api.app.services.model_registry import ModelRegistry
from api.app.services.report_service import ReportService
from api.app.services.training_service import TrainingService
import api.app.services.report_service as report_module
import api.app.services.training_service as training_module
from model.traffic.initial_placement import LanePlacement, generate_initial_route_file
from model.env.state_provider import TrafficSnapshot
from model.utils.metrics import EpisodeMetrics
from model.training import _higher_is_better_change, _lower_is_better_improvement


def test_api_health_and_model_contract() -> None:
    with TestClient(app) as client:
        assert client.get('/api/health').json() == {'status': 'ok'}
        response = client.get('/api/models')
        assert response.status_code == 200
        for model in response.json():
            if model['compatible']:
                assert model['observation_shape'] == [12]
                assert model['action_count'] == 4


def test_continuous_input_requires_complete_rates_and_normalized_ratios() -> None:
    with pytest.raises(ValueError):
        ContinuousTrafficInput(n_rate=0.2)
    with pytest.raises(ValueError):
        ContinuousTrafficInput(left_ratio=0.5, straight_ratio=0.5, right_ratio=0.5)


def test_initial_route_has_exact_lane_counts(tmp_path: Path) -> None:
    placements = {
        'N': LanePlacement(1, 2, 3, 1),
        'S': LanePlacement(0, 1, 2, 0),
        'E': LanePlacement(2, 0, 1, 1),
        'W': LanePlacement(1, 1, 1, 0),
    }
    route_path, count = generate_initial_route_file(tmp_path / 'initial.rou.xml', placements)
    root = ET.parse(route_path).getroot()
    vehicles = root.findall('vehicle')
    assert count == sum(item.total for item in placements.values()) == len(vehicles)
    per_lane = {f'{direction}_{lane}': 0 for direction in 'NSEW' for lane in range(3)}
    for vehicle in vehicles:
        vehicle_id = vehicle.attrib['id'].split('_')
        per_lane[f'{vehicle_id[1]}_{vehicle.attrib["departLane"]}'] += 1
    for direction, placement in placements.items():
        assert per_lane[f'{direction}_0'] == placement.left
        assert per_lane[f'{direction}_1'] == placement.straight
        assert per_lane[f'{direction}_2'] == placement.lane3_total


def test_event_hub_fanout_and_cleanup() -> None:
    hub = EventHub()
    first = hub.subscribe('session')
    second = hub.subscribe('session')
    event = {'type': 'progress', 'value': 1}
    hub.publish('session', event)
    assert first.get_nowait() == event
    assert second.get_nowait() == event
    hub.unsubscribe('session', first)
    hub.unsubscribe('session', second)
    hub.publish('session', event)
    with pytest.raises(Empty):
        first.get_nowait()


def test_websocket_subscribes_before_initial_snapshot() -> None:
    hub = EventHub()

    class FastWebSocket:
        def __init__(self) -> None:
            self.messages: list[dict] = []

        async def accept(self) -> None:
            return None

        async def send_json(self, message: dict) -> None:
            self.messages.append(message)
            if len(self.messages) == 2:
                from fastapi import WebSocketDisconnect
                raise WebSocketDisconnect()

    websocket = FastWebSocket()

    def initial_snapshot() -> dict:
        hub.publish('fast-session', {'status': 'COMPLETED', 'timesteps': 100})
        return {'status': 'PENDING', 'timesteps': 0}

    asyncio.run(_stream(websocket, 'fast-session', hub, initial_snapshot))  # type: ignore[arg-type]

    assert websocket.messages == [
        {'status': 'PENDING', 'timesteps': 0},
        {'status': 'COMPLETED', 'timesteps': 100},
    ]


def test_training_frame_does_not_replace_recoverable_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training_module, 'upsert_training', lambda _row: None)
    service = TrainingService()
    service.sessions['session'] = {
        'id': 'session', 'status': 'PENDING', 'created_at': 'now', 'updated_at': 'now',
        'config': {'mode': 'fixed_steps', 'total_steps': 100}, 'state': {},
        'history': [], 'latest_frame': None, 'artifact_dir': 'artifacts',
        'runner': None, 'thread': None, 'options': None,
    }
    progress = {
        'type': 'training_progress', 'status': 'TRAINING', 'timesteps': 10,
        'total_timesteps': 100, 'episodes': 1, 'epsilon': 0.9,
        'replay_buffer_size': 10, 'rolling_episode_reward': -1.0, 'elapsed_seconds': 1.0,
    }
    frame = {
        'type': 'training_frame', 'status': 'TRAINING', 'timestep': 10,
        'episode': 1, 'simulation_time': 10.0, 'network_bounds': [[0, 0], [1, 1]],
        'side': {},
    }
    service._on_event('session', progress)
    service._on_event('session', frame)
    recovered = service.get('session')
    assert recovered['detail'] == progress
    assert recovered['history'] == [progress]
    assert recovered['frame'] == frame


def test_training_start_is_persisted_before_runner_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    persisted: list[dict] = []
    monkeypatch.setattr(training_module, 'upsert_training', lambda row: persisted.append(row))
    monkeypatch.setattr(TrainingService, '_launch', lambda _service, _record, _checkpoint: None)

    response = TrainingService().start(TrainingCreate(total_steps=100, episode_seconds=10))

    assert response['status'] == 'PENDING'
    assert persisted[0]['id'] == response['id']
    assert persisted[0]['status'] == 'PENDING'
    assert persisted[0]['state'] == {}


def test_persisted_running_training_is_reported_as_interrupted() -> None:
    row = {
        'id': 'stale', 'status': 'TRAINING', 'created_at': 'created', 'updated_at': 'updated',
        'config_json': '{"mode": "auto_convergence"}',
        'state_json': '{"status": "TRAINING", "timesteps": 100}',
        'artifact_dir': 'artifacts',
    }
    recovered = TrainingService._persisted_response(row)
    assert recovered['status'] == 'INTERRUPTED'
    assert recovered['detail']['status'] == 'INTERRUPTED'
    assert recovered['detail']['stop_reason'] == 'server_restarted'


def test_batch_statistics_use_paired_differences() -> None:
    service = ComparisonService(ModelRegistry())
    results = [
        {'fixed': {'avg_waiting_time': 10, 'max_waiting_time': 20, 'avg_queue': 4, 'max_queue': 8, 'throughput': 10, 'phase_changes': 4, 'clearance_time': 50, 'queue_auc': 200},
         'dqn': {'avg_waiting_time': 8, 'max_waiting_time': 16, 'avg_queue': 3, 'max_queue': 6, 'throughput': 10, 'phase_changes': 5, 'clearance_time': 45, 'queue_auc': 160}},
        {'fixed': {'avg_waiting_time': 14, 'max_waiting_time': 25, 'avg_queue': 6, 'max_queue': 9, 'throughput': 11, 'phase_changes': 4, 'clearance_time': 60, 'queue_auc': 240},
         'dqn': {'avg_waiting_time': 11, 'max_waiting_time': 21, 'avg_queue': 5, 'max_queue': 8, 'throughput': 11, 'phase_changes': 5, 'clearance_time': 54, 'queue_auc': 210}},
    ]
    statistics = service._batch_statistics(results)
    waiting = statistics['avg_waiting_time']
    assert waiting['fixed']['mean'] == 12
    assert waiting['dqn']['mean'] == 9.5
    assert waiting['paired_difference']['mean'] == -2.5
    assert waiting['improvement_percent'] == pytest.approx(20.8333333)


def test_lane3_right_validation() -> None:
    with pytest.raises(ValueError):
        LanePlacementInput(lane3_total=1, lane3_right=2)


def test_throughput_is_accumulated_exactly() -> None:
    metrics = EpisodeMetrics('test', 1, 'uniform')
    snapshot = TrafficSnapshot((0,) * 8, 0, 0, 0, 0, 2)
    metrics.record(snapshot)
    metrics.record(TrafficSnapshot((0,) * 8, 0, 0, 0, 0, 3))
    assert metrics.summary()['throughput'] == 5


def test_fixed_baseline_percentage_helpers() -> None:
    assert _lower_is_better_improvement(9, 10) == pytest.approx(10.0)
    assert _lower_is_better_improvement(11, 10) == pytest.approx(-10.0)
    assert _higher_is_better_change(95, 100) == pytest.approx(-5.0)


def test_model_registry_rejects_paths_outside_registry(tmp_path: Path) -> None:
    invalid = tmp_path / 'foreign.zip'
    invalid.write_bytes(b'not a model')
    with pytest.raises(ValueError, match='not part of the project model registry'):
        ModelRegistry().resolve(str(invalid))


def test_report_export_contains_expected_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(database, 'DATABASE_PATH', tmp_path / 'traffic_control.db')
    monkeypatch.setattr(database, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(database, 'ARTIFACTS_DIR', tmp_path / 'artifacts')
    monkeypatch.setattr(report_module, 'ARTIFACTS_DIR', tmp_path / 'artifacts')
    database.initialize_database()
    source = tmp_path / 'experiment'
    source.mkdir()
    (source / 'raw_timeseries.csv').write_text(
        'simulation_time,fixed_vehicles_remaining,dqn_vehicles_remaining,fixed_phase,dqn_phase\n'
        '1,4,4,0,0\n2,3,2,0,2\n', encoding='utf-8'
    )
    model = next(item for item in ModelRegistry().list_models() if item['compatible'])
    result = {
        'model_path': model['path'],
        'fixed': {'avg_waiting_time': 10, 'max_waiting_time': 20, 'avg_queue': 4, 'throughput': 4, 'waiting_p50': 5, 'waiting_p90': 15, 'waiting_p95': 18},
        'dqn': {'avg_waiting_time': 8, 'max_waiting_time': 16, 'avg_queue': 3, 'throughput': 4, 'waiting_p50': 4, 'waiting_p90': 12, 'waiting_p95': 14},
    }
    database.upsert_experiment({
        'id': 'report-test', 'kind': 'single_initial', 'status': 'COMPLETED',
        'created_at': '2026-01-01T00:00:00+00:00', 'updated_at': '2026-01-01T00:00:00+00:00',
        'config': {'model_path': model['path'], 'seed': 1}, 'result': result,
        'artifact_dir': str(source),
    })
    report = ReportService(ModelRegistry()).generate('report-test')
    output = Path(report['artifact_dir'])
    assert Path(report['zip_path']).exists()
    for relative in ('summary.csv', 'raw_timeseries.csv', 'per_run_metrics.csv', 'experiment_config.json', 'model_metadata.json', 'report.html'):
        assert (output / relative).exists()
    for chart in ('avg_waiting', 'max_waiting', 'avg_queue', 'throughput', 'clearance_curve', 'phase_timeline', 'waiting_distribution', 'scenario_comparison'):
        assert (output / 'charts' / f'{chart}.png').exists()
        assert (output / 'charts' / f'{chart}.svg').exists()
