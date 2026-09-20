from __future__ import annotations

import os
from pathlib import Path
import threading
import time

import pytest

from api.app.schemas.models import ComparisonCreate, InitialTrafficInput, LanePlacementInput
from api.app.services.model_registry import ModelRegistry
from api.app.services.simulation_runner import SynchronizedComparisonRunner
from model.training import TrainingOptions, TrainingRunner


pytestmark = pytest.mark.skipif(
    os.environ.get('RUN_SUMO_INTEGRATION') != '1',
    reason='Set RUN_SUMO_INTEGRATION=1 to launch two synchronized SUMO processes.',
)


def test_real_sumo_initial_placement_and_synchronized_comparison(tmp_path: Path) -> None:
    registry = ModelRegistry()
    model = next(item for item in registry.list_models() if item['compatible'])
    lane = LanePlacementInput(left=1, straight=1, lane3_total=1, lane3_right=0)
    request = ComparisonCreate(
        test_mode='initial',
        run_mode='single',
        initial=InitialTrafficInput(N=lane, S=lane, E=lane, W=lane),
        model_path=model['path'],
        seed=25_001,
        duration=90,
        render_interval=15,
        speed='max',
    )
    runner = SynchronizedComparisonRunner(
        'integration_sumo', request, Path(model['path']), tmp_path / 'comparison'
    )
    result = runner.run()
    assert result['status'] == 'COMPLETED'
    assert result['requested_initial_vehicles'] == 12
    assert result['actual_initial_vehicles'] == 12
    assert runner.timeseries
    assert all(row['simulation_time'] > 0 for row in runner.timeseries)
    assert (tmp_path / 'comparison' / 'raw_timeseries.csv').exists()


def test_training_pause_checkpoint_and_replay_resume(tmp_path: Path) -> None:
    events: list[dict] = []
    paused_runner = TrainingRunner(
        'pause-smoke',
        TrainingOptions(
            total_steps=2_000,
            validation_interval=10_000,
            checkpoint_interval=10_000,
            progress_interval=5,
            episode_seconds=20,
            learning_starts=0,
            scenario='low',
        ),
        output_dir=tmp_path / 'training',
        progress_handler=events.append,
    )
    thread = threading.Thread(target=paused_runner.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if paused_runner.model is not None and paused_runner.model.num_timesteps >= 10:
            break
        time.sleep(0.05)
    paused_runner.request_pause()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert paused_runner.status == 'PAUSED'
    assert paused_runner.model is not None
    paused_step = paused_runner.model.num_timesteps
    assert paused_step < 2_000
    assert (tmp_path / 'training' / 'final.zip').exists()
    assert (tmp_path / 'training' / 'final.replay.pkl').exists()
    live_frames = [event for event in events if event.get('type') == 'training_frame']
    assert live_frames
    assert live_frames[-1]['network_bounds']
    assert live_frames[-1]['side']['metrics']['approach_queues'].keys() == {
        'N_left', 'N_straight', 'S_left', 'S_straight',
        'E_left', 'E_straight', 'W_left', 'W_straight',
    }
    paused_replay_size = paused_runner.model.replay_buffer.size()

    resumed_runner = TrainingRunner(
        'resume-smoke',
        TrainingOptions(
            total_steps=paused_step + 10,
            validation_interval=10_000,
            checkpoint_interval=10_000,
            progress_interval=5,
            episode_seconds=20,
            learning_starts=0,
            scenario='low',
        ),
        output_dir=tmp_path / 'resumed',
        resume_checkpoint=tmp_path / 'training' / 'final.zip',
    )
    result = resumed_runner.run()
    assert result['status'] == 'COMPLETED'
    assert result['timesteps'] >= paused_step + 10
    assert resumed_runner.model is not None
    assert resumed_runner.model.replay_buffer.size() >= paused_replay_size


def test_training_validation_compares_dqn_with_fixed_time(tmp_path: Path) -> None:
    events: list[dict] = []
    runner = TrainingRunner(
        'fixed-baseline-smoke',
        TrainingOptions(
            total_steps=1,
            validation_interval=1,
            validation_episodes=1,
            checkpoint_interval=100,
            progress_interval=1,
            episode_seconds=10,
            learning_starts=0,
            scenario='low',
        ),
        output_dir=tmp_path / 'fixed-baseline-training',
        progress_handler=events.append,
    )

    result = runner.run()
    validation_events = [event for event in events if event.get('validation')]

    assert result['status'] == 'COMPLETED'
    assert result['fixed_baseline'] is not None
    assert result['validation'] is not None
    assert validation_events
    validation = validation_events[-1]['validation']
    assert validation['fixed_avg_waiting_time'] == result['fixed_baseline']['avg_waiting_time']
    assert 0 <= validation['throughput_retention_pct']
    assert isinstance(validation['beats_fixed'], bool)
    assert (tmp_path / 'fixed-baseline-training' / 'validation_metrics.csv').exists()
