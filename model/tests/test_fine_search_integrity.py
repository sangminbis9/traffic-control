"""Technical audit checks; no simulator, policy training or experiment ranking."""

from types import SimpleNamespace

import numpy as np
import torch

from model.experiments import fine_search_integrity as audit


def test_replication_scope_is_exactly_the_five_active_peer_jobs():
    assert set(audit.SOURCE_JOBS) == {(22, 11), (22, 13), (22, 14), (22, 15), (42, 11)}
    assert len(audit.SOURCE_JOBS) == 5
    assert (42, 13) not in audit.SOURCE_JOBS  # The failed main job has its own fresh retry.


def test_nested_comparison_checks_optimizer_tensors_and_non_tensor_state():
    original = {'state': {0: {'step': torch.tensor(7.), 'exp_avg': torch.tensor([.1, .2])}},
                'groups': [{'lr': .0003}], 'held': np.array([2])}
    replica = {'state': {0: {'step': torch.tensor(7.), 'exp_avg': torch.tensor([.1, .2])}},
               'groups': [{'lr': .0003}], 'held': np.array([2])}
    assert audit.compare_nested(original, replica)['all_equal']
    replica['state'][0]['exp_avg'][1] += .001
    replica['groups'][0]['lr'] = .0004
    replica['held'][0] = 3
    result = audit.compare_nested(original, replica)
    assert not result['all_equal']
    assert result['tensor_count'] == 2 and result['tensor_elements'] == 3
    assert result['mismatch_count'] == 3
    assert any('exp_avg' in item['path'] for item in result['mismatches'])


def model():
    network = {'weight': torch.tensor([[1., 2.]])}
    target = {'weight': torch.tensor([[3., 4.]])}
    optimizer = {'state': {0: {'exp_avg': torch.tensor([.1, .2])}}, 'param_groups': [{'lr': .0003}]}
    value = SimpleNamespace(
        q_net=SimpleNamespace(state_dict=lambda: network),
        q_net_target=SimpleNamespace(state_dict=lambda: target),
        policy=SimpleNamespace(optimizer=SimpleNamespace(state_dict=lambda: optimizer)),
        observation_space=SimpleNamespace(shape=(60,)), action_space=SimpleNamespace(n=8),
        num_timesteps=150000, _total_timesteps=150000, _n_updates=36249, _n_calls=149999,
        exploration_rate=.01, _current_progress_remaining=1 / 150000, _hold_remaining=0,
        _held_action=np.array([3]), gamma=.95, learning_starts=5000, n_steps=5,
        target_update_interval=1000,
    )
    return value, target, optimizer


def test_checkpoint_comparison_cannot_pass_on_q_net_alone(tmp_path, monkeypatch):
    source, replica = tmp_path / 'source.zip', tmp_path / 'replica.zip'
    source.write_bytes(b'different zip metadata A')
    replica.write_bytes(b'different zip metadata B')
    left, _, _ = model()
    right, target, optimizer = model()
    monkeypatch.setattr(audit.TrafficDQN, 'load', lambda path, **_kwargs: left if path == str(source) else right)
    assert audit.compare_checkpoint(source, replica)['all_equal']
    target['weight'][0, 0] += 1
    optimizer['state'][0]['exp_avg'][0] += .01
    right._n_updates -= 1
    result = audit.compare_checkpoint(source, replica)
    assert not result['all_equal']
    assert result['components']['q_net']['all_equal']
    assert not result['components']['q_net_target']['all_equal']
    assert not result['components']['optimizer']['all_equal']
    assert not result['components']['training_state']['all_equal']


def bundle(folder, seconds):
    folder.mkdir()
    for name in ('training_routes.json', 'evaluation_routes.json'):
        audit.write_json(folder / name, [{'seed': 8001, 'route_sha256': 'same-demand'}])
    audit.search.write_csv(folder / 'evaluation_metrics.csv', [
        {'seed': seed, 'avg_waiting_time': 20, 'throughput': 100} for seed in range(8001, 8021)
    ])
    audit.search.write_csv(folder / 'training_progress.csv', [
        {'status': 'TRAINING', 'timesteps': 150000, 'total_timesteps': 150000,
         'elapsed_seconds': seconds, 'steps_per_second': 150000 / seconds}
    ])
    (folder / 'final.zip').write_bytes(b'same-policy')


def test_metric_comparison_ignores_runtime_speed_but_not_evaluation_changes(tmp_path, monkeypatch):
    original, replica = tmp_path / 'original', tmp_path / 'replica'
    bundle(original, 1000)
    bundle(replica, 2000)
    monkeypatch.setattr(audit, 'compare_checkpoint', lambda *_args: {'all_equal': True})
    source = {'source_folder': str(original), 'candidate_id': 11, 'training_seed': 22}
    result = {'folder': str(replica), 'reused_completed_run': False}
    assert audit.compare_run(source, result)['all_equal']
    rows = audit.csv_rows(replica / 'evaluation_metrics.csv')
    rows[0]['avg_waiting_time'] = 21
    audit.search.write_csv(replica / 'evaluation_metrics.csv', rows)
    comparison = audit.compare_run(source, result)
    assert not comparison['all_equal']
    assert comparison['training_progress_steps_equal']
    assert not comparison['raw_evaluation_metrics_equal']
    assert not comparison['evaluation_rows_equal']
