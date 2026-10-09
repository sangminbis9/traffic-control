"""Orchestration checks with synthetic ledgers; no SUMO or training is launched."""

import itertools
from pathlib import Path

import pytest

from model.experiments import fine_reward_search as search


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    search.write_json(path, value)


def job(parent: Path, *, candidate=search.INCUMBENT_ID, seed=22, steps=50_000):
    return {
        'stage': 'fine', 'candidate_id': candidate,
        'weights': list(search.INCUMBENT), 'training_seed': seed,
        'training_steps': steps, 'parent': str(parent),
    }


def test_grid_and_jobs_keep_the_incumbent_and_independent_fresh_runs(tmp_path):
    expected = set(itertools.product((.2, .3, .4), (.4, .5, .6), (.4, .5, .6)))
    assert len(search.GRID) == 27
    assert set(search.GRID.values()) == expected
    assert search.GRID[search.INCUMBENT_ID] == (.3, .5, .5)
    weights = {str(i): list(value) for i, value in search.GRID.items()}
    jobs = search.jobs_for(tmp_path, 'multiseed', [1, search.INCUMBENT_ID],
                           weights, [22, 42, 62], 150_000)
    assert len({entry['parent'] for entry in jobs}) == len(jobs) == 6
    assert {(entry['candidate_id'], entry['training_seed']) for entry in jobs} == {
        (candidate, seed) for candidate in (1, search.INCUMBENT_ID) for seed in (22, 42, 62)
    }
    assert all(entry['training_steps'] == 150_000 for entry in jobs)
    assert all(entry['weights'] == weights[str(entry['candidate_id'])] for entry in jobs)


def test_boundary_extends_only_top_three_points_one_axis_at_a_time():
    weights = {str(i): list(value) for i, value in search.GRID.items()}
    axes = [[.2, .3, .4], [.4, .5, .6], [.4, .5, .6]]
    assert search.boundary_extensions([search.INCUMBENT_ID], weights, axes) == []
    # The fourth-ranked upper corner must not generate an outward point.
    ranked = [1, search.INCUMBENT_ID, 5, 27]
    additions = search.boundary_extensions(ranked, weights, axes)
    assert set(additions) == {(.1, .4, .4), (.2, .3, .4), (.2, .4, .3), (.1, .5, .5)}
    assert len(additions) == len(set(additions))
    assert len(weights) == 27


def test_boundary_never_produces_negative_or_already_tested_weights():
    weights = {'1': [0, .5, .5], '2': [.1, .5, .5], '3': [.2, .5, .5]}
    axes = [[0, .1], [.4, .5, .6], [.4, .5, .6]]
    # Candidate 1 would extend below zero, while candidate 2's extension exists.
    assert search.boundary_extensions([1, 2], weights, axes) == []


def test_evaluation_demand_pairing_is_order_independent_and_crosses_training_seeds():
    rows = [
        {'scenario': scenario, 'evaluation_seed': traffic, 'training_seed': train,
         'candidate_id': candidate, 'demand_sha256': f'{scenario}-{traffic}'}
        for scenario in ('random', 'heavy') for traffic in (9001, 9002)
        for train in (22, 42) for candidate in (1, search.INCUMBENT_ID)
    ]
    expected = {'verified': True, 'unique_demands': 4, 'evaluation_rows': 16}
    assert search.verify_evaluation_hashes(rows) == expected
    assert search.verify_evaluation_hashes(list(reversed(rows))) == expected
    changed = [dict(row) for row in rows]
    changed[-1]['demand_sha256'] = 'different-route'
    with pytest.raises(RuntimeError, match='Evaluation demand differs'):
        search.verify_evaluation_hashes(changed)


def test_training_pairing_allows_different_seeds_but_rejects_different_candidate_demand(tmp_path):
    results = []
    for seed in (22, 42):
        for candidate in (1, search.INCUMBENT_ID):
            folder = tmp_path / f'{seed}-{candidate}'
            save(folder / 'training_routes.json', [{'seed': seed, 'route_sha256': f'train-{seed}'}])
            save(folder / 'evaluation_routes.json', [{'seed': 8001, 'route_sha256': 'validation'}])
            results.append({'job': {'training_seed': seed}, 'folder': str(folder)})
    search.verify_pairing(list(reversed(results)))
    save(Path(results[-1]['folder']) / 'training_routes.json',
         [{'seed': 42, 'route_sha256': 'different'}])
    with pytest.raises(RuntimeError, match='training/evaluation demand hash mismatch'):
        search.verify_pairing(results)


@pytest.mark.parametrize('changed', ['core', 'runtime'])
def test_resume_refuses_mixed_source_or_runtime_before_reusing_models(tmp_path, monkeypatch, changed):
    config = {'core_sha256': {'worker': 'original'}, 'runtime': {'torch': 'original'}}
    save(tmp_path / 'search_config.json', config)
    monkeypatch.setattr(search, 'core_hashes', lambda: {'worker': 'changed' if changed == 'core' else 'original'})
    monkeypatch.setattr(search, 'current_runtime', lambda: {'torch': 'changed' if changed == 'runtime' else 'original'})
    with pytest.raises(ValueError, match='Immutable|Runtime changed'):
        search.initialize(tmp_path, workers=6, resume=True)


def test_existing_search_requires_explicit_resume(tmp_path):
    save(tmp_path / 'search_config.json', {})
    with pytest.raises(ValueError, match='requires --resume'):
        search.initialize(tmp_path, workers=6, resume=False)


def test_completed_training_reuse_verifies_checkpoint_and_job_identity(tmp_path, monkeypatch):
    task = job(tmp_path / 'run')
    attempt = Path(task['parent']) / 'attempt_001'
    folder = attempt / f'experiment_{task["candidate_id"]}'
    save(attempt / 'job.json', task)
    save(folder / 'summary.json', {'training_steps': 50_000})
    checkpoint = folder / 'final.zip'
    checkpoint.write_bytes(b'audited synthetic checkpoint')
    save(folder / 'experiment_config.json', {
        'completed_training_steps': 50_000, 'model_sha256': search.sha256(checkpoint),
    })
    monkeypatch.setattr(search, 'worker', lambda *_args: pytest.fail('A completed job must not train again'))
    assert search.run_training_job(task)['reused_completed_run'] is True
    with pytest.raises(ValueError, match='protocol mismatch'):
        search.run_training_job({**task, 'training_seed': 42})
    checkpoint.write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='hash mismatch'):
        search.run_training_job(task)


def test_interrupted_training_restarts_in_a_fresh_attempt_with_original_schedule(tmp_path, monkeypatch):
    task = job(tmp_path / 'run')
    first = Path(task['parent']) / 'attempt_001'
    save(first / 'job.json', task)
    marker = first / 'incomplete-model.txt'
    marker.write_text('preserve interrupted evidence', encoding='utf-8')
    calls = []
    monkeypatch.setattr(search, 'worker', lambda candidate, settings: calls.append((candidate, settings)))
    result = search.run_training_job(task)
    assert result['reused_completed_run'] is False
    assert Path(result['folder']).parent.name == 'attempt_002'
    assert marker.read_text(encoding='utf-8') == 'preserve interrupted evidence'
    assert len(calls) == 1
    candidate, settings = calls[0]
    assert candidate == task['candidate_id']
    assert settings['timesteps'] == 50_000
    assert settings['schedule_timesteps'] == 150_000
    assert settings['seed'] == 22
    assert settings['weights'] == list(search.INCUMBENT)
    assert settings.get('resume_checkpoint') is None
    assert settings['eval_seed_start'] == 8001 and settings['eval_episodes'] == 20


def synthetic_result(task):
    """Small complete evidence bundle, with deliberately outward-improving metrics."""
    folder = Path(task['parent']) / 'attempt_001' / f'experiment_{task["candidate_id"]}'
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / 'final.zip'
    checkpoint.write_bytes(f'synthetic-policy-{task["candidate_id"]}-{task["training_seed"]}'.encode())
    save(folder / 'experiment_config.json', {'model_sha256': search.sha256(checkpoint)})
    save(folder / 'training_routes.json', [{'seed': task['training_seed'], 'route_sha256': 'train'}])
    save(folder / 'evaluation_routes.json', [
        {'seed': seed, 'route_sha256': f'route-{seed}'} for seed in range(8001, 8021)
    ])
    search.write_csv(folder / 'evaluation_metrics.csv', [
        {'seed': seed, 'avg_waiting_time': 100 - 10 * sum(task['weights']),
         'max_waiting_time': 60, 'avg_queue': 10, 'phase_changes': 20,
         'throughput': 100, 'collisions': 0, 'teleports': 0}
        for seed in range(8001, 8021)
    ])
    return {'job': task, 'folder': str(folder), 'reused_completed_run': False}


def test_fine_search_stops_after_two_boundary_rounds_and_keeps_incumbent(tmp_path, monkeypatch):
    calls = []

    def fake_jobs(_output, stage, jobs, _workers):
        calls.append((stage, len(jobs)))
        return [synthetic_result(task) for task in jobs]

    monkeypatch.setattr(search, 'run_jobs', fake_jobs)
    config = {
        'candidate_weights': {str(i): list(value) for i, value in search.GRID.items()},
        'grid_axes': [[.2, .3, .4], [.4, .5, .6], [.4, .5, .6]],
        'boundary_protocol': {'max_rounds': 2}, 'validation_seeds': list(range(8001, 8021)),
        'bootstrap_samples': 32, 'bootstrap_seed': 7,
    }
    lock = search.run_fine(tmp_path, config, workers=6)
    state = search.read_json(tmp_path / 'boundary_state.json')
    assert [stage for stage, _count in calls] == ['fine', 'boundary_1', 'boundary_2']
    assert calls[0][1] == 27
    assert len(state['rounds']) == 2
    assert state['boundary_unresolved'] is True
    assert state['pending_outward_points']
    assert 27 < len(state['candidate_weights']) <= 45
    assert search.INCUMBENT_ID in lock['shortlist_candidate_ids']
    assert len(lock['shortlist_candidate_ids']) <= 4
    assert lock['holdout_opened'] is False
    # A resumed, exhausted search cannot silently add a third boundary round.
    resumed = search.run_fine(tmp_path, config, workers=6)
    assert len(calls) == 4 and calls[-1][0] == 'fine'
    assert resumed['shortlist_candidate_ids'] == lock['shortlist_candidate_ids']
    assert len(search.read_json(tmp_path / 'boundary_state.json')['rounds']) == 2


@pytest.mark.parametrize('changed_field', ['selected_candidate_id', 'selected_weights', 'model_fingerprints'])
def test_selection_lock_rejects_reselection_or_changed_evidence(tmp_path, changed_field):
    path = tmp_path / 'selection_lock.json'
    original = {
        'locked_at': 'first-lock', 'selected_candidate_id': 2,
        'selected_weights': [.2, .4, .5], 'model_fingerprints': [{'model_sha256': 'first'}],
    }
    fields = ('selected_candidate_id', 'selected_weights', 'model_fingerprints')
    locked = search.lock_payload(path, original, fields)
    assert search.lock_payload(path, {**original, 'locked_at': 'later'}, fields) == locked
    before = path.read_bytes()
    changed = {
        'selected_candidate_id': 3, 'selected_weights': [.2, .5, .5],
        'model_fingerprints': [{'model_sha256': 'changed'}],
    }
    with pytest.raises(ValueError, match='Locked decision or its evidence changed'):
        search.lock_payload(path, {**original, changed_field: changed[changed_field]}, fields)
    assert path.read_bytes() == before


def test_checkpoint_fingerprints_bind_models_and_validation_evidence(tmp_path):
    results = [synthetic_result(job(tmp_path / str(seed), seed=seed)) for seed in (42, 22)]
    fingerprints = search.checkpoint_fingerprints(results)
    assert fingerprints == search.checkpoint_fingerprints(list(reversed(results)))
    assert [row['training_seed'] for row in fingerprints] == [22, 42]
    csv_path = Path(results[0]['folder']) / 'evaluation_metrics.csv'
    csv_path.write_text(csv_path.read_text(encoding='utf-8-sig') + '\n', encoding='utf-8-sig')
    assert search.checkpoint_fingerprints(results) != fingerprints
    (Path(results[0]['folder']) / 'final.zip').write_bytes(b'changed-policy')
    with pytest.raises(ValueError, match='Model changed before selection was locked'):
        search.checkpoint_fingerprints(results)


def test_cached_evaluation_cannot_hide_a_changed_locked_model(tmp_path):
    result = synthetic_result(job(tmp_path / 'training'))
    source = Path(result['folder'])
    folder = tmp_path / 'evaluation'
    metadata = search.read_json(source / 'experiment_config.json')
    task = {'source': str(source), 'folder': str(folder), 'model_sha256': metadata['model_sha256']}
    save(folder / 'evaluation_config.json', task)
    save(folder / 'results.json', [{'cached': True}])
    assert search.evaluate_job(task) == [{'cached': True}]
    # Even internally consistent replacement metadata cannot override the locked SHA.
    (source / 'final.zip').write_bytes(b'replacement-policy')
    save(source / 'experiment_config.json', {'model_sha256': search.sha256(source / 'final.zip')})
    with pytest.raises(ValueError, match='differs from locked model'):
        search.evaluate_job(task)
