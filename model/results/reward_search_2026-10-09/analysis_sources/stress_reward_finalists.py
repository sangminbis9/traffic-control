"""Diagnostic-only fixed-demand stress check; never changes the locked winner."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.controller.traffic_dqn import TrafficDQN
from model.experiments.quick_reward_search import read_csv, read_json, summarize_episodes
from model.experiments.reward_sensitivity import AuditedEnv, sha256, write_csv, write_json
from model.traffic.route_generator import SCENARIOS
from model.utils.config import ProjectConfig, RewardConfig, SignalConfig, SimulationConfig


STRESS_SCENARIOS = tuple(s for s in SCENARIOS if s != 'random')
SEEDS = [6001, 6002]


def label_for(candidate: int | None) -> str:
    return 'fixed_time' if candidate is None else f'experiment_{candidate}'


def evaluate(output: str, scenario: str, candidate: int | None) -> dict:
    root = Path(output)
    label = label_for(candidate)
    folder = root / 'stress' / scenario / label
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    env = None
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        source = root / f'experiment_{1 if candidate is None else candidate}'
        metadata = read_json(source / 'experiment_config.json')
        sumo_dir = folder / 'sumo'
        sumo_dir.mkdir()
        for name in ('intersection.net.xml', 'simulation.sumocfg'):
            shutil.copy2(source / 'sumo' / name, sumo_dir / name)
        config = replace(
            ProjectConfig(), sumo_dir=sumo_dir, results_dir=folder,
            reward=RewardConfig(**metadata['reward_config']),
            signal=SignalConfig(**metadata['signal_config']),
            simulation=SimulationConfig(**metadata['simulation_config']),
            fixed_green_times={int(k): v for k, v in metadata['fixed_green_times'].items()},
        )
        if sha256(config.network_file) != metadata['network_sha256']:
            raise ValueError('Stress network differs from training network')
        model_path = source / 'final.zip'
        model = None
        if candidate is not None:
            if sha256(model_path) != metadata['model_sha256']:
                raise ValueError('Stress checkpoint differs from saved training checkpoint')
            model = TrafficDQN.load(str(model_path), device='cpu')
            if model.observation_space.shape != (60,) or model.action_space.n != 8:
                raise ValueError('Checkpoint does not use current 60x8 structure')
        episode_seconds = int(metadata['training_options']['episode_seconds'])
        write_json(folder / 'evaluation_config.json', {
            'candidate_id': candidate, 'controller': label, 'scenario': scenario,
            'seeds': SEEDS, 'episode_seconds': episode_seconds,
            'reward_config': metadata['reward_config'],
            'signal_config': metadata['signal_config'],
            'simulation_config': metadata['simulation_config'],
            'fixed_green_times': metadata['fixed_green_times'],
            'network_sha256': metadata['network_sha256'],
            'source_checkpoint': str(model_path) if model is not None else None,
            'source_model_sha256': metadata['model_sha256'] if model is not None else None,
            'deterministic_actions': True, 'used_for_candidate_selection': False,
        })
        write_json(folder / 'status.json', {'status': 'EVALUATING', 'completed_episodes': 0, 'total_episodes': len(SEEDS)})
        env = AuditedEnv(
            config, controller_name='Fixed-Time' if model is None else 'Stress-DQN',
            scenario=scenario, episode_seconds=episode_seconds, cache=True,
            manifest=folder / 'evaluation_routes.json',
        )
        rows = []
        for episode, seed in enumerate(SEEDS):
            observation, _ = env.reset(seed=seed)
            done = False
            while not done:
                action = 0 if model is None else int(model.predict(observation, deterministic=True)[0])
                observation, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            row = env.episode_summary(episode)
            row.update({'candidate_id': candidate, 'controller': label})
            rows.append(row)
            write_csv(folder / 'evaluation_metrics.csv', rows)
            write_json(folder / 'status.json', {
                'status': 'EVALUATING', 'completed_episodes': len(rows),
                'total_episodes': len(SEEDS), 'elapsed_seconds': time.monotonic() - started,
            })
        summary = {'scenario': scenario, 'controller': label, 'candidate_id': candidate,
                   **summarize_episodes(rows)}
        write_json(folder / 'summary.json', summary)
        write_json(folder / 'status.json', {
            'status': 'COMPLETED', 'completed_episodes': len(rows),
            'elapsed_seconds': time.monotonic() - started,
        })
        return summary
    except BaseException as exc:
        write_json(folder / 'status.json', {
            'status': 'FAILED', 'error': repr(exc), 'elapsed_seconds': time.monotonic() - started,
        })
        raise
    finally:
        if env is not None:
            env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Existing quick-search results folder')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.output.resolve()
    if not (root / 'locked_selection.json').is_file():
        parser.error('locked_selection.json must exist before diagnostics run')
    locked = read_json(root / 'locked_selection.json')
    if not locked.get('selection_locked_before_holdout'):
        parser.error('Selection is not locked')
    winner = locked['validation_winner_candidate_id']
    candidates = sorted({1, *([int(winner)] if winner is not None else [])})
    controllers = [*candidates, None]
    settings = read_json(root / 'search_config.json')
    if set(SEEDS) & ({settings['seed']} | set(range(1, 1001)) |
                     set(settings['validation_seeds']) | set(settings['holdout_seeds'])):
        parser.error('Stress seeds overlap training, validation, or holdout seeds')
    destination = root / 'stress'
    if destination.exists():
        parser.error('stress output folder must be fresh')
    destination.mkdir()
    started = time.monotonic()
    tasks = [(scenario, candidate) for scenario in STRESS_SCENARIOS for candidate in controllers]
    write_json(destination / 'stress_config.json', {
        'diagnostic_only': True, 'candidate_reselection_permitted': False,
        'locked_winner_candidate_id': winner, 'baseline_candidate_id': 1,
        'candidate_ids': candidates, 'fixed_time_included': True,
        'scenarios': STRESS_SCENARIOS, 'seeds': SEEDS, 'workers': args.workers,
        'source_sha256': sha256(Path(__file__)),
        'locked_selection_sha256': sha256(root / 'locked_selection.json'),
        'statistical_significance_established': False,
    })
    summaries, failures = [], []
    try:
        write_json(destination / 'status.json', {'status': 'EVALUATING', 'completed_jobs': 0, 'total_jobs': len(tasks)})
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
            pending = {pool.submit(evaluate, str(root), scenario, candidate): (scenario, candidate)
                       for scenario, candidate in tasks}
            for future in as_completed(pending):
                scenario, candidate = pending[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    failures.append({'scenario': scenario, 'candidate_id': candidate, 'error': repr(exc)})
                write_json(destination / 'status.json', {
                    'status': 'EVALUATING', 'completed_jobs': len(summaries),
                    'failed_jobs': len(failures), 'total_jobs': len(tasks),
                    'elapsed_seconds': time.monotonic() - started, 'failures': failures,
                })
        summaries.sort(key=lambda row: (STRESS_SCENARIOS.index(row['scenario']), row['controller']))
        paired = {}
        all_rows = []
        for scenario in STRESS_SCENARIOS:
            scenario_summaries = [s for s in summaries if s['scenario'] == scenario]
            manifests = []
            for summary in scenario_summaries:
                folder = destination / scenario / summary['controller']
                manifests.append(read_json(folder / 'evaluation_routes.json'))
                all_rows.extend(read_csv(folder / 'evaluation_metrics.csv'))
            if manifests and any(manifest != manifests[0] for manifest in manifests[1:]):
                raise RuntimeError(f'Paired stress demand differs: {scenario}')
            paired[scenario] = len(manifests) == len(controllers)
        if summaries:
            write_csv(destination / 'comparison.csv', summaries)
            write_csv(destination / 'evaluation_metrics.csv', all_rows)
        write_json(destination / 'comparison.json', {
            'diagnostic_only': True, 'holdout_reselection_performed': False,
            'paired_route_hashes_verified': all(paired.values()),
            'paired_route_hashes_verified_by_scenario': paired,
            'results': summaries, 'failures': failures,
        })
        write_json(destination / 'status.json', {
            'status': 'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED',
            'completed_jobs': len(summaries), 'total_jobs': len(tasks),
            'elapsed_seconds': time.monotonic() - started, 'failures': failures,
        })
        return 1 if failures else 0
    except BaseException as exc:
        write_json(destination / 'status.json', {
            'status': 'FAILED', 'error': repr(exc),
            'elapsed_seconds': time.monotonic() - started, 'failures': failures,
        })
        raise


if __name__ == '__main__':
    raise SystemExit(main())
