"""Evaluate the four preserved, paired 25,000-step sensitivity checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from model.controller.traffic_dqn import TrafficDQN
from model.experiments.reward_sensitivity import (
    AuditedEnv, METRICS, WEIGHTS, aggregate, sha256, write_csv, write_json)
from model.utils.config import ProjectConfig, RewardConfig


def evaluate(experiment: int, source: str, output: str) -> dict:
    started = time.monotonic()
    source_dir = Path(source)
    folder = Path(output) / f'experiment_{experiment}'
    folder.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source_dir / f'experiment_{experiment}' /
                           'experiment_config.json').read_text(encoding='utf-8'))
    checkpoint = source_dir / 'checkpoints_25000' / f'experiment_{experiment}.zip'
    shutil.copy2(checkpoint, folder / 'final.zip')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = TrafficDQN.load(str(folder / 'final.zip'), device='cpu')
    assert model.num_timesteps == 25000, model.num_timesteps
    assert model.observation_space.shape == (60,) and model.action_space.n == 8
    reward = RewardConfig(**metadata['reward_config'])
    assert (reward.waiting_weight, reward.max_waiting_weight,
            reward.switch_penalty) == WEIGHTS[experiment]
    sumo_dir = folder / 'sumo'
    sumo_dir.mkdir(exist_ok=True)
    for name in ('intersection.net.xml', 'simulation.sumocfg'):
        shutil.copy2(source_dir / f'experiment_{experiment}' / 'sumo' / name,
                     sumo_dir / name)
    config = replace(ProjectConfig(), sumo_dir=sumo_dir, results_dir=folder,
                     reward=reward)
    assert sha256(config.network_file) == metadata['network_sha256']
    training_routes = json.loads((source_dir / f'experiment_{experiment}' /
                                  'training_routes.json').read_text(encoding='utf-8'))
    write_json(folder / 'training_routes.json', training_routes[:84])
    metadata.update({
        'completed_training_steps': 25000,
        'checkpoint_selection': 'preserved checkpoint after 25000 decisions',
        'original_planned_training_steps': 150000,
        'original_learning_schedule_preserved': True,
        'model_sha256': sha256(folder / 'final.zip'),
        'evaluation_source_sha256': sha256(Path(__file__)),
        'evaluation_episodes': 30, 'evaluation_seed_start': 2001,
    })
    write_json(folder / 'experiment_config.json', metadata)
    rows: list[dict] = []
    env = AuditedEnv(config, controller_name='Sensitivity-DQN',
                     scenario='random', episode_seconds=300,
                     manifest=folder / 'evaluation_routes.json')
    try:
        for episode, seed in enumerate(range(2001, 2031)):
            observation, _ = env.reset(seed=seed)
            done = False
            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
            row = env.episode_summary(episode)
            row.update({'experiment': experiment})
            rows.append(row)
            write_csv(folder / 'evaluation_metrics.csv', rows)
            write_json(folder / 'progress.json', {
                'status': 'EVALUATING', 'evaluation_completed': len(rows),
                'evaluation_total': 30, 'training_steps': 25000,
                'elapsed_seconds': time.monotonic() - started,
            })
    except BaseException as exc:
        write_json(folder / 'progress.json', {'status': 'FAILED', 'error': repr(exc)})
        raise
    finally:
        env.close()
    alpha, beta, switch = WEIGHTS[experiment]
    summary = {'experiment': experiment, 'alpha': alpha, 'beta': beta,
               'switch_gamma': switch, 'training_steps': 25000,
               'training_seed': 22, 'scenario': 'random',
               'evaluation_episodes': 30}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        summary[metric] = float(values.mean())
        summary[metric + '_std'] = float(values.std(ddof=1))
    summary['maximum_waiting_worst_episode'] = max(row['max_waiting_time'] for row in rows)
    summary['elapsed_seconds'] = time.monotonic() - started
    write_json(folder / 'summary.json', summary)
    write_json(folder / 'progress.json', {'status': 'COMPLETED', **summary})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('output/reward-sensitivity-2026-10-03'))
    parser.add_argument('--output', type=Path, default=Path('output/reward-sensitivity-25000-2026-10-03'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(evaluate, i, str(args.source.resolve()),
                               str(args.output.resolve())) for i in WEIGHTS]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)
    summaries = aggregate(args.output, list(WEIGHTS))
    comparison_path = args.output / 'comparison.md'
    original = comparison_path.read_text(encoding='utf-8')
    context = ('Training seed 22; random traffic; preserved 25,000-step checkpoints '
               'from a planned 150,000-step schedule. Evaluation seeds 2001–2030, '
               '30 paired 300-second episodes per model, deterministic actions. '
               'These are preliminary results.\n\n')
    comparison_path.write_text(context + original, encoding='utf-8')
    print(json.dumps({'results': summaries}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
