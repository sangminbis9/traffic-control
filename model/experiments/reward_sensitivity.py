"""Train four reward-weight combinations and evaluate paired traffic seeds.

Run: python -m model.experiments.reward_sensitivity --workers 4
Results are isolated from the project's canonical checkpoint.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
import hashlib
import json
import random
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import torch

from model.controller.traffic_dqn import TrafficDQN
from model.env.intersection_env import IntersectionEnv
from model.env.state_provider import SUMOTrafficStateProvider
from model.experiments.subscription_connection import SubscriptionConnection
from model.training import TrainingOptions, TrainingRunner
from model.utils.config import ProjectConfig
from model.utils.reproducibility import collect_reproducibility_metadata


WEIGHTS = {1: (0.05, 0.05, 0.1), 2: (0.1, 0.1, 0.2),
           3: (0.2, 0.2, 0.3), 4: (0.3, 0.3, 0.5)}
METRICS = ('avg_waiting_time', 'max_waiting_time', 'avg_queue',
           'phase_changes', 'throughput')


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    payload = json.dumps(value, indent=2, ensure_ascii=False, default=str)
    for attempt in range(10):
        try:
            temporary.write_text(payload, encoding='utf-8')
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuditedEnv(IntersectionEnv):
    def __init__(self, *args, manifest: Path, cache: bool = True,
                 verify_cache: bool = False, **kwargs):
        self.manifest = manifest
        self.cache = cache
        self.verify_cache = verify_cache
        self.route_sequence: list[dict] = []
        super().__init__(*args, **kwargs)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        if self.cache:
            raw_connection = self.connection
            self._connection = SubscriptionConnection(raw_connection)
            self._state_provider.connection = self._connection
            if self.verify_cache:
                provider = self._state_provider
                raw_provider = SUMOTrafficStateProvider(
                    raw_connection, provider.queue_scale, provider.waiting_scale,
                    provider.max_waiting_scale, provider.detection_distance,
                    provider.approaching_scale)
                cached_snapshot = provider.snapshot

                def checked_snapshot():
                    cached, direct = cached_snapshot(), raw_provider.snapshot()
                    if cached != direct:
                        raise AssertionError(f'TraCI cache differs at {raw_connection.simulation.getTime()}')
                    return cached

                provider.snapshot = checked_snapshot
        self.route_sequence.append({
            'episode': len(self.route_sequence), 'seed': info['seed'],
            'scenario': info['scenario'],
            'route_sha256': sha256(Path(info['route_file'])),
        })
        write_json(self.manifest, self.route_sequence)
        return observation, info


class Progress(BaseCallback):
    def __init__(self, folder: Path, total: int, verify_prefix: str | None = None):
        super().__init__()
        self.folder, self.total = folder, total
        self.started = time.monotonic()
        self.rows: list[dict] = []
        self.verify_prefix = verify_prefix

    def _on_step(self) -> bool:
        step = self.num_timesteps
        if step == 1 or step % 1000 == 0 or step == self.total:
            elapsed = time.monotonic() - self.started
            row = {'status': 'TRAINING', 'timesteps': step,
                   'total_timesteps': self.total, 'elapsed_seconds': elapsed,
                   'steps_per_second': step / max(elapsed, 1e-9)}
            self.rows.append(row)
            write_json(self.folder / 'progress.json', row)
            write_csv(self.folder / 'training_progress.csv', self.rows)
        if step % 25000 == 0:
            self.model.save(str(self.folder / 'latest'))
            shutil.copy2(self.folder / 'latest.zip', self.folder / f'checkpoint_{step}.zip')
        if step == 25000 and self.verify_prefix:
            reference_folder = Path(self.verify_prefix) / self.folder.name
            # SB3 loading seeds/initializes networks; preserve the training RNGs.
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            torch_rng = torch.get_rng_state()
            try:
                reference = TrafficDQN.load(str(reference_folder / 'final.zip'), device='cpu')
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(torch_rng)
            for name in ('q_net', 'q_net_target'):
                current = getattr(self.model, name).state_dict()
                previous = getattr(reference, name).state_dict()
                if current.keys() != previous.keys() or any(
                        not torch.equal(current[key], previous[key]) for key in current):
                    raise RuntimeError(f'25k reproduction differs: {name}')
            for name in ('num_timesteps', '_n_updates', 'exploration_rate', '_hold_remaining'):
                if getattr(self.model, name) != getattr(reference, name):
                    raise RuntimeError(f'25k reproduction differs: {name}')
            if not np.array_equal(self.model._held_action, reference._held_action):
                raise RuntimeError('25k reproduction differs: held action')
            routes = json.loads((self.folder / 'training_routes.json').read_text(encoding='utf-8'))
            previous_routes = json.loads((reference_folder / 'training_routes.json').read_text(encoding='utf-8'))
            if routes != previous_routes:
                raise RuntimeError('25k reproduction differs: traffic routes')
            write_json(self.folder / 'prefix_verification.json', {
                'verified': True, 'training_steps': step,
                'reference': str(reference_folder / 'final.zip'),
                'q_networks_exactly_equal': True, 'training_routes_exactly_equal': True,
                'updates_exploration_and_held_action_equal': True,
            })
        return step < self.total


def worker(experiment: int, settings: dict) -> dict:
    folder = Path(settings['output']) / f'experiment_{experiment}'
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        base = ProjectConfig()
        sumo_folder = folder / 'sumo'
        sumo_folder.mkdir(exist_ok=True)
        for filename in ('intersection.net.xml', 'simulation.sumocfg'):
            shutil.copy2(base.sumo_dir / filename, sumo_folder / filename)
        alpha, beta, switch = settings.get('weights', WEIGHTS.get(experiment))
        config = replace(base, sumo_dir=sumo_folder, results_dir=folder,
                         reward=replace(base.reward, waiting_weight=alpha,
                                        max_waiting_weight=beta,
                                        switch_penalty=switch))
        options = TrainingOptions(total_steps=settings['timesteps'],
                                  seed=settings['seed'],
                                  scenario=settings['scenario'],
                                  episode_seconds=settings['episode_seconds'])
        metadata = collect_reproducibility_metadata()
        metadata.update({
            'experiment': experiment, 'training_options': asdict(options),
            'reward_config': asdict(config.reward),
            'network_sha256': sha256(config.network_file),
            'evaluation_seed_start': settings['eval_seed_start'],
            'evaluation_episodes': settings['eval_episodes'],
            'checkpoint_selection': 'final fixed-step checkpoint',
            'deterministic_algorithms': True,
            'traci_subscriptions': settings['cache'],
            'cache_equivalence_check': settings['verify_cache'],
            'experiment_source_sha256': sha256(Path(__file__)),
            'subscription_source_sha256': sha256(Path(__file__).with_name('subscription_connection.py')),
            'learning_schedule_steps': settings.get('schedule_timesteps') or settings['timesteps'],
            'reproduced_from_scratch': True,
            'prefix_verification_source': settings.get('verify_prefix'),
        })
        write_json(folder / 'experiment_config.json', metadata)
        env = AuditedEnv(config, controller_name='Sensitivity-DQN',
                         scenario=settings['scenario'], seed=settings['seed'],
                         episode_seconds=settings['episode_seconds'],
                         cache=settings['cache'], verify_cache=settings['verify_cache'],
                         manifest=folder / 'training_routes.json')
        try:
            runner = TrainingRunner(f'sensitivity_{experiment}', options,
                                    output_dir=folder)
            runner.config = config
            model = runner._create_model(env)
            model.learn(total_timesteps=settings.get('schedule_timesteps') or settings['timesteps'],
                        callback=Progress(folder, settings['timesteps'], settings.get('verify_prefix')),
                        progress_bar=False)
            if model.num_timesteps != settings['timesteps']:
                raise RuntimeError('Unexpected training step count')
            model.save(str(folder / 'final'))
            model.save_replay_buffer(str(folder / 'final.replay.pkl'))
        finally:
            env.close()
        # Reload the saved model, so the result belongs to a usable checkpoint.
        model = TrafficDQN.load(str(folder / 'final.zip'), device='cpu')
        if model.observation_space.shape != (60,) or model.action_space.n != 8:
            raise RuntimeError('Saved checkpoint does not use current 60x8 structure')
        rows: list[dict] = []
        eval_env = AuditedEnv(config, controller_name='Sensitivity-DQN',
                              scenario=settings['scenario'],
                              episode_seconds=settings['episode_seconds'],
                              cache=settings['cache'], verify_cache=settings['verify_cache'],
                              manifest=folder / 'evaluation_routes.json')
        try:
            for episode in range(settings['eval_episodes']):
                seed = settings['eval_seed_start'] + episode
                write_json(folder / 'progress.json', {
                    'status': 'EVALUATING', 'timesteps': model.num_timesteps,
                    'total_timesteps': settings['timesteps'],
                    'evaluation_completed': episode,
                    'evaluation_total': settings['eval_episodes'],
                    'elapsed_seconds': time.monotonic() - started,
                })
                observation, _ = eval_env.reset(seed=seed)
                done = False
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, _ = eval_env.step(int(action))
                    done = terminated or truncated
                row = eval_env.episode_summary(episode)
                row.update({'experiment': experiment, 'alpha': alpha,
                            'beta': beta, 'switch_gamma': switch})
                rows.append(row)
                write_csv(folder / 'evaluation_metrics.csv', rows)
        finally:
            eval_env.close()
        summary = {'experiment': experiment, 'alpha': alpha, 'beta': beta,
                   'switch_gamma': switch, 'training_steps': model.num_timesteps,
                   'training_seed': settings['seed'],
                   'scenario': settings['scenario'],
                   'evaluation_episodes': len(rows)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            summary[metric] = float(values.mean())
            summary[metric + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary['maximum_waiting_worst_episode'] = max(row['max_waiting_time'] for row in rows)
        summary['elapsed_seconds'] = time.monotonic() - started
        metadata.update({'model_sha256': sha256(folder / 'final.zip'),
                         'completed_training_steps': model.num_timesteps})
        write_json(folder / 'experiment_config.json', metadata)
        write_json(folder / 'summary.json', summary)
        write_json(folder / 'progress.json', {'status': 'COMPLETED', **summary})
        return summary
    except BaseException as exc:
        traceback.print_exc()
        write_json(folder / 'progress.json', {'status': 'FAILED', 'error': repr(exc),
                                             'elapsed_seconds': time.monotonic() - started})
        raise


def aggregate(output: Path, experiments: list[int]) -> list[dict]:
    summaries = [json.loads((output / f'experiment_{i}' / 'summary.json').read_text(
        encoding='utf-8')) for i in experiments]
    for filename in ('training_routes.json', 'evaluation_routes.json'):
        manifests = [json.loads((output / f'experiment_{i}' / filename).read_text(
            encoding='utf-8')) for i in experiments]
        if any(manifest != manifests[0] for manifest in manifests[1:]):
            raise RuntimeError(f'Paired demand verification failed: {filename}')
    summaries.sort(key=lambda row: row['experiment'])
    write_csv(output / 'comparison.csv', summaries)
    write_json(output / 'comparison.json', {'paired_route_hashes_verified': True,
                                            'results': summaries})
    lines = [
        '# Reward weight comparison', '',
        '| Experiment | alpha | beta | switch gamma | Average Waiting (s) | Maximum Waiting (s) | Average Queue (vehicles) | Phase Changes | Throughput (vehicles/episode) |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for row in summaries:
        fields = [str(row['experiment']), str(row['alpha']), str(row['beta']),
                  str(row['switch_gamma'])]
        fields += [f"{row[m]:.2f} ± {row[m + '_std']:.2f}" for m in METRICS]
        lines.append('| ' + ' | '.join(fields) + ' |')
    lines += ['', 'Values are evaluation-episode means ± sample standard deviations.',
              'Maximum Waiting is the mean of each episode\'s maximum.',
              'Average Waiting includes all departed vehicles, including unfinished vehicles at the 300-second cutoff; pending vehicles before departure are excluded.',
              'Average Queue is the time-weighted sum of stopped vehicles across the eight movement groups.',
              'All models use one common training seed; this does not estimate variability across training seeds.',
              'Three weights vary jointly. Results compare combinations rather than isolated coefficient effects.',
              'Training/evaluation route seed sequences and SHA-256 hashes match across experiments.']
    (output / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=150000)
    parser.add_argument('--schedule-timesteps', type=int,
                        help='Keep the original exploration schedule while stopping at --timesteps')
    parser.add_argument('--verify-prefix', type=str,
                        help='Folder containing the previously evaluated 25000-step models')
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--scenario', default='random')
    parser.add_argument('--episode-seconds', type=int, default=300)
    parser.add_argument('--eval-seed-start', type=int, default=2001)
    parser.add_argument('--eval-episodes', type=int, default=30)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--no-cache', action='store_false', dest='cache', default=True)
    parser.add_argument('--verify-cache', action='store_true')
    parser.add_argument('--only', type=int, nargs='+', choices=list(WEIGHTS), default=list(WEIGHTS))
    parser.add_argument('--output', type=Path, default=Path('output/reward-sensitivity-2026-10-03'))
    args = parser.parse_args()
    if args.schedule_timesteps is not None and args.schedule_timesteps < args.timesteps:
        parser.error('--schedule-timesteps must be at least --timesteps')
    settings = vars(args).copy()
    settings['output'] = str(args.output.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'run_config.json', settings)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(args.only))) as pool:
        pending = {pool.submit(worker, experiment, settings): experiment for experiment in args.only}
        for future in as_completed(pending):
            print(json.dumps({'completed': pending[future], 'summary': future.result()}), flush=True)
    print(json.dumps({'results': aggregate(args.output, args.only)}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
