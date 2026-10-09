"""Independent confirmation of candidate 4 after exploratory first-holdout evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from model.experiments.quick_reward_search import (
    confirm_locked_winner, evaluate_holdout, read_csv, read_json, verify_holdout_pairing,
)
from model.experiments.reward_sensitivity import sha256, write_csv, write_json

SEEDS = list(range(7001, 7021))


def evaluate(output: str, candidate: int | None) -> dict:
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return evaluate_holdout(Path(output), candidate, SEEDS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    prior = read_json(root / 'recommendation.json')
    if prior['validation_winner_candidate_id'] != 6 or prior['holdout_confirmation']['confirmed']:
        parser.error('Expected original failed candidate-6 confirmation')
    folder = root / 'followup_confirmation'
    if folder.exists():
        parser.error('followup_confirmation must be a fresh folder')
    folder.mkdir()
    started = time.monotonic()
    locked = {'validation_winner_candidate_id': 4, 'selection_locked_before_holdout': True}
    write_json(folder / 'followup_config.json', {
        'stage': 'exploratory_followup_with_independent_final_test',
        'selection_evidence': 'Original validation and original holdout 5001-5010 informed fixed candidate 4; they are no longer final-test evidence for this new stage.',
        'candidate_id_locked_before_new_test': 4, 'baseline_candidate_id': 1,
        'weights': [0.3, 0.5, 0.5], 'queue_weight': 1.0,
        'fresh_final_test_seeds': SEEDS, 'scenario': 'random', 'episode_seconds': 300,
        'further_reselection_permitted': False, 'training_seeds': [22],
        'training_steps_per_candidate': 50000, 'exploration_schedule_steps': 150000,
        'original_recommendation_preserved': True,
        'original_recommendation_sha256': sha256(root / 'recommendation.json'),
        'source_sha256': sha256(Path(__file__)),
        'statistical_significance_established': False, 'global_optimum_established': False,
    })
    for candidate in (1, 4):
        source = root / f'experiment_{candidate}'
        target = folder / f'experiment_{candidate}'
        (target / 'sumo').mkdir(parents=True)
        for name in ('experiment_config.json', 'final.zip'):
            shutil.copy2(source / name, target / name)
        for name in ('intersection.net.xml', 'simulation.sumocfg'):
            shutil.copy2(source / 'sumo' / name, target / 'sumo' / name)
    if (root / 'runtime_packages.json').is_file():
        shutil.copy2(root / 'runtime_packages.json', folder / 'runtime_packages.json')
    summaries, failures = [], []
    try:
        write_json(folder / 'status.json', {'status': 'EVALUATING', 'completed_jobs': 0, 'total_jobs': 3})
        with ProcessPoolExecutor(max_workers=3) as pool:
            pending = {pool.submit(evaluate, str(folder), candidate): candidate for candidate in (1, 4, None)}
            for future in as_completed(pending):
                candidate = pending[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    failures.append({'candidate_id': candidate, 'error': repr(exc)})
                write_json(folder / 'status.json', {
                    'status': 'EVALUATING', 'completed_jobs': len(summaries), 'total_jobs': 3,
                    'failures': failures, 'elapsed_seconds': time.monotonic() - started,
                })
        verify_holdout_pairing(folder, summaries)
        summaries.sort(key=lambda row: str(row['experiment']))
        if summaries:
            write_csv(folder / 'comparison.csv', summaries)
        write_json(folder / 'comparison.json', {
            'paired_route_hashes_verified': len(summaries) == 3,
            'results': summaries, 'failures': failures,
        })
        candidate_summaries = [s for s in summaries if s['experiment'] != 'fixed_time']
        rows = {int(s['experiment']): read_csv(folder / 'holdout' / f"experiment_{s['experiment']}" / 'evaluation_metrics.csv')
                for s in candidate_summaries}
        confirmation = confirm_locked_winner(locked, candidate_summaries, rows)
        recommendation = {
            'stage': 'exploratory_followup_with_independent_final_test',
            'candidate_id_locked_before_new_test': 4, 'candidate_weights': [0.3, 0.5, 0.5],
            'confirmed': confirmation['confirmed'], 'confirmation': confirmation,
            'recommended_candidate_id': 4 if confirmation['confirmed'] else None,
            'recommended_weights': [0.3, 0.5, 0.5] if confirmation['confirmed'] else None,
            'model_path': str(root / 'experiment_4' / 'final.zip'),
            'fresh_final_test_seeds': SEEDS, 'further_reselection_performed': False,
            'original_recommendation_preserved': True, 'default_configuration_changed': False,
            'training_seeds': [22], 'training_steps_per_candidate': 50000,
            'statistical_significance_established': False, 'global_optimum_established': False,
            'failures': failures,
        }
        write_json(folder / 'recommendation.json', recommendation)
        write_json(folder / 'status.json', {
            'status': 'COMPLETED_WITH_FAILURES' if failures else 'COMPLETED',
            'confirmed': confirmation['confirmed'], 'elapsed_seconds': time.monotonic() - started,
            'failures': failures,
        })
        print(recommendation, flush=True)
        return 1 if failures else 0
    except BaseException as exc:
        write_json(folder / 'status.json', {
            'status': 'FAILED', 'error': repr(exc), 'elapsed_seconds': time.monotonic() - started,
        })
        raise


if __name__ == '__main__':
    raise SystemExit(main())
