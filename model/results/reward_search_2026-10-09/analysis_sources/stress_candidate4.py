"""Supplementary diagnostics for the candidate selected for fresh confirmation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from stress_reward_finalists import evaluate, STRESS_SCENARIOS, read_json, write_json, write_csv


if __name__ == '__main__':
    root = Path('model/results/reward_sensitivity_local/2026-10-09-quick').resolve()
    output = root / 'stress_candidate4'
    output.mkdir(exist_ok=False)
    write_json(output / 'config.json', {'candidate': 4, 'diagnostic_only': True,
               'scenarios': STRESS_SCENARIOS, 'seeds': [6001, 6002],
               'used_for_candidate_selection': False})
    write_json(output / 'status.json', {'status': 'EVALUATING'})
    try:
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(evaluate, str(root), scenario, 4) for scenario in STRESS_SCENARIOS]
            additional = [future.result() for future in futures]
        previous = read_json(root / 'stress' / 'comparison.json')['results']
        summaries = [row for row in previous if row['candidate_id'] in (1, None)] + additional
        for scenario in STRESS_SCENARIOS:
            manifests = [read_json(root / 'stress' / scenario / label / 'evaluation_routes.json')
                         for label in ('experiment_1', 'experiment_4', 'fixed_time')]
            if any(manifest != manifests[0] for manifest in manifests[1:]):
                raise RuntimeError('Unpaired diagnostic demand')
        summaries.sort(key=lambda row: (row['scenario'], row['controller']))
        write_csv(output / 'comparison.csv', summaries)
        write_json(output / 'comparison.json', {'diagnostic_only': True, 'paired_route_hashes_verified': True,
                   'results': summaries})
        write_json(output / 'status.json', {'status': 'COMPLETED'})
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'FAILED', 'error': repr(exc)})
        raise
