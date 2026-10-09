# Reward weight sensitivity experiment

The project adopted queue/waiting/maximum-waiting/switching weights
`1.0 / 0.3 / 0.5 / 0.5` on 2026-10-10. See the
[fixed baseline decision](../README.md#fixed-baseline-decision) for the evidence,
limits, future comparison protocol, and checkpoint compatibility rules.
The experiments below retain their historical explicit candidate weights.
Their former default-weight candidates must not be relabeled as current defaults.
The complete fine-search and paired-comparison source/evidence snapshot is commit
`4b52089d662a6dd5e019040b6fd9a84f51053097`, before adopting the new defaults.

The four experiments keep the queue coefficient at 1.0 and change only
the waiting, maximum waiting, and actual switching coefficients:

| Experiment | Waiting alpha | Maximum waiting beta | Switching coefficient |
|---|---|---|---|
| 1 | 0.05 | 0.05 | 0.1 |
| 2 | 0.1 | 0.1 | 0.2 |
| 3 | 0.2 | 0.2 | 0.3 |
| 4 | 0.3 | 0.3 | 0.5 |

```powershell
python -m model.experiments.reward_sensitivity --timesteps 150000 --seed 22 --scenario random --eval-seed-start 2001 --eval-episodes 30 --workers 4 --output output\reward-sensitivity-2026-10-03
```

Each model starts from scratch using the current 60-input/8-action model,
the project's original `TrainingRunner._create_model()` construction, and
one environment per process. Workers use separate SUMO and output folders.
They do not overwrite `model/results/dqn_intersection.zip`.

All models use the final checkpoint after exactly the requested step count.
There is no early stopping or choice of checkpoint from evaluation results.
All evaluation runs use deterministic actions, the same SUMO network,
300-second episodes, 240 seconds of generated demand, and paired evaluation
seeds 2001 through 2030. Training and evaluation demand manifests record
the episode seed and route-file SHA-256. The comparison is emitted only if
all four manifest sequences match exactly.

SUMO still runs through TraCI. Subscriptions cache vehicle and lane readings
for the current simulation tick, reducing round trips. The original state
provider, signal controller, environment step, reward, and metric calculations
remain in use. `--no-cache` uses ordinary getters. `--verify-cache` compares
every snapshot with ordinary getters and fails on any difference.

Verification performed before the full experiment:

- A 1,000-step/one-evaluation-episode ordinary run and cached run produced
  identical values for all five metrics and identical demand manifests.
- A 300-step training run and one 300-second evaluation episode with
  `--verify-cache` compared every post-tick snapshot exactly.
- These short runs are infrastructure checks; they are not sensitivity results.

Outputs in each experiment folder:

- `final.zip`: the new model.
- `experiment_config.json`: weights, runtime settings, code/network/model hashes.
- `training_routes.json` and `evaluation_routes.json`: paired demand evidence.
- `training_progress.csv` and `progress.json`: progress.
- `evaluation_metrics.csv`: all individual evaluation episodes.
- `summary.json`: metric means and sample standard deviations.

The parent folder contains `comparison.csv`, `comparison.json`, and
`comparison.md` after all requested experiments finish.

## Selected preliminary comparison

The user chose a quicker preliminary comparison using the four saved
25,000-step checkpoints. The 150,000-step run was stopped; its final results
were not produced. The preserved checkpoints are stored under that run's
`checkpoints_25000` directory. To reproduce their evaluation:

```powershell
python -m model.experiments.evaluate_sensitivity_checkpoints
```

This produces `output/reward-sensitivity-25000-2026-10-03`. It uses exactly
25,000-step checkpoints from the originally planned 150,000-step learning
schedule, rather than training fresh models with a compressed 25,000-step
exploration schedule. Each experiment's first 84 training route manifests
are retained and checked; every evaluation uses the same 30 traffic seeds.

## 50,000-step comparison using the original schedule

The original 25,000-step ZIPs have no replay buffers, so a faithful direct
resume is unavailable. Reproduce training from scratch with the same seed
and the original 150,000-step exploration schedule, stopping at 50,000:

```powershell
python -m model.experiments.reward_sensitivity --timesteps 50000 --schedule-timesteps 150000 --verify-prefix output/reward-sensitivity-25000-2026-10-03 --seed 22 --scenario random --eval-seed-start 2001 --eval-episodes 30 --workers 4 --output output/reward-sensitivity-50000-2026-10-04
```

At 25,000 steps the callback verifies exact equality of both Q networks,
update count, exploration rate, held exploratory action, and training routes
against the previously evaluated checkpoints. A mismatch fails the run.
Each 25,000-step checkpoint is preserved separately. Evaluation uses the
same 30 traffic scenarios as the earlier comparison.
The reference-model load preserves Python, NumPy, and Torch RNG states so
verification does not perturb subsequent training. A nonsynchronized local
`--output` folder can avoid file locks from cloud synchronization.

After aggregation, generate the Korean report and changes from 25,000 steps:

```powershell
python -m model.experiments.summarize_sensitivity_steps --current output/reward-sensitivity-50000-2026-10-04
```

## Quick bounded search

The 2026-10-09 run starts six independent current 60-input/8-action models
from scratch, with queue weight 1.0 and `(waiting, maximum waiting, switching)`
weights `(0.1, 0.1, 0.2)`, `(0.3, 0.3, 0.5)`, `(0.5, 0.3, 0.5)`,
`(0.3, 0.5, 0.5)`, `(0.3, 0.3, 0.2)`, and `(0.3, 0.3, 0.8)`.
Candidate 1 is the fresh default-weight baseline. Production defaults and
the canonical checkpoint are not replaced.

```powershell
python -m model.experiments.quick_reward_search --output model/results/reward_sensitivity_local/2026-10-09-quick --workers 6 --timesteps 50000 --eval-episodes 8 --holdout-episodes 10
```

The output directory must not already exist; choose a new name for a rerun.
Each candidate uses training seed 22, the original 150,000-step exploration
schedule, random traffic, 300-second episodes, and 240 seconds of demand.
Validation uses seeds 4001–4008. Candidates with a safety event or less than
95% of baseline throughput are ineligible; the others are ranked by average
waiting, maximum waiting, queue, phase changes, and then throughput.

`locked_selection.json` saves the winner and top two eligible candidates
before holdout evaluation. That shortlist plus candidate 1 and Fixed-Time
are evaluated on paired seeds 5001–5010. Holdout checks the locked winner's
safety, throughput retention, and average waiting against candidate 1;
it never selects a different candidate from the test results.

Outputs include `comparison.csv`, `validation_diagnostics.csv`,
`holdout_comparison.csv`, `recommendation.json`, and `status.json`.
Each model and evaluation folder preserves episode metrics and demand hashes;
diagnostics include pending and unfinished vehicles. The 2026-10-09 run's
`runtime_packages.json` records its actual Python/package versions and
TraCI/SUMO library locations; per-candidate metadata records the SUMO version.

Five focused selection/holdout tests passed, and compilation and CLI help
were checked before starting the run. These checks do not establish training
performance. This is a single-training-seed, random-traffic search: holdout
confirmation is not a significance test, cross-scenario validation, or proof
of a global optimum.

### Archived 2026-10-09 results

The [versioned result snapshot](../results/reward_search_2026-10-09/README.md)
includes all six final models and evaluation evidence. The first validation
winner (candidate 6) failed its initial holdout. An exploratory follow-up fixed
candidate 4 before a new test on seeds 7001–7020; it reduced mean waiting by
4.97% versus a fresh default-weight 50k model. The original failed recommendation
is preserved separately from the final result. See the snapshot for the full
selection history, heavy-traffic limitations, and recorded runtime versions.

## Metric definitions

- Average Waiting: mean observed accumulated waiting over all vehicles that
  departed into the simulation, including vehicles unfinished at the cutoff.
  Vehicles still waiting to enter SUMO are excluded by the project's metric.
- Maximum Waiting: per-episode maximum observed accumulated waiting over the
  same vehicles. The comparison shows the mean of these episode maxima.
  `maximum_waiting_worst_episode` separately records the worst episode.
- Average Queue: time-weighted mean total halted vehicle count across all
  eight incoming movement groups.
- Phase Changes: actual green-phase transitions, including forced transitions.
- Throughput: vehicles arriving at their destinations within an episode.

Means and sample standard deviations describe variability over evaluation traffic
seeds (30 in the original sensitivity comparison). One common training seed
does not characterize variability across model training seeds. In the original
four-experiment comparison, all three coefficients vary together, so it compares
combinations rather than isolating a single coefficient's effect. The switching
coefficient is distinct from DQN's discount gamma 0.95.
