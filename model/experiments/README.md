# Reward weight sensitivity experiment

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

Means and sample standard deviations describe variability over the 30 traffic
seeds. One common training seed does not characterize variability across model
training seeds. All three coefficients vary together, so the comparison
evaluates weight combinations and does not isolate a single coefficient's
effect. The switching coefficient is distinct from DQN's discount gamma 0.95.
