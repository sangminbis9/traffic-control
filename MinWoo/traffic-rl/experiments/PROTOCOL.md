# Multi-seed learning experiment (2026-09-18)

This protocol is written before inspecting new validation or final test results.

- Preserve the existing network, 300-second episode, 240-second demand, signal clearance/minimum/maximum durations and Fixed-Time [10,4,10,4] baseline.
- First configuration: unchanged default DQN, observation and reward.
- Independent training RNG seeds: 11, 22, 33. Training traffic seeds: 1–1000.
- Trajectory length: 300,000 decisions per seed. Save at 20k, 50k, 100k, 200k, 300k. Exploration decays on the full 300k trajectory; checkpoints are not independently restarted short-budget runs.
- Selection/validation traffic seeds: 3001–3005, all six scenarios. These may be reused for model selection and subsequent justified configuration changes.
- Previously reported 2001–2005 results are development evidence, not new final tests.
- Reserved final traffic seeds: 4001–4030, all six scenarios. Do not inspect until freezing the selected configuration and checkpoint rule.
- Compare identical route hashes and SUMO seeds for Fixed/DQN; use equal scenario weights.
- Success target: lower mean waiting and queue with no lower aggregate throughput on final tests; require improvement across multiple training seeds, not only one lucky model. Report all attempted models, per-scenario results, paired differences and confidence intervals. A target 5% waiting improvement provides a useful practical margin; confidence intervals quantify uncertainty.
- If final results fail, report failure and continue using validation. Any later final test must use a newly declared unused seed range; never relabel previously inspected final tests as untouched.
- No deletion of unfavorable results, baseline weakening, score-definition changes or choosing only favorable traffic scenarios.
- Use libsumo for speed only after its trajectory is shown equal to TraCI. Keep actual TraCI reproduction for a selected model.

## Validation-driven addition: switching cost (before final tests)

At 50k, all three default seeds switched about 56–59 times/episode, versus 33 for Fixed. The first 100k validation checkpoint still underperformed. Action traces show minimum-green changes while recently queued vehicles are still moving toward the stop line.

Add a variant with only switching penalty increased from 0.2 to 1.0. All observations, physical settings, training traffic and other reward weights stay identical. Seeds remain 11/22/33; save 20k/50k/100k/200k. Train to 200k, with exploration_fraction=0.45 so epsilon decays over exactly the same first 90k steps as the 300k default runs (0.3 × 300k = 0.45 × 200k). This preserves exploration schedule at matched budgets. Select on 3001–3005 only. Final 4001–4030 is still uninspected.

## Additional pilot: delayed-action credit

The transition consumes two seconds before a requested new green can discharge vehicles. Add an SB3-supported 5-step-return DQN pilot with switching cost 1.0, otherwise the same settings and 90k epsilon schedule, training seed 11 and 200k budget. This is a training algorithm setting, not a new signal or baseline. If validation is promising, replicate with seeds 22 and 33 before final testing. Preserve the unsuccessful one-step runs as comparisons.

## Queue-focused pilot (still validation only)

A separately labeled queue-only heuristic achieved 20.15% lower waiting and 11.73% lower queue with virtually equal throughput on validation. It is diagnostic evidence that the existing observations allow improvements, not a DQN result or a teacher used by DQN.

Add one pure DQN pilot: queue scale 10 rather than 40 (stronger queue features and queue reward), total/max waiting weights 0.1/0.1 (initial maximum-wait cost dominated reward), switching 0.2 unchanged, gamma 0.95, n_steps 5, learning rate 3e-4. Budget 100k, epsilon decays for first 30k. This is a combined hyperparameter candidate, not an isolated causal ablation. If promising, replicate seeds 22/33. No heuristic actions or demonstrations enter DQN training or inference. Physical signal rules, vehicles, demand and Fixed remain unchanged.

## Stopping default runs and exploring green duration

All three default 200k checkpoints underperformed Fixed: waiting increased approximately 27–55% and throughput fell 22–28%. Stop the default workers after preserving all 20k/50k/100k/200k checkpoints; unsaved continuation had reached roughly 210k–230k. Cancel the planned 300k checkpoints for this family and reallocate compute. This is a validation-based futility decision, not a claim that further training could never work.

Independent uniformly random phase requests switch with probability 3/4 every eligible second, so exploration rarely observes long green holds. Add `TrafficDQN`, whose Bellman training and greedy inference remain SB3 DQN, but exploration holds a sampled action for 5–17 decision steps. Safety controller still enforces all timing limits. Use queue-focused configuration, 100k budget and seeds 11/22/33, save 20k/50k/100k. Final epsilon 0.01 limits time spent in correlated exploration blocks. No heuristic is used in this policy. Final test split remains unused.

## Approaching-vehicle observations

Queue-focused seed 11 at 100k reduced validation waiting by 3.72% but increased queue by 2.69% and reduced throughput by 5.77%; it does not meet success criteria. A stopped-queue count drops as soon as cars accelerate, even before they cross the junction. Add eight moving-vehicle counts within 50m of the stop line, in the same N-left/N-through/... grouping. Divide by 10 and clip to [0,1], append after the original 26 features (34 total). This information is locally measurable by future camera ROI tracking; no future routes or privileged demand forecasts are exposed.

Use the temporal-exploration configuration, seeds 11/22/33, 150k budget, 20k/50k/100k/150k checkpoints. Set exploration_fraction=0.2 to preserve the 30k-step epsilon decay of the 100k temporal runs. Traffic, physical rules and baseline stay fixed. An integration test checks exact TraCI/libsumo parity with and without these extra sensors. Select only on validation 3001–3005; final 4001–4030 still unused.

## Freeze rule (before final results)

All three approaching-vehicle models meet the primary validation criteria at 50k. Finish evaluating the declared 100k and 150k checkpoints. Among common checkpoint budgets where all three seeds improve waiting and queue and do not reduce aggregate throughput, choose the budget with the lowest mean waiting across the three seeds. Freeze all three models at that same budget, with SHA-256 checksums. Choose the default/demo model as the lowest-validation-waiting member among those three. Do not choose it from final test results. Then run exactly 4001–4030 × six scenarios for Fixed and each frozen DQN (180 Fixed + 540 DQN episodes).
