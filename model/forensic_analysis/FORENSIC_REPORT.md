# Forensic Research Visualization Report

Generated from repository artifacts by `forensic_analysis.py`.
Frozen model definitions, checkpoints, and historical result folders were read but not modified.

## A. Complete Result Inventory

- Benchmark rows: 480 raw rows across 4 conditions.
- Training logs: 82 files, 2202 epoch records.
- Checkpoint files: 102 `.pth` files.
- Benchmark JSON families per condition: raw, aggregated, Pareto, threshold, and statistical outputs; recovery uses `statistical_tests.json` rather than `statistical_comparison.json`.
- No CSV, NPY, PKL, attention matrix, latent-state trace, readout distribution, or standalone diagnostic artifact was found.

The detailed file-level inventory is in `inventory.csv`; the normalized benchmark table is in `master_benchmark.csv`; training records are in `training_logs.csv`.

## B. Master Experiment Matrix

| Condition | Architecture | Tasks | Seeds | k | D in result rows |
|---|---|---|---|---|---|
| 126K | CoT, Recurrent Latent | task_a, task_b | 42, 43, 44, 45, 46 | 1, 2, 4, 8, 12, 16 | absent |
| 1M | CoT, Recurrent Latent | task_a, task_b | 42, 43, 44, 45, 46 | 1, 2, 4, 8, 12, 16 | absent |
| 5M original | CoT, Recurrent Latent | task_a, task_b | 42, 43, 44, 45, 46 | 1, 2, 4, 8, 12, 16 | absent |
| 5M recovery | CoT, Recurrent Latent | task_a, task_b | 42, 43, 44, 45, 46 | 1, 2, 4, 8, 12, 16 | absent |

Problem depth D is present in the generator/API (2 through 16), but not recorded per benchmark row, so no empirical accuracy-vs-D figure is justified from these outputs.

## C. Data Consistency Audit

- Raw-to-aggregate checks: all aggregate means and standard deviations agree with raw seed rows within tolerance.
- Model naming is inconsistent across generations: `AutoregressiveCoT`/`RecurrentLatentReasoner` versus `cot`/`latent`; analysis canonicalizes these to CoT and Recurrent Latent.
- 5M original and 5M recovery are different training conditions and are never pooled.
- Latency values are measured wall-clock outputs, but cross-condition hardware/runtime comparability is not established by the files; interpret latency frontiers within condition.
- Statistical outputs test same-k seed differences, not latency-matched differences. Five seeds are the independent replicates; 1,000 predictions are repeated test instances.
- Seed 46 is retained. The raw files show seed-level variation, but no causal explanation is established.

### Conflicts

- No value conflicts detected between raw rows and aggregate files.

## D. Recommended Graphs

| Figure | Scientific question | Source/variables | Aggregation | Rank |
|---|---|---|---|---|
| 01 Task A accuracy vs k | Does extra inference compute continue helping? | raw benchmark: task_a, k, accuracy, model, condition | mean +/- SEM across five seeds | ESSENTIAL |
| 02 Task B accuracy vs k | Is the pattern task-specific or near chance? | raw benchmark: task_b, k, accuracy | mean +/- SEM across five seeds | SUPPORTING |
| 03 Scaling at k=16 | How do scale and training condition change final accuracy? | raw benchmark: task_a, k=16, scale/condition, accuracy | mean +/- SEM | ESSENTIAL |
| 04 Latency Pareto | Which observed points trade latency for accuracy? | raw benchmark: task_a, latency_ms, accuracy, k | observed non-dominated points only | ESSENTIAL |
| 05 Original vs recovery | Did the recovery condition change the 5M outcome? | raw benchmark: 5M conditions, task_a, k, accuracy | mean +/- SEM | ESSENTIAL |
| 06 Training loss | How do optimization trajectories differ? | export training_log.json: epoch, loss | mean across seeds | SUPPORTING |
| 07 Seed variability | How much do conclusions vary across training seeds? | raw benchmark: task_a, k=16, seed, accuracy | individual points + mean | SUPPORTING |

Analytical FLOPs can support a parallel Pareto figure, but it is largely a deterministic linear function of k within each architecture/condition and is redundant with the latency figure for an 8-minute presentation. No D, latent transition, attention, readout, or Hebbian diagnostic graph is recommended because the required raw measurements are absent.

## E. Generated Graphs

- `01_taskA_accuracy_vs_k.png`
- `02_taskB_accuracy_vs_k.png`
- `03_scaling_k16.png`
- `04_pareto_latency_accuracy.png`
- `05_5m_original_vs_recovery.png`
- `06_training_loss_5m.png`
- `07_seed_variability_k16.png`

## F. Presentation Recommendation

1. Task A accuracy vs k: CoT improves with budget in successful conditions, while latent accuracy rises early and then changes little; this is an observed pattern, not a universal architectural claim.
2. Scaling at k=16: scale alone does not explain outcomes because the 5M original condition remains near chance while 5M recovery succeeds.
3. Latency Pareto: the relevant trade-off is observed accuracy versus measured latency, separated by condition because comparability is limited.
4. 5M original versus recovery: recovery changes Task A substantially, but multiple optimization variables changed, so causality is not isolated.
5. Task B accuracy vs k: results remain close to the eight-way random baseline (0.125) and do not support a strong general conclusion.
6. Training loss: optimization trajectories provide context for the 5M recovery result.

## G. Data That Should Not Be Graphed

- Problem-depth accuracy: D is not attached to benchmark rows.
- Latent state transition magnitude, cosine similarity, effective rank, attention, and readout diagnostics: no raw diagnostic arrays or records were found.
- A universal cross-scale latency frontier: measurement comparability is not established.
- Aggregated means without seed uncertainty when comparing models: five-seed variability is material, especially at larger k.

## Main Verified Pattern

Task A supports continued CoT improvement with k in 126K, 1M, and 5M recovery. Recurrent Latent improves sharply from k=1 to k=2 in those conditions, then is approximately flat. The original 5M condition is a failure/near-chance condition for both architectures and must remain separate. Task B is weak and mostly near chance. Internal-state causal explanations are not testable from the stored outputs.
