# Ablation `_14_only_mlp`: early-stopping sanity (epoch budget)

Sanity-check whether the fixed **30-epoch** training budget affects learned pair scores and community detection.

**Does not overwrite:** `output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp`.

## Held fixed vs baseline

| Item | Value |
|------|--------|
| Pair CSV | `_13` no-timestamp `pair_training_dataset.csv` |
| Features / MLP | `explicit_only`, 21 dims, same arch |
| π | 0.1 (nnPU) |
| Split | `pair_split_seed: 42`, val/test 0.1 |
| GT + community sweep | expanded GT, same grid as exp62 |

## Changed only

| Setting | Baseline | Early-stopping sanity |
|---------|----------|------------------------|
| `epochs` | 30 | **100** |
| `early_stopping_patience` | 7 | **10** |
| Checkpoint | lowest val nnPU loss | same |

## Commands

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase all
```

Stepwise:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase train
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase community
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_early_stopping_sanity_14_only_mlp.py --phase consolidate
```

## Output paths

| Artifact | Path |
|----------|------|
| MLP run | `output/runs/14_only_mlp__early_stopping_sanity/` |
| Metrics | `.../mlp/metrics.csv` |
| Community sweep | `seed_candidate_workflow/output/scoring_runs/14_only_mlp__early_stopping_sanity__expanded_gt/seed_candidate/community/anchor_community_sweep__ground_truth.csv` |
| Comparison | `seed_candidate_workflow/output/early_stopping_sanity_14_only_mlp/` |

Manifest: `seed_candidate_workflow/configs/early_stopping_sanity/early_stopping_sanity_14_only_mlp.manifest.json`
