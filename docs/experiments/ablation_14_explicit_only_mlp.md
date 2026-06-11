# Ablation `_14_only_mlp`: explicit pair-feature scorer (no GNN embeddings)

## What changes vs `_13`

| Held fixed | Changed |
|------------|---------|
| `_12` hetero graph `.pt` | `pair_encoder_backend`: `explicit_only` |
| `_13` `pair_training_dataset.csv` (same splits via `pair_split_seed: 42`) | `pair_scorer_use_embedding_features`: `false` |
| Same PU hyperparameters (LR, epochs, nnPU, cluster sampling, etc.) | Checkpoints under `mlp/models/` (no GNN training) |
| Same community sweep configs | No `z_i`, `z_j`, `\|z_i-z_j\|`, `z_i ⊙ z_j` in scorer input |

## Enable in `pipeline_config.json`

Merge keys from `seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_14_only_mlp.json`:

- `run_id` → `main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp`
- `graph.*` (same as `_13` fragment)
- `pair_training.backends`: `{ "gnn": false, "mlp": true }`
- `pair_training.pair_encoder_backend`: `"explicit_only"`
- `pair_training.pair_scorer_use_embedding_features`: `false`
- `pair_training.pair_dataset_csv`: `_13` bundle path (reuse)

Leave `training.*` (LR, epochs, early stopping, `torch_seed`, etc.) unchanged from `_13`.

## Commands (repo root, `.venv311`)

### 1. Train explicit-only pair scorer

```powershell
# After updating pipeline_config.json run_id + pair_training keys above:
.\.venv311\Scripts\python.exe core/main.py gnn
```

Artifacts: `output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp/mlp/models/best_model.pt`

### 2. Pair score separation (same GT as `_13`)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json `
  --pair-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv
```

Uses `mlp/training_config.json` + checkpoint automatically.

### 3a. Community detection — dedup GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp61.dedup_task_identity_14_only_mlp.pu.score_only.json
```

### 3b. Community detection — expanded GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp62.dedup_task_identity_14_only_mlp.pu.score_only.expanded_gt.json
```

`graph_id` stays `_13` (reuse seed/candidate bundle); `pu_run.run_dir` is `_14_only_mlp`.

### 3c. Threshold stability (Leiden, resolution 3.0 only)

See [`ablation_14_threshold_stability.md`](ablation_14_threshold_stability.md) and  
`exp63.dedup_task_identity_14_only_mlp.threshold_stability.leiden.resolution_3.expanded_gt.json`.

### 3d. nnPU prior sensitivity (\(\pi \in \{0.05,0.10,0.20,0.30\}\))

See [`ablation_14_prior_sensitivity.md`](ablation_14_prior_sensitivity.md) and  
`seed_candidate_workflow/configs/prior_sensitivity/prior_sensitivity_14_only_mlp.manifest.json`.

## Revert for normal GNN+explicit runs

Set `pair_training.backends.gnn: true`, `mlp: false`, remove `pair_encoder_backend` / `pair_scorer_use_embedding_features` overrides (defaults restore GNN + full scorer).
