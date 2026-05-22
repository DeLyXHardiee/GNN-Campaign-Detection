# Ablation `_18_raw_email_only`: pre-GNN email features only

## What changes vs `_13` / `_14` / `_15`

| Held fixed | `_18` only |
|------------|------------|
| `_12` hetero `.pt`, `_13` `pair_training_dataset.csv`, splits (`pair_split_seed: 42`) | **No** hetero message passing (`model=None`) |
| LR, epochs, nnPU, cluster sampling, community sweep | Scorer input: **only** `x_i`, `x_j`, `\|x_i-x_j\|`, `x_i ⊙ x_j` from `data['email'].x` |
| Same MLP head dims (256 hidden, 0.2 dropout) | `pair_encoder_backend: mlp_raw_email_x` |
| | `pair_scorer_use_explicit_features: false` |
| | `gnn_encoder_ablation.enabled: false` |
| | Checkpoints: `mlp/models/best_model.pt` (not `gnn/models/`) |

**Explicit pair columns and provenance flags stay in the CSV** — they are not passed to the MLP for this run.

## Model family comparison

| Run | Scorer input |
|-----|----------------|
| `_13` | GNN embeddings + explicit pair features |
| `_14_only_mlp` | Explicit pair features only |
| `_15_gnn_only_scorer` | GNN embeddings only |
| `_18_raw_email_only` | Raw `email.x` only (no graph propagation) |

## Interpretation

- If `_18` ≈ `_15`: message passing adds little; signal is mostly in per-email features.
- If `_18` ≪ `_15`: the GNN is using neighborhood structure usefully.

## `pipeline_config.json`

Merge `pipeline_fragment.dedup_task_identity_18_raw_email_only.json` (or use the fragment fields already applied for step 0).

## Run sequence (repo root, `.venv311`)

### 1. Train raw-email pair scorer (MLP backend, no GNN)

```powershell
.\.venv311\Scripts\python.exe core/main.py gnn
```

Confirm `output/runs/.../mlp/training_config.json` has `pair_encoder_backend: mlp_raw_email_x` and `gnn_encoder_ablation.enabled: false`.

### 2. Pair score separation

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_18_raw_email_only `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json `
  --pair-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv `
  --high-cross-score-min 0.80 `
  --mid-cross-score-min 0.70
```

### 3a. Scoring + community — dedup GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp69.dedup_task_identity_18_raw_email_only.pu.score_only.json
```

### 3b. Scoring + community — expanded GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp70.dedup_task_identity_18_raw_email_only.pu.score_only.expanded_gt.json
```

## Revert

Restore `_15` / `_17` via `pipeline_fragment.dedup_task_identity_15_gnn_only_scorer.json` or `_17` fragment; set `pair_training.backends.gnn: true`, `mlp: false`, `pair_encoder_backend: gnn`.
