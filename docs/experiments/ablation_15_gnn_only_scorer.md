# Ablation `_15_gnn_only_scorer`: GNN embedding interactions only

## What changes vs `_13` / `_14`

| Held fixed | `_15` only |
|------------|------------|
| `_12` hetero `.pt`, `_13` `pair_training_dataset.csv`, splits (`pair_split_seed: 42`) | Scorer input: **only** `z_i`, `z_j`, `\|z_i-z_j\|`, `z_i ⊙ z_j` |
| LR, epochs, nnPU, cluster sampling, community sweep | `pair_scorer_use_explicit_features: false` |
| Same MLP head dims (256 hidden, 0.2 dropout) | `pair_encoder_backend: gnn` (full message passing) |
| | Checkpoints: `gnn/models/best_model.pt` |

**Explicit pair columns remain in the CSV and codebase** — they are not passed to the MLP for this run.

## `pipeline_config.json` (already set for `_15`)

See `pipeline_fragment.dedup_task_identity_15_gnn_only_scorer.json` to restore `_14` or `_13` later.

## Run sequence (repo root, `.venv311`)

### 1. Train GNN + embedding-only scorer

```powershell
.\.venv311\Scripts\python.exe core/main.py gnn
```

### 2. Pair score separation

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_15_gnn_only_scorer `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json `
  --pair-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv
```

### 3a. Scoring + community — dedup GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp63.dedup_task_identity_15_gnn_only_scorer.pu.score_only.json
```

### 3b. Scoring + community — expanded GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp64.dedup_task_identity_15_gnn_only_scorer.pu.score_only.expanded_gt.json
```

## Compare ablations

| Run | Scorer input |
|-----|----------------|
| `_13` | GNN embeddings + explicit features (default) |
| `_14_only_mlp` | Explicit features only |
| `_15_gnn_only_scorer` | GNN embeddings only |
