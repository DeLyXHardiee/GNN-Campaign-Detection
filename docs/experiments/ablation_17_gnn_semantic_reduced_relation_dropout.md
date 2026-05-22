# Ablation `_17`: reduced semantic email block + relation edge dropout (GNN-only scorer)

Fair comparison to `_15_gnn_only_scorer`: same pair CSV, splits, PU hyperparameters, and **no explicit pair features in the scorer**. Only the **GNN encoder path** changes.

## What changes

| Component | `_15` | `_17` |
|-----------|--------|--------|
| Hetero graph | `_12` full (domain + html fp) | **Same** `_12` `.pt` |
| Email input to GNN | Baked 128-d semantic + 32-d structured (160) | Train-time adapter → **32** semantic + **64** structured (**96**) |
| Semantic dropout | none | **0.30** on semantic block (train only) |
| Edge dropout | none | **0.20** default; **0.30** on `email→domain`, `email→html_structure_fingerprint` |
| Pair scorer | GNN embeddings only | **Same** |

## Enable in `pipeline_config.json`

Merge `seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_17_gnn_semantic_reduced_relation_dropout.json`:

- `run_id` → `main_gnn_pu_1_no_ts_dedup_task_identity_17_gnn_semantic_reduced_relation_dropout_v1`
- `graph.graph_pt_path_override` → `_12` hetero (not `_16` stripped graph)
- `gnn_encoder_ablation.enabled` → `true`
- `pair_training` → same as `_15` (GNN backend, `pair_scorer_use_explicit_features: false`)

Set `gnn_encoder_ablation.enabled: false` (or remove block) to restore `_15` behavior without code changes.

## Run sequence (repo root, `.venv311`)

### 1. Train

```powershell
.\.venv311\Scripts\python.exe core/main.py gnn
```

Check `output/runs/.../gnn/training_config.json` and `pair_training_setup_summary.json` for `gnn_encoder_ablation`.

### 2. Pair score separation

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_17_gnn_semantic_reduced_relation_dropout_v1 `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json `
  --pair-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv `
  --high-cross-score-min 0.80 `
  --mid-cross-score-min 0.70
```

### 3a. Scoring + community — dedup GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp67.dedup_task_identity_17_gnn_semantic_reduced_relation_dropout.pu.score_only.json
```

### 3b. Scoring + community — expanded GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp68.dedup_task_identity_17_gnn_semantic_reduced_relation_dropout.pu.score_only.expanded_gt.json
```

## Revert

- Set `gnn_encoder_ablation.enabled: false` in `pipeline_config.json`, or merge `_15` pipeline fragment.
- Use `_15` run dir / checkpoint for scoring.
