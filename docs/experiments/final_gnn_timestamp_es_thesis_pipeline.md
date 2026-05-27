# Final thesis GNN pair scoring (timestamp heterograph + ES100)

Replicates legacy `_13` (GNN + explicit pair features) and `_15_gnn_only_scorer` (GNN-only) under the **same protocol as** `final_14_only_mlp__timestamp_feature__early_stopping`:

- Timestamp-enabled deduplicated heterograph (`zero_email_timestamps: false`)
- Final timestamp-materialized pair universe (`final_14_only_mlp__timestamp_feature__early_stopping/pair_training_dataset.csv`)
- nnPU training: max **100** epochs, early stopping patience **10**, best checkpoint = lowest **validation nnPU loss**
- Community sweep: Louvain + Leiden, thresholds **0.0–0.9** (step 0.1), resolutions **1.0, 1.5, 2.0, 3.0**, expanded GT `data/groundtruth/ground_truth.json`

**New run IDs (do not overwrite old runs):**

| Variant | `run_id` |
|---------|----------|
| GNN + explicit pair features | `main_gnn_pu_1_ts_dedup_task_identity_gnn_plus_pair_features_thesis_es100` |
| GNN-only scorer | `main_gnn_pu_1_ts_dedup_task_identity_gnn_only_scorer_thesis_es100` |

**Heterograph:** `core/graph/output/main_gnn_pu_1_ts_dedup_task_identity_thesis_hetero.pt`

**Thesis outputs:** `seed_candidate_workflow/output/final_gnn_pair_scoring_timestamp_es_thesis/`

## Prerequisites

1. Final MLP pair CSV exists (run once if needed):

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_14_only_mlp_step02_materialize.py
```

2. Repo root, Python 3.11 venv active.

## Overnight: full pipeline (recommended)

```powershell
cd C:\Users\aar\Desktop\GNN-Campaign-Detection

.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\run_final_gnn_timestamp_es_thesis_pipeline.py `
  --skip-existing 2>&1 | Tee-Object -FilePath seed_candidate_workflow\output\final_gnn_timestamp_es_steps\overnight_run.log
```

Resume from a failed step (example: training GNN+features only):

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\run_final_gnn_timestamp_es_thesis_pipeline.py `
  --from-step 03 --skip-existing
```

## Step-by-step (if you prefer separate commands)

### 1. Build timestamp heterograph

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step01_build_graph.py
```

### 2. Verify pair CSV + graph + no time-gating

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step02_verify_inputs.py
```

### 3. Train GNN + explicit pair features (~hours)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step03_train_gnn_plus.py
```

### 4. Train GNN-only scorer

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step04_train_gnn_only.py
```

### 5–6. Scoring + community sweep

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step05_community_gnn_plus.py
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step06_community_gnn_only.py
```

### 7–8. Score diagnostics + training loss plots

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step07_score_diagnostics.py
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step08_training_plots.py
```

### 9. Consolidate thesis folder

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow\scripts\final_gnn_timestamp_es_step09_consolidate.py
```

## Timestamp feature (heterograph)

- Raw: Unix seconds from MISP `date` on each deduplicated email.
- Stored as the first scalar in the email feature matrix before projection.
- **Not used raw in the GNN:** `normalize_graph` applies per-node-type IQR outlier clipping + z-score (`core/graph/normalizer.py`).
- Seed/candidate **generation is unchanged**; timestamps are **not** used as candidate time-gating filters.

See `seed_candidate_workflow/output/final_gnn_pair_scoring_timestamp_es_thesis/graph_timestamp_summary.json` after step 1.

## Key artifact paths after completion

| Artifact | Path |
|----------|------|
| GNN+features checkpoint | `output/runs/main_gnn_pu_1_ts_dedup_task_identity_gnn_plus_pair_features_thesis_es100/gnn/models/best_model.pt` |
| GNN-only checkpoint | `output/runs/main_gnn_pu_1_ts_dedup_task_identity_gnn_only_scorer_thesis_es100/gnn/models/best_model.pt` |
| Community sweeps | `.../community/anchor_community_sweep__ground_truth.csv` (per run) |
| Thesis bundle | `seed_candidate_workflow/output/final_gnn_pair_scoring_timestamp_es_thesis/paths_manifest.json` |
