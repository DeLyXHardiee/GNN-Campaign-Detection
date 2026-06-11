# Ablation `_16`: GNN-only scorer on hetero graph without `domain` / `html_structure_fingerprint`

Fair comparison to `_15_gnn_only_scorer`: same pair CSV, splits, PU hyperparameters, and scorer settings; **only** the hetero graph structure changes.

## What changes vs `_15`

| Held fixed | `_16` only |
|------------|------------|
| `_13` `pair_training_dataset.csv`, `pair_split_seed: 42` | Hetero graph: **no** `domain` or `html_structure_fingerprint` nodes/edges |
| GNN-only scorer (`pair_scorer_use_explicit_features: false`) | New artifact: `main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_hetero.pt` |
| LR 0.0002, 30 epochs, nnPU, cluster sampling, etc. | Seed/candidate **bundle** still `_13` (anchor graph unchanged) |

## Create stripped hetero graph (one-time)

From repo root (does not rebuild MISP; strips `_12` checkpoint):

```powershell
.\.venv311\Scripts\python.exe core/graph/scripts/strip_hetero_node_types.py `
  --input-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --output-stem main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp `
  --strip domain html_structure_fingerprint
```

Outputs:

- `core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_hetero.pt`
- `core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_hetero.meta.json`
- `core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_hetero.strip_manifest.json` (records source path for restore)

**Restore full hetero graph later:** set `graph_pt_path_override` back to `_12` hetero `.pt` (see manifest `source_graph_pt`).

## Enable in `pipeline_config.json`

Merge keys from `seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_16_no_domain_html_fp_gnn_only_scorer.json`:

- `run_id` → `main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_gnn_only_scorer`
- `graph.hetero_graph_stem` / `graph.graph_pt_path_override` → `_16` paths above
- `graph.exclude_node_types` → add `"domain"`, `"html_structure_fingerprint"` (plus existing mail-header types)
- `pair_training` → same as `_15` fragment (GNN backend, embedding-only scorer)

Leave `training.*` unchanged from `_15`.

## Run sequence (repo root, `.venv311`)

### 0. Strip graph (if not done)

See command above.

### 1. Train GNN + embedding-only scorer

```powershell
.\.venv311\Scripts\python.exe core/main.py gnn
```

Artifacts: `output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_gnn_only_scorer/gnn/models/best_model.pt`

### 2. Pair score separation

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_gnn_only_scorer `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_16_no_domain_html_fp_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json `
  --pair-csv seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training/main_gnn_pu_1_no_ts_dedup_task_identity_13/pair_training_dataset.csv `
  --high-cross-score-min 0.80 `
  --mid-cross-score-min 0.70
```

### 3a. Scoring + community — dedup GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp65.dedup_task_identity_16_no_domain_html_fp_gnn_only_scorer.pu.score_only.json
```

### 3b. Scoring + community — expanded GT

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp66.dedup_task_identity_16_no_domain_html_fp_gnn_only_scorer.pu.score_only.expanded_gt.json
```

## Compare GNN-only ablations

| Run | Hetero graph |
|-----|----------------|
| `_15_gnn_only_scorer` | `_12` full (includes domain + html fp nodes) |
| `_16_no_domain_html_fp_gnn_only_scorer` | `_12` minus domain + html_structure_fingerprint |
