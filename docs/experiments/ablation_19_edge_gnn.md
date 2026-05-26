# Ablation `_19_edge_gnn`: Edge-GNN on candidate pairs (line graph)

## What changes vs `_14`

| Held fixed | `_19` only |
|------------|------------|
| `_13` `pair_training_dataset.csv`, same 19 explicit pair features as `_14` | Each candidate pair is an **edge-node** in a line graph |
| nnPU loss, row split (`pair_split_seed: 42`) | 2-layer GraphSAGE over edge-nodes sharing an email endpoint |
| Community/scoring graph bundle (`graph_id` `_13`) | Checkpoints under `edge_gnn/models/`; scores in `edge_gnn_pair_scores.csv` |
| Per-epoch train sampling (`train_balance` 2:1, etc.) | Same helpers as `_14` explicit-only MLP (`build_train_epoch_cluster_aware` + emphasis) |

**Not used:** hetero `HeteroSAGE`, `email.x` embeddings, `EmailPairMLPScorer` z_i/z_j head.

This is **Ablation 1** only (Edge-GNN replaces the MLP scorer). No residual MLP+Edge-GNN stack yet.

## Merge config (training)

Apply [`pipeline_fragment.dedup_task_identity_19_edge_gnn.json`](../../seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_19_edge_gnn.json) into root `pipeline_config.json`.

## Step 1 — Train Edge-GNN

```powershell
.\.venv311\Scripts\python.exe core/main.py gnn
```

Confirm artifacts under `output/runs/<run_id>/`:

- `edge_gnn/pair_training_setup_summary.json` → `pair_encoder_backend: edge_gnn`
- `edge_gnn/models/best_model.pt`
- `edge_gnn/edge_line_graph.pt`
- `edge_gnn_pair_scores.csv` (columns include `email_i`, `email_j`, `pu_score`)

Default training `run_id`: `main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_v1`.

### Diagnostic: no message passing (`num_gnn_layers: 0`)

Merge [`pipeline_fragment.dedup_task_identity_19_edge_gnn_no_mp.json`](../../seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_19_edge_gnn_no_mp.json), then train:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/merge_pipeline_fragment.py seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_19_edge_gnn_no_mp.json
$env:PYTHONIOENCODING='utf-8'
.\.venv311\Scripts\python.exe core/main.py gnn
```

Run ID: `main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_no_mp`. Line graph is skipped; same nnPU / sampling / features as `_19`.

Compare metrics:

```powershell
.\.venv311\Scripts\python.exe core/GNN/debug/compare_edge_gnn_ablation_metrics.py `
  --mlp-run output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp `
  --edge-gnn-run output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_v1 `
  --edge-gnn-no-mp-run output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_no_mp
```

### Ablation 1: no-MP, `_14`-compatible local MLP head

Fair baseline: same `EmailPairMLPScorer` explicit-only head as `_14` (`hidden=256`, `dropout=0.2`), inside the Edge-GNN training branch with `num_gnn_layers=0`.

Merge fragment:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/merge_pipeline_fragment.py `
  seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_19_edge_gnn_no_mp_mlp_compatible.json
```

Train:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv311\Scripts\python.exe core/main.py gnn
```

Run ID: `main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_no_mp_mlp_compatible`

Artifacts:

- `output/runs/<run_id>/edge_gnn_pair_scores.csv`
- `output/runs/<run_id>/edge_gnn/models/best_model.pt`
- `output/runs/<run_id>/edge_gnn/metrics.csv`

Pair score separation (uses `edge_gnn_pair_scores.csv` when present):

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_no_mp_mlp_compatible `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json
```

Deduped GT community eval:

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp73.dedup_task_identity_19_edge_gnn_no_mp_mlp_compatible.score_only.json
```

Expanded GT community eval:

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp74.dedup_task_identity_19_edge_gnn_no_mp_mlp_compatible.score_only.expanded_gt.json
```

### Ablation 2: 1-layer GraphSAGE, top-k 8, MLP-compatible local head

Conservative message passing: `_14`-compatible local encoder (`Linear→ReLU→Dropout`, hidden=256), **one** GraphSAGE layer on the sparse candidate-edge line graph, final logits from `concat(h_local, h_graph) → Linear(512, 1)` (`combine_mode=concat_local_graph`).

#### Top-k 8 (Ablation 2)

Merge:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/merge_pipeline_fragment.py `
  seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_19_edge_gnn_1layer_topk8.json
```

Train:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv311\Scripts\python.exe core/main.py gnn
```

Run ID: `main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_1layer_topk8`

Pair score separation:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_1layer_topk8 `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json
```

Deduped GT community eval:

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp75.dedup_task_identity_19_edge_gnn_1layer_topk8.score_only.json
```

Expanded GT community eval:

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp76.dedup_task_identity_19_edge_gnn_1layer_topk8.score_only.expanded_gt.json
```

### Ablation 3: 1-layer GraphSAGE, top-k 16

Same architecture and training as Ablation 2 (top-k 8); **only** `max_neighbors_per_endpoint` changes from **8 → 16** (denser line graph, fewer isolated edge-nodes).

**Deduped GT is intentionally skipped for this ablation** — use expanded GT only (`exp78`).

#### Merge config

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/merge_pipeline_fragment.py `
  seed_candidate_workflow/configs/experiments/pipeline_fragment.dedup_task_identity_19_edge_gnn_1layer_topk16.json
```

#### Train

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv311\Scripts\python.exe core/main.py gnn
```

Run ID: `main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_1layer_topk16`

**Artifacts:**

| Artifact | Path |
|----------|------|
| Pair scores | `output/runs/<run_id>/edge_gnn_pair_scores.csv` |
| Training metrics | `output/runs/<run_id>/edge_gnn/metrics.csv` |
| Line graph stats | `output/runs/<run_id>/edge_gnn/pair_training_setup_summary.json` → `line_graph_stats` |
| Line graph tensor | `output/runs/<run_id>/edge_gnn/edge_line_graph.pt` |
| Checkpoint | `output/runs/<run_id>/edge_gnn/models/best_model.pt` |

#### Pair score separation

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py `
  --run-dir output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_1layer_topk16 `
  --graph-pt core/graph/output/main_gnn_pu_1_no_ts_dedup_task_identity_12_hetero.pt `
  --gt-path data/groundtruth/ground_truth.dedup_task_identity.json
```

Outputs under: `output/runs/<run_id>/pair_score_separation/`

#### Expanded GT community eval

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp78.dedup_task_identity_19_edge_gnn_1layer_topk16.score_only.expanded_gt.json
```

**Community results:** `seed_candidate_workflow/output/scoring_runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_1layer_topk16__expanded_full_gt/seed_candidate/community/`

Best row: `anchor_community_best__ground_truth.json` (sorted by `v-measure`).

#### Compare top-k 8 vs 16 vs `_14`

| Run | Expanded GT eval |
|-----|------------------|
| `_14_only_mlp` | exp62 |
| top-k 8 | exp76 |
| top-k 16 | exp78 |

## Step 2 — Community evaluation (same as `_14`)

Evaluation uses score mode `seed_candidate_edge_gnn_v1`: it loads the **same** unscored seed/candidate PairGraph as `_14` (`graph_id` `main_gnn_pu_1_no_ts_dedup_task_identity_13`) and maps `pu_score` from `edge_gnn_pair_scores.csv` with the same seed edge weight and non-seed `weight_mode` transform as PU runs.

### Update eval config with your training run

Edit `edge_gnn_run.run_dir` (or `pair_scores_csv`) in:

- [`exp71.dedup_task_identity_19_edge_gnn.score_only.json`](../../seed_candidate_workflow/configs/experiments/exp71.dedup_task_identity_19_edge_gnn.score_only.json)
- [`exp72.dedup_task_identity_19_edge_gnn.score_only.expanded_gt.json`](../../seed_candidate_workflow/configs/experiments/exp72.dedup_task_identity_19_edge_gnn.score_only.expanded_gt.json)

Example:

```json
"edge_gnn_run": {
  "run_dir": "output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_19_edge_gnn_v1"
}
```

Or point directly at the CSV:

```json
"pair_scores_csv": "output/runs/<run_id>/edge_gnn_pair_scores.csv"
```

### Run community sweep

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp71.dedup_task_identity_19_edge_gnn.score_only.json
```

Expanded GT (comparable to `_14` / `_62`):

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp72.dedup_task_identity_19_edge_gnn.score_only.expanded_gt.json
```

Dry-run config load:

```powershell
.\.venv311\Scripts\python.exe -m seed_candidate_workflow.pipelines.run_experiment `
  --config seed_candidate_workflow/configs/experiments/exp71.dedup_task_identity_19_edge_gnn.score_only.json `
  --dry-run
```

## Outputs

| Artifact | Path |
|----------|------|
| Scoring run root | `seed_candidate_workflow/output/scoring_runs/<scoring_run_id>/` |
| Run manifest | `.../run_manifest.json` |
| Edge-GNN score diagnostics | `.../seed_candidate/community/edge_gnn_scoring.json` |
| Per-GT sweep CSVs | `.../seed_candidate/community/anchor_community_sweep__<gt>.csv` |
| Best expanded-GT row | Top row of `anchor_community_sweep__*.csv` sorted by `v_measure` (see `sort_by` in config) |
| Optional scorer diagnostics | `.../seed_candidate/community/scorer_diagnostics.json` if `scoring.diagnostics.enabled` is true |

Community results mirror `_14` layout under `seed_candidate_workflow/output/scoring_runs/`.

## Compare

| Run | Scorer | Community eval |
|-----|--------|----------------|
| `_14_only_mlp` | `seed_candidate_pu_v1` + MLP checkpoint | exp61 / exp62 |
| `_15` GNN-only | `seed_candidate_pu_v1` | exp63 / exp64 |
| `_18` raw email.x | `seed_candidate_pu_v1` | exp69 / exp70 |
| `_19_edge_gnn` | `seed_candidate_edge_gnn_v1` + `edge_gnn_pair_scores.csv` | exp71 / exp72 |
| `_19_edge_gnn_no_mp_mlp_compatible` | same score mode + MLP-compatible no-MP CSV | exp73 / exp74 |
| `_19_edge_gnn_1layer_topk8` | same + 1-layer sparse line graph CSV | exp75 / exp76 |
| `_19_edge_gnn_1layer_topk16` | same; Ablation 3 uses **expanded GT only** | exp78 |
