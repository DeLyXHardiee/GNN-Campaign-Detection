# Best B2 (1:2) workflow — seed-candidate pipeline

Runbook for the current best pair-scorer setup from root-cause screening:

| Knob | Value |
|------|--------|
| Feature set | **B2** — drop Jaccard/body similarity pair features |
| Train sampling | **1:2** pos:unlabeled (`target_pos_to_unl_ratio: 0.5`) |
| PU prior | `pi = 0.01` |
| Graph bundle | `fcn_hrnt_ablation` (reparsed URLs, no collapse, HTML + return-path channels) |
| Hetero graph | `core/graph/output/incidents_lake_misp_reparsed_urls_dedup_task_identity_no_collapse_hetero.pt` |
| Community GT | `expanded_gt_only` + `dedup_collapse_out_dir` for member expansion |

**Single entrypoint for graph setup + PU scoring + community sweeps:**

```text
python seed_candidate_workflow/pipelines/run_experiment.py --config <experiment.json>
```

Pair-score separation is a **separate** step (same trained checkpoint).

---

## Prerequisites

- Repo venv active (e.g. `.venv311\Scripts\activate`)
- Hetero graph and pair CSV present (already built on this machine):
  - `core/graph/output/incidents_lake_misp_reparsed_urls_dedup_task_identity_no_collapse_hetero.pt`
  - `seed_candidate_workflow/output/graph_bundles/fcn_hrnt_ablation/pair_training/fcn_hrnt_ablation/pair_training_dataset.csv`
- Ground truth: `data/groundtruth/ground_truth.json` (see `gt_sets.json`)

---

## Config files (canonical)

| Purpose | File |
|---------|------|
| GT sets | `seed_candidate_workflow/configs/experiments/gt_sets.json` |
| Anchor graph (fcn_hrnt) | `seed_candidate_workflow/configs/anchor_graph.fcn_hrnt_ablation.json` |
| Pipeline — explicit MLP only | `seed_candidate_workflow/configs/experiments/pipeline_fragment.best_b2_1to2_mlp.json` |
| Pipeline — edge-GNN + local head | `seed_candidate_workflow/configs/experiments/pipeline_fragment.best_b2_1to2_edge_gnn.json` |
| Full train + score + community (MLP) | `seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_mlp.setup_gnn_score.json` |
| Full train + score + community (edge-GNN) | `seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_edge_gnn.setup_gnn_score.json` |
| Score + community only (MLP) | `seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_mlp.score_only.json` |
| Score + community only (edge-GNN) | `seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_edge_gnn.score_only.json` |

---

## A. Explicit MLP pair scorer (B2, 1:2)

### 1. Apply training settings to `pipeline_config.json`

```text
python seed_candidate_workflow/scripts/merge_pipeline_fragment.py seed_candidate_workflow/configs/experiments/pipeline_fragment.best_b2_1to2_mlp.json
```

This sets `run_id` → `best_b2_1to2_mlp_es100`, enables **mlp** backend only, `explicit_only`, B2 feature excludes, and 1:2 train balance.

### 2. Dry-run (path check, no execution)

```text
python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_mlp.setup_gnn_score.json --dry-run
```

### 3. Full pipeline: rebuild bundle → train → PU score → Louvain/Leiden sweep

```text
python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_mlp.setup_gnn_score.json
```

`setup_gnn_score` runs:

1. Graph setup (anchor → seed → candidate → seed_candidate → pair_training) under `graph_bundles/fcn_hrnt_ablation/`
2. Pair training to `output/runs/best_b2_1to2_mlp_es100/mlp/`
3. PU scoring + community under `scoring_runs/best_b2_1to2_mlp__expanded_gt/seed_candidate/community/`

**First outputs to open:**

- `seed_candidate_workflow/output/scoring_runs/best_b2_1to2_mlp__expanded_gt/run_manifest.json`
- `.../seed_candidate/community/anchor_community_multi_gt_summary.json`
- `output/runs/best_b2_1to2_mlp_es100/mlp/metrics.csv`

### 4. Pair score separation analysis

After training (checkpoint at `output/runs/best_b2_1to2_mlp_es100/mlp/models/best_model.pt`):

```text
python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py ^
  --run-dir output/runs/best_b2_1to2_mlp_es100/mlp ^
  --graph-pt core/graph/output/incidents_lake_misp_reparsed_urls_dedup_task_identity_no_collapse_hetero.pt ^
  --gt-path data/groundtruth/ground_truth.json
```

Outputs land in `output/runs/best_b2_1to2_mlp_es100/mlp/pair_score_separation/` (summary JSON, band tables, optional plots).

Equivalent module invocation:

```text
python -m seed_candidate_workflow.utils.pair_score_separation --run-dir output/runs/best_b2_1to2_mlp_es100/mlp --graph-pt core/graph/output/incidents_lake_misp_reparsed_urls_dedup_task_identity_no_collapse_hetero.pt --gt-path data/groundtruth/ground_truth.json
```

---

## B. Edge-GNN + pair scorer (B2, 1:2)

Same graph bundle and community eval; different `pipeline_config` backend.

### 1. Merge edge-GNN fragment

```text
python seed_candidate_workflow/scripts/merge_pipeline_fragment.py seed_candidate_workflow/configs/experiments/pipeline_fragment.best_b2_1to2_edge_gnn.json
```

`run_id` → `best_b2_1to2_edge_gnn_es100`, **edge_gnn** backend only.

### 2. Dry-run + full run

```text
python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_edge_gnn.setup_gnn_score.json --dry-run

python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_edge_gnn.setup_gnn_score.json
```

Training artifacts: `output/runs/best_b2_1to2_edge_gnn_es100/edge_gnn/`  
Community: `scoring_runs/best_b2_1to2_edge_gnn__expanded_gt/`

### 3. Pair score separation

```text
python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py ^
  --run-dir output/runs/best_b2_1to2_edge_gnn_es100/edge_gnn ^
  --graph-pt core/graph/output/incidents_lake_misp_reparsed_urls_dedup_task_identity_no_collapse_hetero.pt ^
  --gt-path data/groundtruth/ground_truth.json
```

---

## C. Rerun scoring + community only (no retrain)

If the bundle and checkpoint already exist:

1. Merge the correct pipeline fragment (MLP **or** edge-GNN).
2. Run the matching `*.score_only.json`:

```text
python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_mlp.score_only.json
```

or `exp.best_b2_1to2_edge_gnn.score_only.json`.

---

## D. Train only (no community sweep)

Reuse bundle CSV; skip experiment runner:

```text
python seed_candidate_workflow/scripts/merge_pipeline_fragment.py seed_candidate_workflow/configs/experiments/pipeline_fragment.best_b2_1to2_mlp.json

python seed_candidate_workflow/scripts/train_pair_supervision_run.py ^
  --run-id best_b2_1to2_mlp_es100 ^
  --pair-dataset-csv seed_candidate_workflow/output/graph_bundles/fcn_hrnt_ablation/pair_training/fcn_hrnt_ablation/pair_training_dataset.csv ^
  --graph-pt core/graph/output/incidents_lake_misp_reparsed_urls_dedup_task_identity_no_collapse_hetero.pt
```

Then run **C** (`*.score_only.json`) and pair separation (section A.4).

---

## What was removed / restored

Many one-off ablation scripts were deleted. This workflow uses only:

| Step | Tool |
|------|------|
| Experiment orchestration | `pipelines/run_experiment.py` |
| Pipeline knobs | `merge_pipeline_fragment.py` + `pipeline_config.json` |
| Pair separation | `scripts/run_pair_score_separation_analysis.py` → `utils/pair_score_separation.py` |
| Optional train-only | `scripts/train_pair_supervision_run.py` |

`gt_sets.json` was restored under `configs/experiments/` (required by `run_experiment`).

---

## Sanity checks after script cleanup

```text
python -m pytest seed_candidate_workflow/tests/test_run_experiment.py -q

python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp.best_b2_1to2_mlp.setup_gnn_score.json --dry-run

python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py --help
```

---

## Notes

- **Do not** enable both `mlp` and `edge_gnn` in one `pipeline_config` unless you intend to train and score both; `setup_gnn_score` loops all enabled backends.
- `setup_gnn_score` forces `setup.policy.on_present = rebuild` for the bundle (fresh seed/candidate graphs). To only refresh pair CSV, use `setup_only` with `on_present: reuse` or delete only `pair_training/fcn_hrnt_ablation/`.
- Community metrics use **expanded** GT when `dedup_collapse_out_dir` is set in the experiment `community` block (see `anchor_community_multi_gt_summary.json` → `gt_metric_email_expansion`).
