# Thesis baselines: unlearned graph community detection (`_14_only_mlp` pair universe)

Two comparable **Louvain / Leiden** sweeps on deduplicated graph nodes (4,970), evaluated with **expanded** ground truth via `dedup_collapse_out_dir`.

No PU/MLP/GNN training. Edge weights are **uniform 1.0** (`baseline_uniform_v1`).

## Graph definitions (bundle `main_gnn_pu_1_no_ts_dedup_task_identity_13`)

| Baseline | Target | Edge CSV | Definition |
|----------|--------|----------|------------|
| **Expanded anchor graph** | `anchor` | `anchor_graph_edges_expanded_unscored.csv` | Union of `anchor_graph_edges_unscored.csv` ∪ `seed_candidate_pairgraph_unscored.csv` (deduplicated canonical pairs) |
| **Seed+candidate graph** | `seed_candidate` | `seed_candidate_pairgraph_unscored.csv` | 49,030 pairs used by learned pair scoring (`pair_training_dataset.csv`); **strict subset** of expanded anchor edges |

**Nodes (both):** `anchor_graph_nodes.csv` (4,970 deduplicated `external_id`s).

**Materialize expanded anchor (required before anchor baseline run):**

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/materialize_expanded_anchor_graph.py
```

Writes:

- `seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_task_identity_13/anchor/main_gnn_pu_1_no_ts_dedup_task_identity_13/anchor_graph_edges_expanded_unscored.csv`
- `.../anchor_graph_edges_expanded_summary.json`

Does **not** change generator rules or recompute Jaccard; only unions existing pipeline artifacts.

## Ground truth

- **GT set:** `default_multi_gt`
- **Thesis tables:** slug `ground_truth` (expanded lake labels)
- **Dedup → expanded eval:** `data/misp/misp_lake_dedup_task_identity`

## Edge weights

- **Both baselines:** `edge_weight = 1.0` on every edge (`baseline_uniform_v1` / pre-set in expanded CSV).
- With uniform weight 1.0, thresholds `0.0`–`0.9` all retain the full edge set.

## Sweep (identical)

- **Algorithms:** `louvain`, `leiden`
- **Thresholds:** `0.0 … 0.9` (step 0.1)
- **Resolutions:** `1.0`, `2.0`, `3.0`, `4.0`
- **80 settings** per baseline per GT file; best row by `v-measure`

## Commands (repo root)

### 0. Build expanded anchor edge file

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/materialize_expanded_anchor_graph.py
```

### 1. Expanded anchor graph community detection

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp_thesis_anchor_graph_community_detection_14_pair_universe_expanded_gt.json
```

### 2. Seed+candidate graph community detection (unchanged)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp_thesis_candidate_graph_community_detection_14_pair_universe_expanded_gt.json
```

### 3. Consolidated thesis tables

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/consolidate_thesis_baseline_community_summaries.py
```

## Result paths

| Baseline | Scoring run ID |
|----------|----------------|
| Expanded anchor | `thesis_expanded_anchor_graph_community_detection__14_only_mlp_pair_universe__expanded_gt` |
| Seed+candidate | `thesis_candidate_graph_community_detection__14_only_mlp_pair_universe__expanded_gt` |

Raw sweeps: `seed_candidate_workflow/output/scoring_runs/<scoring_run_id>/<target>/community/anchor_community_sweep__ground_truth.csv`

Consolidated: `seed_candidate_workflow/output/thesis_baseline_community_detection/`

## Thesis wording (draft)

**Anchor pair graph:** We define the anchor pair graph as the deduplicated union of all unlearned email–email evidence edges produced by the pipeline: multi-channel anchor edges plus the seed-and-candidate pair universe (49,030 pairs), yielding one broad edge set over 4,970 emails with uniform weight for graph baselines.

**Seed and candidate pair sets:** Seed-positive and candidate pairs are the 49,030-edge pair universe used for PU/MLP training; this set is a strict subset of the expanded anchor pair graph.

**Graph-based baselines:** We compare Louvain/Leiden on (i) the expanded anchor pair graph and (ii) the seed+candidate pair graph alone, both unweighted and evaluated on expanded ground truth.
