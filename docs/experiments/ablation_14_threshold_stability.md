# Threshold stability: `_14_only_mlp` learned pair scores

Isolates **edge-threshold sensitivity** while holding community detection fixed:

- **Algorithm:** Leiden only  
- **Resolution:** 3.0 only  
- **Thresholds:** 0.0, 0.1, …, 0.9  

Does **not** retrain the MLP. Does **not** overwrite the existing `__expanded_full_gt` scoring run (exp62).

## Config

`seed_candidate_workflow/configs/experiments/exp63.dedup_task_identity_14_only_mlp.threshold_stability.leiden.resolution_3.expanded_gt.json`

**Scoring run id:** `14_only_mlp__thresh_stab__leiden_r3__expanded_gt` (short id avoids Windows MAX_PATH on sweep CSV)

## Run (repo root, `.venv311`)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp63.dedup_task_identity_14_only_mlp.threshold_stability.leiden.resolution_3.expanded_gt.json
```

Dry-run plan only:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp63.dedup_task_identity_14_only_mlp.threshold_stability.leiden.resolution_3.expanded_gt.json `
  --dry-run
```

## Primary output (community sweep CSV)

`seed_candidate_workflow/output/scoring_runs/14_only_mlp__thresh_stab__leiden_r3__expanded_gt/seed_candidate/community/anchor_community_sweep__ground_truth.csv`

Also written: `anchor_community_best__ground_truth.json`, `anchor_community_multi_gt_summary.json`, `run_manifest.json` under the same scoring run root.

## What is held constant vs swept

| Held fixed | Swept |
|------------|--------|
| `_14_only_mlp` checkpoint (`mlp/models/best_model.pt`) | `min_edge_weight` ∈ {0.0 … 0.9} |
| Bundle `_13` pair universe (49,030 pairs) | — |
| `seed_candidate_pu_v1` scoring (`raw_score`, seed weight 1.0) | — |
| Expanded GT + dedup member expansion (same as exp62) | — |
| Leiden, resolution 3.0 | — |

`score_only` re-runs **MLP inference** to attach `pu_score` / `edge_weight` (deterministic, ~minutes on CPU). Training is not repeated.

## Reference run (unchanged)

Full method × resolution × threshold grid:  
`seed_candidate_workflow/output/scoring_runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp__expanded_full_gt/` (exp62).
