# Ablation `_14_only_mlp`: timestamp / temporal pair feature

Sanity-check: baseline `_14_only_mlp` (all `time_gap_seconds_min` = 0 from `zero_email_timestamps`) vs same pair universe with MISP-derived `log1p(time_gap_seconds)` in the MLP input column.

**Does not overwrite:** `output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp` or `...__expanded_full_gt`.

## Where timestamps are disabled today

| Stage | Baseline `_14_only_mlp` |
|-------|-------------------------|
| Hetero graph `email.x` | `pipeline_config.json` → `graph.zero_email_timestamps: true` → `email_attrs.ts = 0` in `core/graph/assembler.py` |
| Seed/candidate `nodes_df.ts` | Copied from graph meta → all zeros |
| Pair CSV `time_gap_seconds_min` | Aggregated from zero `ts_map` → **0.0** for all rows (non-null but uninformative) |
| MLP training | `explicit_only`: 21 explicit features; **no** `email.x` |
| Community detection | Uses PU pair scores only (not raw timestamps) |

## Pair universe decision

**Reuse** `main_gnn_pu_1_no_ts_dedup_task_identity_13` seed/candidate graph and pair keys.

Active generators in `_13` have `time_gating_enabled: false` (semantic reciprocal, 2hop). Component expansion is off. Pair **counts** stay identical; only `time_gap_seconds_min` values change.

Optional full rebuild (hetero `zero_email_timestamps: false` + setup) is only needed if you later enable time-gated generators.

## Temporal feature (timestamp branch)

- **Raw:** `|unix_ts(email_i) - unix_ts(email_j)|` from MISP `date_raw` / `timestamp_utc`
- **MLP column:** `time_gap_seconds_min` = `log1p(raw_seconds)` (missing → 0.0 at train time, same as other numerics)
- **Diagnostics column:** `time_gap_seconds_raw` in materialized CSV only

Typical raw gaps: seconds–weeks; `log1p` keeps values on a modest scale (not raw Unix timestamps).

## Commands (repo root, `.venv311`)

### 1. Materialize timestamp pair CSV (fast)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/materialize_timestamp_pair_training_dataset.py
```

Output:

`seed_candidate_workflow/output/graph_bundles/14_only_mlp__with_timestamp__timestamp_ablation/pair_training/14_only_mlp__with_timestamp__timestamp_ablation/pair_training_dataset.csv`

Summary:

`.../pair_training_dataset_timestamp_summary.json`

### 2. Train + community + consolidate (full ablation)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase all
```

Or stepwise:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase materialize
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase train
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase community
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_timestamp_ablation_14_only_mlp.py --phase consolidate
```

### Artifacts

| Item | Path |
|------|------|
| MLP run | `output/runs/14_only_mlp__with_timestamp__timestamp_ablation/` |
| Community sweep (expanded GT) | `seed_candidate_workflow/output/scoring_runs/14_only_mlp__with_timestamp__timestamp_ablation__expanded_gt/seed_candidate/community/anchor_community_sweep__ground_truth.csv` |
| Comparison CSV/JSON/LaTeX | `seed_candidate_workflow/output/timestamp_ablation_14_only_mlp/` |

Manifest: `seed_candidate_workflow/configs/timestamp_ablation/timestamp_ablation_14_only_mlp.manifest.json`

Experiment config: `seed_candidate_workflow/configs/experiments/exp65.14_only_mlp.with_timestamp.timestamp_ablation.expanded_gt.json`

## Interpretation

After `consolidate`, read `interpretation` in `timestamp_ablation_14_only_mlp_comparison.json`.

- **|ΔV| &lt; 0.01:** timestamp does not materially change the learned scorer at community level.
- **Larger ΔV:** rerun score-separation, threshold-stability, and prior-sensitivity on the timestamp branch.
