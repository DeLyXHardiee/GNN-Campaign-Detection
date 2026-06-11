# nnPU prior sensitivity: `_14_only_mlp`

Evaluates sensitivity of the explicit 21-feature MLP pair scorer to the nnPU class prior \(\pi \in \{0.05, 0.10, 0.20, 0.30\}\).

**Held fixed (same as baseline `_14_only_mlp`):**

- Pair universe: bundle `_13` (`pair_training_dataset.csv`, 49,030 pairs)
- `pair_encoder_backend`: `explicit_only`, 21 features, MLP 256 hidden
- Train/val/test split: `pair_split_seed=42`, `pair_val_ratio=0.1`, `pair_test_ratio=0.1`
- Optimizer: `lr=2e-4`, `wd=2e-5`, `epochs=30`, early stopping on **lowest validation nnPU loss** (`early_stopping_patience=7`)
- Community: Louvain + Leiden, thresholds **0.0–0.9** (step 0.1), resolutions 1.0–3.0 (80 settings per π before filtering)
- GT: `data/groundtruth/ground_truth.json` with dedup member expansion

**Does not overwrite:** `output/runs/main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp` or `...__expanded_full_gt` scoring outputs.

## Manifest

`seed_candidate_workflow/configs/prior_sensitivity/prior_sensitivity_14_only_mlp.manifest.json`

| \(\pi\) | Training `run_id` | Scoring run id |
|--------|-------------------|----------------|
| 0.05 | `14_only_mlp__prior_sensitivity__pi_0p05` | `14_only_mlp__prior_sensitivity__pi_0p05__expanded_gt` |
| 0.10 | `14_only_mlp__prior_sensitivity__pi_0p10` | `14_only_mlp__prior_sensitivity__pi_0p10__expanded_gt` |
| 0.20 | `14_only_mlp__prior_sensitivity__pi_0p20` | `14_only_mlp__prior_sensitivity__pi_0p20__expanded_gt` |
| 0.30 | `14_only_mlp__prior_sensitivity__pi_0p30` | `14_only_mlp__prior_sensitivity__pi_0p30__expanded_gt` |

## Run everything (train → community → consolidate)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase all
```

Resume after partial progress:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase all --skip-existing
```

Per-prior training only:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase train --pi 0.05
```

Community only (after all checkpoints exist):

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/run_prior_sensitivity_14_only_mlp.py --phase community
```

Or single experiment:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/pipelines/run_experiment.py `
  --config seed_candidate_workflow/configs/experiments/exp64.14_only_mlp.prior_sensitivity.pi_0p05.expanded_gt.json
```

## Individual outputs

**Training (per \(\pi\)):**

- `output/runs/14_only_mlp__prior_sensitivity__pi_<slug>/mlp/models/best_model.pt`
- `output/runs/14_only_mlp__prior_sensitivity__pi_<slug>/mlp/metrics.csv` (checkpoint = min `val_loss`)

**Community sweep CSV (per \(\pi\)):**

`seed_candidate_workflow/output/scoring_runs/14_only_mlp__prior_sensitivity__pi_<slug>__expanded_gt/seed_candidate/community/anchor_community_sweep__ground_truth.csv`

## Consolidated thesis outputs

After all sweeps exist:

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/consolidate_prior_sensitivity_14_only_mlp.py
```

| Artifact | Path |
|----------|------|
| Best row per \(\pi\) (CSV) | `seed_candidate_workflow/output/prior_sensitivity_14_only_mlp/prior_sensitivity_14_only_mlp_best_by_pi.csv` |
| JSON | `.../prior_sensitivity_14_only_mlp_best_by_pi.json` |
| LaTeX table | `.../prior_sensitivity_14_only_mlp_best_by_pi.tex` |

Columns: `pi`, `algorithm`, `threshold`, `resolution`, `homogeneity`, `completeness`, `v_measure`.

Best row = highest **V-measure** over the full community sweep for that prior (same rule as exp62).
