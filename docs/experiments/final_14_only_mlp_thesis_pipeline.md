# Final canonical `_14_only_mlp` thesis pipeline (stepwise)

Explicit MLP pair scorer with **log1p timestamp feature**, **early stopping** (max 100 epochs, patience 10), π=0.1, expanded GT. Does **not** overwrite `main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp`.

Manifest: `seed_candidate_workflow/configs/final_14_only_mlp/final_14_only_mlp.manifest.json`

Run from repo root with `.venv311`.

---

## Step 1 — Verify configuration

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step01_verify_config.py
```

**Outputs:** `seed_candidate_workflow/output/final_14_only_mlp_timestamp_es_steps/step01_verify_config_report.json` (+ `.md`)

---

## Step 2 — Materialize pair CSV

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step02_materialize.py
```

**Outputs:**

- `seed_candidate_workflow/output/graph_bundles/final_14_only_mlp__timestamp_feature__early_stopping/pair_training/final_14_only_mlp__timestamp_feature__early_stopping/pair_training_dataset.csv`
- `.../pair_training_dataset_timestamp_summary.json`
- `.../steps/step02_materialize_report.json`

---

## Step 3 — Train final MLP

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step03_train.py
```

**Outputs:**

- `output/runs/final_14_only_mlp__timestamp_feature__early_stopping/mlp/models/best_model.pt`
- `.../mlp/metrics.csv`
- `.../mlp/models/best_val_epochs/epoch_*.pt` (each validation improvement)
- `.../steps/step03_train_report.json`

---

## Step 4 — Community sweep (full grid)

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step04_community.py
```

**Outputs:**

- `seed_candidate_workflow/output/scoring_runs/final_14_only_mlp__timestamp_feature__early_stopping__expanded_gt/seed_candidate/community/anchor_community_sweep__ground_truth.csv`
- `.../steps/step04_community_report.json`

---

## Step 5 — Threshold sensitivity

Uses **best method + resolution from step 4**; sweeps threshold 0.0–0.9 only.

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step05_threshold_sensitivity.py
```

**Outputs:**

- `seed_candidate_workflow/output/scoring_runs/final_14_only_mlp__timestamp_feature__early_stopping__thresh_stab__expanded_gt/.../anchor_community_sweep__ground_truth.csv`
- `.../steps/step05_threshold_sensitivity_table.csv` (+ report JSON)

---

## Step 6 — Prior sensitivity

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step06_prior_sensitivity.py --phase train
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step06_prior_sensitivity.py --phase community
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step06_prior_sensitivity.py --phase consolidate
```

Or one π: `--pi 0.05`. Skip finished work: `--skip-existing`.

**Outputs:** `seed_candidate_workflow/output/final_14_only_mlp_timestamp_es_thesis/prior_sensitivity/prior_sensitivity_best_by_pi.{csv,tex,json}`

---

## Step 7 — Score separation + thesis pair diagnostics + KDE

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step07_score_diagnostics.py
```

**Outputs:**

- `output/runs/final_14_only_mlp__timestamp_feature__early_stopping/pair_score_separation/thesis_score_diagnostics/`
- `.../pair_score_separation/plots/score_density_kde_*.png`

---

## Step 8 — Training loss plot

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step08_training_plot.py
```

**Outputs:** `output/runs/final_14_only_mlp__timestamp_feature__early_stopping/plots/loss_over_epochs_best_val_marked.png`

---

## Step 9 — Epoch community diagnostic (not for model selection)

Fixed community setting from step 4; one row per `best_val_epochs/epoch_*.pt`.

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step09_epoch_community_diagnostic.py
```

**Outputs:** `.../steps/step09_epoch_community_diagnostic.{csv,json,tex}`

---

## Step 10 — Consolidate thesis folder

```powershell
.\.venv311\Scripts\python.exe seed_candidate_workflow/scripts/final_14_only_mlp_step10_consolidate.py
```

**Outputs:** `seed_candidate_workflow/output/final_14_only_mlp_timestamp_es_thesis/` (`THESIS_SUMMARY.md`, `paths_manifest.json`, copies of key CSV/TeX/plots)

---

## Legacy comparison

Baseline community reference: **Leiden, threshold 0.3, resolution 3.0, V=0.936** (`main_gnn_pu_1_no_ts_dedup_task_identity_14_only_mlp`).

Step 10 reports ΔV vs that baseline and whether |ΔV| &lt; 0.01.
