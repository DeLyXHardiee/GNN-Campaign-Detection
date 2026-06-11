# Dedup strict uplift — evaluation order

Current **graph id / run id / hetero stem** (stable dedup-strict track): `main_gnn_pu_1_no_ts_dedup_strict` (writes `core/graph/output/main_gnn_pu_1_no_ts_dedup_strict_hetero.pt` and bundle under `seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_strict/`).

Use this order after changing anchor graph, hetero graph, seeds, candidates, or pair-dataset logic so paths and indices stay aligned.

**Restoration / regression guardrails:** After a bad run (mega seed components, PU collapse), configs were rolled back toward stable dedup_strict: no `exact_return_path_domain` hard seed; corroborated weak channels trimmed and `min_semantic_score` 0.92; anchor infra channels disabled again; candidate semantic thresholds restored; GNN `hidden` / `pair_scorer_hidden_dim` back to 128 / 256. Run [`check_seed_bundle_health.py`](../../seed_candidate_workflow/scripts/check_seed_bundle_health.py) after step 2; see [`seed_union_acceptance_thresholds.md`](seed_union_acceptance_thresholds.md).

**`graph_meta_json`:** `exp03.seedcand.setupscore.pu.json` must list the `.meta.json` that matches `pipeline_config.json` hetero stem. If `setup.paths.pair_training.graph_meta_json` is omitted, `run_graph_setup` derives it from `default_hetero_graph_pt_path()`.

### PU calibration run (pair CSV + training only)

Use this when you change **`training.dropout` / `training.wd`**, **`pair_training.reliable_negative_pool`**, or **`pair_training.easy_positive_capping`** in [`pipeline_config.json`](../../pipeline_config.json) but **not** anchor/seed/candidate graphs. `run_graph_setup` applies **`pair_training.reliable_negative_pool`** from that file when rebuilding the pair dataset.

- Set **`pipeline_config.json`** `run_id` to a new value (e.g. `main_gnn_pu_1_no_ts_dedup_strict_restore_v2`) so `output/runs/<run_id>/` stays comparable to prior runs.
- Remove the pair-training folder only: `seed_candidate_workflow/output/graph_bundles/<graph_id>/pair_training/<graph_id>/`, **or** set **`setup.policy.on_present`** to **`rebuild`** in [`exp03.seedcand.setupscore.pu.json`](../../seed_candidate_workflow/configs/experiments/exp03.seedcand.setupscore.pu.json) for one setup run (rebuilds stages according to policy).
- Run **`setup_only`** as in step 2 below.
- Open **`pair_training_dataset_summary.json`** and confirm **`n_reliable_negative_pairs`** &gt; 0 when the RN pool is enabled.
- Train (step 4 below), then **`run_pair_score_separation_analysis.py`** using **`output/runs/<run_id>`**. Before **`score_only`**, set **`exp03.scoring.params.pu.pu_run.run_dir`** to the same **`run_id`** directory.

1. **Hetero graph** (if `pipeline_config.json` `graph.*` or MISP lake changed): run graph creation from repo root (e.g. `python core/main.py` with `run_graph_creation` enabled, or your usual graph-build entrypoint). Confirm `graph_pt_path_override` and `hetero_graph_stem` match the intended run id.

2. **Seed–candidate bundle**: run experiment setup for your graph id (e.g. `python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp03.seedcand.setupscore.pu.json` with `--run-mode setup_only` or full setup as needed). Confirm `candidate_eval_summary.json` reports `ready_for_pu_dataset_construction`. Then run **seed bundle health** (paths from your latest `seed_generation_*` / `pair_training`):

   ```text
   python seed_candidate_workflow/scripts/check_seed_bundle_health.py --anchor-seed-summary <path/to/anchor_seed_summary.json> --pair-training-summary <path/to/pair_training_dataset_summary.json>
   ```

3. **Pair training CSV**: produced under `seed_candidate_workflow/output/graph_bundles/<graph_id>/pair_training/<graph_id>/pair_training_dataset.csv`.

4. **GNN training**: ensure `pipeline_config.json` `pair_training.pair_dataset_csv` and `run_id` match the bundle; train pair supervision.

5. **Pair score separation** (optional diagnostics):

   ```text
   python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py --run-dir output/runs/<run_id> --graph-pt core/graph/output/<stem>_hetero.pt --gt-path data/groundtruth/ground_truth.dedup_strict.json
   ```

6. **Community / PU scoring on GT**: `python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp03.seedcand.setupscore.pu.json --run-mode score_only`

7. **Community sweep (homogeneity vs completeness):** After pair AUROC recovers on GT-covered pairs (~0.9+ same-vs-cross), tune `community.weight_thresholds` and `community.resolutions` in exp03 for best **v-measure** (balances homogeneity and completeness). Log the winning row from the sweep CSV under the scoring run directory.

8. **Timestamp ablation** (optional): copy keys from `seed_candidate_workflow/configs/experiments/pipeline_fragment.ts_ablation.example.json` into a dedicated `pipeline_config` (or merge manually), rebuild hetero with `zero_email_timestamps: false`, then repeat steps 2–6. Dedup MISP events include a `date` attribute on `Event.Attribute`; use this branch only when you want email-level time features in the hetero graph.

**Metric log**: record `run_id`, anchor channel set file version, candidate cosine thresholds, `zero_email_timestamps`, best community sweep `v-measure` (from scoring run artifacts), and pair AUROC from step 5.
