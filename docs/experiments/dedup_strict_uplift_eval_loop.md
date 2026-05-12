# Dedup strict uplift — evaluation order

Current uplift **graph id / run id / hetero stem**: `main_gnn_pu_1_no_ts_dedup_strict_uplift` (writes `core/graph/output/main_gnn_pu_1_no_ts_dedup_strict_uplift_hetero.pt` and bundle under `seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts_dedup_strict_uplift/`).

Use this order after changing anchor graph, hetero graph, seeds, candidates, or pair-dataset logic so paths and indices stay aligned.

1. **Hetero graph** (if `pipeline_config.json` `graph.*` or MISP lake changed): run graph creation from repo root (e.g. `python core/main.py` with `run_graph_creation` enabled, or your usual graph-build entrypoint). Confirm `graph_pt_path_override` and `hetero_graph_stem` match the intended run id.

2. **Seed–candidate bundle**: run experiment setup for your graph id (e.g. `python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp03.seedcand.setupscore.pu.json` with `--run-mode setup_only` or full setup as needed). Confirm `candidate_eval_summary.json` reports `ready_for_pu_dataset_construction`.

3. **Pair training CSV**: produced under `seed_candidate_workflow/output/graph_bundles/<graph_id>/pair_training/<graph_id>/pair_training_dataset.csv`.

4. **GNN training**: ensure `pipeline_config.json` `pair_training.pair_dataset_csv` and `run_id` match the bundle; train pair supervision.

5. **Pair score separation** (optional diagnostics):

   ```text
   python seed_candidate_workflow/scripts/run_pair_score_separation_analysis.py --run-dir output/runs/<run_id> --graph-pt core/graph/output/<stem>_hetero.pt --gt-path data/groundtruth/ground_truth.dedup_strict.json
   ```

6. **Community / PU scoring on GT**: `python seed_candidate_workflow/pipelines/run_experiment.py --config seed_candidate_workflow/configs/experiments/exp03.seedcand.setupscore.pu.json --run-mode score_only`

7. **Timestamp ablation** (optional): copy keys from `seed_candidate_workflow/configs/experiments/pipeline_fragment.ts_ablation.example.json` into a dedicated `pipeline_config` (or merge manually), rebuild hetero with `zero_email_timestamps: false`, then repeat steps 2–6. Dedup MISP events include a `date` attribute on `Event.Attribute`; use this branch only when you want email-level time features in the hetero graph.

**Metric log**: record `run_id`, anchor channel set file version, candidate cosine thresholds, `zero_email_timestamps`, best community sweep `v-measure` (from scoring run artifacts), and pair AUROC from step 5.
