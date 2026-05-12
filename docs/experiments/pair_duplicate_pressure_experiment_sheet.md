# Pair Duplicate Pressure Experiment Sheet

Use this file as the control document for all duplicate-pressure remediation runs.

## Scope Lock

- Base graph: `main_gnn_pu_1_no_ts`
- Base pair dataset: `seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts/pair_training/main_gnn_pu_1_no_ts/pair_training_dataset.csv`
- Base training run for comparison: `output/runs/main_gnn_pu_1_no_ts`
- Duplicate diagnostics baseline:
  - `seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts/pair_training/main_gnn_pu_1_no_ts/pair_duplicate_pressure_20260507T134051Z/pair_duplicate_pressure_summary.json`
  - `.../pair_duplicate_pressure_top_clusters.csv`
  - `.../pair_duplicate_pressure_by_stratum.csv`

No candidate/seed/graph regeneration during Phase 1 and Phase 2 unless explicitly noted.

## Baseline Snapshot (Locked)

- Pair rows analyzed: `126929`
- Strict duplicate same-cluster rows: `24113` (`0.18997`)
- Strict duplicate same-cluster among positives: `24113 / 75920 = 0.31761`
- Strict duplicate same-cluster among unlabeled: `0 / 51009 = 0.0`
- Strict realized over potential: `1.0` (`24113 / 24113`)
- Concentration (strict):
  - top-1 cluster contributes `4950` duplicate-positive rows (`20.53%` of strict dup-positive mass)
  - top-2 cumulative: `9510` (`39.44%`)
  - top-20 cumulative: `18028` (`74.76%`)
- Split parity check (strict dup fraction):
  - train `0.19014`
  - val `0.19351`
  - test `0.18508`

## Decision Metrics (Track Every Run)

Primary:
- `val_epoch_score_separation` (best epoch and final epoch)
- `val_separation_at_threshold` (best epoch and final epoch)
- `pair_score_separation` cross-component same-campaign vs cross-campaign overlap (qual + quant summary)

Duplicate-pressure:
- strict `frac_dup_same_cluster_rows`
- strict `by_pair_status.positive.frac_dup_same_cluster`
- strict top-20 cumulative duplicate-positive mass

Stability:
- early stop epoch
- val loss trend monotonicity / reversals
- train-vs-val separation divergence

Guardrails:
- Do not decrease best `val_epoch_score_separation` by more than `0.01` vs baseline unless cross-component frontier clearly improves.
- Do not increase strict positive duplicate share.
- Do not introduce split skew > `0.02` absolute between train/val duplicate fractions.

## Phase Plan

### Phase 1 (Executable Now): Easy-positive capping ablations

All Phase 1 runs keep these controls fixed:
- `pair_loss_type: nnpu_with_reliable_negatives`
- `hard_positive_emphasis.enabled: false`
- `hard_unlabeled_emphasis.enabled: false`
- `reliable_negative_pool.enabled: false`
- `reliable_negative_emphasis.enabled: false`
- `pair_split_seed: 42`, `pair_val_ratio: 0.1`, `pair_test_ratio: 0.1`
- `pair_dataset_csv` and `graph` unchanged from baseline (`main_gnn_pu_1_no_ts`)

Minimal run matrix (exact values):

1. `E01` / `main_gnn_pu_1_no_ts_epc_or_080`
   - `easy_positive_capping.enabled: true`
   - `easy_positive_capping.downsample_fraction: 0.80`
   - `easy_positive_capping.same_seed_component_only: true`
   - `easy_positive_capping.min_semantic_cosine: 0.97`
   - `easy_positive_capping.min_source_count: 2`
   - `easy_positive_capping.or_rule_across_conditions: true`
2. `E02` / `main_gnn_pu_1_no_ts_epc_or_060`
   - same as E01 except `downsample_fraction: 0.60`
3. `E04` / `main_gnn_pu_1_no_ts_epc_and_080`
   - `easy_positive_capping.enabled: true`
   - `easy_positive_capping.downsample_fraction: 0.80`
   - `easy_positive_capping.same_seed_component_only: true`
   - `easy_positive_capping.min_semantic_cosine: 0.95`
   - `easy_positive_capping.min_source_count: 2`
   - `easy_positive_capping.or_rule_across_conditions: false`
Deferred for now (to reduce complexity / compute):
- `E03`, `E05`, `E06` (additional strength sweeps)
- `E07`, `E08`, `E09` (blocked by missing cluster-aware/quota training knobs)

## Run Execution Contract

For each experiment run, always produce:
- training metrics (`output/runs/<run_id>/metrics.csv`)
- pair score separation diagnostics
- pair duplicate pressure diagnostics against the same pair CSV

Keep `pair_split_seed`, `pair_val_ratio`, `pair_test_ratio` fixed unless the row explicitly says otherwise.

### Command Template (PowerShell)

Use this template for E01, E02, and E04 by substituting values from the matrix above:

```powershell
Set-Location "c:\Users\aar\Desktop\GNN-Campaign-Detection"

$runId = "main_gnn_pu_1_no_ts_epc_or_080"
# For each run, change the literals in the Python block below to the E0x values.

@'
import json
from pathlib import Path

p = Path("pipeline_config.json")
cfg = json.loads(p.read_text(encoding="utf-8"))

cfg["run_id"] = "main_gnn_pu_1_no_ts_epc_or_080"
cfg["graph"]["hetero_graph_stem"] = "main_gnn_pu_1_no_ts"
cfg["graph"]["graph_pt_path_override"] = "core/graph/output/main_gnn_pu_1_no_ts_hetero.pt"
cfg["pair_training"]["pair_dataset_csv"] = "seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts/pair_training/main_gnn_pu_1_no_ts/pair_training_dataset.csv"

pt = cfg["pair_training"]
pt["pair_loss_type"] = "nnpu_with_reliable_negatives"
pt["hard_positive_emphasis"]["enabled"] = False
pt["hard_unlabeled_emphasis"]["enabled"] = False
pt["reliable_negative_pool"]["enabled"] = False
pt["reliable_negative_emphasis"]["enabled"] = False

pt["easy_positive_capping"]["enabled"] = True
pt["easy_positive_capping"]["downsample_fraction"] = 0.80
pt["easy_positive_capping"]["same_seed_component_only"] = True
pt["easy_positive_capping"]["min_semantic_cosine"] = 0.97
pt["easy_positive_capping"]["min_source_count"] = 2
pt["easy_positive_capping"]["or_rule_across_conditions"] = True

p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print("updated pipeline_config.json for run_id", cfg["run_id"])
'@ | python -

# Train
python core/main.py

# Duplicate pressure diagnostics for this run's pair CSV
python seed_candidate_workflow/scripts/analyze_pair_training_duplicate_pressure.py `
  --pair-csv "seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts/pair_training/main_gnn_pu_1_no_ts/pair_training_dataset.csv" `
  --email-cluster-parquet "data/misp/duplicate_email_analysis/email_duplicate_cluster.parquet" `
  --misp-loaded-ids-parquet "data/misp/duplicate_email_analysis/misp_loaded_external_ids.parquet" `
  --graph-meta-json "core/graph/output/main_gnn_pu_1_no_ts_hetero.meta.json" `
  --apply-split `
  --training-config-json ("output/runs/" + $runId + "/training_config.json") `
  --out-dir ("seed_candidate_workflow/output/graph_bundles/main_gnn_pu_1_no_ts/pair_training/main_gnn_pu_1_no_ts/pair_duplicate_pressure_" + $runId)
```

## Success Gates

Gate A (after Phase 1):
- at least one policy improves or preserves best `val_epoch_score_separation` while reducing duplicate concentration.

## Logging

Log each run in:
- `docs/experiments/pair_duplicate_pressure_experiment_log.csv`

Update status as one of:
- `planned`, `running`, `done`, `blocked`, `drop`

