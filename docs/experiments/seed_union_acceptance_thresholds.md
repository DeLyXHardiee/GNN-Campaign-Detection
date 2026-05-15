# Seed union acceptance thresholds (manual QA)

After regenerating **seed** / **pair_training**, sanity-check `anchor_seed_summary.json` and `pair_training_dataset_summary.json` before committing to a full GNN retrain.

Rough targets for the dedup-strict track (tune if your cohort shifts):

| Metric | Healthy ballpark | Bad signal |
|--------|------------------|------------|
| `union_edges.metrics.n_components` | ≥ ~450–550 | Large drop vs prior run (mega-merge) |
| `union_edges.component_size_distribution_top50[0]` | ≤ ~130–250 | **> ~400** (giant component) |
| `pair_training.component_context`: `n_pairs_same_seed_component` / `n_unique_pairs_final` | ≤ ~0.35–0.45 | **> ~0.48** (PU dominated by same-island pairs) |

Automated check:

```text
python seed_candidate_workflow/scripts/check_seed_bundle_health.py ^
  --anchor-seed-summary <path/to/anchor_seed_summary.json> ^
  --pair-training-summary <path/to/pair_training_dataset_summary.json>
```

Defaults match `check_seed_bundle_health.py` flags; override with `--max-union-largest-component`, `--min-union-components`, `--max-same-seed-component-fraction` as needed.
