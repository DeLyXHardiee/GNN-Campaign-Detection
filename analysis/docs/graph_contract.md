# Graph Contract And Terminology

This document defines the canonical graph terminology and table contracts used by
the refactored anchor pipeline.

## Terminology

- `anchor graph`: Email-email graph derived from the hetero graph universe.
- `seed graph`: Graph induced by seed edges generated from anchor evidence.
- `candidate graph`: Graph induced by candidate pair generators.
- `seed_candidate graph`: Union graph of seed and candidate pairs.
- `unscored graph`: Pair graph with evidence features, without `edge_weight`.
- `scored graph`: Unscored graph plus `score_mode` + `edge_weight`.

## Channel semantics

Channel controls are interpreted as:

- `edge_create_enabled`: channel may independently create candidate pairs.
- `evidence_enabled`: channel evidence is computed and attached to existing pairs.
- `score_enabled`: channel contributes to scorer-specific edge weighting.

This supersedes ambiguous naming such as `enabled`, `candidate_enabled`,
`score_enabled` while preserving existing hyper-parameter values.

## Canonical pair table (`PairGraph`) fields

Required identity:

- `email_i` (string)
- `email_j` (string)
- `graph_kind` (string; one of `anchor`, `seed`, `candidate`, `seed_candidate`, `semantic_shard`)
- `graph_id` (string): graph bundle / anchor run directory name.

Legacy CSVs may still use the column name `graph_run_id` with the same meaning; `analysis.utils.pair_graph_contract` normalizes it to `graph_id` (`migrate_unscored_graph_id_column`, `ensure_unscored_contract`).

Required provenance:

- `from_seed` (bool)
- `from_semantic` (bool)
- `from_rare_artifact` (bool)
- `from_component` (bool)
- `from_2hop` (bool)
- `source_count` (int)

Common optional evidence/context:

- `semantic_cosine_max` (float)
- `component_cosine_max` (float)
- `rare_artifact_rarity_max` (float)
- `twohop_rarity_max` (float)
- `time_gap_seconds_min` (float)
- `seed_component_i` (int)
- `seed_component_j` (int)
- `same_seed_component` (bool)

Scored extension:

- `score_mode` (string)
- `edge_weight` (float)
- `score_diagnostics_json` (string JSON payload; optional)

## Scoring contract

Any scorer must:

1. Input: canonical unscored `PairGraph`.
2. Output: canonical scored table with `edge_weight` and `score_mode`.
3. Keep pair identity (`email_i`, `email_j`) stable.
4. Avoid modifying generation hyper-parameter values.
