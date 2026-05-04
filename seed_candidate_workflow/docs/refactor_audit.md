# Refactor Audit: Unscored Graphs + Pluggable Scoring

## Scope

This audit covers the structural refactor that separates:

1. unscored graph construction
2. scoring
3. community sweep consumption

while preserving existing hyper-parameter values.

## New artifacts and stage outputs

### Anchor graph (construction + scoring split)

- New unscored artifact:
  - `anchor_graph_edges_unscored.csv`
- Existing scored artifact retained:
  - `anchor_graph_edges_weighted.csv`
- New standalone scoring stage output:
  - `anchor_graph_scoring_summary.json`

### Seed + candidate unified construction

- New unified stage output directory:
  - `seed_candidate_workflow/output/seed_candidate_graph/<graph_id>/seed_candidate_graph_<timestamp>/` (Path B layout; removed)
- New canonical unscored artifact:
  - `seed_candidate_pairgraph_unscored.csv`
- Stage summary:
  - `seed_candidate_graph_summary.json`

### Scoring plugins

- New scorer registry:
  - `seed_candidate_workflow/utils/graph_scorer_registry.py`
- Implemented scorer entries:
  - `seed_candidate_handcrafted_v1`
  - `seed_candidate_pu_v1`

### Community sweep weighted/unweighted contract

- `run_anchor_multi_gt_community_sweep` now expects canonical pair ids:
  - `email_i,email_j`
- `sweep.score_mode` controls execution mode:
  - non-empty: scorer registry applies scoring and weighted threshold sweep is active.
  - empty: Option A unweighted community path runs, with threshold filtering disabled.

## Config reorganization (values preserved)

Added configs during this audit (later removed with Path B; Path A uses experiment JSON + `seed_candidate_workflow/configs/anchor_*.default.json` only):

- ~~`seed_candidate_workflow/configs/scoring/anchor_graph_scoring.default.json`~~ (removed)
- ~~`seed_candidate_workflow/configs/seed_candidate_graph.default.json`~~ (removed)

Channel semantic aliases were introduced to reduce ambiguity:

- `edge_create_enabled` (alias for candidate creation intent)
- legacy `candidate_enabled` remains supported
- `score_enabled` retained

## Stage/API wiring changes

- New pipeline stages:
  - `score_anchor_graph`
  - `build_seed_candidate_graph`
- Added CLI config flags:
  - `--anchor-scoring-config`
  - `--seed-candidate-graph-config`

## Invariant checks performed in this pass

- Python compile smoke for all modified modules: PASS
  - command: `python -m compileall ...`
- Contract-level guarantees implemented:
  - canonical pair identity normalization (`email_i <= email_j`)
  - unscored/scored required column validation helpers

## Old -> new mapping (high level)

- Old: anchor build always directly emitted scored edges  
  New: anchor build emits unscored + scored; scoring callable as independent stage.

- Old: seed and candidate were separate pipeline stages only  
  New: unified orchestrator stage added that emits canonical unscored seed+candidate pairgraph.

- Old: handcrafted/PU scoring logic embedded in stage-specific helpers  
  New: common scorer registry introduced; helpers now route through scorer functions.

- Old: community sweep only had weighted/thresholded behavior.
  New: community sweep has explicit weighted and unweighted (Option A) paths.

## Forward-only runtime

- Canonical runtime path is now `seed_candidate_workflow/pipelines/run_experiment.py`.
- User-facing runtime configs live under `seed_candidate_workflow/configs/experiments/`.
- Legacy scored-clustering wrappers and PU stage script were removed from canonical runtime.

## Superseded / removed (Path B)

The following were **removed** in favor of Path A only (`run_experiment.py` + `graph_setup_pipeline.py`):

- `seed_candidate_workflow/pipelines/anchor_graph_pipeline.py` (per-stage CLI / `run_anchor_graph_pipeline`).
- `seed_candidate_workflow/utils/seed_candidate_graph_helpers.py` and `seed_candidate_workflow/configs/seed_candidate_graph.default.json` (Path B unified seed+candidate output under `seed_candidate_workflow/output/seed_candidate_graph/...`). Canonical seed–candidate pairgraph is built **inside the graph bundle** by `run_graph_setup`.
- `seed_candidate_workflow/utils/anchor_graph_scoring_helpers.py` and `seed_candidate_workflow/configs/scoring/anchor_graph_scoring.default.json` (standalone anchor handcrafted scoring stage only used by Path B).

The **Stage/API wiring** bullets above that reference `score_anchor_graph`, `build_seed_candidate_graph`, and those JSON paths describe the pre-consolidation state; use `run_experiment` + `setup` / `selection.score_targets` instead.
