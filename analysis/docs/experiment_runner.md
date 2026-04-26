# Experiment Runner (Two-Phase)

Use one canonical entrypoint:

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/<name>.json`

## Pipeline model

The system runs in two explicit phases:

1. **Graph setup phase** (repo-local graph artifacts)
2. **Scoring/community phase** (separate scoring artifact root)

Run mode is controlled by `experiment.mode`:

- `setup_only`
- `score_only`
- `setup_and_score`

## Required config shape

Top-level keys:

- `experiment` (must include `graph_id` and `scoring_run_id`; legacy `graph_run_id` / `run_id` are still read with a deprecation warning)
- `artifacts`
- `setup`
- `selection`
- `scoring`
- `community`

Legacy experiment keys are normalized with a warning: `experiment.run_id` → `scoring_run_id`, `experiment.graph_run_id` → `graph_id`.

## Artifact boundaries

Graph bundle root:

- `artifacts.graph_bundle_root/<graph_id>/`
- contains: `anchor/`, `seed/`, `candidate/`, `seed_candidate/`, `pair_training/`

Scoring bundle root:

- `artifacts.scoring_output_root/<scoring_run_id>/`
- contains per target output (`<target>/community/...`) plus `run_manifest.json`

Community output filenames are unchanged:

- `anchor_community_multi_gt_summary.json`
- `anchor_community_sweep__{gt_slug}.csv`
- `anchor_community_best__{gt_slug}.json`

## Setup behavior

`setup.enable` toggles each component:

- `anchor`
- `seed`
- `candidate`
- `seed_candidate`
- `pair_training`

`setup.policy` controls reuse behavior:

- `on_missing`: `build` or `fail`
- `on_present`: `reuse` or `rebuild`

`pair_training` is auto-created in setup when enabled and prerequisites are present.

## Scoring targets

Select graph surfaces via `selection.score_targets`, for example:

- `anchor`
- `seed`
- `candidate`
- `seed_candidate`

`seed_candidate_handcrafted_v1` and `seed_candidate_pu_v1` currently target `seed_candidate`.

## Quickstart

- PU end-to-end:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/seed_candidate.pu.default.json`
- Unweighted setup+score:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/seed_candidate.none.default.json`
- Handcrafted score-only on existing bundle:
  - `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/seed_candidate.handcrafted.default.json`

## Useful CLI overrides

- `--scoring-run-id` (alias: `--run-id`)
- `--run-mode` (`setup_only|score_only|setup_and_score`)
- `--mode-override` (scoring mode)
- `--graph-id` (alias: `--graph-run-id`)
- `--gt-set`
