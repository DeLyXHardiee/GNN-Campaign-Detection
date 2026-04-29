# Seed-Candidate Pipeline Quick Guide

Fast runbook for teammates who already know the method and need to:

- run a specific experiment quickly,
- know exactly which folder gets created,
- know exactly which files to open first.

## One entrypoint

Use only this command pattern:

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/<config>.json`

## Mental model (2 phases)

- graph setup writes/reuses artifacts in `analysis/output/graph_bundles/<graph_id>/`
- scoring + community writes artifacts in `analysis/output/scoring_runs/<scoring_run_id>/`

Mode is controlled by `experiment.mode`:

- `setup_only`
- `score_only`
- `setup_and_score`

## Quick command index (intent -> config -> output folder)

| Goal | Config | Command | Graph bundle | Scoring run output |
|---|---|---|---|---|
| PU end-to-end on seed+candidate | `analysis/configs/experiments/exp03.seedcand.setupscore.pu.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp03.seedcand.setupscore.pu.json` | `analysis/output/graph_bundles/bundle_seedcand_pairtrain/` | `analysis/output/scoring_runs/run_seedcand_pu/` |
| Seed+candidate unweighted end-to-end baseline | `analysis/configs/experiments/exp01.seedcand.setupscore.unweighted.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp01.seedcand.setupscore.unweighted.json` | `analysis/output/graph_bundles/bundle_seedcand/` | `analysis/output/scoring_runs/run_seedcand_unweighted_setupscore/` |
| Seed+candidate handcrafted score-only on base bundle | `analysis/configs/experiments/exp02.seedcand.scoreonly.handcrafted_on_base.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp02.seedcand.scoreonly.handcrafted_on_base.json` | `analysis/output/graph_bundles/bundle_seedcand/` | `analysis/output/scoring_runs/run_seedcand_handcrafted_scoreonly/` |
| Anchor unweighted score-only on PU bundle | `analysis/configs/experiments/exp04.anchor.scoreonly.unweighted_on_seedcand_pu.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp04.anchor.scoreonly.unweighted_on_seedcand_pu.json` | `analysis/output/graph_bundles/bundle_seedcand_pairtrain/` | `analysis/output/scoring_runs/run_anchor_unweighted_scoreonly/` |
| Seed+candidate unweighted score-only on PU bundle | `analysis/configs/experiments/exp05.seedcand.scoreonly.unweighted_on_seedcand_pu.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp05.seedcand.scoreonly.unweighted_on_seedcand_pu.json` | `analysis/output/graph_bundles/bundle_seedcand_pairtrain/` | `analysis/output/scoring_runs/run_seedcand_unweighted_scoreonly/` |
| Seed+candidate handcrafted score-only on PU bundle | `analysis/configs/experiments/exp06.seedcand.scoreonly.handcrafted_on_seedcand_pu.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp06.seedcand.scoreonly.handcrafted_on_seedcand_pu.json` | `analysis/output/graph_bundles/bundle_seedcand_pairtrain/` | `analysis/output/scoring_runs/run_seedcand_handcrafted_scoreonly_on_pairtrain/` |
| Semantic shard unweighted end-to-end | `analysis/configs/experiments/exp07.shard.setupscore.unweighted.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp07.shard.setupscore.unweighted.json` | `analysis/output/graph_bundles/bundle_shard/` | `analysis/output/scoring_runs/run_shard_unweighted/` |
| Semantic shard weighted (notebook-like) end-to-end | `analysis/configs/experiments/exp08.shard.setupscore.weighted_notebook.json` | `python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp08.shard.setupscore.weighted_notebook.json` | `analysis/output/graph_bundles/bundle_shard/` | `analysis/output/scoring_runs/run_shard_weighted_notebook/` |

## Where artifacts are written

Configured by `artifacts` in experiment config:

- graph bundle root: `artifacts.graph_bundle_root/<graph_id>/`
- scoring run root: `artifacts.scoring_output_root/<scoring_run_id>/`

Inside graph bundle (only enabled pieces are created):

- `anchor/`
- `seed/`
- `candidate/`
- `seed_candidate/`
- `pair_training/`
- `semantic_shard/` (if used)

Inside scoring run:

- `<target>/community/` (per-target sweep artifacts)
- `run_manifest.json` (single source of truth for resolved inputs/outputs)

## First files to open after a run

Start in `analysis/output/scoring_runs/<scoring_run_id>/`.

| Target | Read first | Then read |
|---|---|---|
| `anchor`, `seed`, `candidate`, `seed_candidate` | `<target>/community/anchor_community_multi_gt_summary.json` | `<target>/community/anchor_community_best__<gt_slug>.json`, `<target>/community/anchor_community_sweep__<gt_slug>.csv` |
| `semantic_shard` | `<target>/community/semantic_shard_community_multi_gt_summary.json` | `<target>/community/semantic_shard_community_best__<gt_slug>.json`, `<target>/community/semantic_shard_community_sweep__<gt_slug>.csv` |
| any target | `run_manifest.json` | `targets[*].artifacts` entries for exact paths |

## Most edited config fields

In `analysis/configs/experiments/*.json`:

- `experiment.graph_id`: which graph bundle to build/reuse
- `experiment.scoring_run_id`: name of scoring output folder
- `experiment.mode`: `setup_only`, `score_only`, `setup_and_score`
- `selection.score_targets`: `anchor`, `seed`, `candidate`, `seed_candidate`, `semantic_shard`
- `scoring.score_mode`: `none`, `seed_candidate_handcrafted_v1`, `seed_candidate_pu_v1`, `semantic_shard_handcrafted_v1`, `semantic_shard_affine_v1`
- `setup.enable`: stage toggles (`anchor`, `seed`, `candidate`, `seed_candidate`, `pair_training`, `semantic_shard`)
- `setup.policy`:
  - `on_missing`: `build` or `fail`
  - `on_present`: `reuse` or `rebuild`
- `community.sweep`: community method/sweep/ranking settings

## Compatibility cheatsheet

- `seed_candidate_pu_v1` -> `seed_candidate`
- `seed_candidate_handcrafted_v1` -> `seed_candidate`
- `semantic_shard_handcrafted_v1` -> `semantic_shard`
- `semantic_shard_affine_v1` -> `semantic_shard`
- `none` -> unweighted topology baseline

If target/mode is incompatible, validation fails fast.

## Dry-run before long jobs

Path-resolution check without executing setup/community:

`python analysis/pipelines/run_experiment.py --config analysis/configs/experiments/exp03.seedcand.setupscore.pu.json --dry-run`

## Typical workflow for a new variant

1. Copy the closest config in `analysis/configs/experiments/`.
2. Set unique `experiment.graph_id` and `experiment.scoring_run_id`.
3. Set `experiment.mode`:
   - first run: `setup_and_score`
   - rerun with new scorer/sweep only: `score_only`
4. Set `selection.score_targets` and `scoring.score_mode`.
5. Tune `community.sweep`.
6. Run once with `--dry-run`, then run normally.

## Troubleshooting

- Missing file in score-only run:
  - check `experiment.graph_id` exists under `analysis/output/graph_bundles/`.
- Unexpected reuse/rebuild:
  - check `setup.policy.on_present`.
- Outputs not where expected:
  - verify `artifacts.graph_bundle_root` and `artifacts.scoring_output_root`.
- Unsure what was resolved at runtime:
  - open `analysis/output/scoring_runs/<scoring_run_id>/run_manifest.json`.
